// Static, shareable flood-sim viewer for Washington DC.
// WebFlood-style: the shallow-water solver runs entirely in the browser GPU
// (see sim.js). This file loads the baked DEM + satellite imagery + OSM
// city, renders the 3D scene, and pulls live flows/stage from USGS NWIS.

import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { N8AOPass } from './vendor/N8AO.m.js';
import { GPUFloodSim } from './sim.js?v=6';

const state = {
  nx: 0, ny: 0, dx: 1,
  zmin: 0, zmax: 1, zRange: 1,
  west: 0, south: 0, east: 0, north: 0,
  lastDiag: 0,
  alphaScale: 0.85,
  worldScale: 1.0, worldDx: 1.0,
  sim: null,
  subSteps: 8,
  paused: false,
  gauges: [],
  sliderApply: {},
  buildings: null,
  terrainMesh: null, waterMesh: null, terrainMat: null, waterMat: null,
  framesSinceLog: 0, lastFpsLog: performance.now(),
  lastSimTime: 0, lastRateT: performance.now(), simRate: 0,
};

const SUN = new THREE.Vector3(0.40, 0.82, 0.42).normalize();

function fatal(msg) {
  const el = document.getElementById('err');
  el.textContent = msg;
  el.style.display = 'block';
  document.getElementById('loading').style.display = 'none';
}

// ----------------------------------------------------------------------
// Asset loading (gzip sniffed and decompressed client-side).
// ----------------------------------------------------------------------
async function fetchBinary(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  let buf = await r.arrayBuffer();
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const ds = new DecompressionStream('gzip');
    buf = await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
  }
  return buf;
}

// ----------------------------------------------------------------------
// Renderer + scene + navigation
// ----------------------------------------------------------------------
const pane3d = document.getElementById('pane-3d');
const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(pane3d.clientWidth, pane3d.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.72;
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0a0c10, 6, 18);

const camera = new THREE.PerspectiveCamera(
  45, pane3d.clientWidth / pane3d.clientHeight, 0.0005, 20);
camera.position.set(0.15, 1.05, 1.15);     // south of the city, looking north

// Map-style navigation: left-drag pans the ground plane, right-drag rotates,
// wheel zooms toward the cursor — all the way down to street level.
const controls = new MapControls(camera, renderer.domElement);
controls.target.set(0, 0, -0.05);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomToCursor = true;
controls.minDistance = 0.002;              // ~20 m — street level
controls.maxDistance = 6;
controls.maxPolarAngle = Math.PI * 0.495;  // don't go below the horizon
controls.keyPanSpeed = 12;
controls.listenToKeyEvents(window);        // arrow keys pan
controls.update();
state.camera = camera;
state.controls = controls;

window.addEventListener('resize', () => {
  camera.aspect = pane3d.clientWidth / pane3d.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(pane3d.clientWidth, pane3d.clientHeight);
  composer.setSize(pane3d.clientWidth, pane3d.clientHeight);
});

// ----------------------------------------------------------------------
// Render shaders. Vertical scale is TRUE (1 m of elevation = 1 m of extent).
// Terrain albedo: baked Esri World Imagery drape, or hypsometric ramp.
// ----------------------------------------------------------------------
const TERRAIN_VS = /* glsl */`
  uniform sampler2D zTex;
  uniform float zmin;
  uniform float vScale;   // world units per metre of elevation
  varying vec2 vUv;
  varying float vBedZ;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    float z = texture2D(zTex, uv).r;
    vBedZ = z;
    float y = (z - zmin) * vScale;
    vec3 newPos = vec3(position.x, y, position.z);
    vec4 wp = modelMatrix * vec4(newPos, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const TERRAIN_FS = /* glsl */`
  precision highp float;
  uniform sampler2D zTex;
  uniform sampler2D satTex;
  uniform float useSat;   // 0 = elevation ramp, 1 = satellite drape
  uniform float texelU;
  uniform float texelV;
  uniform float dxm;      // cell size in metres
  uniform vec3 sunDir;
  varying vec2 vUv;
  varying float vBedZ;
  varying vec3 vWorldPos;

  vec3 elevRamp(float zMeters) {
    if (zMeters < 3.0) {
      float u = clamp(zMeters / 3.0, 0.0, 1.0);
      return mix(vec3(0.18, 0.28, 0.36), vec3(0.30, 0.40, 0.32), u);
    } else if (zMeters < 15.0) {
      float u = (zMeters - 3.0) / 12.0;
      return mix(vec3(0.30, 0.40, 0.32), vec3(0.42, 0.58, 0.30), u);
    } else if (zMeters < 40.0) {
      float u = (zMeters - 15.0) / 25.0;
      return mix(vec3(0.42, 0.58, 0.30), vec3(0.74, 0.66, 0.38), u);
    } else if (zMeters < 80.0) {
      float u = (zMeters - 40.0) / 40.0;
      return mix(vec3(0.74, 0.66, 0.38), vec3(0.86, 0.74, 0.52), u);
    }
    float u = clamp((zMeters - 80.0) / 40.0, 0.0, 1.0);
    return mix(vec3(0.86, 0.74, 0.52), vec3(0.96, 0.92, 0.84), u);
  }

  void main() {
    float zL = texture2D(zTex, vec2(vUv.x - texelU, vUv.y)).r;
    float zR = texture2D(zTex, vec2(vUv.x + texelU, vUv.y)).r;
    float zB = texture2D(zTex, vec2(vUv.x, vUv.y - texelV)).r;
    float zT = texture2D(zTex, vec2(vUv.x, vUv.y + texelV)).r;
    float slopeX = (zR - zL) / (2.0 * dxm);
    float slopeZ = (zT - zB) / (2.0 * dxm);   // gradient toward north
    // World north = -z, so the north gradient appears with +sign on z.
    vec3 normal = normalize(vec3(-slopeX, 1.0, slopeZ));

    float diff = max(dot(normal, sunDir), 0.0);
    float skyFill = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
    vec3 albedo = mix(elevRamp(vBedZ), texture2D(satTex, vUv).rgb, useSat);
    // Imagery has its own baked shading — light it more gently.
    float lit = mix(0.35 + 0.85 * diff + 0.10 * skyFill,
                    0.55 + 0.42 * diff, useSat);
    gl_FragColor = vec4(albedo * lit, 1.0);
  }
`;

const WATER_VS = /* glsl */`
  uniform sampler2D zTex;     // bed
  uniform sampler2D hTex;     // water depth (sim state, R channel)
  uniform float zmin;
  uniform float vScale;
  varying vec2 vUv;
  varying float vDepth;
  varying float vSurfaceZ;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    float zb = texture2D(zTex, uv).r;
    float h  = max(texture2D(hTex, uv).r, 0.0);
    float zw = zb + h;
    vDepth = h;
    // Blend dry vertices onto the terrain and wet vertices onto the water
    // surface so shorelines don't form vertical cliffs.
    float blend = smoothstep(0.1, 1.5, h);
    float yEff = mix(zb, zw, blend);
    vSurfaceZ = yEff;
    float y = (yEff - zmin) * vScale;
    vec3 newPos = vec3(position.x, y, position.z);
    vec4 wp = modelMatrix * vec4(newPos, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const WATER_FS = /* glsl */`
  precision highp float;
  uniform float alphaScale;
  uniform vec3 sunDir;
  uniform vec3 viewerPos;
  uniform vec3 skyTint;
  uniform sampler2D qTex;   // face discharge (m²/s) from the solver
  uniform float time;
  varying vec2 vUv;
  varying float vDepth;
  varying float vSurfaceZ;
  varying vec3 vWorldPos;

  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i),               hash(i + vec2(1., 0.)), f.x),
               mix(hash(i + vec2(0., 1.)), hash(i + vec2(1., 1.)), f.x), f.y);
  }

  void main() {
    if (vDepth < 0.08) discard;

    vec3 dpdx_ = dFdx(vWorldPos);
    vec3 dpdy_ = dFdy(vWorldPos);
    vec3 normal = normalize(cross(dpdy_, dpdx_));
    if (normal.y < 0.0) normal = -normal;

    // Flow velocity from the sim's discharge field — ripples drift downstream.
    vec2 q = texture2D(qTex, vUv).rg;
    vec2 vel = q / max(vDepth, 0.3);
    float speed = length(vel);
    vec2 fuv = vUv * 700.0 - vel * time * 0.10;   // ~30 m ripple wavelength
    float n1 = vnoise(fuv + vec2(0.0, time * 0.6));
    float n2 = vnoise(fuv * 2.1 + vec2(time * 0.8, 0.0));
    float ripple = (n1 * 0.65 + n2 * 0.35) - 0.5;
    normal = normalize(normal +
      vec3(ripple * 0.20, 0.0, ripple * 0.20) * clamp(vDepth, 0.3, 1.0));

    // Murky river water: green-brown when shallow, near-black when deep.
    float dN = clamp(vDepth / 6.0, 0.0, 1.0);
    vec3 shallowMud = vec3(0.30, 0.30, 0.20);
    vec3 deepWater  = vec3(0.04, 0.07, 0.08);
    vec3 base = mix(shallowMud, deepWater, sqrt(dN));

    float diff = max(dot(normal, sunDir), 0.0);
    vec3 view = normalize(viewerPos - vWorldPos);
    vec3 H = normalize(sunDir + view);
    float spec = pow(max(dot(normal, H), 0.0), 120.0);
    vec3 color = base * (0.45 + 0.60 * diff);
    color += vec3(1.0, 0.97, 0.90) * spec * 0.4;

    // Grazing angles pick up the sky.
    float fres = pow(1.0 - max(dot(normal, view), 0.0), 4.0);
    color = mix(color, skyTint * 0.7, fres * 0.45);

    // Foam at the wetting front and where the flow is fast.
    float edge = smoothstep(0.08, 0.13, vDepth) * (1.0 - smoothstep(0.13, 0.5, vDepth));
    float rapids = smoothstep(3.0, 6.0, speed);
    float foam = clamp(edge * 0.7 + rapids * 0.35, 0.0, 1.0) * (0.55 + 0.45 * n2);
    color = mix(color, vec3(0.82, 0.85, 0.84), foam * 0.4);

    float a = smoothstep(0.03, 0.4, vDepth) * alphaScale;
    a = max(a, foam * 0.5 * alphaScale);
    gl_FragColor = vec4(color, a);
  }
`;

// ----------------------------------------------------------------------
// Mesh construction. The render mesh is decimated (the sim still runs at
// full resolution — vertex/fragment shaders sample the full-res textures).
// ----------------------------------------------------------------------
function buildScene() {
  const { nx, ny, dx, sim } = state;
  const worldW = nx * dx;
  const worldH = ny * dx;
  state.worldScale = 2.0 / Math.max(worldW, worldH);
  state.worldDx = dx * state.worldScale;
  const s = state.worldScale;
  const cx = 0.5 * worldW * s;
  const cy = 0.5 * worldH * s;

  const step = Math.max(1, Math.round(Math.max(nx, ny) / 700));
  const nvx = Math.floor((nx - 1) / step) + 1;
  const nvy = Math.floor((ny - 1) / step) + 1;

  const nv = nvx * nvy;
  const positions = new Float32Array(nv * 3);
  const uvs = new Float32Array(nv * 2);
  for (let vi = 0; vi < nvx; vi++) {
    const i = Math.min(vi * step, nx - 1);
    const x = i * dx * s - cx;
    const u = i / (nx - 1);
    for (let vj = 0; vj < nvy; vj++) {
      const j = Math.min(vj * step, ny - 1);
      const idx = vi * nvy + vj;
      positions[3 * idx + 0] = x;
      positions[3 * idx + 1] = 0;
      // Right-handed world: +x = east, +y = up, NORTH = -z (else the map
      // renders mirrored east-west).
      positions[3 * idx + 2] = cy - j * dx * s;
      uvs[2 * idx + 0] = u;
      uvs[2 * idx + 1] = j / (ny - 1);
    }
  }
  const nq = (nvx - 1) * (nvy - 1);
  const indices = new (nv > 65535 ? Uint32Array : Uint16Array)(nq * 6);
  let k = 0;
  for (let i = 0; i < nvx - 1; i++) {
    for (let j = 0; j < nvy - 1; j++) {
      const v00 = i * nvy + j;
      const v10 = (i + 1) * nvy + j;
      const v01 = i * nvy + (j + 1);
      const v11 = (i + 1) * nvy + (j + 1);
      indices[k++] = v00; indices[k++] = v10; indices[k++] = v11;
      indices[k++] = v00; indices[k++] = v11; indices[k++] = v01;
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));

  const tUniforms = {
    zTex:    { value: sim.zTexture },
    satTex:  { value: sim.zTexture },   // placeholder until imagery loads
    useSat:  { value: 0 },
    texelU:  { value: 1.0 / (nx - 1) },
    texelV:  { value: 1.0 / (ny - 1) },
    zmin:    { value: state.zmin },
    vScale:  { value: state.worldScale },
    dxm:     { value: dx },
    sunDir:  { value: SUN.clone() },
  };
  state.terrainMat = new THREE.ShaderMaterial({
    uniforms: tUniforms,
    vertexShader: TERRAIN_VS,
    fragmentShader: TERRAIN_FS,
    side: THREE.DoubleSide,
  });
  state.terrainMesh = new THREE.Mesh(geom, state.terrainMat);
  scene.add(state.terrainMesh);

  const wGeom = new THREE.BufferGeometry();
  wGeom.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
  wGeom.setAttribute('uv',       new THREE.BufferAttribute(uvs.slice(), 2));
  wGeom.setIndex(geom.getIndex());
  const wUniforms = {
    zTex:       { value: sim.zTexture },
    hTex:       { value: sim.hTexture },
    qTex:       { value: sim.qTexture },
    zmin:       { value: state.zmin },
    vScale:     { value: state.worldScale },
    alphaScale: { value: state.alphaScale },
    sunDir:     { value: SUN.clone() },
    skyTint:    { value: new THREE.Color(0xd9b083) },
    viewerPos:  { value: new THREE.Vector3() },
    time:       { value: 0 },
  };
  state.waterMat = new THREE.ShaderMaterial({
    uniforms: wUniforms,
    vertexShader: WATER_VS,
    fragmentShader: WATER_FS,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  state.waterMesh = new THREE.Mesh(wGeom, state.waterMat);
  state.waterMesh.renderOrder = 2;
  scene.add(state.waterMesh);
}

// Satellite imagery drape (optional asset).
function loadBasemap() {
  new THREE.TextureLoader().load('data/basemap.jpg', (tex) => {
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
    state.satTex = tex;
    if (state.terrainMat) {
      state.terrainMat.uniforms.satTex.value = tex;
      state.terrainMat.uniforms.useSat.value =
        document.getElementById('sat').checked ? 1 : 0;
    }
  }, undefined, () => {
    console.warn('[basemap] unavailable — using elevation ramp');
    const cb = document.getElementById('sat');
    cb.checked = false;
    cb.disabled = true;
  });
}

// ----------------------------------------------------------------------
// 3D city layer: OSM building footprints baked to oriented boxes
// (data/buildings.f32.gz, 7 floats per building), drawn as one InstancedMesh.
// ----------------------------------------------------------------------
async function loadBuildings() {
  let rec;
  try {
    rec = new Float32Array(await fetchBinary('data/buildings.f32.gz'));
  } catch (e) {
    console.warn('[buildings] unavailable:', e.message);
    return;
  }
  const n = Math.floor(rec.length / 7);
  if (!n) return;
  const geom = new THREE.BoxGeometry(1, 1, 1);
  const mat = new THREE.MeshStandardMaterial({
    color: 0xffffff, roughness: 0.85, metalness: 0.05,
  });
  const mesh = new THREE.InstancedMesh(geom, mat, n);
  mesh.frustumCulled = false;          // instances span the whole domain
  mesh.renderOrder = 1;                // after terrain, before water blend
  // Per-instance color: muted facade tones, hash-varied shade, taller
  // buildings tinted toward glass blue.
  const palette = [0x80868e, 0x8e887d, 0x757c84, 0x8a8278, 0x7d8584]
    .map(c => new THREE.Color(c));
  const glass = new THREE.Color(0x7f98ad);
  const col = new THREE.Color();
  for (let k = 0; k < n; k++) {
    const h1 = ((k * 2654435761) >>> 0) % 1000 / 1000;
    col.copy(palette[Math.floor(h1 * palette.length)]);
    col.multiplyScalar(0.82 + 0.30 * (((k * 1597334677) >>> 0) % 1000 / 1000));
    const hgt = rec[7 * k + 6];
    if (hgt > 35) col.lerp(glass, Math.min((hgt - 35) / 60, 0.55));
    mesh.setColorAt(k, col);
  }
  mesh.instanceColor.needsUpdate = true;
  state.buildings = { mesh, rec, n };
  updateBuildingMatrices();
  scene.add(mesh);
  const lbl = document.querySelector('label[for="bldg"]');
  if (lbl) lbl.textContent = `City buildings (${n.toLocaleString()})`;
  console.log(`[buildings] ${n.toLocaleString()} instances`);
}

function updateBuildingMatrices() {
  const b = state.buildings;
  if (!b) return;
  const { nx, ny, dx, zmin } = state;
  const s = state.worldScale;          // true scale: vertical == horizontal
  const cx = 0.5 * nx * dx * s;
  const cy = 0.5 * ny * dx * s;
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  for (let k = 0; k < b.n; k++) {
    const o = 7 * k;
    const x = b.rec[o], y = b.rec[o + 1];
    const w = b.rec[o + 2], l = b.rec[o + 3];
    const ang = b.rec[o + 4];
    const ground = b.rec[o + 5], hgt = b.rec[o + 6];
    // angle is from +x(east) toward +y(north); world north = -z, and
    // rotateY(α) maps +x → (cosα, 0, −sinα), so α = +angle lands on
    // (cosθ, 0, −sinθ) = east/north correctly.
    q.setFromAxisAngle(up, ang);
    pos.set(x * s - cx, (ground - zmin) * s + 0.5 * hgt * s, cy - y * s);
    scl.set(w * s, Math.max(hgt * s, 1e-6), l * s);
    m.compose(pos, q, scl);
    b.mesh.setMatrixAt(k, m);
  }
  b.mesh.instanceMatrix.needsUpdate = true;
}

// ----------------------------------------------------------------------
// Sky, lights, post-processing (AO + tone mapping)
// ----------------------------------------------------------------------
const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);
const sunLight = new THREE.DirectionalLight(0xfff2d8, 1.2);
scene.add(sunLight);
const hemi = new THREE.HemisphereLight(0xb6cce3, 0x4a4032, 0.40);
scene.add(hemi);

// Gradient sky dome with a sun glow — hand-tuned, plays nicely with ACES.
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: {
    zenith:    { value: new THREE.Color(0.30, 0.47, 0.70) },
    horizon:   { value: new THREE.Color(0.82, 0.80, 0.74) },
    sunDirSky: { value: new THREE.Vector3(0, 1, 0) },
  },
  vertexShader: /* glsl */`
    varying vec3 vDir;
    void main() {
      vDir = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec3 vDir;
    uniform vec3 zenith;
    uniform vec3 horizon;
    uniform vec3 sunDirSky;
    void main() {
      vec3 d = normalize(vDir);
      float h = clamp(d.y, 0.0, 1.0);
      vec3 c = mix(horizon, zenith, pow(h, 0.28));
      float cosSun = max(dot(d, sunDirSky), 0.0);
      c += vec3(1.0, 0.85, 0.60) * pow(cosSun, 16.0) * 0.35;   // haze glow
      c += vec3(1.0, 0.95, 0.85) * smoothstep(0.9991, 0.9997, cosSun) * 2.0;
      gl_FragColor = vec4(c, 1.0);
    }
  `,
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(9, 32, 16), skyMat);
sky.frustumCulled = false;
sky.renderOrder = -1;
scene.add(sky);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const n8ao = new N8AOPass(scene, camera, pane3d.clientWidth, pane3d.clientHeight);
n8ao.configuration.screenSpaceRadius = true;
n8ao.configuration.aoRadius = 40;          // px
n8ao.configuration.distanceFalloff = 1.0;
n8ao.configuration.intensity = 5.0;
n8ao.setQualityMode('Medium');
composer.addPass(n8ao);
composer.addPass(new OutputPass());
state.n8ao = n8ao;
state.composer = composer;

// Sun elevation drives the sky model, the lights, the shader sun vector,
// and the fog/sky tints (warm dusk at low angles, pale blue at noon).
const DUSK_FOG = new THREE.Color(0xc89a6e);
const DAY_FOG = new THREE.Color(0xc3d2e2);
const DUSK_SKYTINT = new THREE.Color(0xc99a6e);
const DAY_SKYTINT = new THREE.Color(0x7397b8);
function setSun(elevDeg) {
  const az = THREE.MathUtils.degToRad(215);          // SW
  const el = THREE.MathUtils.degToRad(elevDeg);
  // world: +x = east, north = -z
  const sd = new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    -Math.cos(az) * Math.cos(el),
  ).normalize();
  skyMat.uniforms.sunDirSky.value.copy(sd);
  sunLight.position.copy(sd).multiplyScalar(4);
  const t = THREE.MathUtils.clamp((elevDeg - 6) / 44, 0, 1);
  skyMat.uniforms.zenith.value.lerpColors(
    new THREE.Color(0.22, 0.28, 0.46), new THREE.Color(0.26, 0.44, 0.72), t);
  skyMat.uniforms.horizon.value.lerpColors(
    new THREE.Color(0.93, 0.64, 0.42), new THREE.Color(0.60, 0.69, 0.78), t);
  sunLight.color.setHSL(0.085, 0.55 * (1 - t), 0.62 + 0.3 * t);
  sunLight.intensity = 0.9 + 0.7 * t;
  ambient.intensity = 0.20 + 0.15 * t;
  hemi.intensity = 0.18 + 0.22 * t;
  scene.fog.color.lerpColors(DUSK_FOG, DAY_FOG, t);
  if (state.terrainMat) state.terrainMat.uniforms.sunDir.value.copy(sd);
  if (state.waterMat) {
    state.waterMat.uniforms.sunDir.value.copy(sd);
    state.waterMat.uniforms.skyTint.value.lerpColors(DUSK_SKYTINT, DAY_SKYTINT, t);
  }
  state.sunDir = sd;
}

// ----------------------------------------------------------------------
// UI bindings
// ----------------------------------------------------------------------
function bindRiverSlider(domId, river, labelId) {
  const el = document.getElementById(domId);
  const apply = (v) => {
    document.getElementById(labelId).textContent = `${v.toFixed(0)} m³/s`;
    const g = state.gauges.find(g => g.role === 'inflow' && g.river === river);
    if (g && state.sim) state.sim.setInflow(river, g.rect, v);
  };
  el.addEventListener('input', (e) => apply(parseFloat(e.target.value)));
  state.sliderApply[river] = (v) => { el.value = v; apply(v); };
}

function setupUI() {
  bindRiverSlider('potomac',      'potomac',      'potomac-label');
  bindRiverSlider('anacostia_nw', 'anacostia_nw', 'anacostia_nw-label');
  bindRiverSlider('anacostia_ne', 'anacostia_ne', 'anacostia_ne-label');
  bindRiverSlider('rock_creek',   'rock_creek',   'rock_creek-label');

  document.getElementById('sun').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById('sun-label').textContent = `${v.toFixed(0)}°`;
    setSun(v);
  });

  document.getElementById('alpha').addEventListener('input', (e) => {
    state.alphaScale = parseFloat(e.target.value);
    document.getElementById('alpha-label').textContent = state.alphaScale.toFixed(2);
    if (state.waterMat) state.waterMat.uniforms.alphaScale.value = state.alphaScale;
  });

  document.getElementById('speed').addEventListener('input', (e) => {
    state.subSteps = parseInt(e.target.value, 10);
    document.getElementById('speed-label').textContent =
      state.subSteps === 0 ? 'paused' : `${state.subSteps} steps/frame`;
  });

  document.getElementById('bldg').addEventListener('change', (e) => {
    if (state.buildings) state.buildings.mesh.visible = e.target.checked;
  });

  document.getElementById('sat').addEventListener('change', (e) => {
    if (state.terrainMat && state.satTex) {
      state.terrainMat.uniforms.useSat.value = e.target.checked ? 1 : 0;
    }
  });

  document.getElementById('btn-pause').addEventListener('click', (e) => {
    state.paused = !state.paused;
    e.currentTarget.textContent = state.paused ? 'Resume' : 'Pause';
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    state.sim.reset();
  });

  document.getElementById('btn-flood').addEventListener('click', () => {
    // Roughly 100-yr peaks (Little Falls ~12,000 m³/s; Anacostia branches
    // ~1,200 each; Rock Creek major events ~250).
    state.sliderApply.potomac?.(12000);
    state.sliderApply.anacostia_nw?.(1200);
    state.sliderApply.anacostia_ne?.(1200);
    state.sliderApply.rock_creek?.(250);
  });

  // Shift-click rain (raycast against the terrain; world XZ → cell indices)
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  renderer.domElement.addEventListener('click', (e) => {
    if (!e.shiftKey || !state.terrainMesh) return;
    const rect = renderer.domElement.getBoundingClientRect();
    ndc.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    ndc.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hits = raycaster.intersectObject(state.terrainMesh, false);
    if (!hits.length) return;
    const { nx, ny, dx, worldScale } = state;
    const s = worldScale;
    const cx = 0.5 * nx * dx * s;
    const cy = 0.5 * ny * dx * s;
    const p = hits[0].point;
    const i = Math.round((p.x + cx) / (dx * s));
    const j = Math.round((cy - p.z) / (dx * s));
    if (i < 0 || i >= nx || j < 0 || j >= ny) return;
    state.sim.splash(i, j, Math.max(8, Math.floor(Math.min(nx, ny) / 40)), 2.0);
  });
}

// ----------------------------------------------------------------------
// USGS NWIS: live discharge for the inflow sliders + tidal stage for the
// initial ("base") water level. Auto-refreshes; falls back when offline.
// ----------------------------------------------------------------------
const FT_TO_M = 0.3048;
const CFS_TO_CMS = 0.0283168;
// Tidal gauges whose gage datum is NAVD88 (≈ the DEM's vertical datum):
// Potomac @ Wisconsin Ave, Anacostia @ Buzzard Point.
const STAGE_SITES = ['01647600', '01651827'];
const USGS_REFRESH_MS = 10 * 60 * 1000;

async function fetchUSGS(meta) {
  const inflowIds = meta.gauges.filter(g => g.role === 'inflow').map(g => g.id);
  const sites = [...inflowIds, ...STAGE_SITES].join(',');
  const url = 'https://waterservices.usgs.gov/nwis/iv/?format=json' +
              `&sites=${sites}&parameterCd=00060,00065&siteStatus=all`;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 6000);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    if (!r.ok) throw new Error(`NWIS HTTP ${r.status}`);
    const j = await r.json();
    const out = { discharge: {}, stage: {} };
    for (const ts of j.value?.timeSeries ?? []) {
      const site = ts.sourceInfo?.siteCode?.[0]?.value;
      const param = ts.variable?.variableCode?.[0]?.value;
      const v = parseFloat(ts.values?.[0]?.value?.[0]?.value);
      if (!Number.isFinite(v) || v < 0) continue;
      if (param === '00060') out.discharge[site] = v * CFS_TO_CMS;
      else if (param === '00065') out.stage[site] = v * FT_TO_M;
    }
    return out;
  } finally {
    clearTimeout(timer);
  }
}

function applyLiveFlows(meta, usgs) {
  let n = 0;
  for (const g of meta.gauges) {
    if (g.role !== 'inflow' || usgs.discharge[g.id] == null) continue;
    // Exact live value (slider steps would snap small creek flows to zero).
    state.sliderApply[g.river]?.(usgs.discharge[g.id]);
    n++;
  }
  return n;
}

// Fill every cell hydraulically connected to the river seeds whose bed sits
// below the water-surface elevation `wse`. BFS keeps isolated inland pits dry.
function computeBaseDepth(z, nx, ny, wse, seeds) {
  const h0 = new Float32Array(nx * ny);
  const visited = new Uint8Array(nx * ny);
  const queue = new Int32Array(nx * ny);
  let qh = 0, qt = 0;
  for (const k of seeds) {
    if (k >= 0 && k < z.length && z[k] < wse && !visited[k]) {
      visited[k] = 1; queue[qt++] = k;
    }
  }
  while (qh < qt) {
    const k = queue[qh++];
    h0[k] = wse - z[k];
    const i = k % nx, j = (k / nx) | 0;
    if (i > 0      && !visited[k - 1]  && z[k - 1]  < wse) { visited[k - 1]  = 1; queue[qt++] = k - 1; }
    if (i < nx - 1 && !visited[k + 1]  && z[k + 1]  < wse) { visited[k + 1]  = 1; queue[qt++] = k + 1; }
    if (j > 0      && !visited[k - nx] && z[k - nx] < wse) { visited[k - nx] = 1; queue[qt++] = k - nx; }
    if (j < ny - 1 && !visited[k + nx] && z[k + nx] < wse) { visited[k + nx] = 1; queue[qt++] = k + nx; }
  }
  return h0;
}

async function applyUSGSBaseline(meta, z) {
  const statEl = document.getElementById('s-usgs');
  let usgs = null;
  try {
    usgs = await fetchUSGS(meta);
  } catch (e) {
    console.warn('[usgs] live fetch failed:', e.message);
  }

  const liveGauges = usgs ? applyLiveFlows(meta, usgs) : 0;

  // Tidal stage → base water-surface elevation (NAVD88 m).
  let wse = 0.6;
  if (usgs) {
    for (const s of STAGE_SITES) {
      if (usgs.stage[s] != null) { wse = usgs.stage[s]; break; }
    }
  }
  wse = Math.min(Math.max(wse, 0.15), 2.5);   // sanity clamp (m NAVD88)

  // Seeds: known in-river stage gauges + the deepest cell in the domain.
  const seeds = meta.gauges.filter(g => g.role === 'stage')
                           .map(g => g.j * meta.nx + g.i);
  let kmin = 0;
  for (let k = 1; k < z.length; k++) if (z[k] < z[kmin]) kmin = k;
  seeds.push(kmin);

  state.sim.setInitialWater(computeBaseDepth(z, meta.nx, meta.ny, wse, seeds));
  const stamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  statEl.textContent = usgs
    ? `live · ${liveGauges} flows · stage ${wse.toFixed(2)} m · ${stamp}`
    : `offline defaults · stage ${wse.toFixed(2)} m`;

  // Keep the inflows tracking the river: re-poll NWIS periodically.
  setInterval(async () => {
    try {
      const fresh = await fetchUSGS(meta);
      const n = applyLiveFlows(meta, fresh);
      const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      statEl.textContent = `live · ${n} flows · stage ${wse.toFixed(2)} m · ${t}`;
    } catch (e) {
      console.warn('[usgs] refresh failed:', e.message);
    }
  }, USGS_REFRESH_MS);
}

// ----------------------------------------------------------------------
// Main loop
// ----------------------------------------------------------------------
function tick(now) {
  controls.update();
  const sim = state.sim;
  if (sim) {
    if (!state.paused && state.subSteps > 0) sim.step(state.subSteps);

    if (now - state.lastDiag > 250) {
      const d = sim.readDiagnostics();
      if (d.nan) {
        console.warn('NaN detected in sim state — resetting');
        sim.reset();
      } else {
        document.getElementById('s-hmax').textContent = `${d.hmax.toFixed(2)} m`;
        document.getElementById('s-vol').textContent =
          `${Math.round(d.volume).toLocaleString()} m³`;
      }
      const wall = (now - state.lastRateT) / 1000;
      if (wall > 0) state.simRate = (sim.simTime - state.lastSimTime) / wall;
      state.lastSimTime = sim.simTime;
      state.lastRateT = now;
      document.getElementById('s-time').textContent = `${sim.simTime.toFixed(0)} s`;
      document.getElementById('s-rate').textContent =
        `${state.simRate.toFixed(0)}× real time`;
      document.getElementById('s-inflow').textContent =
        `${Math.round(sim.totalInflow())} m³/s`;
      state.lastDiag = now;
    }
    if (state.waterMat) {
      state.waterMat.uniforms.hTex.value = sim.hTexture;
      state.waterMat.uniforms.qTex.value = sim.qTexture;
      state.waterMat.uniforms.viewerPos.value.copy(camera.position);
      state.waterMat.uniforms.time.value = now / 1000;
    }
  }
  composer.render();

  state.framesSinceLog++;
  const dtLog = now - state.lastFpsLog;
  if (dtLog > 500) {
    document.getElementById('s-fps').textContent =
      `${((state.framesSinceLog * 1000) / dtLog).toFixed(0)} fps`;
    state.framesSinceLog = 0;
    state.lastFpsLog = now;
  }
  requestAnimationFrame(tick);
}

// ----------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------
async function boot() {
  const meta = await (await fetch('data/meta.json')).json();
  const zRaw = new Uint16Array(await fetchBinary('data/dem_z.u16.gz'));
  if (zRaw.length !== meta.nx * meta.ny) {
    throw new Error(`DEM size mismatch: ${zRaw.length} != ${meta.nx}×${meta.ny}`);
  }
  const z = new Float32Array(zRaw.length);
  for (let k = 0; k < zRaw.length; k++) z[k] = meta.zmin + zRaw[k] * meta.zscale;

  Object.assign(state, {
    nx: meta.nx, ny: meta.ny, dx: meta.dx,
    zmin: meta.zmin, zmax: meta.zmax,
    zRange: Math.max(meta.zmax - meta.zmin, 1.0),
    west: meta.west, south: meta.south, east: meta.east, north: meta.north,
    gauges: meta.gauges,
  });

  state.sim = new GPUFloodSim(renderer, {
    nx: meta.nx, ny: meta.ny, dx: meta.dx, z, manning: meta.manning,
  });

  // Default inflows from gauge metadata (replaced by live USGS values below)
  for (const g of meta.gauges) {
    if (g.role === 'inflow' && g.rect) {
      state.sim.setInflow(g.river, g.rect, g.default_cms);
    }
  }

  document.getElementById('s-grid').textContent =
    `${meta.nx}×${meta.ny}  Δx=${meta.dx.toFixed(0)} m`;

  buildScene();
  setupUI();
  setSun(28);
  loadBasemap();
  document.getElementById('loading').style.display = 'none';

  // Base water level + live inflows from USGS (async; falls back silently).
  applyUSGSBaseline(meta, z).catch(e => console.warn('[usgs]', e));

  // 3D city layer (async, optional).
  loadBuildings();

  requestAnimationFrame(tick);
}

window.FLOOD = state;        // debugging handle
window.FLOOD_TICK = (n) => tick(n ?? performance.now());
boot().catch(e => { console.error(e); fatal(`Failed to start: ${e.message}`); });
