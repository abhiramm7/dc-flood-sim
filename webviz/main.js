// Browser viewer for the Key Bridge flood sim.
// WebFlood-style rendering: terrain + water are flat plane meshes whose
// vertex Y is read from a height DataTexture in the vertex shader, and
// per-pixel hillshade normals are computed in the fragment shader from
// the same texture's gradient. The CPU just shovels h floats into a
// DataTexture each frame — no vertex-buffer rebuilds.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ----------------------------------------------------------------------
const state = {
  nx: 0, ny: 0, dx: 1,
  zmin: 0, zmax: 1, zRange: 1,
  west: 0, south: 0, east: 0, north: 0,
  map: null, floodCanvas: null, floodCtx: null, floodOverlay: null,
  floodplainGeo: null, floodplainLayer: null,
  lastMapUpdate: 0,
  zExag: 1.0,
  yScale: 1.0 * 0.2,           // world Y units per (z-zmin)/zRange
  alphaScale: 0.85,
  worldScale: 1.0,
  worldDx: 1.0,
  z: null, h: null,
  zTex: null, wTex: null,
  terrainMesh: null, waterMesh: null,
  terrainMat: null, waterMat: null,
  gauges: [],
  framesSinceLog: 0,
  lastFpsLog: performance.now(),
  fps: 0,
};

const SUN = new THREE.Vector3(0.40, 0.82, 0.42).normalize();

// ----------------------------------------------------------------------
// WebSocket + bridge fetch
// ----------------------------------------------------------------------
const ws = (() => {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return new WebSocket(`${proto}://${location.host}/ws`);
})();
ws.binaryType = 'arraybuffer';

const statusEl = document.getElementById('status');
ws.addEventListener('open',  () => { statusEl.textContent = 'connected'; statusEl.classList.add('connected'); });
ws.addEventListener('close', () => { statusEl.textContent = 'disconnected'; statusEl.classList.remove('connected'); statusEl.classList.add('error'); });
ws.addEventListener('error', (e) => console.error('ws error', e));

// ----------------------------------------------------------------------
// Renderer + scene
// ----------------------------------------------------------------------
const pane3d = document.getElementById('pane-3d');
const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(pane3d.clientWidth, pane3d.clientHeight);
renderer.setClearColor(0x0a0c10, 1);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0a0c10, 4, 14);

const camera = new THREE.PerspectiveCamera(45, pane3d.clientWidth / pane3d.clientHeight, 0.01, 50);
camera.position.set(1.1, 0.9, 1.3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.3;
controls.maxDistance = 8;
controls.update();

// North arrow (red=east, cyan=north)
{
  const arrow = new THREE.Group();
  const matN = new THREE.MeshBasicMaterial({ color: 0x4ecdc4 });
  const matE = new THREE.MeshBasicMaterial({ color: 0xff6b6b });
  const make = (mat, dir) => {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.20, 8), mat);
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.018, 0.04, 12), mat);
    if (dir === 'z') {
      arm.rotation.x = Math.PI/2; arm.position.set(0, 0, 0.10);
      head.rotation.x = Math.PI/2; head.position.set(0, 0, 0.22);
    } else {
      arm.rotation.z = Math.PI/2; arm.position.set(0.10, 0, 0);
      head.rotation.z = -Math.PI/2; head.position.set(0.22, 0, 0);
    }
    arrow.add(arm, head);
  };
  make(matN, 'z'); make(matE, 'x');
  arrow.position.set(-1.0, 0.01, -0.9);
  scene.add(arrow);
}

window.addEventListener('resize', () => {
  camera.aspect = pane3d.clientWidth / pane3d.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(pane3d.clientWidth, pane3d.clientHeight);
  if (state.map) state.map.invalidateSize();
});

// ----------------------------------------------------------------------
// GLSL shaders (WebFlood-style: vertex-shader displacement + frag hillshade)
// ----------------------------------------------------------------------
const TERRAIN_VS = /* glsl */`
  uniform sampler2D zTex;
  uniform float zmin;
  uniform float zRange;
  uniform float yScale;
  varying vec2 vUv;
  varying float vBedZ;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    float z = texture2D(zTex, uv).r;
    vBedZ = z;
    float y = (z - zmin) / zRange * yScale;
    vec3 newPos = vec3(position.x, y, position.z);
    vec4 wp = modelMatrix * vec4(newPos, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const TERRAIN_FS = /* glsl */`
  precision highp float;
  uniform sampler2D zTex;
  uniform float texelU;     // 1/(nx-1)
  uniform float texelV;     // 1/(ny-1)
  uniform float zmin;
  uniform float zRange;
  uniform float yScale;
  uniform float worldDx;
  uniform vec3 sunDir;
  varying vec2 vUv;
  varying float vBedZ;
  varying vec3 vWorldPos;

  vec3 elevRamp(float zMeters) {
    // Use ABSOLUTE elevation in metres (more meaningful than normalized t):
    //   z < 3 m   → river bed / tidal mudflat (blue-grey)  -> "delineates" the rivers
    //   3..15 m  → low floodplain (dark green)
    //   15..40 m → wooded plain (meadow green)
    //   40..80 m → uplands (tan / sandstone)
    //   > 80 m   → ridgetops (off-white)
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
    // Sample 4 neighbors of the height texture to build a world-space normal.
    float zL = texture2D(zTex, vec2(vUv.x - texelU, vUv.y)).r;
    float zR = texture2D(zTex, vec2(vUv.x + texelU, vUv.y)).r;
    float zB = texture2D(zTex, vec2(vUv.x, vUv.y - texelV)).r;
    float zT = texture2D(zTex, vec2(vUv.x, vUv.y + texelV)).r;
    float slopeX = (zR - zL) * yScale / (2.0 * worldDx * zRange);
    float slopeZ = (zT - zB) * yScale / (2.0 * worldDx * zRange);
    vec3 normal = normalize(vec3(-slopeX, 1.0, -slopeZ));

    float diff = max(dot(normal, sunDir), 0.0);
    vec3 albedo = elevRamp(vBedZ);
    float skyFill = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
    vec3 color = albedo * (0.35 + 0.85 * diff + 0.10 * skyFill);
    gl_FragColor = vec4(color, 1.0);
  }
`;

const WATER_VS = /* glsl */`
  uniform sampler2D zTex;     // bed
  uniform sampler2D wTex;     // surface = z + h
  uniform float zmin;
  uniform float zRange;
  uniform float yScale;
  varying vec2 vUv;
  varying float vDepth;
  varying float vSurfaceZ;
  varying vec3 vWorldPos;
  void main() {
    vUv = uv;
    float zb = texture2D(zTex, uv).r;
    float zw = texture2D(wTex, uv).r;
    float h = max(zw - zb, 0.0);
    vDepth = h;
    // Blend between terrain (dry) and water surface (wet). Without this,
    // dry vertices sit at bed elevation and wet vertices at lake level,
    // producing tall vertical cliffs at every shoreline — those are the
    // spikes you were seeing. With smoothstep blending, dry vertices ride
    // flush with terrain, wet vertices ride at the actual surface, and
    // the transition takes a couple of cells to ramp.
    float blend = smoothstep(0.1, 1.5, h);
    float yEff = mix(zb, zw, blend);
    vSurfaceZ = yEff;
    float y = (yEff - zmin) / zRange * yScale;
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
  uniform float zmin;
  uniform float zRange;
  varying vec2 vUv;
  varying float vDepth;
  varying float vSurfaceZ;
  varying vec3 vWorldPos;

  void main() {
    if (vDepth < 0.08) discard;

    // Normal from screen-space derivatives of world position. Because the
    // vertex shader already smoothed bed→surface across the shoreline,
    // dFdx/dFdy give a clean normal that matches the rendered surface
    // (no texture-gradient cliff artifacts).
    vec3 dpdx = dFdx(vWorldPos);
    vec3 dpdy = dFdy(vWorldPos);
    vec3 normal = normalize(cross(dpdy, dpdx));
    if (normal.y < 0.0) normal = -normal;

    // Color tinted by absolute surface elevation (WebFlood-style).
    float surfN = clamp((vSurfaceZ - zmin) / zRange, 0.0, 1.0);
    vec3 deep    = vec3(0.04, 0.30, 0.55);
    vec3 shallow = vec3(0.50, 0.82, 0.95);
    vec3 base = mix(deep, shallow, surfN);

    float diff = max(dot(normal, sunDir), 0.0);
    vec3 view = normalize(viewerPos - vWorldPos);
    vec3 H = normalize(sunDir + view);
    float spec = pow(max(dot(normal, H), 0.0), 96.0);

    vec3 color = base * (0.50 + 0.55 * diff);
    color += vec3(0.95, 0.97, 1.0) * spec * 0.85;

    float fres = pow(1.0 - max(dot(normal, view), 0.0), 4.0);
    color += vec3(0.30, 0.50, 0.65) * fres * 0.20;

    float a = smoothstep(0.05, 1.0, vDepth) * alphaScale;
    gl_FragColor = vec4(color, a);
  }
`;

// ----------------------------------------------------------------------
// Data-texture helpers
// ----------------------------------------------------------------------
function makeHeightTexture(nx, ny) {
  // RGBA32F so we don't depend on RedFormat support quirks. We store the
  // elevation in the R channel; G/B/A are unused (set to 0).
  const data = new Float32Array(nx * ny * 4);
  const tex = new THREE.DataTexture(data, nx, ny, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return { tex, data };
}

// Python sends z and h as `(nx, ny)` C-order, so arr[i*ny + j] is solver cell
// (i, j). Three.js DataTexture treats `data` as row-major with width=nx,
// meaning pixel (x=i, y=j) lives at data[(j*nx + i) * 4]. Therefore we have
// to *transpose* on upload — otherwise the height field gets aliased into a
// striped/spiky pattern when nx != ny.
function uploadHeightToTexture(arr, texObj) {
  const data = texObj.data;
  const nx = state.nx, ny = state.ny;
  for (let i = 0; i < nx; i++) {
    const srcRow = i * ny;
    for (let j = 0; j < ny; j++) {
      data[(j * nx + i) * 4] = arr[srcRow + j];
    }
  }
  texObj.tex.needsUpdate = true;
}

function uploadSumToTexture(zArr, hArr, texObj) {
  const data = texObj.data;
  const nx = state.nx, ny = state.ny;
  for (let i = 0; i < nx; i++) {
    const srcRow = i * ny;
    for (let j = 0; j < ny; j++) {
      data[(j * nx + i) * 4] = zArr[srcRow + j] + hArr[srcRow + j];
    }
  }
  texObj.tex.needsUpdate = true;
}

// ----------------------------------------------------------------------
// Mesh construction. Flat XZ plane; Y comes from the height texture.
// ----------------------------------------------------------------------
function buildScene() {
  const { nx, ny, dx, z } = state;
  const worldW = nx * dx;
  const worldH = ny * dx;
  state.worldScale = 2.0 / Math.max(worldW, worldH);
  state.worldDx = dx * state.worldScale;
  const s = state.worldScale;
  const cx = 0.5 * worldW * s;
  const cy = 0.5 * worldH * s;

  // Build a flat plane mesh: positions at (x, 0, z), UVs (i/(nx-1), j/(ny-1))
  const nv = nx * ny;
  const positions = new Float32Array(nv * 3);
  const uvs = new Float32Array(nv * 2);
  for (let i = 0; i < nx; i++) {
    const x = i * dx * s - cx;
    const u = i / (nx - 1);
    for (let j = 0; j < ny; j++) {
      const idx = i * ny + j;
      const y = j * dx * s - cy;
      positions[3*idx + 0] = x;
      positions[3*idx + 1] = 0;
      positions[3*idx + 2] = y;
      uvs[2*idx + 0] = u;
      uvs[2*idx + 1] = j / (ny - 1);
    }
  }
  const nq = (nx - 1) * (ny - 1);
  const indices = new (nv > 65535 ? Uint32Array : Uint16Array)(nq * 6);
  let k = 0;
  for (let i = 0; i < nx - 1; i++) {
    for (let j = 0; j < ny - 1; j++) {
      const v00 = i * ny + j;
      const v10 = (i + 1) * ny + j;
      const v01 = i * ny + (j + 1);
      const v11 = (i + 1) * ny + (j + 1);
      indices[k++] = v00; indices[k++] = v10; indices[k++] = v11;
      indices[k++] = v00; indices[k++] = v11; indices[k++] = v01;
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('uv',       new THREE.BufferAttribute(uvs, 2));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));

  // Height textures
  state.zTex = makeHeightTexture(nx, ny);
  state.wTex = makeHeightTexture(nx, ny);
  uploadHeightToTexture(z, state.zTex);
  // Initialize water surface = bed (water depth zero everywhere)
  uploadHeightToTexture(z, state.wTex);

  // Terrain material — pure hillshade + elevation ramp, no satellite overlay.
  const tUniforms = {
    zTex:    { value: state.zTex.tex },
    texelU:  { value: 1.0 / (nx - 1) },
    texelV:  { value: 1.0 / (ny - 1) },
    zmin:    { value: state.zmin },
    zRange:  { value: state.zRange },
    yScale:  { value: state.yScale },
    worldDx: { value: state.worldDx },
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

  // Material: water (shares the geometry topology)
  const wGeom = new THREE.BufferGeometry();
  wGeom.setAttribute('position', new THREE.BufferAttribute(positions.slice(), 3));
  wGeom.setAttribute('uv',       new THREE.BufferAttribute(uvs.slice(), 2));
  wGeom.setIndex(geom.getIndex());
  const wUniforms = {
    zTex:       { value: state.zTex.tex },
    wTex:       { value: state.wTex.tex },
    texelU:     { value: 1.0 / (nx - 1) },
    texelV:     { value: 1.0 / (ny - 1) },
    zmin:       { value: state.zmin },
    zRange:     { value: state.zRange },
    yScale:     { value: state.yScale },
    worldDx:    { value: state.worldDx },
    alphaScale: { value: state.alphaScale },
    sunDir:     { value: SUN.clone() },
    viewerPos:  { value: new THREE.Vector3() },
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

// Scene lights — for the building mesh (MeshStandardMaterial). The terrain
// and water shade themselves via their custom shaders + the SUN uniform.
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const _sun = new THREE.DirectionalLight(0xfff2d8, 1.0);
_sun.position.set(SUN.x * 4, SUN.y * 4, SUN.z * 4);
scene.add(_sun);
scene.add(new THREE.HemisphereLight(0xb6cce3, 0x4a4032, 0.40));

const statGauges = document.getElementById('s-gauges');
const setStatus = (msg) => { if (statGauges) statGauges.textContent = msg; };

async function loadGauges() {
  try {
    const r = await fetch('./gauges.json');
    if (!r.ok) { setStatus(`HTTP ${r.status}`); return; }
    const d = await r.json();
    state.gauges = d.gauges || [];
    setStatus(`${state.gauges.length} loaded`);
    if (state.map) plotGaugeMarkers();
  } catch (e) {
    setStatus(`error: ${e.message}`);
    console.error('[gauges] load', e);
  }
}
loadGauges();

async function loadFloodplain() {
  try {
    const r = await fetch('./fema_floodplain.geojson');
    if (!r.ok) return;
    state.floodplainGeo = await r.json();
    if (state.map) addFloodplainLayer();
  } catch (e) { console.error('[fema] load', e); }
}
loadFloodplain();

function addFloodplainLayer() {
  if (!state.map || !state.floodplainGeo || state.floodplainLayer) return;
  state.floodplainLayer = L.geoJSON(state.floodplainGeo, {
    style: f => {
      const z  = f.properties.FLD_ZONE;
      const sb = f.properties.ZONE_SUBTY;
      if (sb === 'FLOODWAY') {
        return { color: '#ff5b35', weight: 0.4,
                 fillColor: '#ff5b35', fillOpacity: 0.32 };
      }
      if (z === 'AE' || z === 'A' || z === 'AH' || z === 'AO' || z === 'VE') {
        return { color: '#d97a4a', weight: 0.3,
                 fillColor: '#d97a4a', fillOpacity: 0.18 };
      }
      return { color: '#888', weight: 0.3, fillOpacity: 0.10 };
    },
    interactive: false,
  }).addTo(state.map);
  // Make sure the live flood overlay (and gauge markers, added later) draw
  // on top of the FEMA polygons.
  if (state.floodOverlay && state.floodOverlay.bringToFront)
    state.floodOverlay.bringToFront();
}

function plotGaugeMarkers() {
  if (!state.map || !state.gauges) return;
  for (const g of state.gauges) {
    const inflow = g.role === 'inflow';
    const m = L.circleMarker([g.lat, g.lon], {
      radius:        inflow ? 8       : 5,
      color:         inflow ? '#ffd28a' : '#a5d6ff',
      fillColor:     inflow ? '#fb923c' : '#3b82f6',
      fillOpacity:   inflow ? 0.92    : 0.78,
      weight:        2,
    }).addTo(state.map);
    const role = inflow ? 'inflow source' : 'stage observation';
    m.bindTooltip(
      `<b>${g.name}</b><br>USGS ${g.id} · ${role}<br>` +
      `<span style="color:#888">cell (${g.i}, ${g.j}) · bed ${g.elev_m.toFixed(1)} m</span>`,
      { direction: 'top', offset: [0, -4] }
    );
  }
}

// ----------------------------------------------------------------------
// 2D map pane (Leaflet) + flood depth overlay.
// ----------------------------------------------------------------------
function initMapPane() {
  if (state.map) return;     // only do this once
  if (!window.L) { console.warn('Leaflet failed to load'); return; }
  const { west, south, east, north, nx, ny } = state;
  const bounds = L.latLngBounds([south, west], [north, east]);

  state.map = L.map('pane-map', {
    preferCanvas: true,
    zoomControl: true,
    attributionControl: false,
  }).fitBounds(bounds);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png', {
    subdomains: 'abcd', maxZoom: 19,
  }).addTo(state.map);

  state.floodCanvas = document.createElement('canvas');
  state.floodCanvas.width = nx;
  state.floodCanvas.height = ny;
  state.floodCtx = state.floodCanvas.getContext('2d');
  // Image overlay using the canvas's data URL — we'll refresh src each tick.
  state.floodOverlay = L.imageOverlay(state.floodCanvas.toDataURL(),
                                      bounds, { opacity: 0.78, interactive: false })
                        .addTo(state.map);
  if (state.floodplainGeo) addFloodplainLayer();
}

function updateFloodOverlay() {
  const { nx, ny, h, floodCtx, floodCanvas, floodOverlay } = state;
  if (!floodCtx || !h) return;
  const img = floodCtx.createImageData(nx, ny);
  const d = img.data;
  // Solver layout: h[i*ny + j] with i=east, j=north.
  // Canvas pixel (px, py) where py=0 is the top of the image (=north).
  // So  px = i,  py = ny - 1 - j.
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const depth = h[i * ny + j];
      const px = i;
      const py = ny - 1 - j;
      const off = (py * nx + px) * 4;
      if (depth > 0.1) {
        // Color ramp: shallow cyan → deep navy. Alpha grows with depth.
        const t = Math.min(depth / 5.0, 1.0);
        d[off + 0] = Math.round((0.18 - 0.10 * t) * 255);
        d[off + 1] = Math.round((0.55 - 0.20 * t) * 255);
        d[off + 2] = Math.round((0.90 - 0.05 * t) * 255);
        d[off + 3] = Math.round(Math.min(0.85, 0.30 + depth * 0.10) * 255);
      } else {
        d[off + 3] = 0;
      }
    }
  }
  floodCtx.putImageData(img, 0, 0);
  floodOverlay._image.src = floodCanvas.toDataURL();
}

// ----------------------------------------------------------------------
// WebSocket protocol parsing
// ----------------------------------------------------------------------
const INIT_TAG  = 0x01;
const FRAME_TAG = 0x02;

ws.addEventListener('message', (ev) => {
  const buf = ev.data;
  if (!(buf instanceof ArrayBuffer)) return;
  const view = new DataView(buf);
  const tag = view.getInt32(0, true);

  if (tag === INIT_TAG) {
    // 48-byte header — see server's build_init_packet().
    const nx = view.getInt32(4, true);
    const ny = view.getInt32(8, true);
    const dx = view.getFloat32(12, true);
    const zmin = view.getFloat32(16, true);
    const zmax = view.getFloat32(20, true);
    const manning = view.getFloat32(24, true);
    const west  = view.getFloat32(28, true);
    const south = view.getFloat32(32, true);
    const east  = view.getFloat32(36, true);
    const north = view.getFloat32(40, true);
    const zBytes = view.getUint32(44, true);
    const z = new Float32Array(buf, 48, zBytes / 4).slice();
    state.nx = nx; state.ny = ny; state.dx = dx;
    state.zmin = zmin; state.zmax = zmax;
    state.zRange = Math.max(zmax - zmin, 1.0);
    state.west = west; state.south = south; state.east = east; state.north = north;
    state.z = z;
    state.h = new Float32Array(nx * ny);
    state.yScale = state.zExag * 0.2;
    document.getElementById('s-grid').textContent = `${nx}×${ny}  Δx=${dx.toFixed(1)} m`;
    console.log('init', { nx, ny, dx, zmin, zmax, west, south, east, north });

    // Rebuild scene
    if (state.terrainMesh) {
      scene.remove(state.terrainMesh); state.terrainMesh.geometry.dispose();
    }
    if (state.waterMesh) {
      scene.remove(state.waterMesh); state.waterMesh.geometry.dispose();
    }
    buildScene();
    initMapPane();
    plotGaugeMarkers();

  } else if (tag === FRAME_TAG) {
    const sim_t  = view.getFloat32(4, true);
    const hmax   = view.getFloat32(8, true);
    const vol    = view.getFloat32(12, true);
    const inflow = view.getFloat32(16, true);
    state.h = new Float32Array(buf, 20);
    if (state.wTex && state.z) {
      uploadSumToTexture(state.z, state.h, state.wTex);
    }
    // Throttle the 2D-map update — toDataURL costs ~5 ms and we don't need
    // 30 Hz on the map.
    const now = performance.now();
    if (state.floodCtx && (now - state.lastMapUpdate) > 220) {
      updateFloodOverlay();
      state.lastMapUpdate = now;
    }
    document.getElementById('s-time').textContent   = `${sim_t.toFixed(1)} s`;
    document.getElementById('s-hmax').textContent   = `${hmax.toFixed(2)} m`;
    document.getElementById('s-vol').textContent    = `${Math.round(vol).toLocaleString()} m³`;
    document.getElementById('s-inflow').textContent = `${Math.round(inflow)} m³/s`;
  }
});

// ----------------------------------------------------------------------
// UI bindings
// ----------------------------------------------------------------------
const sendCmd = (obj) => {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
};

function bindRiverSlider(domId, riverName, labelId, unit = 'm³/s') {
  const el = document.getElementById(domId);
  el.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    document.getElementById(labelId).textContent = `${v.toFixed(0)} ${unit}`;
    sendCmd({ cmd: 'inflow', river: riverName, value: v });
  });
}
bindRiverSlider('potomac',      'potomac',      'potomac-label');
bindRiverSlider('anacostia_nw', 'anacostia_nw', 'anacostia_nw-label');
bindRiverSlider('anacostia_ne', 'anacostia_ne', 'anacostia_ne-label');
bindRiverSlider('rock_creek',   'rock_creek',   'rock_creek-label');


document.getElementById('alpha').addEventListener('input', (e) => {
  state.alphaScale = parseFloat(e.target.value);
  document.getElementById('alpha-label').textContent = state.alphaScale.toFixed(2);
  if (state.waterMat) state.waterMat.uniforms.alphaScale.value = state.alphaScale;
});

document.getElementById('btn-pause').addEventListener('click', (e) => {
  const paused = e.currentTarget.textContent.trim() === 'Resume';
  sendCmd({ cmd: 'pause', value: !paused });
  e.currentTarget.textContent = paused ? 'Pause' : 'Resume';
});

document.getElementById('btn-reset').addEventListener('click', () => sendCmd({ cmd: 'reset' }));

document.getElementById('btn-flood').addEventListener('click', () => {
  // Roughly 100-yr peaks at each gauge (Little Falls peak ~12,000 m³/s;
  // Anacostia branches each ~1,000–1,500; Rock Creek major events ~250).
  const setSlider = (id, v) => {
    const el = document.getElementById(id);
    el.value = v; el.dispatchEvent(new Event('input'));
  };
  setSlider('potomac',      12000);
  setSlider('anacostia_nw',  1200);
  setSlider('anacostia_ne',  1200);
  setSlider('rock_creek',     250);
});

// Shift-click rain (raycast against flat plane; XZ -> cell indices)
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
  const j = Math.round((p.z + cy) / (dx * s));
  if (i < 0 || i >= nx || j < 0 || j >= ny) return;
  sendCmd({ cmd: 'rain', i, j, radius: Math.max(8, Math.floor(Math.min(nx, ny) / 40)), depth: 2.0 });
});

// ----------------------------------------------------------------------
// Render loop
// ----------------------------------------------------------------------
function tick(now) {
  controls.update();
  if (state.waterMat) state.waterMat.uniforms.viewerPos.value.copy(camera.position);
  renderer.render(scene, camera);
  state.framesSinceLog++;
  const dtLog = now - state.lastFpsLog;
  if (dtLog > 500) {
    state.fps = (state.framesSinceLog * 1000) / dtLog;
    document.getElementById('s-fps').textContent = `${state.fps.toFixed(0)} fps`;
    state.framesSinceLog = 0;
    state.lastFpsLog = now;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
