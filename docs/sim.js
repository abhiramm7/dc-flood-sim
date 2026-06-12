// GPU shallow-water solver — WebFlood-style: the entire local-inertial
// (LISFLOOD-FP) scheme from 03_solver_swe.py, ported to WebGL2 fragment
// shaders with ping-pong float render targets. No server needed; the
// browser's GPU does the hydraulics.
//
// Scheme (identical to the Taichi solver):
//   face flux   q_new = (q - g·h_face·dt·d(z+h)/dx)
//                       / (1 + g·dt·n²·|q| / h_face^(7/3))
//   mass        h_new = h - dt/dx·(qE - qW + qN - qS) + dt·src
//   boundaries  critical-flow weir outflow on all four edges, damped 0.5
//   wet/dry     h_face ≤ H_MIN → face dry, q = 0; h clamped ≥ 0
//
// Textures (texelFetch everywhere — no filtering dependence in the sim):
//   zTex  R32F (nx,ny)        bed elevation, static
//   srcTex R32F (nx,ny)       depth-injection rate (m/s), rebuilt on CPU
//   h     R32F (nx,ny)        ping-pong pair
//   q     RG32F (nx+1,ny+1)   ping-pong pair; r = qx at x-face, g = qy at y-face
//   diag  RGBA32F (nx/8,ny/8) r = block max(h), g = block sum(h)
//   ovl   RGBA8 (nx,ny)       colorized depth for the 2D map overlay

import * as THREE from 'three';

const G = 9.81;
const H_MIN = 1e-3;
const CFL_SAFETY = 0.30;   // Taichi used 0.4 with exact hmax; ours is ~250 ms stale
const DIAG_BLOCK = 8;

const PASS_VS = /* glsl */`
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const Q_PASS_FS = /* glsl */`
  precision highp float;
  precision highp sampler2D;
  layout(location = 0) out highp vec4 frag;
  uniform sampler2D hTex;
  uniform sampler2D zTex;
  uniform sampler2D qTex;
  uniform float dt;
  uniform float dx;
  uniform float manning;
  uniform int nx;
  uniform int ny;

  float faceQ(float q0, float zu, float hu, float zd, float hd) {
    float eu = zu + hu;
    float ed = zd + hd;
    float hflow = max(eu, ed) - max(zu, zd);
    if (hflow <= ${H_MIN}) return 0.0;
    float dedx  = (ed - eu) / dx;
    float numer = q0 - ${G} * hflow * dt * dedx;
    float denom = 1.0 + ${G} * dt * manning * manning * abs(q0)
                        / pow(hflow, 7.0 / 3.0);
    return numer / denom;
  }

  // Critical-flow weir outflow at a domain edge, damped (see Taichi solver).
  float weir(float h) { return (h > ${H_MIN}) ? 0.5 * 1.7 * pow(h, 1.5) : 0.0; }

  void main() {
    ivec2 f = ivec2(gl_FragCoord.xy);   // face index: x in [0,nx], y in [0,ny]
    vec2 q0 = texelFetch(qTex, f, 0).rg;
    float qx = 0.0;
    float qy = 0.0;

    // x-face between cells (f.x-1, f.y) and (f.x, f.y); valid rows f.y < ny
    if (f.y < ny) {
      if (f.x == 0) {
        qx = -weir(texelFetch(hTex, ivec2(0, f.y), 0).r);
      } else if (f.x == nx) {
        qx =  weir(texelFetch(hTex, ivec2(nx - 1, f.y), 0).r);
      } else {
        float zu = texelFetch(zTex, ivec2(f.x - 1, f.y), 0).r;
        float hu = texelFetch(hTex, ivec2(f.x - 1, f.y), 0).r;
        float zd = texelFetch(zTex, f, 0).r;
        float hd = texelFetch(hTex, f, 0).r;
        qx = faceQ(q0.r, zu, hu, zd, hd);
      }
    }
    // y-face between cells (f.x, f.y-1) and (f.x, f.y); valid cols f.x < nx
    if (f.x < nx) {
      if (f.y == 0) {
        qy = -weir(texelFetch(hTex, ivec2(f.x, 0), 0).r);
      } else if (f.y == ny) {
        qy =  weir(texelFetch(hTex, ivec2(f.x, ny - 1), 0).r);
      } else {
        float zu = texelFetch(zTex, ivec2(f.x, f.y - 1), 0).r;
        float hu = texelFetch(hTex, ivec2(f.x, f.y - 1), 0).r;
        float zd = texelFetch(zTex, f, 0).r;
        float hd = texelFetch(hTex, f, 0).r;
        qy = faceQ(q0.g, zu, hu, zd, hd);
      }
    }
    frag = vec4(qx, qy, 0.0, 0.0);
  }
`;

const H_PASS_FS = /* glsl */`
  precision highp float;
  precision highp sampler2D;
  layout(location = 0) out highp vec4 frag;
  uniform sampler2D hTex;
  uniform sampler2D qTex;
  uniform sampler2D srcTex;
  uniform float dt;
  uniform float dx;
  uniform vec4 splat;     // (i, j, radius, depth) — depth ≤ 0 disables

  void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    float h  = texelFetch(hTex, c, 0).r;
    float qW = texelFetch(qTex, c, 0).r;
    float qE = texelFetch(qTex, ivec2(c.x + 1, c.y), 0).r;
    float qS = texelFetch(qTex, c, 0).g;
    float qN = texelFetch(qTex, ivec2(c.x, c.y + 1), 0).g;
    float src = texelFetch(srcTex, c, 0).r;
    h += -(dt / dx) * (qE - qW + qN - qS) + dt * src;
    if (splat.w > 0.0) {
      vec2 d = vec2(c) - splat.xy;
      if (dot(d, d) <= splat.z * splat.z) h = max(h, splat.w);
    }
    frag = vec4(max(h, 0.0), 0.0, 0.0, 0.0);
  }
`;

const DIAG_FS = /* glsl */`
  precision highp float;
  precision highp sampler2D;
  layout(location = 0) out highp vec4 frag;
  uniform sampler2D hTex;
  uniform int nx;
  uniform int ny;
  void main() {
    ivec2 b = ivec2(gl_FragCoord.xy) * ${DIAG_BLOCK};
    float m = 0.0;
    float s = 0.0;
    for (int dy = 0; dy < ${DIAG_BLOCK}; dy++) {
      for (int dxx = 0; dxx < ${DIAG_BLOCK}; dxx++) {
        ivec2 c = b + ivec2(dxx, dy);
        if (c.x < nx && c.y < ny) {
          float h = texelFetch(hTex, c, 0).r;
          m = max(m, h);
          s += h;
        }
      }
    }
    frag = vec4(m, s, 0.0, 0.0);
  }
`;

const OVERLAY_FS = /* glsl */`
  precision highp float;
  precision highp sampler2D;
  layout(location = 0) out highp vec4 frag;
  uniform sampler2D hTex;
  uniform int ny;
  void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    // Flip vertically so the readback buffer's row 0 (GL bottom) becomes the
    // image's top row = north (j = ny-1) after putImageData.
    float depth = texelFetch(hTex, ivec2(p.x, ny - 1 - p.y), 0).r;
    if (depth > 0.1) {
      float t = min(depth / 5.0, 1.0);
      frag = vec4(0.18 - 0.10 * t, 0.55 - 0.20 * t, 0.90 - 0.05 * t,
                  min(0.85, 0.30 + depth * 0.10));
    } else {
      frag = vec4(0.0);
    }
  }
`;

const COPY_FS = /* glsl */`
  precision highp float;
  precision highp sampler2D;
  layout(location = 0) out highp vec4 frag;
  uniform sampler2D srcCopyTex;
  void main() {
    frag = vec4(texelFetch(srcCopyTex, ivec2(gl_FragCoord.xy), 0).r, 0.0, 0.0, 0.0);
  }
`;

function floatTarget(w, h, format) {
  return new THREE.WebGLRenderTarget(w, h, {
    format, type: THREE.FloatType,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    depthBuffer: false, stencilBuffer: false, generateMipmaps: false,
  });
}

export class GPUFloodSim {
  constructor(renderer, { nx, ny, dx, z, manning = 0.035 }) {
    this.renderer = renderer;
    this.nx = nx; this.ny = ny; this.dx = dx;
    this.cellArea = dx * dx;
    this.simTime = 0;
    this.hmaxEst = 0.05;     // refreshed from diagnostics; drives the CFL dt
    this._splat = null;
    this.inflows = {};       // river -> {rect:[i0,i1,j0,j1], cms}

    const gl = renderer.getContext();
    if (!renderer.capabilities.isWebGL2) {
      throw new Error('WebGL2 is required');
    }
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('EXT_color_buffer_float is required (float render targets)');
    }
    this.floatLinear = !!gl.getExtension('OES_texture_float_linear');

    // --- static textures -------------------------------------------------
    this.zTex = new THREE.DataTexture(z, nx, ny, THREE.RedFormat, THREE.FloatType);
    const filt = this.floatLinear ? THREE.LinearFilter : THREE.NearestFilter;
    this.zTex.minFilter = filt; this.zTex.magFilter = filt;
    this.zTex.generateMipmaps = false;
    this.zTex.needsUpdate = true;

    this.srcData = new Float32Array(nx * ny);
    this.srcTex = new THREE.DataTexture(this.srcData, nx, ny,
                                        THREE.RedFormat, THREE.FloatType);
    this.srcTex.minFilter = THREE.NearestFilter;
    this.srcTex.magFilter = THREE.NearestFilter;
    this.srcTex.needsUpdate = true;

    // --- ping-pong targets ------------------------------------------------
    this.hRT = [floatTarget(nx, ny, THREE.RedFormat),
                floatTarget(nx, ny, THREE.RedFormat)];
    this.qRT = [floatTarget(nx + 1, ny + 1, THREE.RGFormat),
                floatTarget(nx + 1, ny + 1, THREE.RGFormat)];
    // Rendering samples h and q with uv — give them a friendly filter
    // (the sim itself uses texelFetch, which ignores filtering).
    for (const rt of [...this.hRT, ...this.qRT]) {
      rt.texture.minFilter = filt; rt.texture.magFilter = filt;
    }
    this.hPing = 0; this.qPing = 0;

    this.diagW = Math.ceil(nx / DIAG_BLOCK);
    this.diagH = Math.ceil(ny / DIAG_BLOCK);
    this.diagRT = floatTarget(this.diagW, this.diagH, THREE.RGBAFormat);
    this.diagBuf = new Float32Array(this.diagW * this.diagH * 4);

    this.ovlRT = new THREE.WebGLRenderTarget(nx, ny, {
      format: THREE.RGBAFormat, type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
      depthBuffer: false, stencilBuffer: false,
    });
    this.ovlBuf = new Uint8Array(nx * ny * 4);

    // --- pass plumbing ----------------------------------------------------
    this.scene = new THREE.Scene();
    this.camera = new THREE.Camera();
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;   // VS outputs clip space directly
    this.scene.add(this.quad);

    const mat = (fs, uniforms) => new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: PASS_VS, fragmentShader: fs, uniforms,
      depthTest: false, depthWrite: false,
    });
    this.qMat = mat(Q_PASS_FS, {
      hTex: { value: null }, zTex: { value: this.zTex }, qTex: { value: null },
      dt: { value: 0 }, dx: { value: dx }, manning: { value: manning },
      nx: { value: nx }, ny: { value: ny },
    });
    this.hMat = mat(H_PASS_FS, {
      hTex: { value: null }, qTex: { value: null },
      srcTex: { value: this.srcTex },
      dt: { value: 0 }, dx: { value: dx },
      splat: { value: new THREE.Vector4(0, 0, 0, -1) },
    });
    this.diagMat = mat(DIAG_FS, {
      hTex: { value: null }, nx: { value: nx }, ny: { value: ny },
    });
    this.ovlMat = mat(OVERLAY_FS, {
      hTex: { value: null }, ny: { value: ny },
    });
    this.copyMat = mat(COPY_FS, { srcCopyTex: { value: null } });

    this.initTex = null;     // optional baseline water state (see setInitialWater)
    this.h0max = 0;

    this.reset();
  }

  get hTexture() { return this.hRT[this.hPing].texture; }
  get qTexture() { return this.qRT[this.qPing].texture; }
  get zTexture() { return this.zTex; }

  _runPass(material, target) {
    this.quad.material = material;
    const prev = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prev);
  }

  // ---------------------------------------------------------------- control
  setInflow(river, rect, cms) {
    this.inflows[river] = { rect, cms: Math.max(0, cms) };
    this._rebuildSources();
  }

  totalInflow() {
    return Object.values(this.inflows).reduce((a, f) => a + f.cms, 0);
  }

  _rebuildSources() {
    this.srcData.fill(0);
    this.maxSrcRate = 0;
    for (const { rect, cms } of Object.values(this.inflows)) {
      const [i0, i1, j0, j1] = rect;
      const nCells = Math.max(1, (i1 - i0) * (j1 - j0));
      const rate = cms / (nCells * this.cellArea);   // m/s of depth per cell
      for (let j = j0; j < j1; j++) {
        for (let i = i0; i < i1; i++) {
          const v = this.srcData[j * this.nx + i] += rate;
          if (v > this.maxSrcRate) this.maxSrcRate = v;
        }
      }
    }
    this.srcTex.needsUpdate = true;
  }

  splash(i, j, radius, depth) {
    this._splat = new THREE.Vector4(i, j, radius, depth);
    this.hmaxEst = Math.max(this.hmaxEst, depth);   // keep dt CFL-safe now
  }

  // Baseline water state (e.g. rivers pre-filled to the USGS stage). Becomes
  // what reset() restores. `data` is depth in metres, texture layout, or null
  // to go back to a dry start.
  setInitialWater(data) {
    if (this.initTex) { this.initTex.dispose(); this.initTex = null; }
    this.h0max = 0;
    if (data) {
      for (let k = 0; k < data.length; k++) {
        if (data[k] > this.h0max) this.h0max = data[k];
      }
      this.initTex = new THREE.DataTexture(data, this.nx, this.ny,
                                           THREE.RedFormat, THREE.FloatType);
      this.initTex.minFilter = THREE.NearestFilter;
      this.initTex.magFilter = THREE.NearestFilter;
      this.initTex.needsUpdate = true;
    }
    this.reset();
  }

  reset() {
    const prev = this.renderer.getRenderTarget();
    const prevColor = new THREE.Color();
    this.renderer.getClearColor(prevColor);
    const prevAlpha = this.renderer.getClearAlpha();
    this.renderer.setClearColor(0x000000, 0);
    for (const rt of [...this.hRT, ...this.qRT]) {
      this.renderer.setRenderTarget(rt);
      this.renderer.clear(true, false, false);
    }
    this.renderer.setRenderTarget(prev);
    this.renderer.setClearColor(prevColor, prevAlpha);
    this.hPing = 0; this.qPing = 0;
    if (this.initTex) {
      this.copyMat.uniforms.srcCopyTex.value = this.initTex;
      this._runPass(this.copyMat, this.hRT[0]);
    }
    this.simTime = 0;
    this.hmaxEst = Math.max(0.05, this.h0max);
  }

  // ------------------------------------------------------------------ step
  cflDt() {
    const h = Math.max(this.hmaxEst * 1.3, 0.5);    // margin for stale hmax
    return Math.min(CFL_SAFETY * this.dx / Math.sqrt(G * h), 2.0);
  }

  step(subSteps = 6) {
    for (let s = 0; s < subSteps; s++) {
      const dt = this.cflDt();
      // q pass: read q[ping], h[ping] → write q[1-ping]
      this.qMat.uniforms.qTex.value = this.qRT[this.qPing].texture;
      this.qMat.uniforms.hTex.value = this.hRT[this.hPing].texture;
      this.qMat.uniforms.dt.value = dt;
      this._runPass(this.qMat, this.qRT[1 - this.qPing]);
      this.qPing = 1 - this.qPing;
      // h pass: read h[ping], q[ping] (just written) → write h[1-ping]
      this.hMat.uniforms.hTex.value = this.hRT[this.hPing].texture;
      this.hMat.uniforms.qTex.value = this.qRT[this.qPing].texture;
      this.hMat.uniforms.dt.value = dt;
      if (this._splat) {
        this.hMat.uniforms.splat.value.copy(this._splat);
        this._splat = null;
      } else {
        this.hMat.uniforms.splat.value.set(0, 0, 0, -1);
      }
      this._runPass(this.hMat, this.hRT[1 - this.hPing]);
      this.hPing = 1 - this.hPing;
      this.simTime += dt;
      // Diagnostics only refresh hmaxEst every ~250 ms, but with large
      // inflows the depth can outgrow the stale estimate within that gap
      // and break the CFL bound (NaN → reset loop). Grow the estimate
      // pessimistically by the injection rate; readDiagnostics corrects it.
      this.hmaxEst += dt * (this.maxSrcRate || 0);
    }
  }

  // ------------------------------------------------------------ diagnostics
  readDiagnostics() {
    this.diagMat.uniforms.hTex.value = this.hRT[this.hPing].texture;
    this._runPass(this.diagMat, this.diagRT);
    this.renderer.readRenderTargetPixels(this.diagRT, 0, 0,
                                         this.diagW, this.diagH, this.diagBuf);
    let hmax = 0, sum = 0;
    for (let k = 0; k < this.diagW * this.diagH; k++) {
      const m = this.diagBuf[4 * k];
      if (m > hmax) hmax = m;
      sum += this.diagBuf[4 * k + 1];
    }
    const nan = !Number.isFinite(hmax) || !Number.isFinite(sum);
    if (!nan) this.hmaxEst = hmax;
    return { hmax, volume: sum * this.cellArea, nan };
  }

  // Colorize depth → RGBA8 readback for the Leaflet overlay. Returns the
  // internal Uint8Array (north-up row order, ready for putImageData).
  renderOverlay() {
    this.ovlMat.uniforms.hTex.value = this.hRT[this.hPing].texture;
    this._runPass(this.ovlMat, this.ovlRT);
    this.renderer.readRenderTargetPixels(this.ovlRT, 0, 0,
                                         this.nx, this.ny, this.ovlBuf);
    return this.ovlBuf;
  }
}
