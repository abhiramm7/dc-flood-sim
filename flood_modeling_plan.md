# GPU-Accelerated 2D Flood Modeling on Mac — Project Plan
### Potomac River at Key Bridge, Washington DC

## 1. The core idea (and why it works)

Full 3D CFD (Navier–Stokes) is overkill for river/floodplain inundation. Water spreading over a landscape is shallow relative to its horizontal extent, so the standard, physically sound simplification is the **2D Shallow-Water Equations (SWE)**, also called the Saint-Venant equations. These solve depth-averaged conservation of mass and momentum on a 2D grid — exactly the regime real flood models (HEC-RAS 2D, LISFLOOD-FP, TUFLOW) operate in.

The reason this connects to game engines: SWE is a *stencil computation on a grid* — each cell updates from its neighbors every timestep. That's the same data-parallel pattern GPUs were built for (it's how fluid effects in games work). So you get genuine hydraulics, but the heavy loop runs on the GPU. On a Mac that means **Metal**, reached through one of the frameworks below — no Navier–Stokes, no mesh generation, no days-long runs.

This is a legitimate middle ground, not a toy: research-grade GPU SWE solvers (e.g. the LISFLOOD-FP CUDA version, TRITON, Floodwater) run city-scale floods in seconds to minutes.

## 2. What "fast" buys us vs. traditional CFD

| Approach | Physics | Typical runtime (small domain) | Hardware |
|---|---|---|---|
| 3D CFD (OpenFOAM) | Full Navier–Stokes | hours–days | CPU cluster |
| 2D SWE on CPU (HEC-RAS) | Depth-averaged | minutes–hours | single CPU |
| **2D SWE on GPU (this plan)** | Depth-averaged | **seconds–minutes** | Mac GPU (Metal) |

## 3. Study area

Potomac at **Key Bridge** (Francis Scott Key Bridge), ~38.902 N, −77.069 W. Good first target: the river is well-gauged, the terrain is steep-banked near Georgetown, and high-quality DEM + bathymetry are freely available. A ~3 km × 2 km box around the bridge is small enough to iterate quickly.

## 4. Open-source toolchain (all run on Mac, Apple Silicon)

**Terrain / GIS (data prep)**
- **GDAL** — read/clip/reproject DEM rasters (`brew install gdal`).
- **rasterio / numpy / xarray** — load the DEM into Python arrays.
- **WhiteboxTools** or **richdem** — hydrological terrain processing (fill sinks, flow accumulation, stream extraction).
- **QGIS** (optional GUI) — sanity-check rasters and channels visually.

**GPU compute on Mac — three viable paths (pick based on comfort):**

1. **Taichi (recommended — VERIFIED).** Python-native, JIT-compiles kernels to the **Metal** backend on Mac automatically (`ti.init(arch=ti.metal)`); it auto-probes available GPU backends at runtime. You write the SWE stencil in Python-like syntax; it runs on the Apple GPU. Best effort-to-payoff ratio. **Important caveat confirmed in Taichi docs: the Metal backend does NOT support 64-bit data types — use `ti.f32` (single precision) throughout.** For SWE this is fine if you keep the numerics well-scaled.
2. **JAX with Metal backend (NOT recommended as primary — VERIFIED unstable).** Apple's `jax-metal` is officially "experimental, not all functionality supported," and current bug reports confirm real pain: version mismatches between `jax` and `jax-metal` causing run failures, and deadlocks reported as recently as 2026. Treat as a research curiosity, not a dependable path.
3. **Raw Metal / WebGPU shader.** Maximum control and speed, most work. A `.metal` compute shader (or WebGPU/WGSL for a browser demo) implementing the SWE update. Reserve for later optimization.

Recommendation: **prototype in Taichi-Metal**, because it gets a real GPU solver running fastest while staying in Python for the GIS glue.

## 5. Data sources (all free, US federal) — VERIFIED June 2026

- **⭐ 2021 USGS Topobathy Lidar: Potomac River** — *the key find.* Topobathymetric lidar captures **both land elevation AND submerged riverbed in one dataset**, covering 120+ river miles of the Potomac including DC (USGS technical announcement, Aug 2023). This largely **eliminates the separate bathymetry-merge step** (old step [2]) — the channel bed is already in the data. Access via NOAA Digital Coast **Data Access Viewer (DAV)**. *This is now the primary DEM source for this project.*
- **USGS 3DEP 1m / S1M** — fallback land DEM. USGS now publishes **S1M (Seamless 1-Meter DEM)**, explicitly built "to support surface water modeling." Programmatic access via `py3dep` (PyPI) or OpenTopography; browse via The National Map 3DEP Viewer. Note: 3DEP/S1M is land-only, so if used you'd still need bathymetry.
- **Bathymetry fallbacks** (if not using topobathy lidar) — USACE **eHydro** hydrographic surveys (ArcGIS dashboard, downloadable) and NOAA's **National Bathymetric Source (NBS)**, also mirrored on the AWS Registry of Open Data.
- **USGS NWIS gauge data** — discharge (cfs) and stage at nearby gauges (e.g. Little Falls, 01646500) to drive the inflow boundary and to *validate* results.
- **NOAA Atlas 14** — design rainfall depths if you later want a rainfall-driven scenario.

## 6. Pipeline (step by step)

```
[1] Acquire DEM        → 2021 USGS Topobathy Lidar (Potomac) via NOAA DAV, clip to
                          Key Bridge bounding box (GDAL). Already includes riverbed.
[2] (Bathymetry)       → MOSTLY DONE by topobathy lidar. Only needed if you fall back
                          to land-only 3DEP/S1M, in which case merge eHydro/NBS bed data.
[3] Extract river      → fill sinks → flow accumulation → threshold = stream network
                          (WhiteboxTools/richdem); define channel mask + inflow cells
[4] Build grid         → resample to a uniform DEM grid (e.g. 2–5 m); arrays: z(bed),
                          h(depth), qx, qy (momentum); roughness map (Manning's n)
[5] Boundary/forcing   → inflow hydrograph at upstream cells from USGS discharge;
                          free outflow downstream
[6] Solve on GPU       → 2D SWE finite-volume update each timestep (Taichi/Metal):
                          mass + momentum, with wetting/drying and CFL-limited dt
[7] Output             → water depth & velocity rasters per timestep; max-depth map
[8] Visualize          → flood extent overlay (matplotlib/QGIS), or animation;
                          optional WebGPU viewer
[9] Validate           → compare modeled stage vs. USGS gauge for a known flood event
```

## 7. The solver, concretely

State per grid cell: bed elevation `z`, water depth `h`, momentum/discharge `qx, qy`.
Each GPU timestep:
1. Compute fluxes between neighboring cells (a finite-volume scheme — start simple, e.g. a local-inertial/"diffusive-wave" formulation as in LISFLOOD-FP, which is stable and cheap; upgrade to a full shallow-water Riemann solver later).
2. Update `h` from net flux (mass conservation).
3. Update `qx, qy` from water-surface slope, gravity, and Manning friction (momentum).
4. Apply wetting/drying so dry cells stay dry.
5. Limit `dt` by the CFL condition for stability.

Starting with the **local-inertial (LISFLOOD-FP-style)** scheme is the pragmatic choice: it's the same math used in operational large-scale flood models, maps cleanly to a GPU stencil, and avoids the complexity of shock-capturing Riemann solvers on day one.

## 8. Validation strategy

Don't trust pretty pictures. Pick a **historical Potomac flood** with known peak stage at the Key Bridge / Georgetown area, drive the model with that event's USGS discharge, and check that modeled water-surface elevation matches the recorded gauge stage within a reasonable margin. Also confirm mass conservation (water in ≈ water stored + water out) as a numerical sanity check.

## 9. Suggested milestones

1. **Data**: clip DEM + bathymetry for the Key Bridge box, view it.
2. **Channel**: extract the river / inflow cells from the DEM.
3. **Solver v0**: Taichi-Metal local-inertial SWE on a flat test grid (verify it runs on the Mac GPU and is stable).
4. **Coupled run**: solver on the real DEM with a steady inflow → first flood-extent map.
5. **Event run + validation**: drive with a real discharge hydrograph, compare to gauge.
6. **Polish**: animation / WebGPU viewer; optionally upgrade the numerical scheme.

## 10. Honest caveats

- **Bathymetry** is normally the gotcha — but using the 2021 Potomac topobathy lidar sidesteps it, since bed elevation is already in the dataset. (If you fall back to land-only 3DEP, the gotcha returns.)
- **Use single precision (`ti.f32`).** Taichi's Metal backend has no 64-bit support. Keep elevations/depths well-scaled to avoid precision loss.
- **Resolution vs. speed.** Finer grids are more accurate but cost GPU memory and time; 2–5 m is a sensible start for this domain.
- **Boundary conditions drive everything.** A flood result is only as good as the inflow hydrograph and downstream condition you impose.
- **Don't rely on `jax-metal`.** Verified unstable (version mismatches, deadlocks as of 2026). Taichi-Metal is the dependable Mac GPU path.
- This produces *research-/planning-grade* results, not a regulatory FEMA submission (those require HEC-RAS and formal QA).
```

