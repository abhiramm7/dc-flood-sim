# CLAUDE.md — Potomac Key Bridge Flood Model

Context for Claude Code working in this repo. Read this first.

## Goal

Fast flood modeling using **GPU-accelerated 2D shallow-water (Saint-Venant)
equations** instead of full 3D CFD. The SWE update is a grid stencil — the same
data-parallel pattern game engines use for fluids — so the heavy loop runs on
the **Mac GPU via Metal**. Genuine hydraulics, seconds-to-minutes runtimes.

First target: a ~3 km × 2 km reach of the **Potomac River at Francis Scott Key
Bridge, Washington DC** (≈38.902 N, −77.069 W).

This is research-/planning-grade, NOT a regulatory FEMA submission.

## Platform

- macOS, Apple Silicon. **GPU = Metal.**
- Compute framework: **Taichi** (`pip install taichi`), initialized with
  `ti.init(arch=ti.metal)`. It auto-detects the Apple GPU.
- **CRITICAL: Metal has no 64-bit float support. Use `ti.f32` everywhere.**
  Keep elevations/depths well-scaled to avoid precision loss.
- Do **NOT** use `jax-metal` — verified unstable (version mismatches, deadlocks
  as of 2026). Taichi-Metal is the dependable path.

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install taichi numpy matplotlib \
            py3dep rioxarray rasterio xarray \
            richdem whitebox            # terrain/hydrology (step 2)
brew install gdal                       # raster CLI utilities
```

## Pipeline & status

Scripts are numbered in run order.

| Step | File | Status | Purpose |
|---|---|---|---|
| 1 | `01_acquire_dem.py` | ✅ DONE | Fetch + clip DEM to Key Bridge box → `keybridge_dem_clipped.tif` |
| 2 | `02_extract_channel.py` | ⬜ TODO | Fill sinks → flow accumulation → channel mask + inflow cells |
| 3 | `03_solver_swe.py` | ⬜ TODO | Taichi-Metal local-inertial SWE solver (test on flat grid first) |
| 4 | `04_run_keybridge.py` | ⬜ TODO | Solver on real DEM + steady inflow → flood-extent map |
| 5 | `05_validate.py` | ⬜ TODO | Drive with USGS discharge event; compare stage to gauge |

`flood_modeling_plan.md` is the full design doc — consult it for rationale.

## Conventions

- All rasters in a **projected CRS in meters**: UTM Zone 18N = **EPSG:32618**.
- Default solver grid resolution: **3 m** (`TARGET_RES`). Finer = slower/more GPU memory.
- Bounding box and Key Bridge coords live at the top of `01_acquire_dem.py`
  (`BBOX`, `KEY_BRIDGE`) — single source of truth; import or copy from there.
- State arrays for the solver: `z` (bed elev), `h` (water depth), `qx`, `qy`
  (momentum/discharge per unit width). Roughness: Manning's `n` map.
- Outputs (GeoTIFF, PNG) write to the repo root. Prefix previews `*_preview.png`.

## Data sources (verified June 2026)

- **DEM (primary):** 2021 USGS Topobathy Lidar: Potomac River — includes the
  submerged riverbed (so no separate bathymetry merge). Download via NOAA
  Digital Coast Data Access Viewer (https://coast.noaa.gov/dataviewer/), then
  `python 01_acquire_dem.py --local-dem <file.tif>`.
- **DEM (fallback, automated):** USGS 3DEP 1m / S1M via `py3dep` — land-only.
  `python 01_acquire_dem.py` (no args). Good enough to build/test the pipeline.
- **Bathymetry (only if using land-only DEM):** USACE eHydro, NOAA NBS.
- **Forcing + validation:** USGS NWIS discharge/stage. Nearby gauge:
  Little Falls **01646500**.

## The solver, concretely (step 3)

Per Taichi kernel, each timestep:
1. Compute fluxes between neighbor cells. **Start with the local-inertial
   (LISFLOOD-FP-style) scheme** — stable, cheap, maps cleanly to a GPU stencil.
   Upgrade to a full shallow-water Riemann solver only later.
2. Update `h` from net flux (mass conservation).
3. Update `qx, qy` from water-surface slope, gravity, Manning friction (momentum).
4. Wetting/drying: dry cells stay dry.
5. CFL-limit `dt` for stability.

Boundary conditions: inflow hydrograph at upstream cells (from USGS discharge),
free outflow downstream. **Results are only as good as these BCs.**

## Validation (don't trust pretty pictures)

Pick a historical Potomac flood with a known peak stage near Key Bridge, drive
the model with that event's USGS discharge, and check modeled water-surface
elevation against the recorded gauge within a reasonable margin. Also verify
mass conservation (in ≈ stored + out).

## Safety / scope notes for the agent

- Network-restricted sandboxes may block PyPI/USGS; these run fine on the user's
  Mac. If a fetch fails in a sandbox, that's the environment, not the code.
- Don't claim physical accuracy without running the validation step.
