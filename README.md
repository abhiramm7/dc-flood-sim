# DC Flood Sim — open flood modeling from open public data

**Live demo:** https://abhiramm7.github.io/dc-flood-sim/

A GPU-accelerated 2D flood model of Washington DC (Potomac + Anacostia) that
runs **entirely in your browser** — real shallow-water hydraulics, a 3D city
built from open data, and live USGS river conditions, shareable as a static
web page.

![100-year flood scenario over Washington DC](docs/screenshot.jpg)
*Simulated 100-yr flood (~14,650 m³/s combined inflow). National Airport and
the Mall flood while the city's buildings stand in the water. Everything you
see is computed live on the viewer's GPU.*

## Vision

Flood risk information is usually locked in static FEMA panels or expensive
proprietary models. Every dataset this project uses is **free and public** —
the goal is a template for building interactive, physics-based flood-risk
models for any US city from open data alone:

- **terrain** from USGS lidar,
- **river forcing** from live USGS gauges,
- **the built environment** from OpenStreetMap and municipal open data,
- **reference hazard zones** from FEMA,

…combined with a solver fast enough to explore "what if" scenarios in real
time, in a browser, with no installation.

This is research/education grade, **not** a regulatory product. See
[Limitations](#limitations).

## How it works

Two implementations of the same physics, sharing one codebase philosophy:

1. **Python + Taichi (Metal GPU)** — the numbered pipeline scripts
   (`01_…` → `05_…`) for data prep, experimentation, and validation on a Mac.
2. **WebGL2 fragment shaders** — [`docs/`](docs/) is a fully static site:
   the identical local-inertial scheme ported to ping-pong float textures,
   in the spirit of [WebFlood](https://aeplay.github.io/WebFlood/). Hosted on
   GitHub Pages; the visitor's GPU does the hydraulics.

**The scheme** is the LISFLOOD-FP-style *local-inertial* approximation of the
shallow-water equations (Bates, Horritt & Fewtrell 2010; de Almeida et
al. 2012): explicit face fluxes with semi-implicit Manning friction, mass
conservation per cell, wetting/drying, CFL-limited timestep, critical-flow
weir outflow at the domain edges. ~1 M cells at 20 m resolution run
thousands of times faster than real time on an ordinary laptop GPU.

## Data sources (all open)

| Data | Source | How it's obtained |
|---|---|---|
| Terrain (DEM) | **USGS 3DEP** 1 m / 10 m lidar via [`py3dep`](https://github.com/hyriver/py3dep); optionally **2021 USGS Topobathy Lidar: Potomac River** (includes submerged riverbed) via [NOAA Digital Coast](https://coast.noaa.gov/dataviewer/) | `01_acquire_dem.py` downloads and clips to the model box (EPSG:32618); `export_web_assets.py` downsamples to 20 m, quantizes to uint16, gzips |
| River discharge + tidal stage | **USGS NWIS** instantaneous values ([waterservices.usgs.gov](https://waterservices.usgs.gov)) — gauges 01646500 Little Falls, 01649500/01651003 Anacostia branches, 01648000 Rock Creek; tidal stage 01647600/01651827 | fetched **live by the web page** on load and every 10 min; sets initial water level (connectivity flood-fill to the NAVD88 stage) and the inflow sliders |
| Buildings | **OpenStreetMap** via the [Overpass API](https://overpass-api.de) | `export_web_assets.py` fetches all `building` ways in the bbox (tiled, auto-subdividing on timeout), fits an oriented box + height per footprint, packs ~236k of them into a 5 MB binary rendered as one `InstancedMesh` |
| Flood hazard zones | **FEMA NFHL** (National Flood Hazard Layer) MapServer | `webviz_server.py` queries layer 28 for AE/A/VE/Floodway polygons; slimmed + gzipped for the site |
| Aerial imagery | **Esri World Imagery** tile service (© Esri and contributors; free use with attribution) | `export_web_assets.py` fetches ~130 web-mercator tiles and resamples them onto the UTM model grid as the terrain drape |

## Quick start

Run the shared site locally (no Python needed — it's static):

```bash
python3 -m http.server 8901 --directory docs
# open http://localhost:8901/
```

Rebuild the model/data pipeline (macOS, Apple Silicon):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install taichi numpy matplotlib py3dep rioxarray rasterio xarray scipy pillow

python 01_acquire_dem.py            # fetch + clip the DEM
python export_web_assets.py         # bake docs/data/* (DEM, buildings, imagery, gauges)
python webviz_server.py             # optional: live Taichi/Metal viewer on :8765
```

See [`docs/README.md`](docs/README.md) for the web app internals, hosting
options, and browser requirements, and [`flood_modeling_plan.md`](flood_modeling_plan.md)
for the modeling design doc.

## Repository map

| Path | What |
|---|---|
| `01_acquire_dem.py` … `05_validate.py` | numbered modeling pipeline (acquire → channels → solver → run → validate) |
| `03_solver_swe.py` | the reference Taichi/Metal local-inertial SWE solver |
| `webviz_server.py` + `webviz/` | live local viewer (Taichi solver → WebSocket → browser) |
| `export_web_assets.py` | bakes all static assets for the shareable site |
| `docs/` | **the shareable site** (GitHub Pages root): `sim.js` GPU solver, `main.js` viewer, `data/` baked assets |

## Limitations

- **Not validated yet**: the planned check — driving the model with a
  historical flood's USGS discharge record and comparing modeled stage
  against the gauge — hasn't been run. Treat outputs as illustrative.
- Buildings are visual context only; they don't yet block flow (bare-earth
  DEM, uniform Manning roughness).
- Steady inflows at four gauges; free weir outflow at domain edges. Results
  are only as good as these boundary conditions.
- 20 m grid, single precision, no storm-drain network, no tides/surge
  forcing.

## Roadmap

- Validation against historical events (Isabel 2003, the 1936 flood)
- Buildings as flow obstructions (footprints stamped into bed elevation /
  roughness)
- Hydrograph playback of real events from NWIS daily values
- Click-to-probe depth, depth-colored water, FEMA comparison draped in 3D
- Generalize the bake pipeline to any US bbox

## License

Code is licensed under the **GNU General Public License v3.0** — see
[LICENSE](LICENSE).

Data remains under its providers' terms: USGS and FEMA data are US public
domain; OpenStreetMap data is © OpenStreetMap contributors under the
[ODbL](https://www.openstreetmap.org/copyright); aerial imagery in
`docs/data/basemap.jpg` is © Esri and its imagery partners, used here for
visualization with attribution.

## Acknowledgements

- [WebFlood](https://aeplay.github.io/WebFlood/) for demonstrating
  browser-GPU flood simulation
- Bates, Horritt & Fewtrell (2010), *J. Hydrology* and de Almeida et
  al. (2012), *Water Resources Research* for the numerical scheme
- [Taichi](https://www.taichi-lang.org/), [three.js](https://threejs.org/),
  and the [HyRiver](https://docs.hyriver.io/) stack
