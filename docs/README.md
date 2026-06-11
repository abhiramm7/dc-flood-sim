# DC Flood Sim — shareable static site

Everything in this folder is a **fully static website**: the 2D shallow-water
solver (the same local-inertial scheme as `03_solver_swe.py`) runs in WebGL2
fragment shaders on the *viewer's* GPU, WebFlood-style
(https://aeplay.github.io/WebFlood/). No Python, no server, no install —
anyone with the link gets the live simulation.

The 3D scene is the whole show: satellite imagery draped on the USGS
topobathy DEM at **true vertical scale**, ~236k OSM buildings, and the live
flood surface. Navigation is map-style — drag to pan, right-drag to rotate,
scroll zooms toward the cursor down to street level, arrow keys pan.

On load the site fetches **live USGS NWIS data**: current tidal stage
(gauges 01647600/01651827, NAVD88) sets the initial river level via a
connectivity flood-fill, and the four inflow sliders preset to the rivers'
current discharges (re-polled every 10 min). Offline it falls back to
defaults.

## Files

| File | What |
|---|---|
| `index.html` | UI shell (HUD, sliders, legend) |
| `main.js` | 3D rendering (Three.js), navigation, USGS NWIS polling |
| `sim.js` | The GPU solver: ping-pong float textures, flux/mass/diagnostic passes |
| `data/dem_z.u16.gz` | Baked DEM (1105×854 @ 20 m, uint16-quantized, ~2 mm steps) |
| `data/meta.json` | Grid dims, WGS84 bounds, USGS gauges + inflow cells |
| `data/basemap.jpg` | Esri World Imagery resampled onto the grid (terrain drape) |
| `data/buildings.f32.gz` | 3D city: OSM footprints as oriented boxes (7 × f32 each), one `InstancedMesh` |
| `data/fema_floodplain.geojson.gz` | FEMA NFHL polygons (kept for future overlay use) |

Regenerate the `data/` assets after changing the DEM or gauges:

```bash
source .venv/bin/activate
python export_web_assets.py          # run from the repo root
```

Buildings (Overpass) and the basemap (Esri tiles) are fetched over the
network and cached — delete the file in `data/` to refetch, or pass
`--skip-buildings`. Dense Overpass tiles are subdivided automatically on
timeout.

**Cache busting:** `index.html` loads `main.js?v=N` and `main.js` imports
`sim.js?v=N`. Bump both `N`s whenever you edit the JS, or returning
visitors (and CDNs) may run the old version.

## Preview locally

```bash
python3 -m http.server 8901 --directory docs
# open http://localhost:8901/
```

(A plain `file://` open won't work — ES modules and fetch need HTTP.)

## Host it as a public website

**GitHub Pages (free, simplest):**
1. Push this repo to GitHub.
2. Repo **Settings → Pages → Build and deployment**:
   source *Deploy from a branch*, branch `main`, folder `/docs`.
3. Live in ~1 minute at `https://<your-username>.github.io/<repo-name>/`.

**Alternatives** (all free tiers, all work unchanged):
- **Cloudflare Pages / Netlify / Vercel** — connect the repo, set the
  publish directory to `docs/`. Better CDN + custom-domain ergonomics.
- **Netlify Drop** (drag-and-drop the `docs/` folder, no git needed).
- Any S3/GCS-style static bucket behind a CDN.

The `.gz` assets are decompressed in the browser, so no special server
configuration is needed anywhere. Total payload ≈ 10 MB on first load.

## Browser requirements

WebGL2 with `EXT_color_buffer_float` — all current Chrome, Edge, Firefox,
Safari 16+. A clear error message is shown if missing.

## Caveats

- Research/education grade, **not** a regulatory flood product. The
  validation step (driving a historical USGS event and comparing modeled
  stage to the gauge record) has not been run on this port.
- Buildings are visual only — they don't block flow in the solver (the DEM
  is bare-earth).
- Steady inflows at four gauges; free critical-flow weir outflow at the
  domain edges. Results are only as good as those boundary conditions.
- Single-precision solver (same as the Taichi/Metal original).
