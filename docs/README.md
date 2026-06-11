# The shareable site

This folder is a complete static website. The shallow-water solver (the
same local-inertial scheme as `03_solver_swe.py`) runs in WebGL2 fragment
shaders on the visitor's GPU, an approach borrowed from
[WebFlood](https://aeplay.github.io/WebFlood/). There is no backend. Anyone
with the link gets the live simulation.

The 3D scene carries the whole interface: satellite imagery draped over the
USGS DEM at true vertical scale, ~236k OSM buildings, and the simulated
water surface. Navigation works like a map app. Drag to pan, right-drag to
rotate, scroll to zoom toward the cursor (down to street level), arrow keys
to move.

On load the page calls USGS NWIS for current conditions. The tidal stage
(gauges 01647600 and 01651827, NAVD88) sets the initial river level through
a connectivity flood-fill, and the four inflow sliders preset to whatever
the rivers are doing right now. It re-polls every 10 minutes and falls back
to defaults when offline.

## Files

| File | What |
|---|---|
| `index.html` | UI shell: HUD, sliders, legend |
| `main.js` | 3D rendering (three.js), navigation, NWIS polling |
| `sim.js` | the GPU solver: ping-pong float textures with flux, mass, and diagnostic passes |
| `data/dem_z.u16.gz` | baked DEM, 1105×854 at 20 m, quantized to uint16 |
| `data/meta.json` | grid dimensions, WGS84 bounds, gauges, inflow cells |
| `data/basemap.jpg` | Esri World Imagery resampled onto the grid |
| `data/buildings.f32.gz` | OSM footprints as oriented boxes, 7 floats each, drawn as one `InstancedMesh` |
| `data/fema_floodplain.geojson.gz` | FEMA NFHL polygons (kept for a future overlay) |

To regenerate the assets after changing the DEM or gauges, run
`python export_web_assets.py` from the repo root (inside the venv).
Buildings come from Overpass and the basemap from Esri's tile service; both
are cached, so delete the file in `data/` when you want a fresh copy.
Overpass tiles that time out get subdivided and retried automatically.

One habit to keep: `index.html` loads `main.js?v=N`, and `main.js` imports
`sim.js?v=N`. Bump those numbers whenever you edit the JS. Browsers and
CDNs cache ES modules aggressively, and returning visitors will otherwise
run stale code.

## Preview locally

```bash
python3 -m http.server 8901 --directory docs
# open http://localhost:8901/
```

Opening `index.html` directly from the filesystem won't work; ES modules
and `fetch` need an HTTP server.

## Hosting

The site is live at https://abhiramm7.github.io/dc-flood-sim/ via GitHub
Pages (Settings → Pages → deploy from branch, `main`, `/docs`).

It works unchanged on any static host. Cloudflare Pages, Netlify, and
Vercel just need the publish directory set to `docs/`. Netlify Drop takes a
drag-and-dropped folder if you don't want git involved. The `.gz` assets
are decompressed in the browser, so no server configuration is needed
anywhere. First load is about 10 MB.

## Browser requirements

WebGL2 with the `EXT_color_buffer_float` extension, which means any current
Chrome, Edge, Firefox, or Safari 16+. The page shows an error message if
either is missing.

## Caveats

- Research and education grade, not a regulatory product. The validation
  step (a historical event driven through the model and compared to the
  gauge record) hasn't been run on this port.
- Buildings don't block flow; the solver sees bare earth.
- Steady inflows at four gauges, free weir outflow at the edges. The
  results are only as good as those boundary conditions.
- Single precision, same as the Taichi/Metal original.
