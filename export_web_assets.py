#!/usr/bin/env python3
"""
Bake the DC DEM + gauge metadata into static assets for the shareable
in-browser flood sim (docs/). After this runs, docs/ is a fully static
site: the SWE solver itself executes in WebGL2 fragment shaders
(WebFlood-style), so it can be hosted on GitHub Pages.

Outputs (under docs/data/):
    dem_z.u16.gz   — bed elevation, uint16-quantized, gzip, texture layout
                     (row-major rows = j south→north, cols = i west→east)
    meta.json      — grid dims/dx, quantization, WGS84 bounds, gauges,
                     inflow source rectangles
    fema_floodplain.geojson.gz — slimmed FEMA NFHL overlay (optional)

Run:
    source .venv/bin/activate
    python export_web_assets.py            # defaults: dc_dem.tif, downsample 2
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

# Reuse the DEM loader + gauge-snapping logic from the live-server viz so
# the static site and the Taichi server agree on grid layout and gauges.
import importlib.util
_spec = importlib.util.spec_from_file_location("webviz_server", "webviz_server.py")
wvs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wvs)


def smooth(z, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(z, sigma=float(sigma), mode="nearest").astype(np.float32)
    except ImportError:
        # Separable 3-tap binomial blur applied twice ≈ small gaussian.
        for _ in range(2):
            z = 0.25 * (np.roll(z, 1, 0) + np.roll(z, -1, 0)) + 0.5 * z
            z = 0.25 * (np.roll(z, 1, 1) + np.roll(z, -1, 1)) + 0.5 * z
        return z.astype(np.float32)


def parse_height(tags):
    """Building height from OSM tags, with type-based fallbacks (metres)."""
    if (hs := tags.get("height")):
        try:
            return float(hs.split()[0])
        except (ValueError, IndexError):
            pass
    if (ls := tags.get("building:levels")):
        try:
            return max(2.5, float(ls) * 3.2)
        except ValueError:
            pass
    btype = tags.get("building", "yes")
    if btype in ("house", "detached", "residential", "garage", "hut", "shed"):
        return 7.0
    if btype in ("apartments", "commercial", "office"):
        return 16.0
    if btype in ("church", "cathedral"):
        return 22.0
    return 9.0


def oriented_box(utm_pts):
    """Fit an oriented rectangle to a footprint polygon (UTM metres).
    Returns (cx, cy, w, l, angle) with angle from +x(east) toward +y(north),
    or None for degenerate polygons."""
    import math
    pts = np.asarray(utm_pts, dtype=np.float64)
    if len(pts) < 3:
        return None
    # Dominant direction = longest edge.
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    lengths = np.hypot(d[:, 0], d[:, 1])
    e = d[int(np.argmax(lengths))]
    ang = math.atan2(e[1], e[0])
    ca, sa = math.cos(-ang), math.sin(-ang)
    rx = pts[:, 0] * ca - pts[:, 1] * sa
    ry = pts[:, 0] * sa + pts[:, 1] * ca
    w = float(rx.max() - rx.min())
    l = float(ry.max() - ry.min())
    if w < 1.0 or l < 1.0:
        return None
    mx = 0.5 * (rx.max() + rx.min())
    my = 0.5 * (ry.max() + ry.min())
    cb, sb = math.cos(ang), math.sin(ang)
    cx = mx * cb - my * sb
    cy = mx * sb + my * cb
    return cx, cy, w, l, ang


def export_buildings(out_dir, z, dx, utm_xmin, utm_ymin, wgs_bbox,
                     tiles=3, min_area=40.0):
    """Fetch OSM building footprints via Overpass (tiled to keep responses
    manageable), reduce each to an oriented box + ground elev + height, and
    write a flat float32 record stream:
        [cx, cy, w, l, angle, ground_m, height_m] * N
    cx/cy are metres east/north of the domain's SW corner."""
    import json as _json
    import subprocess
    import time

    out_path = out_dir / "buildings.f32.gz"
    if out_path.exists():
        print(f"[buildings] reusing {out_path.name} "
              f"({out_path.stat().st_size/1e6:.1f} MB) — delete it to refetch")
        return out_path.stat().st_size

    west, south, east, north = wgs_bbox
    nx, ny = z.shape
    seen = set()
    recs = []
    n_skip_small = 0

    import pyproj
    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                      always_xy=True)

    def fetch_bbox(s, w_, n_, e_):
        query = ("[out:json][timeout:180];"
                 f'(way["building"]({s:.5f},{w_:.5f},{n_:.5f},{e_:.5f}););'
                 "out geom;")
        for attempt in range(2):
            p = subprocess.run(
                ["curl", "-sL", "--max-time", "300", "-X", "POST",
                 "-H", "User-Agent: flood-sim/0.1 (research)",
                 "--data-urlencode", f"data={query}",
                 "https://overpass-api.de/api/interpreter"],
                capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.startswith("{"):
                return _json.loads(p.stdout)
            print(f"[buildings]   retrying (rc={p.returncode})…")
            time.sleep(10)
        return None

    # Dense tiles time out on the public endpoint — subdivide on failure.
    stack = []
    for ti in range(tiles):
        for tj in range(tiles):
            stack.append((south + (north - south) * tj / tiles,
                          west + (east - west) * ti / tiles,
                          south + (north - south) * (tj + 1) / tiles,
                          west + (east - west) * (ti + 1) / tiles, 0))
    while stack:
        s, w_, n_, e_, depth = stack.pop()
        print(f"[buildings] bbox ({w_:.3f},{s:.3f})→({e_:.3f},{n_:.3f}) "
              f"depth {depth}…", flush=True)
        payload = fetch_bbox(s, w_, n_, e_)
        if payload is None:
            if depth < 2:
                sm = 0.5 * (s + n_); wm = 0.5 * (w_ + e_)
                stack += [(s, w_, sm, wm, depth + 1),
                          (s, wm, sm, e_, depth + 1),
                          (sm, w_, n_, wm, depth + 1),
                          (sm, wm, n_, e_, depth + 1)]
                print("[buildings]   subdividing into 4 quarters")
            else:
                print("[buildings]   bbox FAILED at max depth — skipped")
            continue
        n_tile = 0
        for el in payload.get("elements", []):
            if el.get("type") != "way" or "geometry" not in el:
                continue
            if el["id"] in seen:
                continue
            seen.add(el["id"])
            ll = [(g["lon"], g["lat"]) for g in el["geometry"]]
            if len(ll) >= 2 and ll[0] == ll[-1]:
                ll = ll[:-1]
            if len(ll) < 3:
                continue
            lons = [c[0] for c in ll]
            lats = [c[1] for c in ll]
            ux, uy = fwd.transform(lons, lats)
            utm = list(zip(ux, uy))
            box = oriented_box(utm)
            if box is None:
                continue
            cx, cy, w, l, ang = box
            if w * l < min_area:
                n_skip_small += 1
                continue
            gx = cx - utm_xmin
            gy = cy - utm_ymin
            i = max(0, min(nx - 1, int(gx / dx)))
            j = max(0, min(ny - 1, int(gy / dx)))
            ground = float(z[i, j])
            height = parse_height(el.get("tags", {}))
            recs.append((gx, gy, w, l, ang, ground, height))
            n_tile += 1
        print(f"[buildings]   +{n_tile} (total {len(recs)})")
        time.sleep(2)   # be polite to the public Overpass endpoint

    arr = np.asarray(recs, dtype=np.float32)
    payload = arr.tobytes(order="C")
    out_path.write_bytes(gzip.compress(payload, 9))
    print(f"[buildings] wrote {len(recs)} boxes "
          f"(skipped {n_skip_small} < {min_area} m²) — "
          f"{len(payload)/1e6:.1f} MB raw → "
          f"{out_path.stat().st_size/1e6:.1f} MB gz")
    return len(recs)


def export_basemap(out_dir, utm_xmin, utm_ymin, nx, ny, dx, zoom=14, scale=2):
    """Bake an aerial-imagery texture aligned to the solver grid: fetch Esri
    World Imagery web-mercator tiles, then resample per output pixel into the
    UTM grid. Written north-up (row 0 = north); three.js flipY puts v=0 at
    the south edge, matching the mesh UVs."""
    import io
    import math
    import subprocess
    from PIL import Image
    import pyproj

    out = out_dir / "basemap.jpg"
    if out.exists():
        print(f"[basemap] reusing {out.name} "
              f"({out.stat().st_size/1e6:.1f} MB) — delete to refetch")
        return

    W, H = nx * scale, ny * scale
    xs = utm_xmin + (np.arange(W) + 0.5) * (dx / scale)
    ys = utm_ymin + (np.arange(H) + 0.5) * (dx / scale)
    back = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326",
                                       always_xy=True)
    XX, YY = np.meshgrid(xs, ys)            # (H, W)
    lon, lat = back.transform(XX, YY)

    n = (2 ** zoom) * 256
    px = (lon + 180.0) / 360.0 * n
    lat_r = np.radians(lat)
    py = (1.0 - np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r)) / math.pi) / 2.0 * n
    tx0, tx1 = int(px.min() // 256), int(px.max() // 256)
    ty0, ty1 = int(py.min() // 256), int(py.max() // 256)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    print(f"[basemap] fetching {n_tiles} Esri World Imagery tiles (z{zoom})…",
          flush=True)

    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3),
                      np.uint8)
    done = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            url = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                   f"World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}")
            try:
                # curl (system certs) — python.org urllib lacks SSL CAs here
                p = subprocess.run(
                    ["curl", "-sL", "--max-time", "30",
                     "-H", "User-Agent: flood-sim/0.1 (research)", url],
                    capture_output=True)
                if p.returncode != 0 or not p.stdout:
                    raise RuntimeError(f"curl rc={p.returncode}")
                img = Image.open(io.BytesIO(p.stdout)).convert("RGB")
                mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                       (tx - tx0) * 256:(tx - tx0 + 1) * 256] = np.asarray(img)
            except Exception as e:
                print(f"[basemap]   tile {tx},{ty} failed: {e}")
            done += 1
            if done % 40 == 0:
                print(f"[basemap]   {done}/{n_tiles}", flush=True)

    ix = np.clip((px - tx0 * 256).astype(np.int32), 0, mosaic.shape[1] - 1)
    iy = np.clip((py - ty0 * 256).astype(np.int32), 0, mosaic.shape[0] - 1)
    out_img = mosaic[iy, ix][::-1]          # flip: row 0 = north
    Image.fromarray(out_img).save(out, quality=82)
    print(f"[basemap] wrote {W}x{H} → {out.stat().st_size/1e6:.1f} MB jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", default="dc_dem.tif")
    ap.add_argument("--downsample", type=int, default=2,
                    help="block-average + decimate DEM by this factor")
    ap.add_argument("--smooth-sigma", type=float, default=1.2)
    ap.add_argument("--manning", type=float, default=0.035)
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--skip-buildings", action="store_true",
                    help="don't fetch OSM building footprints")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DEM → solver-layout grid (i=east, j=0 at south) ------------------
    z, dx = wvs.load_dem(args.dem)
    ds = args.downsample
    if ds > 1:
        nx_trim = (z.shape[0] // ds) * ds
        ny_trim = (z.shape[1] // ds) * ds
        z = z[:nx_trim, :ny_trim].reshape(nx_trim // ds, ds,
                                          ny_trim // ds, ds).mean(axis=(1, 3))
        z = z.astype(np.float32).copy()
        dx *= ds
    if args.smooth_sigma > 0:
        z = smooth(z, args.smooth_sigma)
    nx, ny = z.shape
    print(f"[grid] {nx}x{ny}  dx={dx:.1f} m  elev {z.min():.1f}..{z.max():.1f} m")

    # --- WGS84 bounds (for the Leaflet pane) -------------------------------
    import pyproj
    import xarray as xr
    import rioxarray  # noqa: F401
    da = xr.open_dataarray(args.dem, engine="rasterio").squeeze()
    tr = da.rio.transform()
    h_rows, w_cols = da.shape
    xmin = tr.c; ymax = tr.f
    xmax = xmin + w_cols * tr.a
    ymin = ymax + h_rows * tr.e
    back = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    west, south = back.transform(xmin, ymin)
    east, north = back.transform(xmax, ymax)

    # --- Gauges + inflow source rectangles ---------------------------------
    gauge_info = wvs.find_gauge_inflows(z, wvs.USGS_GAUGES, args.dem, ds)
    gauges = []
    for g in gauge_info:
        entry = {
            "id": g["id"], "name": g["name"], "role": g["role"],
            "river": g["river"], "lat": g["lat"], "lon": g["lon"],
            "i": g["i"], "j": g["j"], "elev_m": round(g["elev_m"], 2),
            "default_cms": g.get("default_cms", 0.0),
        }
        if g["role"] == "inflow":
            islc, jslc = g["slice"]
            entry["rect"] = [int(islc.start), int(islc.stop),
                             int(jslc.start), int(jslc.stop)]
        gauges.append(entry)
        print(f"  [{g['role']:6s}] {g['id']}  {g['name']:38s} "
              f"cell=({g['i']},{g['j']}) z={g['elev_m']:.1f}m")

    # --- Quantize + write heightmap ----------------------------------------
    zmin = float(z.min()); zmax = float(z.max())
    zscale = (zmax - zmin) / 65535.0
    q = np.round((z - zmin) / zscale).astype(np.uint16)
    # Texture layout: data[(j*nx + i)] — transpose so rows are j.
    payload = q.T.copy(order="C").tobytes()
    (out_dir / "dem_z.u16.gz").write_bytes(gzip.compress(payload, 9))
    print(f"[dem] wrote dem_z.u16.gz "
          f"({len(payload)/1e6:.1f} MB raw → "
          f"{(out_dir / 'dem_z.u16.gz').stat().st_size/1e6:.1f} MB gz, "
          f"quantization step {zscale*100:.2f} cm)")

    meta = {
        "nx": nx, "ny": ny, "dx": round(float(dx), 3),
        "zmin": round(zmin, 3), "zmax": round(zmax, 3),
        "zscale": zscale,
        "west": west, "south": south, "east": east, "north": north,
        "manning": args.manning,
        "source_dem": args.dem,
        "gauges": gauges,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[meta] wrote meta.json ({len(gauges)} gauges)")

    # --- Satellite basemap (terrain drape) ----------------------------------
    try:
        export_basemap(out_dir, xmin, ymin, nx, ny, dx)
    except Exception as e:
        print(f"[basemap] failed: {e!r} — site falls back to elevation ramp")

    # --- OSM buildings (3D city layer) --------------------------------------
    if not args.skip_buildings:
        try:
            export_buildings(out_dir, z, dx, xmin, ymin,
                             (west, south, east, north))
        except Exception as e:
            print(f"[buildings] failed: {e!r} — site works without them")

    # --- FEMA floodplain (slim + gzip) --------------------------------------
    src = Path("webviz/fema_floodplain.geojson")
    if src.exists():
        geo = json.loads(src.read_text())

        def rnd(coords):
            if isinstance(coords[0], (int, float)):
                return [round(coords[0], 5), round(coords[1], 5)]
            return [rnd(c) for c in coords]

        feats = []
        for f in geo.get("features", []):
            p = f.get("properties", {})
            feats.append({
                "type": "Feature",
                "properties": {"FLD_ZONE": p.get("FLD_ZONE"),
                               "ZONE_SUBTY": p.get("ZONE_SUBTY")},
                "geometry": {"type": f["geometry"]["type"],
                             "coordinates": rnd(f["geometry"]["coordinates"])},
            })
        slim = json.dumps({"type": "FeatureCollection", "features": feats},
                          separators=(",", ":"))
        outp = out_dir / "fema_floodplain.geojson.gz"
        outp.write_bytes(gzip.compress(slim.encode(), 9))
        print(f"[fema] {len(feats)} polygons, "
              f"{len(slim)/1e6:.1f} MB slim → {outp.stat().st_size/1e6:.1f} MB gz")
    else:
        print("[fema] webviz/fema_floodplain.geojson not found — skipping overlay")


if __name__ == "__main__":
    main()
