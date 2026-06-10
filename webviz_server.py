#!/usr/bin/env python3
"""
Browser-based live viewer for the Key Bridge SWE solver, in the spirit of
WebFlood (aeplay/WebFlood). The physics runs in Python+Taichi on the Mac GPU
(reusing 03_solver_swe.py); a small aiohttp server streams the water-depth
grid over a WebSocket to a Three.js viewer in the browser.

Run:
    source .venv/bin/activate
    python webviz_server.py
then open http://127.0.0.1:8765/

Wire protocol (binary, little-endian):
    server -> client INIT  (sent once on connect)
        u8  tag = 0x01
        i32 nx
        i32 ny
        f32 dx (meters)
        f32 zmin
        f32 zmax
        f32 manning_default
        f32[nx*ny] z (bed elevation, row-major i-major i.e. i*ny + j)

    server -> client FRAME (sent ~30 Hz)
        u8  tag = 0x02
        f32 sim_time (s)
        f32 h_max    (m)
        f32 volume   (m^3)
        f32 inflow_cms
        f32[nx*ny] h (water depth, same layout as z)

    client -> server (JSON text frames)
        {"cmd":"inflow","value":2500}
        {"cmd":"reset"}
        {"cmd":"pause","value":true}
        {"cmd":"rain","i":150,"j":120,"radius":12,"depth":1.5}
"""
import argparse
import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti
from aiohttp import web, WSMsgType

# ---- Solver (from 03) ----------------------------------------------------
import importlib.util
_spec = importlib.util.spec_from_file_location("swe", "03_solver_swe.py")
swe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(swe)
SWESolver = swe.SWESolver


# --- DEM loading + inflow placement (copied from 04 so this script is self-contained) ---
def load_dem(path: str):
    import rioxarray  # noqa
    import xarray as xr
    da = xr.open_dataarray(path, engine="rasterio").squeeze()
    z = da.values.astype(np.float32)
    tr = da.rio.transform()
    dx = float(abs(tr.a))
    nodata = da.rio.nodata
    if nodata is not None:
        mask = (z == nodata) | ~np.isfinite(z)
        if mask.any():
            z[mask] = np.nanmin(z[~mask])
    # raster is (y, x) north-up; we want (nx, ny) with j=0 south
    z = np.flipud(z).T.copy()
    return z, dx


# USGS streamgauges in/around DC. role="inflow" gauges become source cells;
# role="stage" gauges are validation/observation points (plotted as markers,
# no injection). Lat/lon from the USGS NWIS metadata pages (verified 2026).
USGS_GAUGES = [
    # ---- Inflows (water enters the domain at these gauge locations) ----
    {"id": "01646500", "role": "inflow",
     "name": "Potomac @ Little Falls",
     "river": "potomac",
     "lat": 38.9498, "lon": -77.1276,
     "default_cms": 3000.0},
    {"id": "01651003", "role": "inflow",
     "name": "Anacostia NW Branch @ Brentwood",
     "river": "anacostia_nw",
     "lat": 38.9490, "lon": -76.9562,
     "default_cms": 400.0},
    {"id": "01649500", "role": "inflow",
     "name": "Anacostia NE Branch @ Riverdale",
     "river": "anacostia_ne",
     "lat": 38.9603, "lon": -76.9260,
     "default_cms": 400.0},
    {"id": "01648000", "role": "inflow",
     "name": "Rock Creek @ Sherrill Dr",
     "river": "rock_creek",
     "lat": 38.9725, "lon": -77.0400,
     "default_cms": 50.0},
    # ---- Validation / stage observation gauges (no injection) ----
    {"id": "01651760", "role": "stage",
     "name": "Anacostia @ Kenilworth",
     "river": "anacostia",
     "lat": 38.9092, "lon": -76.9553},
    {"id": "01651827", "role": "stage",
     "name": "Anacostia @ Buzzard Point",
     "river": "anacostia",
     "lat": 38.8652, "lon": -77.0103},
    {"id": "01652500", "role": "stage",
     "name": "Fourmile Run @ Alexandria",
     "river": "potomac",
     "lat": 38.8433, "lon": -77.0859},
]


def find_gauge_inflows(z, gauges, dem_path, downsample, window=4):
    """Project each USGS gauge's lon/lat to a cell index in the (possibly
    downsampled) grid and return per-gauge inflow slices. The slice is
    snapped to the lowest-elevation cell in a small window around the
    nominal location (so it sits in the river channel, not on a nearby
    bank if the gauge coords drift slightly from the channel)."""
    import pyproj
    import rioxarray  # noqa
    import xarray as xr

    da = xr.open_dataarray(dem_path, engine="rasterio").squeeze()
    tr = da.rio.transform()
    h_rows, w_cols = da.shape
    utm_xmin = tr.c
    utm_ymax = tr.f
    pixel = abs(tr.a)
    # After downsampling, dx becomes pixel*downsample and the (i,j) -> UTM
    # mapping uses the same origin.
    dx_ds = pixel * downsample
    utm_ymin = utm_ymax - h_rows * pixel
    # The downsampled grid covers [utm_xmin, utm_xmin + nx_ds*dx_ds] in x
    # and [utm_ymin, utm_ymin + ny_ds*dx_ds] in y (the south-to-north flip
    # is already applied in load_dem).
    nx_ds, ny_ds = z.shape

    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                      always_xy=True)
    out = []
    for g in gauges:
        ux, uy = fwd.transform(g["lon"], g["lat"])
        i_c = int(round((ux - utm_xmin) / dx_ds))
        j_c = int(round((uy - utm_ymin) / dx_ds))
        i_c = max(0, min(nx_ds - 1, i_c))
        j_c = max(0, min(ny_ds - 1, j_c))
        # Snap to local channel minimum
        i0 = max(0, i_c - window); i1 = min(nx_ds, i_c + window + 1)
        j0 = max(0, j_c - window); j1 = min(ny_ds, j_c + window + 1)
        sub = z[i0:i1, j0:j1]
        if sub.size:
            di, dj = np.unravel_index(int(np.argmin(sub)), sub.shape)
            i_c = i0 + di; j_c = j0 + dj
        # Source footprint: a tiny rectangle around the snapped cell
        w = 2
        islc = slice(max(0, i_c - w), min(nx_ds, i_c + w + 1))
        jslc = slice(max(0, j_c - w), min(ny_ds, j_c + w + 1))
        out.append({
            **g,                           # carries id/name/role/river/lat/lon/default_cms
            "i": int(i_c), "j": int(j_c),
            "slice": (islc, jslc),
            "elev_m": float(z[i_c, j_c]),
        })
    return out


# --- Sim wrapper ----------------------------------------------------------
class SimState:
    """Multi-inflow SWE state. Each named river gets its own discharge and
    edge slice; we re-apply all of them whenever any one changes."""

    def __init__(self, dem_path, downsample=2, inflows=None, manning=0.035,
                 smooth_sigma=1.2):
        ti.init(arch=ti.metal, default_fp=ti.f32)
        self.dem_path = dem_path
        # Geographic bbox (lon/lat, WGS84) from the DEM file — sent to the
        # browser so the 2D pane can anchor its tile basemap.
        import pyproj, rioxarray  # noqa
        import xarray as xr
        _da = xr.open_dataarray(dem_path, engine="rasterio").squeeze()
        _h_rows, _w_cols = _da.shape
        _tr = _da.rio.transform()
        _xmin = _tr.c; _ymax = _tr.f
        _xmax = _xmin + _w_cols * _tr.a
        _ymin = _ymax + _h_rows * _tr.e
        _back = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326",
                                            always_xy=True)
        self.west,  self.south = _back.transform(_xmin, _ymin)
        self.east,  self.north = _back.transform(_xmax, _ymax)

        z, dx = load_dem(dem_path)
        if downsample > 1:
            # Block-average first to anti-alias the high-frequency DEM noise,
            # then take every Nth sample. Average is computed by reshaping.
            ds = downsample
            ny_full, nx_full = z.shape  # remember: z is (nx, ny) in solver layout
            nx_trim = (z.shape[0] // ds) * ds
            ny_trim = (z.shape[1] // ds) * ds
            z = z[:nx_trim, :ny_trim].reshape(nx_trim // ds, ds,
                                              ny_trim // ds, ds).mean(axis=(1, 3))
            z = z.astype(np.float32).copy()
            dx *= ds
        if smooth_sigma and smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter
                z = gaussian_filter(z, sigma=float(smooth_sigma),
                                    mode="nearest").astype(np.float32)
            except ImportError:
                pass
        self.z = z
        self.dx = float(dx)
        self.nx, self.ny = z.shape
        self.manning = float(manning)

        self.solver = SWESolver(self.nx, self.ny, self.dx)
        self.solver.load_terrain(z, manning=manning)

        # Snap every gauge to its nearest channel cell. Inflows get source
        # rectangles; stage gauges are markers only.
        gauge_info = find_gauge_inflows(z, USGS_GAUGES, dem_path, downsample)
        default_cms = inflows or {}
        self.inflows = {}
        self.gauges = []
        for g in gauge_info:
            self.gauges.append({
                "id": g["id"], "name": g["name"], "role": g["role"],
                "river": g["river"],
                "lat": g["lat"], "lon": g["lon"],
                "i": g["i"], "j": g["j"],
                "elev_m": g["elev_m"],
                "default_cms": g.get("default_cms", 0.0),
            })
            if g["role"] != "inflow":
                continue
            slc = g["slice"]
            n_cells = ((slc[0].stop - slc[0].start) *
                       (slc[1].stop - slc[1].start))
            river = g["river"]
            self.inflows[river] = {
                "slice": slc,
                "n_cells": n_cells,
                "cms": float(default_cms.get(river, g["default_cms"])),
            }

        self.cell_area = self.dx * self.dx
        self._apply_all_inflows()

        self.sim_t = 0.0
        self.paused = False
        self.clients: set = set()

    def _apply_all_inflows(self):
        # Zero src field, then layer each inflow on top.
        self.solver.src.fill(0.0)
        for name, info in self.inflows.items():
            rate = info["cms"] / max(1, info["n_cells"] * self.cell_area)
            slc = info["slice"]
            self.solver.set_source_rect(
                slc[0].start, slc[0].stop,
                slc[1].start, slc[1].stop, rate,
            )

    def set_inflow(self, name, cms):
        if name not in self.inflows:
            return
        self.inflows[name]["cms"] = max(0.0, float(cms))
        self._apply_all_inflows()

    @property
    def total_inflow_cms(self):
        return sum(info["cms"] for info in self.inflows.values())

    def reset(self):
        self.solver._zero_state()
        self._apply_all_inflows()
        self.sim_t = 0.0

    def rain(self, i, j, radius=10, depth=1.0):
        """Add water in a disc centered at (i, j) -- 'rain' / click-to-flood."""
        i = int(i); j = int(j); r = int(radius); d = float(depth)
        h = self.solver.h.to_numpy()
        ii, jj = np.ogrid[:self.nx, :self.ny]
        m = (ii - i) ** 2 + (jj - j) ** 2 <= r * r
        h[m] = np.maximum(h[m], d)
        self.solver.h.from_numpy(h)

    def step(self, sub_steps=4):
        if self.paused:
            return
        for _ in range(sub_steps):
            dt = self.solver.cfl_dt()
            self.solver.step(dt)
            self.sim_t += dt

    # snapshot helpers (CPU copies)
    def snapshot_z(self):
        return self.z.astype(np.float32, copy=False)

    def snapshot_h(self):
        return self.solver.h.to_numpy().astype(np.float32, copy=False)


# --- WebSocket handlers ---------------------------------------------------
INIT_TAG  = 0x01
FRAME_TAG = 0x02


def build_init_packet(sim: SimState) -> bytes:
    # 4-byte aligned header. Layout (all little-endian):
    #   i32  tag
    #   i32  nx
    #   i32  ny
    #   f32  dx
    #   f32  zmin
    #   f32  zmax
    #   f32  manning
    #   f32  west, south, east, north   (lon/lat, WGS84)
    #   u32  zbytes
    # Total = 48 bytes; z grid begins at offset 48 (still 4-aligned).
    z = sim.snapshot_z()
    zmin = float(z.min()); zmax = float(z.max())
    header = struct.pack("<iiiffffffffI",
        INIT_TAG, sim.nx, sim.ny,
        sim.dx, zmin, zmax, sim.manning,
        float(sim.west), float(sim.south),
        float(sim.east), float(sim.north),
        z.size * 4)
    return header + z.tobytes(order="C")


def build_frame_packet(sim: SimState) -> bytes:
    # Header total = 20 bytes (multiple of 4).
    h = sim.snapshot_h()
    hmax = float(h.max())
    vol  = float(h.sum()) * sim.cell_area
    header = struct.pack("<iffff",
        FRAME_TAG, sim.sim_t, hmax, vol, sim.total_inflow_cms)
    return header + h.tobytes(order="C")


async def ws_handler(request: web.Request):
    sim: SimState = request.app["sim"]
    ws = web.WebSocketResponse(max_msg_size=64 * 1024 * 1024, heartbeat=20.0)
    await ws.prepare(request)
    sim.clients.add(ws)
    addr = request.remote
    print(f"[ws] client connected from {addr} ({len(sim.clients)} total)")

    try:
        await ws.send_bytes(build_init_packet(sim))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    cmd = json.loads(msg.data)
                except Exception:
                    continue
                op = cmd.get("cmd")
                if op == "inflow":
                    # Backward-compat: "inflow" with just a value drives the
                    # main (Potomac) channel; pass "river" to target either.
                    river = cmd.get("river", "potomac")
                    sim.set_inflow(river, cmd.get("value", 0.0))
                elif op == "reset":
                    sim.reset()
                elif op == "pause":
                    sim.paused = bool(cmd.get("value", not sim.paused))
                elif op == "rain":
                    sim.rain(cmd.get("i", 0), cmd.get("j", 0),
                             cmd.get("radius", 10), cmd.get("depth", 1.0))
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        sim.clients.discard(ws)
        print(f"[ws] client {addr} disconnected ({len(sim.clients)} left)")
    return ws


# --- broadcast loop ------------------------------------------------------
async def broadcast_loop(app: web.Application):
    sim: SimState = app["sim"]
    target_fps = app["target_fps"]
    sub_steps  = app["sub_steps"]
    period = 1.0 / target_fps
    last = time.time()
    next_log = time.time() + 5.0
    while True:
        t0 = time.time()
        # Advance the simulation
        sim.step(sub_steps=sub_steps)
        # Broadcast frame
        if sim.clients:
            packet = build_frame_packet(sim)
            dead = []
            for ws in list(sim.clients):
                if ws.closed:
                    dead.append(ws); continue
                try:
                    await ws.send_bytes(packet)
                except (ConnectionResetError, RuntimeError):
                    dead.append(ws)
            for d in dead:
                sim.clients.discard(d)
        # Pace
        elapsed = time.time() - t0
        if elapsed < period:
            await asyncio.sleep(period - elapsed)
        else:
            await asyncio.sleep(0)
        if time.time() > next_log:
            next_log += 5.0
            print(f"[sim] t={sim.sim_t:7.1f}s  hmax={float(sim.solver.h_max[None]):.2f}m  "
                  f"clients={len(sim.clients)}  step-rate={1/max(elapsed,1e-3):.1f}/s")


# --- app wiring ----------------------------------------------------------
def write_bridge_json(sim: SimState, dem_path: str, out_path: Path):
    """Compute Francis Scott Key Bridge endpoints in mesh world coords and dump
    to a JSON file the browser can fetch alongside the basemap."""
    import json as _json
    import pyproj
    import rioxarray  # noqa
    import xarray as xr

    # Key Bridge endpoints (Rosslyn ↔ Georgetown), approx from OSM.
    bridge_lonlat = {
        "west": (-77.0731, 38.9013),
        "east": (-77.0676, 38.9036),
    }
    xform = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                        always_xy=True)
    utm_w = xform.transform(*bridge_lonlat["west"])
    utm_e = xform.transform(*bridge_lonlat["east"])

    da = xr.open_dataarray(dem_path, engine="rasterio").squeeze()
    h, w = da.shape
    tr = da.rio.transform()
    utm_xmin = tr.c
    utm_ymax = tr.f
    utm_xmax = utm_xmin + w * tr.a
    utm_ymin = utm_ymax + h * tr.e  # tr.e is negative
    utm_cx = 0.5 * (utm_xmin + utm_xmax)
    utm_cy = 0.5 * (utm_ymin + utm_ymax)

    # World scale identical to the client's: based on the FULL footprint
    # (downsampling doesn't change the extent, only the cell count).
    world_w = sim.nx * sim.dx
    world_h = sim.ny * sim.dx
    s = 2.0 / max(world_w, world_h)

    def utm_to_world(utm_x, utm_y):
        return ((utm_x - utm_cx) * s, (utm_y - utm_cy) * s)

    wx, wz = utm_to_world(*utm_w)
    ex, ez = utm_to_world(*utm_e)
    out = {
        "endpoints": [
            {"x": wx, "z": wz, "label": "Rosslyn"},
            {"x": ex, "z": ez, "label": "Georgetown"},
        ],
        "deck_elev_m": 35.0,   # Key Bridge deck ≈ 35 m above MSL near midspan
        "width_m": 22.0,       # 6 lanes + walkways
        "n_spans": 7,          # five Roman-style concrete arches + approaches
    }
    out_path.write_text(_json.dumps(out, indent=2))
    print(f"[bridge] wrote {out_path}  endpoints: "
          f"({wx:+.3f},{wz:+.3f}) -> ({ex:+.3f},{ez:+.3f})")


def write_roads_json(sim: SimState, dem_path: str, out_path: Path):
    """Fetch OSM road centerlines for the DEM bbox via Overpass API and
    write a JSON file the browser can render as a polyline mesh."""
    import json as _json
    import subprocess
    import pyproj
    import rioxarray  # noqa
    import xarray as xr

    da = xr.open_dataarray(dem_path, engine="rasterio").squeeze()
    h_rows, w_cols = da.shape
    tr = da.rio.transform()
    utm_xmin = tr.c
    utm_ymax = tr.f
    utm_xmax = utm_xmin + w_cols * tr.a
    utm_ymin = utm_ymax + h_rows * tr.e
    utm_cx = 0.5 * (utm_xmin + utm_xmax)
    utm_cy = 0.5 * (utm_ymin + utm_ymax)
    back = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326",
                                       always_xy=True)
    west, south = back.transform(utm_xmin, utm_ymin)
    east, north = back.transform(utm_xmax, utm_ymax)
    bbox_key = [west, south, east, north]

    if out_path.exists():
        try:
            cached = _json.loads(out_path.read_text())
            if cached.get("bbox") == bbox_key:
                print(f"[roads] reusing cache ({cached.get('n', 0)} roads)")
                return
        except Exception:
            pass

    # Only fetch the road types worth drawing — skip footways/cycleways which
    # would add noise.
    types = ("motorway|trunk|primary|secondary|tertiary|residential|"
             "unclassified|service|motorway_link|trunk_link|primary_link|"
             "secondary_link|tertiary_link")
    query = (
        "[out:json][timeout:90];"
        f'(way["highway"~"^({types})$"]'
        f'({south:.5f},{west:.5f},{north:.5f},{east:.5f}););'
        "out geom;"
    )
    print(f"[roads] fetching Overpass for bbox "
          f"({west:.3f},{south:.3f}) → ({east:.3f},{north:.3f})…")
    p = subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-X", "POST",
         "-H", "User-Agent: flood-sim/0.1 (research)",
         "--data-urlencode", f"data={query}",
         "https://overpass-api.de/api/interpreter"],
        capture_output=True, text=True,
    )
    if p.returncode != 0 or not p.stdout:
        print(f"[roads] Overpass call failed (rc={p.returncode}): "
              f"{(p.stderr or '')[:200]}")
        return
    try:
        payload = _json.loads(p.stdout)
    except _json.JSONDecodeError:
        print(f"[roads] non-JSON response: {p.stdout[:200]}")
        return

    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                      always_xy=True)
    s_world = 2.0 / max(sim.nx * sim.dx, sim.ny * sim.dx)

    z_full = da.values.astype(np.float32)
    nd = da.rio.nodata
    if nd is not None:
        z_full = np.where(z_full == nd, 0.0, z_full)
    pixel_w = abs(tr.a); pixel_h = abs(tr.e)

    def elev_at_utm(ux, uy):
        col = int((ux - utm_xmin) / pixel_w)
        row = int((utm_ymax - uy) / pixel_h)
        col = max(0, min(w_cols - 1, col))
        row = max(0, min(h_rows - 1, row))
        v = float(z_full[row, col])
        return v if np.isfinite(v) else 0.0

    # Class -> visual weight (higher = thicker line in the renderer).
    CLASS_RANK = {
        "motorway": 4, "trunk": 4,
        "primary": 3, "secondary": 3,
        "tertiary": 2,
        "residential": 1, "unclassified": 1,
        "service": 0,
    }
    def rank_of(highway):
        if "_link" in highway:
            highway = highway.replace("_link", "")
        return CLASS_RANK.get(highway, 1)

    import math
    STEP = 20.0   # densification step in metres along each road segment

    elements = payload.get("elements", [])
    roads = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords_ll = [(g["lon"], g["lat"]) for g in el["geometry"]]
        if len(coords_ll) < 2:
            continue
        # Project all node lon/lats to UTM first.
        utm_pts = [fwd.transform(lon, lat) for lon, lat in coords_ll]
        # Walk each consecutive UTM pair and densify so we never skip the
        # DEM in between two distant OSM nodes (otherwise roads would cut
        # straight through hills).
        pts = []
        for i in range(len(utm_pts) - 1):
            ux0, uy0 = utm_pts[i]
            ux1, uy1 = utm_pts[i + 1]
            d_m = math.hypot(ux1 - ux0, uy1 - uy0)
            n_sub = max(1, int(math.ceil(d_m / STEP)))
            for k in range(n_sub):
                t = k / n_sub
                ux = ux0 + t * (ux1 - ux0)
                uy = uy0 + t * (uy1 - uy0)
                wx = (ux - utm_cx) * s_world
                wz = (uy - utm_cy) * s_world
                pts.append([wx, wz, float(elev_at_utm(ux, uy))])
        ux_e, uy_e = utm_pts[-1]
        pts.append([(ux_e - utm_cx) * s_world, (uy_e - utm_cy) * s_world,
                    float(elev_at_utm(ux_e, uy_e))])

        tags = el.get("tags", {})
        roads.append({
            "p": pts,                            # [worldX, worldZ, ground_m]
            "r": rank_of(tags.get("highway", "")),
        })

    out_path.write_text(_json.dumps({
        "bbox": bbox_key,
        "n": len(roads),
        "roads": roads,
    }, allow_nan=False))
    print(f"[roads] wrote {len(roads)} road ways → {out_path}")


def write_buildings_json(sim: SimState, dem_path: str, out_path: Path):
    """Pull OSM building footprints for the DEM's bbox via Overpass API, then
    project each polygon to MESH-WORLD coordinates so the browser can build an
    ExtrudeGeometry directly. Heights come from `height` or `building:levels`
    tags; otherwise we apply a sensible default."""
    import json as _json
    import subprocess
    import pyproj
    import rioxarray  # noqa
    import xarray as xr

    # Discover the DEM bbox in UTM and convert to lon/lat for Overpass.
    da = xr.open_dataarray(dem_path, engine="rasterio").squeeze()
    h_rows, w_cols = da.shape
    tr = da.rio.transform()
    utm_xmin = tr.c
    utm_ymax = tr.f
    utm_xmax = utm_xmin + w_cols * tr.a
    utm_ymin = utm_ymax + h_rows * tr.e
    utm_cx = 0.5 * (utm_xmin + utm_xmax)
    utm_cy = 0.5 * (utm_ymin + utm_ymax)
    back = pyproj.Transformer.from_crs("EPSG:32618", "EPSG:4326",
                                       always_xy=True)
    west, south = back.transform(utm_xmin, utm_ymin)
    east, north = back.transform(utm_xmax, utm_ymax)
    bbox_key = [west, south, east, north]

    # Cache: skip re-fetch if file already covers this bbox.
    if out_path.exists():
        try:
            cached = _json.loads(out_path.read_text())
            if cached.get("bbox") == bbox_key:
                print(f"[buildings] reusing cache ({cached.get('n', 0)} buildings)")
                return
        except Exception:
            pass

    query = (
        "[out:json][timeout:90];"
        f'(way["building"]({south:.5f},{west:.5f},{north:.5f},{east:.5f}););'
        "out body; >; out skel qt;"
    )
    print(f"[buildings] fetching Overpass for bbox "
          f"({west:.3f},{south:.3f}) → ({east:.3f},{north:.3f})…")
    p = subprocess.run(
        ["curl", "-sL", "--max-time", "180", "-X", "POST",
         "-H", "User-Agent: flood-sim/0.1 (research)",
         "--data-urlencode", f"data={query}",
         "https://overpass-api.de/api/interpreter"],
        capture_output=True, text=True,
    )
    if p.returncode != 0 or not p.stdout:
        print(f"[buildings] Overpass call failed (rc={p.returncode}): "
              f"{(p.stderr or '')[:300]}")
        return
    try:
        payload = _json.loads(p.stdout)
    except _json.JSONDecodeError:
        print(f"[buildings] non-JSON response: {p.stdout[:200]}")
        return

    elements = payload.get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in elements if el["type"] == "node"}
    ways = [el for el in elements
            if el["type"] == "way" and el.get("tags", {}).get("building")]
    print(f"[buildings] {len(elements)} elements, {len(nodes)} nodes, "
          f"{len(ways)} building ways")

    # Forward transform lon/lat → UTM 18N
    fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32618",
                                      always_xy=True)
    # Mesh-world scale (identical to client's calc)
    s_world = 2.0 / max(sim.nx * sim.dx, sim.ny * sim.dx)

    # Bare-earth elevation lookup at UTM coords (using the FULL-res DEM).
    z_full = da.values.astype(np.float32)
    nd = da.rio.nodata
    if nd is not None:
        z_full = np.where(z_full == nd, 0.0, z_full)
    pixel_w = abs(tr.a); pixel_h = abs(tr.e)

    def elev_at_utm(ux, uy):
        col = int((ux - utm_xmin) / pixel_w)
        row = int((utm_ymax - uy) / pixel_h)
        col = max(0, min(w_cols - 1, col))
        row = max(0, min(h_rows - 1, row))
        return float(z_full[row, col])

    def parse_height(tags):
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
        if btype in ("house", "detached", "residential", "garage", "hut"):
            return 7.0
        if btype in ("apartments", "commercial", "office"):
            return 16.0
        if btype in ("church", "cathedral"):
            return 22.0
        return 9.0   # default for "yes" / unknown

    out = []
    for way in ways:
        node_ids = way.get("nodes", [])
        ll = [nodes[i] for i in node_ids if i in nodes]
        if len(ll) < 3:
            continue
        # Drop the closing duplicate node OSM appends.
        if ll[0] == ll[-1]:
            ll = ll[:-1]
        if len(ll) < 3:
            continue
        utm = [fwd.transform(lon, lat) for lon, lat in ll]
        # Shoelace area (m²)
        n_pts = len(utm)
        area = 0.0
        cx_sum = 0.0; cy_sum = 0.0
        for k in range(n_pts):
            x0, y0 = utm[k]
            x1, y1 = utm[(k + 1) % n_pts]
            area += (x0 * y1 - x1 * y0)
            cx_sum += x0; cy_sum += y0
        area = abs(area) * 0.5
        if area < 25.0:
            continue   # skip tiny garages, sheds, etc.
        footprint = [
            [(ux - utm_cx) * s_world, (uy - utm_cy) * s_world]
            for (ux, uy) in utm
        ]
        ground_m = elev_at_utm(cx_sum / n_pts, cy_sum / n_pts)
        height_m = parse_height(way.get("tags", {}))
        # Sanitize: NaN is illegal JSON and trips browsers' JSON.parse.
        if not np.isfinite(ground_m): ground_m = 0.0
        if not np.isfinite(height_m): height_m = 9.0
        out.append({
            "footprint": footprint,
            "ground_m": float(ground_m),
            "height_m": float(height_m),
        })

    out_path.write_text(_json.dumps({
        "bbox": bbox_key,
        "n": len(out),
        "buildings": out,
    }, allow_nan=False))
    print(f"[buildings] wrote {len(out)} buildings (area ≥ 25 m²) → {out_path}")


def write_fema_floodplain(sim: SimState, out_path: Path):
    """Pull FEMA NFHL flood hazard zones (layer 28) for the sim bbox and
    save as GeoJSON. Client renders these as a translucent overlay on the
    2D map for visual comparison against the live flood extent."""
    import json as _json
    import subprocess
    if out_path.exists() and out_path.stat().st_size > 1024:
        print(f"[fema] reusing cache {out_path.name}")
        return
    bbox = f"{sim.west},{sim.south},{sim.east},{sim.north}"
    print(f"[fema] fetching NFHL flood hazard zones for {bbox}…")
    p = subprocess.run(
        ["curl", "-sL", "--max-time", "120", "-G",
         "--data-urlencode", "where=FLD_ZONE IN ('AE','A','AH','AO','VE','FLOODWAY')",
         "--data-urlencode", f"geometry={bbox}",
         "--data-urlencode", "geometryType=esriGeometryEnvelope",
         "--data-urlencode", "inSR=4326",
         "--data-urlencode", "outFields=FLD_ZONE,STATIC_BFE,ZONE_SUBTY",
         "--data-urlencode", "outSR=4326",
         "--data-urlencode", "f=geojson",
         "--data-urlencode", "resultRecordCount=2000",
         "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"],
        capture_output=True, text=True,
    )
    if p.returncode != 0 or not p.stdout.startswith("{"):
        print(f"[fema] fetch failed (rc={p.returncode})")
        return
    out_path.write_text(p.stdout)
    try:
        n = len(_json.loads(p.stdout).get("features", []))
        print(f"[fema] wrote {n} polygons → {out_path}")
    except Exception:
        pass


async def on_startup(app):
    app["broadcast"] = asyncio.create_task(broadcast_loop(app))


async def on_cleanup(app):
    app["broadcast"].cancel()
    try:
        await app["broadcast"]
    except asyncio.CancelledError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="dc_dem.tif")
    ap.add_argument("--downsample", type=int, default=6,
                    help="block-average + decimate DEM by this factor")
    ap.add_argument("--smooth-sigma", type=float, default=1.2,
                    help="Gaussian smoothing sigma (in cells) applied to z")
    ap.add_argument("--potomac-cms",      type=float, default=3000.0)
    ap.add_argument("--anacostia-nw-cms", type=float, default=400.0)
    ap.add_argument("--anacostia-ne-cms", type=float, default=400.0)
    ap.add_argument("--rock-creek-cms",   type=float, default=50.0)
    ap.add_argument("--manning", type=float, default=0.035)
    ap.add_argument("--sub-steps", type=int, default=4,
                    help="solver sub-steps per broadcast frame")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if not os.path.exists(args.dem):
        print(f"DEM not found: {args.dem}\n  run 01_acquire_dem.py first",
              file=sys.stderr); sys.exit(1)

    sim = SimState(
        args.dem, downsample=args.downsample,
        inflows={
            "potomac":      args.potomac_cms,
            "anacostia_nw": args.anacostia_nw_cms,
            "anacostia_ne": args.anacostia_ne_cms,
            "rock_creek":   args.rock_creek_cms,
        },
        manning=args.manning, smooth_sigma=args.smooth_sigma,
    )
    print(f"[sim] grid {sim.nx}x{sim.ny}  dx={sim.dx:.2f} m  "
          f"elev {sim.z.min():.1f}..{sim.z.max():.1f} m")
    for name, info in sim.inflows.items():
        print(f"[sim] {name:>10s}  Q={info['cms']:>5.0f} m^3/s  "
              f"slice={info['slice']}  ({info['n_cells']} cells)")

    static_root = Path(__file__).parent / "webviz"
    # Wipe any artefacts from previous experiments so the client doesn't
    # accidentally render them.
    for stale in ("bridge.json", "basemap.png",
                  "buildings.json", "roads.json"):
        p = static_root / stale
        if p.exists():
            p.unlink()
    # Publish gauge metadata so the 2D map pane can place markers
    # (orange = inflow, blue = stage/validation).
    import json as _json
    (static_root / "gauges.json").write_text(_json.dumps({
        "gauges": [
            {k: g[k] for k in ("id", "name", "role", "river",
                               "lat", "lon", "i", "j", "elev_m",
                               "default_cms")}
            for g in sim.gauges
        ]
    }, indent=2))
    print(f"[gauges] wrote {len(sim.gauges)} → {static_root / 'gauges.json'}")
    try:
        write_fema_floodplain(sim, static_root / "fema_floodplain.geojson")
    except Exception as e:
        print(f"[fema] skipped: {e!r}")
    for g in sim.gauges:
        role = g["role"]
        info = (f"  Q={sim.inflows[g['river']]['cms']:>5.0f} m^3/s"
                if role == "inflow" and g["river"] in sim.inflows else
                "  (stage only)")
        print(f"  [{role:6s}] {g['id']}  {g['name']:38s} "
              f"cell=({g['i']},{g['j']}) z={g['elev_m']:.1f}m {info}")

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app["sim"] = sim
    app["target_fps"] = args.fps
    app["sub_steps"] = args.sub_steps
    app.router.add_get("/ws", ws_handler)

    async def root(_request):
        return web.FileResponse(static_root / "index.html")
    app.router.add_get("/", root)
    app.router.add_static("/", path=str(static_root), show_index=False)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print(f"\nOpen http://{args.host}:{args.port}/  in your browser.\n")
    web.run_app(app, host=args.host, port=args.port, print=lambda *a, **k: None)


if __name__ == "__main__":
    main()
