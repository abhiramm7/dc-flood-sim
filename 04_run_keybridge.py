#!/usr/bin/env python3
"""
Milestone 4 — Game-like live 3D visualization of the SWE solver running on the
real Key Bridge DEM.

Runs the local-inertial solver from 03_solver_swe.py on `keybridge_dem_clipped.tif`,
imposes a steady inflow at the upstream edge of the river channel, and renders
the terrain + water surface in a real-time orbit-camera 3D scene using
`ti.ui.Scene`.

Controls:
    Mouse drag       orbit camera
    W/A/S/D          pan camera
    E / Q            zoom in / out
    Space            pause / resume simulation
    R                reset water (clear depth + momentum)
    F                +25% inflow,  G  -25% inflow
    Esc / window X   quit

Usage:
    python 04_run_keybridge.py                 # uses keybridge_dem_clipped.tif
    python 04_run_keybridge.py --dem other.tif
    python 04_run_keybridge.py --synthetic     # procedural valley (no DEM needed)
    python 04_run_keybridge.py --downsample 2  # 2x downsample for speed
"""
import argparse
import sys
import time
import numpy as np
import taichi as ti

# Local-inertial SWE solver written in 03_solver_swe.py
import importlib.util
spec = importlib.util.spec_from_file_location("swe", "03_solver_swe.py")
swe = importlib.util.module_from_spec(spec); spec.loader.exec_module(swe)
SWESolver = swe.SWESolver


# -----------------------------------------------------------------------------
# Terrain loading
# -----------------------------------------------------------------------------
def load_dem(path: str):
    """Returns z (nx,ny) float32 in meters, and dx (meters/pixel)."""
    import rioxarray  # noqa
    import xarray as xr
    da = xr.open_dataarray(path, engine="rasterio").squeeze()
    z = da.values.astype(np.float32)
    # The raster is in EPSG:32618 with TARGET_RES=3 m; pull resolution from transform
    tr = da.rio.transform()
    dx = float(abs(tr.a))
    # Replace nodata with the local minimum (so dry land is sensible)
    nodata = da.rio.nodata
    if nodata is not None:
        mask = (z == nodata) | ~np.isfinite(z)
        if mask.any():
            z[mask] = np.nanmin(z[~mask])
    # The DEM is stored as (y, x) row-major (north up). Solver wants (nx, ny)
    # with x = first index, y = second. Transpose and flip y so j=0 is south.
    z = np.flipud(z).T.copy()
    return z, dx


def synthetic_valley(nx=512, ny=384, dx=3.0):
    """Procedural V-shaped river valley running west-east."""
    xs = np.arange(nx, dtype=np.float32) * dx
    ys = np.arange(ny, dtype=np.float32) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    # Channel center y as a function of x (gentle sinusoid)
    yc = (ny * dx) * 0.5 + 60.0 * np.sin(2 * np.pi * X / (nx * dx) * 1.5)
    # Cross-section: parabolic banks
    cross = ((Y - yc) / 90.0) ** 2
    # Down-valley slope
    slope = -0.0015 * X
    z = 30.0 + 25.0 * cross + slope
    # Add a little ridged roughness on the banks
    z += 0.4 * np.sin(X / 40.0) * np.cos(Y / 35.0)
    return z.astype(np.float32), float(dx)


# -----------------------------------------------------------------------------
# Inflow placement: pick a row of cells along the upstream edge that sit in the
# lowest part of the valley (the channel).
# -----------------------------------------------------------------------------
def find_inflow_cells(z, edge="west", width=12):
    """Return (i_slice, j_slice) of inflow cells along the chosen edge.

    Looks at the elevation profile along the edge, finds the lowest segment
    (the channel), and returns a `width`-cell-wide rectangle straddling it.
    """
    nx, ny = z.shape
    if edge == "west":
        col = z[:5, :].min(axis=0)  # min over first 5 columns
        jc = int(np.argmin(col))
        j0 = max(0, jc - width // 2); j1 = min(ny, j0 + width)
        return (slice(0, 3), slice(j0, j1))
    if edge == "east":
        col = z[-5:, :].min(axis=0)
        jc = int(np.argmin(col))
        j0 = max(0, jc - width // 2); j1 = min(ny, j0 + width)
        return (slice(nx - 3, nx), slice(j0, j1))
    raise ValueError(edge)


# -----------------------------------------------------------------------------
# Renderer: build vertex arrays for terrain + water, color terrain by elevation,
# render water as a translucent surface that only shows where h > h_show.
# -----------------------------------------------------------------------------
@ti.data_oriented
class Renderer:
    def __init__(self, solver, z_np, world_scale=1.0, z_exag=2.0):
        """world_scale stretches/shrinks the XY footprint for the camera; z_exag
        vertically exaggerates relief for visual punch."""
        self.solver = solver
        self.nx, self.ny = solver.nx, solver.ny
        self.dx = solver.dx
        self.z_exag = float(z_exag)

        # Normalize world coords roughly to [-1,1] x [-1,1] for an easy camera
        world_w = self.nx * self.dx
        world_h = self.ny * self.dx
        self.world_w = world_w; self.world_h = world_h
        s = world_scale / max(world_w, world_h)
        self.s = s

        # Elevation normalization for color + vertical placement
        zmin = float(np.nanmin(z_np)); zmax = float(np.nanmax(z_np))
        self.zmin = zmin; self.zmax = zmax
        self.z_range = max(zmax - zmin, 1.0)

        # Vertex / index buffers for terrain mesh
        nv = self.nx * self.ny
        nt = 2 * (self.nx - 1) * (self.ny - 1)
        self.terrain_verts  = ti.Vector.field(3, ti.f32, shape=nv)
        self.terrain_colors = ti.Vector.field(3, ti.f32, shape=nv)
        self.terrain_idx    = ti.field(ti.i32, shape=nt * 3)

        self.water_verts  = ti.Vector.field(3, ti.f32, shape=nv)
        self.water_colors = ti.Vector.field(4, ti.f32, shape=nv)  # rgba
        # Reuse the same index buffer for water (same topology)

        self._build_indices()
        self._build_terrain()

    @ti.kernel
    def _build_indices(self):
        for i, j in ti.ndrange(self.nx - 1, self.ny - 1):
            tri = (i * (self.ny - 1) + j) * 6
            v00 = i * self.ny + j
            v10 = (i + 1) * self.ny + j
            v01 = i * self.ny + (j + 1)
            v11 = (i + 1) * self.ny + (j + 1)
            # tri 1
            self.terrain_idx[tri + 0] = v00
            self.terrain_idx[tri + 1] = v10
            self.terrain_idx[tri + 2] = v11
            # tri 2
            self.terrain_idx[tri + 3] = v00
            self.terrain_idx[tri + 4] = v11
            self.terrain_idx[tri + 5] = v01

    @ti.kernel
    def _build_terrain(self):
        # XY centered at origin, Z is elevation (exaggerated, normalized).
        cx = 0.5 * self.nx * self.dx * self.s
        cy = 0.5 * self.ny * self.dx * self.s
        for i, j in ti.ndrange(self.nx, self.ny):
            idx = i * self.ny + j
            x = i * self.dx * self.s - cx
            y = j * self.dx * self.s - cy
            z = self.solver.z[i, j]
            zn = (z - self.zmin) / self.z_range          # 0..1
            zw = zn * self.z_exag * 0.2                  # vertical world units
            self.terrain_verts[idx] = ti.Vector([x, zw, y])  # y-up
            # Color ramp: deep blue (low) -> green -> tan -> white (high)
            r = 0.0; g = 0.0; b = 0.0
            if zn < 0.25:
                t = zn / 0.25
                r = 0.05 + t * 0.10
                g = 0.20 + t * 0.40
                b = 0.30 + t * 0.20
            elif zn < 0.65:
                t = (zn - 0.25) / 0.40
                r = 0.15 + t * 0.45
                g = 0.60 - t * 0.10
                b = 0.50 - t * 0.30
            else:
                t = (zn - 0.65) / 0.35
                r = 0.60 + t * 0.35
                g = 0.50 + t * 0.45
                b = 0.20 + t * 0.70
            self.terrain_colors[idx] = ti.Vector([r, g, b])

    @ti.kernel
    def _build_water(self, h_show: ti.f32):
        cx = 0.5 * self.nx * self.dx * self.s
        cy = 0.5 * self.ny * self.dx * self.s
        for i, j in ti.ndrange(self.nx, self.ny):
            idx = i * self.ny + j
            x = i * self.dx * self.s - cx
            y = j * self.dx * self.s - cy
            z = self.solver.z[i, j]
            h = self.solver.h[i, j]
            zn = (z - self.zmin) / self.z_range
            terrain_w = zn * self.z_exag * 0.2
            # Lift water by depth in the SAME vertical units
            hw = (h / self.z_range) * self.z_exag * 0.2
            yw = terrain_w + hw
            self.water_verts[idx] = ti.Vector([x, yw, y])
            # Alpha 0 when nearly dry, ramp up with depth; deeper = darker blue
            a = 0.0
            if h > h_show:
                a = ti.min(0.85, 0.35 + h * 0.4)
            # Color: shallow turquoise -> deep navy
            d = ti.min(h * 0.4, 1.0)
            r = 0.20 - 0.18 * d
            g = 0.55 - 0.30 * d
            b = 0.85 - 0.10 * d
            self.water_colors[idx] = ti.Vector([r, g, b, a])


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", default="keybridge_dem_clipped.tif")
    ap.add_argument("--synthetic", action="store_true",
                    help="use a procedural V-shaped valley instead of the DEM")
    ap.add_argument("--downsample", type=int, default=1,
                    help="downsample DEM by an integer factor (speeds up sim)")
    ap.add_argument("--inflow-cms", type=float, default=600.0,
                    help="upstream discharge in m^3/s (Potomac mean ~300, flood ~3000+)")
    ap.add_argument("--sub-steps", type=int, default=8,
                    help="solver sub-steps per rendered frame")
    ap.add_argument("--z-exag", type=float, default=4.0,
                    help="vertical exaggeration for the visuals")
    ap.add_argument("--no-vsync", action="store_true")
    args = ap.parse_args()

    ti.init(arch=ti.metal, default_fp=ti.f32)

    # --- Terrain
    if args.synthetic:
        z, dx = synthetic_valley()
        print(f"Synthetic valley: {z.shape}, dx={dx} m")
    else:
        try:
            z, dx = load_dem(args.dem)
        except FileNotFoundError:
            print(f"DEM not found: {args.dem}\n"
                  f"Run `python 01_acquire_dem.py` first, or pass --synthetic.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"DEM loaded: {z.shape}, dx={dx:.2f} m, "
              f"elev range {z.min():.1f}..{z.max():.1f} m")

    if args.downsample > 1:
        s = args.downsample
        z = z[::s, ::s].copy()
        dx *= s
        print(f"Downsampled to: {z.shape}, dx={dx:.2f} m")

    nx, ny = z.shape

    # --- Solver
    solver = SWESolver(nx, ny, dx)
    solver.load_terrain(z, manning=0.035)

    # --- Inflow: discharge -> per-cell depth-injection rate
    inflow_slc = find_inflow_cells(z, edge="west", width=max(8, ny // 30))
    n_inflow_cells = (inflow_slc[0].stop - inflow_slc[0].start) * \
                     (inflow_slc[1].stop - inflow_slc[1].start)
    cell_area = dx * dx
    inflow_rate = args.inflow_cms / (n_inflow_cells * cell_area)  # m/s
    solver.set_source_rect(inflow_slc[0].start, inflow_slc[0].stop,
                           inflow_slc[1].start, inflow_slc[1].stop,
                           inflow_rate)
    print(f"Inflow: {args.inflow_cms} m^3/s spread over {n_inflow_cells} cells "
          f"= {inflow_rate*1000:.2f} mm/s depth injection")

    # --- Renderer
    renderer = Renderer(solver, z, world_scale=2.0, z_exag=args.z_exag)

    # --- ti.ui scene
    window = ti.ui.Window("Potomac at Key Bridge — live flood sim",
                          (1280, 800), vsync=not args.no_vsync)
    canvas = window.get_canvas()
    scene  = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 1.2, 1.8)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(45)

    paused = False
    sim_time = 0.0
    frame = 0
    t_wall = time.time()
    current_inflow = args.inflow_cms

    print("\nControls: drag=orbit  WASD=pan  E/Q=zoom  Space=pause  R=reset  "
          "F/G=inflow +/-  Esc=quit", flush=True)

    while window.running:
        if frame % 60 == 0:
            print(f"  frame {frame}  sim_t={sim_time:.2f}s  running={window.running}",
                  flush=True)
        # Handle keys
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            if window.event.key == ti.ui.SPACE:
                paused = not paused
                print(f"  [{'paused' if paused else 'running'}]")
            if window.event.key == "r":
                solver._zero_state()
                solver.set_source_rect(inflow_slc[0].start, inflow_slc[0].stop,
                                       inflow_slc[1].start, inflow_slc[1].stop,
                                       inflow_rate)
                sim_time = 0.0
                print("  reset")
            if window.event.key == "f":
                current_inflow *= 1.25
                rate = current_inflow / (n_inflow_cells * cell_area)
                solver.set_source_rect(inflow_slc[0].start, inflow_slc[0].stop,
                                       inflow_slc[1].start, inflow_slc[1].stop,
                                       rate)
                print(f"  inflow = {current_inflow:.0f} m^3/s")
            if window.event.key == "g":
                current_inflow /= 1.25
                rate = current_inflow / (n_inflow_cells * cell_area)
                solver.set_source_rect(inflow_slc[0].start, inflow_slc[0].stop,
                                       inflow_slc[1].start, inflow_slc[1].stop,
                                       rate)
                print(f"  inflow = {current_inflow:.0f} m^3/s")

        # Simulation sub-steps
        if not paused:
            for _ in range(args.sub_steps):
                dt = solver.cfl_dt()
                solver.step(dt)
                sim_time += dt

        # Camera control
        camera.track_user_inputs(window, movement_speed=0.03,
                                 hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.55))
        scene.point_light(pos=(1.5, 2.5, 1.5), color=(1.0, 0.97, 0.92))
        scene.point_light(pos=(-1.5, 1.5, -1.0), color=(0.30, 0.35, 0.45))

        # Build water vertices on GPU (terrain is static)
        renderer._build_water(h_show=swe.H_MIN * 5.0)

        scene.mesh(renderer.terrain_verts,
                   indices=renderer.terrain_idx,
                   per_vertex_color=renderer.terrain_colors,
                   two_sided=True)
        scene.mesh(renderer.water_verts,
                   indices=renderer.terrain_idx,
                   per_vertex_color=renderer.water_colors,
                   two_sided=True)

        canvas.scene(scene)

        # Lightweight HUD via window title
        if frame % 30 == 0:
            hmax = float(solver.h_max[None])
            vol  = float(solver.vol[None])
            wall = time.time() - t_wall
            fps  = (frame + 1) / max(wall, 1e-3)
            window.GUI.begin("HUD", 0.01, 0.01, 0.22, 0.16)
            window.GUI.text(f"sim t   {sim_time:7.1f} s")
            window.GUI.text(f"hmax    {hmax:6.2f} m")
            window.GUI.text(f"volume  {vol:11.0f} m^3")
            window.GUI.text(f"Q in    {current_inflow:6.0f} m^3/s")
            window.GUI.text(f"fps     {fps:5.1f}")
            window.GUI.end()
        else:
            window.GUI.begin("HUD", 0.01, 0.01, 0.22, 0.16)
            window.GUI.text(f"sim t   {sim_time:7.1f} s")
            window.GUI.text(f"Q in    {current_inflow:6.0f} m^3/s")
            window.GUI.end()

        window.show()
        frame += 1


if __name__ == "__main__":
    main()
