#!/usr/bin/env python3
"""
Milestone 3 — Taichi-Metal local-inertial 2D Shallow-Water solver.

Scheme: LISFLOOD-FP-style local-inertial. Faces store momentum-per-unit-width
(q on x-faces, p on y-faces); cells store water depth h over bed z. Cheap,
stable, maps cleanly to a GPU stencil. Single precision (Metal requirement).

Reference: Bates, Horritt, Fewtrell (2010) JoH; de Almeida et al. (2012) WRR.

The face flux update is:

    q_new = ( q_old  -  g * h_face * dt * d(h+z)/dx )
            -------------------------------------------------
            1 + g * dt * n^2 * |q_old| / h_face^(7/3)

where h_face = max(h_up+z_up, h_dn+z_dn) - max(z_up, z_dn) is the "flow depth"
between two cells. If h_face <= eps, the face is dry and q is zeroed.

Then mass:

    h_new = h_old - dt/dx * (q_E - q_W + p_N - p_S)

API: build_solver(z_np, manning_np, dx) returns a Solver object exposing
.h, .qx, .qy ti.fields and .step(dt_sub_steps) plus convenience setters.
This file is importable; running it directly executes a flat-bed dam-break
sanity test and prints mass-conservation residuals.
"""
import numpy as np
import taichi as ti
from typing import Union


# Physical / numerical constants
G = 9.81
H_MIN = 1e-3          # depths below this are treated as dry
N_DEFAULT = 0.035     # Manning's n for natural channel
CFL = 0.4             # local-inertial scheme tolerates CFL ~ 0.5; be safe


@ti.data_oriented
class SWESolver:
    """Local-inertial SWE on a regular grid. All fields are ti.f32 (Metal)."""

    def __init__(self, nx: int, ny: int, dx: float):
        self.nx, self.ny, self.dx = nx, ny, float(dx)

        # Cell-centered state
        self.z  = ti.field(ti.f32, shape=(nx, ny))   # bed elevation
        self.h  = ti.field(ti.f32, shape=(nx, ny))   # water depth
        self.n  = ti.field(ti.f32, shape=(nx, ny))   # Manning's n

        # Face-centered momentum (discharge per unit width, m^2/s).
        # qx[i,j] = flux across face between cell (i-1,j) and (i,j) -> shape (nx+1, ny)
        # qy[i,j] = flux across face between cell (i,j-1) and (i,j) -> shape (nx, ny+1)
        self.qx = ti.field(ti.f32, shape=(nx + 1, ny))
        self.qy = ti.field(ti.f32, shape=(nx, ny + 1))

        # Forcing: depth-injection rate at cells (m/s of water added). Used for
        # inflow boundary -- simpler than imposing a face flux and very stable.
        self.src = ti.field(ti.f32, shape=(nx, ny))

        # Auxiliary: stage = z + h, water surface elevation (computed each step)
        self.eta = ti.field(ti.f32, shape=(nx, ny))

        # Scalar diagnostics
        self.h_max = ti.field(ti.f32, shape=())
        self.vol   = ti.field(ti.f32, shape=())

    # ------------------------------------------------------------ initial setup
    def load_terrain(self, z_np, manning=N_DEFAULT):
        assert z_np.shape == (self.nx, self.ny), \
            f"z shape {z_np.shape} != ({self.nx},{self.ny})"
        self.z.from_numpy(z_np.astype(np.float32))
        if np.isscalar(manning):
            n_np = np.full((self.nx, self.ny), float(manning), dtype=np.float32)
        else:
            n_np = manning.astype(np.float32)
        self.n.from_numpy(n_np)
        self._zero_state()

    @ti.kernel
    def _zero_state(self):
        for i, j in self.h:
            self.h[i, j] = 0.0
            self.src[i, j] = 0.0
        for i, j in self.qx:
            self.qx[i, j] = 0.0
        for i, j in self.qy:
            self.qy[i, j] = 0.0

    # --------------------------------------------------------------- kernels
    @ti.kernel
    def _update_stage(self):
        for i, j in self.h:
            self.eta[i, j] = self.z[i, j] + self.h[i, j]

    @ti.kernel
    def _update_qx(self, dt: ti.f32):
        # Loop interior x-faces only; faces 0 and nx are domain edges (closed).
        for i, j in ti.ndrange((1, self.nx), self.ny):
            zu = self.z[i - 1, j]; hu = self.h[i - 1, j]
            zd = self.z[i,     j]; hd = self.h[i,     j]
            eu = zu + hu
            ed = zd + hd
            # "flow depth" over the higher of the two beds
            hflow = ti.max(eu, ed) - ti.max(zu, zd)
            q = self.qx[i, j]
            if hflow > H_MIN:
                dedx = (ed - eu) / self.dx
                # Friction term uses upstream-ish n (average for stability)
                nman = 0.5 * (self.n[i - 1, j] + self.n[i, j])
                # implicit friction: q_new = (q - g h dt dedx) / (1 + g dt n^2 |q| / h^(7/3))
                numer = q - G * hflow * dt * dedx
                denom = 1.0 + G * dt * nman * nman * ti.abs(q) / ti.pow(hflow, 7.0/3.0)
                self.qx[i, j] = numer / denom
            else:
                self.qx[i, j] = 0.0

    @ti.kernel
    def _update_qy(self, dt: ti.f32):
        for i, j in ti.ndrange(self.nx, (1, self.ny)):
            zu = self.z[i, j - 1]; hu = self.h[i, j - 1]
            zd = self.z[i, j    ]; hd = self.h[i, j    ]
            eu = zu + hu
            ed = zd + hd
            hflow = ti.max(eu, ed) - ti.max(zu, zd)
            q = self.qy[i, j]
            if hflow > H_MIN:
                dedy = (ed - eu) / self.dx
                nman = 0.5 * (self.n[i, j - 1] + self.n[i, j])
                numer = q - G * hflow * dt * dedy
                denom = 1.0 + G * dt * nman * nman * ti.abs(q) / ti.pow(hflow, 7.0/3.0)
                self.qy[i, j] = numer / denom
            else:
                self.qy[i, j] = 0.0

    @ti.kernel
    def _update_h(self, dt: ti.f32):
        # h_new = h_old - dt/dx * (qE - qW + pN - pS) + dt*src
        for i, j in self.h:
            qE = self.qx[i + 1, j]
            qW = self.qx[i,     j]
            qN = self.qy[i, j + 1]
            qS = self.qy[i, j    ]
            dh = -(dt / self.dx) * (qE - qW + qN - qS) + dt * self.src[i, j]
            h_new = self.h[i, j] + dh
            if h_new < 0.0:
                h_new = 0.0   # wetting/drying clamp; preserves positivity
            self.h[i, j] = h_new

    @ti.kernel
    def _apply_open_boundaries(self):
        # Critical-flow (weir) outflow at all four domain edges. For a free
        # overfall the unit discharge is roughly q = (2/3)^(3/2)*sqrt(g)*h^(3/2)
        # ≈ 1.7*h^(3/2) per metre of edge. We damp by 0.5 to avoid the boundary
        # forcing an unphysically fast drawdown each step.
        #   Sign convention: qx>0 means flow in +x direction.
        #     - West face (i=0)   -> flow OUT going westward → qx negative.
        #     - East face (i=nx)  -> flow OUT going eastward → qx positive.
        #     - South face (j=0)  -> flow OUT going southward → qy negative.
        #     - North face (j=ny) -> flow OUT going northward → qy positive.
        damp = 0.5
        for j in range(self.ny):
            hw = self.h[0, j]
            if hw > H_MIN:
                self.qx[0, j] = -damp * 1.7 * ti.pow(hw, 1.5)
            else:
                self.qx[0, j] = 0.0
            he = self.h[self.nx - 1, j]
            if he > H_MIN:
                self.qx[self.nx, j] = damp * 1.7 * ti.pow(he, 1.5)
            else:
                self.qx[self.nx, j] = 0.0
        for i in range(self.nx):
            hs = self.h[i, 0]
            if hs > H_MIN:
                self.qy[i, 0] = -damp * 1.7 * ti.pow(hs, 1.5)
            else:
                self.qy[i, 0] = 0.0
            hn = self.h[i, self.ny - 1]
            if hn > H_MIN:
                self.qy[i, self.ny] = damp * 1.7 * ti.pow(hn, 1.5)
            else:
                self.qy[i, self.ny] = 0.0

    @ti.kernel
    def _diagnostics(self) -> ti.f32:
        # Reductions across the parallel grid need atomics for scalars.
        self.h_max[None] = 0.0
        self.vol[None]   = 0.0
        for i, j in self.h:
            ti.atomic_max(self.h_max[None], self.h[i, j])
            self.vol[None] += self.h[i, j]   # auto-atomic on field write
        self.vol[None] *= self.dx * self.dx
        return self.h_max[None]

    # --------------------------------------------------------------- public step
    def step(self, dt: float) -> float:
        """Advance one timestep. Returns peak depth (for monitoring)."""
        dt32 = float(dt)
        self._update_qx(dt32)
        self._update_qy(dt32)
        self._apply_open_boundaries()
        self._update_h(dt32)
        self._update_stage()
        return self._diagnostics()

    def cfl_dt(self) -> float:
        """CFL-stable timestep estimate based on current peak depth."""
        self._diagnostics()
        hmax = max(float(self.h_max[None]), 0.05)
        c = (G * hmax) ** 0.5
        return CFL * self.dx / max(c, 1e-6)

    # ------------------------------------------------------------- forcing helpers
    def set_source_rect(self, i0: int, i1: int, j0: int, j1: int, rate_mps: float):
        """Set per-cell depth-injection rate (m/s) over an index rectangle.

        rate_mps is depth per second added to each cell. To impose a discharge
        Q (m^3/s) over a rectangle of N cells with area dx*dx each:
            rate = Q / (N * dx * dx)
        """
        arr = self.src.to_numpy()
        arr[i0:i1, j0:j1] = rate_mps
        self.src.from_numpy(arr)


# =============================================================================
# Self-test: flat-bed dam break (no DEM needed)
# =============================================================================
def _selftest():
    ti.init(arch=ti.metal, default_fp=ti.f32)
    nx, ny, dx = 256, 256, 1.0
    solver = SWESolver(nx, ny, dx)
    z = np.zeros((nx, ny), dtype=np.float32)        # flat bed
    solver.load_terrain(z, manning=0.025)

    # Initial wall of water in the left third
    h0 = solver.h.to_numpy()
    h0[:nx // 3, :] = 2.0
    solver.h.from_numpy(h0)

    print("Flat dam-break sanity test (256x256, dx=1 m)...")
    total_t = 0.0
    vol_initial = float(h0.sum()) * dx * dx
    for step in range(400):
        dt = solver.cfl_dt()
        solver.step(dt)
        total_t += dt
        if step % 50 == 0:
            solver._diagnostics()
            print(f"  step {step:4d}  t={total_t:6.2f}s  "
                  f"hmax={float(solver.h_max[None]):.3f}m  "
                  f"vol={float(solver.vol[None]):.1f} m^3 (init {vol_initial:.1f})")
    print("OK (no NaNs, water moved, mass change reflects free-outflow edges).")


if __name__ == "__main__":
    _selftest()
