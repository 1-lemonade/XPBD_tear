"""Particle-grid transfer schemes. First (and only) scheme: MLS-MPM with APIC and
quadratic B-spline weights (Hu et al. 2018), formulas as in the 2026-10-30 code.

P2G affine term (D6): A_p = m C - dt V0 (4/dx^2) tau.
G2P: v, C from the grid; F_trial = (I + dt C) F; model.update -> F, tau, flag (and the
model's own state); x += dt v with the grid's safety clamp; grows the grid's active box.

Interface: p2g(P, model, dt), g2p(P, model, dt) are ti.funcs holding one top-level
particle loop each, called from the solver's substep kernel; dt is the solver's 0-d
field, read inside the loop (D21, see solver.py).
"""
import taichi as ti

from .config import check_keys


@ti.data_oriented
class MLSAPICQuadratic:
    name = "mls_apic_quadratic"

    def __init__(self, grid, tau_cache: bool = True):
        self.grid = grid
        self.dtype = grid.dtype
        self.tau_cache = bool(tau_cache)   # D2: cached tau (default) or recompute in P2G

    def describe(self) -> dict:
        return dict(scheme=self.name, tau_cache=self.tau_cache)

    @ti.func
    def _weights(self, x):
        base = (x * self.grid.inv_dx - 0.5).cast(int)
        fx = x * self.grid.inv_dx - base.cast(self.dtype)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        return base, fx, w[0], w[1], w[2]

    @ti.func
    def p2g(self, P: ti.template(), model: ti.template(), dt: ti.template()):
        for p in range(P.shape[0]):
            h = dt[None]
            base, fx, w0, w1, w2 = self._weights(P.x[p])
            w = [w0, w1, w2]
            tau = P.tau[p]
            if ti.static(not self.tau_cache):
                tau = model.stress(P, p, P.F[p])
            stress_scale = -h * P.V0[p] * 4.0 * self.grid.inv_dx * self.grid.inv_dx
            affine = stress_scale * tau + P.m[p] * P.C[p]
            mv = P.m[p] * P.v[p]
            for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = (off.cast(self.dtype) - fx) * self.grid.dx
                weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                self.grid.v[base + off] += weight * (mv + affine @ dpos)
                self.grid.m[base + off] += weight * P.m[p]

    @ti.func
    def g2p(self, P: ti.template(), model: ti.template(), dt: ti.template()):
        for p in range(P.shape[0]):
            h = dt[None]
            base, fx, w0, w1, w2 = self._weights(P.x[p])
            w = [w0, w1, w2]
            new_v = ti.Vector.zero(self.dtype, 3)
            new_C = ti.Matrix.zero(self.dtype, 3, 3)
            for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                dpos = off.cast(self.dtype) - fx
                g_v = self.grid.v[base + off]
                weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                new_v += weight * g_v
                new_C += 4.0 * self.grid.inv_dx * weight * g_v.outer_product(dpos)
            P.v[p] = new_v
            P.C[p] = new_C

            F_trial = (ti.Matrix.identity(self.dtype, 3) + h * new_C) @ P.F[p]
            F_new, tau, flag = model.update(P, p, F_trial)
            P.F[p] = F_new
            P.tau[p] = tau
            P.flag[p] = ti.cast(flag, ti.i8)

            x_c = self.grid.clamp(P.x[p] + h * new_v)
            P.x[p] = x_c
            self.grid.note_particle(x_c)


SCHEMES = {MLSAPICQuadratic.name: MLSAPICQuadratic}


def build(spec: dict, grid):
    check_keys("transfer", spec, ("scheme", "tau_cache"))
    if spec["scheme"] not in SCHEMES:
        raise KeyError(f"unknown transfer scheme {spec['scheme']!r}; known {sorted(SCHEMES)}")
    return SCHEMES[spec["scheme"]](grid, tau_cache=spec.get("tau_cache", True))
