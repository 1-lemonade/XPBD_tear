"""Solver diagnostics (), one row per call. Reductions are accumulated in f64.

Particle side (2026-10-30 keys): mass, momentum, kinetic, vmax, com_y, CFL, flag
fractions, inverted, clamped_total, nonfinite (particles whose x or v is inf/NaN or
beyond 1e30; bit test, see guards.py). t is the in-kernel clock (f64 sum of dt).

Momentum balance (U2). With P = sum m_p v_p, M = sum m_p, G = sum of the dt*g the
kernel added (solver clock, gdt) and J_s the impulse delivered to each momentum sink
(solver.momentum_sinks()), measured from the moment this object is created:
    r = dP - M dG + sum_s dJ_s        (zero in exact arithmetic)
    bal_scale = M |dG| + sum_s |dJ_s| + sum_p m_p |v_p|
    bal_rel = |r| / bal_scale         (0 when r = 0, e.g. the reference row)
The scale holds every term that carries round-off (gravity, sinks, the momentum the
transfers move), so it stays meaningful without gravity or without boundaries.
P2G and G2P conserve momentum for quadratic B-splines (sum_i w_ip (x_i - x_p) = 0,
same weights both ways), the grid update changes it only by gravity and the sinks,
and the position clamp leaves velocities alone; r holds round-off only. Sink
impulses must not be reset while this object is in use.

Grid check, inside grid.box() only (nodes outside are undefined): after the last
substep the grid holds the P2G mass and the updated velocity that G2P read, so
grid_mass_rel = (sum m_i - M) / M (a run-time safety net for the box; the per-substep
box test is tests/test_pipeline.py) and g2p_rel = |sum m_i v_i - P| / sum m_i |v_i|.
NaN before the first substep.
"""
import math

import numpy as np
import taichi as ti

from .guards import not_finite

NAN = float("nan")

# column -> (unit, meaning); sink columns are generated per sink
COLUMNS = {
    "t": ("s", "simulated time (in-kernel sum of dt)"),
    "mass": ("kg", "sum of particle masses"),
    "kinetic": ("J", "particle kinetic energy"),
    "vmax": ("m/s", "max particle speed"),
    "cfl": ("1", "dt (c_s + vmax) / dx"),
    "com_y": ("m", "centre of mass height"),
    "frac_tip": ("1", "fraction of particles with flag 0 (cone tip)"),
    "frac_elastic": ("1", "fraction with flag 1"),
    "frac_plastic": ("1", "fraction with flag 2"),
    "inverted": ("count", "particles with det F <= 0"),
    "clamped_total": ("count", "position clamps since start"),
    "nonfinite": ("count", "particles whose x or v is inf/NaN or |.| > 1e30"),
    "px": ("kg m/s", "particle momentum x"), "py": ("kg m/s", "particle momentum y"),
    "pz": ("kg m/s", "particle momentum z"),
    "p_abs": ("kg m/s", "sum of m_p |v_p|"),
    "step": ("count", "substeps done"),
    "box_nodes": ("count", "grid nodes in the active box of the last substep"),
    "grid_mass": ("kg", "sum of grid node masses in the box after the last substep"),
    "grid_mass_rel": ("1", "(grid_mass - mass) / mass"),
    "g2p_rel": ("1", "|sum m_i v_i - P| / sum m_i |v_i| over the box"),
    "nonfinite_events": ("count", "(particle, substep) pairs flagged by the G2P guard since start"),
    "nonfinite_step": ("count", "substep of the first guard event, -1 if none"),
    "gdt_x": ("m/s", "sum of dt*g added to grid velocities since start, x"),
    "gdt_y": ("m/s", "same, y"), "gdt_z": ("m/s", "same, z"),
    "bal_px": ("N s", "momentum balance residual x"), "bal_py": ("N s", "momentum balance residual y"),
    "bal_pz": ("N s", "momentum balance residual z"),
    "bal_scale": ("N s", "M |d gdt| + sum_s |d J_s| + p_abs"),
    "bal_rel": ("1", "|residual| / bal_scale (0 when the residual is 0)"),
}
# contact block (P1-U4): the export contract v1 reserved names plus a few more; the G
# measure columns contact_JG_x/y/z are the momentum sink 'contact' (sink columns)
CONTACT_COLUMNS = {
    "contact_JP_x": ("N s", "contact impulse material -> cloth, particle-side target measure (P), cumulative"),
    "contact_JP_y": ("N s", "same, y"), "contact_JP_z": ("N s", "same, z"),
    "contact_Jabs_x": ("N s", "G_on_pinned (named J_absorbed up to U4): part of the G measure attributed to "
                              "pinned cloth vertices, cumulative"),
    "contact_Jabs_y": ("N s", "same, y"), "contact_Jabs_z": ("N s", "same, z"),
    "contact_KG_x": ("N m s", "angular impulse of the realised grid impulse (G) about the world origin, "
                              "sum_g x_g x dJ_g, cumulative (U5)"),
    "contact_KG_y": ("N m s", "same, y"), "contact_KG_z": ("N m s", "same, z"),
    "contact_epoch": ("count", "exchange periods H completed (= stub step_index)"),
    "contact_active": ("count", "particles constrained (gap <= d_hat) in the last substep"),
    "contact_candidates": ("count", "particles in the candidate table of the current period"),
    "contact_overflow": ("count", "candidate-table overflows since start (the scene stops on > 0)"),
    "contact_bin_overflow": ("count", "bin-table entries dropped since start (the scene stops on > 0)"),
    "contact_ambiguous": ("count", "candidates whose near-tied faces disagree on the side, last substep"),
    "contact_pen_max": ("m", "deepest candidate on the illegal side (z = 1): its distance r behind the sheet, "
                             "last substep (recorded only)"),
    "contact_penetrated": ("count", "candidates on the illegal side (z = 1) in the last substep"),
    "contact_pen_max_run": ("m", "largest contact_pen_max over all substeps since start"),
    "contact_penetrated_max_run": ("count", "largest contact_penetrated over all substeps since start"),
    "contact_overlap_max": ("m", "largest r0 - r of a legal-side (z = 0) constrained particle, last substep"),
    "contact_overlap_max_run": ("m", "largest contact_overlap_max over all substeps since start"),
    "contact_missed": ("count", "particles found in the active zone at a rebuild that were not candidates before"),
    "contact_rq": ("m", "candidate query radius (unsigned distance) of the current period"),
    "contact_gain": ("1", "realised fraction of the last substep's targets: sum m (v - v_in).r / sum m |r|^2"),
    "contact_JG_split_rel": ("1", "|sum of per-vertex G attribution - realised G| / |realised G|"),
}
PARTICLE_KEYS = ["mass", "kinetic", "vmax", "cfl", "com_y", "frac_tip", "frac_elastic", "frac_plastic",
                 "inverted", "clamped_total", "nonfinite"]


@ti.data_oriented
class Diagnostics:
    def __init__(self, sim):
        self.sim = sim
        self._f = ti.field(ti.f64, 10)
        self._i = ti.field(ti.i32, 6)
        self._g = ti.field(ti.f64, 6)       # grid: m, m v (3), m |v|, nodes in box
        self.sinks = sim.momentum_sinks()
        self.ref = None                     # balance reference, taken on the first call

    def sink_columns(self) -> dict:
        cols = {}
        plates = {f"plate{k}" for k, b in enumerate(self.sim.bset) if b.kind == "plate"}
        names = self.sim.sink_columns()
        for label, _ in self.sinks:
            if label in plates:
                cols[f"{label}_pos"] = ("m", f"{label} position along its axis")
            for c, name in zip("xyz", names[label]):
                what = "realised grid impulse (G measure) of the contact block" if label == "contact" else "impulse"
                cols[name] = ("N s", f"{what} delivered by the material to {label}, {c}, since start")
        return cols

    def columns(self) -> dict:
        """Ordered column spec of a diag row: v0 columns (t ... pz, plate read-outs), then U2 ones."""
        sink = self.sink_columns()
        v0 = ["t"] + PARTICLE_KEYS + ["px", "py", "pz"]
        u2 = ["step", "p_abs", "box_nodes", "grid_mass", "grid_mass_rel", "g2p_rel", "nonfinite_events",
              "nonfinite_step", "gdt_x", "gdt_y", "gdt_z", "bal_px", "bal_py", "bal_pz", "bal_scale", "bal_rel"]
        out = {k: COLUMNS[k] for k in v0}
        out.update(sink)
        out.update({k: COLUMNS[k] for k in u2})
        if self.sim.contact is not None:
            out.update(CONTACT_COLUMNS)
        return out

    @ti.kernel
    def _reduce(self):
        for i in range(10):
            self._f[i] = 0.0
        for i in range(6):
            self._i[i] = 0
        for i in range(6):
            self._g[i] = 0.0
        for p in range(self.sim.P.shape[0]):
            v = ti.cast(self.sim.P.v[p], ti.f64)
            m = ti.cast(self.sim.P.m[p], ti.f64)
            ti.atomic_add(self._f[0], m)
            ti.atomic_add(self._f[1], m * v[0])
            ti.atomic_add(self._f[2], m * v[1])
            ti.atomic_add(self._f[3], m * v[2])
            ti.atomic_add(self._f[4], 0.5 * m * v.dot(v))
            ti.atomic_max(self._f[5], v.norm())
            ti.atomic_add(self._f[6], m * ti.cast(self.sim.P.x[p][1], ti.f64))
            ti.atomic_add(self._f[7], m * v.norm())
            finite = 1
            for d in ti.static(range(3)):
                x_d = self.sim.P.x[p][d]
                v_d = self.sim.P.v[p][d]
                if not_finite(x_d, self.sim.dtype) or not_finite(v_d, self.sim.dtype):
                    finite = 0
                if ti.abs(x_d) > 1e30 or ti.abs(v_d) > 1e30:
                    finite = 0
            if finite == 0:
                ti.atomic_add(self._i[0], 1)
            f = ti.cast(self.sim.P.flag[p], ti.i32)
            ti.atomic_add(self._i[1 + f], 1)          # 1 tip, 2 elastic, 3 plastic
            if self.sim.P.F[p].determinant() <= 0:
                ti.atomic_add(self._i[4], 1)
        g = self.sim.grid
        for I in ti.grouped(ti.ndrange((g.lo[None][0], g.hi[None][0]), (g.lo[None][1], g.hi[None][1]),
                                       (g.lo[None][2], g.hi[None][2]))):
            m = ti.cast(g.m[I], ti.f64)
            mv = m * ti.cast(g.v[I], ti.f64)
            ti.atomic_add(self._g[0], m)
            ti.atomic_add(self._g[1], mv[0])
            ti.atomic_add(self._g[2], mv[1])
            ti.atomic_add(self._g[3], mv[2])
            ti.atomic_add(self._g[4], mv.norm())
            ti.atomic_add(self._g[5], 1.0)

    def __call__(self) -> dict:
        self._reduce()
        f = self._f.to_numpy()
        i = self._i.to_numpy()
        gsum = self._g.to_numpy()
        n = max(self.sim.n, 1)
        mass = f[0]
        P = f[1:4]
        clock = self.sim.clock()
        J = {label: field[None].to_numpy().astype(np.float64) for label, field in self.sinks}
        d = dict(
            mass=mass, momentum=P.tolist(), kinetic=f[4], vmax=f[5],
            com_y=f[6] / mass if mass > 0 else 0.0,
            cfl=self.sim.cfl(f[5]),
            nonfinite=int(i[0]),
            frac_tip=i[1] / n, frac_elastic=i[2] / n, frac_plastic=i[3] / n,
            inverted=int(i[4]), clamped_total=int(self.sim.grid.clamp_count[None]),
        )
        for k, b in enumerate(self.sim.bset):
            if b.kind == "plate":
                d[f"plate{k}_pos"] = float(b.pos[None])
        names = self.sim.sink_columns()
        for label, val in J.items():
            for name, x in zip(names[label], val):
                d[name] = float(x)
        if self.sim.contact is not None:
            d.update(self.sim.contact.readout())

        d["t"] = clock["time"]
        d["step"] = clock["step"]
        d["p_abs"] = f[7]
        d["box_nodes"] = int(round(gsum[5]))
        if clock["step"] > 0:
            d["grid_mass"] = gsum[0]
            d["grid_mass_rel"] = (gsum[0] - mass) / mass if mass > 0 else NAN
            d["g2p_rel"] = float(np.linalg.norm(gsum[1:4] - P) / gsum[4]) if gsum[4] > 0 else NAN
        else:
            d["grid_mass"] = d["grid_mass_rel"] = d["g2p_rel"] = NAN
        gr = self.sim.guard.readout()
        d["nonfinite_events"] = gr["events"]
        d["nonfinite_step"] = gr["first_step"]

        gdt = np.asarray(clock["gdt"], np.float64)
        for c, x in zip("xyz", gdt):
            d[f"gdt_{c}"] = float(x)
        if self.ref is None:
            self.ref = dict(P=P.copy(), M=mass, gdt=gdt.copy(), J={k: v.copy() for k, v in J.items()})
        dG = gdt - self.ref["gdt"]
        dJ = [J[k] - self.ref["J"][k] for k in J]
        r = (P - self.ref["P"]) - self.ref["M"] * dG + sum(dJ, np.zeros(3))
        scale = self.ref["M"] * float(np.linalg.norm(dG)) + sum(float(np.linalg.norm(x)) for x in dJ) + f[7]
        for c, x in zip("xyz", r):
            d[f"bal_p{c}"] = float(x)
        d["bal_scale"] = scale
        rn = float(np.linalg.norm(r))
        d["bal_rel"] = rn / scale if scale > 0 else (0.0 if rn == 0 else math.inf)
        return d

    def balance_ref(self) -> dict:
        return None if self.ref is None else dict(M=self.ref["M"], P=self.ref["P"].tolist(), gdt=self.ref["gdt"].tolist())


def nanmax(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(vals) if vals else NAN
