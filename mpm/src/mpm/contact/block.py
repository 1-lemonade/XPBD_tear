"""MPM-side contact block against a stub cloth surface . Structure after SoftMAC's mixed contact (soft_cloth/engine/mpm_simulator.py
grid_op_mixed1-4, primitive/primitive_cloth.py collide_mixed); parameters derived here.

Per exchange period H = exchange_substeps * dt (orchestrated by Solver.substep):
  begin_period()  bins + candidate table from the particle positions at the period start
  n_sub substeps, each running apply() between the grid update and G2P
  end_period()    the surface settles the period's impulse batch and advances its snapshot

Candidate table. Query radius (unsigned distance) R_q = r0 + d_hat + H max|v_cloth| +
c_q (v_max + |g| H) H (v_max: largest particle speed at the period start). Every
triangle is registered in the uniform bins (edge bin_cells * dx) overlapped by its swept
AABB over H inflated by R_q (CSR: count, serial prefix sum, fill; entries past
bin_capacity are counted in bin_overflow). A particle whose unsigned distance r to the
nearest triangle of its bin is <= R_q becomes a candidate (capacity cand_capacity,
excess counted in cand_overflow) and keeps that bin for the whole period: any triangle
within r0 + d_hat of it during the period is in that bin's list as long as the particle
moved less than c_q (v_max + |g| H) H, an estimate, not a bound .
contact_missed counts particles that a rebuild finds in the active zone but that were
not candidates in the previous period; it detects, it does not repair. Overflows stop
the scene at the end of the frame (scenes/_cli.py), like a non-finite value.
The broad phase and the active zone use the unsigned r, never the signed gap: with the
signed gap every particle behind the sheet would qualify .

Geometry and side. Nearest triangle of the bin
(exact closest point, geometry.py) in two passes: r_min, then the lowest triangle row
with r <= r_min + tie_tol, so the choice does not depend on the bin order (atomics);
the window counts as ambiguous when its triangles disagree on the side. On a face, interior edge or interior vertex
the feature's pseudo-normal n gives the side sign(n . (x - cp)); rim features (boundary
edges and their vertices, later also crack edges) only have a distance (z = 0) and push
along (x - cp) / r, so material can flow round a free edge.
  side = winding: the legal side is +n for every particle (z = 1 on the -n side).
  side = tracked (persistent side tracking): each particle remembers its side s_p = +-1
    (member contact_side; 0 = none). A particle entering the candidate table learns the
    side it is on (on a rim feature: the side of the triangle's plane); while its
    nearest feature is a rim, s_p follows the plane side (flowing round an edge or through
    a hole or crack is legal); on a face / interior feature z = 1 when its side differs
    from s_p; s_p is forgotten when the particle leaves the table. No face adjacency is
    stored, so topology changes (cracks) need nothing beyond the rim flags; the mesh must
    be consistently wound (orientable). winding is tracked with s_p = +1 for all. Idea
    from SoftMAC's per-particle penetration memory (mpm_simulator.py trace_penetration_*),
    without its static neighbour table, which a tearing mesh would invalidate.
Signed distance dist = (1 - 2z) r, signed gap d = dist - r0.

Particle-side target velocity (), for candidates in the active zone r <= r0 + d_hat
(constrained); all velocities in the bracket are relative to the cloth:
  v_in from the updated grid (G2P weights); v_c = barycentric cloth velocity at cp;
  v_rel = v_in - v_c, v_n = v_rel . n (n towards the legal side)
  approaching (v_n < 0): normal part removed, tangential part scaled by
  max(0, 1 - mu |v_n| / |v_t|) (Coulomb in impulse form; |v_t| = 0 gives 0, no 0/0)
  -> v_out (relative); if d < 0 the relative normal velocity is raised to at least
  min(kappa (-d) / dt, push_vmax) (push-out, kappa of the gap per substep, capped);
  s = min(exp(-beta d), 1); v_tgt = v_c + s v_out + (1 - s) v_rel; r_p = v_tgt - v_in.
A particle on the illegal side (z = 1) deeper than r0 + d_hat is counted (penetrated),
not pulled back through the sheet.
Grid correction (), scattered into the scratch dv and applied by a node loop:
  A  dv_g = alpha sum_p w_gp r_p                         (single gradient step, alpha 0.2)
  B  dv_g = alpha sum_p w_gp m_p r_p / sum_p w_gp m_p    (mass-weighted Jacobi, alpha 1)
  C  dv_g = alpha sum_p w_gp m_p r_p / m_g               (impulse splat, alpha 1; added in U4)
     m_g is the node's full P2G mass, so the grid receives exactly alpha sum_p m_p r_p:
     G = alpha P identically, and no node moves by more than alpha max r_p. With A the
     grid receives alpha (sum_g w_gp m_g / m_p) P (more or less than P), with B about
     (m_g / m_A,g) P >= P. In all three the free particles sharing a node get the same
     node change as the constrained ones; the correction also enters the APIC C of the
     particles (G2P of the corrected grid), not only their velocity.
Accounting (), impulse delivered BY the material TO the cloth, f64, cumulative
(the negative of v2's J_p = m_p (v' - v_in), which is the impulse on the particle; D20):
  JP  sum_p m_p (v_in - v_tgt), the target impulse (P measure). The grid receives
      G = (A) alpha (sum_g w_gp m_g / m_p) P per particle, i.e. alpha ppc_eff P (;
      F3 n48: G/P = 0.89, so more or less than P depending on alpha and the stencil
      mass), (B) about (m_g / m_A,g) P >= P, (C) exactly alpha P. contact_gain is a
      different quantity: the fraction of its own residual that a constrained particle
      receives through G2P (projection on r_p); the rest of the node change goes to the
      free particles sharing the nodes, so gain < G/P in general 
  JG  sum_g m_g (v_before - v_after) of the correction, the realised grid impulse: the
      momentum sink 'contact' (D17), closes the balance to round-off
  per vertex: each particle's P impulse and the grid impulse it caused (A: alpha r_p
  sum_g w_gp m_g; B: alpha m_p r_p sum_g w_gp m_g / sum_q w_gq m_q) are split over the
  three vertices of its face with the barycentric weights; JG_attr is the sum of these
  (equals JG up to round-off, contact_JG_split_rel); G_on_pinned is the part on pinned
  vertices (named J_absorbed up to U4; diag columns keep the contract v1 names contact_Jabs_*).
  KG  sum_g x_g x m_g (v_before - v_after), the angular impulse of the realised grid
      impulse about the world origin (U5; x_g = I dx). With C, since the quadratic B-spline
      weights reproduce linear functions (sum_g w_gp x_g = x_p), KG = sum_p x_p x G_p,
      while the vertex attribution carries sum_i x_i x G_i = sum_p cp_p x G_p (barycentric,
      linear): the two differ by sum_p (x_p - cp_p) x G_p, the lever arm of the contact
      layer, which only the tangential part of G_p feeds on faces (x_p - cp_p is along the
      normal there); without friction they agree to round-off (tests/test_replay.py).

Window clock and hooks (U5): end_period() settles the window with H_ns = round(H 1e9) when
H = exchange_substeps dt is a whole number of ns (else None) and then calls every function
in period_hooks with the block (I/O only, e.g. scenes/f3_replay.py copies the batch).

D21: every run-time quantity a loop uses (dt, time, the parameters, R_q, the candidate
count) is read from a field inside the loop body; serial code uses locals of its own.
"""
import math

import numpy as np
import taichi as ti

from ..config import check_keys
from .geometry import closest_point
from .surface import build_surface

CORRECTIONS = ("A", "B", "C")
SIDES = ("winding", "tracked")
NONE, CANDIDATE, CONSTRAINED = 0, 1, 2
BIG = 1e30


def _pick(spec, direct, scaled, unit, where):
    """Exactly one of spec[direct] (absolute) or spec[scaled] (times unit)."""
    has_d, has_s = spec.get(direct) is not None, spec.get(scaled) is not None
    if has_d == has_s:
        raise KeyError(f"{where}: give exactly one of {direct!r} or {scaled!r} (no built-in default)")
    return float(spec[direct]) if has_d else float(spec[scaled]) * unit


@ti.data_oriented
class ContactBlock:
    label = "contact"
    KEYS = ("surface", "exchange_substeps", "side", "correction", "alpha", "mu", "kappa", "push_vmax",
            "r0", "c_r", "d_hat", "c_d", "beta", "beta_dhat", "c_q", "bin_cells",
            "cand_capacity", "bin_capacity", "tie_tol")
    RUNTIME = ("alpha", "mu", "kappa", "push_vmax", "r0", "d_hat", "beta", "c_q")

    def __init__(self, spec, grid, transfer, dtype, n_particles, ppc):
        where = "contact"
        check_keys(where, spec, self.KEYS)
        for k in ("surface", "exchange_substeps", "mu", "kappa", "push_vmax"):
            if spec.get(k) is None:
                raise KeyError(f"{where}: missing {k!r} (no built-in default)")
        self.correction = spec.get("correction", "A")
        self.side_mode = spec.get("side", "winding")
        if self.correction not in CORRECTIONS or self.side_mode not in SIDES:
            raise ValueError(f"{where}: correction in {list(CORRECTIONS)}, side in {list(SIDES)}")
        self.n_sub = int(spec["exchange_substeps"])
        if self.n_sub < 1:
            raise ValueError(f"{where}.exchange_substeps must be >= 1")
        self.grid, self.tr, self.dtype = grid, transfer, dtype
        self.n_particles = int(n_particles)
        dx = grid.dx
        self.s_p = dx / ppc ** (1.0 / 3.0)
        r0 = _pick(spec, "r0", "c_r", self.s_p, where)
        d_hat = _pick(spec, "d_hat", "c_d", self.s_p, where)
        beta = _pick(spec, "beta", "beta_dhat", 1.0 / d_hat if d_hat > 0 else math.inf, where)
        if r0 < 0 or d_hat < 0 or beta < 0:
            raise ValueError(f"{where}: r0, d_hat, beta must be >= 0")
        alpha = float(spec.get("alpha", 0.2 if self.correction == "A" else 1.0))   # A: original paper; B, C: 1
        self.surface = build_surface(spec["surface"], dtype, where=f"{where}.surface")
        S = self.surface

        self.bin_cells = int(spec.get("bin_cells", 2))
        self.h_b = self.bin_cells * dx
        self.nb = int(math.ceil(grid.domain / self.h_b - 1e-9))
        self.nb3 = self.nb ** 3
        self.cand_cap = int(spec.get("cand_capacity") or n_particles)
        self.bin_cap = int(spec.get("bin_capacity") or max(4096, 64 * S.nt))
        self.tie_tol = float(spec["tie_tol"]) if spec.get("tie_tol") is not None else \
            (1e-9 if dtype == ti.f64 else 1e-5) * dx
        self.tiny = 1e-9 * dx
        self.inv_hb = 1.0 / self.h_b
        self.spec = dict(spec, correction=self.correction, side=self.side_mode)

        f = lambda t=dtype: ti.field(t, ())
        self.alpha, self.mu, self.kappa, self.push_vmax, self.r0, self.d_hat, self.beta, self.c_q = (f() for _ in range(8))
        # candidate table
        C = self.cand_cap
        self.cand_p = ti.field(ti.i32, C)
        self.cand_bin = ti.field(ti.i32, C)
        self.cand_face = ti.field(ti.i32, C)
        self.cand_r = ti.Vector.field(3, dtype, C)        # residual v_tgt - v_in of the last substep
        self.cand_vin = ti.Vector.field(3, dtype, C)      # v_in of the last substep (gain diagnostic)
        self.cand_bary = ti.Vector.field(3, dtype, C)
        self.n_cand = ti.field(ti.i32, ())
        self.n_cand_raw = ti.field(ti.i32, ())
        # bins (CSR)
        self.bin_count = ti.field(ti.i32, self.nb3)
        self.bin_cursor = ti.field(ti.i32, self.nb3)
        self.bin_start = ti.field(ti.i32, self.nb3 + 1)
        self.bin_items = ti.field(ti.i32, self.bin_cap)
        self.bin_used = ti.field(ti.i32, ())
        # grid scratch: A holds alpha sum w r; B holds sum w m r and sum w m (den)
        n = grid.n
        self.dv = ti.Vector.field(3, dtype, (n, n, n))
        self.den = ti.field(dtype, (n, n, n) if self.correction == "B" else ())
        # accounting [N s], f64, cumulative from construction
        v3 = lambda: ti.Vector.field(3, ti.f64, ())
        self.JP, self.JG, self.JG_attr, self.G_on_pinned = v3(), v3(), v3(), v3()
        self.KG = v3()                                    # angular impulse of JG about the world origin [N m s]
        # per-substep and running diagnostics
        self.n_active = ti.field(ti.i32, ())
        self.n_pen = ti.field(ti.i32, ())
        self.n_ambig = ti.field(ti.i32, ())
        self.pen_max = ti.field(dtype, ())                # deepest z = 1 candidate (r behind the face)
        self.pen_max_run = ti.field(dtype, ())
        self.n_pen_max_run = ti.field(ti.i32, ())
        self.overlap_max = ti.field(dtype, ())            # largest r0 - r of a z = 0 constrained particle
        self.overlap_max_run = ti.field(dtype, ())
        self._gain = ti.field(ti.f64, 2)                  # sum m (v - v_in).r, sum m |r|^2 (last substep)
        self.cand_overflow = ti.field(ti.i32, ())
        self.bin_overflow = ti.field(ti.i32, ())
        self.missed = ti.field(ti.i32, ())
        self.rebuilds = ti.field(ti.i32, ())
        self.Rq = ti.field(dtype, ())
        self.vmax_p = ti.field(dtype, ())
        self.vmax_c = ti.field(dtype, ())
        self.set_params(alpha=alpha, mu=spec["mu"], kappa=spec["kappa"], push_vmax=spec["push_vmax"], r0=r0,
                        d_hat=d_hat, beta=beta, c_q=spec.get("c_q", 2.0))
        self.P = self.dt = self.time = self.gravity = None
        self.period_hooks = []                            # f(block) after each settled window (python, I/O only)
        self._sim = None

    # ------------------------------------------------------------ python side
    def state_members(self) -> dict:
        m = dict(contact_state=ti.i8,           # 0 none, 1 candidate, 2 constrained (last substep)
                 contact_face=ti.i32,           # row of the nearest triangle, -1 none
                 contact_dist=self.dtype)       # signed distance (1 - 2z) r to it [m] (not minus r0)
        if self.side_mode == "tracked":
            m["contact_side"] = ti.i8           # remembered side +1 / -1 of the surface, 0 none
        return m

    def bind(self, sim):
        """Solver fields used by the kernels (after the particle field is built)."""
        self.P, self.dt, self.time, self.gravity, self.n = sim.P, sim.dt, sim.time, sim.gravity, sim.n
        self._sim = sim

    def window_ns(self):
        """H in whole nanoseconds (int), or None when exchange_substeps * dt is not one
        (python value of dt, no device read)."""
        H = self.n_sub * float(self._sim.dt_host) * 1e9
        k = round(H)
        return int(k) if k > 0 and abs(H - k) <= 1e-6 else None

    def set_params(self, **kw):
        check_keys("contact.set_params", kw, self.RUNTIME)
        for k, v in kw.items():
            if v is None:
                continue
            getattr(self, k)[None] = float(v)
            self.spec[f"{k}_now"] = float(v)

    def params(self) -> dict:
        return {k: float(getattr(self, k)[None]) for k in self.RUNTIME}

    @property
    def STAGES(self):
        if self.correction in ("A", "C"):
            return ("contact_eval", "contact_scatter", "contact_apply")
        return ("contact_zero", "contact_eval", "contact_scatter", "contact_apply", "contact_attr")

    def begin_period(self):
        self._rebuild_radius()
        self._rebuild_tables()

    def end_period(self):
        self.surface.settle_window(self.time, self.window_ns())
        for hook in self.period_hooks:
            hook(self)

    def describe(self) -> dict:
        return dict(spec=self.spec, resolved=dict(self.params(), s_p=self.s_p, H=self.n_sub * float(self.dt[None]),
                                                   exchange_substeps=self.n_sub),
                    bins=dict(cells=self.bin_cells, edge=self.h_b, per_axis=self.nb, capacity=self.bin_cap),
                    cand_capacity=self.cand_cap, tie_tol=self.tie_tol, surface=self.surface.describe(),
                    stages=list(self.STAGES))

    def readout(self) -> dict:
        """Diag values (cumulative impulses, counters of the last substep / since start)."""
        JP, JG, JA, Gpin, KG = (np.asarray(f[None].to_numpy(), np.float64)
                                for f in (self.JP, self.JG, self.JG_attr, self.G_on_pinned, self.KG))
        d = {}
        for c, i in zip("xyz", range(3)):
            d[f"contact_JP_{c}"] = float(JP[i])
            d[f"contact_Jabs_{c}"] = float(Gpin[i])          # G_on_pinned (contract v1 column name)
            d[f"contact_KG_{c}"] = float(KG[i])
        nG = float(np.linalg.norm(JG))
        self._gain_kernel()
        gn, gd = self._gain.to_numpy()
        d.update(contact_epoch=int(self.surface.step_index[None]), contact_active=int(self.n_active[None]),
                 contact_candidates=int(self.n_cand[None]),
                 contact_overflow=int(self.cand_overflow[None]), contact_bin_overflow=int(self.bin_overflow[None]),
                 contact_ambiguous=int(self.n_ambig[None]), contact_pen_max=float(self.pen_max[None]),
                 contact_penetrated=int(self.n_pen[None]), contact_pen_max_run=float(self.pen_max_run[None]),
                 contact_penetrated_max_run=int(self.n_pen_max_run[None]),
                 contact_overlap_max=float(self.overlap_max[None]), contact_overlap_max_run=float(self.overlap_max_run[None]),
                 contact_missed=int(self.missed[None]), contact_rq=float(self.Rq[None]),
                 contact_gain=float(gn / gd) if gd > 0 else float("nan"),
                 contact_JG_split_rel=float(np.linalg.norm(JA - JG) / nG) if nG > 0 else 0.0)
        return d

    @ti.kernel
    def _gain_kernel(self):
        """Realised fraction of the last substep's targets: G2P gives v = sum w v_g(after) and
        v_in = sum w v_g(before) with the same weights, so v - v_in is exactly the velocity
        the correction gave the particle; gain = sum m (v - v_in).r / sum m |r|^2."""
        self._gain[0] = 0.0
        self._gain[1] = 0.0
        for c in range(self.n_cand[None]):
            r = ti.cast(self.cand_r[c], ti.f64)
            p = self.cand_p[c]
            m = ti.cast(self.P.m[p], ti.f64)
            dvp = ti.cast(self.P.v[p], ti.f64) - ti.cast(self.cand_vin[c], ti.f64)
            ti.atomic_add(self._gain[0], m * dvp.dot(r))
            ti.atomic_add(self._gain[1], m * r.dot(r))

    # ------------------------------------------------------------ geometry helpers
    @ti.func
    def _bin_index(self, x):
        b = ti.cast(ti.floor(x * self.inv_hb), ti.i32)
        b = ti.min(ti.max(b, 0), self.nb - 1)
        return (b[0] * self.nb + b[1]) * self.nb + b[2]

    @ti.func
    def _tri_bins(self, t):
        """Bin range [lo, hi] (inclusive) of triangle t's AABB swept over H, inflated by R_q."""
        H = ti.cast(self.n_sub * self.dt[None], self.dtype)
        lo = ti.Vector([BIG, BIG, BIG], self.dtype)
        hi = ti.Vector([-BIG, -BIG, -BIG], self.dtype)
        for k in ti.static(range(3)):
            v = self.surface.tri[t][k]
            a = self.surface.vertex_at(v, 0.0)
            b = self.surface.vertex_at(v, H)
            lo = ti.min(lo, ti.min(a, b))
            hi = ti.max(hi, ti.max(a, b))
        rq = self.Rq[None]
        ilo = ti.cast(ti.floor((lo - rq) * self.inv_hb), ti.i32)
        ihi = ti.cast(ti.floor((hi + rq) * self.inv_hb), ti.i32)
        ilo = ti.min(ti.max(ilo, 0), self.nb - 1)
        ihi = ti.min(ti.max(ihi, 0), self.nb - 1)
        return ilo, ihi

    @ti.func
    def _tri_query(self, x, t, tau, s):
        """Distance r, z, barycentric, contact normal (towards the legal side), rim flag and
        plane side of x against triangle t at tau; s = the particle's legal side (+1 / -1)."""
        a = self.surface.vertex_at(self.surface.tri[t][0], tau)
        bb = self.surface.vertex_at(self.surface.tri[t][1], tau)
        c = self.surface.vertex_at(self.surface.tri[t][2], tau)
        cp, bary, feat = closest_point(x, a, bb, c, self.dtype)
        diff = x - cp
        r = diff.norm()
        pn, rim = self.surface.side(t, feat, diff)
        z = 0
        if ti.static(self.side_mode == "tracked"):
            side = 1
            if pn.dot(diff) < 0:
                side = -1
            if rim == 0 and side != s:
                z = 1
        else:
            if rim == 0 and pn.dot(diff) < 0:
                z = 1
        n_c = pn
        if ti.static(self.side_mode == "tracked"):
            if s < 0:
                n_c = -pn
        if r > self.tiny:
            n_c = diff / r
            if z == 1:
                n_c = -n_c
        return r, z, bary, n_c, rim, self.surface.plane_side(t, diff)

    @ti.func
    def _nearest(self, x, b, tau, s):
        """Nearest triangle of bin b at tau = t - t_s for a particle of legal side s: (row or
        -1, signed distance, contact normal towards the legal side, barycentric, z,
        ambiguous, rim flag and plane side of the chosen triangle). Two passes, so that the
        result does not depend on the order of the bin list (filled by atomics): r_min
        first, then the lowest row with r <= r_min + tie_tol; ambiguous when the triangles
        in that window disagree on the side ."""
        lo = self.bin_start[b]
        hi = ti.min(self.bin_start[b + 1], self.bin_cap)
        r_min = ti.cast(BIG, self.dtype)
        for k in range(lo, hi):
            t = self.bin_items[k]
            if self.surface.degenerate[t] == 0:
                r, z, bary, n_c, rim, ps = self._tri_query(x, t, tau, s)
                r_min = ti.min(r_min, r)
        best_t = -1
        best_r = ti.cast(BIG, self.dtype)
        best_z = 0
        best_bary = ti.Vector.zero(self.dtype, 3)
        best_n = ti.Vector.zero(self.dtype, 3)
        best_rim, best_ps = 0, 1
        z_lo, z_hi = 1, 0
        for k in range(lo, hi):
            t = self.bin_items[k]
            if self.surface.degenerate[t] == 0:
                r, z, bary, n_c, rim, ps = self._tri_query(x, t, tau, s)
                if r <= r_min + self.tie_tol:
                    z_lo = ti.min(z_lo, z)
                    z_hi = ti.max(z_hi, z)
                    if best_t < 0 or t < best_t:
                        best_t, best_r, best_z, best_bary, best_n = t, r, z, bary, n_c
                        best_rim, best_ps = rim, ps
        ambig = 0
        if z_hi > z_lo:
            ambig = 1
        dist = best_r
        if best_z == 1:
            dist = -best_r
        return best_t, dist, best_n, best_bary, best_z, ambig, best_rim, best_ps

    # ------------------------------------------------------------ period start
    @ti.kernel
    def _rebuild_radius(self):
        """R_q from the speeds at the period start. Its own kernel: a field written in a serial
        part and read by later loops of the same kernel is forwarded by Taichi into a global
        temporary (found by the IR test, U4), which D21 keeps out of every kernel."""
        self.vmax_p[None] = 0.0
        self.vmax_c[None] = 0.0
        for p in range(self.n):
            ti.atomic_max(self.vmax_p[None], self.P.v[p].norm())
        for v in range(self.surface.nv):
            ti.atomic_max(self.vmax_c[None], self.surface.v_s[v].norm())
        H = self.n_sub * self.dt[None]
        g = self.gravity[None].norm()
        self.Rq[None] = (self.r0[None] + self.d_hat[None] + H * self.vmax_c[None]
                         + self.c_q[None] * (self.vmax_p[None] + g * H) * H)

    @ti.kernel
    def _rebuild_tables(self):
        """Bins (CSR) and the candidate table for this period."""
        for b in range(self.nb3):
            self.bin_count[b] = 0
        for t in range(self.surface.nt):
            ilo, ihi = self._tri_bins(t)
            for i in range(ilo[0], ihi[0] + 1):
                for j in range(ilo[1], ihi[1] + 1):
                    for k in range(ilo[2], ihi[2] + 1):
                        ti.atomic_add(self.bin_count[(i * self.nb + j) * self.nb + k], 1)
        self.bin_start[0] = 0
        ti.loop_config(serialize=True)
        for b in range(self.nb3):
            self.bin_start[b + 1] = self.bin_start[b] + self.bin_count[b]
            self.bin_cursor[b] = self.bin_start[b]
        self.bin_used[None] = self.bin_start[self.nb3]
        for t in range(self.surface.nt):
            ilo, ihi = self._tri_bins(t)
            for i in range(ilo[0], ihi[0] + 1):
                for j in range(ilo[1], ihi[1] + 1):
                    for k in range(ilo[2], ihi[2] + 1):
                        slot = ti.atomic_add(self.bin_cursor[(i * self.nb + j) * self.nb + k], 1)
                        if slot < self.bin_cap:
                            self.bin_items[slot] = t
                        else:
                            ti.atomic_add(self.bin_overflow[None], 1)
        self.n_cand_raw[None] = 0
        for p in range(self.n):
            x = self.P.x[p]
            b = self._bin_index(x)
            prev = ti.cast(self.P.contact_state[p], ti.i32)
            state = NONE
            face = -1
            dist_out = ti.cast(0.0, self.dtype)
            side_p = 1
            side_out = 0
            learn = 1
            if ti.static(self.side_mode == "tracked"):
                if prev != NONE and self.P.contact_side[p] != 0:
                    side_p = ti.cast(self.P.contact_side[p], ti.i32)
                    learn = 0
            if self.bin_start[b + 1] > self.bin_start[b]:
                t, dist, n_c, bary, z, amb, rim, ps = self._nearest(x, b, ti.cast(0.0, self.dtype), side_p)
                if t >= 0:
                    r = ti.abs(dist)
                    if r <= self.Rq[None]:
                        slot = ti.atomic_add(self.n_cand_raw[None], 1)
                        if slot < self.cand_cap:
                            self.cand_p[slot] = p
                            self.cand_bin[slot] = b
                            state = CANDIDATE
                            face = t
                            dist_out = dist
                            if ti.static(self.side_mode == "tracked"):
                                side_out = side_p
                                if rim == 1:
                                    side_out = ps                 # round a rim: follow the plane side
                                elif learn == 1:
                                    side_out = 1 - 2 * z          # entering: learn the side it is on
                                    dist_out = r
                        else:
                            ti.atomic_add(self.cand_overflow[None], 1)
                    if r <= self.r0[None] + self.d_hat[None] and prev == NONE and self.rebuilds[None] > 0:
                        ti.atomic_add(self.missed[None], 1)
            self.P.contact_state[p] = ti.cast(state, ti.i8)
            self.P.contact_face[p] = face
            self.P.contact_dist[p] = dist_out
            if ti.static(self.side_mode == "tracked"):
                self.P.contact_side[p] = ti.cast(side_out, ti.i8)    # 0: forgotten off the table
        self.n_cand[None] = ti.min(self.n_cand_raw[None], self.cand_cap)
        self.rebuilds[None] += 1

    # ------------------------------------------------------------ substep (in Solver._substep)
    @ti.func
    def apply(self):
        """Top-level loops between the grid update and G2P (D13 slot)."""
        self.n_active[None] = 0
        self.n_pen[None] = 0
        self.n_ambig[None] = 0
        self.pen_max[None] = 0.0
        self.overlap_max[None] = 0.0
        if ti.static(self.correction == "B"):
            for c in range(self.n_cand[None]):
                base, fx, w0, w1, w2 = self.tr._weights(self.P.x[self.cand_p[c]])
                for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    self.den[base + off] = 0.0
        self._evaluate()
        self.pen_max_run[None] = ti.max(self.pen_max_run[None], self.pen_max[None])
        self.n_pen_max_run[None] = ti.max(self.n_pen_max_run[None], self.n_pen[None])
        self.overlap_max_run[None] = ti.max(self.overlap_max_run[None], self.overlap_max[None])
        self._scatter()
        self.grid.for_each_node(self)
        if ti.static(self.correction == "B"):
            self._attribute_b()

    @ti.func
    def _evaluate(self):
        for c in range(self.n_cand[None]):
            p = self.cand_p[c]
            x = self.P.x[p]
            base, fx, w0, w1, w2 = self.tr._weights(x)
            w = [w0, w1, w2]
            v_in = ti.Vector.zero(self.dtype, 3)
            for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                v_in += w[off[0]][0] * w[off[1]][1] * w[off[2]][2] * self.grid.v[base + off]
            tau = ti.cast(self.time[None] - ti.cast(self.dt[None], ti.f64) - self.surface.t_s[None], self.dtype)
            side_p = 1                  # remembered side (int); not `s`, which below is the blend factor
            if ti.static(self.side_mode == "tracked"):
                side_p = ti.cast(self.P.contact_side[p], ti.i32)
            t, dist, n_c, bary, z, amb, rim, ps = self._nearest(x, self.cand_bin[c], tau, side_p)
            r = ti.Vector.zero(self.dtype, 3)
            state = CANDIDATE
            if ti.static(self.side_mode == "tracked"):
                if t >= 0 and rim == 1:
                    self.P.contact_side[p] = ti.cast(ps, ti.i8)     # round a rim: follow the plane side
            if t >= 0:
                r_abs = ti.abs(dist)
                gap = dist - self.r0[None]
                self.P.contact_face[p] = t
                self.P.contact_dist[p] = dist
                if amb == 1:
                    ti.atomic_add(self.n_ambig[None], 1)
                if z == 1:
                    ti.atomic_add(self.n_pen[None], 1)
                    ti.atomic_max(self.pen_max[None], r_abs)
                elif gap < 0:
                    ti.atomic_max(self.overlap_max[None], -gap)
                if r_abs <= self.r0[None] + self.d_hat[None]:
                    state = CONSTRAINED
                    ti.atomic_add(self.n_active[None], 1)
                    vc = ti.Vector.zero(self.dtype, 3)
                    for k in ti.static(range(3)):
                        vc += bary[k] * self.surface.v_s[self.surface.tri[t][k]]
                    v_rel = v_in - vc
                    vn = v_rel.dot(n_c)
                    v_out = v_rel
                    if vn < 0:
                        vt = v_rel - vn * n_c
                        vt_norm = vt.norm()
                        scale = ti.cast(0.0, self.dtype)
                        if vt_norm > self.tiny:
                            scale = ti.max(0.0, 1.0 - self.mu[None] * (-vn) / vt_norm)
                        v_out = vt * scale
                    if gap < 0:
                        push = ti.min(self.kappa[None] * (-gap) / self.dt[None], self.push_vmax[None])
                        vno = v_out.dot(n_c)
                        if vno < push:
                            v_out += (push - vno) * n_c
                    s = ti.cast(1.0, self.dtype)
                    if gap > 0:
                        s = ti.min(ti.exp(-self.beta[None] * gap), 1.0)
                    r = s * v_out + (1.0 - s) * v_rel - v_rel
                    JP = -ti.cast(self.P.m[p], ti.f64) * ti.cast(r, ti.f64)
                    ti.atomic_add(self.JP[None], JP)
                    for k in ti.static(range(3)):
                        ti.atomic_add(self.surface.JP_per[self.surface.tri[t][k]], ti.cast(bary[k], ti.f64) * JP)
            self.cand_r[c] = r
            self.cand_vin[c] = v_in
            self.cand_face[c] = t
            self.cand_bary[c] = bary
            self.P.contact_state[p] = ti.cast(state, ti.i8)

    @ti.func
    def _scatter(self):
        for c in range(self.n_cand[None]):
            r = self.cand_r[c]
            if r[0] != 0 or r[1] != 0 or r[2] != 0:
                p = self.cand_p[c]
                base, fx, w0, w1, w2 = self.tr._weights(self.P.x[p])
                w = [w0, w1, w2]
                msum = ti.cast(0.0, ti.f64)
                if ti.static(self.correction == "A"):
                    a = self.alpha[None]
                    for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                        self.dv[base + off] += (a * weight) * r
                        msum += ti.cast(weight * self.grid.m[base + off], ti.f64)
                    JG = -ti.cast(a, ti.f64) * msum * ti.cast(r, ti.f64)
                    self._split_G(self.cand_face[c], self.cand_bary[c], JG)
                elif ti.static(self.correction == "C"):
                    mp = self.P.m[p]
                    for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                        self.dv[base + off] += (weight * mp) * r
                    JG = -ti.cast(self.alpha[None] * mp, ti.f64) * ti.cast(r, ti.f64)
                    self._split_G(self.cand_face[c], self.cand_bary[c], JG)
                else:
                    mp = self.P.m[p]
                    for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                        weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                        self.dv[base + off] += (weight * mp) * r
                        self.den[base + off] += weight * mp

    @ti.func
    def _split_G(self, t, bary, JG):
        ti.atomic_add(self.JG_attr[None], JG)
        for k in ti.static(range(3)):
            vtx = self.surface.tri[t][k]
            part = ti.cast(bary[k], ti.f64) * JG
            ti.atomic_add(self.surface.JG_per[vtx], part)
            if self.surface.pinned[vtx] == 1:
                ti.atomic_add(self.G_on_pinned[None], part)

    @ti.func
    def node(self, I):
        """Node loop (grid.for_each_node): apply the scattered correction, record the realised
        impulse m (v_before - v_after) in f64 (as the walls and plates do), clear dv."""
        d = self.dv[I]
        if d[0] != 0 or d[1] != 0 or d[2] != 0:
            m = self.grid.m[I]
            if m > 0:
                dvel = d
                if ti.static(self.correction == "B"):
                    dvel = self.alpha[None] * d / self.den[I]
                elif ti.static(self.correction == "C"):
                    dvel = self.alpha[None] * d / m
                v_old = self.grid.v[I]
                v_new = v_old + dvel
                self.grid.v[I] = v_new
                dJ = ti.cast(m, ti.f64) * (ti.cast(v_old, ti.f64) - ti.cast(v_new, ti.f64))
                ti.atomic_add(self.JG[None], dJ)
                xg = ti.cast(I, ti.f64) * self.grid.dx
                ti.atomic_add(self.KG[None], xg.cross(dJ))
            self.dv[I] = ti.Vector.zero(self.dtype, 3)

    @ti.func
    def _attribute_b(self):
        for c in range(self.n_cand[None]):
            r = self.cand_r[c]
            if r[0] != 0 or r[1] != 0 or r[2] != 0:
                p = self.cand_p[c]
                base, fx, w0, w1, w2 = self.tr._weights(self.P.x[p])
                w = [w0, w1, w2]
                msum = ti.cast(0.0, ti.f64)
                for off in ti.static(ti.grouped(ti.ndrange(3, 3, 3))):
                    weight = w[off[0]][0] * w[off[1]][1] * w[off[2]][2]
                    den = self.den[base + off]
                    if den > 0:
                        msum += ti.cast(weight * self.grid.m[base + off] / den, ti.f64)
                JG = -ti.cast(self.alpha[None] * self.P.m[p], ti.f64) * msum * ti.cast(r, ti.f64)
                self._split_G(self.cand_face[c], self.cand_bary[c], JG)


def build(spec: dict, grid, transfer, dtype, n_particles, ppc):
    return ContactBlock(spec, grid, transfer, dtype, n_particles, ppc) if spec else None
