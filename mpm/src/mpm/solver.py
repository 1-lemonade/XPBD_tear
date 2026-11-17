"""Pipeline assembly: config -> modules -> particle field; one Taichi kernel per substep.

Substep ():
  clock                   (step += 1, time += dt, gdt += dt g; f64 except step)
  grid.begin_substep      (active box of this substep)
  boundaries.pre_substep  (e.g. moving plate advances)
  grid.clear              (active box or whole grid)
  transfer.p2g
  grid.update_nodes       (momentum -> velocity, gravity, boundary list in order)
  contact.apply           (U4, only with a `contact` section: candidate evaluation,
                           correction scatter, node apply; see contact/block.py)
  grid.reset_box; transfer.g2p (constitutive update, advection, clamp, next active box)
  guard.scan              (non-finite x, v, F: count, first substep; detection, see guards.py)
stages() names the top-level parallel loops in this order (kernel profiler labels).

The kernel is `_substep`; `substep()` is the Python entry that also runs the contact
block's exchange period around it (begin_period before the first substep of a period,
end_period after its last; counted in Python, no device read). Without a contact block
substep() only calls the kernel, which is then the U3 kernel unchanged.

Run-time data (no recompilation): dt, gravity, the constitutive parameter table,
boundary parameters. Structural: precision, grid size, model and options, transfer
scheme, boundary composition and friction types, active box.

Kernel-level locals (D21): the parallel loops get dt and gravity as the 0-d fields and
read them inside the loop body; the kernel never holds them in a local used by a
loop. Taichi 1.7.4 turns such a local into a global temporary and does not align
vector-typed ones: after the active box's i32 range bounds the f64 gravity vector sat
at offsets 20 and 44 B, which CUDA rejects (CUDA_ERROR_MISALIGNED_ADDRESS; x86 accepts
it). Serial code may still use locals of its own. U4's contact loops follow the same rule.
"""
import numpy as np
import taichi as ti

from . import boundaries, constitutive, contact, transfer
from .constitutive import FLAG_ELASTIC
from .grid import DenseGrid
from .guards import NonFiniteGuard
from .schema import ParticleSchema


@ti.data_oriented
class Solver:
    STAGES = ("clear", "p2g", "grid_update", "g2p", "guard")     # without a contact block
    SUBSTEP_KERNEL = "_substep"

    def __init__(self, cfg: dict, positions: np.ndarray, mat_id=None):
        self.cfg = cfg
        self.dtype = ti.f64 if cfg["precision"] == "f64" else ti.f32
        self.np_dtype = np.float64 if cfg["precision"] == "f64" else np.float32
        gc = cfg["grid"]
        self.grid = DenseGrid(gc["n"], gc["domain"], gc["buffer_cells"], self.dtype, gc["active_box"])
        self.model = constitutive.build(cfg["constitutive"], self.dtype)
        self.transfer = transfer.build(cfg["transfer"], self.grid)
        self.bset = boundaries.build(cfg["boundaries"], self.grid)
        self.schema = ParticleSchema(self.dtype)
        self.schema.add(self.model.name, self.model.state_members())

        positions = np.asarray(positions, np.float64)
        lo, hi = self.grid.x_min, self.grid.x_max
        if not np.isfinite(positions).all() or positions.min() < lo or positions.max() > hi:
            raise ValueError(f"initial positions must be finite and inside [{lo:.4g}, {hi:.4g}] m (grid clamp range)")
        self.n = n = int(positions.shape[0])
        self.contact = contact.build(cfg.get("contact") or {}, self.grid, self.transfer, self.dtype, n,
                                     cfg["particles"]["ppc"])
        self.has_contact = self.contact is not None          # structural (ti.static)
        if self.has_contact:
            self.schema.add(self.contact.label, self.contact.state_members())
        self._period_pos = 0                 # substeps done in the current exchange period
        self.P = self.schema.build(n)
        self.dt = ti.field(self.dtype, ())
        self.gravity = ti.Vector.field(3, self.dtype, ())
        # in-kernel clock: substeps started, simulated time and sum of the dt*g actually
        # added to grid velocities (the gravity term of the momentum balance)
        self.step = ti.field(ti.i64, ())
        self.time = ti.field(ti.f64, ())
        self.gdt = ti.Vector.field(3, ti.f64, ())
        self.guard = NonFiniteGuard(self.dtype, self.step, 0.5 * self.grid.domain)
        self.set_dt(cfg["time"]["dt"])
        self.set_gravity(cfg["gravity"])
        if self.contact is not None:
            self.contact.bind(self)

        mat_id = np.zeros(n, np.int32) if mat_id is None else np.asarray(mat_id, np.int32)
        if mat_id.min() < 0 or mat_id.max() >= self.model.n_mat:
            raise ValueError("mat_id outside the constitutive parameter table")
        self.ppc = cfg["particles"]["ppc"]
        p_vol = self.grid.dx ** 3 / self.ppc
        rho = np.array([self.model.material_info(i)["rho"] for i in range(self.model.n_mat)])
        self.P.x.from_numpy(positions.astype(self.np_dtype))
        self.P.mat_id.from_numpy(mat_id)
        self.P.V0.from_numpy(np.full(n, p_vol).astype(self.np_dtype))
        self.P.m.from_numpy((p_vol * rho[mat_id]).astype(self.np_dtype))
        self._init_particles()
        self.grid.init_box(self.P.x)

    # ------------------------------------------------------------ run-time parameters
    def set_dt(self, dt: float):
        self.dt[None] = float(dt)
        self.dt_host = float(dt)            # python copy (window clock, no device read)

    def set_gravity(self, g):
        self.gravity[None] = [float(c) for c in g]

    # ------------------------------------------------------------ kernels
    @ti.kernel
    def _init_particles(self):
        for p in range(self.n):
            self.P.v[p] = ti.Vector.zero(self.dtype, 3)
            self.P.C[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.P.F[p] = ti.Matrix.identity(self.dtype, 3)
            self.P.tau[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.P.flag[p] = ti.cast(FLAG_ELASTIC, ti.i8)
            self.model.init_state(self.P, p)
            if ti.static(self.has_contact):
                self.P.contact_face[p] = -1

    @ti.kernel
    def _substep(self):
        self.step[None] += 1
        self.time[None] += ti.cast(self.dt[None], ti.f64)
        self.gdt[None] += ti.cast(self.dt[None] * self.gravity[None], ti.f64)
        self.grid.begin_substep()
        self.bset.pre_substep(self.dt[None])
        self.grid.clear()
        self.transfer.p2g(self.P, self.model, self.dt)
        self.grid.update_nodes(self.dt, self.gravity, self.bset)
        if ti.static(self.has_contact):
            self.contact.apply()
        self.grid.reset_box()
        self.transfer.g2p(self.P, self.model, self.dt)
        self.guard.scan(self.P)

    def substep(self):
        c = self.contact
        if c is None:
            self._substep()
            return
        if self._period_pos == 0:
            c.begin_period()
        self._substep()
        self._period_pos += 1
        if self._period_pos == c.n_sub:
            c.end_period()
            self._period_pos = 0

    def advance(self, n_substeps: int):
        for _ in range(n_substeps):
            self.substep()

    def stages(self) -> tuple:
        """Labels of the substep kernel's top-level parallel loops, in order."""
        if self.contact is None:
            return self.STAGES
        return self.STAGES[:3] + tuple(self.contact.STAGES) + self.STAGES[3:]

    def momentum_sinks(self) -> list:
        """(label, 0-d f64 vec3 impulse field) of every module that changes grid momentum
        besides gravity: the boundaries and the contact block (U4, its realised grid impulse)."""
        sinks = self.bset.sinks()
        if self.contact is not None:
            sinks.append((self.contact.label, self.contact.JG))
        return sinks

    def sink_columns(self) -> dict:
        """label -> its three impulse column names (contact: the contract's contact_JG_x/y/z)."""
        return {label: ([f"contact_JG_{c}" for c in "xyz"] if self.contact is not None and label == self.contact.label
                        else [f"{label}_J{c}" for c in "xyz"]) for label, _ in self.momentum_sinks()}

    def clock(self) -> dict:
        return dict(step=int(self.step[None]), time=float(self.time[None]), gdt=self.gdt[None].to_numpy().tolist())

    # ------------------------------------------------------------ derived quantities
    def wave_speed(self) -> float:
        return max(self.model.material_info(i)["wave_speed"] for i in range(self.model.n_mat))

    def cfl(self, vmax: float = 0.0) -> float:
        return float(self.dt[None]) * (self.wave_speed() + vmax) / self.grid.dx

    def derived(self) -> dict:
        dx = self.grid.dx
        masses = self.P.m.to_numpy().astype(np.float64)
        return dict(dx=dx, ppc=self.ppc, p_vol=dx ** 3 / self.ppc, particle_spacing=dx / self.ppc ** (1 / 3),
                    n_particles=self.n, total_mass=float(masses.sum()),
                    p_mass_per_material=[dx ** 3 / self.ppc * self.model.material_info(i)["rho"] for i in range(self.model.n_mat)],
                    wave_speed=self.wave_speed(), cfl0=self.cfl(), cfl_limit=self.cfg["time"]["cfl_limit"])

    def report(self, vmax_hint: float = 0.0) -> list:
        """Print derived quantities; return warnings (parameters are never changed)."""
        d = self.derived()
        lines = [
            f"dx={d['dx']:.4g} m  n={self.grid.n}  ppc={d['ppc']}  s_p={d['particle_spacing']:.4g} m  active_box={self.grid.active_box}",
            f"dt={float(self.dt[None]):.3g} s  c_s={d['wave_speed']:.4g} m/s  CFL(v=0)={self.cfl():.3g}  CFL(v={vmax_hint})={self.cfl(vmax_hint):.3g}",
            f"N={self.n}  total mass={d['total_mass']:.4g} kg  materials={self.model.n_mat}  model={self.model.name}",
            f"transfer={self.transfer.name} tau_cache={self.transfer.tau_cache}  precision={self.cfg['precision']}  boundaries={[b.kind for b in self.bset]}",
        ]
        warns = []
        if self.cfl(vmax_hint) > d["cfl_limit"]:
            warns.append(f"CFL {self.cfl(vmax_hint):.3g} exceeds limit {d['cfl_limit']}")
        if self.grid.buffer_cells < 2:
            warns.append("buffer_cells < 2: quadratic stencil may leave the grid")
        for k, b in enumerate(self.bset):
            if b.kind == "plate":
                step = float(np.abs(b.vel[None].to_numpy()).max()) * float(self.dt[None])
                if step >= self.grid.dx:
                    warns.append(f"boundaries[{k}] plate moves {step:.3g} m per substep >= dx; it can skip a node layer")
        if self.contact is not None:
            c = self.contact
            pr = c.params()
            S = c.surface
            nva, nta = S.n_active if hasattr(S, "n_active") else (S.nv, S.nt)    # live surface: active / capacity
            cap = f" (capacity {S.V_cap}/{S.T_cap})" if hasattr(S, "V_cap") else ""
            lines.append(f"contact: {nva} vertices {nta} triangles{cap} pinned={S.spec['pinned']} motion={S.spec['motion']}  "
                         f"H={c.n_sub} substeps  correction={c.correction} alpha={pr['alpha']:.3g} "
                         f"(alpha*ppc={pr['alpha'] * d['ppc']:.3g})  r0={pr['r0']:.3g} d_hat={pr['d_hat']:.3g} "
                         f"beta={pr['beta']:.4g} mu={pr['mu']:.3g} kappa={pr['kappa']:.3g}")
            if c.correction == "A" and pr["alpha"] * d["ppc"] >= 2.0:
                warns.append(f"contact: alpha*ppc = {pr['alpha'] * d['ppc']:.3g} >= 2; the single-step correction A "
                             "can overshoot ( estimate, not a measured limit)")
        for s in lines:
            print("[config]", s)
        for w in warns:
            print("[config][WARN]", w)
        return warns

    def describe(self) -> dict:
        return dict(grid=self.grid.describe(), transfer=self.transfer.describe(),
                    constitutive=self.model.describe(), boundaries=self.bset.describe(),
                    contact=self.contact.describe() if self.contact is not None else None,
                    stages=list(self.stages()),
                    guard=self.guard.describe(), momentum_sinks=[k for k, _ in self.momentum_sinks()],
                    particle_schema=self.schema.describe(), derived=self.derived())
