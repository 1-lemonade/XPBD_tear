"""Composable grid boundaries (). The configuration is an ordered list; the
boundaries are applied in that order to every node velocity after gravity.

Interface of a boundary module:
    KEYS                              accepted config keys
    set_params(**kw)                  run-time parameters (0-d fields), no recompilation
    readout() -> dict                 at least 'impulse' (see below); a plate adds 'pos'
    describe() -> dict                full parameters for the run record
    impulse                           0-d f64 vec3 field, see below
    pre_substep(dt)        [ti.func]  serial, once per substep before P2G
    apply(I, v, m, dt)     [ti.func]  -> constrained node velocity

Every boundary is a momentum sink (U2 balance): impulse accumulates, from construction,
sum m_g (v_before - v_after) over the nodes it changes, i.e. the impulse the material
delivers to the boundary [N s] (resting on a floor: about -M g t along y). It is
evaluated in f64 from the working-precision values, so the record is much finer than
the transfer round-off. Friction type (slip | sticky | coulomb) is structural; the
Coulomb mu is run-time.
"""
import taichi as ti

from .config import check_keys

AXES = {"x": 0, "y": 1, "z": 2}
FRICTIONS = ("slip", "sticky", "coulomb")


@ti.data_oriented
class _Plane:
    """Shared normal-direction constraint and impulse record for planar boundaries
    (needs self.a, self.friction, self.dtype and self.impulse)."""

    def reset_impulse(self):
        self.impulse[None] = [0.0, 0.0, 0.0]

    @ti.func
    def _record(self, m, v_before, v_after):
        dJ = ti.cast(m, ti.f64) * (ti.cast(v_before, ti.f64) - ti.cast(v_after, ti.f64))
        for d in ti.static(range(3)):
            ti.atomic_add(self.impulse[None][d], dJ[d])

    @ti.func
    def _constrain(self, v, mu):
        """v moves into the solid along axis a: remove the normal part per friction type.
        The Coulomb branch is the 2026-10-30 ground formula, generalised to any axis."""
        a = ti.static(self.a)
        if ti.static(self.friction == "sticky"):
            v = ti.Vector.zero(self.dtype, 3)
        elif ti.static(self.friction == "slip"):
            v[a] = 0.0
        else:
            vn = ti.abs(v[a])
            vt = v
            vt[a] = 0.0
            vt_norm = vt.norm()
            v = ti.Vector.zero(self.dtype, 3)
            if vt_norm > 1e-20:
                v = vt * ti.max(0.0, 1.0 - mu * vn / vt_norm)
        return v


def _common(spec, where, keys):
    check_keys(where, spec, keys)
    axis, side, friction = spec.get("axis"), spec.get("side"), spec.get("friction", "slip")
    if axis not in AXES or side not in ("lo", "hi") or friction not in FRICTIONS:
        raise ValueError(f"{where}: need axis in {list(AXES)}, side in [lo, hi], friction in {list(FRICTIONS)}; got {spec}")
    return AXES[axis], side, friction


@ti.data_oriented
class Wall(_Plane):
    """Static domain face. Constrained nodes: lo side I[a] < off, hi side I[a] > n - off
    (the 2026-10-30 criterion, i.e. nodes strictly outside the planes off*dx and
    domain - off*dx), only while the node moves outward. Note the inherited asymmetry:
    with off = 3 the lo side holds nodes 0..2 (3 layers), the hi side n-2..n-1 (2 layers)."""
    kind = "wall"
    KEYS = ("type", "axis", "side", "friction", "mu", "offset_cells")

    def __init__(self, spec, grid, where="boundary"):
        self.a, self.side, self.friction = _common(spec, where, self.KEYS)
        self.off = int(spec.get("offset_cells", grid.buffer_cells))
        self.dtype = grid.dtype
        self.n = grid.n
        self.spec = dict(spec, offset_cells=self.off, friction=self.friction)
        self.mu = ti.field(grid.dtype, ())
        self.impulse = ti.Vector.field(3, ti.f64, ())
        self.set_params(mu=spec.get("mu", 0.0))

    def set_params(self, **kw):
        check_keys("wall.set_params", kw, ("mu",))
        if "mu" in kw:
            self.mu[None] = float(kw["mu"])
            self.spec["mu"] = float(kw["mu"])

    def readout(self) -> dict:
        return dict(impulse=self.impulse[None].to_numpy().tolist())

    def describe(self) -> dict:
        return dict(self.spec)

    @ti.func
    def pre_substep(self, dt):
        pass

    @ti.func
    def apply(self, I, v, m, dt):
        a = ti.static(self.a)
        hit = False
        if ti.static(self.side == "lo"):
            hit = I[a] < self.off and v[a] < 0
        else:
            hit = I[a] > self.n - self.off and v[a] > 0
        if hit:
            v_new = self._constrain(v, self.mu[None])
            self._record(m, v, v_new)
            v = v_new
        return v


@ti.data_oriented
class MovingPlate(_Plane):
    """Half-space plate with prescribed velocity (interface for loading plates, ).

    side = lo: the solid occupies coordinate < pos along axis (a floor for axis y);
    side = hi: the solid occupies coordinate > pos. Nodes strictly inside the solid
    whose velocity relative to the plate points into it are constrained (friction
    applied to the relative velocity). pos += dt * vel[axis] every substep.
    impulse: see the module docstring; the mean force over a read-out interval is
    dJ / dt_interval.
    With pos = (off - 0.5) dx and vel = 0 the constrained node set equals Wall(off).
    """
    kind = "plate"
    KEYS = ("type", "axis", "side", "friction", "mu", "pos", "vel")

    def __init__(self, spec, grid, where="boundary"):
        self.a, self.side, self.friction = _common(spec, where, self.KEYS)
        if "pos" not in spec:
            raise KeyError(f"{where}: plate needs 'pos' [m]")
        self.dx = grid.dx
        self.spec = dict(spec, friction=self.friction)
        self.dtype = dtype = grid.dtype
        self.mu = ti.field(dtype, ())
        self.pos = ti.field(ti.f64, ())          # f64: accumulated every substep
        self.vel = ti.Vector.field(3, dtype, ())
        self.impulse = ti.Vector.field(3, ti.f64, ())
        self.set_params(mu=spec.get("mu", 0.0), pos=spec["pos"], vel=spec.get("vel", [0.0, 0.0, 0.0]))

    def set_params(self, **kw):
        check_keys("plate.set_params", kw, ("mu", "pos", "vel"))
        if "mu" in kw:
            self.mu[None] = float(kw["mu"])
        if "pos" in kw:
            self.pos[None] = float(kw["pos"])
        if "vel" in kw:
            self.vel[None] = [float(c) for c in kw["vel"]]

    def readout(self) -> dict:
        return dict(pos=float(self.pos[None]), impulse=self.impulse[None].to_numpy().tolist())

    def describe(self) -> dict:
        return dict(self.spec, pos_now=float(self.pos[None]), vel_now=self.vel[None].to_numpy().tolist(),
                    mu_now=float(self.mu[None]))

    @ti.func
    def pre_substep(self, dt):
        self.pos[None] += dt * self.vel[None][self.a]

    @ti.func
    def apply(self, I, v, m, dt):
        a = ti.static(self.a)
        xg = ti.cast(I[a] * self.dx, ti.f64)
        vp = self.vel[None]
        vr = v - vp
        inside = False
        if ti.static(self.side == "lo"):
            inside = xg < self.pos[None] and vr[a] < 0
        else:
            inside = xg > self.pos[None] and vr[a] > 0
        if inside:
            v_new = self._constrain(vr, self.mu[None]) + vp
            self._record(m, v, v_new)
            v = v_new
        return v


KINDS = {Wall.kind: Wall, MovingPlate.kind: MovingPlate}


@ti.data_oriented
class BoundarySet:
    """Ordered list of boundary modules, applied in sequence (empty list = no boundary)."""

    def __init__(self, items):
        self.items = list(items)

    def __getitem__(self, i):
        return self.items[i]

    def __len__(self):
        return len(self.items)

    def readout(self) -> list:
        return [dict(kind=b.kind, **b.readout()) for b in self.items]

    def sinks(self) -> list:
        """(label, impulse field) per boundary; label = kind + list index, e.g. wall1, plate0."""
        return [(f"{b.kind}{k}", b.impulse) for k, b in enumerate(self.items)]

    def describe(self) -> list:
        return [b.describe() for b in self.items]

    @ti.func
    def pre_substep(self, dt):
        for b in ti.static(self.items):
            b.pre_substep(dt)

    @ti.func
    def apply(self, I, v, m, dt):
        for b in ti.static(self.items):
            v = b.apply(I, v, m, dt)
        return v


def build(specs, grid) -> BoundarySet:
    items = []
    for i, spec in enumerate(specs):
        kind = spec.get("type")
        if kind not in KINDS:
            raise KeyError(f"boundaries[{i}]: unknown type {kind!r}; known {sorted(KINDS)}")
        items.append(KINDS[kind](spec, grid, where=f"boundaries[{i}]"))
    return BoundarySet(items)
