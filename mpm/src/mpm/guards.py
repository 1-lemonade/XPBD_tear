"""Non-finite detection (: the only hard stop).

Taichi compiles with fast_math by default, which lets LLVM assume no NaNs: a check
written as `x == x` is folded to true and never fires (measured, Taichi 1.7.4, f32
and f64). The test here looks at the IEEE bits instead (exponent all ones = inf or
NaN), which no floating-point flag can remove.

NonFiniteGuard.scan is its own particle loop at the end of every substep (after G2P),
over the new x, v, F. Inlined into G2P it changed LLVM's fast-math contraction of the
G2P arithmetic (ulp-level differences from U1, single thread); as a separate loop the
physics code compiles exactly as before. It detects and does not repair: the scene
stops the run at the end of the frame. One safety action only: a particle whose
position is inf/NaN is moved to the grid centre, because the next P2G would turn it
into an out-of-range node index (measured: segmentation fault). Velocity and F stay
as they are, so the state remains visibly bad until the stop. The grid also clamps
the active box to the grid for the same reason (grid.begin_substep).
"""
import taichi as ti

NONE = (1 << 31) - 1


@ti.func
def not_finite(a, dtype: ti.template()):
    """True for +-inf and NaN; a is an f32 or f64 scalar of the given dtype."""
    r = False
    if ti.static(dtype == ti.f64):
        r = (ti.bit_cast(a, ti.i64) >> 52) & 0x7ff == 0x7ff
    else:
        r = (ti.bit_cast(a, ti.i32) >> 23) & 0xff == 0xff
    return r


@ti.data_oriented
class NonFiniteGuard:
    def __init__(self, dtype, step, centre):
        self.dtype = dtype
        self.step = step                              # solver clock: index of the current substep (i64)
        self.centre = centre                          # safe position [m] for a non-finite x
        self.events = ti.field(ti.i32, ())            # (particle, substep) pairs with a non-finite value
        self.first_step = ti.field(ti.i32, ())        # substep in which the first event happened (i32 atomics)
        self.min_particle = ti.field(ti.i32, ())      # lowest particle index ever flagged
        self.moved = ti.field(ti.i32, ())             # non-finite positions moved to the centre
        self.reset()

    def reset(self):
        self.events[None] = 0
        self.first_step[None] = NONE
        self.min_particle[None] = NONE
        self.moved[None] = 0

    def readout(self) -> dict:
        n = int(self.events[None])
        return dict(events=n, first_step=int(self.first_step[None]) if n else -1,
                    min_particle=int(self.min_particle[None]) if n else -1, moved=int(self.moved[None]))

    def describe(self) -> dict:
        return dict(kind="nonfinite_guard", checked=["x", "v", "F"], where="own loop after g2p",
                    action="detect; non-finite x moved to the grid centre (memory safety only)")

    @ti.func
    def scan(self, P: ti.template()):
        for p in range(P.shape[0]):
            x, v, F = P.x[p], P.v[p], P.F[p]
            x_bad = 0
            n_bad = 0
            for d in ti.static(range(3)):
                x_bad += ti.cast(not_finite(x[d], self.dtype), ti.i32)
                n_bad += ti.cast(not_finite(v[d], self.dtype), ti.i32)
                for e in ti.static(range(3)):
                    n_bad += ti.cast(not_finite(F[d, e], self.dtype), ti.i32)
            if x_bad + n_bad > 0:
                ti.atomic_add(self.events[None], 1)
                ti.atomic_min(self.first_step[None], ti.cast(self.step[None], ti.i32))
                ti.atomic_min(self.min_particle[None], p)
            if x_bad > 0:
                P.x[p] = ti.Vector([self.centre] * 3, self.dtype)
                ti.atomic_add(self.moved[None], 1)
