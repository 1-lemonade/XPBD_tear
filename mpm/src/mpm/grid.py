"""Dense background grid () with all node traversal in one place.

Optional active box (structural switch): grid loops visit only the nodes that the
particle stencils can touch. G2P accumulates [lo_next, hi_next) from the clamped new
positions; begin_substep() copies it to [lo, hi), which the clear and node update use.
After a substep [lo, hi) is exactly the range that substep's P2G wrote (U2 balances,
U4 contact traversal read only there). Node data outside [lo, hi) is undefined (zero
or stale) and must not be read. The box starts as the whole grid.
Sparse storage would change this file (storage and traversal), not the transfer code.
"""
import taichi as ti

BIG = 1 << 30


@ti.data_oriented
class DenseGrid:
    def __init__(self, n: int, domain: float, buffer_cells: int, dtype, active_box: bool = True):
        self.n = n
        self.domain = domain
        self.dx = domain / n                 # python floats: folded into kernels as constants
        self.inv_dx = 1.0 / self.dx
        self.buffer_cells = buffer_cells
        self.dtype = dtype
        self.active_box = bool(active_box)
        self.m = ti.field(dtype, (n, n, n))
        self.v = ti.Vector.field(3, dtype, (n, n, n))    # momentum after P2G, velocity after update
        self.lo = ti.Vector.field(3, ti.i32, ())         # active box of this substep, half-open [lo, hi)
        self.hi = ti.Vector.field(3, ti.i32, ())
        self.lo_next = ti.Vector.field(3, ti.i32, ())    # accumulated in G2P for the next substep
        self.hi_next = ti.Vector.field(3, ti.i32, ())
        self.clamp_count = ti.field(ti.i32, ())          # particle position clamps (never silent)
        # particle safety clamp keeps every quadratic stencil inside the grid
        self.x_min = 1.0 * self.dx
        self.x_max = (n - 2.0) * self.dx
        for f, val in ((self.lo, 0), (self.lo_next, 0), (self.hi, n), (self.hi_next, n)):
            f[None] = [val] * 3

    def describe(self) -> dict:
        return dict(kind="dense", n=self.n, domain=self.domain, dx=self.dx,
                    buffer_cells=self.buffer_cells, active_box=self.active_box)

    def box(self):
        """[lo, hi) written by the last substep's P2G (whole grid when the box is off)."""
        return self.lo[None].to_numpy().tolist(), self.hi[None].to_numpy().tolist()

    # ------------------------------------------------------------ traversal
    @ti.func
    def _clear_node(self, I):
        self.m[I] = 0.0
        self.v[I] = ti.Vector.zero(self.dtype, 3)

    @ti.func
    def clear(self):
        if ti.static(self.active_box):
            for I in ti.grouped(ti.ndrange((self.lo[None][0], self.hi[None][0]),
                                           (self.lo[None][1], self.hi[None][1]),
                                           (self.lo[None][2], self.hi[None][2]))):
                self._clear_node(I)
        else:
            for I in ti.grouped(self.m):
                self._clear_node(I)

    @ti.func
    def _update_node(self, I, dt: ti.template(), g: ti.template(), bset: ti.template()):
        m = self.m[I]
        if m > 0:
            h = dt[None]
            v = self.v[I] / m + h * g[None]
            self.v[I] = bset.apply(I, v, m, h)

    @ti.func
    def update_nodes(self, dt: ti.template(), g: ti.template(), bset: ti.template()):
        """Momentum -> velocity, gravity, then the boundary list in order. dt and g are the
        solver's 0-d fields, read per node (D21, see solver.py)."""
        if ti.static(self.active_box):
            for I in ti.grouped(ti.ndrange((self.lo[None][0], self.hi[None][0]),
                                           (self.lo[None][1], self.hi[None][1]),
                                           (self.lo[None][2], self.hi[None][2]))):
                self._update_node(I, dt, g, bset)
        else:
            for I in ti.grouped(self.m):
                self._update_node(I, dt, g, bset)

    @ti.func
    def for_each_node(self, op: ti.template()):
        """Top-level loop over the active box (whole grid when off) calling op.node(I);
        used by modules that post-process node velocities (the contact block, U4)."""
        if ti.static(self.active_box):
            for I in ti.grouped(ti.ndrange((self.lo[None][0], self.hi[None][0]),
                                           (self.lo[None][1], self.hi[None][1]),
                                           (self.lo[None][2], self.hi[None][2]))):
                op.node(I)
        else:
            for I in ti.grouped(self.m):
                op.node(I)

    # ------------------------------------------------------------ particle side
    @ti.func
    def clamp(self, x):
        x_c = ti.min(ti.max(x, self.x_min), self.x_max)
        if (x_c - x).norm() > 0:
            ti.atomic_add(self.clamp_count[None], 1)
        return x_c

    @ti.func
    def begin_substep(self):
        """Serial: the box accumulated by the previous G2P becomes this substep's box,
        clamped to the grid. Clamped particles always give a box inside [0, n]; the clamp
        only matters if a non-finite position reached note_particle (cast of NaN to int),
        where an unclamped box would make clear/update loop far outside the grid."""
        if ti.static(self.active_box):
            for d in ti.static(range(3)):
                self.lo[None][d] = ti.min(ti.max(self.lo_next[None][d], 0), self.n)
                self.hi[None][d] = ti.min(ti.max(self.hi_next[None][d], 0), self.n)

    @ti.func
    def reset_box(self):
        """Serial, before G2P: start accumulating the next box."""
        if ti.static(self.active_box):
            self.lo_next[None] = [BIG, BIG, BIG]
            self.hi_next[None] = [-BIG, -BIG, -BIG]

    @ti.func
    def note_particle(self, x):
        """Grow the active box to cover the stencil of a particle at x.
        Check-then-atomic: lo only decreases and hi only increases, so a stale read
        costs at most one redundant atomic and never skips a needed update."""
        if ti.static(self.active_box):
            base = (x * self.inv_dx - 0.5).cast(int)
            for d in ti.static(range(3)):
                if base[d] < self.lo_next[None][d]:
                    ti.atomic_min(self.lo_next[None][d], base[d])
                if base[d] + 3 > self.hi_next[None][d]:
                    ti.atomic_max(self.hi_next[None][d], base[d] + 3)

    @ti.kernel
    def init_box(self, x: ti.template()):
        """Box from current positions; call after positions are set from outside a substep."""
        self.reset_box()
        for p in range(x.shape[0]):
            self.note_particle(x[p])
        self.begin_substep()
