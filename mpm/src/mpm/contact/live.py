"""Live cloth surface (P2-U1): a deformable triangle mesh read every exchange window from a
completed cloth state, held in Taichi fields behind the interface the contact block uses on
StubSurface (block.py: nv, nt, tri, vertex_at, side, plane_side, degenerate, v_s, t_s,
step_index, JP_per, JG_per, pinned, settle_window), so the block runs unchanged. Design:
interface specification

Snapshot. Field names follow XPBD's ClothSnapshot (read by attribute or dict key, no XPBD
import): positions, velocity (N, 3) in the native frame, triangles (T, 3) row indices,
vertex_ids / triangle_ids (stable int64, unique, >= 0, not rows), pinned (N,), epoch,
step_index, topology_version, time, dt. Rows are only meaningful inside one snapshot; the
stable IDs carry identity across snapshots.

ingest(snapshot), between windows only:
  python  checks (shapes, finite values, IDs, version rules, capacity); when the topology
          changes (first snapshot or a new topology_version) the combinatorial checks and
          tables (opposite face of every edge, vertex -> incident (face, corner) CSR), built on
          the host and uploaded. Rows are window-local: under the same topology_version the
          rows may be permuted (same vertex IDs, same triangles in ID space with the same
          triangle IDs and orientation); the tables are then rebuilt for the new rows and the
          per-vertex state follows the IDs . A change of the topology in
          ID space without a new topology_version is rejected.
  kernel  native -> world (X = C + R (x - c), V = R v) with the measures of the step
          (max |X_new - X_old|, max |X_new - frozen-velocity prediction|, atomic max; None
          after a topology change), face normals and
          degenerate flags, edge pseudo-normals and rim flags, angle-weighted vertex
          pseudo-normals and vertex rim flags. The rules are those of surface.side_data (rim =
          an edge with one non-degenerate face, a folded edge or any edge of a degenerate face;
          a rim vertex is an end of a rim edge), computed as gathers in face-row order (the
          measures are max reductions), so the result does not depend on the thread count. Normals come from f64 geometry and are
          stored in the working precision, as in the stub.
settle_window(time, H_ns), called by the block at the window end: the stub's settle kernel
(batch -> last and cumulative, snapshot advanced along its frozen velocity to the window-end
prediction, t_s = solver time, step_index + 1) and window clock, then the batch is cached on
the host in the world frame and in the native frame (G_native = R^T G_world, kernel).

Capacity. Fields hold V_cap + 1 vertices (the last one is a parking vertex at PARK, outside
the domain [0, L]^3 of D10) and T_cap triangles; nv and nt, the contact block's loop bounds,
are these capacities. Inactive triangles point at the parking vertex and are degenerate (the
block skips degenerate faces and bins them in corner bin 0 only); inactive vertices rest at the
parking point with zero velocity and impulse. The capacity cannot grow during a run.

Time. t_s is the MPM substep clock (written by settle, 0 at the start, so the first snapshot
must be read at solver time 0: settle_window checks that every window spans H on that clock,
and H must be a whole number of ns, D30);
the snapshot's time is recorded and handed back with the batch, never used for the in-window
prediction tau = time - dt - t_s ().

Known limits : normals and rims belong to the window-start geometry (frozen
inside the window, exact under rigid translation); no vertex-fan (bow-tie) check here (XPBD's
validate_surface checks the fans); geometric self-intersection is checked by neither side
(validate_surface is combinatorial). Export (P2-U2): mesh_arrays() and cloth_state(t).
"""
import math
from dataclasses import dataclass

import numpy as np
import taichi as ti

from ..config import check_keys
from .xpbd_mesh import RigidMap, proposed_transform

PARK = -1.0                  # parking coordinate on every axis [m]: outside the domain (D10)
FOLD_TOL = 1e-12             # |n_a + n_b| below this: a folded edge, treated as a rim (side_data)
SNAPSHOT_FIELDS = ("positions", "velocity", "triangles", "vertex_ids", "triangle_ids", "pinned",
                   "epoch", "step_index", "topology_version", "time", "dt")
IDENTITY = dict(R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], c=[0.0, 0.0, 0.0], C=[0.0, 0.0, 0.0],
                status="identity", user_signoff="not-applicable")
PROPOSED_STATUS = dict(status="accepted-by-mpm-p1u5", user_signoff="pending")


@dataclass(frozen=True, eq=False)
class SurfaceSnapshot:
    """MPM-side carrier of one completed cloth state, field names as XPBD's ClothSnapshot
    (no checks here: LiveSurface.ingest checks)."""
    positions: np.ndarray
    triangles: np.ndarray
    vertex_ids: np.ndarray
    triangle_ids: np.ndarray
    velocity: np.ndarray
    pinned: np.ndarray
    epoch: int
    step_index: int
    topology_version: int
    time: float
    dt: float

    @property
    def version(self) -> tuple:
        return (self.epoch, self.step_index, self.topology_version)


# ---------------------------------------------------------------- host checks (I/O)
def _get(obj, name):
    if isinstance(obj, dict):
        if name not in obj:
            raise ValueError(f"live surface: snapshot has no {name!r}")
        return obj[name]
    if not hasattr(obj, name):
        raise ValueError(f"live surface: snapshot has no {name!r}")
    return getattr(obj, name)


def _int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0:
        raise ValueError(f"live surface: {name} must be a nonnegative integer, got {v!r}")
    return int(v)


def _real(v, name, positive=False):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"live surface: {name} must be a real scalar, got {v!r}")
    v = float(v)
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        raise ValueError(f"live surface: {name} must be finite and {'> 0' if positive else '>= 0'}, got {v!r}")
    return v


def _ids(a, name, n):
    a = np.asarray(a)
    if a.shape != (n,) or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"live surface: {name} must be an integer array of shape ({n},), got {a.dtype}{a.shape}")
    if a.dtype.kind == "u" and a.size and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"live surface: {name} exceed int64")
    a = a.astype(np.int64)
    if (a < 0).any():
        raise ValueError(f"live surface: {name} must be nonnegative")
    if len(np.unique(a)) != n:
        raise ValueError(f"live surface: {name} must be unique")
    return a


def read_snapshot(obj) -> dict:
    """Checked host copy of a snapshot (native frame): positions / velocity f64 (N, 3),
    triangles int64 (T, 3), IDs int64, pinned bool, version (epoch, step_index,
    topology_version), time, dt. Raises ValueError on the first violation."""
    X = np.asarray(_get(obj, "positions"))
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0 or X.dtype.kind not in "fiu":
        raise ValueError(f"live surface: positions must be a nonempty real (N, 3) array, got {X.dtype}{X.shape}")
    N = X.shape[0]
    V = np.asarray(_get(obj, "velocity"))
    if V.shape != (N, 3) or V.dtype.kind not in "fiu":
        raise ValueError(f"live surface: velocity must be a real ({N}, 3) array, got {V.dtype}{V.shape}")
    X, V = X.astype(np.float64), V.astype(np.float64)
    for name, a in (("positions", X), ("velocity", V)):
        if not np.isfinite(a).all():
            raise ValueError(f"live surface: {name} has non-finite values (row {int(np.argwhere(~np.isfinite(a))[0][0])})")
    T = np.asarray(_get(obj, "triangles"))
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] == 0 or T.dtype.kind not in "iu":
        raise ValueError(f"live surface: triangles must be a nonempty integer (T, 3) array, got {T.dtype}{T.shape}")
    if (T < 0).any() or (T >= N).any():
        raise ValueError(f"live surface: triangle row indices must lie in [0, {N})")
    T = T.astype(np.int64)
    vid = _ids(_get(obj, "vertex_ids"), "vertex_ids", N)
    tid = _ids(_get(obj, "triangle_ids"), "triangle_ids", len(T))
    pin = np.asarray(_get(obj, "pinned"))
    if pin.shape != (N,) or not (pin.dtype.kind == "b" or (pin.dtype.kind in "iu" and np.isin(pin, (0, 1)).all())):
        raise ValueError(f"live surface: pinned must be a boolean ({N},) array")
    version = tuple(_int(_get(obj, k), k) for k in ("epoch", "step_index", "topology_version"))
    return dict(positions=X, velocity=V, triangles=T, vertex_ids=vid, triangle_ids=tid, pinned=pin.astype(bool),
                version=version, time=_real(_get(obj, "time"), "time"), dt=_real(_get(obj, "dt"), "dt", positive=True))


def topology_tables(T, n_vertices) -> dict:
    """Combinatorial checks and tables of one topology (host, once per topology_version).
    Rejects a triangle that repeats a vertex, two triangles on the same three vertices, and
    a directed edge (a, b) used twice: that is exactly 'an edge with more than two faces or
    two faces traversing it the same way' (non-manifold or inconsistent winding; the stub's
    check_orientation plus side_data's non-manifold check). Returns opp (T, 3): the row of
    the face on the other side of edge k = (tri[k], tri[(k + 1) % 3]), -1 on a boundary, and
    the CSR vf_start (N + 1), vf_items (3 T) of the corners 3 t + k at each vertex in
    ascending face row (the summation order of side_data)."""
    T = np.asarray(T, np.int64)
    n = int(n_vertices)
    rep = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2])
    if rep.any():
        raise ValueError(f"live surface: triangle row {int(np.argmax(rep))} repeats a vertex")
    s = np.sort(T, axis=1)
    _, first, cnt = np.unique(s, axis=0, return_index=True, return_counts=True)
    if (cnt > 1).any():
        raise ValueError(f"live surface: two triangles on the same vertices {s[first[np.argmax(cnt > 1)]].tolist()}")
    a = T.ravel()
    b = T[:, [1, 2, 0]].ravel()
    key = a * n + b
    order = np.argsort(key, kind="stable")
    sk = key[order]
    dup = np.flatnonzero(sk[1:] == sk[:-1])
    if dup.size:
        j1, j2 = int(order[dup[0]]), int(order[dup[0] + 1])
        raise ValueError(f"live surface: faces {j1 // 3} and {j2 // 3} traverse edge ({int(a[j1])}, {int(b[j1])}) "
                         "the same way (non-manifold edge or inconsistent winding)")
    rkey = b * n + a
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    opp = np.where(sk[pos] == rkey, order[pos] // 3, -1).reshape(-1, 3).astype(np.int32)
    vf_items = np.argsort(a, kind="stable").astype(np.int32)        # corner j = 3 t + k, ascending t per vertex
    vf_start = np.concatenate(([0], np.cumsum(np.bincount(a, minlength=n)))).astype(np.int32)
    return dict(opp=opp, vf_start=vf_start, vf_items=vf_items)


def id_topology(T, vertex_ids, triangle_ids):
    """Row-independent form of a topology: sorted vertex IDs, sorted triangle IDs and, in that
    order, each triangle's vertex IDs rotated to start at the smallest (orientation kept)."""
    F = np.asarray(vertex_ids)[np.asarray(T)]
    r = np.argmin(F, axis=1)
    F = np.take_along_axis(F, (r[:, None] + np.arange(3)) % 3, axis=1)
    o = np.argsort(triangle_ids, kind="stable")
    return np.sort(vertex_ids), np.asarray(triangle_ids)[o], F[o]


def resolve_transform(spec_tf) -> dict:
    """None or a partial dict -> the XPBD proposal (as type mesh) with its status; 'identity'."""
    if spec_tf == "identity":
        return dict(IDENTITY)
    if spec_tf is None or isinstance(spec_tf, dict):
        check_keys("contact.surface.transform", spec_tf or {}, ("R", "c", "C", "status", "user_signoff"))
        # later keys win (a config may state its own status / user_signoff; P2-U4: dict(..., **a, **b) raised
        # TypeError on a repeated keyword)
        return {**proposed_transform(), **PROPOSED_STATUS, **(spec_tf or {})}
    raise ValueError("contact.surface.transform must be null, 'identity' or {R, c, C[, status, user_signoff]}")


# ---------------------------------------------------------------- surface
@ti.data_oriented
class LiveSurface:
    KEYS = ("type", "capacity", "transform", "area_rtol")

    def __init__(self, spec, dtype, where="contact.surface"):
        check_keys(where, spec, self.KEYS)
        cap = spec.get("capacity")
        if not isinstance(cap, dict):
            raise KeyError(f"{where}.capacity: give {{vertices, triangles}} (no default: the surface is empty "
                           "until its first snapshot)")
        check_keys(f"{where}.capacity", cap, ("vertices", "triangles"))
        for k, lo in (("vertices", 3), ("triangles", 1)):
            v = cap.get(k)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < lo:
                raise ValueError(f"{where}.capacity.{k} must be an integer >= {lo}, got {v!r}")
        Vc, Tc = int(cap["vertices"]), int(cap["triangles"])
        tf = resolve_transform(spec.get("transform"))
        for k in ("c", "C"):
            a = np.asarray(tf[k], np.float64)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"{where}.transform.{k} must be 3 finite numbers, got {tf[k]!r}")
        self.rigid = RigidMap(tf["R"], tf["c"], tf["C"])
        self.transform_status = dict(status=tf["status"], user_signoff=tf["user_signoff"])
        area_rtol = _real(spec["area_rtol"], f"{where}.area_rtol") if spec.get("area_rtol") is not None else 1e-12
        self.dtype = dtype
        self.V_cap, self.T_cap = Vc, Tc
        self.pv = Vc                                        # parking vertex row
        self.nv, self.nt = Vc + 1, Tc                        # the contact block's loop bounds
        self.spec = dict(spec, type="live", capacity=dict(vertices=Vc, triangles=Tc),
                         transform=dict(self.rigid.describe(), **self.transform_status), area_rtol=area_rtol,
                         pinned="snapshot", motion=dict(type="live"))
        self.source = None                                  # the stub's mesh source record: none here
        V, T = self.nv, self.nt
        f64v = lambda n: ti.Vector.field(3, ti.f64, n)
        # geometry: native staging (host upload), world (f64 positions as in the stub)
        self.xn, self.vn = f64v(V), f64v(V)
        self.x_s, self.x_prev = f64v(V), f64v(V)
        self.v_s = ti.Vector.field(3, dtype, V)
        # topology (uploaded once per topology_version)
        self.tri = ti.Vector.field(3, ti.i32, T)
        self.opp = ti.Vector.field(3, ti.i32, T)
        self.vf_start = ti.field(ti.i32, V + 1)
        self.vf_items = ti.field(ti.i32, 3 * T)
        # side information (recomputed every window)
        self.fn64 = f64v(T)
        self.face_n = ti.Vector.field(3, dtype, T)
        self.edge_n = ti.Matrix.field(3, 3, dtype, T)      # row k: pseudo-normal of edge k
        self.edge_rim = ti.Vector.field(3, ti.i32, T)
        self.vert_n = ti.Vector.field(3, dtype, V)
        self.vert_rim = ti.field(ti.i32, V)
        self.degenerate = ti.field(ti.i32, T)
        self.pinned = ti.field(ti.i32, V)
        # run-time data
        self.n_vert = ti.field(ti.i32, ())
        self.n_tri = ti.field(ti.i32, ())
        self.measure = ti.field(ti.i32, ())                  # 1: same rows as the last snapshot
        self.metric = ti.field(ti.f64, 2)                    # max |X_new - X_old|, max |X_new - prediction|
        self.Rm = ti.Matrix.field(3, 3, ti.f64, ())
        self.c_n = ti.Vector.field(3, ti.f64, ())
        self.C_w = ti.Vector.field(3, ti.f64, ())
        self.area_rtol = ti.field(ti.f64, ())
        # clock, version and impulses [N s], as StubSurface
        self.t_s = ti.field(ti.f64, ())
        self.step_index = ti.field(ti.i64, ())
        self.JP_per, self.JG_per = f64v(V), f64v(V)
        self.JP_last, self.JG_last = f64v(V), f64v(V)
        self.JP_cum, self.JG_cum = f64v(V), f64v(V)
        self.JP_nat, self.JG_nat = f64v(V), f64v(V)          # last batch in the native frame
        self.last_t = ti.Vector.field(2, ti.f64, ())
        self.last_step = ti.field(ti.i64, ())
        self.t_ns = 0
        self.last_ns = None
        self.H_ns = None
        # host state
        self._phase = "between"                              # "between" windows or inside a "window"
        self._snap = None                                    # version, time, dt of the current snapshot
        self.T = np.zeros((0, 3), np.int64)
        self.vertex_ids = np.zeros(0, np.int64)
        self.triangle_ids = np.zeros(0, np.int64)
        self.pinned_np = np.zeros(0, bool)
        self._id_order = np.zeros(0, np.int64)
        self._batch = None
        self._report = None
        self.dropped_cum = dict(JP=np.zeros(3), JG=np.zeros(3))   # cumulative impulse of IDs that disappeared
        self.rebuilds = 0                                    # topology uploads (new topology or permuted rows)
        self.reorders = 0                                    # of these: permuted rows, same topology

        park = np.full((V, 3), PARK)
        for f in (self.xn, self.x_s, self.x_prev):
            f.from_numpy(park)
        self.tri.from_numpy(np.full((T, 3), self.pv, np.int32))
        self.opp.from_numpy(np.full((T, 3), -1, np.int32))
        self.edge_rim.from_numpy(np.ones((T, 3), np.int32))
        self.degenerate.from_numpy(np.ones(T, np.int32))
        self.Rm.from_numpy(self.rigid.R)
        self.c_n.from_numpy(self.rigid.c)
        self.C_w.from_numpy(self.rigid.C)
        self.area_rtol[None] = area_rtol
        self.last_step[None] = -1

    # ------------------------------------------------------------ python side
    def set_params(self, **kw):
        check_keys("surface.set_params", kw, ("area_rtol",))
        if kw.get("area_rtol") is not None:
            v = _real(kw["area_rtol"], "area_rtol")
            self.area_rtol[None] = v
            self.spec["area_rtol"] = v

    @property
    def n_active(self) -> tuple:
        return len(self.vertex_ids), len(self.T)

    def version(self):
        """Version (epoch, step_index, topology_version) of the current snapshot (None before
        the first one)."""
        return None if self._snap is None else self._snap["version"]

    def ingest(self, snapshot) -> dict:
        """Read one completed state (native frame) before a window; see the module docstring.
        Every check runs before any field changes. Returns the ingest report."""
        if self._phase != "between":
            raise RuntimeError("live surface: ingest inside a window (the window must be settled first)")
        s = read_snapshot(snapshot)
        N, Tn = len(s["positions"]), len(s["triangles"])
        prev = self._snap
        if prev is not None:
            (ep, st, tv), (pep, pst, ptv) = s["version"], prev["version"]
            if ep != pep:
                raise ValueError(f"live surface: epoch {ep} after {pep} (a new epoch is a new stream and run)")
            if st != pst + 1:
                raise ValueError(f"live surface: step_index {st} after {pst} (must be +1)")
            if tv < ptv:
                raise ValueError(f"live surface: topology_version {tv} after {ptv} (must not decrease)")
            if not s["time"] > prev["time"]:
                raise ValueError(f"live surface: time {s['time']!r} after {prev['time']!r} (must increase)")
        rebuild = prev is None or s["version"][2] != prev["version"][2]
        if N > self.V_cap or Tn > self.T_cap:
            raise ValueError(f"live surface: {N} vertices / {Tn} triangles exceed the capacity "
                             f"{self.V_cap} / {self.T_cap} (it cannot grow during a run)")
        reorder = False
        if not rebuild and not (np.array_equal(s["triangles"], self.T) and np.array_equal(s["vertex_ids"], self.vertex_ids)
                                and np.array_equal(s["triangle_ids"], self.triangle_ids)):
            new, cur = (id_topology(s["triangles"], s["vertex_ids"], s["triangle_ids"]),
                        id_topology(self.T, self.vertex_ids, self.triangle_ids))
            if not all(a.shape == b.shape and np.array_equal(a, b) for a, b in zip(new, cur)):
                raise ValueError("live surface: the topology changed (in stable IDs) without a topology_version change")
            reorder = True                                   # same topology, rows permuted
        tables = topology_tables(s["triangles"], N) if (rebuild or reorder) else None
        # ---- checks passed: change state
        if rebuild or reorder:
            self._upload_topology(s, tables, carry_geometry=reorder)
        V = self.nv
        xn = np.full((V, 3), PARK)
        vn = np.zeros((V, 3))
        pin = np.zeros(V, np.int32)
        xn[:N], vn[:N], pin[:N] = s["positions"], s["velocity"], s["pinned"]
        self.xn.from_numpy(xn)
        self.vn.from_numpy(vn)
        self.pinned.from_numpy(pin)
        self.pinned_np = s["pinned"].copy()
        self.measure[None] = 0 if rebuild else 1          # a reorder carried the previous geometry by ID
        self.metric.fill(0.0)
        self._geometry()
        m = self.metric.to_numpy()
        self._snap = dict(version=s["version"], time=s["time"], dt=s["dt"])
        self._report = dict(version=s["version"], time=s["time"], dt=s["dt"], rebuilt=bool(rebuild),
                            reordered=bool(reorder), n_vertices=N, n_triangles=Tn,
                            max_step=None if rebuild else float(m[0]),
                            max_prediction_deviation=None if rebuild else float(m[1]))
        self._phase = "window"
        return dict(self._report)

    def _upload_topology(self, s, tables, carry_geometry=False):
        """Topology tables of the new rows; the cumulative per-vertex impulses (and, for a pure row
        permutation, the previous and predicted positions used by the measures) follow the IDs."""
        N, Tn = len(s["positions"]), len(s["triangles"])
        V, T = self.nv, self.nt
        tri = np.full((T, 3), self.pv, np.int32)
        opp = np.full((T, 3), -1, np.int32)
        tri[:Tn], opp[:Tn] = s["triangles"], tables["opp"]
        vfs = np.full(V + 1, tables["vf_start"][-1], np.int32)
        vfs[:N + 1] = tables["vf_start"]
        vfi = np.zeros(3 * T, np.int32)
        vfi[:3 * Tn] = tables["vf_items"]
        if self._snap is not None:                         # cumulative per-vertex totals follow the stable IDs
            rows, hit = self._match(s["vertex_ids"], self.vertex_ids)
            n_old = len(self.vertex_ids)
            for name, f in (("JP", self.JP_cum), ("JG", self.JG_cum)):
                old = f.to_numpy()[:n_old]
                new = np.zeros((V, 3))
                new[rows[hit]] = old[hit]
                self.dropped_cum[name] = self.dropped_cum[name] + old[~hit].sum(0)
                f.from_numpy(new)
            if carry_geometry:
                for f in (self.x_s, self.x_prev):
                    old = f.to_numpy()[:n_old]
                    new = np.full((V, 3), PARK)
                    new[rows[hit]] = old[hit]
                    f.from_numpy(new)
        self.tri.from_numpy(tri)
        self.opp.from_numpy(opp)
        self.vf_start.from_numpy(vfs)
        self.vf_items.from_numpy(vfi)
        self.n_vert[None], self.n_tri[None] = N, Tn
        self.T = s["triangles"].copy()
        self.vertex_ids, self.triangle_ids = s["vertex_ids"].copy(), s["triangle_ids"].copy()
        self._id_order = np.argsort(self.vertex_ids, kind="stable")
        self.rebuilds += 1
        self.reorders += int(carry_geometry)

    @staticmethod
    def _match(new_ids, old_ids):
        """Row in new_ids of every old ID (rows) and whether it is there (hit)."""
        order = np.argsort(new_ids, kind="stable")
        srt = new_ids[order]
        pos = np.minimum(np.searchsorted(srt, old_ids), max(len(srt) - 1, 0))
        hit = srt[pos] == old_ids
        return order[pos], hit

    def rows_of(self, ids) -> np.ndarray:
        """Rows of stable vertex IDs in the current snapshot; unknown IDs raise."""
        ids = np.asarray(ids, np.int64)
        srt = self.vertex_ids[self._id_order]
        pos = np.minimum(np.searchsorted(srt, ids), max(len(srt) - 1, 0))
        if len(srt) == 0 or not (srt[pos] == ids).all():
            raise ValueError("live surface: unknown vertex IDs")
        return self._id_order[pos]

    def ingest_report(self):
        return None if self._report is None else dict(self._report)

    def settle_window(self, time, H_ns):
        """End of a window (ContactBlock.end_period): settle as the stub, advance the integer
        window clock, cache the batch (world and native frame) with the IDs and version of the
        snapshot it belongs to."""
        if self._phase != "window":
            raise RuntimeError("live surface: a window ended without a snapshot (ingest before every window)")
        if H_ns is None:                                  # D30: a coupling window is a whole number of ns
            raise RuntimeError("live surface: the exchange window H = exchange_substeps * dt must be a whole "
                               "number of nanoseconds")
        span, H = float(time[None]) - float(self.t_s[None]), int(H_ns) / 1e9
        if abs(span - H) > 1e-6 * H:                       # t_s must be the solver time at the window start
            raise RuntimeError(f"live surface: the window spans {span!r} s on the solver clock, not H = {H!r} s "
                               "(the first snapshot must be read at solver time 0)")
        self.settle(time)
        self.last_ns = (self.t_ns, self.t_ns + int(H_ns))
        self.t_ns += int(H_ns)
        self.H_ns = H_ns
        self._to_native()
        N = len(self.vertex_ids)
        ts, te = self.last_t[None].to_numpy().tolist()
        b = dict(window_index=int(self.last_step[None]), cloth_version=self._snap["version"],
                 snapshot_time=self._snap["time"], snapshot_dt=self._snap["dt"], H=int(H_ns) / 1e9, H_ns=int(H_ns),
                 t_solver=(ts, te), t_start=ts, t_end=te, t_start_ns=None, t_end_ns=None,
                 vertex_ids=self.vertex_ids.copy(),
                 impulse_P=self.JP_last.to_numpy()[:N], impulse_G=self.JG_last.to_numpy()[:N],
                 impulse_P_native=self.JP_nat.to_numpy()[:N], impulse_G_native=self.JG_nat.to_numpy()[:N],
                 frame_native="xpbd-native (G_native = R^T G_world)")
        b.update(t_start_ns=self.last_ns[0], t_end_ns=self.last_ns[1],
                 t_start=self.last_ns[0] / 1e9, t_end=self.last_ns[1] / 1e9)
        self._batch = b
        self._phase = "between"

    def last_batch(self) -> dict:
        """The last settled batch (copy); None before the first window ends. t_start / t_end
        (and *_ns, t_solver) are the MPM window clock, as in StubSurface; the snapshot it belongs
        to is cloth_version at snapshot_time. A LoadBatch (P2-U2) takes t_start = snapshot_time and
        t_end = snapshot_time + h (XPBD's float arithmetic), not the MPM clock (); snapshot_dt is XPBD's previous step, not h. H is the MPM exchange window."""
        if self._batch is None:
            return None
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._batch.items()}

    def positions_at(self, t: float) -> np.ndarray:
        """Predicted world positions of the active vertices at solver time t."""
        N = len(self.vertex_ids)
        return self.x_s.to_numpy()[:N] + (t - float(self.t_s[None])) * self.v_s.to_numpy()[:N].astype(np.float64)

    def side_arrays(self) -> dict:
        """Side information of the active rows (analysis and tests)."""
        N, Tn = self.n_active
        return dict(face_n=self.face_n.to_numpy()[:Tn], edge_n=self.edge_n.to_numpy()[:Tn],
                    edge_rim=self.edge_rim.to_numpy()[:Tn], vert_n=self.vert_n.to_numpy()[:N],
                    vert_rim=self.vert_rim.to_numpy()[:N], degenerate=self.degenerate.to_numpy()[:Tn])

    def mesh_arrays(self) -> dict:
        """cloth/mesh.npz of a run (P2-U2): the topology and positions of the current snapshot
        (the scene writes it after reading the first one). Rows are window-local: every cloth
        frame (cloth_state) carries its own IDs and triangles."""
        if self._snap is None:
            raise RuntimeError("live surface: no snapshot read yet (read the first window before exporting)")
        N, Tn = self.n_active
        return dict(vertex_ids=self.vertex_ids.copy(), triangle_ids=self.triangle_ids.copy(),
                    triangles=self.T.astype(np.int32), x0=self.x_s.to_numpy()[:N],
                    pinned=self.pinned_np.astype(np.int8), degenerate=self.degenerate.to_numpy()[:Tn].astype(np.int8))

    def cloth_state(self, t) -> dict:
        """One cloth frame (export.py): the stub's fields on the active rows, plus the rows'
        stable IDs and triangles of this snapshot."""
        if self._snap is None:
            raise RuntimeError("live surface: no snapshot read yet")
        N, Tn = self.n_active
        cum = lambda c, p: (c.to_numpy() + p.to_numpy())[:N]
        return dict(version=np.asarray(self.version(), np.int64), x=self.positions_at(t),
                    v=self.v_s.to_numpy()[:N].astype(np.float32), JP=cum(self.JP_cum, self.JP_per),
                    JG=cum(self.JG_cum, self.JG_per), vertex_ids=self.vertex_ids.copy(),
                    triangle_ids=self.triangle_ids.copy(), triangles=self.T.astype(np.int32))

    def describe(self) -> dict:
        d = dict(self.spec, n_vertices=len(self.vertex_ids), n_triangles=len(self.T),
                 n_pinned=int(self.pinned_np.sum()), version_now=self.version(), rebuilds=self.rebuilds,
                 reorders=self.reorders, last_ingest=self.ingest_report(), park=PARK)
        return d

    # ------------------------------------------------------------ taichi side (contact block interface)
    @ti.func
    def vertex_at(self, v, tau):
        """Predicted position of vertex v at tau = t - t_s (working precision)."""
        return ti.cast(self.x_s[v], self.dtype) + tau * self.v_s[v]

    @ti.func
    def plane_side(self, t, diff):
        """+1 or -1: side of the triangle's plane (face normal) that diff points to (0 -> +1)."""
        s = 1
        if self.face_n[t].dot(diff) < 0:
            s = -1
        return s

    @ti.func
    def side(self, t, feat, diff):
        """(pseudo-normal of the feature, rim flag) for the closest feature of triangle t."""
        pn = self.face_n[t]
        rim = 0
        if feat >= 1 and feat <= 3:
            k = feat - 1
            pn = ti.Vector([self.edge_n[t][k, 0], self.edge_n[t][k, 1], self.edge_n[t][k, 2]])
            rim = self.edge_rim[t][k]
        elif feat >= 4:
            vtx = self.tri[t][feat - 4]
            pn = self.vert_n[vtx]
            rim = self.vert_rim[vtx]
        return pn, rim

    @ti.kernel
    def settle(self, time: ti.template()):
        """End of an exchange period (StubSurface.settle, same code): batch -> last and
        cumulative, then the snapshot is advanced to the solver time along its frozen velocity
        (the window-end prediction, compared with the next snapshot by the ingest measures)."""
        for v in range(self.nv):
            self.JP_last[v] = self.JP_per[v]
            self.JG_last[v] = self.JG_per[v]
            self.JP_cum[v] += self.JP_per[v]
            self.JG_cum[v] += self.JG_per[v]
            self.JP_per[v] = ti.Vector.zero(ti.f64, 3)
            self.JG_per[v] = ti.Vector.zero(ti.f64, 3)
            self.x_s[v] += (time[None] - self.t_s[None]) * ti.cast(self.v_s[v], ti.f64)
        self.last_t[None] = ti.Vector([self.t_s[None], time[None]])
        self.last_step[None] = self.step_index[None]
        self.t_s[None] = time[None]
        self.step_index[None] += 1

    @ti.kernel
    def _to_native(self):
        """Last batch in the native frame: G_native = R^T G_world (vectors rotate only)."""
        for v in range(self.nv):
            Rt = self.Rm[None].transpose()
            self.JP_nat[v] = Rt @ self.JP_last[v]
            self.JG_nat[v] = Rt @ self.JG_last[v]

    @ti.kernel
    def _geometry(self):
        """World geometry and side information of the uploaded snapshot (four top-level
        loops, each a gather; run-time data read inside the loop bodies, D21)."""
        # vertices: native -> world, measures against the previous snapshot, parking
        for v in range(self.nv):
            if v < self.n_vert[None]:
                X = self.C_w[None] + self.Rm[None] @ (self.xn[v] - self.c_n[None])
                if self.measure[None] == 1:
                    ti.atomic_max(self.metric[0], (X - self.x_prev[v]).norm())
                    ti.atomic_max(self.metric[1], (X - self.x_s[v]).norm())
                self.x_s[v] = X
                self.x_prev[v] = X
                self.v_s[v] = ti.cast(self.Rm[None] @ self.vn[v], self.dtype)
            else:
                P = ti.Vector([PARK, PARK, PARK], ti.f64)
                self.x_s[v] = P
                self.x_prev[v] = P
                self.v_s[v] = ti.Vector.zero(self.dtype, 3)
        # faces: f64 normal and degenerate flag (side_data: area2 <= area_rtol |e1| |e2|)
        for t in range(self.nt):
            fn = ti.Vector.zero(ti.f64, 3)
            deg = 1
            if t < self.n_tri[None]:
                x0 = self.x_s[self.tri[t][0]]
                e1 = self.x_s[self.tri[t][1]] - x0
                e2 = self.x_s[self.tri[t][2]] - x0
                cr = e1.cross(e2)
                area2 = cr.norm()
                scale = ti.max(e1.norm() * e2.norm(), 1e-300)
                if area2 > self.area_rtol[None] * scale:
                    deg = 0
                    fn = cr / area2
            self.fn64[t] = fn
            self.face_n[t] = ti.cast(fn, self.dtype)
            self.degenerate[t] = deg
        # edges: pseudo-normal (sum of the two face normals) or rim (one non-degenerate face,
        # folded, or any edge of a degenerate face) with the face normal
        for t in range(self.nt):
            for k in ti.static(range(3)):
                en = self.fn64[t]
                rim = 1
                if self.degenerate[t] == 0:
                    o = self.opp[t][k]
                    if o >= 0:
                        if self.degenerate[o] == 0:
                            s = self.fn64[t] + self.fn64[o]
                            ns = s.norm()
                            if ns >= FOLD_TOL:
                                en = s / ns
                                rim = 0
                for j in ti.static(range(3)):
                    self.edge_n[t][k, j] = ti.cast(en[j], self.dtype)
                self.edge_rim[t][k] = rim
        # vertices: angle-weighted pseudo-normal over the incident non-degenerate faces in face
        # row order; rim when either edge at the corner is a rim (degenerate faces included)
        for v in range(self.nv):
            acc = ti.Vector.zero(ti.f64, 3)
            rim = 0
            if v < self.n_vert[None]:
                for j in range(self.vf_start[v], self.vf_start[v + 1]):
                    item = self.vf_items[j]
                    t = item // 3
                    k = item - 3 * t
                    for kk in ti.static(range(3)):
                        if k == kk:
                            if self.edge_rim[t][kk] == 1 or self.edge_rim[t][(kk + 2) % 3] == 1:
                                rim = 1
                            if self.degenerate[t] == 0:
                                xi = self.x_s[self.tri[t][kk]]
                                a = self.x_s[self.tri[t][(kk + 1) % 3]] - xi
                                b = self.x_s[self.tri[t][(kk + 2) % 3]] - xi
                                ang = ti.atan2(a.cross(b).norm(), a.dot(b))
                                acc += ang * self.fn64[t]
            nrm = acc.norm()
            if nrm > 0:
                acc = acc / nrm
            self.vert_n[v] = ti.cast(acc, self.dtype)
            self.vert_rim[v] = rim
