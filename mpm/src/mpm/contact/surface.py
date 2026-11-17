"""Stub cloth surface for P1-U4: a triangle mesh with fixed or prescribed rigid motion,
held in Taichi fields . It does not respond to the impulses it
receives; it records them per vertex and settles one batch per exchange period H.

Semantics follow the XPBD side's frozen Phase 2.0 snapshot : rows vs stable int64 vertex/triangle IDs (IDs need not equal rows), winding
gives the face normal, version (epoch, step_index, topology_version). Here epoch is the
reset generation (1 for a run), step_index counts completed stub steps (one per H, equal
to the exchange periods completed), topology_version never changes (no tearing in P1).

Snapshot and prediction (): at the start of a period the surface is at x_s with
velocity v_s at time t_s; inside the period the contact block uses x_s + (t - t_s) v_s.
settle() closes the period: per-vertex batch -> last batch and cumulative totals, then
x_s += (t - t_s) v_s, t_s = t, step_index += 1 (motion types: fixed, translate).

Side information (precomputed, constant under rigid translation): face normals, edge
pseudo-normals (sum of the two adjacent face normals) and angle-weighted vertex
pseudo-normals (Baerentzen and Aanaes 2005), and rim flags for boundary edges and the
vertices on them. Non-manifold edges are rejected.

Surface types: `sheet` (generated rectangle, U4) and `mesh` (U5): a given triangle mesh in
a native frame placed in the world by a rigid map X = C + R (x - c); vertex and triangle
IDs come from the mesh. The one source so far is the XPBD coupling reference mesh
(`source: xpbd-cloth52-v1`, mpm/contact/xpbd_mesh.py), read from its published mesh.npz
(`path`, optional `expect_sha256`) or, without a path, rebuilt from the documented formula
(development only; not bound to a file). `pinned` may then also name a support mask of the
mesh (e.g. pinned_two_rows); the stub stays rigid either way, the mask only decides which
vertices' G share is G_on_pinned.

Window clock (U5): the exchange window k covers [k H, (k + 1) H). When H is an integer
number of nanoseconds the batch endpoints are kept as integers (t_start_ns, t_end_ns, exact
sums of H_ns) and their floats are t_ns / 1e9 (correctly rounded binary64); the solver's
own accumulated clock is recorded beside them (t_solver) and still drives the prediction.
The batch carries window_index = k (windows completed before it).
"""
import math

import numpy as np
import taichi as ti

from ..config import check_keys

AXES = {"x": 0, "y": 1, "z": 2}
PINNED = ("all", "rim", "none")
MOTIONS = ("fixed", "translate")
TYPES = ("sheet", "mesh")
MESH_SOURCES = ("xpbd-cloth52-v1",)


def build_surface(spec, dtype, where="contact.surface"):
    """Surface factory of the contact block (P2-U1): type sheet / mesh -> StubSurface (rigid,
    unchanged), type live -> LiveSurface (a completed cloth state read every window,
    mpm/contact/live.py). Both expose the same interface to the block."""
    kind = spec.get("type") if isinstance(spec, dict) else None
    if kind == "live":
        from .live import LiveSurface
        return LiveSurface(spec, dtype, where=where)
    if kind not in TYPES:
        raise KeyError(f"{where}.type must be one of {list(TYPES) + ['live']}, got {kind!r}")
    return StubSurface(spec, dtype, where=where)


def sheet(spec):
    """Regular rectangular sheet: (vertices (V,3) f64, triangles (T,3) i32), winding such
    that every face normal points along normal_sign * normal_axis. Optional `holes`: list
    of {"center": [u, w], "size": [su, sw]} in the world coordinates of the two in-plane
    axes (u, w in axis order, e.g. x and z for a y-normal sheet); triangles whose centroid
    lies inside a hole are removed and unused vertices dropped, so the hole boundary is a
    rim (a stub for pre-cut slits and cracks)."""
    a = AXES[spec["normal_axis"]]
    u, w = [k for k in range(3) if k != a]
    nu, nw = (int(c) for c in spec["cells"])
    su, sw = (float(s) for s in spec["size"])
    c = np.asarray(spec["center"], np.float64)
    iu, iw = np.meshgrid(np.arange(nu + 1), np.arange(nw + 1), indexing="ij")
    X = np.zeros(((nu + 1) * (nw + 1), 3))
    X[:, a] = c[a]
    X[:, u] = c[u] - su / 2 + su * iu.ravel() / nu
    X[:, w] = c[w] - sw / 2 + sw * iw.ravel() / nw
    vid = lambda i, j: i * (nw + 1) + j
    tris = []
    for i in range(nu):
        for j in range(nw):
            v00, v10, v01, v11 = vid(i, j), vid(i + 1, j), vid(i, j + 1), vid(i + 1, j + 1)
            tris += [(v00, v10, v11), (v00, v11, v01)]
    T = np.asarray(tris, np.int32)
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    flip = n[:, a] * float(spec.get("normal_sign", 1)) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    holes = spec.get("holes") or []
    if holes:
        cen = X[T].mean(1)
        keep = np.ones(len(T), bool)
        for h in holes:
            (hu, hw), (hsu, hsw) = h["center"], h["size"]
            keep &= ~((np.abs(cen[:, u] - hu) < hsu / 2) & (np.abs(cen[:, w] - hw) < hsw / 2))
        T = T[keep]
        used = np.unique(T)
        remap = -np.ones(len(X), np.int64)
        remap[used] = np.arange(len(used))
        X, T = X[used], remap[T].astype(np.int32)
    return X, T


def check_orientation(T):
    """Consistent winding: every edge shared by two faces is traversed in opposite
    directions by them (an orientable, consistently wound mesh; D25 tracked side)."""
    directed = {}
    for t, tri in enumerate(np.asarray(T)):
        for k in range(3):
            e = (int(tri[k]), int(tri[(k + 1) % 3]))
            if e in directed:
                raise ValueError(f"stub surface: faces {directed[e]} and {t} traverse edge {e} in the same "
                                 "direction (inconsistent winding)")
            directed[e] = t


def mesh_surface(spec):
    """type mesh: (world vertices, triangles i32, vertex IDs, triangle IDs, support masks,
    source record) from the configured source and rigid map. The winding must be
    consistent (check_orientation); non-manifold edges are rejected by side_data."""
    from . import xpbd_mesh
    if spec.get("source") not in MESH_SOURCES:
        raise ValueError(f"contact.surface.source must be one of {list(MESH_SOURCES)}")
    src = (xpbd_mesh.load(spec["path"], spec.get("expect_sha256"), spec.get("expect_manifest_sha256"))
           if spec.get("path") else xpbd_mesh.reconstructed())
    tf = dict(xpbd_mesh.proposed_transform(), **(spec.get("transform") or {}))
    rigid = xpbd_mesh.RigidMap(tf["R"], tf["c"], tf["C"])
    a = src["arrays"]
    X = rigid.to_world(a["positions"])
    T = a["triangles"].astype(np.int32)
    check_orientation(T)
    masks = {k: a[k].astype(np.int32) for k in xpbd_mesh.SUPPORTS}
    record = dict(source=spec["source"], binding=src["binding"], path=src["path"], sha256=src["sha256"],
                  bytes=src["bytes"], manifest_sha256=src["manifest_sha256"], checks=src["checks"],
                  transform=rigid.describe(), transform_status=tf.get("status", "accepted-by-mpm-p1u5"),
                  native_frame=xpbd_mesh.FRAME)
    return X, T, a["vertex_ids"].astype(np.int64), a["triangle_ids"].astype(np.int64), masks, record, rigid, a


def side_data(X, T, area_rtol=1e-12):
    """Face normals, edge/vertex pseudo-normals, rim flags, degenerate faces (NumPy, init only)."""
    e1, e2 = X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]]
    cr = np.cross(e1, e2)
    area2 = np.linalg.norm(cr, axis=1)
    scale = np.maximum(np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1), 1e-300)
    degen = area2 <= area_rtol * scale
    fn = np.where(degen[:, None], 0.0, cr / np.where(degen, 1.0, area2)[:, None])
    edges = {}
    for t, tri in enumerate(T):
        if degen[t]:
            continue
        for k in range(3):
            key = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            edges.setdefault(key, []).append(t)
    bad = [k for k, f in edges.items() if len(f) > 2]
    if bad:
        raise ValueError(f"stub surface: non-manifold edges {bad[:5]}")
    en = np.zeros((len(T), 3, 3))
    erim = np.zeros((len(T), 3), np.int32)
    vrim = np.zeros(len(X), np.int32)
    for t, tri in enumerate(T):
        for k in range(3):
            key = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
            faces = edges.get(key, [t])
            s = fn[faces].sum(0)
            ns = np.linalg.norm(s)
            if len(faces) == 1 or ns < 1e-12:          # boundary (or folded flat onto itself)
                erim[t, k] = 1
                vrim[list(key)] = 1
                en[t, k] = fn[t]
            else:
                en[t, k] = s / ns
    vn = np.zeros((len(X), 3))
    for t, tri in enumerate(T):
        if degen[t]:
            continue
        for k in range(3):
            i, j, l = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            a, b = X[j] - X[i], X[l] - X[i]
            ang = math.atan2(np.linalg.norm(np.cross(a, b)), float(a @ b))
            vn[i] += ang * fn[t]
    nv = np.linalg.norm(vn, axis=1)
    vn = vn / np.where(nv > 0, nv, 1.0)[:, None]
    return dict(face_n=fn, edge_n=en, edge_rim=erim, vert_n=vn, vert_rim=vrim, degenerate=degen.astype(np.int32))


@ti.data_oriented
class StubSurface:
    KEYS = ("type", "center", "size", "cells", "normal_axis", "normal_sign", "pinned", "id_base", "motion", "holes")
    MESH_KEYS = ("type", "source", "path", "expect_sha256", "expect_manifest_sha256", "transform", "pinned", "motion")

    def __init__(self, spec, dtype, where="contact.surface"):
        kind = spec.get("type")
        if kind not in TYPES:
            raise KeyError(f"{where}.type must be one of {list(TYPES)}, got {kind!r}")
        check_keys(where, spec, self.KEYS if kind == "sheet" else self.MESH_KEYS)
        if kind == "sheet":
            for k in ("center", "size", "cells", "normal_axis"):
                if k not in spec:
                    raise KeyError(f"{where}: missing {k!r}")
            if spec["normal_axis"] not in AXES:
                raise ValueError(f"{where}.normal_axis must be one of {list(AXES)}")
        pinned = spec.get("pinned", "all")
        motion = dict(spec.get("motion", {"type": "fixed"}))
        check_keys(f"{where}.motion", motion, ("type", "vel"))
        if motion.get("type") not in MOTIONS:
            raise ValueError(f"{where}.motion.type must be one of {list(MOTIONS)}")
        self.dtype = dtype
        self.source = None                                  # mesh: source record (file hash, checks, transform)
        self.rigid = None                                   # mesh: native <-> world map
        masks = {}
        if kind == "sheet":
            self.spec = dict(spec, pinned=pinned, motion=motion, normal_sign=int(spec.get("normal_sign", 1)),
                             id_base=int(spec.get("id_base", 0)))
            X, T = sheet(self.spec)
            vertex_ids = self.spec["id_base"] + np.arange(len(X), dtype=np.int64)
            triangle_ids = self.spec["id_base"] + np.arange(len(T), dtype=np.int64)
            self.native = None
        else:
            self.spec = dict(spec, pinned=pinned, motion=motion)
            X, T, vertex_ids, triangle_ids, masks, self.source, self.rigid, self.native = mesh_surface(self.spec)
        if pinned not in PINNED and pinned not in masks:
            raise ValueError(f"{where}.pinned must be one of {list(PINNED) + sorted(masks)}")
        sd = side_data(X, T)
        self.X0, self.T = X, T
        self.nv, self.nt = len(X), len(T)
        self.vertex_ids, self.triangle_ids = vertex_ids, triangle_ids
        self.support_masks = masks
        pin = np.ones(self.nv, np.int32) if pinned == "all" else (
            sd["vert_rim"].copy() if pinned == "rim" else (
                np.zeros(self.nv, np.int32) if pinned == "none" else masks[pinned].copy()))
        self.pinned_np = pin
        self.vert_rim_np = sd["vert_rim"]
        self.degenerate_np = sd["degenerate"]

        V, Tn = self.nv, self.nt
        self.x_s = ti.Vector.field(3, ti.f64, V)          # snapshot position (accumulated in f64)
        self.v_s = ti.Vector.field(3, dtype, V)           # snapshot velocity
        self.tri = ti.Vector.field(3, ti.i32, Tn)
        self.face_n = ti.Vector.field(3, dtype, Tn)
        self.edge_n = ti.Matrix.field(3, 3, dtype, Tn)    # row k: pseudo-normal of edge k
        self.edge_rim = ti.Vector.field(3, ti.i32, Tn)
        self.vert_n = ti.Vector.field(3, dtype, V)
        self.vert_rim = ti.field(ti.i32, V)
        self.degenerate = ti.field(ti.i32, Tn)
        self.pinned = ti.field(ti.i32, V)
        # clock and version
        self.t_s = ti.field(ti.f64, ())
        self.step_index = ti.field(ti.i64, ())
        # per-vertex impulse material -> cloth [N s], f64: this period, last settled batch, cumulative
        vec = lambda: ti.Vector.field(3, ti.f64, V)
        self.JP_per, self.JG_per = vec(), vec()
        self.JP_last, self.JG_last = vec(), vec()
        self.JP_cum, self.JG_cum = vec(), vec()
        self.last_t = ti.Vector.field(2, ti.f64, ())     # [t_start, t_end) of the last batch, solver clock
        self.last_step = ti.field(ti.i64, ())             # step_index the last batch belongs to
        # window clock (python ints, exact): start of the open window and the last batch [ns]
        self.t_ns = 0
        self.last_ns = None
        self.H_ns = None

        self.x_s.from_numpy(X)
        self.tri.from_numpy(T)
        self.face_n.from_numpy(sd["face_n"].astype(np.float64 if dtype == ti.f64 else np.float32))
        self.edge_n.from_numpy(sd["edge_n"].astype(np.float64 if dtype == ti.f64 else np.float32))
        self.edge_rim.from_numpy(sd["edge_rim"])
        self.vert_n.from_numpy(sd["vert_n"].astype(np.float64 if dtype == ti.f64 else np.float32))
        self.vert_rim.from_numpy(sd["vert_rim"])
        self.degenerate.from_numpy(sd["degenerate"])
        self.pinned.from_numpy(pin)
        self.last_step[None] = -1
        self.set_params(vel=motion.get("vel", [0.0, 0.0, 0.0]))

    # ------------------------------------------------------------ python side
    def set_params(self, **kw):
        """Run-time: vel (translate). Change it at period boundaries: inside a period the
        prediction x_s + (t - t_s) v_s would jump."""
        check_keys("surface.set_params", kw, ("vel",))
        if "vel" in kw:
            vel = [float(c) for c in kw["vel"]]
            if self.spec["motion"]["type"] == "fixed" and any(vel):
                raise ValueError("fixed surface: velocity must be zero (use motion.type = translate)")
            self.spec["motion"]["vel"] = vel
            self.v_s.from_numpy(np.tile(np.asarray(vel, np.float64), (self.nv, 1)).astype(
                np.float64 if self.dtype == ti.f64 else np.float32))

    def version(self) -> tuple:
        return (1, int(self.step_index[None]), 0)

    def positions_at(self, t: float) -> np.ndarray:
        """Predicted vertex positions at time t (export and analysis)."""
        return self.x_s.to_numpy() + (t - float(self.t_s[None])) * self.v_s.to_numpy().astype(np.float64)

    def last_batch(self) -> dict:
        """The last settled batch (P21-01 LoadBatch semantics, stub side): read-only copy.
        window_index k = windows completed before it (= its cloth_version step_index);
        t_start/t_end follow the window clock (t_ns / 1e9) when H is a whole number of ns,
        else the solver clock; t_solver is always the solver clock (accumulated dt)."""
        ts, te = self.last_t[None].to_numpy().tolist()
        k = int(self.last_step[None])
        out = dict(window_index=k, cloth_version=(1, k, 0), t_solver=(ts, te), t_start=ts, t_end=te,
                   t_start_ns=None, t_end_ns=None, vertex_ids=self.vertex_ids.copy(),
                   impulse_P=self.JP_last.to_numpy(), impulse_G=self.JG_last.to_numpy())
        if self.last_ns is not None:
            out.update(t_start_ns=self.last_ns[0], t_end_ns=self.last_ns[1],
                       t_start=self.last_ns[0] / 1e9, t_end=self.last_ns[1] / 1e9)
        return out

    def settle_window(self, time, H_ns):
        """End of a window: settle (kernel) and advance the integer window clock by H_ns
        (None: H is not a whole number of ns; the window clock is then dropped for good)."""
        self.settle(time)
        if H_ns is None or self.t_ns is None:
            self.t_ns = self.last_ns = None
        else:
            self.last_ns = (self.t_ns, self.t_ns + int(H_ns))
            self.t_ns += int(H_ns)
        self.H_ns = H_ns

    def mesh_arrays(self) -> dict:
        return dict(vertex_ids=self.vertex_ids, triangle_ids=self.triangle_ids, triangles=self.T,
                    x0=self.X0, pinned=self.pinned_np.astype(np.int8), degenerate=self.degenerate_np.astype(np.int8))

    def describe(self) -> dict:
        d = dict(self.spec, n_vertices=self.nv, n_triangles=self.nt, n_pinned=int(self.pinned_np.sum()),
                 n_rim_vertices=int(self.vert_rim_np.sum()),
                 n_degenerate=int(self.degenerate_np.sum()), version_now=self.version())
        if self.spec["type"] == "sheet":
            d["edge_length"] = [float(s) / float(c) for s, c in zip(self.spec["size"], self.spec["cells"])]
        else:
            e = self.X0[self.T[:, 1]] - self.X0[self.T[:, 0]]
            d.update(edge_length_mean=float(np.linalg.norm(e, axis=1).mean()), source_record=self.source,
                     support_masks={k: int(v.sum()) for k, v in self.support_masks.items()})
        return d

    # ------------------------------------------------------------ taichi side
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
        """End of an exchange period: batch -> last and cumulative, then advance the snapshot
        to the solver time (time is the solver's 0-d f64 clock, read in the loop, D21)."""
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
