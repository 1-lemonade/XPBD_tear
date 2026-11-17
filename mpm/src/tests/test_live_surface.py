"""P2-U1 LiveSurface : side information
of deformed snapshots against the NumPy reference (surface.side_data); bit-for-bit equality
with StubSurface under rigid motion (F3 on the XPBD mesh through the proposed transform, F4
translating sheet through the identity, with and without spare capacity); topology rebuild
after a tear; rejection of bad snapshots and phase errors; oriented corners, normal, velocity
and impulse round trip of the native -> world map; XPBD snapshot objects and the tick
prediction (read-only import of the XPBD mirror, skipped when it is absent)."""
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import taichi as ti

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "scenes"))
sys.path.insert(0, str(WORK / "tests"))
from mpm import Diagnostics, Solver, load_config  # noqa: E402
from mpm.contact import LiveSurface, StubSurface, side_data, sheet, xpbd_mesh  # noqa: E402
from mpm.contact.live import PARK, read_snapshot, topology_tables  # noqa: E402
import f3_replay  # noqa: E402
import stub_contact  # noqa: E402
import live_fixtures as lf  # noqa: E402

CONF = WORK / "scenes" / "configs"
F3M, F4 = CONF / "f3_replay_n64.json", CONF / "stub_drop_impact.json"
NV, NT = 52 * 52, 2 * 51 * 51
PARTICLE_KEYS = ("x", "v", "C", "F", "tau", "q", "vc", "alpha", "contact_state", "contact_face", "contact_dist")
BLOCK_KEYS = ("JP", "JG", "JG_attr", "G_on_pinned", "KG", "n_active", "n_pen", "n_ambig", "missed", "rebuilds",
              "cand_overflow", "bin_overflow", "pen_max_run", "overlap_max_run", "Rq", "vmax_c", "n_cand")
SIDE_KEYS = ("face_n", "edge_n", "edge_rim", "vert_n", "vert_rim", "degenerate")


def init(threads=1, prec=ti.f64):
    ti.init(arch=ti.cpu, default_fp=prec, cpu_max_num_threads=threads, random_seed=0, log_level=ti.WARN)


def surface_alone(spec, dtype=ti.f64):
    return LiveSurface(spec, dtype)


def settle(S, t, H_ns=1_000_000):
    f = ti.field(ti.f64, ())
    f[None] = t
    S.settle_window(f, H_ns)


def state(sim):
    out = {k: getattr(sim.P, k).to_numpy() for k in PARTICLE_KEYS}
    out.update({f"block.{k}": getattr(sim.contact, k).to_numpy() for k in BLOCK_KEYS})
    return out


def stub_side(S, N, Tn):
    return {k: getattr(S, k).to_numpy()[:N if k.startswith("vert") else Tn] for k in SIDE_KEYS}


def assert_same(a, b, what, bits=False):
    """Equal values (np.array_equal: +0.0 == -0.0), or with bits=True equal bytes."""
    assert a.keys() == b.keys()
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        assert x.shape == y.shape and x.dtype == y.dtype, f"{what}: {k} shape/dtype"
        same = x.tobytes() == y.tobytes() if bits else np.array_equal(x, y)
        assert same, f"{what}: {k} differs (max {np.abs(x.astype(float) - y.astype(float)).max()})"


# ------------------------------------------------------------------ 1 side information vs NumPy
def _deformed_cases(a):
    """(name, world positions, triangles): deformations of the reference mesh in the world."""
    rng = np.random.default_rng(7)
    X0, T = a["positions"], a["triangles"]
    out = []
    bump = X0.copy()
    bump[:, 2] = 0.05 * np.sin(2 * np.pi * X0[:, 0] / 0.8) * np.cos(np.pi * X0[:, 1] / 0.8)
    bump += rng.normal(scale=1e-3, size=bump.shape)
    out.append(("bump_jitter", bump, T))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    out.append(("rotated", bump @ q.T + [0.3, 0.2, 0.1], T))
    cen = X0[T].mean(1)
    hole = (np.abs(cen[:, 0] - 0.4) < 0.08) & (np.abs(cen[:, 1] - 0.4) < 0.08)
    out.append(("hole_unused_vertices", bump, T[~hole]))     # vertices of the hole keep no face
    flat = X0.copy()
    t0 = 1000                                                # collapse a face onto one of its edges
    i, j, k = T[t0]
    flat[j] = 0.5 * (flat[i] + flat[k])
    out.append(("degenerate_face", flat, T))
    fold = X0.copy()
    m = fold[:, 0] > 0.6 + 1e-9                              # fold the last columns back onto the sheet
    fold[m, 0] = 1.2 - fold[m, 0]
    out.append(("folded_edge", fold, T))
    return out


def test_side_information_matches_numpy_reference_on_deformed_snapshots():
    init(threads=3)
    a = lf.reference_mesh()
    first = None
    for name, X, T in _deformed_cases(a):
        S = surface_alone(lf.live_spec(NV, NT, "identity"))
        S.ingest(lf.snapshot(X, T, a["vertex_ids"], a["triangle_ids"][:len(T)]))
        got = S.side_arrays()
        ref = side_data(X, T)                                # the input geometry (identity map), not the kernel's
        assert np.array_equal(S.x_s.to_numpy()[:NV], X)
        for k in ("edge_rim", "vert_rim", "degenerate"):
            assert np.array_equal(got[k], ref[k]), (name, k)
        # on the flat 180-degree fold the angle sums of the two sides cancel, so the pseudo-normal of
        # a fold vertex is the sign of round-off; the block never uses a rim vertex's pseudo-normal
        # beyond r <= tiny (rim features push along (x - cp) / r), so those vertices are left out
        vsel = (ref["vert_rim"] == 0) if name == "folded_edge" else slice(None)
        for k, sel in (("face_n", slice(None)), ("edge_n", slice(None)), ("vert_n", vsel)):
            assert np.allclose(got[k][sel], ref[k][sel], rtol=0, atol=1e-12), (name, k, np.abs(got[k][sel] - ref[k][sel]).max())
        if name == "bump_jitter":
            first = {k: v.copy() for k, v in got.items()}
            assert not got["degenerate"].any() and got["vert_rim"].sum() == 4 * 51
        if name == "hole_unused_vertices":
            assert got["edge_rim"].sum() > 204 and not got["vert_n"][np.isin(np.arange(NV), T, invert=True)].any()
        if name == "degenerate_face":
            assert got["degenerate"].sum() == 1 and got["degenerate"][1000] == 1 and got["edge_rim"][1000].all()
        if name == "folded_edge":
            assert got["edge_rim"].sum() > 204                # the fold line is a rim (n_a + n_b = 0)
    # the kernels are gathers: one thread gives the same bits as three
    init(threads=1)
    name, X, T = _deformed_cases(a)[0]
    S = surface_alone(lf.live_spec(NV, NT, "identity"))
    S.ingest(lf.snapshot(X, T, a["vertex_ids"], a["triangle_ids"]))
    assert_same(S.side_arrays(), first, "threads 1 vs 3", bits=True)


# ------------------------------------------------------------------ 2 rigid motion: bit for bit with the stub
def _f3_mesh_cfg(surface=None):
    sets = ["grid.n=24", "precision=f64", "scene.block.size=[0.3,0.1,0.3]"]
    if surface is not None:
        sets.append("contact.surface=" + json.dumps(surface))
    return load_config(str(F3M), sets, f3_replay.SCENE_DEFAULTS)


def test_f3_mesh_fixed_live_equals_stub_bit_for_bit():
    """F3 on the XPBD 52 x 52 mesh, fixed, proposed transform: the live surface reads the native
    reference mesh every window; one thread; 30 windows of 5 substeps."""
    W = 30
    runs = []
    for live in (False, True):
        init(threads=1)
        a = lf.reference_mesh()
        cfg = _f3_mesh_cfg(lf.live_spec(NV, NT, None) if live else None)
        pos, _ = f3_replay.initial_positions(cfg, 0.3)
        sim = Solver(cfg, pos)
        S, c = sim.contact.surface, sim.contact
        diag = Diagnostics(sim)
        diag()
        batches = []
        for k in range(W):
            if live:
                S.ingest(lf.reference_snapshot(a, step=k, time=k * 1e-3))
            sim.advance(c.n_sub)
            batches.append(S.last_batch())
        d = diag()
        runs.append((state(sim), stub_side(S, NV, NT), batches, d, S.x_s.to_numpy()[:NV]))   # host copies: the next ti.init frees the fields
    (s0, side0, b0, d0, stub), (s1, side1, b1, d1, live) = runs
    assert_same(s0, s1, "particles and block", bits=True)
    assert_same(side0, side1, "side information")
    for k in range(W):
        for key in ("impulse_G", "impulse_P", "vertex_ids"):
            assert b0[k][key].tobytes() == b1[k][key].tobytes(), (k, key)
        for key in ("window_index", "t_start_ns", "t_end_ns", "t_solver"):
            assert b0[k][key] == b1[k][key], (k, key)
        assert b1[k]["cloth_version"] == (1, k, 0)
    assert np.array_equal(stub, live)
    assert d1["bal_rel"] < 1e-10 and d1["contact_active"] > 50
    # impulse direction : the resting sand pushes the cloth down, world -Y
    # = native -Z, and G_native_z = G_world_y exactly (R^T maps world y to native z)
    G = np.sum([b["impulse_G"] for b in b1], axis=0)
    Gn = np.sum([b["impulse_G_native"] for b in b1], axis=0)
    assert G[:, 1].sum() < 0 and np.array_equal(Gn[:, 2], G[:, 1]) and np.array_equal(Gn[:, 0], G[:, 0])
    assert np.array_equal(Gn[:, 1], -G[:, 2])


def test_f4_translating_sheet_live_equals_stub_with_and_without_spare_capacity():
    """F4 (f32 as configured) with the sheet translating along all three axes and the block
    falling onto it, identity transform. The live surface reads, every window, the stub's own
    x_s and v_s at that window's start; a third run carries 60 spare vertices and 150 spare
    triangles (inert: parked, degenerate). One thread; 70 windows of 10 substeps."""
    W = 70
    vel = [0.1, 0.15, 0.05]
    base = ["grid.n=16", "precision=f32", "scene.block.bottom_gap=0.05"]
    cfg0 = load_config(str(F4), base + ["contact.surface.motion=" + json.dumps(dict(type="translate", vel=vel))],
                       stub_contact.SCENE_DEFAULTS)
    pos, _ = stub_contact.initial_positions(cfg0)
    init(threads=1, prec=ti.f32)
    sim = Solver(cfg0, pos)
    S, c = sim.contact.surface, sim.contact
    nv, nt = S.nv, S.nt
    snaps, b_stub = [], []
    for k in range(W):
        snaps.append(lf.snapshot_of_stub(S, step=k, time=k * 2e-3))
        sim.advance(c.n_sub)
        b_stub.append(S.last_batch())
    ref = (state(sim), stub_side(S, nv, nt), S.x_s.to_numpy())
    # the stub really translated: its window-start positions follow X0 + k H v (analytic, f32 velocity;
    # dt is f32(2e-4) in this f32 run, so the solver's H differs from 2 ms by ~5e-8 relative)
    v32 = np.asarray(vel, np.float32).astype(np.float64)
    for k in (1, W // 2, W - 1):
        assert np.allclose(snaps[k].positions, snaps[0].positions + k * 2e-3 * v32, rtol=0, atol=1e-9), k
    for extra in ((0, 0), (60, 150)):
        cfg = load_config(str(F4), base + ["contact.surface=" + json.dumps(lf.live_spec(nv, nt, "identity", extra))],
                          stub_contact.SCENE_DEFAULTS)
        init(threads=1, prec=ti.f32)
        sim = Solver(cfg, pos)
        L, c = sim.contact.surface, sim.contact
        assert (L.nv, L.nt) == (nv + extra[0] + 1, nt + extra[1])
        for k in range(W):
            rep = L.ingest(snaps[k])
            if k and extra == (0, 0):                         # live settle advanced x_s exactly as the stub did
                assert rep["max_prediction_deviation"] == 0.0
                assert rep["max_step"] == pytest.approx(2e-3 * np.linalg.norm(vel), rel=1e-6)
            sim.advance(c.n_sub)
            b = L.last_batch()
            for key in ("impulse_G", "impulse_P", "vertex_ids"):
                assert b_stub[k][key].tobytes() == b[key].tobytes(), (extra, k, key)
        got = state(sim)
        if extra != (0, 0):                                  # the padding adds entries to corner bin 0 only
            assert sim.contact.bin_overflow[None] == 0
        assert_same(ref[0], got, f"particles and block, spare {extra}", bits=True)
        assert_same(ref[1], stub_side(L, nv, nt), f"side, spare {extra}")
        assert np.array_equal(ref[2], L.x_s.to_numpy()[:nv])
        assert not L.JG_cum.to_numpy()[nv:].any() and np.all(L.x_s.to_numpy()[nv:] == PARK)
        assert L.degenerate.to_numpy()[nt:].all()
    assert got["block.rebuilds"] == W and np.abs(L.JG_cum.to_numpy()).sum() > 0   # the block met the moving sheet


# ------------------------------------------------------------------ 3 topology change
def test_topology_rebuild_after_a_tear():
    init(threads=3)
    a = lf.reference_mesh()
    X, T, ids, tids, pin, dup = lf.slit(a)
    cap = lf.live_spec(NV, NT, None, extra=(64, 0))
    S = surface_alone(cap)
    S.ingest(lf.reference_snapshot(a, step=0, time=0.0))
    G = np.zeros((S.nv, 3))
    G[:NV] = np.random.default_rng(3).normal(size=(NV, 3))
    S.JG_per.from_numpy(G)
    settle(S, 1e-3)
    rep = S.ingest(lf.snapshot(X, T, ids, tids, pinned=pin, step=1, time=1e-3, topo=1))
    assert rep["rebuilt"] and S.rebuilds == 2 and S.n_active == (len(X), NT)
    # same tables and side information as a surface built from the torn snapshot directly
    F = surface_alone(cap)
    F.ingest(lf.snapshot(X, T, ids, tids, pinned=pin, step=1, time=1e-3, topo=1))
    assert_same(S.side_arrays(), F.side_arrays(), "rebuild vs fresh")
    for name in ("tri", "opp", "vf_start", "vf_items", "pinned"):
        assert np.array_equal(getattr(S, name).to_numpy(), getattr(F, name).to_numpy()), name
    Xw = S.rigid.to_world(X)                                 # NumPy map of the input, independent of the kernel
    assert np.allclose(S.x_s.to_numpy()[:len(X)], Xw, rtol=0, atol=1e-15)
    ref = side_data(Xw, T)
    got = S.side_arrays()
    for k in ("face_n", "edge_n", "vert_n"):
        assert np.allclose(got[k], ref[k], rtol=0, atol=1e-12), k
    for k in ("edge_rim", "vert_rim", "degenerate"):
        assert np.array_equal(got[k], ref[k]), k
    # both lips are rims: 29 slit vertices and their copies, 30 edges on each side
    lips = np.concatenate([S.rows_of(dup), S.rows_of(dup + 100000)])
    assert got["vert_rim"][lips].all() and got["edge_rim"].sum() == 204 + 2 * 30
    tips = S.rows_of([10 * 52 + 25, 40 * 52 + 25])
    assert got["vert_rim"][tips].all()
    # cumulative per-vertex impulse follows the stable IDs; the copies start from zero
    cum = S.JG_cum.to_numpy()
    assert np.array_equal(cum[S.rows_of(a["vertex_ids"])], G[:NV])
    assert not cum[S.rows_of(dup + 100000)].any() and not S.dropped_cum["JG"].any()
    # a later topology without some IDs: their cumulative impulse is reported as dropped
    settle(S, 2e-3)
    keep = np.ones(len(X), bool)
    gone = S.rows_of([0])                                    # corner vertex 0 and its single face
    keep[gone] = False
    face_keep = ~np.isin(T, gone).any(1)
    remap = -np.ones(len(X), np.int64)
    remap[keep] = np.arange(keep.sum())
    S.ingest(lf.snapshot(X[keep], remap[T[face_keep]], ids[keep], tids[face_keep], pinned=pin[keep], step=2,
                         time=2e-3, topo=2))
    assert np.array_equal(S.dropped_cum["JG"], G[0])
    with pytest.raises(ValueError):
        S.rows_of([0])


def test_contact_continues_across_a_topology_rebuild():
    """F3 on the live mesh: 4 windows untorn, then a slit under the sand (topology_version 1)
    and 6 more windows: finite, balance closed, no overflow, faces bound to valid rows."""
    init(threads=3)
    a = lf.reference_mesh()
    X, T, ids, tids, pin, _ = lf.slit(a)
    cfg = _f3_mesh_cfg(lf.live_spec(NV, NT, None, extra=(64, 0)))
    pos, _ = f3_replay.initial_positions(cfg, 0.3)
    sim = Solver(cfg, pos)
    S, c = sim.contact.surface, sim.contact
    diag = Diagnostics(sim)
    for k in range(10):
        if k < 4:
            S.ingest(lf.reference_snapshot(a, step=k, time=k * 1e-3))
        else:
            S.ingest(lf.snapshot(X, T, ids, tids, pinned=pin, step=k, time=k * 1e-3, topo=1))
        sim.advance(c.n_sub)
    d = diag()
    face = sim.P.contact_face.to_numpy()
    assert S.rebuilds == 2 and d["nonfinite_events"] == 0 and d["bal_rel"] < 1e-10
    assert d["contact_overflow"] == 0 and d["contact_bin_overflow"] == 0 and d["contact_active"] > 50
    assert face.max() < NT and S.last_batch()["vertex_ids"].shape == (len(X),)
    # fewer active rows: drop the corner vertex 0 and its face (far from the sand), two more windows
    gone = S.rows_of([0])
    keep = np.ones(len(X), bool)
    keep[gone] = False
    fk = ~np.isin(T, gone).any(1)
    remap = -np.ones(len(X), np.int64)
    remap[keep] = np.arange(keep.sum())
    for k in (10, 11):
        S.ingest(lf.snapshot(X[keep], remap[T[fk]], ids[keep], tids[fk], pinned=pin[keep], step=k, time=k * 1e-3,
                             topo=2))
        sim.advance(c.n_sub)
    d = diag()
    n_v, n_t = S.n_active
    assert (n_v, n_t) == (len(X) - 1, NT - 1) and d["bal_rel"] < 1e-10 and d["contact_bin_overflow"] == 0
    assert np.all(S.x_s.to_numpy()[n_v:] == PARK) and S.degenerate.to_numpy()[n_t:].all()
    assert sim.P.contact_face.to_numpy().max() < n_t and not S.JG_last.to_numpy()[n_v:].any()


# ------------------------------------------------------------------ 4 rejections and phase
def _small():
    X, T = sheet(dict(lf.SHEET, cells=[3, 3]))
    return X, T.astype(np.int64), np.arange(len(X)) + 7, np.arange(len(T)) + 50


def _bad_cases():
    X, T, vid, tid = _small()
    nonfinite = X.copy()
    nonfinite[3, 1] = np.nan
    flip = T.copy()
    flip[4] = flip[4][[0, 2, 1]]
    dup_face = np.vstack([T, T[2][[0, 2, 1]]])
    third = np.vstack([T, [[T[0][0], T[0][1], 15]]])          # a third face on edge (T00, T01), same way
    rep = T.copy()
    rep[0, 2] = rep[0, 0]
    return {
        "nonfinite": dict(positions=nonfinite),
        "index_range": dict(triangles=np.where(T == 5, 16, T)),
        "duplicate_ids": dict(vertex_ids=np.where(vid == 8, 7, vid)),
        "negative_triangle_id": dict(triangle_ids=-tid),
        "repeated_vertex": dict(triangles=rep),
        "inconsistent_winding": dict(triangles=flip),
        "duplicate_face": dict(triangles=dup_face, triangle_ids=np.arange(len(dup_face))),
        "third_face_on_edge": dict(triangles=third, triangle_ids=np.arange(len(third))),
        "pinned_shape": dict(pinned=np.zeros(3, bool)),
        "velocity_shape": dict(velocity=np.zeros((len(X), 2))),
        "capacity": dict(positions=np.vstack([X, X[:1] + 1e-3]), velocity=np.zeros((len(X) + 1, 3)),
                         vertex_ids=np.arange(len(X) + 1), pinned=np.zeros(len(X) + 1, bool)),
    }


@pytest.mark.parametrize("case", list(_bad_cases()))
def test_first_snapshot_rejections(case):
    init(threads=1)
    X, T, vid, tid = _small()
    S = surface_alone(lf.live_spec(len(X), len(T) + 1))
    fields = dict(positions=X, triangles=T, vertex_ids=vid, triangle_ids=tid, velocity=np.zeros((len(X), 3)),
                  pinned=np.zeros(len(X), bool))
    fields.update(_bad_cases()[case])
    with pytest.raises(ValueError):
        S.ingest(lf.snapshot(**fields))
    assert S.version() is None and S.rebuilds == 0
    S.ingest(lf.snapshot(X, T, vid, tid))                    # a rejected snapshot changed nothing
    assert S.version() == (1, 0, 0)


@pytest.mark.parametrize("case", ["step_jump", "epoch", "time", "topology_down", "changed_without_version",
                                  "missing_field"])
def test_sequence_rejections(case):
    init(threads=1)
    X, T, vid, tid = _small()
    S = surface_alone(lf.live_spec(len(X), len(T)))
    S.ingest(lf.snapshot(X, T, vid, tid, step=3, time=0.5, topo=2))
    settle(S, 1e-3)
    nxt = dict(step=4, time=0.501, topo=2)
    kw = dict(step_jump=dict(step=5), epoch=dict(epoch=2), time=dict(time=0.5), topology_down=dict(topo=1)).get(case, {})
    tri = T[::-1].copy() if case == "changed_without_version" else T
    snap = lf.snapshot(X, tri, vid, tid, **dict(nxt, **kw))
    if case == "missing_field":
        snap = {k: getattr(snap, k) for k in ("positions", "velocity", "triangles", "vertex_ids", "triangle_ids",
                                              "pinned", "epoch", "step_index", "topology_version", "time")}
    with pytest.raises(ValueError):
        S.ingest(snap)
    S.ingest(lf.snapshot(X, T, vid, tid, **nxt))              # unchanged state: the right one still passes
    assert S.version() == (1, 4, 2)


def test_phase_errors():
    init(threads=1)
    X, T, vid, tid = _small()
    S = surface_alone(lf.live_spec(len(X), len(T)))
    with pytest.raises(RuntimeError):
        settle(S, 1e-3)                                      # a window without a snapshot
    S.ingest(lf.snapshot(X, T, vid, tid))
    with pytest.raises(RuntimeError):
        S.ingest(lf.snapshot(X, T, vid, tid, step=1, time=1e-3))   # twice inside one window
    settle(S, 1e-3)
    m = S.mesh_arrays()                                     # P2-U2: the current snapshot's topology
    assert m["triangles"].shape == T.shape and np.array_equal(m["vertex_ids"], vid)
    with pytest.raises(RuntimeError):
        surface_alone(lf.live_spec(len(X), len(T))).mesh_arrays()    # before any snapshot
    with pytest.raises(KeyError):
        surface_alone(dict(type="live"))                    # capacity is required
    S.ingest(lf.snapshot(X, T, vid, tid, step=1, time=1e-3))
    f = ti.field(ti.f64, ())
    f[None] = 2e-3
    with pytest.raises(RuntimeError, match="whole"):
        S.settle_window(f, None)                            # D30: H must be a whole number of ns
    for bad in (dict(capacity=dict(vertices=16.0, triangles=18)), dict(area_rtol=float("nan")),
                dict(transform=dict(c=[0.0, float("nan"), 0.0]))):
        with pytest.raises(ValueError):
            surface_alone(dict(lf.live_spec(len(X), len(T)), **bad))
    with pytest.raises(ValueError):
        S.set_params(area_rtol=-1.0)


def test_topology_tables_match_the_stub_checks():
    """Directed-edge uniqueness = consistent winding + at most two faces per edge (the stub's
    check_orientation and side_data's non-manifold test), on the reference mesh and a hole."""
    a = lf.reference_mesh(published=False)
    T = a["triangles"]
    tab = topology_tables(T, NV)
    sd = side_data(a["positions"], T)
    assert np.array_equal((tab["opp"] < 0).astype(np.int32), sd["edge_rim"])
    o = tab["opp"]
    for t in (0, 777, 5201):                                 # the opposite face really shares edge k reversed
        for k in range(3):
            if o[t, k] >= 0:
                e = (T[t, k], T[t, (k + 1) % 3])
                other = T[o[t, k]]
                assert any((other[j], other[(j + 1) % 3]) == e[::-1] for j in range(3))
    assert np.array_equal(np.diff(tab["vf_start"]), np.bincount(T.ravel(), minlength=NV))
    cen = a["positions"][T].mean(1)                          # a hole: inner boundary edges are boundaries too
    Th = T[~((np.abs(cen[:, 0] - 0.4) < 0.1) & (np.abs(cen[:, 1] - 0.4) < 0.1))]
    tab_h = topology_tables(Th, NV)
    assert np.array_equal((tab_h["opp"] < 0).astype(np.int32), side_data(a["positions"], Th)["edge_rim"])
    assert (tab_h["opp"] < 0).sum() > 204


# ------------------------------------------------------------------ 5 transform: corners, normal, round trip
def test_native_to_world_oriented_corners_normal_velocity_and_impulse_round_trip():
    init(threads=3)
    a = lf.reference_mesh()
    S = surface_alone(lf.live_spec(NV, NT, None))
    vn = np.array([0.01, 0.02, 0.03])
    S.ingest(lf.reference_snapshot(a, velocity=np.tile(vn, (NV, 1)), pinned=np.ones(NV, bool)))
    X = S.x_s.to_numpy()[:NV]
    rows = S.rows_of([0, 51, 2652, 2703])                   # row 0 col 0, row 0 col 51, row 51 col 0, row 51 col 51
    assert np.allclose(X[rows], [[0.1, 0.3, 0.1], [0.9, 0.3, 0.1], [0.1, 0.3, 0.9], [0.9, 0.3, 0.9]], rtol=0, atol=1e-15)
    e_x, e_y = X[rows[1]] - X[rows[0]], X[rows[0]] - X[rows[2]]   # native +x, native +y
    assert np.allclose(e_x / np.linalg.norm(e_x), [1, 0, 0], atol=1e-15)
    assert np.allclose(e_y / np.linalg.norm(e_y), [0, 0, -1], atol=1e-15)
    side = S.side_arrays()                                   # native +Z (winding) -> world +Y, exactly
    assert np.array_equal(np.abs(side["face_n"]), np.tile([0.0, 1.0, 0.0], (NT, 1))) and (side["face_n"][:, 1] == 1).all()
    assert (side["vert_n"][:, 1] == 1).all() and S.pinned.to_numpy()[:NV].all()
    assert np.array_equal(S.v_s.to_numpy()[0], [0.01, 0.03, -0.02])
    assert np.allclose(S.rigid.to_native(X), a["positions"], rtol=0, atol=1e-15)
    # impulses: world -> native (kernel, R^T) -> world (R), bit for bit for this signed permutation
    Gw, Pw = np.random.default_rng(5).normal(size=(2, NV, 3))
    for f, val in ((S.JG_per, Gw), (S.JP_per, Pw)):
        pad = np.zeros((S.nv, 3))
        pad[:NV] = val
        f.from_numpy(pad)
    settle(S, 1e-3)
    b = S.last_batch()
    for key, w in (("G", Gw), ("P", Pw)):
        assert np.array_equal(b[f"impulse_{key}"], w) and np.array_equal(b[f"impulse_{key}_native"], w @ S.rigid.R)
        assert np.array_equal(b[f"impulse_{key}_native"] @ S.rigid.R.T, w)
    # angular impulse about the native origin from the one about the world origin
    K_w, J_w = np.cross(X, Gw).sum(0), Gw.sum(0)
    K_n = np.cross(a["positions"], b["impulse_G_native"]).sum(0)
    assert np.allclose(S.rigid.torque_to_native(K_w, J_w), K_n, rtol=1e-12, atol=1e-12)
    assert S.describe()["transform"]["user_signoff"] == "pending"
    with pytest.raises(ValueError):                         # a reflection is not a proper rotation
        surface_alone(dict(lf.live_spec(NV, NT), transform=dict(R=[[1, 0, 0], [0, 1, 0], [0, 0, -1]])))


# ------------------------------------------------------------------ 6 XPBD snapshot objects (read-only mirror)
XPBD_WORK = pathlib.Path(os.environ.get("MPM_XPBD_WORK", WORK.parents[1]))
# Expected source digest for this fixture.
XPBD_SOURCES = {"project/coupling/kinematic_surface.py": "5cf1942dfcd6a762e104a3f1d55c64c5e4de766590b2ad0435867a84a9654312",
                "project/coupling/contracts.py": "02af24df3b377285c962f242082d102e32eb3e97e84e36f58f13d520fe122a95",
                "project/coupling/exchange.py": "da93067bb5d193ffae418d04f6f88c628e4985847483b2f18fdef4d709531807",
                "project/coupling/loads.py": "6998dbd955768ed1c72d452d0c0be338c582060d0d845bb4a41b7c333acfa019"}


@pytest.fixture
def xpbd():
    if not (XPBD_WORK / "project" / "coupling" / "kinematic_surface.py").exists():
        pytest.skip(f"XPBD mirror not found at {XPBD_WORK} (set MPM_XPBD_WORK)")
    for rel, digest in XPBD_SOURCES.items():
        got = hashlib.sha256((XPBD_WORK / rel).read_bytes()).hexdigest()
        assert got == digest, f"XPBD mirror {rel} changed ({got}): re-check the U13 sources before use"
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); "
             "import project.coupling.kinematic_surface, project.coupling.exchange, project.coupling.loads; "
             "print('taichi' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(XPBD_WORK)], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "False", out   # importing them starts no Taichi
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True                           # read-only: no __pycache__ in the mirror
    sys.path.insert(0, str(XPBD_WORK))
    try:
        from project.coupling.contracts import ClothSnapshot
        from project.coupling.exchange import SCHEMA, ExchangeAdapter, ImpulseWindow
        from project.coupling.kinematic_surface import KinematicSurfaceBridge
        from project.coupling.loads import LoadBatch, SourceIdentity
        yield dict(ClothSnapshot=ClothSnapshot, SCHEMA=SCHEMA, ExchangeAdapter=ExchangeAdapter,
                   ImpulseWindow=ImpulseWindow, KinematicSurfaceBridge=KinematicSurfaceBridge,
                   LoadBatch=LoadBatch, SourceIdentity=SourceIdentity)
    finally:
        sys.path.remove(str(XPBD_WORK))
        sys.dont_write_bytecode = old
        for m in [m for m in sys.modules if m == "project" or m.startswith("project.")]:
            del sys.modules[m]


@ti.kernel
def _predict(S: ti.template(), tau: ti.f64, out: ti.template()):
    for v in range(S.nv):
        out[v] = ti.cast(S.vertex_at(v, ti.cast(tau, S.dtype)), ti.f64)


@pytest.mark.parametrize("vel", [[0.0, 0.0, 0.0], [0.2, -0.3, 0.5]], ids=["fixed", "translate"])
def test_xpbd_snapshots_tick_prediction_and_impulse_path(xpbd, vel):
    """KinematicSurfaceBridge (U13 candidate) snapshots of the reference mesh, 3 windows (h 1 ms,
    particle_dt 0.2 ms): offer.snapshot -> ingest; the live prediction at every tick equals the
    XPBD frozen-velocity predictor mapped to the world; a world impulse set goes out through
    the native batch (dense, shuffled IDs) and XPBD's receipt maps back to it. Synthetic,
    self-declared prescribed source: a geometry and impulse-path check, not a coupling run."""
    init(threads=3)
    X = xpbd
    a = lf.reference_mesh()
    src = X["SourceIdentity"]("mpm-p2u1-test", "prescribed", 1)
    snap0 = X["ClothSnapshot"](a["positions"], a["triangles"], a["vertex_ids"], a["triangle_ids"],
                               np.tile(vel, (NV, 1)), np.zeros(NV), np.ones(NV, bool), 1, 0, 0, 0.0, 1e-3)
    bridge = X["KinematicSurfaceBridge"](snap0, source=src, stream_id="p2u1-test", h=1e-3)
    adapter = X["ExchangeAdapter"](bridge, src)
    S = surface_alone(lf.live_spec(NV, NT, None))
    out = ti.Vector.field(3, ti.f64, S.nv)
    t = ti.field(ti.f64, ())
    rng = np.random.default_rng(11)
    for k in range(3):
        offer = adapter.open_window(h=1e-3, particle_dt=2e-4)
        rep = S.ingest(offer.snapshot)
        assert np.array_equal(S.vertex_ids, offer.snapshot.vertex_ids) and S.version() == offer.snapshot.version
        if k:
            assert rep["max_prediction_deviation"] < 1e-15
            assert rep["max_step"] == pytest.approx(1e-3 * np.linalg.norm(vel), rel=1e-9, abs=1e-15)
        for i in range(offer.substeps):
            _predict(S, i * offer.particle_dt, out)
            exp = S.rigid.to_world(offer.geometry_at_tick(i).positions)
            assert np.allclose(out.to_numpy()[:NV], exp, rtol=0, atol=1e-14), (k, i)
        Gw = rng.normal(size=(NV, 3)) * 1e-3
        pad = np.zeros((S.nv, 3))
        pad[:NV] = Gw
        S.JG_per.from_numpy(pad)
        t[None] = (k + 1) * 1e-3
        S.settle_window(t, 1_000_000)
        b = S.last_batch()
        perm = rng.permutation(NV)
        batch = X["LoadBatch"](stream_id=offer.stream_id, source=offer.source, cloth_version=offer.snapshot.version,
                               batch_id=f"p2u1-test/w{k:05d}", t_start=offer.snapshot.time,
                               t_end=offer.snapshot.time + offer.h, vertex_ids=b["vertex_ids"][perm],
                               impulse=b["impulse_G_native"][perm])
        receipt = adapter.consume(X["ImpulseWindow"](X["SCHEMA"], offer.window_index, batch))
        G_rec = np.asarray(receipt["impulse"])
        assert np.array_equal(G_rec, b["impulse_G_native"]) and np.array_equal(G_rec @ S.rigid.R.T, Gw)
        assert receipt["window_index"] == k and not receipt["load_applied_to_geometry"]


# ------------------------------------------------------------------ 7 regression checks 
def _permuted(a, rng, shift=np.zeros(3)):
    """The reference mesh with its vertex rows and triangle rows permuted and every triangle
    rotated cyclically (same topology in stable IDs, same orientation)."""
    pv, pt = rng.permutation(NV), rng.permutation(NT)
    inv = np.empty(NV, np.int64)
    inv[pv] = np.arange(NV)
    T = inv[a["triangles"][pt]]
    T = np.take_along_axis(T, (rng.integers(0, 3, NT)[:, None] + np.arange(3)) % 3, axis=1)
    return (a["positions"][pv] + shift, T, a["vertex_ids"][pv], a["triangle_ids"][pt], a["pinned_two_rows"][pv])


def test_rows_permuted_within_one_topology_version():
    """Rows are window-local: the same topology (stable IDs, orientation) with permuted rows is
    accepted without a new topology_version, the tables follow the new rows and the per-vertex
    state follows the IDs; a real change in ID space without a new version is rejected."""
    init(threads=3)
    a = lf.reference_mesh()
    rng = np.random.default_rng(2)
    S = surface_alone(lf.live_spec(NV, NT, None))
    S.ingest(lf.reference_snapshot(a, step=0, time=0.0))
    G = np.zeros((S.nv, 3))
    G[:NV] = rng.normal(size=(NV, 3))
    S.JG_per.from_numpy(G)
    settle(S, 1e-3)
    shift = np.array([1e-3, -2e-3, 5e-4])                    # native displacement of the whole sheet
    X, T, vid, tid, pin = _permuted(a, rng, shift)
    rep = S.ingest(lf.snapshot(X, T, vid, tid, pinned=pin, step=1, time=1e-3))
    assert rep["reordered"] and not rep["rebuilt"] and S.reorders == 1
    assert rep["max_step"] == pytest.approx(np.linalg.norm(shift), rel=1e-9)
    assert rep["max_prediction_deviation"] == pytest.approx(np.linalg.norm(shift), rel=1e-9)   # v = 0: x_s carried by ID
    rows = S.rows_of(a["vertex_ids"])
    assert np.array_equal(S.vertex_ids[rows], a["vertex_ids"])
    assert np.array_equal(S.JG_cum.to_numpy()[rows], G[:NV])                 # cumulative impulse by ID
    R = surface_alone(lf.live_spec(NV, NT, None))                            # same geometry, original rows
    R.ingest(lf.reference_snapshot(a, positions=a["positions"] + shift))
    got, ref = S.side_arrays(), R.side_arrays()
    pt = np.argsort(tid)                                                     # triangle rows of R are ID order
    assert np.allclose(got["vert_n"][rows], ref["vert_n"], rtol=0, atol=1e-12)
    assert np.array_equal(got["vert_rim"][rows], ref["vert_rim"])
    assert np.allclose(got["face_n"][pt], ref["face_n"], rtol=0, atol=1e-15)
    assert got["edge_rim"].sum() == ref["edge_rim"].sum() == 204
    assert np.allclose(S.x_s.to_numpy()[rows], R.x_s.to_numpy()[:NV], rtol=0, atol=0)
    settle(S, 2e-3)
    T2 = T.copy()
    T2[0] = T2[0][[0, 2, 1]]                                 # one face flipped: not the same topology
    with pytest.raises(ValueError, match="without a topology_version"):
        S.ingest(lf.snapshot(X, T2, vid, tid, pinned=pin, step=2, time=2e-3))
    keep = np.arange(1, NT)                                  # one face removed, same version
    with pytest.raises(ValueError, match="without a topology_version"):
        S.ingest(lf.snapshot(X, T[keep], vid, tid[keep], pinned=pin, step=2, time=2e-3))
    S.ingest(lf.snapshot(X, T[keep], vid, tid[keep], pinned=pin, step=2, time=2e-3, topo=1))   # with a new version
    assert S.rebuilds == 3


def test_contact_with_rows_permuted_every_window():
    """F3 on the live mesh, 10 windows: rows shuffled every window (same topology) against the
    original rows. Equal up to round-off: the tie-break (lowest row) and the order of the per-vertex
    sums depend on the rows; compared per stable ID."""
    out = []
    for shuffle in (False, True):
        init(threads=1)
        a = lf.reference_mesh()
        rng = np.random.default_rng(4)
        cfg = _f3_mesh_cfg(lf.live_spec(NV, NT, None))
        pos, _ = f3_replay.initial_positions(cfg, 0.3)
        sim = Solver(cfg, pos)
        S, c = sim.contact.surface, sim.contact
        diag = Diagnostics(sim)
        diag()
        Gid = np.zeros((NV, 3))
        for k in range(10):
            if shuffle:
                X, T, vid, tid, pin = _permuted(a, rng)
                S.ingest(lf.snapshot(X, T, vid, tid, pinned=pin, step=k, time=k * 1e-3))
            else:
                S.ingest(lf.reference_snapshot(a, step=k, time=k * 1e-3))
            sim.advance(c.n_sub)
            b = S.last_batch()
            Gid[b["vertex_ids"]] += b["impulse_G"]               # reference IDs are 0..NV-1
        d = diag()
        out.append((Gid, sim.P.x.to_numpy(), d))
    (G0, x0, d0), (G1, x1, d1) = out
    assert d1["bal_rel"] < 1e-10 and d0["contact_active"] == d1["contact_active"] > 50
    assert np.abs(G1 - G0).max() <= 1e-9 * np.abs(G0).max(), np.abs(G1 - G0).max() / np.abs(G0).max()
    assert np.abs(x1 - x0).max() < 1e-12


def test_ingest_measures_and_version_counters():
    """max_step and max_prediction_deviation against a known motion; the batch keeps the
    snapshot's version (cloth_version) apart from the MPM window count (window_index)."""
    init(threads=3)
    X, T, vid, tid = _small()
    S = surface_alone(lf.live_spec(len(X), len(T), None))
    v = np.array([0.3, -0.1, 0.2])
    S.ingest(lf.snapshot(X, T, vid, tid, velocity=np.tile(v, (len(X), 1)), step=5, time=0.005))
    settle(S, 1e-3)
    b = S.last_batch()
    assert b["window_index"] == 0 and b["cloth_version"] == (1, 5, 0) and S.step_index[None] == 1
    delta = np.array([0.0, 0.0, 2e-4])                       # off the frozen-velocity prediction (native)
    rep = S.ingest(lf.snapshot(X + 1e-3 * v + delta, T, vid, tid, velocity=np.tile(v, (len(X), 1)), step=6, time=0.006))
    assert rep["max_step"] == pytest.approx(np.linalg.norm(1e-3 * v + delta), rel=1e-9)
    assert rep["max_prediction_deviation"] == pytest.approx(2e-4, rel=1e-9)
    with pytest.raises(RuntimeError, match="spans"):          # t_s is the window start on the solver clock
        settle(S, 5e-3)


def _function_bodies(path, cls):
    tree = ast.parse(pathlib.Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    out = {}
    for f in node.body:
        if isinstance(f, ast.FunctionDef):
            body = f.body[1:] if (f.body and isinstance(f.body[0], ast.Expr)
                                  and isinstance(f.body[0].value, ast.Constant)) else f.body
            out[f.name] = [ast.dump(x) for x in body]
    return out


def test_shared_surface_functions_have_the_stub_code():
    """vertex_at, plane_side, side and settle are copies of the stub's (the block reaches both
    surfaces through them): their code, docstrings aside, must stay identical ."""
    stub = _function_bodies(WORK / "mpm" / "contact" / "surface.py", "StubSurface")
    live = _function_bodies(WORK / "mpm" / "contact" / "live.py", "LiveSurface")
    for name in ("vertex_at", "plane_side", "side", "settle"):
        assert stub[name] == live[name], name
