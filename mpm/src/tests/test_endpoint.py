"""P2-U2 MPM endpoint : E1 fixed contact
bit for bit with the f3_replay stub; E2-E4 against the expectations written before the first
run ; offer and batch rejections (fail-stop, nothing advanced); the pre-registered stop
rules  on synthetic rows and in two runs; stable IDs with rows shuffled by the peer;
provenance, archive, verify, tamper detection and replay; the live_endpoint scene CLI with the
live cloth export; the XPBD mirror (optional): wire compatibility of the batch, the bridge's
refusal of a real source and the endpoint's refusal of a prescribed-source offer."""
import dataclasses
import json
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest
import taichi as ti

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "scenes"))
sys.path.insert(0, str(WORK / "tests"))
from mpm import Diagnostics, Solver, load_config, replay  # noqa: E402
from mpm.contact import xpbd_mesh  # noqa: E402
from coupling import (SCHEMA, EndpointStop, ImpulseWindow, LoadBatch, MPMEndpoint, PrescribedPeer,  # noqa: E402
                      SourceIdentity, StopRules, archive)
import f3_replay  # noqa: E402
import live_endpoint  # noqa: E402
import live_fixtures as lf  # noqa: E402
from test_live_surface import XPBD_WORK, xpbd  # noqa: E402,F401  (fixture: read-only XPBD mirror)

CONF = WORK / "scenes" / "configs"
F3M, LIVE = CONF / "f3_replay_n64.json", CONF / "live_endpoint_f3.json"
NV, NT = 52 * 52, 2 * 51 * 51
RIGID = xpbd_mesh.RigidMap(**{k: xpbd_mesh.proposed_transform()[k] for k in ("R", "c", "C")})
PARTICLE_KEYS = ("x", "v", "C", "F", "tau", "q", "vc", "alpha", "flag", "contact_state", "contact_face", "contact_dist")
BLOCK_KEYS = ("JP", "JG", "JG_attr", "G_on_pinned", "KG", "n_active", "n_pen", "missed", "rebuilds", "cand_overflow",
              "bin_overflow", "pen_max_run", "Rq", "n_cand")
SMALL = ["grid.n=24", "scene.block.size=[0.3,0.0625,0.3]"]      # pre-registered : 3 sample layers at n24


def init(threads=3):
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=threads, random_seed=0, log_level=ti.WARN)


def build(*sets, threads=3, vel=(0.0, 0.0, 0.0), shuffle=None, stop=None, stream="u2-test"):
    init(threads)
    cfg = load_config(str(LIVE), [*SMALL, *sets], live_endpoint.SCENE_DEFAULTS)
    a = lf.reference_mesh()
    pos, _ = f3_replay.initial_positions(cfg, 0.3)
    sim = Solver(cfg, pos)
    ep = MPMEndpoint(sim, scene="test", cfg=cfg, stop=stop)
    peer = PrescribedPeer(a["positions"], a["triangles"], a["vertex_ids"], a["triangle_ids"],
                          RIGID.vec_to_native(np.asarray(vel, float)), h=ep.H, stream_id=stream,
                          expected_source=ep.source, shuffle_rows=shuffle)
    return sim, ep, peer, a


def run(sim, ep, peer, W, frame_every=None):
    diag = Diagnostics(sim)
    diag()
    rows = []
    for k in range(W):
        msg = ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
        ep.acknowledge(peer.consume(msg))
        if frame_every and (k + 1) % frame_every == 0:
            rows.append(diag())
            ep.check_frame(rows[-1])
    return rows


def state(sim):
    out = {k: getattr(sim.P, k).to_numpy() for k in PARTICLE_KEYS}
    out.update({f"block.{k}": getattr(sim.contact, k).to_numpy() for k in BLOCK_KEYS})
    return out


def load_ratio(ep, sim):
    MgH = float(sim.P.m.to_numpy().sum()) * 9.81 * ep.H
    return np.array([-r["sumG_y"] / MgH for r in ep.ledger])


# ------------------------------------------------------------------ E1: bit for bit with the f3_replay stub
def test_e1_fixed_contact_equals_the_f3_replay_stub():
    W = 30
    init(threads=1)
    cfg = load_config(str(F3M), [*SMALL, "precision=f64", "contact.surface.pinned=all"], f3_replay.SCENE_DEFAULTS)
    pos, _ = f3_replay.initial_positions(cfg, 0.3)
    sim = Solver(cfg, pos)
    rec = replay.ReplayRecorder(sim.contact, W)
    sim.advance(W * sim.contact.n_sub)
    ref = state(sim)
    G_native = RIGID.vec_to_native(rec.G)
    J_native = RIGID.vec_to_native(rec.J_material)
    t_ns, counts = rec.t_ns.copy(), rec.counts.copy()

    sim, ep, peer, _ = build(threads=1)
    run(sim, ep, peer, W)
    got = state(sim)
    for k in ref:
        assert ref[k].tobytes() == got[k].tobytes(), k
    for k, r in enumerate(ep.records):
        # native batch = R^T stub G: equal values; the bytes may differ only in the sign of zeros (the kernel's
        # R^T under fast-math vs NumPy's G @ R), the world-frame physics above is equal byte for byte
        g, ref_g = r["impulse"], G_native[k]
        assert np.array_equal(g, ref_g), k
        diff = g.view(np.uint64) != ref_g.view(np.uint64)
        assert not g[diff].any() and not ref_g[diff].any(), k
        assert r["J_material"].tobytes() == J_native[k].tobytes(), k
        assert (r["t_start_ns"], r["t_end_ns"]) == tuple(t_ns[k]) and np.array_equal(r["counts"], counts[k])
        assert r["cloth_version"] == (1, k, 0) and r["delivered"]
    rcp = peer.receipts
    assert len(rcp) == W and all(np.array_equal(rc["impulse"], ep.records[k]["impulse"]) for k, rc in enumerate(rcp))
    assert ep.ledger[-1]["impulse_rel"] < 1e-12 and ep.stopped is None


# ------------------------------------------------------------------ E2-E4 
def test_e2_fixed_sheet_drop_separation_then_contact():
    sim, ep, peer, _ = build("scene.block.bottom_gap=0.05")
    run(sim, ep, peer, 150)
    nonzero = [bool(r["impulse"].any()) for r in ep.records]
    onset = nonzero.index(True)
    assert 55 <= onset <= 90, onset
    assert not any(nonzero[:onset])                                          # exactly zero before contact
    assert sum(r["sumG_y"] for r in ep.ledger[onset:]) < 0                    # the sand pushes the cloth down
    assert ep.stopped is None and ep.k == 150


def test_e3_sheet_moving_down_separates_then_recontacts():
    sim, ep, peer, _ = build(vel=(0.0, -0.3, 0.0))
    run(sim, ep, peer, 150)
    L = load_ratio(ep, sim)
    moved = ep.surface.x_s.to_numpy()[:NV] - RIGID.to_world(ep.snapshot0["positions"])
    assert np.allclose(moved, 150 * ep.H * np.array([0.0, -0.3, 0.0]), rtol=0, atol=1e-12)   # world down 
    assert np.abs(L[3:16]).max() < 0.01, L[3:16]                             # separating: no constraint, no push-out
    assert 0.5 <= L[100:150].mean() <= 1.5, L[100:150].mean()
    assert ep.stopped is None


def test_e4_horizontal_translation_drags_the_layer():
    sim, ep, peer, _ = build("contact.mu=0.8", vel=(0.2, 0.0, 0.0))
    rows = run(sim, ep, peer, 300, frame_every=1)                            # rows[k]: state at the end of window k
    assert sum(r["sumG_x"] for r in ep.ledger) < 0                           # friction pulls the cloth back
    vx = np.mean([r["momentum"][0] / r["mass"] for r in rows[150:300]])       # windows 150-299 (pre-registered)
    assert vx == pytest.approx(0.2, rel=0.15), vx
    L = load_ratio(ep, sim)
    assert 0.8 <= L[150:300].mean() <= 1.2, L[150:300].mean()
    # native x = world x (R maps x to x); the two sums are reduced in a different order, so compare to round-off
    assert np.allclose([r["sumG_native_x"] for r in ep.ledger], [r["sumG_x"] for r in ep.ledger], rtol=1e-12, atol=1e-18)
    assert ep.stopped is None


# ------------------------------------------------------------------ rejections (fail-stop, nothing advanced)
def _offer(o, snap=None, **kw):
    f = dict(schema=o.schema, stream_id=o.stream_id, source=o.source, window_index=o.window_index,
             snapshot=o.snapshot if snap is None else dataclasses.replace(o.snapshot, **snap),
             particle_dt=o.particle_dt, h=o.h)
    f.update(kw)
    return types.SimpleNamespace(**f)


FIRST = {"schema": dict(schema="xpbd-exchange-v0"),
         "source_prescribed": dict(source=SourceIdentity("u13-local-synthetic-fixture", "prescribed", 1)),
         "window_index": dict(window_index=1),
         "h_not_H": dict(h=0.0005, particle_dt=0.0001),
         "particle_dt": dict(particle_dt=0.00025),
         "start_time": dict(snap=dict(time=0.001)),
         "start_step": dict(snap=dict(step_index=5)),
         "nonfinite_snapshot": "S1"}
LATER = {"stream": dict(stream_id="another-stream"),
         "replayed_window": dict(window_index=0),
         "time_gap": dict(snap=dict(time=0.0011)),
         "epoch": dict(snap=dict(epoch=2)),
         "step_jump": dict(snap=dict(step_index=2))}


@pytest.mark.parametrize("case", list(FIRST) + list(LATER) + ["bin_capacity", "advance_without_offer"])
def test_offer_rejections_stop_without_advancing(case):
    sets = ["contact.bin_capacity=4096"] if case == "bin_capacity" else []
    sim, ep, peer, _ = build(*sets)
    if case in LATER:
        ep.acknowledge(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
    o = peer.open_window(h=ep.H, particle_dt=ep.dt)
    step0, ver0 = int(sim.step[None]), ep.surface.version()
    code = "S1" if case == "nonfinite_snapshot" else "S8"                    # a non-finite snapshot is S1
    with pytest.raises(EndpointStop, match=code):
        if case == "advance_without_offer":
            ep.advance()
        elif case == "nonfinite_snapshot":
            X = o.snapshot.positions.copy()
            X[7, 1] = np.nan
            ep.receive(_offer(o, dict(positions=X)))
        else:
            kw = dict(FIRST.get(case) or LATER.get(case) or {})
            ep.receive(_offer(o, kw.pop("snap", None), **kw))
    assert int(sim.step[None]) == step0 and ep.surface.version() == ver0 and ep.stopped["reason"].startswith(code)
    assert ep.surface._phase == "between"
    with pytest.raises(EndpointStop, match="stopped earlier"):
        ep.receive(o)                                                         # fail-stop: a new stream is needed


def test_peer_rejects_bad_batches_and_stays_retryable():
    sim, ep, peer, _ = build()
    o = peer.open_window(h=ep.H, particle_dt=ep.dt)
    msg = ep.step(o)
    b = msg.batch
    other = SourceIdentity("mpm-couple.other", "real_backend", 1)

    def with_(**kw):
        f = {k: getattr(b, k) for k in ("stream_id", "source", "cloth_version", "batch_id", "t_start", "t_end",
                                        "vertex_ids", "impulse")}
        f.update(kw)
        return ImpulseWindow(SCHEMA, 0, LoadBatch(**f))

    bad = [with_(stream_id="x"), with_(source=other), with_(cloth_version=(1, 1, 0)), with_(t_start=1e-4),
           with_(t_end=b.t_end * 2), with_(vertex_ids=np.r_[b.vertex_ids[:-1], 999999]),
           with_(vertex_ids=b.vertex_ids[:-1], impulse=b.impulse[:-1]), ImpulseWindow(SCHEMA, 1, b)]
    for m in bad:
        with pytest.raises(ValueError):
            peer.consume(m)
    receipt = peer.consume(msg)                                               # unchanged: the right batch passes
    # P2-U3: receipts are JSON-native (lists), like XPBD's; same check as U2 on the tuple of the version
    assert tuple(receipt["committed_version"]) == (1, 1, 0) and receipt["digest"]
    with pytest.raises(ValueError):                                           # the mirror refuses duplicates, NaN
        LoadBatch(**{**{k: getattr(b, k) for k in ("stream_id", "source", "cloth_version", "batch_id", "t_start",
                                                   "t_end")}, "vertex_ids": np.r_[b.vertex_ids[:-1], b.vertex_ids[0]],
                     "impulse": b.impulse})
    with pytest.raises(ValueError):
        LoadBatch(**{**{k: getattr(b, k) for k in ("stream_id", "source", "cloth_version", "batch_id", "t_start",
                                                   "t_end", "vertex_ids")}, "impulse": np.where(b.impulse == 0, np.nan, b.impulse)})


# ------------------------------------------------------------------ stop rules
def _row(**kw):
    r = dict(guard_new=0, impulse_finite=True, n_active=100, n_pen_active=0, pen_max_run=0.0, impulse_rel=0.0)
    r.update(kw)
    return r


def test_stop_rules_on_synthetic_rows():
    lim = 0.03
    assert StopRules(lim).window(_row(guard_new=1))[0].startswith("S1")
    assert StopRules(lim).window(_row(impulse_finite=False))[0].startswith("S1")
    assert StopRules(lim).period(2, 0).startswith("S2")
    r = StopRules(lim)
    assert [r.period(0, m) for m in (1, 1, 0, 1, 1)] == [None] * 5            # a clean window resets
    assert r.period(0, 1).startswith("S3")
    r = StopRules(lim)
    reason, warns = r.ingest(dict(max_prediction_deviation=0.6 * lim))
    assert reason is None and warns and r.ingest(dict(max_prediction_deviation=None)) == (None, [])
    assert r.ingest(dict(max_prediction_deviation=1.01 * lim))[0].startswith("S4")
    r = StopRules(lim)
    seq = [_row(n_pen_active=1), _row(n_pen_active=1), _row(n_active=10, n_pen_active=5), _row(n_pen_active=1),
           _row(n_pen_active=1)]
    assert [r.window(x)[0] for x in seq] == [None] * 5                        # 1 % > 0.5 %; n_active < 20 skipped
    assert r.window(_row(n_pen_active=1))[0].startswith("S5")
    assert StopRules(lim).window(_row(pen_max_run=1.1 * lim))[0].startswith("S5")
    assert StopRules(lim).window(_row(impulse_rel=2e-9))[0].startswith("S6")
    assert StopRules(lim).window(_row(impulse_rel=float("inf")))[0].startswith("S6")
    assert StopRules(lim).frame(dict(nonfinite=0, bal_rel=2e-10)).startswith("S7")
    assert StopRules(lim).frame(dict(nonfinite=1, bal_rel=0.0)).startswith("S1")
    assert StopRules(lim).frame(dict(nonfinite=0, bal_rel=1e-14)) is None
    with pytest.raises(KeyError):
        StopRules(lim, missed=1)


class _TeleportPeer(PrescribedPeer):
    """Moves the sheet 5 cm off its constant velocity after window `at` (not KinematicSurfaceBridge)."""
    at = 2

    def consume(self, msg):
        r = super().consume(msg)
        if r["window_index"] == self.at:
            self._S = dataclasses.replace(self._S, positions=self._S.positions + [0.0, 0.0, 0.05])
        return r


def test_stop_on_prediction_deviation_and_on_overflow_keeps_the_prefix(tmp_path):
    sim, ep, peer, a = build()
    peer.__class__ = _TeleportPeer
    with pytest.raises(EndpointStop, match="S4"):
        run(sim, ep, peer, 10)
    assert ep.k == 3 and ep.stopped["window"] == 3 and [r["delivered"] for r in ep.records] == [True] * 3
    assert len(peer.receipts) == 3 and int(sim.step[None]) == 3 * ep.n_sub   # stopped at the read: no substep on it
    sim, ep, peer, _ = build("contact.cand_capacity=10")
    with pytest.raises(EndpointStop, match="S2"):
        run(sim, ep, peer, 3)
    assert ep.k == 0 and ep.records == [] and ep.ledger[0]["stop"].startswith("S2")
    assert int(sim.step[None]) == 0                                           # stopped before any substep 
    rep = archive.write(tmp_path / "stopped", ep)                             # the diagnosable state is archived
    v = archive.verify(tmp_path / "stopped")
    assert rep["windows"] == 0 and not rep["complete"] and v["delivered"] == 0 and v["stop_reason"].startswith("S2")


# ------------------------------------------------------------------ stable IDs, archive, provenance, replay
def test_rows_shuffled_by_the_peer_give_the_same_impulse_per_id():
    out = []
    for shuffle in (None, 7):
        sim, ep, peer, a = build(shuffle=shuffle)
        run(sim, ep, peer, 10)
        G = np.zeros((10, NV, 3))
        for k, rc in enumerate(peer.receipts):
            G[k][rc["vertex_ids"]] = rc["impulse"]                           # reference IDs are 0 .. NV-1
        out.append((G, ep))
    (G0, e0), (G1, e1) = out
    assert e1.ledger[1]["reordered"] and not e0.ledger[1]["reordered"]
    assert np.abs(G1 - G0).max() <= 1e-9 * np.abs(G0).max()


def test_provenance_archive_verify_tamper_and_replay(tmp_path):
    sim, ep, peer, a = build()
    run(sim, ep, peer, 12)
    reg = ep.registration
    assert reg["source"]["kind"] == "real_backend" and reg["source"]["version"] == int(reg["code_digest"][:8], 16)
    assert reg["peer"]["stream_id"] == "u2-test" and reg["peer"]["n_vertices"] == NV
    assert all(r["step_after"] - r["step_before"] == ep.n_sub for r in ep.ledger)
    assert max(r["impulse_rel"] for r in ep.ledger) < 1e-12
    d = tmp_path / "arch"
    archive.write(d, ep)
    rep = archive.verify(d, work=WORK)
    assert rep["windows"] == rep["delivered"] == 12 and rep["code_unchanged"] and rep["chain_final"] == ep.chain
    assert rep["complete"] and rep["stop_reason"] is None
    got = archive.refeed(d)                                                   # peer rebuilt from the archive alone
    assert [r["digest"] for r in got] == [r["digest"] for r in peer.receipts]
    ep.records[5]["cloth_version"] = (1, 5, 1)                                # one topology per archive
    with pytest.raises(ValueError, match="one topology"):
        archive.write(tmp_path / "two", ep)
    ep.records[5]["cloth_version"] = (1, 5, 0)
    z = dict(np.load(d / "batches.npz"))
    z["impulse"][5, 100, 1] += 1e-12                                          # tamper with one value
    (d / "batches.npz").unlink()
    np.savez(d / "batches.npz", **z)
    with pytest.raises(ValueError):
        archive.verify(d)
    # ledger tampered consistently with the manifest's file digest: the substep invariant catches it 
    d2 = tmp_path / "arch2"
    archive.write(d2, ep)
    led = (d2 / "ledger.csv").read_text().splitlines()
    cols = led[0].split(",")
    cells = led[3].split(",")
    cells[cols.index("step_after")] = str(int(cells[cols.index("step_before")]) + 1)
    led[3] = ",".join(cells)
    (d2 / "ledger.csv").write_text("\n".join(led) + "\n")
    man = json.loads((d2 / "manifest.json").read_text())
    man["files"]["ledger.csv"]["sha256"] = coupling_sha(d2 / "ledger.csv")
    (d2 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="substeps"):
        archive.verify(d2)
    man["files"]["ledger.csv"]["sha256"] = coupling_sha(d2 / "ledger.csv")
    man["delivered"][0] = False                                               # flags differ from the batches
    (d2 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError):
        archive.verify(d2)


def coupling_sha(path):
    import hashlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_archive_states_shuffled_refeed_reject_pending_nonfinite_and_code(tmp_path):
    # shuffled rows: receipt digests are by stable ID, so the archive alone re-feeds the stream 
    sim, ep, peer, _ = build(shuffle=7)
    run(sim, ep, peer, 6)
    archive.write(tmp_path / "shuf", ep)
    assert [r["digest"] for r in archive.refeed(tmp_path / "shuf")] == [r["digest"] for r in peer.receipts]
    # a window the peer rejected is not delivered and not re-fed; a read but not advanced window is pending (R1)
    sim, ep, peer, _ = build()
    run(sim, ep, peer, 3)
    ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(EndpointStop):
        ep.reject("synthetic")
    rep = archive.write(tmp_path / "rej", ep)
    v = archive.verify(tmp_path / "rej")
    assert not rep["complete"] and v["delivered"] == 3 and len(archive.refeed(tmp_path / "rej")) == 3
    sim, ep, peer, _ = build()
    ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))
    rep = archive.write(tmp_path / "pending", ep)
    assert not rep["complete"] and archive.verify(tmp_path / "pending")["pending_window"] == 0
    # a non-finite batch (S1) is archived as not delivered and the archive verifies 
    sim, ep, peer, _ = build()
    run(sim, ep, peer, 2)
    orig = ep.surface.last_batch

    def bad():
        b = orig()
        for key in ("impulse_G", "impulse_G_native"):
            b[key] = b[key].copy()
            b[key][0, 0] = np.nan
        return b

    ep.surface.last_batch = bad
    with pytest.raises(EndpointStop, match="S1"):
        ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    archive.write(tmp_path / "nan", ep)
    v = archive.verify(tmp_path / "nan")
    assert v["windows"] == 3 and v["delivered"] == 2 and v["stop_reason"].startswith("S1")
    # code_unchanged covers the registered scene files : a copied work tree with one scene file changed
    import shutil
    w = tmp_path / "work"
    for sub in ("mpm", "coupling", "scenes"):
        shutil.copytree(WORK / sub, w / sub, ignore=shutil.ignore_patterns("__pycache__"))
    init()
    cfg = load_config(str(LIVE), SMALL, live_endpoint.SCENE_DEFAULTS)
    sim = Solver(cfg, f3_replay.initial_positions(cfg, 0.3)[0])
    ep = MPMEndpoint(sim, scene="test", cfg=cfg, scene_files=[WORK / "scenes" / "live_endpoint.py"])
    a = lf.reference_mesh()
    peer = PrescribedPeer(a["positions"], a["triangles"], a["vertex_ids"], a["triangle_ids"], np.zeros(3), h=ep.H,
                          stream_id="u2-test", expected_source=ep.source)
    run(sim, ep, peer, 1)
    archive.write(tmp_path / "code", ep)
    assert archive.verify(tmp_path / "code", work=w)["code_unchanged"]
    (w / "scenes" / "live_endpoint.py").write_text((w / "scenes" / "live_endpoint.py").read_text() + "\n# changed\n")
    assert not archive.verify(tmp_path / "code", work=w)["code_unchanged"]


def test_endpoint_drives_the_window_as_solver_advance():
    """The endpoint's window (begin_period, n_sub x _substep, end_period) against sim.advance on the same live
    surface fed directly: particle state, solver clock and step, period position ."""
    out = []
    for direct in (True, False):
        sim, ep, peer, a = build(threads=1)
        for k in range(8):
            o = peer.open_window(h=ep.H, particle_dt=ep.dt)
            if direct:
                sim.contact.surface.ingest(o.snapshot)
                sim.advance(ep.n_sub)
                b = sim.contact.surface.last_batch()
                peer.consume(ImpulseWindow(SCHEMA, k, LoadBatch(
                    stream_id=o.stream_id, source=ep.source, cloth_version=o.snapshot.version, batch_id=f"d/{k}",
                    t_start=o.snapshot.time, t_end=o.snapshot.time + o.h, vertex_ids=b["vertex_ids"],
                    impulse=b["impulse_G_native"])))
            else:
                ep.acknowledge(peer.consume(ep.step(o)))
        st = state(sim)
        st.update(time=np.float64(sim.time[None]), step=np.int64(sim.step[None]), pos=np.int64(sim._period_pos))
        out.append(st)
    for k in out[0]:
        assert out[0][k].tobytes() == out[1][k].tobytes(), k


# ------------------------------------------------------------------ scene CLI and live cloth export
def test_live_endpoint_scene_cli_exports_live_cloth(tmp_path):
    cmd = [sys.executable, "-B", str(WORK / "scenes" / "live_endpoint.py"), "--set", "grid.n=16", "--set",
           "scene.windows=20", "--set", "scene.windows_per_frame=10", "--set", "scene.block.size=[0.3,0.1,0.3]",
           "--set", "scene.shuffle_rows=3", "--out", str(tmp_path), "--threads", "3"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    run_dir = tmp_path / "live_endpoint_f3"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "ok" and s["windows_delivered"] == 20 and s["archive"]["verify"]["delivered"] == 20
    man = json.loads((run_dir / "endpoint" / "manifest.json").read_text())
    assert man["complete"] and man["run_anchor"]["run_dir"] == str(run_dir) and "run_info.json_sha256" in man["run_anchor"]
    assert any(k.startswith("scenes/") for k in json.loads((run_dir / "endpoint" / "provenance.json").read_text())
               ["registration"]["code_files"])
    mesh = np.load(run_dir / "cloth" / "mesh.npz")
    assert mesh["triangles"].shape == (NT, 3) and mesh["vertex_ids"].shape == (NV,)
    f2 = np.load(run_dir / "cloth" / "00002.npz")
    assert {"vertex_ids", "triangle_ids", "triangles", "x", "v", "JG", "version"} <= set(f2.files)
    assert tuple(f2["version"]) == (1, 19, 0)                                 # frame after window 19, before any read
    info = json.loads((run_dir / "run_info.json").read_text())
    assert "vertex_ids" in info["export"]["cloth"]["frame_fields"]
    assert "same-numbered" in info["export"]["frame_fields"]["contact_face"]["meaning"]
    # contact_face indexes the same-numbered cloth frame's triangles (rows shuffled every window): the distance from
    # a constrained particle to that triangle equals |contact_dist|; with the first window's rows it does not (R6)
    fr = np.load(run_dir / "frames" / "00002.npz")
    on = np.flatnonzero(fr["contact_state"] == 2)
    assert len(on) > 20
    X = f2["x"]

    def dist(rows):
        return np.array([_point_triangle_distance(fr["x"][p].astype(float), *X[rows[fr["contact_face"][p]]]) for p in on])

    err = np.abs(dist(f2["triangles"]) - np.abs(fr["contact_dist"][on].astype(float)))
    assert err.max() < 1e-4, err.max()
    wrong = np.abs(dist(mesh["triangles"]) - np.abs(fr["contact_dist"][on].astype(float)))
    assert (wrong > 1e-3).mean() > 0.5


def _point_triangle_distance(p, a, b, c):
    """Exact point-triangle distance (NumPy reference, Ericson 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return np.linalg.norm(ap)
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return np.linalg.norm(bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return np.linalg.norm(p - (a + d1 / (d1 - d3) * ab))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return np.linalg.norm(cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return np.linalg.norm(p - (a + d2 / (d2 - d6) * ac))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return np.linalg.norm(p - (b + (d4 - d3) / ((d4 - d3) + (d5 - d6)) * (c - b)))
    den = 1.0 / (va + vb + vc)
    return np.linalg.norm(p - (a + ab * vb * den + ac * vc * den))


def test_live_endpoint_scene_cli_stops_with_code_4_and_a_summary(tmp_path):
    """An endpoint stop (S2 before the first substep) in the scene: exit code 4, summary and archive written (B2)."""
    cmd = [sys.executable, "-B", str(WORK / "scenes" / "live_endpoint.py"), "--set", "grid.n=16", "--set",
           "scene.windows=5", "--set", "contact.cand_capacity=10", "--set", "scene.block.size=[0.3,0.1,0.3]",
           "--out", str(tmp_path), "--threads", "3", "--no-frames"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 4, p.stdout[-2000:] + p.stderr[-2000:]
    run_dir = tmp_path / "live_endpoint_f3"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"].startswith("stopped: S2") and s["windows_delivered"] == 0
    man = json.loads((run_dir / "endpoint" / "manifest.json").read_text())
    assert not man["complete"] and man["stop_reason"].startswith("S2") and s["archive"]["verify"]["windows"] == 0


# ------------------------------------------------------------------ XPBD mirror (optional, read-only)
def test_xpbd_wire_compatibility_and_the_real_source_gap(xpbd):
    X = xpbd
    sim, ep, peer, a = build()
    o = peer.open_window(h=ep.H, particle_dt=ep.dt)
    msg = ep.step(o)
    b = msg.batch
    real = X["SourceIdentity"](ep.source.source_id, ep.source.kind, ep.source.version)
    xb = X["LoadBatch"](stream_id=b.stream_id, source=real, cloth_version=b.cloth_version, batch_id=b.batch_id,
                        t_start=b.t_start, t_end=b.t_end, vertex_ids=b.vertex_ids, impulse=b.impulse)
    X["ImpulseWindow"](X["SCHEMA"], msg.window_index, xb)                     # same wire format
    S = o.snapshot
    xs = X["ClothSnapshot"](S.positions, S.triangles, S.vertex_ids, S.triangle_ids, S.velocity, np.zeros(NV),
                            np.ones(NV, bool), S.epoch, S.step_index, S.topology_version, S.time, S.dt)
    rows = xb.validate_for(xs, stream_id=b.stream_id, source=real, h=ep.H, capacity=NV)
    assert np.array_equal(np.sort(rows), np.arange(NV))                       # dense, every ID once
    with pytest.raises(ValueError, match="prescribed"):                       # the gap: the bridge refuses real loads
        X["KinematicSurfaceBridge"](xs, source=real, stream_id="u2-test", h=ep.H)
    # an XPBD offer naming a prescribed load source is refused by the endpoint, nothing advances
    sim2, ep2, _, _ = build()
    pres = X["SourceIdentity"]("u13-local-synthetic-fixture", "prescribed", 1)
    xs0 = dataclasses.replace(xs, step_index=0, time=0.0)
    adapter = X["ExchangeAdapter"](X["KinematicSurfaceBridge"](xs0, source=pres, stream_id="u13", h=ep2.H), pres)
    xo = adapter.open_window(h=ep2.H, particle_dt=ep2.dt)
    with pytest.raises(EndpointStop, match="registered"):
        ep2.receive(xo)
    assert int(sim2.step[None]) == 0 and ep2.surface.version() is None


# ------------------------------------------------------------------ phases and the peer's terminal failure 
def test_endpoint_phases_reject_and_receipts():
    sim, ep, peer, _ = build()
    msg = ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    step1 = int(sim.step[None])
    with pytest.raises(EndpointStop, match="phase"):
        ep.advance()                                                          # a second advance of the same window
    assert int(sim.step[None]) == step1                                       # refused before any substep
    sim, ep, peer, _ = build()
    msg = ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(EndpointStop, match="receipt"):
        ep.acknowledge(dict(window_index=3))
    sim, ep, peer, _ = build()
    ep.acknowledge(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
    ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(EndpointStop, match="peer rejected window 1"):
        ep.reject("synthetic refusal")
    # P2-U3: the ledger keeps the content digest (U2's meaning of receipt["digest"]) computed by the endpoint, and
    # receipt_digest is now the canonical-JSON digest of the receipt as received
    from coupling import wire
    assert ep.ledger[-1]["peer_rejected"] and ep.ledger[0]["receipt_content_digest"] == peer.receipts[0]["digest"]
    assert ep.ledger[0]["receipt_digest"] == wire.json_digest(peer.receipts[0])


class _BadCommitPeer(PrescribedPeer):
    def _next_state(self, C, b):
        return dataclasses.replace(super()._next_state(C, b), step_index=C.step_index + 2)


def test_peer_end_checks_poison_the_stream():
    sim, ep, peer, _ = build()
    peer.__class__ = _BadCommitPeer
    msg = ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(ValueError, match="poisoned"):
        peer.consume(msg)
    assert peer.failed
    with pytest.raises(RuntimeError, match="aborted"):
        peer.open_window(h=ep.H, particle_dt=ep.dt)
    with pytest.raises(RuntimeError, match="aborted"):
        peer.consume(msg)                                                     # no retry after the commit (R5)
