"""P2-U3 adapter : the wire format; the XPBD
service process (spawn, read-only XPBD import, no Taichi / mpm, pinned files); E1 through the
transport bit for bit with the f3_replay stub; a uniform translation through the transport bit
for bit with the U2 in-process stand-in; the replay into the loop (the archive's frozen inputs
through the transport and offline, byte for byte); fault injection (both sides stop, prefix kept,
nothing retried); handshake failures before the first window; the endpoint phase machine with
the delivered phase; archive compatibility; the coupled_u13 scene CLI. Two local processes, n24,
<= 30 windows (A tier)."""
import copy
import dataclasses
import json
import multiprocessing as mp
import os
import pathlib
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "scenes"))
sys.path.insert(0, str(WORK / "tests"))
from mpm import Solver, load_config, replay  # noqa: E402
from coupling import EndpointStop, MPMEndpoint, archive, wire, xpbd_service  # noqa: E402
from coupling.driver import (PINNED_XPBD, PUBLISHED_MESH, CoupledDriver, DriverError, first_mismatch,  # noqa: E402
                             inputs_from_vectors, offline_receipts, reconcile, replay_archive, run_offline,
                             run_transport, snapshot_spec)
from coupling.link import DEFAULT_TIMEOUTS, PeerError, XPBDLink  # noqa: E402
import f3_replay  # noqa: E402
import live_endpoint  # noqa: E402
import live_fixtures as lf  # noqa: E402
import test_endpoint as te  # noqa: E402
from test_live_surface import XPBD_WORK  # noqa: E402

HAVE_XPBD = (XPBD_WORK / "project" / "coupling" / "registered_kinematic.py").exists()
needs_xpbd = pytest.mark.skipif(not HAVE_XPBD, reason=f"XPBD mirror not found at {XPBD_WORK} (set MPM_XPBD_WORK)")
FILE_MESH = (pathlib.Path(PUBLISHED_MESH["path"]) / "mesh.npz").exists()
FAST = dict(DEFAULT_TIMEOUTS, idle=120.0)
VECTORS = WORK.parent / "data" / "contract-vectors.json"
VECTORS_SHA256 = "620c0ed9a65af177d5ed28cf39d52f2d97d06b4c990b01604ae9248f3940658d"   # XPBD interface specification


def _pycache_dirs():
    return sorted(str(p) for p in XPBD_WORK.rglob("__pycache__")) if HAVE_XPBD else []


PYCACHE_BEFORE = _pycache_dirs()


def build_h(*sets, threads=3):
    te.init(threads)
    cfg = load_config(str(te.LIVE), [*te.SMALL, *sets], live_endpoint.SCENE_DEFAULTS)
    pos, _ = f3_replay.initial_positions(cfg, 0.3)
    sim = Solver(cfg, pos)
    return sim, MPMEndpoint(sim, scene="test", cfg=cfg, handshake=True)


def spec_for(ep, vel=(0.0, 0.0, 0.0)):
    if FILE_MESH:
        return snapshot_spec(ep, velocity_world=vel), None
    return snapshot_spec(ep, velocity_world=vel, mesh="inline"), lf.reference_mesh(published=False)


def drive(ep, W, tmp, vel=(0.0, 0.0, 0.0), *, timeouts=None, fault=None, driver_fault=None, hooks=None, **kw):
    spec, mesh = spec_for(ep, vel)
    drv = CoupledDriver(ep, snapshot=spec, mesh_arrays=mesh, log_dir=tmp / "coupling", xpbd_root=XPBD_WORK,
                        timeouts=dict(FAST, **(timeouts or {})), fault=fault, driver_fault=driver_fault,
                        expect_mesh=kw.pop("expect_mesh", lf.reference_mesh()), **kw)
    return drv.run(W, **(hooks or {})), drv


def service_log(tmp):
    d = tmp / "coupling" / "xpbd_service"
    rows = [json.loads(x) for x in (d / "receipts.jsonl").read_text().splitlines()] if (d / "receipts.jsonl").exists() else []
    final = json.loads((d / "final.json").read_text()) if (d / "final.json").exists() else None
    return rows, final


# ------------------------------------------------------------------ 1. wire format (no process)
def _frame(**over):
    meta = dict(x=-0.0, t=0.001, nested=dict(a=[1, 2.5, None, True]))
    arrays = dict(p=np.array([[0.1, -0.0, 3.0]], "<f8"), i=np.arange(4, dtype="<i8"), b=np.array([True, False]))
    return wire.encode("OFFER", over.get("seq", 0), meta, arrays), meta, arrays


def _raw(head: str, payload=b""):
    h = head.encode()
    return wire.MAGIC + len(h).to_bytes(4, "big") + h + payload


def test_wire_roundtrip_and_rejections():
    buf, meta, arrays = _frame()
    fr = wire.decode(buf, expect_seq=0, modes=wire.FROM_SERVICE)
    assert fr.mode == "OFFER" and fr.meta == meta and set(fr.arrays) == set(arrays)
    for k, a in arrays.items():
        assert fr.arrays[k].tobytes() == a.tobytes() and fr.arrays[k].dtype == a.dtype and not fr.arrays[k].flags.writeable
    assert np.signbit(fr.meta["x"]) and np.signbit(fr.arrays["p"][0, 1])                  # -0.0 survives
    assert wire.encode("OFFER", 0, meta, arrays) == buf                                     # deterministic bytes
    good_head = buf[8:8 + int.from_bytes(buf[4:8], "big")].decode()
    payload = buf[8 + len(good_head):]
    bad = {
        "magic": b"XXXX" + buf[4:],
        "header_length": buf[:4] + (10 ** 9).to_bytes(4, "big") + buf[8:],
        "non_canonical_header": _raw(good_head.replace('"seq":0', '"seq": 0'), payload),
        "negative_zero_int": _raw(good_head.replace('"seq":0', '"seq":-0'), payload),
        "duplicate_key": _raw(good_head.replace('"seq":0', '"seq":0,"seq":0'), payload),
        "nan": _raw(good_head.replace('"t":0.001', '"t":NaN'), payload),
        "overflow_number": _raw(good_head.replace('"t":0.001', '"t":1e999'), payload),
        "extra_header_key": _raw(good_head[:-1] + ',"zz":1}', payload),
        "wire_version": _raw(good_head.replace(wire.WIRE, "mpm-xpbd-wire-v0"), payload),
        "sha": buf[:-1] + bytes([buf[-1] ^ 1]),
        "payload_extra": buf + b"\0",
        "payload_short": buf[:-1],
        "dtype": _raw(good_head.replace('"<i8"', '"<i4"'), payload),
    }
    for name, b in bad.items():
        with pytest.raises(wire.WireError):
            wire.decode(b)
        assert name
    with pytest.raises(wire.WireError, match="seq"):
        wire.decode(buf, expect_seq=1)
    with pytest.raises(wire.WireError, match="mode"):
        wire.decode(buf, modes=("RECEIPT",))
    with pytest.raises(wire.WireError):
        wire.encode("OFFER", 0, {}, dict(f=np.zeros(2, np.float32)))                        # no silent cast
    for v in [(1, 2), np.float64(1.0), float("inf"), {1: 2}]:
        with pytest.raises(wire.WireError):
            wire.canonical_json(dict(v=v))
    assert wire.is_canonical(wire.canonical_json(meta)) and not wire.is_canonical(json.dumps(meta))


def test_wire_oversize_frame_breaks_the_channel():
    """recv_bytes(maxlength) refuses a frame longer than max_frame (POSIX: after reading only the length; Windows:
    after the first 128 bytes of the message): the channel is terminal .
    the frame (about 1.8 KB) exceeds max_frame (1 KiB) but stays well inside the pipe buffer of every platform,
    so sending it before receiving it in the same thread cannot block. A Windows multiprocessing pipe holds
    BUFSIZE = 8 KiB and a longer write waits until the other end reads (Linux socketpair buffers are 208 KiB)."""
    from multiprocessing import connection
    a, b = mp.Pipe(duplex=True)
    tx, rx = wire.Channel(a), wire.Channel(b, max_frame=1024)
    n = tx.send("OFFER", {}, dict(x=np.zeros(200)))
    assert 1024 < n <= connection.BUFSIZE // 2, n
    with pytest.raises(wire.WireError, match="max_frame"):
        rx.recv(1.0)
    assert rx.broken
    with pytest.raises(wire.WireClosed, match="out of step"):
        rx.recv(1.0)
    a.close()
    b.close()


@needs_xpbd
def test_canonical_digest_is_the_xpbd_rule():
    X, _ = xpbd_service.import_xpbd(XPBD_WORK, PINNED_XPBD)
    rk = sys.modules["project.coupling.registered_kinematic"]
    sim, ep = build_h()
    assert rk._digest(json.loads(ep.registration_frozen)) == ep.registration_digest == wire.json_digest(
        json.loads(ep.registration_frozen))
    assert rk._json(json.loads(ep.registration_frozen)) == ep.registration_frozen


# ------------------------------------------------------------------ 2. the service process
@needs_xpbd
def test_service_hello_imports_only_pinned_xpbd_and_no_taichi(tmp_path):
    link = XPBDLink(dict(xpbd_root=str(XPBD_WORK), expected_sha256=PINNED_XPBD, log_dir=str(tmp_path), fault={}), FAST)
    hello = link.start()
    rec = link.stop("test")
    assert hello["forbidden_modules_at_start"] == [] and hello["forbidden_modules_after_import"] == []
    assert hello["flags_B"] and hello["dont_write_bytecode"] and hello["pid"] != __import__("os").getpid()
    files = hello["xpbd_files"]
    for rel in ("project/coupling/registered_kinematic.py", "project/coupling/contracts.py",
                "project/coupling/exchange.py", "project/coupling/loads.py"):
        assert files[rel] == PINNED_XPBD[rel]                                   # the imported pinned subset
    assert all(k.startswith("project/") for k in files)
    assert rec["bye"]["reason"].startswith("STOP") and rec["exitcode"] == 0 and not rec["terminated"]
    # a wrong pin or a pin set without the imported core files: the service refuses to start
    for pins in ({**PINNED_XPBD, "project/coupling/loads.py": "0" * 64},
                 {"project/coupling/kinematic_surface.py": PINNED_XPBD["project/coupling/kinematic_surface.py"]}):
        link = XPBDLink(dict(xpbd_root=str(XPBD_WORK), expected_sha256=pins, fault={}), FAST)
        with pytest.raises(PeerError, match="hello"):
            link.start()
        assert link.stop("test")["exitcode"] == 0
    sim, ep = build_h()
    with pytest.raises(ValueError, match="empty"):
        CoupledDriver(ep, snapshot=spec_for(ep)[0], expected_sha256={})


# ------------------------------------------------------------------ 3-5. E1 and translation through the transport, replay
@pytest.fixture(scope="module")
def e1_run(tmp_path_factory):
    """E1 through the transport: fixed sheet, single thread, 30 windows; archive kept for the replay."""
    if not HAVE_XPBD:
        pytest.skip("XPBD mirror not found")
    W = 30
    te.init(threads=1)
    cfg = load_config(str(te.F3M), [*te.SMALL, "precision=f64", "contact.surface.pinned=all"], f3_replay.SCENE_DEFAULTS)
    pos, _ = f3_replay.initial_positions(cfg, 0.3)
    sim = Solver(cfg, pos)
    rec = replay.ReplayRecorder(sim.contact, W)
    sim.advance(W * sim.contact.n_sub)
    ref = dict(state=te.state(sim), G_native=te.RIGID.vec_to_native(rec.G), J_native=te.RIGID.vec_to_native(rec.J_material))
    tmp = tmp_path_factory.mktemp("e1")
    sim, ep = build_h(threads=1)
    result, drv = drive(ep, W, tmp)
    archive.write(tmp / "endpoint", ep)
    return dict(ref=ref, got=te.state(sim), result=result, ep=ep, tmp=tmp, records=ep.records, ledger=ep.ledger)


@needs_xpbd
def test_e1_through_the_transport_equals_the_f3_replay_stub(e1_run):
    r, ref, ep = e1_run["result"], e1_run["ref"], e1_run["ep"]
    assert r["status"] == "ok" and r["windows_acknowledged"] == 30 and ep.stopped is None
    for k in ref["state"]:
        assert ref["state"][k].tobytes() == e1_run["got"][k].tobytes(), k
    for k, rr in enumerate(e1_run["records"]):
        g, ref_g = rr["impulse"], ref["G_native"][k]
        assert np.array_equal(g, ref_g), k                                     # values equal; zero signs as in U2 E1
        diff = g.view(np.uint64) != ref_g.view(np.uint64)
        assert not g[diff].any() and not ref_g[diff].any(), k
        assert rr["J_material"].tobytes() == ref["J_native"][k].tobytes() and rr["acknowledged"]
    # handshake and receipts: the XPBD digest of the frozen registration, the service log = the MPM ledger
    assert r["bound"]["registration_digest"] == ep.registration_digest == r["registration_digest"]
    assert ep.handshake["registration_confirmed"]["digest"] == ep.registration_digest and ep.handshake["peer_confirmed"]
    rows, final = service_log(e1_run["tmp"])
    assert [x["receipt_digest"] for x in rows] == [x["receipt_digest"] for x in e1_run["ledger"]]
    assert final["committed_windows"] == 30 and final["reason"].startswith("STOP") and r["xpbd_stop"]["exitcode"] == 0
    rc = reconcile(e1_run["tmp"] / "endpoint", e1_run["tmp"] / "coupling" / "xpbd_service")
    assert rc["consistent"] and rc["extra_xpbd_windows"] == 0
    v = archive.verify(e1_run["tmp"] / "endpoint")
    assert v["complete"] and v["acknowledged"] == 30
    assert r["bound"]["snapshot"]["mesh_binding"] == ("file" if FILE_MESH else "inline")
    assert r["bound"]["snapshot"]["topology_version"] == (1 if FILE_MESH else 0)          # manifest's 


@pytest.fixture(scope="module")
def translation_run(tmp_path_factory):
    """Uniform translation 0.2 m/s, mu = 0.8, 30 windows, single thread: the U2 in-process stand-in and the
    XPBD registered bridge through the transport, with the offered positions of every window."""
    if not HAVE_XPBD:
        pytest.skip("XPBD mirror not found")
    W, vel = 30, (0.2, 0.0, 0.0)
    sim, ep, peer, _ = te.build("contact.mu=0.8", threads=1, vel=vel)
    pos_u2 = []
    for k in range(W):
        o = peer.open_window(h=ep.H, particle_dt=ep.dt)
        pos_u2.append(np.array(o.snapshot.positions))
        ep.acknowledge(peer.consume(ep.step(o)))
    u2 = dict(state=te.state(sim), impulse=[r["impulse"].copy() for r in ep.records], positions=pos_u2)
    tmp = tmp_path_factory.mktemp("translation")
    sim, ep = build_h("contact.mu=0.8", threads=1)
    pos_u3 = []
    result, _ = drive(ep, W, tmp, vel, hooks=dict(on_received=lambda k: pos_u3.append(np.array(ep._offer.snapshot.positions))))
    archive.write(tmp / "endpoint", ep)
    return dict(u2=u2, u3=dict(state=te.state(sim), impulse=[r["impulse"].copy() for r in ep.records], positions=pos_u3),
                result=result, tmp=tmp, ep=ep)


@needs_xpbd
def test_translation_through_the_transport_equals_the_in_process_stand_in(translation_run):
    u2, u3, r = translation_run["u2"], translation_run["u3"], translation_run["result"]
    assert r["status"] == "ok" and r["windows_acknowledged"] == 30
    for k in u2["state"]:
        assert u2["state"][k].tobytes() == u3["state"][k].tobytes(), k
    assert all(a.tobytes() == b.tobytes() for a, b in zip(u2["positions"], u3["positions"]))   # offers: X + k h v
    assert all(a.tobytes() == b.tobytes() for a, b in zip(u2["impulse"], u3["impulse"]))
    assert np.abs(u3["positions"][-1] - u3["positions"][0]).max() > 0                       # it moved


@needs_xpbd
@pytest.mark.parametrize("which", ["e1_run", "translation_run"])
def test_replay_into_the_loop_is_byte_identical(which, request, tmp_path):
    run_ = request.getfixturevalue(which)
    d = run_["tmp"] / "endpoint"
    man = archive.read(d)["manifest"]
    A = replay_archive(d, xpbd_root=XPBD_WORK, log_dir=tmp_path / "replay", timeouts=FAST)
    B = offline_receipts(d, xpbd_root=XPBD_WORK)
    assert A["registration_digest"] == B["registration_digest"] == man["registration_digest"]
    assert A["snapshot0_equal"] and B["snapshot0_equal"]
    assert A["offer_digests"] == B["offer_digests"] == [w["offer_digest"] for w in run_["result"]["windows"]]
    assert A["receipt_digests"] == B["receipt_digests"] == man["receipt_digests"] and len(A["receipt_digests"]) == 30
    assert A["receipt_content_digests"] == B["receipt_content_digests"] == man["receipt_content_digests"]
    assert A["receipt_canonical"] == B["receipt_canonical"]                                   # the bytes themselves
    # every committed snapshot, every field, byte for byte 
    assert len(A["committed"]) == len(B["committed"]) == 30 and first_mismatch(A, B) is None
    assert A["stop"]["exitcode"] == 0 and A["stop"]["bye"] is not None
    # the MPM stand-in rebuilt from the archive reproduces the content of XPBD's receipts
    assert [r["digest"] for r in archive.refeed(d)] == man["receipt_content_digests"]


@needs_xpbd
@pytest.mark.parametrize("case", ["fixed", "translation"])
def test_xpbd_contract_vectors_through_the_transport_and_offline(case, tmp_path):
    """XPBD's 4-point host vectors (a small fixture, not the 52 mesh): the canonical registration and scalar bytes
    are reproduced; the vector inputs through the transport and offline give identical receipts and committed
    snapshots on this machine (the acceptance pair). Equality with the host's own outputs is recorded in the
    evidence summary, not asserted: no cross-platform bitwise claim ."""
    if not VECTORS.exists():
        pytest.skip("XPBD contract vectors not in the mirror")
    assert te.coupling_sha(VECTORS) == VECTORS_SHA256
    d = json.loads(VECTORS.read_text())
    assert wire.canonical_json(d["registration"]["value"]) == d["registration"]["canonical_json"]
    assert wire.json_digest(d["registration"]["value"]) == d["registration"]["sha256"]
    for v in d["scalar_vectors"]:
        assert wire.canonical_json(v["value"]) == v["canonical_json"] and wire.json_digest(v["value"]) == v["sha256"]
    inp = inputs_from_vectors(VECTORS, case)
    A = run_transport(inp, xpbd_root=XPBD_WORK, log_dir=tmp_path / "svc", timeouts=FAST)
    B = run_offline(inp, xpbd_root=XPBD_WORK)
    assert A["registration_digest"] == B["registration_digest"] == d["registration"]["sha256"]
    assert A["receipt_canonical"] == B["receipt_canonical"] and len(A["receipt_canonical"]) == 3
    assert first_mismatch(A, B) is None and A["stop"]["exitcode"] == 0


def test_send_watchdog_unblocks_a_stalled_send():
    """8 MB that nobody reads: the send watchdog ends the other side (closes its end) and the blocked send fails as
    WireClosed. the send runs in a worker thread and the test waits at most
    10 s, so a platform on which closing the other end does not release the pending write (Windows: an overlapped
    WriteFile waited on with INFINITE) makes this test FAIL with that message instead of hanging the host tests step
    until its wall budget. The real transport's watchdog (link._stalled: terminate the service) relies on the same
    release, so on Windows this test is its first check."""
    a, b = mp.Pipe(duplex=True)
    tx = wire.Channel(a)
    out = {}

    def send():
        t0 = time.monotonic()
        try:
            tx.send("OFFER", {}, dict(x=np.zeros(2 ** 20)), timeout=0.5, on_stall=b.close)    # 8 MB, nobody reads
            out["result"] = "sent"
        except Exception as e:                                                            # judged below
            out["result"] = e
        out["wall_s"] = time.monotonic() - t0

    worker = threading.Thread(target=send, name="stalled-send", daemon=True)
    worker.start()
    worker.join(10.0)
    if worker.is_alive():
        a.close()                                                                         # the test has failed
        pytest.fail("the send watchdog closed the other end but the pending send was not released within 10 s")
    assert isinstance(out["result"], wire.WireClosed) and "send watchdog" in str(out["result"]), out
    assert out["wall_s"] < 5, out
    a.close()


class _StalledRead:
    """Stand-in connection whose frame has begun but never completes: poll() sees data, recv_bytes() blocks until the
    other side is ended (close), then fails the way a pipe whose writer went away does (EOFError)."""

    def __init__(self):
        self.ended = threading.Event()

    def poll(self, timeout):
        return True

    def recv_bytes(self, maxlength=None):
        if not self.ended.wait(30.0):
            raise AssertionError("the read watchdog never ended the other side")
        raise EOFError

    def close(self):
        self.ended.set()


def test_read_watchdog_unblocks_a_half_frame():
    """A frame whose start arrived but whose body never does . no skip on any platform: a
    stand-in connection drives Channel.recv's read watchdog everywhere; on POSIX the real half frame as well
    (Connection's own 4-byte length header and one byte of body written to the socket). A Windows pipe is message-mode
    (a message arrives whole) and its fileno() is a pipe handle, not a file descriptor: writing to it with
    os.write, which cannot work there, so the real half frame exists only on POSIX."""
    stand_in = _StalledRead()
    t0 = time.monotonic()
    with pytest.raises(wire.WireClosed, match="read watchdog"):
        wire.Channel(stand_in).recv(1.0, on_stall=stand_in.close)
    assert stand_in.ended.is_set() and time.monotonic() - t0 < 5
    if os.name != "posix":
        return                                                                          # no half frame on Windows
    a, b = mp.Pipe(duplex=True)
    rx = wire.Channel(b)
    os.write(a.fileno(), struct.pack("!i", 100) + b"x")                                 # Connection's own length header
    t0 = time.monotonic()
    with pytest.raises(wire.WireClosed, match="read watchdog"):
        rx.recv(1.0, on_stall=a.close)
    assert time.monotonic() - t0 < 5
    b.close()


@needs_xpbd
def test_service_checks_open_and_impulse_itself():
    """The service state machine checks OPEN's h, particle_dt, window index and IMPULSE's window index itself,
    not only through XPBD ; XPBD's 4-point vector inputs, no process."""
    if not VECTORS.exists():
        pytest.skip("XPBD contract vectors not in the mirror")
    X, _ = xpbd_service.import_xpbd(XPBD_WORK, PINNED_XPBD)
    inp = inputs_from_vectors(VECTORS, "fixed")
    core = xpbd_service.ServiceCore(X, {}, xpbd_service.ServiceLog(None))

    def fr(mode, meta, arrays=None):
        return wire.Frame(mode, 0, meta, arrays or {}, 0)
    core.handle(fr("REGISTER", dict(registration_text=inp["registration_text"], stream_id=inp["stream_id"], h=inp["h"],
                                    particle_dt=inp["dt"], snapshot=inp["spec"]),
                   {k: np.asarray(v) for k, v in inp["mesh"].items()}))
    for bad in (dict(window_index=0, h=2 * inp["h"], particle_dt=inp["dt"]), dict(window_index=0, h=inp["h"],
                particle_dt=inp["dt"] / 2), dict(window_index=False, h=inp["h"], particle_dt=inp["dt"]),
                dict(window_index=1, h=inp["h"], particle_dt=inp["dt"])):
        with pytest.raises(xpbd_service.ServiceError):
            core.handle(fr("OPEN", bad))
        assert core.open is None
    assert core.handle(fr("OPEN", dict(window_index=0, h=inp["h"], particle_dt=inp["dt"])))[0] == "OFFER"
    core.handle(fr("PEER", dict(peer=inp["peer"])))
    meta, arrays = inp["batches"][0]
    for w in (1, False):
        with pytest.raises(xpbd_service.ServiceError, match="IMPULSE for window"):
            core.handle(fr("IMPULSE", dict(meta, window_index=w), arrays))
    assert core.committed == 0 and core.handle(fr("IMPULSE", meta, arrays))[0] == "RECEIPT" and core.committed == 1


@needs_xpbd
def test_malformed_receipt_and_unclean_stop_are_recorded(tmp_path):
    """A RECEIPT without its fields is a protocol error (S8), coupled.json is written; a run whose service gives
    no BYE is not 'ok' ."""
    sim, ep = build_h()
    spec, mesh = spec_for(ep)
    drv = CoupledDriver(ep, snapshot=spec, mesh_arrays=mesh, log_dir=tmp_path / "a", xpbd_root=XPBD_WORK, timeouts=FAST)
    orig = drv.link.request

    def request(mode, meta, arrays=None, **kw):
        out = orig(mode, meta, arrays, **kw)
        if mode == "IMPULSE" and meta["window_index"] == 1:
            out = wire.Frame(out.mode, out.seq, {}, out.arrays, out.nbytes)
        return out
    drv.link.request = request
    r = drv.run(3)
    assert r["status"].startswith("stopped: S8 DriverError") and "lacks" in r["status"], r["status"]
    assert ep.awaiting_receipt == 1 and (tmp_path / "a" / "coupled.json").exists() and r["xpbd_stop"]["exitcode"] == 0
    sim, ep = build_h()
    drv = CoupledDriver(ep, snapshot=spec, mesh_arrays=mesh, log_dir=tmp_path / "b", xpbd_root=XPBD_WORK, timeouts=FAST)
    orig_stop = drv.link.stop
    drv.link.stop = lambda reason: dict(orig_stop(reason), bye=None)
    r = drv.run(2)
    assert ep.acknowledged() == 2 and r["status"].startswith("incomplete"), r["status"]


def test_replay_environment_must_be_the_sealed_one():
    from coupling.driver import check_environment
    inp = dict(environment=dict(python="3.10.22", numpy="1.26.4", xpbd_files={"a": "1"}))
    assert check_environment(inp, "3.10.22", "1.26.4", {"a": "1"})
    for got in (("3.10.21", "1.26.4", {"a": "1"}), ("3.10.22", "1.26.3", {"a": "1"}), ("3.10.22", "1.26.4", {"a": "2"})):
        with pytest.raises(DriverError, match="environment"):
            check_environment(inp, *got)
    assert check_environment(dict(environment=None), "x", "y", {}) is None                # vectors: recorded only


def test_receipt_field_set_is_fixed():
    from coupling.driver import RECEIPT_FIELDS, check_receipt_fields
    r = {k: None for k in RECEIPT_FIELDS}
    r["schema"] = "xpbd-registered-kinematic-v1"
    check_receipt_fields(r)
    for bad in (dict(r, extra=1), {k: v for k, v in r.items() if k != "drive_work"}, dict(r, schema="other")):
        with pytest.raises(DriverError):
            check_receipt_fields(bad)


# ------------------------------------------------------------------ 6. fault injection: both sides stop
FAULTS = {
    "reject": dict(fault=dict(reject_at=2)),
    "receipt_timeout": dict(fault=dict(delay_receipt_at=1, delay_s=3.0), timeouts=dict(impulse=1.0)),
    "nan_offer": dict(fault=dict(nan_offer_at=1)),
    "nan_impulse": dict(driver_fault=dict(nan_impulse_at=1)),
    "service_exit": dict(fault=dict(exit_at=1)),
    "open_timeout": dict(fault=dict(delay_open_at=1, delay_s=3.0), timeouts=dict(open=1.0)),
    "idle_watchdog": dict(timeouts=dict(idle=3.0)),   # > window 0 incl. compile
    "mpm_nonfinite_batch": dict(),
    "committed_receipt_not_digestible": dict(fault=dict(bad_receipt_at=1)),     # 
}


@needs_xpbd
@pytest.mark.parametrize("case", list(FAULTS))
def test_fault_injection_stops_both_sides_and_keeps_the_prefix(case, tmp_path):
    sim, ep = build_h()
    kw = copy.deepcopy(FAULTS[case])
    hooks = {}
    if case == "idle_watchdog":
        hooks["on_window"] = lambda k: time.sleep(6.0) if k == 0 else None
    if case == "mpm_nonfinite_batch":
        orig = ep.surface.last_batch

        def bad():
            b = orig()
            if b["window_index"] == 1:
                for key in ("impulse_G", "impulse_G_native"):
                    b[key] = b[key].copy()
                    b[key][0, 0] = np.nan
            return b
        ep.surface.last_batch = bad
    t0 = time.monotonic()
    r, drv = drive(ep, 5, tmp_path, hooks=hooks, **kw)
    wall = time.monotonic() - t0
    rows, final = service_log(tmp_path)
    xs = r["xpbd_stop"]
    assert r["status"].startswith("stopped:") and ep.stopped is not None and ep.phase == "stopped"
    assert xs["exitcode"] is not None and not drv.link.proc.is_alive()                     # the service ended too
    with pytest.raises(EndpointStop, match="stopped earlier"):
        ep.halt("again")                                                                 # no retry on this stream
    expect = dict(reject=(2, 2, None, "S8 the peer rejected window 2"), receipt_timeout=(1, 2, 1, "S8 transport timeout"),
                  nan_offer=(1, 1, None, "S1"), nan_impulse=(1, 1, None, "S8 the peer rejected window 1"),
                  service_exit=(1, 1, 1, "S8 transport closed"), open_timeout=(1, 1, None, "S8 transport timeout"),
                  idle_watchdog=(1, 1, None, "S8 transport closed"), mpm_nonfinite_batch=(1, 1, None, "S1"),
                  committed_receipt_not_digestible=(1, 2, 1, "S8 the XPBD service refused at IMPULSE"))[case]
    acked, committed, awaiting, reason = expect
    assert ep.acknowledged() == acked and ep.awaiting_receipt == awaiting, (ep.stopped, ep.acknowledged())
    assert ep.stopped["reason"].startswith(reason), ep.stopped["reason"]
    assert len(rows) == committed and [x["receipt_digest"] for x in rows[:acked]] == \
        [x["receipt_digest"] for x in ep.ledger[:acked]]
    archive.write(tmp_path / "endpoint", ep)
    v = archive.verify(tmp_path / "endpoint")
    rc = reconcile(tmp_path / "endpoint", tmp_path / "coupling" / "xpbd_service")
    assert not v["complete"] and v["acknowledged"] == acked and rc["consistent"], rc
    assert rc["extra_xpbd_windows"] == committed - acked
    if case == "service_exit":
        assert final is None and xs["exitcode"] == 3 and wall < 10                       # EOF, not a watchdog
    elif case == "idle_watchdog":
        assert final["reason"].startswith("idle watchdog") and xs["exitcode"] == 0
    else:
        assert final is not None and xs["bye"] is not None and xs["exitcode"] == 0, xs
        assert final["committed_windows"] == committed and final["reason"].startswith("STOP")
    if case == "receipt_timeout":
        assert [f["mode"] for f in xs["late_frames"]] == ["RECEIPT"] and xs["late_frames"][0]["window_index"] == 1
        assert "commit of window 1 is unknown" in ep.stopped["reason"] and rc["note"]
    if case == "open_timeout":
        assert [f["mode"] for f in xs["late_frames"]] == ["OFFER"] and final["open_window"] == 1
    if case in ("nan_offer", "mpm_nonfinite_batch"):
        assert final["open_window"] == 1                                                 # opened, never consumed
    if case == "nan_impulse":                                                            # only the wire copy was bad
        z = np.load(tmp_path / "endpoint" / "batches.npz")
        assert np.isfinite(z["impulse"]).all() and not z["delivered"][1]
    if case == "reject":
        assert final["frames_received"] == 9                                             # REGISTER, 3 OPEN, PEER, 3 IMPULSE, STOP
    if case == "committed_receipt_not_digestible":                                       # counted as committed, MPM halts
        assert final["committed_windows"] == 2 and rows[1]["receipt_digest"] is None and rows[1]["error"]
        assert "commit of window 1 is unknown" in ep.stopped["reason"] and ep.records[1]["delivered"]


# ------------------------------------------------------------------ 7. handshake failures: before the first window
@needs_xpbd
@pytest.mark.parametrize("case", ["non_canonical", "duplicate_key", "digest", "peer_binding", "bound_mesh"])
def test_handshake_failures_stop_before_the_first_window(case, tmp_path):
    sim, ep = build_h()
    kw = {}
    if case == "non_canonical":
        ep.registration_frozen = json.dumps(json.loads(ep.registration_frozen), indent=1)
    elif case == "duplicate_key":
        ep.registration_frozen = ep.registration_frozen.replace('"mpm_epoch":1', '"mpm_epoch":1,"mpm_epoch":1')
    elif case == "digest":
        ep.registration_digest = "0" * 64
    elif case == "peer_binding":
        orig = ep.receive

        def receive(offer):
            rep = orig(offer)
            ep.registration["peer"]["n_vertices"] += 1                          # MPM records a different binding
            return rep
        ep.receive = receive
    elif case == "bound_mesh":
        m = dict(lf.reference_mesh())
        m["positions"] = m["positions"] + 1e-3
        kw["expect_mesh"] = m
    r, drv = drive(ep, 3, tmp_path, **kw)
    assert int(sim.step[None]) == 0 and ep.k == 0 and ep.stopped["reason"].startswith("S8")
    assert r["xpbd_stop"]["exitcode"] == 0 and not drv.link.proc.is_alive()
    rows, final = service_log(tmp_path)
    assert rows == [] and final["committed_windows"] == 0
    if case == "peer_binding":
        assert ep.stopped["pending_window"] == 0                                 # read, not advanced (R9)
        archive.write(tmp_path / "endpoint", ep)
        assert archive.verify(tmp_path / "endpoint")["pending_window"] == 0
    else:
        assert ep.snapshot0 is None


@needs_xpbd
def test_every_run_uses_a_new_stream():
    sim, ep = build_h()
    spec, mesh = spec_for(ep)
    ids = {CoupledDriver(ep, snapshot=spec, mesh_arrays=mesh).stream_id for _ in range(3)}
    assert len(ids) == 3 and all(s.startswith("u3-") for s in ids)


# ------------------------------------------------------------------ 8. endpoint phase machine (no process)
def test_endpoint_delivered_phase_and_receipt_checks():
    sim, ep, peer, _ = te.build()
    msg = ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))
    receipt = peer.consume(msg)
    steps = int(sim.step[None])
    with pytest.raises(EndpointStop, match="phase 'delivered'"):
        ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))                           # no offer before the receipt
    assert int(sim.step[None]) == steps and ep.awaiting_receipt == 0
    for change in ("ulp", "numpy", "version", "window", "bool_window", "h"):
        sim, ep, peer, _ = te.build()
        rc = copy.deepcopy(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
        if change == "ulp":
            i = int(np.flatnonzero(np.abs(np.asarray(rc["impulse"])).sum(1))[0]) if np.abs(rc["impulse"]).sum() else 0
            rc["impulse"][i][1] = float(np.nextafter(rc["impulse"][i][1], 1.0))
        elif change == "numpy":
            rc["impulse"] = np.asarray(rc["impulse"])
        elif change == "version":
            rc["committed_version"] = [1, 2, 0]
        elif change == "bool_window":
            rc["window_index"] = False                                                    # == 0 but not an int (S1)
        elif change == "h":
            rc["h"] = 2 * ep.H                                                             # checked when present (S2)
        else:
            rc["window_index"] = 1
        with pytest.raises(EndpointStop, match="S8 receipt"):
            ep.acknowledge(rc)
        assert ep.records[0]["acknowledged"] is False and ep.ledger[0]["receipt_digest"] is None
    # reject after an acknowledged window stops without touching it 
    sim, ep, peer, _ = te.build()
    ep.acknowledge(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
    with pytest.raises(EndpointStop, match="no delivered window"):
        ep.reject("late refusal")
    assert ep.records[0]["delivered"] and ep.records[0]["acknowledged"] and ep.awaiting_receipt is None
    # a later window must carry the previous window's h as dt 
    sim, ep, peer, _ = te.build()
    ep.acknowledge(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
    o = peer.open_window(h=ep.H, particle_dt=ep.dt)
    with pytest.raises(EndpointStop, match="snapshot dt"):
        ep.receive(dataclasses.replace(o, snapshot=dataclasses.replace(o.snapshot, dt=2 * ep.H)))
    assert int(sim.step[None]) == ep.n_sub
    # the frozen registration (peer = null) does not change when the peer is recorded
    assert json.loads(ep.registration_frozen)["peer"] is None and ep.registration["peer"]["n_vertices"] == te.NV
    assert wire.json_digest(json.loads(ep.registration_frozen)) == ep.registration_digest


def test_handshake_mode_is_enforced_by_the_phase_machine():
    sim, ep = build_h()
    peer = te.PrescribedPeer(*(lf.reference_mesh()[k] for k in ("positions", "triangles", "vertex_ids", "triangle_ids")),
                             np.zeros(3), h=ep.H, stream_id="u3-test", expected_source=ep.source)
    with pytest.raises(EndpointStop, match="registration handshake"):
        ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))
    sim, ep = build_h()
    peer = te.PrescribedPeer(*(lf.reference_mesh()[k] for k in ("positions", "triangles", "vertex_ids", "triangle_ids")),
                             np.zeros(3), h=ep.H, stream_id="u3-test", expected_source=ep.source)
    ep.confirm_registration(ep.registration_digest)
    ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(EndpointStop, match="confirm_peer"):
        ep.advance()
    assert int(sim.step[None]) == 0 and ep.stopped["pending_window"] == 0
    # the endpoint itself refuses a binding that differs from what it recorded 
    sim, ep = build_h()
    peer = te.PrescribedPeer(*(lf.reference_mesh()[k] for k in ("positions", "triangles", "vertex_ids", "triangle_ids")),
                             np.zeros(3), h=ep.H, stream_id="u3-test", expected_source=ep.source)
    ep.confirm_registration(ep.registration_digest)
    ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))
    with pytest.raises(EndpointStop, match="peer binding"):
        ep.confirm_peer(dict(ep.registration["peer"], n_vertices=ep.registration["peer"]["n_vertices"] + 1))
    assert int(sim.step[None]) == 0


# ------------------------------------------------------------------ 9. archive: U2 semantics still read
def test_archive_without_the_u3_revision_keeps_the_u2_semantics(tmp_path):
    sim, ep, peer, _ = te.build(shuffle=7)
    te.run(sim, ep, peer, 4)
    d = tmp_path / "a"
    archive.write(d, ep)
    man = json.loads((d / "manifest.json").read_text())
    assert man["revision"] == archive.REVISION and man["receipt_content_digests"] == [r["digest"] for r in peer.receipts]
    assert archive.verify(d)["complete"] and len(archive.refeed(d)) == 4
    # the same archive as U2 wrote it: no revision, receipt_digests = the peer's content digests
    for key in ("revision", "acknowledged", "receipt_content_digests", "awaiting_receipt", "registration_digest",
                "receipt_digest_rule"):
        man.pop(key)
    man["receipt_digests"] = [r["digest"] for r in peer.receipts]
    led = (d / "ledger.csv").read_text().splitlines()
    cols = led[0].split(",")
    i = cols.index("receipt_digest")
    for n, r in enumerate(peer.receipts, start=1):
        cells = led[n].split(",")
        cells[i] = r["digest"]
        led[n] = ",".join(cells)
    (d / "ledger.csv").write_text("\n".join(led) + "\n")
    man["files"]["ledger.csv"].update(sha256=te.coupling_sha(d / "ledger.csv"), bytes=(d / "ledger.csv").stat().st_size)
    (d / "manifest.json").write_text(json.dumps(man))
    v = archive.verify(d)
    assert v["complete"] and v["acknowledged"] is None
    assert [r["digest"] for r in archive.refeed(d)] == man["receipt_digests"]
    # a U3 manifest that lost one of its U3 keys is refused, not read as U2 
    d2 = tmp_path / "b"
    archive.write(d2, ep)
    man2 = json.loads((d2 / "manifest.json").read_text())
    man2.pop("acknowledged")
    (d2 / "manifest.json").write_text(json.dumps(man2))
    with pytest.raises(ValueError, match="lacks"):
        archive.verify(d2)
    with pytest.raises(ValueError, match="lacks"):
        archive.refeed(d2)


# ------------------------------------------------------------------ 10. the scene CLI
@needs_xpbd
def test_coupled_u13_scene_cli(tmp_path):
    cmd = [sys.executable, "-B", str(WORK / "scenes" / "coupled_u13.py"), "--set", "grid.n=16", "--set",
           "scene.windows=10", "--set", "scene.windows_per_frame=5", "--set", "scene.block.size=[0.3,0.1,0.3]",
           "--set", f"scene.xpbd.root={XPBD_WORK}", "--out", str(tmp_path), "--threads", "3"]
    if not FILE_MESH:
        pytest.skip("the CLI binds the published mesh file")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    run_dir = tmp_path / "coupled_u13"
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["status"] == "ok" and s["windows_acknowledged"] == 10 and s["reconcile"]["consistent"]
    assert s["archive"]["complete"] and s["archive"]["verify"]["acknowledged"] == 10
    hello = json.loads((run_dir / "coupling" / "xpbd_service" / "hello.json").read_text())
    assert hello["forbidden_modules_at_start"] == [] and hello["forbidden_modules_after_import"] == []
    assert hello["fault"] == {}                                                         # no injection in a scene run
    coupled = json.loads((run_dir / "coupling" / "coupled.json").read_text())
    assert coupled["xpbd_stop"]["exitcode"] == 0 and len(coupled["windows"]) == 10
    assert (run_dir / "cloth" / "00002.npz").exists()


@needs_xpbd
def test_zz_no_bytecode_written_into_the_xpbd_mirror():
    assert _pycache_dirs() == PYCACHE_BEFORE
