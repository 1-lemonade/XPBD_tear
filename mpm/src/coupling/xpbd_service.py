"""XPBD service process of the MPM <-> XPBD exchange (P2-U3). Written by MPM, run in its own
process (multiprocessing spawn); it imports only standard library, NumPy, coupling.wire and,
at main() from the configured XPBD root, XPBD's public entry
project.coupling.registered_kinematic.create_registered_kinematic_adapter with the exchange
contract classes. It never imports taichi or mpm (checked at start: a hard stop) and never
writes into the XPBD tree (no bytecode; logs go to the directory the driver gives).

State machine (requests from the driver; one reply each, ERROR possible for any):
    start      -> HELLO (or ERROR and exit)
    REGISTER   -> BOUND      registration text (canonical JSON, duplicate keys refused), stream,
                             h, particle_dt, snapshot spec -> all-pinned snapshot + adapter
    OPEN       -> OFFER      adapter.open_window(h, particle_dt); one window open at a time
    PEER       -> PEER_OK    adapter.validate_peer(peer); once, after OFFER 0, before IMPULSE 0
    IMPULSE    -> RECEIPT    XPBD LoadBatch + ImpulseWindow rebuilt from the frame, adapter.consume
    STOP       -> BYE        final state, then exit
Any failure -> ERROR (stage, retryable in XPBD terms, adapter_failed, committed windows); after
it only STOP is served. Idle watchdog: no request within `idle` seconds -> final.json, exit.
Fault injection (`fault`, test configuration only, reported in HELLO and final.json):
reject_at, delay_receipt_at + delay_s, delay_open_at + delay_s, nan_offer_at, exit_at, bad_receipt_at , log_fail_at (the receipt
log row of a committed window cannot be written, P2-U4 prefix fix B).
Receipt log (receipts.jsonl, one canonical JSON row per committed window, flushed): a row that cannot be written
stops the service like any other failure after the commit (ERROR, committed windows > MPM's acknowledged); the
error is kept in log_errors (state: ERROR, BYE, final.json) so a reconcile can tell a log write failure from a
missing log or a crash (no final.json).
"""
import hashlib
import importlib
import io
import json
import os
import pathlib
import platform
import sys
import time
import traceback

import numpy as np

from . import wire

SERVICE = "mpm-xpbd-service-v1"
FORBIDDEN = ("taichi", "mpm")
XPBD_IMPORTS = ("project.coupling.registered_kinematic", "project.coupling.contracts", "project.coupling.loads",
                "project.coupling.exchange")
MESH_SCHEMA, MESH_FRAME = "xpbd-coupling-reference-mesh-v1", "xpbd-native-xy-v1"
FAULT_KEYS = ("reject_at", "delay_receipt_at", "delay_open_at", "delay_s", "nan_offer_at", "exit_at", "bad_receipt_at",
              "log_fail_at")


class ServiceError(RuntimeError):
    pass


def forbidden_loaded():
    return sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)


def _sha_bytes(b) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ XPBD import (read-only)
def import_xpbd(root, expected=None):
    """Import XPBD's public entry from `root` (its work_place) without writing bytecode; return
    the callables and {path relative to root: SHA-256} of every loaded project module; each must be inside root/project."""
    root = pathlib.Path(root).resolve()
    if not (root / "project" / "coupling" / "registered_kinematic.py").is_file():
        raise ServiceError(f"XPBD root {root} has no project/coupling/registered_kinematic.py")
    sys.dont_write_bytecode = True
    pre = sys.modules.get("project")
    if pre is not None and root not in pathlib.Path(getattr(pre, "__file__", "") or "/").resolve().parents:
        raise ServiceError(f"a module 'project' from {getattr(pre, '__file__', None)} is already loaded (shadowing)")
    if str(root) not in sys.path:
        sys.path.append(str(root))                             # after the MPM paths
    mods = [importlib.import_module(m) for m in XPBD_IMPORTS]
    for m in mods:
        if root not in pathlib.Path(m.__file__).resolve().parents:
            raise ServiceError(f"{m.__name__} was imported from {m.__file__}, not from {root} (shadowing)")
    files = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if name == "project" or name.startswith("project."):
            if not f:
                raise ServiceError(f"XPBD module {name} has no source file")
            p = pathlib.Path(f).resolve()
            if (root / "project") not in p.parents:
                raise ServiceError(f"XPBD module {name} is outside the expected project directory (shadowing)")
            files[p.relative_to(root).as_posix()] = _sha_bytes(p.read_bytes())
    files = dict(sorted(files.items()))
    # pinning : the four imported core files must be pinned when a pin is given;
    # every pinned file that was imported must match; other loaded files are reported, not compared
    core = [m.replace(".", "/") + ".py" for m in XPBD_IMPORTS]
    if expected:
        unpinned = [c for c in core if c not in expected]
        if unpinned:
            raise ServiceError(f"the pin set lacks the imported core files {unpinned}")
    for rel, want in (expected or {}).items():
        if rel in files and files[rel] != want:
            raise ServiceError(f"XPBD file {rel}: SHA-256 {files[rel]} differs from the pinned {want}")
    rk, contracts, loads, exchange = mods
    X = dict(create=rk.create_registered_kinematic_adapter, ClothSnapshot=contracts.ClothSnapshot,
             LoadBatch=loads.LoadBatch, SourceIdentity=loads.SourceIdentity, ImpulseWindow=exchange.ImpulseWindow,
             SCHEMA=exchange.SCHEMA)
    return X, files


# ------------------------------------------------------------------ snapshot
def load_mesh(path, mesh_sha256, manifest_sha256) -> dict:
    """The published mesh.npz (NumPy, allow_pickle=False) bound by both digests: the bytes that are
    hashed are the bytes that are loaded; the manifest's file record must match."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "mesh.npz"
    if not mesh_sha256 or not manifest_sha256:
        raise ServiceError("a file mesh needs both mesh_sha256 and manifest_sha256")
    data = p.read_bytes()
    if _sha_bytes(data) != mesh_sha256:
        raise ServiceError(f"mesh {p}: SHA-256 differs from {mesh_sha256}")
    mb = (p.parent / "manifest.json").read_bytes()
    if _sha_bytes(mb) != manifest_sha256:
        raise ServiceError(f"mesh manifest: SHA-256 differs from {manifest_sha256}")
    man = wire.strict_loads(mb)
    rec = man.get("file", {})
    if (man.get("schema") != MESH_SCHEMA or man.get("frame_id") != MESH_FRAME or man.get("complete") is not True
            or rec.get("sha256") != mesh_sha256 or rec.get("bytes") != len(data)):
        raise ServiceError("mesh manifest: schema, frame, complete flag or file record differs")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        arrays = {k: np.array(z[k]) for k in ("positions", "triangles", "vertex_ids", "triangle_ids")}
    return dict(arrays=arrays, binding="file", path=str(p), sha256=mesh_sha256, manifest_sha256=manifest_sha256,
                manifest_topology_version=man.get("topology_version"))


def build_snapshot(X, spec, mesh_arrays, h):
    """All vertices pinned, inverse_mass 0, one native velocity for all (U13 registered bridge)."""
    a = mesh_arrays
    N = len(a["positions"])
    v = np.asarray(spec["velocity_native"], np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ServiceError("snapshot velocity_native must be 3 finite numbers")
    prev = spec.get("previous_dt")
    return X["ClothSnapshot"](positions=np.asarray(a["positions"], np.float64),
                              triangles=np.asarray(a["triangles"], np.int64),
                              vertex_ids=np.asarray(a["vertex_ids"], np.int64),
                              triangle_ids=np.asarray(a["triangle_ids"], np.int64),
                              velocity=np.tile(v, (N, 1)), inverse_mass=np.zeros(N), pinned=np.ones(N, bool),
                              epoch=int(spec["epoch"]), step_index=0, topology_version=int(spec["topology_version"]),
                              time=0.0, dt=float(h if prev is None else prev))


def mesh_content_digest(arrays) -> str:
    """Digest of the four mesh arrays as used (file or inline): canonical dtype, shape and SHA-256 of the bytes."""
    a = {k: np.asarray(arrays[k], "<f8" if k == "positions" else "<i8")
         for k in ("positions", "triangles", "vertex_ids", "triangle_ids")}
    return payload_digest(dict(kind="xpbd-mesh-arrays"), a)


def parse_registration(text):
    """Strict JSON (duplicate keys, non-finite refused) in canonical form only."""
    if type(text) is not str:
        raise ServiceError("registration_text must be a string")
    reg = wire.strict_loads(text)
    if type(reg) is not dict:
        raise ServiceError("the registration must be a JSON object")
    if wire.canonical_json(reg) != text:
        raise ServiceError("registration_text is not in canonical form (digest bytes must be agreed)")
    return reg


def offer_payload(offer):
    """(meta, arrays) of an XPBD WindowOffer on the wire."""
    S = offer.snapshot
    src = offer.source
    meta = dict(schema=offer.schema, stream_id=offer.stream_id,
                source=dict(source_id=src.source_id, kind=src.kind, version=int(src.version)),
                window_index=int(offer.window_index), particle_dt=float(offer.particle_dt), h=float(offer.h),
                substeps=int(offer.substeps), cut_edges=[list(map(int, e)) for e in sorted(offer.cut_edges)],
                epoch=int(S.epoch), step_index=int(S.step_index), topology_version=int(S.topology_version),
                time=float(S.time), dt=float(S.dt))
    arrays = dict(positions=np.asarray(S.positions, "<f8"), triangles=np.asarray(S.triangles, "<i8"),
                  vertex_ids=np.asarray(S.vertex_ids, "<i8"), triangle_ids=np.asarray(S.triangle_ids, "<i8"),
                  velocity=np.asarray(S.velocity, "<f8"), inverse_mass=np.asarray(S.inverse_mass, "<f8"),
                  pinned=np.asarray(S.pinned, "|b1"))
    return meta, arrays


def payload_digest(meta, arrays) -> str:
    """Digest of a (meta, arrays) message content: canonical JSON of meta + per-array dtype, shape, SHA-256."""
    return wire.json_digest(dict(meta=meta, arrays={k: [np.asarray(a).dtype.str, list(np.asarray(a).shape),
                                                        _sha_bytes(np.ascontiguousarray(a).tobytes())]
                                                    for k, a in sorted(arrays.items())}))


def batch_from_wire(X, meta, arrays):
    """XPBD LoadBatch + ImpulseWindow from an IMPULSE frame (XPBD's own constructors check it)."""
    b = meta["batch"]
    batch = X["LoadBatch"](stream_id=b["stream_id"], source=X["SourceIdentity"](**b["source"]),
                           cloth_version=tuple(b["cloth_version"]), batch_id=b["batch_id"], t_start=b["t_start"],
                           t_end=b["t_end"], vertex_ids=arrays["vertex_ids"], impulse=arrays["impulse"],
                           coordinate_frame=b["coordinate_frame"], units=b["units"])
    return X["ImpulseWindow"](meta["schema"], meta["window_index"], batch)


# ------------------------------------------------------------------ log
class ServiceLog:
    def __init__(self, log_dir):
        self.dir = None if log_dir is None else pathlib.Path(log_dir)
        self.errors = []                                       # receipt rows that could not be written
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name, obj):
        if self.dir is not None:
            (self.dir / name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")

    def receipt(self, row):
        if self.dir is not None:
            with (self.dir / "receipts.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(wire.canonical_json(row) + "\n")
                fh.flush()


# ------------------------------------------------------------------ state machine
class ServiceCore:
    def __init__(self, X, cfg, log):
        self.X, self.cfg, self.log = X, cfg, log
        self.fault = dict(cfg.get("fault") or {})
        self.adapter = None
        self.stream_id = None
        self.h = self.particle_dt = None
        self.open = None                                       # window index of the open window
        self.committed = 0                                     # windows consumed (receipts returned)
        self.peer_ok = False
        self.failed = None                                     # first error; afterwards only STOP
        self.mesh = None

    def state(self):
        a = self.adapter
        return dict(committed_windows=self.committed, open_window=self.open, peer_validated=self.peer_ok,
                    log_errors=list(self.log.errors),
                    adapter_failed=None if a is None else bool(a.failed), failed=self.failed)

    def _at(self, key, k):
        return self.fault.get(key) is not None and int(self.fault[key]) == k

    def _log_receipt(self, k, row):
        """One receipt log row of committed window k; a write failure stops the service and is kept (fix B)."""
        try:
            if self._at("log_fail_at", k):
                raise OSError(f"injected receipt log write failure at window {k} (test configuration)")
            self.log.receipt(row)
        except OSError as e:
            self.log.errors.append(dict(window_index=k, error=f"{type(e).__name__}: {e}"[:500]))
            raise ServiceError(f"window {k} was committed but its receipt log row could not be written: {e}") from e

    def handle(self, fr):
        if fr.mode == "REGISTER":
            return self._register(fr)
        if self.adapter is None:
            raise ServiceError(f"{fr.mode} before REGISTER")
        if fr.mode == "OPEN":
            return self._open(fr)
        if fr.mode == "PEER":
            return self._peer(fr)
        if fr.mode == "IMPULSE":
            return self._impulse(fr)
        raise ServiceError(f"unexpected request {fr.mode}")

    def _register(self, fr):
        if self.adapter is not None:
            raise ServiceError("REGISTER twice (one stream per service process)")
        m = fr.meta
        text = m["registration_text"]
        reg = parse_registration(text)
        spec = m["snapshot"]
        if spec.get("mesh") == "file":
            self.mesh = load_mesh(spec["mesh_path"], spec.get("mesh_sha256"), spec.get("manifest_sha256"))
            tv = self.mesh["manifest_topology_version"]
            if tv is not None and spec["topology_version"] != tv:     # interface specification: the manifest's version
                raise ServiceError(f"snapshot topology_version {spec['topology_version']} differs from the mesh "
                                   f"manifest's {tv}")
        elif spec.get("mesh") == "inline":
            self.mesh = dict(arrays={k: fr.arrays[k] for k in ("positions", "triangles", "vertex_ids", "triangle_ids")},
                             binding="inline", path=None, sha256=None, manifest_sha256=None)
        else:
            raise ServiceError(f"snapshot mesh must be 'file' or 'inline', got {spec.get('mesh')!r}")
        h, dt = m["h"], m["particle_dt"]
        snap = build_snapshot(self.X, spec, self.mesh["arrays"], h)
        adapter = self.X["create"](snap, registration=reg, stream_id=m["stream_id"], h=h, particle_dt=dt)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if adapter.registration_digest != digest:
            raise ServiceError("adapter registration digest differs from the digest of the registration text")
        self.adapter, self.stream_id, self.h, self.particle_dt = adapter, m["stream_id"], h, dt
        out = dict(registration_digest=adapter.registration_digest, binding=adapter.binding,
                   snapshot=dict(n_vertices=len(snap.vertex_ids), n_triangles=len(snap.triangles), epoch=snap.epoch,
                                 topology_version=snap.topology_version, dt=snap.dt, time=snap.time,
                                 previous_dt=snap.dt, mesh_binding=self.mesh["binding"], mesh_sha256=self.mesh["sha256"],
                                 manifest_sha256=self.mesh["manifest_sha256"],
                                 mesh_content_digest=mesh_content_digest(self.mesh["arrays"])))
        self.log.write_json("bound.json", dict(out, stream_id=self.stream_id, h=h, particle_dt=dt))
        return "BOUND", out, None

    def _open(self, fr):
        if self.open is not None:
            raise ServiceError(f"OPEN while window {self.open} is open")
        k = self.committed
        w = fr.meta.get("window_index")
        if type(w) is not int or w != k:                       # no bool for 0 / 1 
            raise ServiceError(f"OPEN for window {w!r}, the next window is {k}")
        h, dt = fr.meta.get("h"), fr.meta.get("particle_dt")
        if type(h) is not float or type(dt) is not float or h != self.h or dt != self.particle_dt:
            raise ServiceError(f"OPEN h / particle_dt {h!r} / {dt!r} differ from the registered {self.h!r} / "
                               f"{self.particle_dt!r}")
        offer = self.adapter.open_window(h=h, particle_dt=dt)
        self.open = int(offer.window_index)
        meta, arrays = offer_payload(offer)
        if self._at("nan_offer_at", k):                       # the wire copy only (XPBD's snapshot stays finite)
            X = arrays["positions"].copy()
            X[7, 1] = np.nan
            arrays["positions"] = X
        meta["offer_digest"] = payload_digest(meta, arrays)    # of what is actually sent
        if self._at("delay_open_at", k):
            time.sleep(float(self.fault.get("delay_s", 1.0)))
        return "OFFER", meta, arrays

    def _peer(self, fr):
        if self.peer_ok or self.committed != 0 or self.open != 0:
            raise ServiceError("PEER is validated once, after OFFER 0 and before IMPULSE 0")
        self.adapter.validate_peer(fr.meta["peer"])
        self.peer_ok = True
        return "PEER_OK", dict(binding=self.adapter.binding), None

    def _impulse(self, fr):
        k = self.open
        if k is None:
            raise ServiceError("IMPULSE without an open window")
        if k == 0 and not self.peer_ok:
            raise ServiceError("IMPULSE 0 before the peer binding was validated")
        if self._at("exit_at", k):
            os._exit(3)                                        # simulated crash (test configuration only)
        if self._at("reject_at", k):
            raise ServiceError(f"injected rejection of window {k} (test configuration)")
        w = fr.meta.get("window_index")
        if type(w) is not int or w != k:                       # the state machine checks it itself 
            raise ServiceError(f"IMPULSE for window {w!r}, the open window is {k}")
        t0 = time.perf_counter()
        msg = batch_from_wire(self.X, fr.meta, fr.arrays)      # XPBD's constructors: finite, IDs, frame, units
        receipt = self.adapter.consume(msg)
        wall = time.perf_counter() - t0
        self.committed += 1                                    # committed: count it before anything else can fail
        self.open = None
        if self._at("bad_receipt_at", k):
            receipt = dict(receipt, committed_version=tuple(receipt["committed_version"]))   # not JSON-native
        try:
            digest = wire.json_digest(receipt)
        except Exception as e:
            self._log_receipt(k, dict(window_index=k, batch_id=str(receipt.get("batch_id")), receipt_digest=None,
                                      error=f"{type(e).__name__}: {e}"[:500], consume_s=wall))
            raise ServiceError(f"window {k} was committed but its receipt cannot be digested: {e}") from e
        self._log_receipt(k, dict(window_index=int(receipt["window_index"]), batch_id=receipt["batch_id"],
                                  receipt_digest=digest, committed_version=receipt["committed_version"],
                                  t_end=receipt["t_end"], consume_s=wall))
        if self._at("delay_receipt_at", k):
            time.sleep(float(self.fault.get("delay_s", 1.0)))
        return "RECEIPT", dict(receipt=receipt, receipt_digest=digest, consume_s=wall), None


def _error_meta(stage, e, core=None):
    retry = None
    if core is not None and core.adapter is not None:
        retry = not bool(core.adapter.failed)                  # XPBD: validation failures are retryable
    out = dict(stage=stage, error_type=type(e).__name__, message=str(e)[:2000], retryable_in_xpbd=retry,
               note="the joint protocol never retries: both sides stop")
    if core is not None:
        out.update(core.state())
    return out


def main(conn, cfg):
    """Entry of the spawned service process (top-level, importable without side effects)."""
    t_start = time.time()
    chan = wire.Channel(conn, max_frame=int(cfg.get("max_frame_bytes", wire.MAX_FRAME)), recv_modes=wire.TO_SERVICE)
    log = ServiceLog(cfg.get("log_dir"))
    send_timeout = float(cfg.get("send_timeout", 30.0))

    def stalled():                     # the driver stopped reading, or its frame never completed: record, end
        log.write_json("final.json", dict(reason=f"send or read watchdog: a frame was not taken or did not complete "
                                                 f"in time", wall_s=time.time() - t_start))
        os._exit(4)

    def send(mode, meta, arrays=None):
        return chan.send(mode, meta, arrays, timeout=send_timeout, on_stall=stalled)
    fault = {k: v for k, v in (cfg.get("fault") or {}).items()}
    unknown = set(fault) - set(FAULT_KEYS)
    hello = dict(service=SERVICE, wire=wire.WIRE, pid=os.getpid(), python=sys.version.split()[0], numpy=np.__version__,
                 platform=platform.platform(), xpbd_root=str(cfg.get("xpbd_root")), fault=fault,
                 dont_write_bytecode=bool(sys.dont_write_bytecode), flags_B=bool(sys.flags.dont_write_bytecode),
                 forbidden_modules_at_start=forbidden_loaded())
    final = dict(hello=hello, reason=None, frames_received=0, frames_sent=0)
    core = None
    try:
        if unknown:
            raise ServiceError(f"unknown fault keys {sorted(unknown)}")
        if hello["forbidden_modules_at_start"]:
            raise ServiceError(f"forbidden modules loaded in the service process: {hello['forbidden_modules_at_start']}")
        X, files = import_xpbd(cfg["xpbd_root"], cfg.get("expected_sha256"))
        hello.update(xpbd_files=files, forbidden_modules_after_import=forbidden_loaded())
        if hello["forbidden_modules_after_import"]:
            raise ServiceError(f"importing XPBD loaded forbidden modules: {hello['forbidden_modules_after_import']}")
        core = ServiceCore(X, cfg, log)
    except Exception as e:  # noqa: BLE001
        hello["error"] = f"{type(e).__name__}: {e}"
        log.write_json("hello.json", hello)
        try:
            send("ERROR", dict(_error_meta("hello", e), hello=hello))
        except wire.WireClosed:
            pass
        final.update(reason=f"start failed: {hello['error']}", wall_s=time.time() - t_start)
        log.write_json("final.json", final)
        chan.close()
        return
    log.write_json("hello.json", hello)
    send("HELLO", hello)
    idle = float(cfg.get("idle_timeout", 300.0))
    reason = None
    while True:
        try:
            fr = chan.recv(idle, strict_seq=core.failed is None, on_stall=stalled)
        except wire.WireTimeout:
            reason = f"idle watchdog: no request within {idle} s"
            break
        except wire.WireClosed:
            reason = "the driver closed the connection"
            break
        except wire.WireError as e:
            if core.failed is None:
                core.failed = f"wire: {e}"
            try:
                send("ERROR", _error_meta("wire", e, core))
            except wire.WireClosed:
                reason = "the driver closed the connection"
                break
            if chan.broken is not None:                        # out of step: terminal, no BYE 
                reason = f"wire out of step: {chan.broken}"
                break
            continue
        if fr.mode == "STOP":
            reason = f"STOP from the driver: {fr.meta.get('reason')}"
            try:
                send("BYE", dict(reason=reason, **core.state(), frames_received=chan.received,
                                      frames_sent=chan.sent + 1))
            except wire.WireClosed:
                pass
            break
        if core.failed is not None:
            try:
                send("ERROR", dict(_error_meta("stopping", ServiceError("only STOP is served after a failure"),
                                                    core)))
            except wire.WireClosed:
                break
            continue
        try:
            mode, meta, arrays = core.handle(fr)
        except Exception as e:  # noqa: BLE001
            core.failed = f"{fr.mode}: {type(e).__name__}: {e}"
            meta = _error_meta(fr.mode, e, core)
            meta["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
            try:
                send("ERROR", meta)
            except wire.WireClosed:
                reason = "the driver closed the connection"
                break
            continue
        try:
            send(mode, meta, arrays)
        except wire.WireClosed:
            reason = "the driver closed the connection"
            break
    final.update(reason=reason, frames_received=chan.received, frames_sent=chan.sent, wall_s=time.time() - t_start,
                 **core.state())
    log.write_json("final.json", final)
    chan.close()
