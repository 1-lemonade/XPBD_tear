"""Two-process driver of the MPM <-> XPBD exchange (P2-U3; MPM process, python orchestration).

    ep = MPMEndpoint(sim, scene=..., cfg=..., handshake=True)
    drv = CoupledDriver(ep, snapshot=snapshot_spec(ep, velocity_world=...), log_dir=run_dir / "coupling")
    result = drv.run(W)                      # both sides stop together on any failure; nothing is retried

Per window: OPEN -> OFFER -> ep.receive (window 0: PEER -> PEER_OK, ep.confirm_peer) -> ep.advance
-> IMPULSE -> XPBD consume -> RECEIPT -> ep.acknowledge -> next window. Handshake before window 0:
spawn + HELLO, REGISTER (the endpoint's frozen registration text, a fresh stream id) -> BOUND,
ep.confirm_registration(registration_digest).

Joint stop: an ERROR from the service, a watchdog timeout, a closed link, a wire error, a
malformed offer or an MPM stop rule ends the endpoint (reject when the delivered window was
refused, halt otherwise; a timeout while waiting for a receipt leaves the window delivered but
unacknowledged: the XPBD commit is unknown) and then the service (STOP -> BYE, reap). Recovery is a
new endpoint, solver, service process and stream.

replay_archive / offline_receipts: paired transport and offline replay:
the same frozen inputs of an archive (registration text, stream, h, dt, snapshot spec, batches)
through the transport into a fresh service process, and consumed by the same XPBD adapter in
one process without transport; offers, registration and receipts compared byte for byte (by
SHA-256). reconcile: the service's receipt log against the endpoint archive, row by row.

P2-U4 prefix fixes : the U13 registered bridge keeps
its topology fixed , so every offer and every receipt (start and committed version)
must carry the initial topology_version (the endpoint keeps the general "not decreasing" rule of U2/U3);
reconcile checks each log row's window index (consecutive from 0) and batch id against the archive and reports
a missing log, a missing final.json (crash or kill) and a log write failure separately. The published mesh
directory is $MPM_PUBLISHED_MESH when set (a host test package lives elsewhere and the shared directory has had two
spellings on the host), else the package data/cloth52-v1-20261109 directory.
"""
import hashlib
import json
import os
import pathlib
import sys
import time
import uuid

import numpy as np

from mpm.contact.live import SurfaceSnapshot
from . import archive
from . import wire
from . import xpbd_service
from .contracts import SCHEMA, SourceIdentity, WindowOffer
from .endpoint import EndpointStop
from .link import PeerError, XPBDLink
from .prescribed import receipt_digest as receipt_content_digest

WORK = pathlib.Path(__file__).resolve().parents[1]
EXCHANGE = WORK.parents[1]                                     # Work_place -> MPM_Couple -> the shared root
PUBLISHED_MESH = dict(path=pathlib.Path(os.environ.get("MPM_PUBLISHED_MESH")
                                        or WORK.parent / "data" / "cloth52-v1-20261109"),
                      mesh_sha256="e6b155ccf4bb7f18ad4ef5be058c291194815c7996af343f6161498dc4cfe198",
                      manifest_sha256="45a469a9c1d68655b9cacf0431cc153706bc3683b25d0ba6cac3e8a90bd5dca5")
# the XPBD mirror version this adapter was written against 
PINNED_XPBD = {
    "project/coupling/registered_kinematic.py": "9a64547ac25270945db588bfb25b2415dd8989d50726ca86fcbd2f10ada7b8ac",
    "project/coupling/kinematic_surface.py": "5cf1942dfcd6a762e104a3f1d55c64c5e4de766590b2ad0435867a84a9654312",
    "project/coupling/contracts.py": "02af24df3b377285c962f242082d102e32eb3e97e84e36f58f13d520fe122a95",
    "project/coupling/exchange.py": "da93067bb5d193ffae418d04f6f88c628e4985847483b2f18fdef4d709531807",
    "project/coupling/loads.py": "6998dbd955768ed1c72d452d0c0be338c582060d0d845bb4a41b7c333acfa019",
}
OFFER_ARRAYS = frozenset(("positions", "triangles", "vertex_ids", "triangle_ids", "velocity", "inverse_mass", "pinned"))
DRIVER_FAULTS = ("nan_impulse_at",)
# XPBD interface specification: the complete key set of the registered bridge's receipt (schema xpbd-registered-kinematic-v1,
# window_index added by ExchangeAdapter.consume); a change must come with a new schema / contract, never be ignored
RECEIPT_SCHEMA = "xpbd-registered-kinematic-v1"
RECEIPT_FIELDS = frozenset((
    "schema", "candidate_only", "kinematic", "free_dof_count", "motion", "source", "source_authenticity",
    "authenticated", "external_process_observed", "external_files_verified", "config_verified",
    "registration_digest", "registration_binding", "mpm_run_id", "mpm_epoch", "code_digest", "config_digest",
    "stream_id", "batch_id", "t_start", "t_end", "h", "previous_dt", "start_version", "committed_version",
    "vertex_ids", "impulse", "load_received", "load_applied_to_geometry", "support_balance",
    "reaction_measurement", "drive_work", "device_solve", "real_coupling", "window_index"))
SNAPSHOT_FIELDS = ("positions", "triangles", "vertex_ids", "triangle_ids", "velocity", "inverse_mass", "pinned",
                   "epoch", "step_index", "topology_version", "time", "dt")


class DriverError(RuntimeError):
    """A protocol inconsistency found by the driver (malformed offer, stated digest differs, ...)."""


def resolve_xpbd_root(root=None) -> pathlib.Path:
    """Explicit root, else $MPM_XPBD_WORK, else the enclosing XPBD candidate (no drive letters)."""
    if root:
        return pathlib.Path(root).resolve()
    if os.environ.get("MPM_XPBD_WORK"):
        return pathlib.Path(os.environ["MPM_XPBD_WORK"]).resolve()
    return EXCHANGE.resolve()


def snapshot_spec(ep, *, velocity_world=(0.0, 0.0, 0.0), mesh="file", mesh_path=None, mesh_sha256=None,
                  manifest_sha256=None, epoch=1, topology_version=None, previous_dt=None) -> dict:
    """The service's initial snapshot: published mesh (both digests) or inline arrays, all pinned, one
    native velocity = R^T v_world (the endpoint's transform). topology_version None: the mesh manifest's
     for a file mesh, 0 for an inline one."""
    v = ep.surface.rigid.vec_to_native(np.asarray(velocity_world, np.float64))
    spec = dict(mesh=mesh, velocity_native=[float(x) for x in v], velocity_world=[float(x) for x in velocity_world],
                epoch=int(epoch), topology_version=topology_version, previous_dt=previous_dt)
    if mesh == "file":
        spec.update(mesh_path=str(mesh_path or PUBLISHED_MESH["path"]),
                    mesh_sha256=mesh_sha256 or PUBLISHED_MESH["mesh_sha256"],
                    manifest_sha256=manifest_sha256 or PUBLISHED_MESH["manifest_sha256"])
        if topology_version is None:
            mp_ = pathlib.Path(spec["mesh_path"])
            man = wire.strict_loads((mp_ if mp_.is_dir() else mp_.parent).joinpath("manifest.json").read_bytes())
            spec["topology_version"] = int(man["topology_version"])
    elif mesh != "inline":
        raise ValueError("snapshot mesh must be 'file' or 'inline'")
    if spec["topology_version"] is None:
        spec["topology_version"] = 0
    spec["topology_version"] = int(spec["topology_version"])
    return spec


def offer_from_frame(fr, stream_id, topology_version=None):
    """MPM WindowOffer (mirror) with a SurfaceSnapshot from an OFFER frame, and its content digest.
    topology_version: the U13 fixed topology (every offer must carry it; P2-U4 prefix fix C)."""
    try:
        m = dict(fr.meta)
        stated = m.pop("offer_digest")
        A = fr.arrays
        if set(A) != OFFER_ARRAYS:
            raise DriverError(f"offer arrays {sorted(A)} != {sorted(OFFER_ARRAYS)}")
        digest = xpbd_service.payload_digest(m, A)
        if stated != digest:
            raise DriverError("offer digest stated by the service differs from the received content")
        if m["stream_id"] != stream_id:
            raise DriverError(f"offer stream {m['stream_id']!r} is not this run's {stream_id!r}")
        if m["cut_edges"]:
            raise DriverError("U13 offers carry no cut edges")
        if not A["pinned"].all() or A["inverse_mass"].any():
            raise DriverError("the U13 registered bridge offers an all-pinned surface with inverse_mass 0")
        if topology_version is not None and (type(m["topology_version"]) is not int
                                             or m["topology_version"] != topology_version):
            raise DriverError(f"offer topology_version {m['topology_version']!r} is not the fixed U13 topology "
                              f"{topology_version} ")
        S = SurfaceSnapshot(positions=A["positions"], triangles=A["triangles"], vertex_ids=A["vertex_ids"],
                            triangle_ids=A["triangle_ids"], velocity=A["velocity"], pinned=A["pinned"],
                            epoch=m["epoch"], step_index=m["step_index"], topology_version=m["topology_version"],
                            time=m["time"], dt=m["dt"])
        offer = WindowOffer(m["schema"], m["stream_id"], SourceIdentity(**m["source"]), m["window_index"], S,
                            m["particle_dt"], m["h"])
        if offer.substeps != m["substeps"]:
            raise DriverError(f"offer substeps {m['substeps']} != h / particle_dt = {offer.substeps}")
    except DriverError:
        raise
    except Exception as e:
        raise DriverError(f"malformed offer: {type(e).__name__}: {e}") from e
    return offer, digest


def need(meta, *keys):
    """Fields of a reply; a missing one is a protocol error (DriverError), not a KeyError ."""
    missing = [k for k in keys if k not in meta]
    if missing:
        raise DriverError(f"reply lacks {missing}")
    return [meta[k] for k in keys] if len(keys) > 1 else meta[keys[0]]


def check_receipt_fields(receipt):
    if type(receipt) is not dict or receipt.get("schema") != RECEIPT_SCHEMA:
        raise DriverError(f"receipt schema {receipt.get('schema') if type(receipt) is dict else None!r} is not "
                          f"{RECEIPT_SCHEMA!r}")
    if set(receipt) != RECEIPT_FIELDS:
        raise DriverError(f"receipt fields differ from {RECEIPT_SCHEMA}: missing {sorted(RECEIPT_FIELDS - set(receipt))}, "
                          f"extra {sorted(set(receipt) - RECEIPT_FIELDS)}")


def check_fixed_topology(receipt, topology_version):
    """U13 fixed topology : the receipt's start and committed
    versions keep the initial topology_version. The endpoint only requires "not decreasing" (general rule); the
    driver of the registered bridge requires equality."""
    for key in ("start_version", "committed_version"):
        v = receipt.get(key)
        if not (type(v) is list and len(v) == 3 and type(v[2]) is int and v[2] == topology_version):
            raise DriverError(f"receipt {key} {v!r} does not keep the fixed U13 topology_version {topology_version}")


def impulse_payload(msg, fault_nan=False):
    b = msg.batch
    meta = dict(schema=msg.schema, window_index=int(msg.window_index),
                batch=dict(stream_id=b.stream_id, source=b.source.as_dict(),
                           cloth_version=[int(v) for v in b.cloth_version], batch_id=b.batch_id,
                           t_start=float(b.t_start), t_end=float(b.t_end), coordinate_frame=b.coordinate_frame,
                           units=b.units))
    imp = np.asarray(b.impulse, "<f8")
    if fault_nan:                                              # transport corruption (test configuration only)
        imp = imp.copy()
        imp[0, 0] = np.nan
    return meta, dict(vertex_ids=np.asarray(b.vertex_ids, "<i8"), impulse=imp)


class CoupledDriver:
    def __init__(self, ep, *, snapshot, xpbd_root=None, mesh_arrays=None, log_dir=None, timeouts=None,
                 expected_sha256=None, fault=None, driver_fault=None, max_frame_bytes=wire.MAX_FRAME, stream_id=None,
                 expect_mesh=None):
        """expect_mesh: the mesh arrays MPM placed the sand against (the scene's own copy); BOUND must report
        the same mesh content digest and counts before any window . expected_sha256: the
        XPBD pin set (default PINNED_XPBD; never empty)."""
        if not ep.handshake["required"]:
            raise ValueError("the two-process driver needs MPMEndpoint(..., handshake=True)")
        if expected_sha256 is not None and not expected_sha256:
            raise ValueError("the XPBD pin set must not be empty")
        unknown = set(driver_fault or {}) - set(DRIVER_FAULTS)
        if unknown:
            raise KeyError(f"driver faults: unknown keys {sorted(unknown)}")
        self.ep = ep
        self.snapshot = dict(snapshot)
        self.mesh_arrays = mesh_arrays
        if self.snapshot["mesh"] == "inline" and mesh_arrays is None:
            raise ValueError("an inline mesh needs mesh_arrays")
        self.xpbd_root = resolve_xpbd_root(xpbd_root)
        self.log_dir = None if log_dir is None else pathlib.Path(log_dir)
        self.stream_id = stream_id or f"u3-{uuid.uuid4().hex}"   # a new stream every run
        self.fault = dict(fault or {})
        self.driver_fault = dict(driver_fault or {})
        self.service_cfg = dict(xpbd_root=str(self.xpbd_root),
                                expected_sha256=PINNED_XPBD if expected_sha256 is None else dict(expected_sha256),
                                log_dir=None if self.log_dir is None else str(self.log_dir / "xpbd_service"),
                                fault=self.fault, max_frame_bytes=int(max_frame_bytes))
        self.expect_mesh = None
        if expect_mesh is not None:
            self.expect_mesh = dict(mesh_content_digest=xpbd_service.mesh_content_digest(expect_mesh),
                                    n_vertices=int(len(expect_mesh["vertex_ids"])),
                                    n_triangles=int(len(expect_mesh["triangles"])))
        self.link = XPBDLink(self.service_cfg, timeouts)
        self.hello = self.bound = self.xpbd_stop = self.error = None
        self.rows = []

    # ------------------------------------------------------------ stop both sides
    def _stop_mpm(self, e):
        ep = self.ep
        commit_unknown = False
        if isinstance(e, PeerError):
            what = f"the XPBD service refused at {e.meta.get('stage')}: {e.meta.get('error_type')}: {e.meta.get('message')}"
            # a failure after XPBD's callback started (poison) or a commit beyond MPM's acknowledged count: the
            # window may be committed, so it is not a plain refusal 
            commit_unknown = bool(e.meta.get("adapter_failed")) or int(e.meta.get("committed_windows") or 0) > ep.acknowledged()
            if commit_unknown and ep.phase == "delivered":
                what += f"; XPBD adapter_failed = {e.meta.get('adapter_failed')}, committed_windows = " \
                        f"{e.meta.get('committed_windows')}: the commit of window {ep.k - 1} is unknown"
        elif isinstance(e, wire.WireTimeout):
            what = f"transport timeout: {e}"
            if ep.phase == "delivered":
                what += f"; the XPBD commit of window {ep.k - 1} is unknown (not retried)"
        elif isinstance(e, wire.WireClosed):
            what = f"transport closed: {e}" + (f"; the XPBD commit of window {ep.k - 1} is unknown"
                                               if ep.phase == "delivered" else "")
        elif isinstance(e, wire.WireError):
            what = f"wire error: {e}"
        else:
            what = f"{type(e).__name__}: {e}"
            if isinstance(e, DriverError) and ep.phase == "delivered":   # a receipt MPM refuses after XPBD's commit
                what += f"; window {ep.k - 1} was delivered, XPBD replied, MPM did not acknowledge it"
        try:
            if ep.stopped is None:
                if isinstance(e, PeerError) and ep.phase == "delivered" and not commit_unknown:
                    ep.reject(what)
                else:
                    ep.halt(f"S8 {what}")
        except EndpointStop:
            pass
        return what

    # ------------------------------------------------------------ run
    def _register(self):
        ep, t = self.ep, self.link.t
        meta = dict(registration_text=ep.registration_frozen, stream_id=self.stream_id, h=ep.H, particle_dt=ep.dt,
                    snapshot=self.snapshot)
        arrays = None
        if self.snapshot["mesh"] == "inline":
            a = self.mesh_arrays
            arrays = dict(positions=np.asarray(a["positions"], "<f8"), triangles=np.asarray(a["triangles"], "<i8"),
                          vertex_ids=np.asarray(a["vertex_ids"], "<i8"), triangle_ids=np.asarray(a["triangle_ids"], "<i8"))
        fr = self.link.request("REGISTER", meta, arrays, expect="BOUND", timeout=t["register"])
        self.bound = fr.meta
        digest, b, snap = need(fr.meta, "registration_digest", "binding", "snapshot")
        if type(b) is not dict or type(snap) is not dict:
            raise DriverError("BOUND binding and snapshot must be JSON objects")
        ep.confirm_registration(digest,
                                info=dict(stream_id=self.stream_id, snapshot_spec=self.snapshot,
                                          xpbd_snapshot=snap, wire=wire.WIRE,
                                          xpbd_files=(self.hello or {}).get("xpbd_files")))
        if b.get("stream_id") != self.stream_id or b.get("expected_source") != ep.source.as_dict():
            raise DriverError(f"BOUND binding {b} does not name this stream and the registered source")
        spec = self.snapshot
        want = dict(epoch=spec["epoch"], topology_version=spec["topology_version"], time=0.0,
                    previous_dt=float(ep.H if spec.get("previous_dt") is None else spec["previous_dt"]))
        if spec["mesh"] == "file":
            want.update(mesh_binding="file", mesh_sha256=spec["mesh_sha256"], manifest_sha256=spec["manifest_sha256"])
        if self.expect_mesh is not None:
            want.update(self.expect_mesh)
        bad = {k: (snap.get(k), v) for k, v in want.items() if snap.get(k) != v}
        if bad:                                                # before any window: the solver has not moved
            raise DriverError(f"BOUND snapshot differs from what MPM expects (got, want): {bad}")

    def _window(self, k, on_received=None):
        ep, link, t = self.ep, self.link, self.link.t
        row = dict(window=k)
        t0 = time.perf_counter()
        fr = link.request("OPEN", dict(window_index=k, h=ep.H, particle_dt=ep.dt), expect="OFFER", timeout=t["open"])
        t1 = time.perf_counter()
        offer, odig = offer_from_frame(fr, self.stream_id, self.snapshot["topology_version"])
        row.update(open_s=t1 - t0, offer_bytes=fr.nbytes, offer_digest=odig)
        ep.receive(offer)
        t2 = time.perf_counter()
        row["receive_s"] = t2 - t1
        if k == 0:
            link.request("PEER", dict(peer=ep.registration["peer"]), expect="PEER_OK", timeout=t["peer"])
            ep.confirm_peer(self.bound["binding"])
        if on_received is not None:                            # e.g. a scene's export setup after the first read
            on_received(k)
        t3 = time.perf_counter()
        row["peer_s"] = t3 - t2
        msg = ep.advance()
        t4 = time.perf_counter()
        row["advance_s"] = t4 - t3
        meta, arrays = impulse_payload(msg, fault_nan=self.driver_fault.get("nan_impulse_at") == k)
        fr = link.request("IMPULSE", meta, arrays, expect="RECEIPT", timeout=t["impulse"])
        t5 = time.perf_counter()
        receipt, stated = need(fr.meta, "receipt", "receipt_digest")
        try:
            computed = wire.json_digest(receipt)
        except wire.WireError as e:
            raise DriverError(f"the receipt is not JSON-native: {e}") from e
        if stated != computed:
            raise DriverError("receipt digest stated by the service differs from the received receipt")
        check_receipt_fields(receipt)
        check_fixed_topology(receipt, self.snapshot["topology_version"])
        ep.acknowledge(receipt)
        t6 = time.perf_counter()
        row.update(impulse_bytes=link.chan.last_sent_bytes, receipt_bytes=fr.nbytes, consume_rtt_s=t5 - t4,
                   xpbd_consume_s=fr.meta.get("consume_s"), acknowledge_s=t6 - t5, window_s=t6 - t0,
                   receipt_digest=ep.ledger[-1]["receipt_digest"])
        return row

    def run(self, W, *, on_received=None, on_window=None) -> dict:
        """on_received(k): after window k is read (and, at k = 0, the peer checked), before it is advanced;
        on_window(k): after window k is acknowledged. An EndpointStop from a hook is a joint stop."""
        status = "ok"
        t0 = time.perf_counter()
        try:
            self.hello = self.link.start()
            self._register()
            for k in range(int(W)):
                self.rows.append(self._window(k, on_received))
                if on_window is not None:                      # e.g. frame rows and checks; may stop the endpoint
                    on_window(k)
        except EndpointStop as e:
            status = f"stopped: {e.reason}"
            self.error = repr(e)
        except (PeerError, wire.WireTimeout, wire.WireClosed, wire.WireError, DriverError) as e:
            self.error = f"{type(e).__name__}: {e}"
            self._stop_mpm(e)
            status = f"stopped: {self.ep.stopped['reason']}"
        except BaseException as e:                             # anything else: still stop both sides, record, raise
            self.error = f"{type(e).__name__}: {e}"
            self._stop_mpm(e)
            self.xpbd_stop = self.link.stop(f"driver exception {self.error}")
            self._record(f"driver exception: {self.error}", W, t0)
            raise
        finally:
            if self.xpbd_stop is None:
                self.xpbd_stop = self.link.stop("run complete" if status == "ok" else status)
        xs = self.xpbd_stop
        if status == "ok" and not (xs.get("bye") and xs.get("exitcode") == 0 and not xs.get("terminated")):
            status = "incomplete: every window was acknowledged but the XPBD service did not stop cleanly (no BYE, " \
                     "nonzero exit or terminated)"                           # 
        return self._record(status, W, t0)

    def _record(self, status, W, t0) -> dict:
        ep = self.ep
        result = dict(status=status, stream_id=self.stream_id, windows_requested=int(W),
                      windows_delivered=ep.k, windows_acknowledged=ep.acknowledged(), awaiting_receipt=ep.awaiting_receipt,
                      mpm_stop=ep.stopped, xpbd_stop=self.xpbd_stop, error=self.error, hello=self.hello or self.link.hello,
                      bound=self.bound, registration_digest=ep.registration_digest, handshake=ep.handshake,
                      transport=dict(wire=wire.WIRE, timeouts=self.link.t, max_frame_bytes=self.service_cfg["max_frame_bytes"],
                                     process="multiprocessing spawn + duplex Pipe"),
                      fault=self.fault, driver_fault=self.driver_fault, windows=self.rows, wall_s=time.perf_counter() - t0)
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / "coupled.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        return result


# ------------------------------------------------------------------ replay into the loop, offline, reconcile
def snapshot_field_digests(meta, arrays) -> dict:
    """Per field of a committed snapshot: arrays by dtype, shape and SHA-256 of the little-endian C-order bytes,
    scalars by canonical JSON ."""
    out = {}
    for f in SNAPSHOT_FIELDS:
        if f in arrays:
            a = np.ascontiguousarray(arrays[f])
            out[f] = [a.dtype.str, list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]
        else:
            out[f] = wire.canonical_json(meta[f])
    return out


def first_mismatch(a, b, key="committed"):
    """(window, field) of the first difference between two per-window field-digest lists, or None."""
    for k, (x, y) in enumerate(zip(a[key], b[key])):
        for f in SNAPSHOT_FIELDS:
            if x[f] != y[f]:
                return dict(window=k, field=f)
    if len(a[key]) != len(b[key]):
        return dict(window=min(len(a[key]), len(b[key])), field="(count)")
    return None


def _inputs_from_archive(dir_):
    a = archive.read(dir_)
    man, prov, arr, s0 = a["manifest"], a["provenance"], a["arrays"], a["snapshot0"]
    reg = prov.get("registration_frozen")
    hs = (prov.get("handshake") or {}).get("registration_confirmed")
    if reg is None or hs is None:
        raise ValueError("replay: the archive has no frozen registration or handshake record (not a U3 run)")
    acked = man.get("acknowledged") or []
    n = next((k for k, x in enumerate(acked) if not x), len(acked))
    batches = []
    for k in range(n):
        meta = dict(schema=SCHEMA, window_index=k,
                    batch=dict(stream_id=man["stream_id"], source=man["source"],
                               cloth_version=[int(v) for v in arr["cloth_version"][k]], batch_id=man["batch_ids"][k],
                               t_start=float(arr["t_start"][k]), t_end=float(arr["t_end"][k]), coordinate_frame="world",
                               units="SI"))
        batches.append((meta, dict(vertex_ids=np.asarray(arr["vertex_ids"][k], "<i8"),
                                   impulse=np.asarray(arr["impulse"][k], "<f8"))))
    mesh = {k: s0[k] for k in ("positions", "triangles", "vertex_ids", "triangle_ids")}
    rt = prov["registration"]["runtime"]
    env = dict(python=rt["python"], numpy=rt["numpy"], xpbd_files=hs.get("xpbd_files"))
    return dict(registration_text=reg, stream_id=man["stream_id"], h=float(man["h"]), dt=float(man["dt"]),
                spec=hs["snapshot_spec"], peer=prov["registration"]["peer"], batches=batches, mesh=mesh, snapshot0=s0,
                environment=env)


def check_environment(inp, python, numpy_version, xpbd_files):
    """The sealed code and environment versions of a replay : the Python and NumPy
    versions and the XPBD files of the original run must be those of the replaying process ."""
    env = inp.get("environment")
    if env is None:
        return None
    got = dict(python=python, numpy=numpy_version, xpbd_files=xpbd_files)
    bad = {k: (env[k], got[k]) for k in env if env[k] is not None and env[k] != got[k]}
    if bad:
        raise DriverError(f"replay environment differs from the sealed one (sealed, now): {bad}")
    return got


def inputs_from_vectors(path, case):
    """Sealed inputs of one case of XPBD's contract-vectors.json (U13-u3-support-20261116, a small host fixture)."""
    d = wire.strict_loads(pathlib.Path(path).read_bytes())
    c = next(x for x in d["cases"] if x["name"] == case)
    S = c["initial_snapshot"]
    v = np.asarray(S["velocity"], np.float64)
    if not (v == v[0]).all():
        raise ValueError("vector snapshot velocity is not uniform")
    spec = dict(mesh="inline", velocity_native=[float(x) for x in v[0]], epoch=S["epoch"],
                topology_version=S["topology_version"], previous_dt=S["dt"])
    batches = []
    for w in c["windows"]:
        iw = w["impulse_window"]
        b = dict(iw["batch"])
        arrays = dict(vertex_ids=np.asarray(b.pop("vertex_ids"), "<i8"),
                      impulse=np.asarray(b.pop("impulse"), "<f8").reshape(-1, 3))
        batches.append((dict(schema=iw["schema"], window_index=iw["window_index"], batch=b), arrays))
    mesh = dict(positions=np.asarray(S["positions"], "<f8"), triangles=np.asarray(S["triangles"], "<i8"),
                vertex_ids=np.asarray(S["vertex_ids"], "<i8"), triangle_ids=np.asarray(S["triangle_ids"], "<i8"))
    host = dict(receipt_digests=[w["receipt"]["sha256"] for w in c["windows"]],
                receipt_canonical=[w["receipt"]["canonical_json"] for w in c["windows"]],
                committed=[snapshot_field_digests(
                    {k: w["committed_snapshot"][k] for k in ("epoch", "step_index", "topology_version", "time", "dt")},
                    dict(positions=np.asarray(w["committed_snapshot"]["positions"], "<f8"),
                         triangles=np.asarray(w["committed_snapshot"]["triangles"], "<i8"),
                         vertex_ids=np.asarray(w["committed_snapshot"]["vertex_ids"], "<i8"),
                         triangle_ids=np.asarray(w["committed_snapshot"]["triangle_ids"], "<i8"),
                         velocity=np.asarray(w["committed_snapshot"]["velocity"], "<f8"),
                         inverse_mass=np.asarray(w["committed_snapshot"]["inverse_mass"], "<f8"),
                         pinned=np.asarray(w["committed_snapshot"]["pinned"], "|b1")))
                    for w in c["windows"]],
                registration_canonical=d["registration"]["canonical_json"], registration_sha256=d["registration"]["sha256"])
    return dict(registration_text=d["registration"]["canonical_json"], stream_id=c["stream_id"], h=float(d["h"]),
                dt=float(d["particle_dt"]), spec=spec, peer=c["binding"]["value"], batches=batches, mesh=mesh,
                snapshot0=None, host=host)


def _same_as_snapshot0(A, m, s0) -> bool:
    keys = ("positions", "velocity", "triangles", "vertex_ids", "triangle_ids", "pinned")
    return (all(np.asarray(A[k]).tobytes() == np.asarray(s0[k]).tobytes() and np.asarray(A[k]).dtype == s0[k].dtype
                for k in keys)
            and [m["epoch"], m["step_index"], m["topology_version"]] == [int(v) for v in s0["version"]]
            and m["time"] == float(s0["time"]) and m["dt"] == float(s0["dt"]))


def run_transport(inp, *, xpbd_root=None, log_dir=None, timeouts=None, expected_sha256=None) -> dict:
    """Path A: sealed inputs through the transport into a fresh XPBD service process (original stream and
    registration reused, never a new identity). One extra OPEN after the last window reads the last committed
    snapshot (the window is left open; the service is then stopped)."""
    cfg = dict(xpbd_root=str(resolve_xpbd_root(xpbd_root)),
               expected_sha256=PINNED_XPBD if expected_sha256 is None else expected_sha256,
               log_dir=None if log_dir is None else str(log_dir), fault={})
    link = XPBDLink(cfg, timeouts)
    out = dict(path="transport", offer_digests=[], committed=[], receipt_digests=[], receipt_content_digests=[],
               receipt_canonical=[], snapshot0_equal=None, registration_digest=None)
    try:
        out["hello"] = h_ = link.start()
        out["environment"] = check_environment(inp, h_["python"], h_["numpy"], h_.get("xpbd_files"))
        t = link.t
        arrays = None
        if inp["spec"]["mesh"] == "inline":
            arrays = {k: np.asarray(v, "<f8" if k == "positions" else "<i8") for k, v in inp["mesh"].items()}
        fr = link.request("REGISTER", dict(registration_text=inp["registration_text"], stream_id=inp["stream_id"],
                                           h=inp["h"], particle_dt=inp["dt"], snapshot=inp["spec"]), arrays,
                          expect="BOUND", timeout=t["register"])
        out["registration_digest"] = fr.meta["registration_digest"]
        W = len(inp["batches"])
        for k in range(W + 1):
            fo = link.request("OPEN", dict(window_index=k, h=inp["h"], particle_dt=inp["dt"]), expect="OFFER",
                              timeout=t["open"])
            m = dict(fo.meta)
            stated = m.pop("offer_digest")
            dig = xpbd_service.payload_digest(m, fo.arrays)
            if dig != stated:
                raise DriverError("replay: offer digest differs from the received content")
            if k > 0:
                out["committed"].append(snapshot_field_digests(m, fo.arrays))
            if k == W:
                break
            out["offer_digests"].append(dig)
            if k == 0:
                if inp.get("snapshot0") is not None:
                    out["snapshot0_equal"] = _same_as_snapshot0(fo.arrays, m, inp["snapshot0"])
                link.request("PEER", dict(peer=inp["peer"]), expect="PEER_OK", timeout=t["peer"])
            meta, arr = inp["batches"][k]
            fr = link.request("IMPULSE", meta, arr, expect="RECEIPT", timeout=t["impulse"])
            if fr.meta["receipt_digest"] != wire.json_digest(fr.meta["receipt"]):
                raise DriverError("replay: stated receipt digest differs from the receipt")
            out["receipt_digests"].append(fr.meta["receipt_digest"])
            out["receipt_canonical"].append(wire.canonical_json(fr.meta["receipt"]))
            out["receipt_content_digests"].append(receipt_content_digest(fr.meta["receipt"]))
    finally:
        out["stop"] = link.stop("replay complete")
    return out


def run_offline(inp, *, xpbd_root=None, expected_sha256=None, keep_snapshots=False) -> dict:
    """Path B: the same sealed inputs consumed by the same XPBD adapter in this process, no transport.
    keep_snapshots: also return the opening snapshot and every committed snapshot (positions, velocity,
    versions, time, dt) for a motion check (P2-U4 host package)."""
    X, files = xpbd_service.import_xpbd(resolve_xpbd_root(xpbd_root),
                                        PINNED_XPBD if expected_sha256 is None else expected_sha256)
    env = check_environment(inp, sys.version.split()[0], np.__version__, files)
    spec = inp["spec"]
    if spec["mesh"] == "file":
        mesh = xpbd_service.load_mesh(spec["mesh_path"], spec["mesh_sha256"], spec["manifest_sha256"])["arrays"]
    else:
        mesh = inp["mesh"]
    snap = xpbd_service.build_snapshot(X, spec, mesh, inp["h"])
    reg = xpbd_service.parse_registration(inp["registration_text"])
    adapter = X["create"](snap, registration=reg, stream_id=inp["stream_id"], h=inp["h"], particle_dt=inp["dt"])
    out = dict(path="offline", registration_digest=adapter.registration_digest, offer_digests=[], committed=[],
               receipt_digests=[], receipt_content_digests=[], receipt_canonical=[], xpbd_files=files,
               snapshot0_equal=None, environment=env)
    if keep_snapshots:
        out["snapshots"] = [_kept(adapter.snapshot)]
    for k, (bm, ba) in enumerate(inp["batches"]):
        offer = adapter.open_window(h=inp["h"], particle_dt=inp["dt"])
        meta, arrays = xpbd_service.offer_payload(offer)
        out["offer_digests"].append(xpbd_service.payload_digest(meta, arrays))
        if k == 0:
            if inp.get("snapshot0") is not None:
                out["snapshot0_equal"] = _same_as_snapshot0(arrays, meta, inp["snapshot0"])
            adapter.validate_peer(inp["peer"])
        receipt = adapter.consume(xpbd_service.batch_from_wire(X, bm, ba))
        out["receipt_digests"].append(wire.json_digest(receipt))
        out["receipt_canonical"].append(wire.canonical_json(receipt))
        out["receipt_content_digests"].append(receipt_content_digest(receipt))
        cm, ca = _snapshot_payload(adapter.snapshot)
        out["committed"].append(snapshot_field_digests(cm, ca))
        if keep_snapshots:
            out["snapshots"].append(_kept(adapter.snapshot))
    return out


def _kept(S):
    return dict(positions=np.array(S.positions, np.float64), velocity=np.array(S.velocity, np.float64),
                epoch=int(S.epoch), step_index=int(S.step_index), topology_version=int(S.topology_version),
                time=float(S.time), dt=float(S.dt))


def _snapshot_payload(S):
    meta = dict(epoch=int(S.epoch), step_index=int(S.step_index), topology_version=int(S.topology_version),
                time=float(S.time), dt=float(S.dt))
    arrays = dict(positions=np.asarray(S.positions, "<f8"), triangles=np.asarray(S.triangles, "<i8"),
                  vertex_ids=np.asarray(S.vertex_ids, "<i8"), triangle_ids=np.asarray(S.triangle_ids, "<i8"),
                  velocity=np.asarray(S.velocity, "<f8"), inverse_mass=np.asarray(S.inverse_mass, "<f8"),
                  pinned=np.asarray(S.pinned, "|b1"))
    return meta, arrays


def replay_archive(dir_, **kw) -> dict:
    return run_transport(_inputs_from_archive(dir_), **kw)


def offline_receipts(dir_, **kw) -> dict:
    return run_offline(_inputs_from_archive(dir_), **kw)


def reconcile(archive_dir, service_log_dir) -> dict:
    """The service's receipt log against the endpoint archive, row by row :
    - every log row is strict JSON with window_index == its row number (consecutive from 0) and the batch_id
      of that window in the MPM archive;
    - the receipt digests are equal on MPM's acknowledged prefix;
    - XPBD may hold exactly one more committed window, and only the one the endpoint stopped waiting for
      (awaiting_receipt == the acknowledged count; its row has that window index and batch id); it is reported,
      never counted as acknowledged;
    - final.json, when present, states the same committed count as the log;
    - log_condition: "final" (final.json present), "no_final" (no final.json: crash, kill or an exit without
      it; not complete, not an inconsistency by itself), "log_missing" (windows were committed or acknowledged
      but there is no receipt log), "log_write_failed" (the service recorded a receipt row it could not write).
    consistent is the self-consistency of the two records, not a complete run and not an authentication."""
    man = archive.read(archive_dir)["manifest"]
    d = pathlib.Path(service_log_dir)
    log_path, final_path = d / "receipts.jsonl", d / "final.json"
    issues = []
    rows = []
    if log_path.exists():
        for i, line in enumerate(log_path.read_text(encoding="utf-8").splitlines()):
            try:
                r = wire.strict_loads(line)
            except wire.WireError as e:
                issues.append(f"receipt log line {i} is not strict JSON: {e}")
                break                                          # nothing after an unreadable row is used
            if type(r) is not dict:
                issues.append(f"receipt log line {i} is not a JSON object")
                break
            rows.append(r)
    final = None
    if final_path.exists():
        try:
            final = wire.strict_loads(final_path.read_text(encoding="utf-8"))
        except wire.WireError as e:
            issues.append(f"final.json is not strict JSON: {e}")
    acked = man.get("acknowledged") or []
    n = next((k for k, x in enumerate(acked) if not x), len(acked))
    batch_ids = list(man.get("batch_ids") or [])
    awaiting = man.get("awaiting_receipt")
    for i, r in enumerate(rows):                               # row by row: window index and batch id
        w = r.get("window_index")
        if type(w) is not int or w != i:
            issues.append(f"receipt log row {i}: window_index {w!r}, expected {i}")
        if i >= len(batch_ids):
            issues.append(f"receipt log row {i}: MPM delivered only {len(batch_ids)} windows")
        elif r.get("batch_id") != batch_ids[i]:
            issues.append(f"receipt log row {i}: batch_id {r.get('batch_id')!r} is not the archive's {batch_ids[i]!r}")
    mpm = list(man["receipt_digests"][:n])
    xp = [r.get("receipt_digest") for r in rows]
    extra = len(xp) - len(mpm)
    prefix = len(xp) >= len(mpm) and xp[:len(mpm)] == mpm
    if not prefix:
        issues.append(f"receipt digests differ on MPM's acknowledged prefix of {n} windows (log rows {len(xp)})")
    note = None
    awaited_row_ok = (extra == 1 and awaiting == n and type(rows[n].get("window_index")) is int
                      and rows[n]["window_index"] == n and n < len(batch_ids) and rows[n].get("batch_id") == batch_ids[n])
    if awaited_row_ok:
        note = f"XPBD committed window {n}; the endpoint stopped before its receipt was acknowledged (commit unknown)"
    elif extra == 1 and awaiting == n:
        pass                                                   # the awaited row itself is wrong: issues above
    elif extra > 0:
        issues.append(f"XPBD logged {len(xp)} committed windows, MPM acknowledged {n} (awaiting {awaiting!r}): "
                      f"more than the one awaited window")
    log_errors = list((final or {}).get("log_errors") or [])
    committed_final = None if final is None else final.get("committed_windows")
    if log_errors:
        condition = "log_write_failed"
        issues.append(f"the service could not write {len(log_errors)} receipt log row(s): {log_errors[0]}")
    elif not log_path.exists() and (n > 0 or (committed_final or 0) > 0):
        condition = "log_missing"
        issues.append(f"no receipt log, yet MPM acknowledged {n} and final.json states {committed_final!r} committed")
    elif final is None:
        condition = "no_final"
    else:
        condition = "final"
    if committed_final is not None and committed_final != len(xp) and not log_errors:
        issues.append(f"final.json says {committed_final} committed, the receipt log has {len(xp)}")
    if committed_final is not None and log_errors and committed_final != len(xp) + len(log_errors):
        issues.append(f"final.json says {committed_final} committed, the log has {len(xp)} rows and "
                      f"{len(log_errors)} unwritten")
    return dict(consistent=not issues, prefix_equal=bool(prefix), mpm_acknowledged=n, xpbd_committed=len(xp),
                extra_xpbd_windows=extra, awaiting_receipt=awaiting, xpbd_final_reason=None if final is None else
                final.get("reason"), xpbd_final_present=final is not None, receipt_log_present=log_path.exists(),
                log_condition=condition, issues=issues, note=note)
