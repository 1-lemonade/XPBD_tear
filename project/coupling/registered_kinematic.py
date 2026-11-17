"""Registered host-only prescribed geometry, NOT a process authenticator.

This opt-in entry binds a declared real load source to a fixed stream and time
schedule. It never starts MPM, verifies external files, measures reactions, or
claims a real joint run. The old prescribed-only candidate remains unchanged.
"""
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

import numpy as np

from .contracts import ClothSnapshot, _integer, _real
from .exchange import ExchangeAdapter, WindowOffer, SCHEMA, validate_surface
from .loads import LoadBatch, LoadLedger, SourceIdentity, _text


def _json(value):
    """Finite JSON only; canonicalization is integrity, not authentication."""
    def check(v):
        if type(v) is dict:
            for k, child in v.items():
                if type(k) is not str:
                    raise ValueError("JSON object keys must be strings")
                k.encode("utf-8")
                check(child)
        elif type(v) is list:
            for child in v:
                check(child)
        elif type(v) is str:
            v.encode("utf-8")
        elif v is None or type(v) in (bool, int):
            pass
        elif type(v) is float and math.isfinite(v):
            pass
        else:
            raise ValueError("only finite JSON-native values are accepted")
    try:
        check(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("registration must be finite JSON") from exc


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _positive_int(value, name):
    value = _integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _code_digest(files):
    """SHA-256 of path<TAB>digest<LF> records sorted by path, UTF-8."""
    data = "".join(f"{k}\t{v}\n" for k, v in sorted(files.items()))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class _RegisteredKinematicSurfaceBridge:
    """Explicit registered source; all geometry remains prescribed.

    The full canonical registration is deep-owned. Its assertions about code,
    config and process execution are NOT independently authenticated.
    """

    def __init__(self, snapshot, *, registration, stream_id, h, particle_dt):
        if not isinstance(snapshot, ClothSnapshot):
            raise TypeError("initial state must be ClothSnapshot")
        if snapshot.step_index != 0 or snapshot.time != 0:
            raise ValueError("initial step/time must be zero; no resume")
        if (not len(snapshot.vertex_ids) or not np.all(snapshot.pinned)
                or np.any(snapshot.inverse_mass != 0)):
            raise ValueError("all topology vertices must be pinned with inverse_mass=0")
        if not np.all(snapshot.velocity == snapshot.velocity[0]):
            raise ValueError("only uniform constant translation is supported")
        validate_surface(snapshot, frozenset())
        stream_id = _text(stream_id, "stream_id")
        h = _real(h, "h", positive=True)
        particle_dt = _real(particle_dt, "particle_dt", positive=True)
        if not isinstance(registration, dict):
            raise TypeError("registration must be a JSON object")
        owned = json.loads(_json(registration))
        required = {"schema", "source", "code_digest", "code_files", "config_digest",
                    "mpm_run_id", "mpm_epoch", "window", "peer"}
        if not required.issubset(owned):
            raise ValueError("incomplete registration")
        if owned["schema"] != "mpm-endpoint-registration-v1":
            raise ValueError("unknown registration schema")
        src = owned["source"]
        if not isinstance(src, dict) or set(src) != {"source_id", "kind", "version"}:
            raise ValueError("invalid registered source")
        source = SourceIdentity(**src)
        if source.kind != "real_backend":
            raise ValueError("registered entry requires a real_backend declaration")
        code_digest = _sha(owned["code_digest"], "code_digest")
        if source.version != int(code_digest[:8], 16):
            raise ValueError("source version differs from code digest prefix")
        files = owned["code_files"]
        if not isinstance(files, dict) or not files:
            raise ValueError("code_files must be a nonempty path-to-digest mapping")
        for name, digest in files.items():
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                    or any(ord(ch) < 32 for ch in name)
                    or PurePosixPath(name).is_absolute()
                    or any(p in ("", ".", "..") for p in name.split("/"))):
                raise ValueError("code_files paths must be canonical relative POSIX paths")
            _sha(digest, "code file digest")
        if _code_digest(files) != code_digest:
            raise ValueError("code_files aggregate differs from code_digest")
        _sha(owned["config_digest"], "config_digest")
        _text(owned["mpm_run_id"], "mpm_run_id")
        _positive_int(owned["mpm_epoch"], "mpm_epoch")
        w = owned["window"]
        if not isinstance(w, dict) or not {"dt", "n_sub", "H_ns"}.issubset(w):
            raise ValueError("incomplete registered window")
        n = _positive_int(w["n_sub"], "n_sub")
        ns = _positive_int(w["H_ns"], "H_ns")
        dt = _real(w["dt"], "registered dt", positive=True)
        if (dt != particle_dt or Fraction(str(h)) * 10**9 != ns
                or Fraction(str(h)) / Fraction(str(particle_dt)) != n):
            raise ValueError("registered clock differs from h/particle_dt")
        # Reuse full existing tick representability and surface validation.
        WindowOffer(SCHEMA, stream_id, source, 0, snapshot, particle_dt, h)
        self._source = source
        self._stream_id = stream_id
        self._h = h
        self._particle_dt = particle_dt
        self._snapshot = replace(snapshot)
        self._registration_json = _json(owned)
        self._registration_digest = _digest(owned)
        self._binding = dict(stream_id=stream_id, expected_source=asdict(source),
                             snapshot_epoch=snapshot.epoch,
                             topology_version=snapshot.topology_version,
                             n_vertices=len(snapshot.vertex_ids), n_triangles=len(snapshot.triangles))
        if owned["peer"] is not None:
            self.validate_peer(owned["peer"])
        self._ledger = LoadLedger(stream_id)
        self._failed = False
        self._advancing = False

    @property
    def registration(self):
        return json.loads(self._registration_json)

    @property
    def registration_digest(self):
        return self._registration_digest

    @property
    def binding(self):
        return json.loads(_json(self._binding))

    def validate_peer(self, peer):
        """Check later handshake binding without rewriting the registration."""
        if not isinstance(peer, dict) or _json(peer) != _json(self._binding):
            raise ValueError("peer binding differs from registered local stream")
        return True

    @property
    def source(self):
        return self._source

    @property
    def stream_id(self):
        return self._stream_id

    @property
    def failed(self):
        return self._failed or self._ledger.poisoned

    @property
    def snapshot(self):
        if self.failed:
            raise RuntimeError("registered kinematic bridge aborted; start a new stream")
        return self._snapshot

    @property
    def cut_edges(self):
        if self.failed:
            raise RuntimeError("registered kinematic bridge aborted")
        return frozenset()

    def validate_h(self, h):
        h = _real(h, "next h", positive=True)
        if h != self._h:
            raise ValueError("candidate requires its explicitly configured fixed h")
        return h

    def _commit(self, state):
        """Only state-changing hook, inside LoadLedger's poisoned callback."""
        self._snapshot = state

    def advance(self, batch, h):
        if self._advancing:
            raise RuntimeError("reentrant kinematic advance is unsupported")
        initial = self.snapshot
        h = self.validate_h(h)
        if not isinstance(batch, LoadBatch):
            raise TypeError("advance requires LoadBatch")
        # Prevalidation is BEFORE ledger.consume and BEFORE callback entry.
        rows = batch.validate_for(initial, stream_id=self.stream_id,
                                  source=self.source, h=h, capacity=len(initial.vertex_ids))
        if (len(rows) != len(initial.vertex_ids)
                or not np.array_equal(np.sort(rows), np.arange(len(rows)))):
            raise ValueError("dense batch must include every topology vertex exactly once, including zero rows")
        dense = np.zeros_like(initial.positions)
        dense[rows] = batch.impulse
        # Stage and validate all fallible derived data before any ledger mutation.
        with np.errstate(over="ignore", invalid="ignore"):
            end = replace(initial, positions=initial.positions + h * initial.velocity,
                          time=batch.t_end, dt=h, step_index=initial.step_index + 1)
        validate_surface(end, frozenset())
        receipt = dict(
            schema="xpbd-registered-kinematic-v1", candidate_only=False,
            kinematic=True, free_dof_count=0, motion=("fixed" if not np.any(initial.velocity)
                                                    else "uniform_constant_translation"),
            source=asdict(self.source), source_authenticity="unverified_real_backend_registration",
            authenticated=False, external_process_observed=False,
            external_files_verified=False, config_verified=False,
            registration_digest=self.registration_digest, registration_binding=self.binding,
            mpm_run_id=self.registration["mpm_run_id"],
            mpm_epoch=self.registration["mpm_epoch"],
            code_digest=self.registration["code_digest"],
            config_digest=self.registration["config_digest"],
            stream_id=self.stream_id, batch_id=batch.batch_id,
            t_start=batch.t_start, t_end=batch.t_end, h=h, previous_dt=initial.dt,
            start_version=list(initial.version), committed_version=list(end.version),
            vertex_ids=initial.vertex_ids.tolist(), impulse=dense.tolist(),
            load_received=True, load_applied_to_geometry=False,
            support_balance=None, reaction_measurement="not_measured",
            drive_work=None, device_solve=False, real_coupling=False)
        entered = False

        def publish(_rows, _impulse):
            nonlocal entered
            entered = True
            self._commit(end)
            return receipt

        self._advancing = True
        try:
            return self._ledger.consume(batch, initial, source=self.source, h=h,
                                        capacity=len(initial.vertex_ids), apply=publish)
        except BaseException:
            if entered:
                # Host-only publication is restored, but the stream stays poisoned.
                # This is not a device/MPM rollback or permission to retry.
                self._snapshot = initial
                self._failed = True
            raise
        finally:
            self._advancing = False



class _RegisteredAdapter(ExchangeAdapter):
    @property
    def registration(self):
        return self._bridge.registration

    @property
    def registration_digest(self):
        return self._bridge.registration_digest

    @property
    def binding(self):
        return self._bridge.binding

    def validate_peer(self, peer):
        return self._bridge.validate_peer(peer)

    def open_window(self, *, h, particle_dt):
        dt = _real(particle_dt, "particle_dt", positive=True)
        if dt != self._bridge._particle_dt:
            raise ValueError("particle_dt differs from the registered schedule")
        return super().open_window(h=h, particle_dt=dt)


def create_registered_kinematic_adapter(snapshot, *, registration, stream_id, h, particle_dt):
    """Public U3 construction entry, no process or transport side effects.

    registration is a JSON-native dict, not JSON text. Caller must reject
    duplicate keys while parsing text. Digests use Python 3 json.dumps with
    sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False,
    then UTF-8. Floats use Python's JSON representation; U3 must agree on these
    exact bytes, not invent a language-independent canonicalization.
    Code digest hashes UTF-8 path<TAB>sha256<LF> lines sorted by path. Registration digest hashes ALL fields.
    Real source claims are not independently authenticated. The caller owns
    unique stream allocation and the provenance handshake before transport.
    """
    bridge = _RegisteredKinematicSurfaceBridge(
        snapshot, registration=registration, stream_id=stream_id, h=h,
        particle_dt=particle_dt)
    return _RegisteredAdapter(bridge, bridge.source)
