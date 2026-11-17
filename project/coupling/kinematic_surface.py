"""Isolated U13 preparation candidate: prescribed geometry, NOT a cloth solve.

No default driver uses this bridge. All vertices follow fixed/uniform constant
translation. Loads are audited, never applied to geometry. Real MPM, measured
reaction, work, mass dynamics, transport and bilateral acceptance are absent.
"""
from dataclasses import asdict, replace

import numpy as np

from .contracts import ClothSnapshot, _real
from .exchange import validate_surface
from .loads import LoadBatch, LoadLedger, SourceIdentity, _text


class KinematicSurfaceBridge:
    """Sequential host-only candidate compatible with ExchangeAdapter.

    `source` describes synthetic load provenance, not the surface trajectory.
    prescribed-only is a preparation safety gate, not the final U13 protocol.
    Snapshot read-only arrays provide API ownership, not a security boundary.
    """

    def __init__(self, snapshot, *, source, stream_id, h):
        if not isinstance(snapshot, ClothSnapshot):
            raise TypeError("initial state must be ClothSnapshot")
        if not isinstance(source, SourceIdentity) or source.kind != "prescribed":
            raise ValueError("candidate accepts only self-declared prescribed loads; real integration is blocked")
        if snapshot.step_index != 0 or snapshot.time != 0:
            raise ValueError("candidate requires initial step 0/time 0; resume is unsupported")
        if (not np.all(snapshot.pinned) or np.any(snapshot.inverse_mass != 0)
                or len(snapshot.vertex_ids) == 0):
            raise ValueError("entire nonempty surface must be prescribed: all pinned and inverse_mass=0")
        if not np.all(snapshot.velocity == snapshot.velocity[0]):
            raise ValueError("only spatially uniform constant translation is supported")
        validate_surface(snapshot, frozenset())
        self._snapshot = replace(snapshot)  # Own fresh arrays, never alias caller state.
        self._source = source
        self._stream_id = _text(stream_id, "stream_id")
        self._h = _real(h, "configured h", positive=True)
        self._ledger = LoadLedger(self._stream_id)
        self._failed = False
        self._advancing = False

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
            raise RuntimeError("kinematic candidate aborted; start a new stream")
        return self._snapshot

    @property
    def cut_edges(self):
        if self.failed:
            raise RuntimeError("kinematic candidate aborted")
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
            schema="xpbd-kinematic-candidate-v1", candidate_only=True,
            kinematic=True, free_dof_count=0, motion=("fixed" if not np.any(initial.velocity)
                                                    else "uniform_constant_translation"),
            source=asdict(self.source), source_authenticity="self_declared_synthetic",
            authenticated=False, stream_id=self.stream_id, batch_id=batch.batch_id,
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
        except Exception:
            if entered:
                # Host-only publication is restored, but the stream stays poisoned.
                # This is not a device/MPM rollback or permission to retry.
                self._snapshot = initial
                self._failed = True
            raise
        finally:
            self._advancing = False
