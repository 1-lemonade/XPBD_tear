"""MPM-side stand-in for the XPBD U13 peer (P2-U2): prescribed snapshots with the semantics of
XPBD's KinematicSurfaceBridge + ExchangeAdapter (read-only reference, not XPBD code).

Semantics kept:
  the whole surface is pinned (inverse mass 0) and moves with one constant velocity for the
  whole stream (zero = fixed); h is fixed and h / particle_dt an integer; the stream starts at
  step 0, time 0; one window open at a time, window_index from 0;
  consume() checks schema, window_index, stream and source, cloth_version == S_k.version,
  t_start == S_k.time, t_end == S_k.time + h, every ID known, dense (every vertex exactly once,
  zero rows included), finite; then S_{k+1}: positions + h velocity, time = t_end, dt = h,
  step + 1. The impulse is recorded, never applied to the geometry. A failed check before the
  commit changes nothing (XPBD: validation failures are retryable). After the commit the
  adapter's end checks run (epoch unchanged, step_index + 1, time == t_end, dt == h,
  topology_version not decreasing); a failure there poisons the peer for good (XPBD: a failure
  after the bridge applied the load is terminal; recovery is a new stream). Here the next state
  comes from the stand-in's own _next_state, so the checks only guard subclasses and later
  external sources of the next state; in XPBD they guard an independent bridge.
Difference (the minimal interface difference handed back to XPBD): XPBD's bridge only accepts
`prescribed` loads; this stand-in is constructed with the source registered by the MPM
endpoint (`real_backend`) and accepts exactly that one.
Option shuffle_rows (seed): every window offers the same topology with permuted rows (XPBD
does not promise stable rows across windows); receipts are in the offered row order.
Receipts are JSON-native (lists, no tuples or arrays) like XPBD's (P2-U3); `digest` is the
row-order independent content digest of U2 (receipt_digest), the endpoint does not read it.
"""
import hashlib

import numpy as np

from mpm.contact.live import SurfaceSnapshot
from .contracts import SCHEMA, ImpulseWindow, SourceIdentity, WindowOffer, substeps_of


def receipt_digest(receipt: dict) -> str:
    """Row-order independent: the impulse rows are hashed in ascending stable vertex ID."""
    ids = np.asarray(receipt["vertex_ids"], np.int64)
    o = np.argsort(ids, kind="stable")
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ids[o], dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(np.asarray(receipt["impulse"], np.float64)[o], dtype="<f8").tobytes())
    h.update(repr((receipt["window_index"], tuple(receipt["start_version"]), tuple(receipt["committed_version"]),
                   float(receipt["t_start"]).hex(), float(receipt["t_end"]).hex())).encode())
    return h.hexdigest()


class PrescribedPeer:
    def __init__(self, positions, triangles, vertex_ids, triangle_ids, velocity, *, h, stream_id,
                 expected_source, epoch=1, topology_version=0, shuffle_rows=None):
        if not isinstance(expected_source, SourceIdentity):
            raise TypeError("expected_source must be SourceIdentity")
        X = np.asarray(positions, np.float64)
        v = np.asarray(velocity, np.float64)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError("one constant velocity (3,) for the whole surface and stream")
        N = len(X)
        self._h = float(h)
        self._stream = str(stream_id)
        self._source = expected_source
        self._S = SurfaceSnapshot(positions=X.copy(), triangles=np.asarray(triangles, np.int64).copy(),
                                  vertex_ids=np.asarray(vertex_ids, np.int64).copy(),
                                  triangle_ids=np.asarray(triangle_ids, np.int64).copy(),
                                  velocity=np.tile(v, (N, 1)), pinned=np.ones(N, bool), epoch=int(epoch),
                                  step_index=0, topology_version=int(topology_version), time=0.0, dt=self._h)
        self._rng = None if shuffle_rows is None else np.random.default_rng(shuffle_rows)
        self._k = 0
        self._offer = None
        self.failed = False
        self.receipts = []

    @property
    def stream_id(self):
        return self._stream

    @property
    def source(self):
        return self._source

    def _view(self):
        """The current state as offered: canonical rows, or a row permutation of them."""
        S = self._S
        if self._rng is None:
            return S
        N, T = len(S.positions), len(S.triangles)
        pv, pt = self._rng.permutation(N), self._rng.permutation(T)
        inv = np.empty(N, np.int64)
        inv[pv] = np.arange(N)
        tri = inv[S.triangles[pt]]
        tri = np.take_along_axis(tri, (self._rng.integers(0, 3, T)[:, None] + np.arange(3)) % 3, axis=1)
        return SurfaceSnapshot(positions=S.positions[pv], triangles=tri, vertex_ids=S.vertex_ids[pv],
                               triangle_ids=S.triangle_ids[pt], velocity=S.velocity[pv], pinned=S.pinned[pv],
                               epoch=S.epoch, step_index=S.step_index, topology_version=S.topology_version,
                               time=S.time, dt=S.dt)

    def _next_state(self, C, b):
        """S_{k+1} of the prescribed motion (the KinematicSurfaceBridge rule)."""
        return SurfaceSnapshot(positions=C.positions + self._h * C.velocity, triangles=C.triangles,
                               vertex_ids=C.vertex_ids, triangle_ids=C.triangle_ids, velocity=C.velocity,
                               pinned=C.pinned, epoch=C.epoch, step_index=C.step_index + 1,
                               topology_version=C.topology_version, time=b.t_end, dt=self._h)

    def open_window(self, *, h, particle_dt):
        if self.failed:
            raise RuntimeError("prescribed peer: aborted after a failed commit; start a new stream")
        if self._offer is not None:
            raise RuntimeError("prescribed peer: a window is already open")
        if float(h) != self._h:
            raise ValueError("prescribed peer: h is fixed for the stream")
        substeps_of(h, particle_dt)
        self._offer = WindowOffer(SCHEMA, self._stream, self._source, self._k, self._view(), float(particle_dt),
                                  self._h)
        return self._offer

    def consume(self, msg):
        if self.failed:
            raise RuntimeError("prescribed peer: aborted after a failed commit; start a new stream")
        offer = self._offer
        if offer is None:
            raise RuntimeError("prescribed peer: no open window")
        if not isinstance(msg, ImpulseWindow) or msg.schema != SCHEMA:
            raise ValueError("prescribed peer: expected an ImpulseWindow of the negotiated schema")
        if msg.window_index != offer.window_index:
            raise ValueError("prescribed peer: missed or replayed window")
        b, S = msg.batch, offer.snapshot
        if b.stream_id != self._stream or b.source != self._source:
            raise ValueError("prescribed peer: load stream/source mismatch")
        if b.cloth_version != (S.epoch, S.step_index, S.topology_version):
            raise ValueError("prescribed peer: stale or foreign cloth version")
        if b.t_start != S.time or b.t_end != S.time + self._h:
            raise ValueError("prescribed peer: load interval does not match the window")
        index = {int(v): i for i, v in enumerate(S.vertex_ids)}
        if any(int(v) not in index for v in b.vertex_ids):
            raise ValueError("prescribed peer: unknown vertex ID")
        rows = np.array([index[int(v)] for v in b.vertex_ids], np.int64)
        if len(rows) != len(S.vertex_ids) or not np.array_equal(np.sort(rows), np.arange(len(rows))):
            raise ValueError("prescribed peer: dense batch must cover every vertex exactly once, zero rows included")
        dense = np.zeros((len(rows), 3))
        dense[rows] = b.impulse
        C = self._S
        nxt = self._next_state(C, b)
        # JSON-native like XPBD's receipts (P2-U3: the endpoint digests the receipt as canonical JSON)
        receipt = dict(schema="mpm-prescribed-peer-v1", window_index=int(offer.window_index), batch_id=b.batch_id,
                       stream_id=b.stream_id, source=b.source.as_dict(), t_start=float(b.t_start), t_end=float(b.t_end),
                       start_version=[int(S.epoch), int(S.step_index), int(S.topology_version)],
                       committed_version=[int(nxt.epoch), int(nxt.step_index), int(nxt.topology_version)],
                       vertex_ids=np.asarray(S.vertex_ids, np.int64).tolist(), impulse=dense.tolist(),
                       load_applied_to_geometry=False,
                       peer="MPM-side stand-in with KinematicSurfaceBridge semantics (not XPBD code)")
        receipt["digest"] = receipt_digest(receipt)
        self._S = nxt                                          # commit
        if (nxt.epoch != C.epoch or nxt.step_index != C.step_index + 1 or nxt.time != b.t_end
                or nxt.dt != self._h or nxt.topology_version < C.topology_version):
            self.failed = True                                 # after the commit: terminal, as XPBD's adapter
            raise ValueError("prescribed peer: inconsistent committed publication (stream poisoned)")
        self._k += 1
        self._offer = None
        self.receipts.append(receipt)
        return receipt
