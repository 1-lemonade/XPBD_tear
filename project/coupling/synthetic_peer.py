"""Deterministic SI point-force peer for the bounded exchange demonstration."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .contracts import MappingConfig, QueryPoints, _array
from .exchange import SCHEMA, ImpulseWindow
from .loads import LoadBatch, SourceIdentity
from .tracking import SurfaceTracker


def _nullable(vectors):
    """Use JSON null for unavailable P20 geometry or HIT-only velocities."""
    return [[float(x) if np.isfinite(x) else None for x in row] for row in vectors]


class SyntheticPeer:
    """Prepare a window privately, then commit its particle state after cloth acceptance.

    Each eligible point exerts its own prescribed world-space force in N. Its
    synthetic particle receives the opposite impulse. Positions use a symplectic
    Euler update per particle tick. No area weighting or continuum force model.
    """
    def __init__(self, source, particle_ids, positions, velocity, mass, force_on_cloth_N,
                 *, max_distance=.025):
        if not isinstance(source, SourceIdentity) or source.kind != "prescribed":
            raise ValueError("synthetic peer requires a prescribed source identity")
        initial = QueryPoints(particle_ids, positions)
        n = len(initial.query_ids)
        self.source = source
        self.particle_ids = initial.query_ids
        self.positions = initial.positions
        self.velocity = _array(velocity, "velocity", np.float64, (n,3))
        self.mass = _array(mass, "mass", np.float64, (n,))
        if np.any(self.mass <= 0):
            raise ValueError("synthetic particles require positive physical masses")
        self.force_on_cloth_N = _array(force_on_cloth_N, "force_on_cloth_N", np.float64, (n,3))
        self.config = MappingConfig(max_distance=max_distance)
        if self.config.max_distance is None or self.config.max_distance <= 0:
            raise ValueError("a finite positive risk radius is required")
        self._pending = None
        self._stream = None
        self._next_window = 0
        self._time = 0.
        self._expected_version = None

    def prepare(self, offer):
        if self._pending is not None:
            raise RuntimeError("a synthetic particle proposal is pending")
        if (offer.schema != SCHEMA or offer.source != self.source or offer.window_index != self._next_window
                or offer.snapshot.time != self._time
                or self._stream is not None and offer.stream_id != self._stream):
            raise ValueError("stale or foreign offer for this synthetic peer")
        if self._expected_version is None:
            if offer.snapshot.step_index != 0:
                raise ValueError("synthetic peer cannot resume a cloth stream")
        elif offer.snapshot.version != self._expected_version:
            raise ValueError("opening cloth version differs from its previous acknowledgement")
        s, dt = offer.snapshot, offer.particle_dt
        rows = {int(identity):i for i,identity in enumerate(s.vertex_ids)}
        cloth = np.zeros((len(rows),3))
        x, v = self.positions.copy(), self.velocity.copy()
        ticks = []
        for i in range(offer.substeps):
            geometry = offer.geometry_at_tick(i)
            # Prediction samples have the same source identity; none is published
            # as the next cloth snapshot or reused as a tracked completed state.
            result = SurfaceTracker(self.config).map(geometry, QueryPoints(self.particle_ids,x),
                                                     stream_id=offer.stream_id, cut_edges=offer.cut_edges)
            mapped = result.mapping
            points = np.where(result.eligible[:,None], self.force_on_cloth_N * dt, 0.)
            for p in np.flatnonzero(result.eligible):
                vertex_rows = [rows[int(identity)] for identity in mapped.vertex_ids[p]]
                np.add.at(cloth, vertex_rows, mapped.barycentric[p,:,None] * points[p])
            next_v = v - points / self.mass[:,None]
            next_x = x + next_v*dt
            if not (np.isfinite(next_x).all() and np.isfinite(next_v).all()):
                raise ValueError("synthetic particle integration overflow")
            ticks.append(dict(index=i, tau=offer.sample_time(i), query_positions=x.tolist(),
                              triangle_ids=mapped.triangle_ids.tolist(), vertex_ids=mapped.vertex_ids.tolist(),
                              closest_point=_nullable(mapped.closest_point),
                              cloth_angular_impulse=np.cross(mapped.closest_point[result.eligible],
                                                            points[result.eligible]).sum(axis=0).tolist(),
                              barycentric=_nullable(mapped.barycentric), status=mapped.status.tolist(),
                              eligible=result.eligible.tolist(), surface_velocity=_nullable(result.velocity),
                              point_impulse_on_cloth=points.tolist(), particle_velocity_after=next_v.tolist(),
                              particle_positions_after=next_x.tolist()))
            x,v = next_x,next_v
        batch = LoadBatch(offer.stream_id, self.source, s.version, f"point-force:{offer.window_index}",
                          s.time, s.time+offer.h, s.vertex_ids, cloth)
        message = ImpulseWindow(SCHEMA, offer.window_index, batch)
        trace = dict(schema="xpbd-synthetic-peer-trace-v1", window_index=offer.window_index,
                     mapping_config=asdict(self.config),
                     particle_ids=self.particle_ids.tolist(), mass=self.mass.tolist(),
                     force_on_cloth_N=self.force_on_cloth_N.tolist(), positions_before=self.positions.tolist(),
                     velocity_before=self.velocity.tolist(), positions_after=x.tolist(), velocity_after=v.tolist(),
                     peer_impulse=((v-self.velocity)*self.mass[:,None]).sum(axis=0).tolist(),
                     cloth_impulse=cloth.sum(axis=0).tolist(), ticks=ticks)
        x.setflags(write=False)
        v.setflags(write=False)
        self._pending = (offer,trace,batch,x,v)
        return message,trace

    def commit(self, offer, trace, receipt):
        pending = self._pending
        if pending is None or pending[0] is not offer or pending[1] is not trace:
            raise ValueError("synthetic commit does not match its pending proposal")
        batch = pending[2]
        start = list(batch.cloth_version)
        solved = [start[0], start[1]+1, start[2]]
        expected = dict(schema="xpbd-exchange-cloth-v1", stream_id=offer.stream_id,
                        window_index=offer.window_index, batch_id=batch.batch_id, source=asdict(batch.source),
                        t_start=batch.t_start, t_end=batch.t_end, h=offer.h, previous_dt=offer.snapshot.dt,
                        start_version=start, solved_version=solved)
        if not isinstance(receipt, dict) or any(receipt.get(k) != v for k,v in expected.items()):
            raise ValueError("cloth acknowledgement differs from the pending window")
        committed = receipt.get("committed_version")
        if ("event" not in receipt or type(committed) is not list or len(committed) != 3
                or any(type(v) is not int for v in committed) or committed[:2] != solved[:2]):
            raise ValueError("invalid committed cloth version")
        event = receipt["event"]
        if event is None:
            valid = committed == solved
        else:
            valid = (isinstance(event,dict) and committed[2] > solved[2] and event.get("before_version") == solved
                     and event.get("after_version") == committed and event.get("time") == batch.t_end)
        if not valid:
            raise ValueError("cloth acknowledgement has an inconsistent topology commitment")
        # The synthetic sender emits a dense batch in opening-snapshot stable ID order.
        for name, value in (("batch_vertex_ids",batch.vertex_ids), ("batch_impulse",batch.impulse),
                            ("vertex_ids",batch.vertex_ids), ("impulse",batch.impulse)):
            if not np.array_equal(receipt.get(name), value):
                raise ValueError(f"cloth acknowledgement differs from pending {name}")
        self.positions, self.velocity = pending[3], pending[4]
        self._stream = offer.stream_id
        self._expected_version = tuple(committed)
        self._next_window += 1
        self._time = offer.snapshot.time + offer.h
        self._pending = None
