"""Window exchange v1, in right-handed world SI. A window is one XPBD step.

A material backend owns its substeps and contact law. It returns a single batch
of material-on-cloth impulses on the stable vertices of the opening surface.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from fractions import Fraction
import math

import numpy as np

from .contracts import ClothSnapshot, _integer, _real
from .loads import LoadBatch, SourceIdentity, _text
from .tracking_topology import canonical_cut_edges

SCHEMA = "xpbd-exchange-v1"


def validate_surface(snapshot, cut_edges):
    """Check oriented edge incidence and each vertex link, in current IDs.

    This is a combinatorial check. Geometric self-intersection and unrelated
    vertices that touch the interior of another face are outside its scope.
    """
    faces = snapshot.vertex_ids[snapshot.triangles].tolist()
    points = snapshot.positions[snapshot.triangles]
    with np.errstate(over="ignore", invalid="ignore"):
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    if not np.isfinite(normals).all() or np.any(np.all(normals == 0, axis=1)):
        raise ValueError("exchange surface contains an unusable triangle")
    seen_faces = set()
    incidence = defaultdict(list)
    links = defaultdict(lambda: defaultdict(set))
    for a, b, c in faces:
        key = tuple(sorted((a, b, c)))
        if len(set(key)) != 3 or key in seen_faces:
            raise ValueError("duplicate or collapsed triangle")
        seen_faces.add(key)
        for center, u, v in ((a, b, c), (b, c, a), (c, a, b)):
            incidence[tuple(sorted((u, v)))].append((u, v))
            links[center][u].add(v)
            links[center][v].add(u)
    if not faces or set(links) != set(snapshot.vertex_ids.tolist()):
        raise ValueError("exchange surface requires faces at every vertex")
    for sides in incidence.values():
        if len(sides) > 2 or (len(sides) == 2 and sides[0] == sides[1]):
            raise ValueError("nonmanifold edge or inconsistent winding")
    for link in links.values():
        reached = set()
        pending = [next(iter(link))]
        while pending:
            vertex = pending.pop()
            if vertex not in reached:
                reached.add(vertex)
                pending.extend(link[vertex] - reached)
        if len(reached) != len(link):
            raise ValueError("disconnected vertex fan")
    if not cut_edges.issubset(incidence):
        raise ValueError("cut edge is absent from the published surface")


@dataclass(frozen=True, eq=False)
class WindowOffer:
    schema: str
    stream_id: str
    source: SourceIdentity
    window_index: int
    snapshot: ClothSnapshot
    particle_dt: float
    h: float
    cut_edges: frozenset = frozenset()

    substeps: int = field(init=False)

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.source, SourceIdentity):
            raise ValueError("unsupported window schema or source")
        _text(self.stream_id, "stream_id")
        _integer(self.window_index, "window_index")
        if not isinstance(self.snapshot, ClothSnapshot):
            raise TypeError("window needs the opening ClothSnapshot")
        h = _real(self.h, "next h", positive=True)
        dt = _real(self.particle_dt, "particle_dt", positive=True)
        # Durations on the wire use the shortest decimal form of each float.
        # This admits 0.0006 / 0.0002 and rejects a neighbouring noninteger H.
        ratio = Fraction(str(h)) / Fraction(str(dt))
        if ratio.denominator != 1 or ratio.numerator > 2**53:
            raise ValueError("H=h must be an integer number of representable particle ticks")
        end = self.snapshot.time + h
        if (not math.isfinite(end) or end <= self.snapshot.time
                or (ratio.numerator > 1 and (dt < math.ulp(end)
                    or self.snapshot.time + (ratio.numerator - 1) * dt >= end))):
            raise ValueError("the exchange clock cannot distinguish particle samples")
        cuts = canonical_cut_edges(self.cut_edges)
        validate_surface(self.snapshot, cuts)
        object.__setattr__(self, "cut_edges", cuts)
        object.__setattr__(self, "substeps", ratio.numerator)

    def sample_time(self, i):
        """Elapsed time of a material sample relative to the opening snapshot."""
        i = _integer(i, "tick")
        if i >= self.substeps:
            raise ValueError("tick must be inside the half-open exchange window")
        return i * self.particle_dt

    def geometry_at_tick(self, i):
        """Frozen-velocity predictor, tagged with its source publication."""
        tau = self.sample_time(i)
        if tau == 0.:
            return self.snapshot
        return replace(self.snapshot, positions=self.snapshot.positions + tau * self.snapshot.velocity)


@dataclass(frozen=True)
class ImpulseWindow:
    schema: str
    window_index: int
    batch: LoadBatch

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.batch, LoadBatch):
            raise ValueError("unsupported impulse window schema or payload")
        _integer(self.window_index, "window_index")


class ExchangeAdapter:
    """Own the only active exchange window and publish only committed states."""

    def __init__(self, bridge, source):
        if not isinstance(source, SourceIdentity) or source != bridge.source:
            raise ValueError("adapter source differs from the configured cloth bridge")
        self._bridge = bridge
        self._source = source
        self._window = None
        self._window_index = 0
        self._failed = False
        initial = bridge.snapshot
        if initial.step_index != 0 or initial.time != 0:
            raise ValueError("resuming an exchange stream is unsupported")
        self._committed = initial
        self._stream_id = bridge.stream_id

    @property
    def failed(self):
        return self._failed or self._bridge.failed

    @property
    def snapshot(self):
        if self.failed:
            raise RuntimeError("exchange aborted; start a new stream")
        if self._bridge.snapshot is not self._committed or self._bridge.stream_id != self._stream_id:
            self._failed = True
            raise RuntimeError("cloth publication changed outside the exchange")
        return self._committed

    def open_window(self, *, h, particle_dt):
        state = self.snapshot
        if self._window is not None:
            raise RuntimeError("an exchange window is already open")
        h = self._bridge.validate_h(h)
        offer = WindowOffer(SCHEMA, self._stream_id, self._source, self._window_index,
                            state, particle_dt, h, self._bridge.cut_edges)
        self._window = offer
        return offer

    def consume(self, message):
        state = self.snapshot
        offer = self._window
        if offer is None:
            raise RuntimeError("no pending exchange window")
        if not isinstance(message, ImpulseWindow) or message.schema != SCHEMA:
            raise ValueError("expected the negotiated impulse message schema")
        if message.window_index != offer.window_index:
            raise ValueError("missed or replayed exchange window")
        batch = message.batch
        batch.validate_for(state, stream_id=offer.stream_id, source=self._source,
                           h=offer.h, capacity=len(state.vertex_ids))
        receipt = self._bridge.advance(batch, offer.h)
        # Once the bridge has applied a load, publication failures are terminal.
        try:
            end = self._bridge.snapshot
            if (end.epoch != state.epoch or end.step_index != state.step_index + 1
                    or end.time != batch.t_end or end.dt != offer.h
                    or end.topology_version < state.topology_version):
                raise ValueError("inconsistent committed cloth publication")
            receipt["window_index"] = offer.window_index
        except Exception:
            self._failed = True
            raise
        self._committed = end
        self._window = None
        self._window_index += 1
        return receipt

