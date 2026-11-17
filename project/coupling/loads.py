"""SI impulse input, separate from the read-only surface query API.

The driver owns the ledger; a callback is the host bridge, never a Taichi field.
Read-only arrays are an API guarantee, not a security boundary.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .contracts import ClothSnapshot, _array, _count, _ids, _integer, _real


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _version(value):
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("cloth_version must be (epoch, step_index, topology_version)")
    return tuple(_integer(v, "cloth_version") for v in value)


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    kind: str
    version: int

    def __post_init__(self):
        _text(self.source_id, "source_id")
        if self.kind not in ("prescribed", "real_backend"):
            raise ValueError("source kind must be prescribed or real_backend")
        object.__setattr__(self, "version", _integer(self.version, "source version"))


@dataclass(frozen=True, eq=False)
class LoadBatch:
    stream_id: str
    source: SourceIdentity
    cloth_version: tuple[int, int, int]
    batch_id: str
    t_start: float
    t_end: float
    vertex_ids: np.ndarray
    impulse: np.ndarray
    coordinate_frame: str = "world"
    units: str = "SI"

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        _text(self.batch_id, "batch_id")
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        object.__setattr__(self, "cloth_version", _version(self.cloth_version))
        for name in ("t_start", "t_end"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if self.t_end <= self.t_start:
            raise ValueError("load interval must advance representable time")
        if self.coordinate_frame != "world" or self.units != "SI":
            raise ValueError("loads require world coordinates and SI (impulse N s)")
        n = _count(self.vertex_ids, "vertex_ids")
        object.__setattr__(self, "vertex_ids", _ids(self.vertex_ids, "vertex_ids", n, nonnegative=True))
        object.__setattr__(self, "impulse", _array(self.impulse, "impulse", np.float64, (n, 3)))

    def validate_for(self, snapshot, *, stream_id, source, h, capacity):
        """Resolve stable IDs to active rows, never treating an ID as a slot."""
        if not isinstance(snapshot, ClothSnapshot):
            raise TypeError("snapshot must be ClothSnapshot")
        h = _real(h, "h", positive=True)
        capacity = _integer(capacity, "capacity")
        if self.stream_id != stream_id or self.source != source:
            raise ValueError("load stream/source revision mismatch")
        if self.cloth_version != snapshot.version:
            raise ValueError("stale or foreign cloth version")
        if self.t_start != snapshot.time or self.t_end != snapshot.time + h:
            raise ValueError("load interval does not match the next physical substep")
        if len(snapshot.vertex_ids) > capacity:
            raise ValueError("active vertex count exceeds device capacity")
        index = {int(v): i for i, v in enumerate(snapshot.vertex_ids)}
        if any(int(v) not in index for v in self.vertex_ids):
            raise ValueError("load contains an unknown active vertex ID")
        # Pinned rows may carry an applied impulse. The bridge uses inv_mass=0;
        # the applied external impulse remains in the audit, not a pin reaction.
        if (np.any(snapshot.inverse_mass[snapshot.pinned] != 0)
                or np.any(snapshot.inverse_mass[~snapshot.pinned] <= 0)):
            raise ValueError("inverse mass and pinned flags are inconsistent")
        rows = np.array([index[int(v)] for v in self.vertex_ids], dtype=np.int64)
        rows.setflags(write=False)
        return rows


class LoadLedger:
    """One aggregated load per physical substep, including empty/zero loads.

Validation failure is retryable. Once application starts, failure poisons the
ledger because a host/device upload might have partially succeeded. Recovery
requires a new driver/ledger and a new epoch (or a new simulation stream).
"""

    def __init__(self, stream_id: str):
        self.stream_id = _text(stream_id, "stream_id")
        self._batch_ids = set()
        self._last = None
        self._last_end = None
        self._last_topology = None
        self.poisoned = False

    def consume(self, batch: LoadBatch, snapshot: ClothSnapshot, *, source: SourceIdentity,
                h: float, capacity: int, apply: Callable):
        if self.poisoned:
            raise RuntimeError("load ledger is poisoned; reset the simulation")
        if not isinstance(batch, LoadBatch) or not callable(apply):
            raise TypeError("consume requires LoadBatch and a bridge callback")
        rows = batch.validate_for(snapshot, stream_id=self.stream_id, source=source,
                                  h=h, capacity=capacity)
        step = snapshot.version[:2]
        if batch.batch_id in self._batch_ids:
            raise ValueError("batch_id already consumed")
        if self._last is not None and step <= self._last:
            raise ValueError("physical step already consumed or moved backwards")
        if self._last is None or step[0] > self._last[0]:
            if step[1] != 0 or batch.t_start != 0:
                raise ValueError("load epoch must start at step 0, time 0; resume is unsupported")
        elif (step[1] != self._last[1] + 1 or batch.t_start != self._last_end
              or snapshot.topology_version < self._last_topology):
            raise ValueError("loads must cover consecutive substeps without time overlap or gaps")
        self._batch_ids.add(batch.batch_id)
        self._last = step
        self._last_end = batch.t_end
        self._last_topology = snapshot.topology_version
        self.poisoned = True
        result = apply(rows, batch.impulse)
        self.poisoned = False
        return result


class SnapshotClock:
    """Validate publications of completed states; owns no solver state.

Frame and outer counters do not advance this clock. After a tear, publish the
same step/time with an increased topology version; after solve, step += 1.
"""

    def __init__(self, stream_id: str):
        self.stream_id = _text(stream_id, "stream_id")
        self._last = None

    def publish(self, snapshot: ClothSnapshot):
        if not isinstance(snapshot, ClothSnapshot):
            raise TypeError("snapshot must be ClothSnapshot")
        if self._last is None or snapshot.epoch > self._last.epoch:
            if snapshot.step_index != 0 or snapshot.time != 0:
                raise ValueError("new epoch must start at step 0, time 0")
        else:
            old = self._last
            if snapshot.epoch != old.epoch or snapshot.topology_version < old.topology_version:
                raise ValueError("snapshot version moved backwards")
            if snapshot.step_index == old.step_index:
                if (snapshot.topology_version <= old.topology_version
                        or snapshot.time != old.time or snapshot.dt != old.dt):
                    raise ValueError("only a topology event may republish an unadvanced step")
            elif snapshot.step_index == old.step_index + 1:
                if snapshot.time != old.time + snapshot.dt or snapshot.time <= old.time:
                    raise ValueError("completed substep time must advance by actual h")
                if snapshot.topology_version != old.topology_version:
                    raise ValueError("publish solved state before the topology event")
            else:
                raise ValueError("step_index must increment once per physical substep")
        self._last = snapshot
