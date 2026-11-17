"""Owned SI provider, reference and diagnostic packets for the new driver.

These types do not apply contact, allocate MPM particles or advance a solver.
The cloth host bridge will return plain arrays; only the driver wraps those
arrays in ConstraintDiagnostics, keeping cloth independent of coupling.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contracts import (ClothSnapshot, QueryPoints, MappingResult, MappingStatus,
                        _array, _count, _ids, _integer, _real)
from .loads import SourceIdentity, _text, _version


def _set_array(obj, name, dtype, shape, *, finite=True):
    value = _array(getattr(obj, name), name, dtype, shape, finite=finite)
    object.__setattr__(obj, name, value)
    return value


@dataclass(frozen=True, eq=False)
class ProviderBoundarySamples:
    """Mapped boundary samples evaluated at cp at snapshot.time.

Keep original query and mapping identity, including non-HIT candidates.
Provider fields on every non-HIT row MUST be NaN (active=False). The packet
is not a quadrature rule: Q statistics require an explicit sampling measure.
"""

    stream_id: str
    source: SourceIdentity
    snapshot: ClothSnapshot
    queries: QueryPoints
    mapping: MappingResult
    sample_time: float
    cloth_velocity: np.ndarray
    phi: np.ndarray
    outward_normal: np.ndarray
    body_velocity: np.ndarray
    pressure: np.ndarray
    active: np.ndarray
    sampling_measure: str = "unweighted_queries"
    measure_weights: np.ndarray | None = None
    coordinate_frame: str = "world"
    units: str = "SI"

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if not isinstance(self.snapshot, ClothSnapshot) or not isinstance(self.queries, QueryPoints):
            raise TypeError("snapshot/queries must use the original query contracts")
        if not isinstance(self.mapping, MappingResult):
            raise TypeError("mapping must be MappingResult")
        if self.units != "SI" or self.coordinate_frame != "world":
            raise ValueError("provider packet requires world coordinates and SI")
        sample_time = _real(self.sample_time, "sample_time")
        object.__setattr__(self, "sample_time", sample_time)
        if self.mapping.version != self.snapshot.version or sample_time != self.snapshot.time:
            raise ValueError("provider packet contains stale state or time")
        if not np.array_equal(self.mapping.query_ids, self.queries.query_ids):
            raise ValueError("query IDs/order must exactly match the mapping")
        q = len(self.queries.query_ids)
        if self.sampling_measure not in ("unweighted_queries", "body_reference_area", "cloth_current_area"):
            raise ValueError("unknown sampling measure")
        if self.measure_weights is None:
            if self.sampling_measure != "unweighted_queries":
                raise ValueError("area measures require explicit Jacobian/area weights [m^2]")
            object.__setattr__(self, "measure_weights", np.ones(q))
        weights = _set_array(self, "measure_weights", np.float64, (q,))
        with np.errstate(over="ignore"):
            total_weight = weights.sum(dtype=np.float64)
        if np.any(weights < 0) or not np.isfinite(total_weight) or (q and total_weight <= 0):
            raise ValueError("sampling weights must be nonnegative with finite positive total measure")
        if self.sampling_measure == "unweighted_queries" and np.any(weights != 1):
            raise ValueError("unweighted query measure uses unit weights, including MISS")
        hit = self.mapping.status == MappingStatus.HIT
        for name in ("cloth_velocity", "outward_normal", "body_velocity", "phi", "pressure"):
            shape = (q, 3) if name in ("cloth_velocity", "outward_normal", "body_velocity") else (q,)
            value = _set_array(self, name, np.float64, shape, finite=False)
            if not np.isfinite(value[hit]).all() or not np.isnan(value[~hit]).all():
                raise ValueError(f"{name}: HIT must be finite, every non-HIT must be NaN")
        active = _set_array(self, "active", bool, (q,))
        if not np.array_equal(active, hit & (self.phi < 0)):
            raise ValueError("active must be HIT and phi < 0")
        if np.any(self.pressure[hit] < 0) or np.any(self.pressure[hit & ~active] != 0):
            raise ValueError("pressure must be nonnegative and zero outside contact")
        if np.any(np.abs(np.linalg.norm(self.outward_normal[hit], axis=1) - 1) > 1e-9):
            raise ValueError("contact normal must be a defined outward unit vector")
        expected = self.interpolated_velocity(self.snapshot, self.queries, self.mapping)
        if not np.allclose(self.cloth_velocity[hit], expected[hit], rtol=1e-10, atol=1e-12):
            raise ValueError("cloth velocity must be interpolated at actual cp")

    def validate_for(self, snapshot, *, stream_id, source):
        if (self.stream_id != stream_id or self.source != source
                or self.snapshot.version != snapshot.version or self.sample_time != snapshot.time):
            raise ValueError("provider/cloth revision or stream mismatch")

    @staticmethod
    def interpolated_velocity(snapshot, queries, mapping):
        """Validate mapping bindings against the producing snapshot by stable ID."""
        if mapping.version != snapshot.version or not np.array_equal(mapping.query_ids, queries.query_ids):
            raise ValueError("mapping/query/snapshot identity mismatch")
        tri_rows = {int(t): i for i, t in enumerate(snapshot.triangle_ids)}
        result = np.full((len(queries.query_ids), 3), np.nan)
        candidate = mapping.status <= MappingStatus.OUTSIDE_THRESHOLD
        for i in np.flatnonzero(candidate):
            tid = int(mapping.triangle_ids[i])
            if tid not in tri_rows:
                raise ValueError("mapped triangle ID is absent from snapshot")
            rows = snapshot.triangles[tri_rows[tid]]
            if not np.array_equal(mapping.vertex_ids[i], snapshot.vertex_ids[rows]):
                raise ValueError("mapped endpoint IDs/order disagree with current triangle")
            cp = mapping.barycentric[i] @ snapshot.positions[rows]
            # Scale is geometric distance from origin in world metres; this is
            # consistency validation, not a replacement nearest-point search.
            if not np.allclose(cp, mapping.closest_point[i], rtol=1e-10, atol=1e-12):
                raise ValueError("cp disagrees with current triangle/barycentric binding")
            distance = np.linalg.norm(queries.positions[i] - mapping.closest_point[i])
            if not np.isclose(distance, mapping.distance[i], rtol=1e-10, atol=1e-12):
                raise ValueError("mapping was not produced for these query positions")
            if mapping.status[i] == MappingStatus.HIT:
                result[i] = mapping.barycentric[i] @ snapshot.velocity[rows]
        result.setflags(write=False)
        return result


class BoundaryProvider(Protocol):
    """P21-03 implementations must declare provenance and physical time.

    A real backend must accept the full packet and trace returned loads to it.
    The empty MPM slot raises NotImplementedError for all physical methods.
    """

    @property
    def source(self) -> SourceIdentity: ...

    def sample_boundary(self, time: float, count: int) -> QueryPoints: ...

    def evaluate_surface(self, positions: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return phi [m], outward normal [1], velocity [m/s] at world positions."""
        ...

    def accept_cloth_samples(self, samples: ProviderBoundarySamples) -> None: ...

    def advance(self, t_start: float, h: float) -> None: ...


@dataclass(frozen=True, eq=False)
class ParticleMassState:
    """Host mass upload payload [kg, kg^-1], including physical pinned mass."""

    stream_id: str
    cloth_version: tuple[int, int, int]
    vertex_ids: np.ndarray
    mass: np.ndarray
    inverse_mass: np.ndarray
    pinned: np.ndarray

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        object.__setattr__(self, "cloth_version", _version(self.cloth_version))
        n = _count(self.vertex_ids, "vertex_ids")
        object.__setattr__(self, "vertex_ids", _ids(self.vertex_ids, "vertex_ids", n, nonnegative=True))
        mass = _set_array(self, "mass", np.float64, (n,))
        inv = _set_array(self, "inverse_mass", np.float64, (n,))
        pinned = _set_array(self, "pinned", bool, (n,))
        if np.any(mass <= 0) or np.any(inv < 0):
            raise ValueError("physical mass must be positive; inverse mass nonnegative")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            expected = np.where(pinned, 0, 1 / mass)
        if not np.isfinite(expected).all() or not np.allclose(inv, expected, rtol=1e-12, atol=0):
            raise ValueError("inverse mass must be 0 for pins, otherwise 1/mass")

    def validate_for(self, snapshot, stream_id, *, capacity, reference):
        capacity = _integer(capacity, "capacity")
        reference.validate_for(snapshot, stream_id)
        if (self.stream_id != stream_id or self.cloth_version != snapshot.version
                or not np.array_equal(self.vertex_ids, snapshot.vertex_ids)
                or len(self.vertex_ids) > capacity
                or not np.array_equal(self.pinned, snapshot.pinned)
                or not np.array_equal(self.pinned, reference.boundary)
                or not np.allclose(self.inverse_mass, snapshot.inverse_mass, rtol=1e-12, atol=0)):
            raise ValueError("mass upload does not match active topology/boundary/capacity")
        if len(np.unique(snapshot.triangles)) != len(snapshot.vertex_ids):
            raise ValueError("isolated vertex has no incident fixed triangle mass")


@dataclass(frozen=True, eq=False)
class ReferenceState:
    """Fixed triangle masses and root provenance, valid for one topology.

    root_vertex_ids may repeat; every root remains a self-rooted active vertex.
    boundary is reference membership, not a current contact/pin decision.
    """

    stream_id: str
    epoch: int
    topology_version: int
    vertex_ids: np.ndarray
    root_vertex_ids: np.ndarray
    reference_positions: np.ndarray
    boundary: np.ndarray
    triangle_ids: np.ndarray
    triangle_mass: np.ndarray

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        for name in ("epoch", "topology_version"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        n, t = _count(self.vertex_ids, "vertex_ids"), _count(self.triangle_ids, "triangle_ids")
        for name, count in (("vertex_ids", n), ("triangle_ids", t)):
            object.__setattr__(self, name, _ids(getattr(self, name), name, count, nonnegative=True))
        roots = _set_array(self, "root_vertex_ids", np.int64, (n,))
        _set_array(self, "reference_positions", np.float64, (n, 3))
        _set_array(self, "boundary", bool, (n,))
        mass = _set_array(self, "triangle_mass", np.float64, (t,))
        if np.any(mass <= 0):
            raise ValueError("fixed triangle rest mass must be positive")
        index = {int(v): i for i, v in enumerate(self.vertex_ids)}
        for i, root in enumerate(roots):
            if int(root) not in index:
                raise ValueError("reference root ID is missing")
            j = index[int(root)]
            if (roots[j] != root or self.boundary[i] != self.boundary[j]
                    or not np.array_equal(self.reference_positions[i], self.reference_positions[j])):
                raise ValueError("duplicate must preserve root reference and boundary membership")

    def validate_for(self, snapshot, stream_id):
        if (self.stream_id != stream_id or self.epoch != snapshot.epoch
                or self.topology_version != snapshot.topology_version
                or not np.array_equal(self.vertex_ids, snapshot.vertex_ids)
                or not np.array_equal(self.triangle_ids, snapshot.triangle_ids)):
            raise ValueError("reference state does not match current cloth topology")

    def validate_successor(self, previous):
        """A tear may append vertices, but never change reference triangle mass."""
        if (self.stream_id != previous.stream_id or self.epoch != previous.epoch
                or self.topology_version <= previous.topology_version
                or not np.array_equal(self.triangle_ids, previous.triangle_ids)
                or not np.array_equal(self.triangle_mass, previous.triangle_mass)):
            raise ValueError("tear must preserve fixed triangle identities/masses")
        n = len(previous.vertex_ids)
        for name in ("vertex_ids", "root_vertex_ids", "reference_positions", "boundary"):
            if not np.array_equal(getattr(self, name)[:n], getattr(previous, name)):
                raise ValueError("tear must append provenance without changing existing rows")
        if not np.isin(self.root_vertex_ids[n:], previous.root_vertex_ids).all():
            raise ValueError("new vertex must descend from a previous reference root")


@dataclass(frozen=True, eq=False)
class ConstraintDiagnostics:
    """One constraint type, in actual device slot/color order, after solve.

    stretch/bend: endpoints (N,2), rest/lambda (N,).
    pin: endpoints (N,1), rest=target (N,3), lambda (N,3).
    No solved lambda after rebuild: do not construct a packet; record N/A.
    Hard-pin lambda is not a support reaction. No tension helper is supplied
    before the independent force-channel acceptance in later milestones.
    """

    stream_id: str
    cloth_version: tuple[int, int, int]
    kind: str
    endpoint_ids: np.ndarray
    rest: np.ndarray
    lambdas: np.ndarray
    color_offsets: np.ndarray
    h: float
    time: float
    precision: str
    backend: str
    coloring_version: int
    repack_count: int
    active_count: int
    export_stage: str = "post_solve_pre_tear"

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        object.__setattr__(self, "cloth_version", _version(self.cloth_version))
        if self.kind not in ("stretch", "bend", "pin"):
            raise ValueError("unknown constraint kind")
        if self.precision not in ("f32", "f64") or self.backend not in ("cpu", "cuda"):
            raise ValueError("actual backend/precision must be explicit")
        width = 1 if self.kind == "pin" else 2
        n = _count(self.endpoint_ids, "endpoint_ids", width)
        for name in ("coloring_version", "repack_count", "active_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if (self.coloring_version != self.cloth_version[2] or self.active_count != n
                or self.export_stage != "post_solve_pre_tear"):
            raise ValueError("diagnostic generation/count/stage mismatch")
        ids = _set_array(self, "endpoint_ids", np.int64, (n, width))
        if np.any(ids < 0) or (width == 2 and np.any(ids[:, 0] == ids[:, 1])):
            raise ValueError("invalid constraint endpoint IDs")
        shape = (n, 3) if self.kind == "pin" else (n,)
        _set_array(self, "rest", np.float64, shape)
        _set_array(self, "lambdas", np.float64, shape)
        if width == 2 and np.any(self.rest <= 0):
            raise ValueError("distance rest lengths must be positive")
        k = _count(self.color_offsets, "color_offsets")
        offsets = _set_array(self, "color_offsets", np.int64, (k,))
        if k == 0 or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
            raise ValueError("color offsets must partition active device slots (empty: [0])")
        for start, end in zip(offsets[:-1], offsets[1:]):
            if len(np.unique(ids[start:end])) != (end - start) * width:
                raise ValueError("constraints in a device color share endpoints")
        object.__setattr__(self, "h", _real(self.h, "h", positive=True))
        object.__setattr__(self, "time", _real(self.time, "time", positive=True))
        if self.time < self.h or self.cloth_version[1] == 0:
            raise ValueError("diagnostics require a completed physical substep")

    def validate_for(self, snapshot, stream_id, *, repack_count):
        if (self.stream_id != stream_id or self.cloth_version != snapshot.version
                or self.time != snapshot.time or self.h != snapshot.dt
                or self.repack_count != _integer(repack_count, "repack_count")
                or not np.isin(self.endpoint_ids, snapshot.vertex_ids).all()):
            raise ValueError("diagnostics are not aligned to this completed cloth state")
