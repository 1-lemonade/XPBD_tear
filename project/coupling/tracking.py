"""Track risk-query IDs on one completed cloth stream using P20 geometry.

A previous HIT supplies only a candidate face. Its current one-ring may be
used only after a conservative device certificate excludes the complement
from the global nearest-face tie window. The tracker owns no solver state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import ClothSnapshot, MappingConfig, MappingFeature, MappingResult, MappingStatus, QueryPoints
from .mapping import _Surface, map_points
from .tracking_device import certify_local_batch, interpolate_surface_velocity
from .tracking_topology import SurfaceTopology, canonical_cut_edges


@dataclass(frozen=True, eq=False)
class TrackingResult:
    """Per-query diagnostics in input order. Only eligible HITs are contact candidates.

    candidate_ids contains the stable triangle IDs actually sent to the
    local P20 narrow phase, including attempts followed by global fallback.
    An empty tuple means no local subset was attempted. candidate_count is
    the length of each tuple; exhaustive face lists are never stored here.
    """
    stream_id: str
    rebuild_generation: int
    mapping: MappingResult
    velocity: np.ndarray
    eligible: np.ndarray
    kind: tuple[str, ...]
    route: tuple[str, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    candidate_count: np.ndarray


_GEOMETRY_FIELDS = ("triangle_ids", "vertex_ids", "barycentric", "closest_point",
                    "distance", "normal", "normal_offset", "status", "feature")
_CONNECTIVITY = ("vertex_ids", "triangle_ids", "triangles")
_STATE_FIELDS = ("positions", "velocity", "pinned", "inverse_mass")


def _submesh(snapshot, rows):
    """Copy only the referenced vertices; peer P20 inputs keep their stable IDs."""
    faces = np.asarray(rows, dtype=np.intp)
    vertices, inverse = np.unique(snapshot.triangles[faces], return_inverse=True)
    return ClothSnapshot(snapshot.positions[vertices], inverse.reshape(-1, 3), snapshot.vertex_ids[vertices],
                         snapshot.triangle_ids[faces], snapshot.velocity[vertices], snapshot.inverse_mass[vertices],
                         snapshot.pinned[vertices], *snapshot.version, snapshot.time, snapshot.dt)


def _select_queries(queries, indices):
    return QueryPoints(queries.query_ids[indices], queries.positions[indices])


def _scatter(columns, count, indices, mapped):
    for name in _GEOMETRY_FIELDS:
        values = getattr(mapped, name)
        if name not in columns:
            columns[name] = np.empty((count,) + values.shape[1:], dtype=values.dtype)
        columns[name][indices] = values


def _map_with_seeds(snapshot, queries, config, topology, seeds, cold_route):
    count = len(queries.query_ids)
    routes = [cold_route] * count
    candidate_ids = [()] * count
    groups = {}
    for i, query_id in enumerate(queries.query_ids):
        triangle_id = seeds.get(int(query_id))
        if triangle_id is None:
            continue
        routes[i] = "global_fallback"
        center = topology.row_for_id.get(triangle_id)
        if center is None:
            continue
        rows = tuple(sorted(topology.one_ring(center)))
        if 0 < len(rows) < len(snapshot.triangles):
            groups.setdefault(rows, []).append(i)
    if not groups:
        return map_points(snapshot, queries, config), routes, candidate_ids

    columns = {}
    local_rows = [()] * count
    distances = np.full(count, np.nan)
    for rows, group in groups.items():
        indices = np.asarray(group, dtype=np.intp)
        stable_ids = tuple(int(snapshot.triangle_ids[row]) for row in rows)
        for i in group:
            candidate_ids[i] = stable_ids
        # A failing subset may have a different overflow/tie window than the
        # full mesh. Only the complete P20 call decides whether to raise.
        try:
            local = map_points(_submesh(snapshot, rows), _select_queries(queries, indices), config)
        except ValueError:
            continue
        _scatter(columns, count, indices, local)
        distances[indices] = local.distance
        for i in group:
            local_rows[i] = rows

    attempt = np.flatnonzero(np.isfinite(distances))
    certified = np.zeros(count, dtype=bool)
    if len(attempt):
        flags = certify_local_batch(snapshot, queries.positions[attempt], [local_rows[i] for i in attempt],
                                    distances[attempt], config)
        if flags.shape != (len(attempt),) or flags.dtype != np.bool_:
            raise ValueError("invalid local certificate readback")
        certified[attempt] = flags
    for i in np.flatnonzero(certified):
        routes[i] = "local_certified"
    if not certified.any():
        return map_points(snapshot, queries, config), routes, candidate_ids
    fallback = np.flatnonzero(~certified)
    if len(fallback):
        full = map_points(snapshot, _select_queries(queries, fallback), config)
        _scatter(columns, count, fallback, full)
    return MappingResult(queries.query_ids, *snapshot.version, **columns), routes, candidate_ids


def _complete_result(snapshot, config, topology, mapped, routes, candidate_ids, stream_id, generation):
    count = len(mapped.query_ids)
    velocity_rows = np.full(count, -1, dtype=np.int64)
    hit = mapped.status == MappingStatus.HIT
    hits = np.flatnonzero(hit)
    # An interior edge needs two usable incident faces in this geometry.
    valid_faces = (_Surface(snapshot, config).valid
                   if np.any(hit & (mapped.feature == MappingFeature.EDGE)) else None)
    kinds = ["outside" if status == MappingStatus.OUTSIDE_THRESHOLD else "missing" for status in mapped.status]
    eligible = np.zeros(count, dtype=bool)
    for i in hits:
        row = topology.row_for_id[int(mapped.triangle_ids[i])]
        velocity_rows[i] = row
        kind = topology.classify(row, mapped.feature[i], mapped.barycentric[i], valid_faces)
        kinds[i] = kind
        eligible[i] = kind in ("face", "interior_edge")
    if len(hits):
        velocity = interpolate_surface_velocity(snapshot, velocity_rows, mapped.barycentric)
        if (velocity.shape != (count, 3) or not np.isfinite(velocity[hit]).all()
                or not np.isnan(velocity[~hit]).all()):
            raise ValueError("invalid current surface velocity readback")
    else:
        velocity = np.full((count, 3), np.nan)
    candidate_count = np.fromiter(map(len, candidate_ids), dtype=np.int64, count=count)
    for array in (velocity, eligible, candidate_count):
        array.setflags(write=False)
    return TrackingResult(stream_id, generation, mapped, velocity, eligible, tuple(kinds), tuple(routes),
                          tuple(candidate_ids), candidate_count)


class SurfaceTracker:
    """Single-consumer tracker; reset before changing streams.

    Supply both lips of any declared cut using current stable vertex IDs.
    An omitted/non-HIT query ID loses its seed at the end of each successful
    call. A failed call leaves the last successful publication intact.
    """
    def __init__(self, config: MappingConfig):
        if not isinstance(config, MappingConfig):
            raise TypeError("SurfaceTracker requires a MappingConfig")
        if config.max_distance is None or config.max_distance <= 0:
            raise ValueError("a finite positive risk radius (max_distance) is required")
        self._config = config
        self.reset()

    def reset(self):
        self._stream_id = None
        self._snapshot = None
        self._generation = 0
        self._topology = None
        self._seeds = {}

    def _transition(self, snapshot, stream_id, generation, cuts):
        old = self._snapshot
        if old is None:
            return "global_cold", True
        if stream_id != self._stream_id:
            raise ValueError("reset the tracker before switching streams")
        if snapshot.epoch < old.epoch:
            raise ValueError("stale snapshot epoch")
        if snapshot.epoch > old.epoch:
            return "global_epoch", True
        if (generation < self._generation or snapshot.topology_version < old.topology_version
                or snapshot.step_index < old.step_index):
            raise ValueError("stale snapshot or rebuild generation")
        if snapshot.step_index == old.step_index:
            if snapshot.time != old.time or snapshot.dt != old.dt:
                raise ValueError("an unadvanced snapshot must retain its physical time and dt")
        elif snapshot.time <= old.time:
            raise ValueError("a later step must advance physical time")
        if snapshot.topology_version != old.topology_version:
            return "global_topology", True
        if generation > self._generation:
            return "global_rebuild", True
        if cuts != self._topology.cut_edges or (snapshot is not old and any(
                not np.array_equal(getattr(snapshot, name), getattr(old, name)) for name in _CONNECTIVITY)):
            raise ValueError("connectivity or declared cuts changed without topology publication or rebuild")
        if snapshot is not old and snapshot.step_index == old.step_index and any(
                not np.array_equal(getattr(snapshot, name), getattr(old, name)) for name in _STATE_FIELDS):
            raise ValueError("inconsistent state for the same snapshot version")
        if snapshot.step_index > old.step_index + 1:
            return "global_cold", False
        return None, False

    def map(self, snapshot: ClothSnapshot, queries: QueryPoints, *, stream_id: str,
            rebuild_generation: int = 0, cut_edges=()) -> TrackingResult:
        """Map a complete risk batch against this snapshot and publish seeds atomically."""
        if not isinstance(snapshot, ClothSnapshot) or not isinstance(queries, QueryPoints):
            raise TypeError("expected ClothSnapshot and QueryPoints")
        if not isinstance(stream_id, str) or not stream_id.strip():
            raise ValueError("stream_id must be a nonempty string")
        if (isinstance(rebuild_generation, (bool, np.bool_))
                or not isinstance(rebuild_generation, (int, np.integer)) or rebuild_generation < 0):
            raise ValueError("rebuild_generation must be a nonnegative integer")
        generation = int(rebuild_generation)
        cuts = canonical_cut_edges(cut_edges)
        cold_route, rebuild = self._transition(snapshot, stream_id, generation, cuts)
        topology = SurfaceTopology(snapshot, cuts) if rebuild else self._topology
        seeds = self._seeds if cold_route is None else {}
        mapped, routes, candidates = _map_with_seeds(snapshot, queries, self._config, topology,
                                                     seeds, cold_route or "global_cold")
        result = _complete_result(snapshot, self._config, topology, mapped, routes, candidates, stream_id, generation)
        new_seeds = {int(query_id): int(face_id) for query_id, face_id, status in
                     zip(mapped.query_ids, mapped.triangle_ids, mapped.status) if status == MappingStatus.HIT}
        self._stream_id, self._snapshot, self._generation = stream_id, snapshot, generation
        self._topology, self._seeds = topology, new_seeds
        return result
