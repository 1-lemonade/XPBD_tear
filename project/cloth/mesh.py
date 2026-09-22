"""Triangle mesh, edge records, and adjacency rebuilding."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class Edge:
    id: int
    a: int
    b: int
    rest_length: float
    adjacent_triangles: list[int]
    active: bool = True
    broken: bool = False
    current_strain: float = 0.0

    @property
    def key(self) -> tuple[int, int]:
        return (min(self.a, self.b), max(self.a, self.b))

    def length(self, positions: np.ndarray) -> float:
        return float(np.linalg.norm(positions[self.a] - positions[self.b]))

    def strain(self, positions: np.ndarray) -> float:
        if self.rest_length <= 1e-12:
            return 0.0
        # Engineering strain: \( \varepsilon = (\ell - \ell_0) / \ell_0 \).
        return (self.length(positions) - self.rest_length) / self.rest_length


class TriangleMesh:
    """Mutable triangle index buffer with vertex/edge/triangle adjacency."""

    def __init__(self, triangles: np.ndarray, positions: np.ndarray):
        triangles = np.asarray(triangles, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (triangle_count, 3)")
        self.triangles = triangles.copy()
        self._initial_positions = np.asarray(positions, dtype=np.float64).copy()
        self.edges: list[Edge] = []
        self.vertex_to_triangles: dict[int, set[int]] = {}
        self.vertex_to_edges: dict[int, set[int]] = {}
        self.edge_lookup: dict[tuple[int, int], Edge] = {}
        self.broken_edge_keys: set[tuple[int, int]] = set()
        self.rebuild_adjacency(positions)

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        if self.triangles.size == 0:
            return 0
        return int(self.triangles.max()) + 1

    def rebuild_adjacency(self, positions: np.ndarray) -> None:
        old_rest = {edge.key: edge.rest_length for edge in self.edges}
        old_broken = set(self.broken_edge_keys)
        vertex_to_triangles: dict[int, set[int]] = defaultdict(set)
        edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
        for triangle_id, triangle in enumerate(self.triangles):
            for vertex in triangle:
                vertex_to_triangles[int(vertex)].add(triangle_id)
            for u, v in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                edge_triangles[(min(int(u), int(v)), max(int(u), int(v)))].append(triangle_id)

        self.vertex_to_triangles = dict(vertex_to_triangles)
        self.edges = []
        self.edge_lookup = {}
        self.vertex_to_edges = defaultdict(set)
        self.broken_edge_keys = old_broken
        for edge_id, (key, adjacent) in enumerate(sorted(edge_triangles.items())):
            a, b = key
            if key in old_rest:
                rest = old_rest[key]
            else:
                rest = float(np.linalg.norm(positions[a] - positions[b]))
            broken = key in self.broken_edge_keys
            edge = Edge(edge_id, a, b, rest, list(adjacent), active=not broken, broken=broken)
            self.edges.append(edge)
            self.edge_lookup[key] = edge
            self.vertex_to_edges[a].add(edge_id)
            self.vertex_to_edges[b].add(edge_id)

    def mark_edge_broken(self, edge_id: int) -> None:
        edge = self.edges[edge_id]
        edge.active = False
        edge.broken = True
        self.broken_edge_keys.add(edge.key)

    def edge_for_vertices(self, a: int, b: int) -> Edge | None:
        return self.edge_lookup.get((min(a, b), max(a, b)))

    def edge_strains(self, positions: np.ndarray, active_only: bool = True) -> dict[int, float]:
        return {
            edge.id: edge.strain(positions)
            for edge in self.edges
            if not active_only or edge.active
        }

    def append_vertex_reference(self, source_vertex: int, new_vertex: int, triangle_ids: set[int]) -> None:
        for triangle_id in triangle_ids:
            triangle = self.triangles[triangle_id]
            self.triangles[triangle_id] = np.where(triangle == source_vertex, new_vertex, triangle)
