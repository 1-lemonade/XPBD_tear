"""Runtime mesh fracture operations."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .mesh import TriangleMesh
from .particles import ParticleSoA


class TopologyManager:
    """Owns vertex splitting and adjacency rebuilds.

    A crack is represented by duplicating both endpoints on one side of a
    broken interior edge. ``split_vertex`` is the public primitive; splitting
    the companion endpoint in the same triangle component is necessary for a
    geometric slit rather than a hinge at the unsplit endpoint.
    """

    def __init__(self, particles: ParticleSoA, mesh: TriangleMesh):
        self.particles = particles
        self.mesh = mesh
        self.fracture_count = 0

    def split_vertex(self, vertex_id: int, edge_id: int, gap: float = 1.0e-3) -> tuple[int, ...]:
        edge = self.mesh.edges[edge_id]
        if edge.broken or len(edge.adjacent_triangles) < 2:
            return ()
        if vertex_id not in (edge.a, edge.b):
            raise ValueError("vertex_id must be an endpoint of edge_id")

        # Start from one side of the edge and flood the endpoint fan while
        # treating the broken edge as a cut. The untouched side keeps the old
        # vertex identifiers.
        side_triangles = self._triangle_component(edge.adjacent_triangles[1], vertex_id, edge.key)
        if not side_triangles:
            return ()
        other_vertex = edge.b if vertex_id == edge.a else edge.a
        split_ids = [vertex_id]
        for source in (vertex_id, other_vertex):
            new_id = self.particles.append(self.particles.position[source], source=source)
            self.mesh.append_vertex_reference(source, new_id, side_triangles)
            split_ids.append(new_id)

        self.mesh.mark_edge_broken(edge_id)
        self.mesh.rebuild_adjacency(self.particles.position)
        # Build rest lengths while the duplicate is coincident with its
        # source; only then open the visible crack by a tiny geometric gap.
        for new_id in split_ids[1:]:
            self._offset_duplicate(new_id, edge, gap)
        self.fracture_count += 1
        return tuple(split_ids)

    def remove_edge(self, edge_id: int) -> None:
        self.mesh.mark_edge_broken(edge_id)
        self.mesh.rebuild_adjacency(self.particles.position)

    def rebuild_adjacency(self) -> None:
        self.mesh.rebuild_adjacency(self.particles.position)

    def _triangle_component(self, start: int, vertex_id: int, cut_key: tuple[int, int]) -> set[int]:
        incident = self.mesh.vertex_to_triangles.get(vertex_id, set())
        visited: set[int] = set()
        queue = deque([start])
        while queue:
            triangle_id = queue.popleft()
            if triangle_id in visited or triangle_id not in incident:
                continue
            visited.add(triangle_id)
            triangle = self.mesh.triangles[triangle_id]
            for u, v in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                key = (min(int(u), int(v)), max(int(u), int(v)))
                if key == cut_key:
                    continue
                if vertex_id not in (u, v):
                    continue
                edge = self.mesh.edge_lookup.get(key)
                if edge is not None:
                    for neighbor in edge.adjacent_triangles:
                        if neighbor not in visited:
                            queue.append(neighbor)
        return visited

    def _offset_duplicate(self, new_id: int, edge, gap: float) -> None:
        direction = self.particles.position[edge.b] - self.particles.position[edge.a]
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            return
        # For the default planar cloth, this is the in-plane normal to the cut.
        normal = np.array([-direction[1], direction[0], 0.0], dtype=np.float64) / length
        self.particles.position[new_id] += normal * (gap * max(edge.rest_length, 1.0))
        self.particles.predicted_position[new_id] = self.particles.position[new_id]
        self.particles.previous_position[new_id] = self.particles.position[new_id]
