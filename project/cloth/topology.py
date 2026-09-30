"""Runtime mesh fracture operations."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .mesh import TriangleMesh
from .particles import ParticleSoA


class TopologyManager:
    """Owns vertex splitting and adjacency rebuilds.

    A cut first removes the shared-edge constraint. A vertex is duplicated
    only when cuts divide its incident triangle fan into separate sectors.
    This preserves intact neighboring triangles and lets successive cuts
    form a connected slit rather than detaching a triangle per event.
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

        endpoints = (vertex_id, edge.b if vertex_id == edge.a else edge.a)
        self.mesh.mark_edge_broken(edge_id)
        sectors = {source: self._fan_components(source) for source in endpoints}
        split_ids = [vertex_id]
        for source in endpoints:
            for side_triangles in sectors[source][1:]:
                new_id = self.particles.append(self.particles.position[source], source=source)
                self.mesh.register_duplicate(source, new_id)
                self.mesh.append_vertex_reference(source, new_id, side_triangles)
                split_ids.append(new_id)

        self.mesh.rebuild_adjacency(self.particles.position)
        for new_id in split_ids[1:]:
            self._offset_duplicate(new_id, edge, gap)
        self.fracture_count += 1
        return tuple(split_ids)

    def remove_edge(self, edge_id: int) -> None:
        self.mesh.mark_edge_broken(edge_id)
        self.mesh.rebuild_adjacency(self.particles.position)

    def rebuild_adjacency(self) -> None:
        self.mesh.rebuild_adjacency(self.particles.position)

    def _fan_components(self, vertex_id: int) -> list[set[int]]:
        incident = self.mesh.vertex_to_triangles.get(vertex_id, set())
        visited: set[int] = set()
        components: list[set[int]] = []
        for start in sorted(incident):
            if start in visited:
                continue
            component: set[int] = set()
            queue = deque([start])
            while queue:
                triangle_id = queue.popleft()
                if triangle_id in visited:
                    continue
                visited.add(triangle_id)
                component.add(triangle_id)
                triangle = self.mesh.triangles[triangle_id]
                for u, v in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                    if vertex_id not in (u, v):
                        continue
                    edge = self.mesh.edge_for_vertices(int(u), int(v))
                    if edge is not None and edge.active:
                        queue.extend(neighbor for neighbor in edge.adjacent_triangles if neighbor in incident and neighbor not in visited)
            components.append(component)
        return components

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
