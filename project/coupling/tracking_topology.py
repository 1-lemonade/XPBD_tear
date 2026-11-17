"""Current face incidence and contact qualification on an open cloth surface."""
from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType

import numpy as np

from .contracts import MappingFeature


def canonical_cut_edges(cut_edges):
    """Normalize undirected pairs of current, stable vertex IDs."""
    edges = set()
    for pair in cut_edges:
        a, b = pair
        for vertex in (a, b):
            if (isinstance(vertex, bool) or not isinstance(vertex, (int, np.integer))
                    or not 0 <= int(vertex) <= np.iinfo(np.int64).max):
                raise ValueError("cut edge requires nonnegative int64 vertex IDs")
        edge = tuple(sorted((int(a), int(b))))
        if a == b or edge in edges:
            raise ValueError("cut endpoints must be distinct and cut edges unique")
        edges.add(edge)
    return frozenset(edges)


def archive_cut_edges(record):
    """Resolve an audited cloth record's cut root pairs to the current lip IDs.

    Roots identify which material edges were cut. They are never used to
    connect triangles. Other exposed boundaries remain free edges.
    """
    arrays = record.arrays
    cuts = canonical_cut_edges(arrays["cut_root_edges"])
    roots = dict(zip(arrays["vertex_ids"], arrays["root_vertex_ids"]))
    return frozenset(tuple(sorted((int(a), int(b)))) for a, b in arrays["edge_vertex_ids"]
                     if tuple(sorted((roots[a], roots[b]))) in cuts)


class SurfaceTopology:
    """One-ring candidates use actual shared edges of consistently wound faces."""

    def __init__(self, snapshot, cut_edges=()):
        self.cut_edges = canonical_cut_edges(cut_edges)
        self.row_for_id = MappingProxyType({int(v): i for i, v in enumerate(snapshot.triangle_ids)})
        self._faces = snapshot.vertex_ids[snapshot.triangles]
        incidence = defaultdict(list)
        for row, (a, b, c) in enumerate(self._faces.tolist()):
            for first, second, opposite in ((a, b, c), (b, c, a), (c, a, b)):
                if first != second:
                    edge = tuple(sorted((first, second)))
                    incidence[edge].append((row, first, opposite))
        if not self.cut_edges.issubset(incidence):
            raise ValueError("cut edge is absent from current surface")
        self._interior = {}
        rings = [{row} for row in range(len(self._faces))]
        for edge, sides in incidence.items():
            if edge in self.cut_edges or len(sides) != 2:
                continue
            (i, start_i, other_i), (j, start_j, other_j) = sides
            # Reject same-winding, duplicate and repeated-vertex faces.
            if (i != j and start_i != start_j and other_i != other_j
                    and other_i not in edge and other_j not in edge):
                self._interior[edge] = (i, j)
                rings[i].add(j)
                rings[j].add(i)
        self._rings = tuple(tuple(sorted(rows)) for rows in rings)

    def one_ring(self, row):
        return self._rings[row]

    def classify(self, row, feature, barycentric, valid_faces=None):
        if feature == MappingFeature.FACE:
            return "face"
        if feature == MappingFeature.VERTEX:
            return "vertex"
        if feature != MappingFeature.EDGE:
            raise ValueError("contact qualification requires a P20 candidate feature")
        # The selected P20 feature has exact closed-region weights. Keep this
        # distinction even for a query lying very close to an open lip.
        endpoints = self._faces[row][np.asarray(barycentric) != 0.]
        if len(endpoints) != 2:
            return "vertex"
        edge = tuple(sorted(map(int, endpoints)))
        if edge in self.cut_edges:
            return "crack_edge"
        neighbours = self._interior.get(edge)
        if neighbours is not None and valid_faces is not None:
            a, b = neighbours
            if valid_faces[a] and valid_faces[b]:
                return "interior_edge"
        return "free_edge"
