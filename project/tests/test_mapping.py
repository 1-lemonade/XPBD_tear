"""Analytic boundary checks for the exhaustive NumPy surface mapping.

Every expected value below is pre-computed by hand (or by an exact closed-form
expression), not by calling the implementation a second time. Geometry is
chosen so the arithmetic that decides the closed-region classification is
exact: distances such as 0.25 and 0.5 are binary fractions, the
threshold-boundary heights are binary multiples of the tolerance, and the
axis-aligned cases keep their non-zero coordinates exactly representable.
(Not every value used here is a binary fraction -- 1e-6 is not -- so the
exactness claims in a given test are about the specific values it compares.)

Run with::

    conda run -n XPBD_tear python -m project.tests.test_mapping
"""

from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
import math
import tracemalloc
import unittest

import numpy as np

from project.coupling.contracts import (
    ClothSnapshot,
    MappingConfig,
    MappingFeature,
    MappingStatus,
    QueryPoints,
)
from project.coupling.mapping import map_points

SQUARE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
SQUARE_TRIANGLES = [(0, 1, 2), (0, 2, 3)]
FACE_QUERY = (0.25, 0.5, 0.25)
FACE_EXPECTED = dict(
    status=MappingStatus.HIT, triangle_id=1, vertex_ids=(0, 2, 3), feature=MappingFeature.FACE,
    barycentric=(0.5, 0.25, 0.25), closest_point=(0.25, 0.5, 0.0), distance=0.25,
    normal=(0.0, 0.0, 1.0), normal_offset=0.25,
)


def extra_fields(n, t):
    """Non-geometric ClothSnapshot fields with distinctive version values."""
    return dict(
        velocity=np.zeros((n, 3)),
        inverse_mass=np.ones(n),
        pinned=np.zeros(n, dtype=bool),
        vertex_ids=np.arange(n, dtype=np.int64),
        triangle_ids=np.arange(t, dtype=np.int64),
        epoch=3,
        step_index=7,
        topology_version=5,
        time=0.5,
        dt=1.0 / 120.0,
    )


def make_snapshot(positions, triangles, **overrides):
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    fields_ = extra_fields(positions.shape[0], triangles.shape[0])
    fields_.update(overrides)
    return ClothSnapshot(positions=positions, triangles=triangles, **fields_)


def make_queries(ids, positions):
    return QueryPoints(
        query_ids=np.asarray(ids, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64).reshape(-1, 3),
    )


def _exact_closest_point(vertices, query):
    """Exact closest point of the closed triangle, in rational arithmetic.

    Independent of the mapper. The plane projection is expressed through the
    cross-product barycentric form, which is the object the implementation
    compares. Exact rational arithmetic is exact for both that form and the
    Gram normal-equation form -- they agree to the last digit and neither loses
    information -- so the choice is about what is being compared, not about
    Fraction precision. The Gram form is only unusable in floating point, where
    ``g11*g22 - g12*g12`` is a cancelling difference of two quantities of order
    ``|edge|^4``; that is a property of the float64 evaluation, not of exact
    arithmetic. Falls back to the exact best edge when the projection leaves the
    triangle.

    Returns ``(point, squared_distance, weights_or_None)``.
    """
    def f(vector):
        return tuple(Fraction(float(value)) for value in vector)

    def sub(u, v):
        return tuple(x - y for x, y in zip(u, v))

    def add(u, v):
        return tuple(x + y for x, y in zip(u, v))

    def mul(u, k):
        return tuple(x * k for x in u)

    def dot(u, v):
        return sum((x * y for x, y in zip(u, v)), Fraction(0))

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    fa, fb, fc = (f(vertex) for vertex in vertices)
    fp = f(query)
    fab, fac, fap = sub(fb, fa), sub(fc, fa), sub(fp, fa)
    fn = cross(fab, fac)
    fn2 = dot(fn, fn)
    if fn2 != 0:
        v = dot(cross(fap, fac), fn) / fn2
        w = dot(cross(fab, fap), fn) / fn2
        u = 1 - v - w
        if min(u, v, w) >= 0:
            point = add(add(mul(fa, u), mul(fb, v)), mul(fc, w))
            offset = sub(point, fp)
            return point, dot(offset, offset), (u, v, w)
    best = None
    for first, second in ((fa, fb), (fa, fc), (fb, fc)):
        edge = sub(second, first)
        parameter = dot(edge, sub(fp, first)) / dot(edge, edge)
        parameter = max(Fraction(0), min(Fraction(1), parameter))
        point = add(first, mul(edge, parameter))
        offset = sub(point, fp)
        value = dot(offset, offset)
        if best is None or value < best[0]:
            best = (value, point)
    return best[1], best[0], None


def _seeded_sliver():
    """Vertices and query of a fixed-seed rotated sliver that defeated naive arithmetic.

    The generator sequence matches the rotated-sliver audit used during
    falsification: QR of a normal draw from ``default_rng(727)``, ten unit-scale
    cases at h=1e-6 consumed first, then the first h=1e-9 case scaled to base
    1e3 (case index 1). The query is the exact one that audit used, not the
    centroid normal, so the pinned configuration is the audited input. Before
    the error-free weight numerators this input sat at the tolerance boundary
    (ratio 0.983) and its batch contained a case that exceeded it.
    """
    rng = np.random.default_rng(727)
    for _ in range(10):
        rng.normal(size=(3, 3))
    for _ in range(2):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    height = 1.0e-9
    vertices = (np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, height, 0.0]]) * 1.0e3) @ rotation.T
    query = (np.asarray([0.75, height * 0.25, 0.5]) * 1.0e3) @ rotation.T
    return vertices, query


def _rotation(axis, angle):
    """Rotation matrix about ``axis``, used to build general-orientation cases.

    General orientation matters: an axis-aligned thin triangle keeps exact
    zero components and hides the cross-product cancellation that a rotated one
    exposes, which is how the rotated-sliver defect escaped this suite.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine, sine = math.cos(angle), math.sin(angle)
    one = 1.0 - cosine
    return np.array([
        [cosine + x * x * one, x * y * one - z * sine, x * z * one + y * sine],
        [y * x * one + z * sine, cosine + y * y * one, y * z * one - x * sine],
        [z * x * one - y * sine, z * y * one + x * sine, cosine + z * z * one],
    ])


def grid_case(rows, columns, query_count):
    xs = np.linspace(0.0, 1.0, columns)
    ys = np.linspace(0.0, 1.0, rows)
    positions = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    triangles = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a = row * columns + column
            b = a + 1
            c = a + columns
            d = c + 1
            triangles.append((a, c, b))
            triangles.append((b, c, d))
    snapshot = make_snapshot(positions, np.asarray(triangles, dtype=np.int64))
    query_ids = np.arange(query_count, dtype=np.int64)
    query_positions = positions[query_ids % positions.shape[0]].copy()
    query_positions[:, 2] = 0.25
    return snapshot, make_queries(query_ids, query_positions)


def assert_row_consistent(case, snapshot, queries, result, index, scale=1.0):
    """Contract-level invariants of one result row, independent of the mapper."""
    status = MappingStatus(int(result.status[index]))
    feature = MappingFeature(int(result.feature[index]))
    if status in (MappingStatus.EMPTY_SURFACE, MappingStatus.DEGENERATE_ONLY):
        case.assertEqual(int(result.triangle_ids[index]), -1)
        case.assertTrue(np.all(result.vertex_ids[index] == -1))
        case.assertEqual(feature, MappingFeature.NONE)
        for name in ("barycentric", "closest_point", "distance", "normal", "normal_offset"):
            case.assertTrue(np.isnan(getattr(result, name)[index]).all(), name)
        return
    tolerance = 1e-9 * max(1.0, scale)
    row = int(np.flatnonzero(snapshot.triangle_ids == int(result.triangle_ids[index]))[0])
    vertices = snapshot.triangles[row]
    weights = result.barycentric[index]
    np.testing.assert_allclose(weights @ snapshot.positions[vertices],
                               result.closest_point[index], atol=tolerance, rtol=0.0)
    np.testing.assert_array_equal(result.vertex_ids[index], snapshot.vertex_ids[vertices])
    offset = queries.positions[index] - result.closest_point[index]
    case.assertAlmostEqual(float(result.distance[index]), float(np.linalg.norm(offset)), places=12)
    case.assertAlmostEqual(float(result.normal_offset[index]),
                           float(offset @ result.normal[index]), places=12)
    case.assertAlmostEqual(float(np.linalg.norm(result.normal[index])), 1.0, places=12)
    case.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
    case.assertGreaterEqual(float(result.distance[index]), 0.0)
    case.assertIn(feature, (MappingFeature.FACE, MappingFeature.EDGE, MappingFeature.VERTEX))


def _exact_closest(positions, triangle, query):
    """Exact closest point of one closed triangle, computed with Fractions.

    This is an independent high-precision oracle, not a second call of the
    implementation: it applies the mathematical definition only (signed-area
    weights, exact zero weights for boundary features, clamped edge
    projections). Zero-area triangles are the only degeneracy it models, which
    is enough for the dyadic cases below. Returns ``None`` without a candidate.
    """
    a, b, c = (tuple(Fraction(float(value)) for value in positions[index])
               for index in triangle)
    p = tuple(Fraction(float(value)) for value in query)

    def sub(u, v):
        return tuple(x - y for x, y in zip(u, v))

    def add(u, v):
        return tuple(x + y for x, y in zip(u, v))

    def mul(u, k):
        return tuple(x * k for x in u)

    def dot(u, v):
        return sum((x * y for x, y in zip(u, v)), Fraction(0))

    def cross(u, v):
        return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])

    ab, ac = sub(b, a), sub(c, a)
    n = cross(ab, ac)
    n2 = dot(n, n)
    if n2 == 0:
        return None
    ap = sub(p, a)
    v = dot(cross(ap, ac), n) / n2
    w = dot(cross(ab, ap), n) / n2
    u = 1 - v - w
    if u >= 0 and v >= 0 and w >= 0:
        zeros = sum(1 for value in (u, v, w) if value == 0)
        feature = (MappingFeature.VERTEX if zeros >= 2
                   else MappingFeature.EDGE if zeros == 1 else MappingFeature.FACE)
        point = add(add(mul(a, u), mul(b, v)), mul(c, w))
        return point, dot(sub(p, point), sub(p, point)), (u, v, w), feature
    candidates = []
    for order, (first, second, weights_of) in enumerate((
        (a, b, lambda t: (1 - t, t, Fraction(0))),
        (a, c, lambda t: (1 - t, Fraction(0), t)),
        (b, c, lambda t: (Fraction(0), 1 - t, t)),
    )):
        edge = sub(second, first)
        length2 = dot(edge, edge)
        if length2 == 0:
            continue
        parameter = dot(edge, sub(p, first)) / length2
        parameter = max(Fraction(0), min(Fraction(1), parameter))
        point = add(first, mul(edge, parameter))
        feature = MappingFeature.VERTEX if parameter in (0, 1) else MappingFeature.EDGE
        candidates.append((dot(sub(p, point), sub(p, point)), order, point,
                           weights_of(parameter), feature))
    if not candidates:
        return None
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return best[2], best[0], best[3], best[4]


def exact_mapping(positions, triangles, triangle_ids, query):
    """Exact winner under the strict rule: minimum distance, then smallest ID."""
    best = None
    for row, triangle in enumerate(triangles):
        result = _exact_closest(positions, triangle, query)
        if result is None:
            continue
        point, squared, weights, feature = result
        entry = (squared, triangle_ids[row], point, weights, feature)
        if best is None or entry[:2] < best[:2]:
            best = entry
    return best


class TestAnalyticRegions(unittest.TestCase):
    def check_single(self, positions, triangles, query, expected, **overrides):
        snapshot = make_snapshot(positions, triangles, **overrides)
        queries = make_queries([0], [query])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(expected["status"]))
        if expected["status"] != MappingStatus.HIT:
            return snapshot, queries, result
        self.assertEqual(int(result.triangle_ids[0]), expected["triangle_id"])
        self.assertEqual(int(result.feature[0]), int(expected["feature"]))
        np.testing.assert_array_equal(result.vertex_ids[0], np.asarray(expected["vertex_ids"]))
        np.testing.assert_allclose(result.barycentric[0], expected["barycentric"], atol=1e-12)
        np.testing.assert_allclose(result.closest_point[0], expected["closest_point"], atol=1e-12)
        self.assertAlmostEqual(float(result.distance[0]), expected["distance"], places=12)
        np.testing.assert_allclose(result.normal[0], expected["normal"], atol=1e-12)
        self.assertAlmostEqual(float(result.normal_offset[0]), expected["normal_offset"], places=12)
        assert_row_consistent(self, snapshot, queries, result, 0)
        return snapshot, queries, result

    def test_face_interior(self):
        self.check_single(SQUARE, SQUARE_TRIANGLES, FACE_QUERY, FACE_EXPECTED)

    def test_edge_interior(self):
        # (0.5, 0, 0.25) projects exactly onto the middle of edge (v0, v1). The
        # in-plane weights are (0.5, 0.5, 0.0): one *exact* zero weight is the
        # threshold-free EDGE classification, not a distance tie.
        self.check_single(SQUARE, SQUARE_TRIANGLES, (0.5, 0.0, 0.25), dict(
            status=MappingStatus.HIT, triangle_id=0, vertex_ids=(0, 1, 2), feature=MappingFeature.EDGE,
            barycentric=(0.5, 0.5, 0.0), closest_point=(0.5, 0.0, 0.0), distance=0.25,
            normal=(0.0, 0.0, 1.0), normal_offset=0.25,
        ))

    def test_vertex(self):
        # Two exact zero weights mean VERTEX; again no spatial tolerance.
        self.check_single(SQUARE, SQUARE_TRIANGLES, (0.0, 0.0, 0.25), dict(
            status=MappingStatus.HIT, triangle_id=0, vertex_ids=(0, 1, 2), feature=MappingFeature.VERTEX,
            barycentric=(1.0, 0.0, 0.0), closest_point=(0.0, 0.0, 0.0), distance=0.25,
            normal=(0.0, 0.0, 1.0), normal_offset=0.25,
        ))

    def test_projection_outside_face_uses_boundary(self):
        # (-0.5, 0.5) is left of the square, closest to edge v0-v3 of row 1.
        self.check_single(SQUARE, SQUARE_TRIANGLES, (-0.5, 0.5, 0.2), dict(
            status=MappingStatus.HIT, triangle_id=1, vertex_ids=(0, 2, 3), feature=MappingFeature.EDGE,
            barycentric=(0.5, 0.0, 0.5), closest_point=(0.0, 0.5, 0.0),
            distance=float(np.sqrt(0.25 + 0.04)), normal=(0.0, 0.0, 1.0), normal_offset=0.2,
        ))

    def test_reversed_winding_flips_normal_and_offset(self):
        reversed_triangles = [(0, 2, 1), (0, 3, 2)]
        expected = dict(FACE_EXPECTED)
        expected.update(triangle_id=1, vertex_ids=(0, 3, 2), normal=(0.0, 0.0, -1.0),
                        normal_offset=-0.25)
        self.check_single(SQUARE, reversed_triangles, FACE_QUERY, expected)

    def test_normal_offset_is_signed_plane_distance(self):
        # Below the surface the offset is negative and equals dot(query, n).
        expected = dict(FACE_EXPECTED)
        expected.update(normal_offset=-0.25, closest_point=(0.25, 0.5, 0.0))
        _, _, result = self.check_single(SQUARE, SQUARE_TRIANGLES, (0.25, 0.5, -0.25), expected)
        self.assertAlmostEqual(float(result.normal_offset[0]), -0.25, places=12)

    def test_barycentric_reconstruction_matches_for_many_queries(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        positions = [(0.1, 0.2, 0.3), (0.9, 0.1, -0.4), (0.6, 0.6, 0.05),
                     (1.4, 0.5, 0.2), (-0.2, 1.3, 0.1), (0.0, 0.0, 0.0)]
        queries = make_queries([5, 4, 3, 2, 1, 0], positions)
        result = map_points(snapshot, queries)
        for index in range(len(positions)):
            assert_row_consistent(self, snapshot, queries, result, index)


class TestSelectionRule(unittest.TestCase):
    def test_shared_edge_tie_prefers_smallest_triangle_id(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [(0.5, 0.5, 0.3)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.triangle_ids[0]), 0)
        self.assertEqual(int(result.feature[0]), int(MappingFeature.EDGE))
        self.assertAlmostEqual(float(result.distance[0]), 0.3, places=12)

    def test_shared_edge_tie_follows_permuted_identity_not_row_order(self):
        for triangle_ids, expected_id, expected_vertices in (
            ([5, 2], 2, (0, 2, 3)),
            ([1, 7], 1, (0, 1, 2)),
        ):
            snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=triangle_ids)
            queries = make_queries([0], [(0.5, 0.5, 0.3)])
            result = map_points(snapshot, queries)
            self.assertEqual(int(result.triangle_ids[0]), expected_id)
            np.testing.assert_array_equal(result.vertex_ids[0], np.asarray(expected_vertices))
            self.assertAlmostEqual(float(result.distance[0]), 0.3, places=12)

    def test_coincident_triangles_resolve_by_identity(self):
        coincident = [(0, 1, 2), (0, 1, 2)]
        snapshot = make_snapshot(SQUARE[:3], coincident, triangle_ids=[9, 3])
        queries = make_queries([0], [(0.5, 0.25, 0.5)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.triangle_ids[0]), 3)
        self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE))
        self.assertAlmostEqual(float(result.distance[0]), 0.5, places=12)

    def test_tie_window_is_not_transitive(self):
        # Distances 10.25, 9.75 and 9.5 with tie_atol=0.5: the window around
        # d_min=9.5 admits 9.75 but not 10.25, even though 10.25 is within 0.5
        # of 9.75. The smallest identity in the window is 20, so a chained
        # comparison would incorrectly return identity 1.
        planes = [-0.25, 0.25, 0.5]
        positions = []
        triangles = []
        for index, z in enumerate(planes):
            base = 3 * index
            positions.extend([(-1.0, -1.0, z), (1.0, -1.0, z), (0.0, 1.0, z)])
            triangles.append((base, base + 1, base + 2))
        snapshot = make_snapshot(positions, triangles, triangle_ids=[1, 20, 30])
        queries = make_queries([0], [(0.0, 0.0, 10.0)])
        result = map_points(snapshot, queries, MappingConfig(tie_atol=0.5, tie_rtol=0.0))
        self.assertEqual(int(result.triangle_ids[0]), 20)
        self.assertAlmostEqual(float(result.distance[0]), 9.75, places=12)
        self.assertAlmostEqual(float(result.closest_point[0][2]), 0.25, places=12)

    def test_max_distance_equal_distance_is_hit(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [(0.0, 0.0, 0.5)])
        result = map_points(snapshot, queries, MappingConfig(max_distance=0.5))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.feature[0]), int(MappingFeature.VERTEX))
        self.assertAlmostEqual(float(result.distance[0]), 0.5, places=12)

    def test_max_distance_nextafter_boundary(self):
        just_above = float(np.nextafter(0.5, 2.0))
        just_below = float(np.nextafter(0.5, 0.0))
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0, 1], [(0.0, 0.0, just_above), (0.0, 0.0, just_below)])
        result = map_points(snapshot, queries, MappingConfig(max_distance=0.5))
        self.assertEqual(int(result.status[0]), int(MappingStatus.OUTSIDE_THRESHOLD))
        self.assertEqual(int(result.status[1]), int(MappingStatus.HIT))
        self.assertAlmostEqual(float(result.distance[0]), just_above, places=15)
        self.assertAlmostEqual(float(result.distance[1]), just_below, places=15)

    def test_outside_threshold_keeps_candidate_geometry(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [FACE_QUERY])
        result = map_points(snapshot, queries, MappingConfig(max_distance=0.1))
        self.assertEqual(int(result.status[0]), int(MappingStatus.OUTSIDE_THRESHOLD))
        self.assertEqual(int(result.triangle_ids[0]), 1)
        self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE))
        np.testing.assert_allclose(result.barycentric[0], (0.5, 0.25, 0.25), atol=1e-12)
        self.assertAlmostEqual(float(result.distance[0]), 0.25, places=12)
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_broad_tie_rtol_is_defined_behaviour(self):
        # The frozen contract allows large finite tie tolerances; everything
        # ties and the smallest identity of the whole candidate set wins.
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [FACE_QUERY])
        result = map_points(snapshot, queries, MappingConfig(tie_rtol=1e30))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.triangle_ids[0]), 0)
        self.assertAlmostEqual(float(result.distance[0]), float(np.sqrt(0.09375)), places=12)

    def test_straddling_tie_uses_selected_candidate_for_max_distance(self):
        # d = 10.25 and 9.75 share one window, so the smaller identity (5) is
        # the farther candidate. The frozen decision tests the selected
        # candidate's distance, not the global minimum, so max_distance = 10.0
        # must report OUTSIDE_THRESHOLD even though the other tied candidate is
        # inside the radius. tie_atol = 0.75 keeps both window borders away from
        # the exact boundary, which the frozen prompt judges by computed values.
        planes = [-0.25, 0.25]
        positions = []
        triangles = []
        for index, z in enumerate(planes):
            base = 3 * index
            positions.extend([(-1.0, -1.0, z), (1.0, -1.0, z), (0.0, 1.0, z)])
            triangles.append((base, base + 1, base + 2))
        snapshot = make_snapshot(positions, triangles, triangle_ids=[5, 9])
        queries = make_queries([0], [(0.0, 0.0, 10.0)])
        result = map_points(snapshot, queries,
                            MappingConfig(tie_atol=0.75, tie_rtol=0.0, max_distance=10.0))
        self.assertEqual(int(result.triangle_ids[0]), 5)
        self.assertAlmostEqual(float(result.distance[0]), 10.25, places=12)
        self.assertEqual(int(result.status[0]), int(MappingStatus.OUTSIDE_THRESHOLD))
        assert_row_consistent(self, snapshot, queries, result, 0, scale=10.25)

    def test_strict_zero_tie_resolves_by_identity(self):
        # Exact strict tie (tie_atol = tie_rtol = 0): the query lies on the
        # shared diagonal, both triangles report the very same point built by
        # the same affine construction, so the distances are bitwise equal and
        # only the smallest identity can decide.
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=[5, 2])
        queries = make_queries([0], [(0.5, 0.5, 0.3)])
        result = map_points(snapshot, queries, MappingConfig(tie_atol=0.0, tie_rtol=0.0))
        self.assertEqual(int(result.triangle_ids[0]), 2)
        np.testing.assert_array_equal(result.vertex_ids[0], (0, 2, 3))
        self.assertEqual(int(result.feature[0]), int(MappingFeature.EDGE))
        np.testing.assert_allclose(result.barycentric[0], (0.5, 0.5, 0.0), atol=1e-15)
        np.testing.assert_allclose(result.closest_point[0], (0.5, 0.5, 0.0), atol=1e-15)
        self.assertAlmostEqual(float(result.distance[0]), 0.3, places=12)
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_mixed_statuses_in_one_batch(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0, 1], [(0.25, 0.5, 0.05), (0.25, 0.5, 0.5)])
        result = map_points(snapshot, queries, MappingConfig(max_distance=0.1))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.status[1]), int(MappingStatus.OUTSIDE_THRESHOLD))
        self.assertAlmostEqual(float(result.distance[0]), 0.05, places=12)
        self.assertAlmostEqual(float(result.distance[1]), 0.5, places=12)
        assert_row_consistent(self, snapshot, queries, result, 0)
        assert_row_consistent(self, snapshot, queries, result, 1)

    def test_unreferenced_position_rows_keep_identity_mapping(self):
        positions = list(SQUARE) + [(7.0, 7.0, 7.0)]
        snapshot = make_snapshot(positions, SQUARE_TRIANGLES)
        queries = make_queries([0], [FACE_QUERY])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.triangle_ids[0]), 1)
        np.testing.assert_array_equal(result.vertex_ids[0], (0, 2, 3))
        self.assertAlmostEqual(float(result.distance[0]), 0.25, places=12)


class TestSurfaceStates(unittest.TestCase):
    def test_empty_triangle_set_is_empty_surface(self):
        snapshot = make_snapshot([(0.0, 0.0, 0.0)], np.zeros((0, 3), dtype=np.int64))
        queries = make_queries([4], [(1.0, 2.0, 3.0)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.EMPTY_SURFACE))
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_empty_vertices_and_triangles(self):
        snapshot = make_snapshot(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        queries = make_queries([0, 1], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        result = map_points(snapshot, queries)
        np.testing.assert_array_equal(result.status, int(MappingStatus.EMPTY_SURFACE))
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_repeated_index_triangle_is_degenerate_only(self):
        # Duplicate position rows are legal identities; the face itself is
        # geometrically degenerate and must not be used as a segment.
        snapshot = make_snapshot(SQUARE, [(0, 0, 1)])
        queries = make_queries([0], [(0.5, 0.0, 0.25)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.DEGENERATE_ONLY))
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_collinear_triangle_is_degenerate_only(self):
        snapshot = make_snapshot([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], [(0, 1, 2)])
        queries = make_queries([0], [(1.0, 0.0, 0.0)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.DEGENERATE_ONLY))

    def test_mixed_valid_and_degenerate_uses_valid_only(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                     (10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (12.0, 10.0, 0.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2), (3, 4, 5)])
        # The query sits exactly on the collinear triple; if degenerate faces
        # competed as segments the distance would be 0.
        queries = make_queries([0], [(11.0, 10.0, 0.0)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.triangle_ids[0]), 0)
        self.assertEqual(int(result.feature[0]), int(MappingFeature.VERTEX))
        self.assertAlmostEqual(float(result.distance[0]), float(np.sqrt(200.0)), places=12)

    def test_sliver_passing_area_filter_keeps_face_hit(self):
        # Height 1e-6 keeps A2 = 1e-12 above the (1e-12 * L2) ** 2 = 1e-24
        # threshold, and the query projects strictly inside.
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1e-6, 0.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2)])
        queries = make_queries([0], [(0.5, 0.9e-6, 0.25)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE))
        self.assertAlmostEqual(float(result.distance[0]), 0.25, places=9)
        np.testing.assert_allclose(result.barycentric[0], (0.05, 0.05, 0.9), atol=1e-9)
        np.testing.assert_allclose(result.closest_point[0], (0.5, 0.9e-6, 0.0), atol=1e-9)
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_very_thin_sliver_stays_contract_valid(self):
        # Height 1e-9 is inside the area filter (A2 = 1e-18 > 1e-24) and the
        # projection is strictly inside, so the exact-region classification must
        # report FACE with the analytic weights instead of falling back to a
        # boundary candidate.
        positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1e-9, 0.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2)])
        queries = make_queries([0], [(0.5, 0.9e-9, 0.25)])
        result = map_points(snapshot, queries)
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE))
        self.assertAlmostEqual(float(result.distance[0]), 0.25, places=9)
        np.testing.assert_allclose(result.barycentric[0], (0.05, 0.05, 0.9), atol=1e-9)
        np.testing.assert_allclose(result.closest_point[0], (0.5, 0.9e-9, 0.0), atol=1e-9)
        assert_row_consistent(self, snapshot, queries, result, 0)

    def test_sliver_projection_is_never_stolen_by_edge_rounding(self):
        # Regression for the review counter-example: the in-plane projection of
        # (0.75, h/4, 0.5) is strictly inside, and the analytic answer is
        # FACE with weights (0.25, 0.5, 0.25) at every height. A boundary
        # candidate that ties only through distance rounding must not win.
        for height in (1e-6, 1e-10, 2e-12):
            positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, height, 0.0)]
            snapshot = make_snapshot(positions, [(0, 1, 2)])
            queries = make_queries([93], [(0.75, height * 0.25, 0.5)])
            result = map_points(snapshot, queries)
            message = f"height={height!r}"
            self.assertEqual(int(result.status[0]), int(MappingStatus.HIT), message)
            self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE), message)
            np.testing.assert_allclose(result.barycentric[0], (0.25, 0.5, 0.25),
                                       atol=1e-9, err_msg=message)
            np.testing.assert_allclose(result.closest_point[0],
                                       (0.75, height * 0.25, 0.0), atol=1e-12, err_msg=message)
            self.assertAlmostEqual(float(result.distance[0]), 0.5, places=12, msg=message)
            assert_row_consistent(self, snapshot, queries, result, 0)

    def test_huge_area_rtol_marks_everything_degenerate(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [FACE_QUERY])
        result = map_points(snapshot, queries, MappingConfig(area_rtol=1e6))
        self.assertEqual(int(result.status[0]), int(MappingStatus.DEGENERATE_ONLY))

    def test_rotated_thin_triangle_keeps_the_true_closest_point(self):
        # Regression for the rotated-sliver failures: the weight numerators are
        # triple products of order |edge|^2 that cancel down to the area scale,
        # and |n|^2 is a difference of nearly parallel edge components, so with
        # plain float64 the weights carried ~5e-7 of error and the reported point
        # landed ~2e-5 away from the closest point -- 20x the 1e-9*scale budget.
        # Error-free transformations in the triangle tables and the two weight
        # numerators fix it.
        #
        # The expected values below are exact rational closest points computed
        # independently of the mapper (cross-product barycentric form with an
        # exact boundary fallback), and the reference error budget is the
        # documented one, not a relaxed one. The inputs cover both construction
        # orders, two rotation axes, a thicker control, and one fixed-seed
        # rotated sample.
        rotation = _rotation((1.0, 2.0, 3.0), math.radians(37.0))
        rotation_311 = _rotation((1.0, 2.0, 3.0), math.radians(311.0))
        geometries = (
            # (a) the original counter-example: a 2e3-long triangle 1e-6 thick,
            # rotated. Built by scaling first, then rotating.
            ("long thin triangle, h=1e-6, 37 deg, scaled then rotated",
             np.asarray([[-1.0e3, 0.0, 0.0], [1.0e3, 0.0, 0.0], [0.0, 1.0e-6, 0.0]]) @ rotation.T),
            # (b) the same nominal geometry written in the other order. This is a
            # separate case, not a duplicate of (a): `raw * 1e3` scales the height
            # too, so the raw height here is 1e-9 to keep the thickness at 1e-6.
            ("long thin triangle, h=1e-6, 37 deg, rotated then scaled",
             np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0e-9, 0.0]]) @ rotation.T * 1.0e3),
            # (c) a genuinely thicker variant (height 1e-4 after the x1000) kept
            # as a second construction-order control.
            ("thicker variant, h=1e-4, 37 deg, rotated then scaled",
             np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0e-7, 0.0]]) @ rotation.T * 1.0e3),
            # (d) the original second counter-example, other axis, height
            # 1e-4/1e3 = 1e-7. Kept verbatim.
            ("long thin triangle, h=1e-7, 311 deg, scaled then rotated",
             np.asarray([[-1.0e3, 0.0, 0.0], [1.0e3, 0.0, 0.0], [0.0, 1.0e-7, 0.0]]) @ rotation_311.T),
            # (e) the seeded case audited before the fix, with the exact query
            # that audit used (not the centroid normal).
            ("seeded QR case, h=1e-9 scaled to 1e-6, 1e3 base", _seeded_sliver()),
        )
        for label, geometry in geometries:
            if isinstance(geometry, tuple):
                vertices, query = geometry
            else:
                vertices = geometry
                normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
                normal = normal / np.linalg.norm(normal)
                query = (vertices[0] + vertices[1] + vertices[2]) / 3.0 + normal * 0.25
            snapshot = make_snapshot(vertices, [(0, 1, 2)], triangle_ids=[7])
            queries = make_queries([5], [query])
            result = map_points(snapshot, queries)
            scale = float(np.abs(vertices).max())
            tolerance = 1e-9 * max(1.0, scale)
            exact_point, exact_squared, exact_weights = _exact_closest_point(vertices, query)
            exact_point = np.asarray([float(value) for value in exact_point])
            self.assertEqual(int(result.status[0]), int(MappingStatus.HIT), label)
            self.assertEqual(int(result.triangle_ids[0]), 7, label)
            point_error = float(np.linalg.norm(result.closest_point[0] - exact_point))
            self.assertLessEqual(point_error, tolerance,
                                 f"{label}: point error {point_error:.3e} > {tolerance:.3e}")
            # The distance must match the exact minimum, and the reported triple
            # has to reproduce the reported point.
            exact_distance = math.sqrt(float(exact_squared))
            self.assertLessEqual(abs(float(result.distance[0]) - exact_distance),
                                 1e-9 * (1.0 + exact_distance), label)
            if exact_weights is not None:
                # An interior projection is where the mapper's classification is
                # testable: it must report FACE, not a boundary fallback.
                self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE), label)
            # The reported point must be the closest one, i.e. no worse than the
            # exact minimiser. For a thin triangle this is the meaningful check:
            # individual weights are not reproducible to 1e-9 there because the
            # spine and the perpendicular are nearly parallel, so a weight shift
            # that leaves the point invariant is expected and allowed. What the
            # contract does require -- weight sum, bounds and reconstruction --
            # is asserted below and is independent of that conditioning.
            reported_squared = sum((float(result.closest_point[0][i]) - float(queries.positions[0][i]))
                                   ** 2 for i in range(3))
            exact_float_squared = float(exact_squared)
            self.assertLessEqual(reported_squared, exact_float_squared
                                 + (tolerance * tolerance) + 1e-18, label)
            weights = result.barycentric[0]
            self.assertTrue(bool((weights >= 0.0).all() and (weights <= 1.0).all()), label)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=12, msg=label)
            np.testing.assert_allclose(weights @ vertices, result.closest_point[0],
                                       atol=1e-9, rtol=1e-9, err_msg=label)
            assert_row_consistent(self, snapshot, queries, result, 0, scale=scale)

    def test_thin_triangle_weights_stay_physically_meaningful(self):
        # The reconstruction of the reported weights must equal the reported
        # point for the pathological orientation too, and the reported normal
        # must be unit length, because the frozen result contract validates
        # both. A thin triangle is exactly where a naive normalisation of the
        # head-only cross product drifts (it was 4e-8 off before the fix).
        rotation = _rotation((1.0, 2.0, 3.0), math.radians(37.0))
        vertices = np.asarray([[-1.0e3, 0.0, 0.0], [1.0e3, 0.0, 0.0],
                               [0.0, 1.0e-6, 0.0]]) @ rotation.T
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        normal = normal / np.linalg.norm(normal)
        query = (vertices[0] + vertices[1] + vertices[2]) / 3.0 + normal * 0.25
        result = map_points(make_snapshot(vertices, [(0, 1, 2)]),
                            make_queries([3], [query]))
        weights = result.barycentric[0]
        self.assertTrue((weights >= 0.0).all() and (weights <= 1.0).all())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=15)
        self.assertAlmostEqual(float(np.linalg.norm(result.normal[0])), 1.0, places=12)
        np.testing.assert_allclose(weights @ vertices, result.closest_point[0],
                                   atol=1e-12, rtol=0.0)

    def test_empty_query_skips_surface_derivation(self):
        # A legal snapshot whose derived products would overflow must still
        # return the documented empty result when there is no query to map:
        # nothing is derived, so nothing can be non-finite. With a query the
        # same snapshot must still raise with the triangle identity.
        positions = [(0.0, 0.0, 0.0), (1.0e200, 0.0, 0.0), (0.0, 1.0e200, 0.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2)], triangle_ids=[11])
        empty = make_queries([], np.empty((0, 3)))
        result = map_points(snapshot, empty)
        self.assertEqual(result.query_ids.shape, (0,))
        self.assertEqual(result.distance.shape, (0,))
        self.assertEqual(result.vertex_ids.shape, (0, 3))
        self.assertEqual(result.status.shape, (0,))
        self.assertEqual(result.feature.shape, (0,))
        self.assertEqual(result.version, snapshot.version)
        with self.assertRaises(ValueError) as raised:
            map_points(snapshot, make_queries([1], [(0.0, 0.0, 1.0)]))
        self.assertIn("triangle_id=11", str(raised.exception))
        # Empty input objects and invalid config are still rejected first.
        with self.assertRaises(ValueError):
            make_queries([1], np.zeros((0, 3)))
        with self.assertRaises(TypeError):
            map_points(snapshot, object())
        with self.assertRaises(TypeError):
            map_points(object(), empty)
        with self.assertRaises(ValueError):
            map_points(snapshot, empty, MappingConfig(area_rtol=-1.0))
        # T=0 with Q=0 and a degenerate-only surface with Q=0.
        bare = make_snapshot(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
        self.assertEqual(map_points(bare, empty).distance.shape, (0,))
        collinear = make_snapshot([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                                  [(0, 1, 2)])
        self.assertEqual(map_points(collinear, empty).status.shape, (0,))

    def test_area_rtol_overflow_raises(self):
        # (area_rtol * L2) ** 2 is a derived quantity: when it overflows, the
        # mapper must diagnose it with the triangle identity instead of
        # silently filtering every triangle into DEGENERATE_ONLY.
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=[4, 9])
        queries = make_queries([77], [(0.25, 0.25, 0.5)])
        for area_rtol in (1e200, 1e308):
            with self.assertRaises(ValueError, msg=f"area_rtol={area_rtol!r}") as raised:
                map_points(snapshot, queries, MappingConfig(area_rtol=area_rtol))
            self.assertIn("area threshold", str(raised.exception))
            self.assertIn("triangle_id=4", str(raised.exception))

    def test_area_filter_boundary_is_inclusive(self):
        # With a unit base, A2 = h**2 and the frozen threshold is
        # (area_rtol * L2) ** 2 = area_rtol**2, so h == area_rtol is degenerate
        # (<=) and 2 * area_rtol is accepted. Both heights are exact binary
        # multiples of the configured tolerance.
        unit = float(MappingConfig().area_rtol)
        for height, expected in ((unit / 2.0, MappingStatus.DEGENERATE_ONLY),
                                 (unit, MappingStatus.DEGENERATE_ONLY),
                                 (2.0 * unit, MappingStatus.HIT)):
            positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, height, 0.0)]
            snapshot = make_snapshot(positions, [(0, 1, 2)])
            queries = make_queries([0], [(0.5, 0.0, 0.25)])
            result = map_points(snapshot, queries)
            self.assertEqual(int(result.status[0]), int(expected), f"height={height!r}")
            if expected == MappingStatus.HIT:
                self.assertAlmostEqual(float(result.distance[0]), 0.25, places=12)


class TestScalesFramesAndBatches(unittest.TestCase):
    def test_scales(self):
        for scale in (1e-3, 1.0, 1e3):
            positions = scale * np.asarray(SQUARE)
            snapshot = make_snapshot(positions, SQUARE_TRIANGLES)
            queries = make_queries([0, 1], [scale * np.asarray(FACE_QUERY),
                                            (0.0, 0.0, 0.25 * scale)])
            result = map_points(snapshot, queries)
            tolerance = 1e-9 * max(1.0, scale)
            np.testing.assert_allclose(result.distance, [0.25 * scale, 0.25 * scale],
                                       atol=tolerance, rtol=0.0)
            np.testing.assert_allclose(result.normal_offset, [0.25 * scale, 0.25 * scale],
                                       atol=tolerance, rtol=0.0)
            np.testing.assert_allclose(result.barycentric[0], (0.5, 0.25, 0.25), atol=1e-12)
            self.assertEqual(int(result.triangle_ids[0]), 1)
            self.assertEqual(int(result.feature[0]), int(MappingFeature.FACE))
            self.assertEqual(int(result.feature[1]), int(MappingFeature.VERTEX))
            np.testing.assert_array_equal(result.status, int(MappingStatus.HIT))

    def test_rigid_rotation_and_translation(self):
        rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        translation = np.array([0.25, -1.5, 2.0])
        positions = np.asarray(SQUARE) @ rotation.T + translation
        snapshot = make_snapshot(positions, SQUARE_TRIANGLES)
        query = np.asarray(FACE_QUERY) @ rotation.T + translation
        queries = make_queries([0], [query])
        result = map_points(snapshot, queries)
        np.testing.assert_allclose(result.normal[0], (0.0, -1.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(result.barycentric[0], (0.5, 0.25, 0.25), atol=1e-12)
        np.testing.assert_allclose(result.closest_point[0],
                                   np.array([0.25, 0.5, 0.0]) @ rotation.T + translation,
                                   atol=1e-12)
        self.assertAlmostEqual(float(result.distance[0]), 0.25, places=12)
        self.assertAlmostEqual(float(result.normal_offset[0]), 0.25, places=12)
        self.assertEqual(int(result.triangle_ids[0]), 1)
        assert_row_consistent(self, snapshot, queries, result, 0, scale=2.0)
        # The exact-zero boundary classification survives the rigid motion:
        # an edge-interior projection stays EDGE with (0.5, 0.5, 0.0) and a
        # vertex projection stays VERTEX with (1.0, 0.0, 0.0).
        for local, feature, weights in (((0.5, 0.0, 0.25), MappingFeature.EDGE, (0.5, 0.5, 0.0)),
                                        ((0.0, 0.0, 0.25), MappingFeature.VERTEX, (1.0, 0.0, 0.0))):
            moved = make_queries([0], [np.asarray(local) @ rotation.T + translation])
            moved_result = map_points(snapshot, moved)
            self.assertEqual(int(moved_result.feature[0]), int(feature), local)
            np.testing.assert_allclose(moved_result.barycentric[0], weights, atol=1e-15)
            self.assertAlmostEqual(float(moved_result.distance[0]), 0.25, places=12)
            assert_row_consistent(self, snapshot, moved, moved_result, 0, scale=2.0)

    def test_query_order_and_identities_are_preserved(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        query_ids = [10, -7, 0, 4]
        positions = [(0.25, 0.5, 0.25), (0.0, 0.0, 0.25), (2.0, 2.0, 0.0), (0.5, 0.5, 0.3)]
        queries = make_queries(query_ids, positions)
        result = map_points(snapshot, queries)
        np.testing.assert_array_equal(result.query_ids, query_ids)
        self.assertEqual(result.query_ids.shape[0], 4)
        np.testing.assert_allclose(
            result.distance, [0.25, 0.25, float(np.hypot(1.0, 1.0)), 0.3], atol=1e-12, rtol=0.0)
        # (2, 2, 0) is closest to vertex v2 = (1, 1, 0), which both triangles
        # share, so the smallest triangle identity must win.
        self.assertEqual(int(result.triangle_ids[2]), 0)
        self.assertEqual(int(result.feature[2]), int(MappingFeature.VERTEX))
        np.testing.assert_array_equal(result.barycentric[2], (0.0, 0.0, 1.0))

    def test_empty_query_batch(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        result = map_points(snapshot, make_queries([], np.zeros((0, 3))))
        self.assertEqual(result.query_ids.shape, (0,))
        self.assertEqual(result.triangle_ids.shape, (0,))
        self.assertEqual(result.vertex_ids.shape, (0, 3))
        self.assertEqual(result.barycentric.shape, (0, 3))
        self.assertEqual(result.closest_point.shape, (0, 3))
        self.assertEqual(result.distance.shape, (0,))
        self.assertEqual(result.normal.shape, (0, 3))
        self.assertEqual(result.normal_offset.shape, (0,))
        self.assertEqual(result.status.shape, (0,))
        self.assertEqual(result.feature.shape, (0,))
        self.assertEqual(result.version, snapshot.version)

    def test_empty_query_batch_on_empty_surface(self):
        snapshot = make_snapshot(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        result = map_points(snapshot, make_queries([], np.zeros((0, 3))))
        # Q = 0 intentionally carries no surface diagnosis.
        self.assertEqual(result.status.shape, (0,))
        self.assertEqual(int(result.status.dtype.itemsize), 1)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        self.queries = make_queries([0], [FACE_QUERY])

    def test_api_object_types(self):
        with self.assertRaises(TypeError):
            map_points(np.zeros((4, 3)), self.queries)
        with self.assertRaises(TypeError):
            map_points(self.snapshot, np.zeros((1, 3)))
        with self.assertRaises(TypeError):
            map_points(self.snapshot, self.queries, {"area_rtol": 1e-12})

    def test_snapshot_shape_errors(self):
        n, t = 4, 2
        with self.assertRaises(ValueError):
            ClothSnapshot(positions=np.zeros((4, 2)), triangles=np.zeros((2, 3), np.int64),
                          **extra_fields(n, t))
        with self.assertRaises(ValueError):
            ClothSnapshot(positions=np.zeros((4, 3)), triangles=np.zeros((2, 4), np.int64),
                          **extra_fields(n, t))
        with self.assertRaises(ValueError):
            ClothSnapshot(positions=np.zeros((4, 3)), triangles=np.zeros((2, 3), np.int64),
                          **dict(extra_fields(n, t), vertex_ids=np.arange(3)))

    def test_query_shape_errors(self):
        with self.assertRaises(ValueError):
            make_queries([0], np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            QueryPoints(query_ids=np.array([0, 1], dtype=np.int64), positions=np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            QueryPoints(query_ids=np.array([], dtype=np.int64), positions=np.zeros((0,)))

    def test_integer_identity_errors(self):
        with self.assertRaises(ValueError):
            QueryPoints(query_ids=np.array([1.5]), positions=np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=np.array([0.5, 1.5]))
        with self.assertRaises(ValueError):
            QueryPoints(query_ids=np.array([True]), positions=np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            QueryPoints(query_ids=np.array([2 ** 63], dtype=np.uint64), positions=np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            make_queries([3, 3], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=[0, 0])

    def test_index_errors(self):
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, [(-1, 0, 1)])
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, [(0, 1, 4)])

    def test_non_finite_inputs(self):
        with self.assertRaises(ValueError):
            make_snapshot([(np.nan, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], [(0, 1, 2)])
        with self.assertRaises(ValueError):
            make_snapshot([(np.inf, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)], [(0, 1, 2)])
        with self.assertRaises(ValueError):
            make_queries([0], [(np.nan, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            make_queries([0], [(0.0, -np.inf, 0.0)])
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, SQUARE_TRIANGLES, velocity=np.full((4, 3), np.nan))
        with self.assertRaises(ValueError):
            make_snapshot(SQUARE, SQUARE_TRIANGLES, inverse_mass=np.array([1.0, 1.0, -1.0, 1.0]))

    def test_config_errors(self):
        for name, value in (
            ("area_rtol", 0.0),
            ("area_rtol", -1e-12),
            ("tie_atol", -1e-10),
            ("tie_atol", np.nan),
            ("tie_rtol", np.inf),
            ("max_distance", -0.5),
            ("max_distance", np.inf),
            ("max_distance", np.nan),
        ):
            with self.assertRaises(ValueError, msg=f"{name}={value}"):
                MappingConfig(**{name: value})

    def test_frozen_config_is_not_mutated(self):
        config = MappingConfig(tie_atol=0.25)
        map_points(self.snapshot, self.queries, config)
        self.assertEqual(float(config.tie_atol), 0.25)

    def test_snapshot_overflow_reports_triangle_id(self):
        positions = [(1e100, 0.0, 0.0), (1e100, 1e100, 0.0), (1e100, 0.0, 1e100)]
        snapshot = make_snapshot(positions, [(0, 1, 2)], triangle_ids=[11])
        with self.assertRaises(ValueError) as raised:
            map_points(snapshot, make_queries([0], [(0.0, 0.0, 0.0)]))
        self.assertIn("triangle_id=11", str(raised.exception))

    def test_query_overflow_reports_query_and_triangle_id(self):
        positions = [(-1e308, 0.0, 0.0), (-1e308, 1.0, 0.0), (-1e308, 0.0, 1.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2)], triangle_ids=[0])
        queries = make_queries([7], [(1e308, 0.0, 0.0)])
        with self.assertRaises(ValueError) as raised:
            map_points(snapshot, queries)
        message = str(raised.exception)
        self.assertIn("query_id=7", message)
        self.assertIn("triangle_id=0", message)

    def test_candidate_overflow_with_finite_siblings_is_reported(self):
        # Review counter-example: a long query offset overflows the squared
        # in-plane candidate metric while the edge candidates stay finite. The
        # overflowing candidate must still be diagnosed, so a finite sibling can
        # not hide it.
        positions = [(0.0, 0.0, 0.0), (1e70, 0.0, 0.0), (0.0, 1e70, 0.0)]
        snapshot = make_snapshot(positions, [(0, 1, 2)], triangle_ids=[3])
        queries = make_queries([5], [(0.5e70, 0.5e70, 1e150)])
        with self.assertRaises(ValueError) as raised:
            map_points(snapshot, queries)
        message = str(raised.exception)
        self.assertIn("face candidate metric", message)
        self.assertIn("query_id=5", message)
        self.assertIn("triangle_id=3", message)

    def test_tie_window_overflow_raises(self):
        # Review counter-example: at d_min = 2 the window
        # 2 + 1e308 + 1e308 * 2 overflows, so it must be diagnosed instead of
        # silently swallowing every candidate.
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=[4, 9])
        queries = make_queries([7], [(0.25, 0.25, 2.0)])
        with self.assertRaises(ValueError) as raised:
            map_points(snapshot, queries,
                       MappingConfig(tie_atol=1e308, tie_rtol=1e308))
        message = str(raised.exception)
        self.assertIn("tie window", message)
        self.assertIn("query_id=7", message)
        self.assertIn("triangle_id=4", message)
        with self.assertRaises(ValueError):
            map_points(snapshot, queries, MappingConfig(tie_rtol=1e308))

    def test_large_finite_tolerances_without_overflow_stay_allowed(self):
        # The same huge configuration is legal when the derived quantities stay
        # finite: at d_min = 0.5 the window is 1.5e308, and a moderate finite
        # window keeps the smallest identity in the (broad) tie set.
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0], [(0.25, 0.25, 0.5)])
        result = map_points(snapshot, queries,
                            MappingConfig(tie_atol=1e308, tie_rtol=1e308))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        self.assertEqual(int(result.triangle_ids[0]), 0)
        self.assertAlmostEqual(float(result.distance[0]), 0.5, places=12)
        result = map_points(snapshot, queries, MappingConfig(tie_atol=1e30, tie_rtol=1e30))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))
        result = map_points(snapshot, queries, MappingConfig(tie_atol=1e308))
        self.assertEqual(int(result.status[0]), int(MappingStatus.HIT))


class TestResultContract(unittest.TestCase):
    def test_version_and_dtypes(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0, 1], [FACE_QUERY, (0.0, 0.0, 0.25)])
        result = map_points(snapshot, queries)
        self.assertEqual(result.version, snapshot.version)
        self.assertEqual(result.version, (3, 7, 5))
        self.assertEqual(result.epoch, 3)
        self.assertEqual(result.step_index, 7)
        self.assertEqual(result.topology_version, 5)
        self.assertEqual(result.query_ids.dtype, np.int64)
        self.assertEqual(result.triangle_ids.dtype, np.int64)
        self.assertEqual(result.vertex_ids.dtype, np.int64)
        self.assertEqual(result.barycentric.dtype, np.float64)
        self.assertEqual(result.closest_point.dtype, np.float64)
        self.assertEqual(result.distance.dtype, np.float64)
        self.assertEqual(result.normal.dtype, np.float64)
        self.assertEqual(result.normal_offset.dtype, np.float64)
        self.assertEqual(result.status.dtype, np.int8)
        self.assertEqual(result.feature.dtype, np.int8)

    def test_arrays_are_read_only_and_unaliased(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0, 1], [FACE_QUERY, (0.0, 0.0, 0.25)])
        result = map_points(snapshot, queries)
        sources = [getattr(snapshot, field.name) for field in fields(snapshot)] + [
            queries.query_ids, queries.positions]
        for field in fields(result):
            value = getattr(result, field.name)
            if not isinstance(value, np.ndarray):
                continue
            self.assertFalse(value.flags.writeable, field.name)
            self.assertTrue(value.flags.owndata, field.name)
            for source in sources:
                self.assertFalse(np.shares_memory(value, source), field.name)
            if value.size:
                with self.assertRaises(ValueError, msg=field.name):
                    value.flat[0] = value.flat[0]

    def test_mapping_never_writes_snapshot_or_queries(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES)
        queries = make_queries([0, 1], [FACE_QUERY, (2.0, -1.0, 0.5)])
        arrays = [getattr(snapshot, field.name) for field in fields(snapshot)
                  if isinstance(getattr(snapshot, field.name), np.ndarray)]
        arrays += [queries.query_ids, queries.positions]
        before = [array.tobytes() for array in arrays]
        map_points(snapshot, queries)
        after = [array.tobytes() for array in arrays]
        self.assertEqual(before, after)

    def test_repeated_calls_are_bitwise_deterministic(self):
        snapshot = make_snapshot(SQUARE, SQUARE_TRIANGLES, triangle_ids=[5, 2])
        queries = make_queries([0, -1, 2, 3], [FACE_QUERY, (0.5, 0.5, 0.3),
                                              (0.0, 0.0, 0.5), (2.0, -1.0, 0.25)])
        first = map_points(snapshot, queries)
        second = map_points(snapshot, queries)
        for name in ("query_ids", "triangle_ids", "vertex_ids", "status", "feature"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        for name in ("barycentric", "closest_point", "distance", "normal", "normal_offset"):
            self.assertTrue(np.array_equal(getattr(first, name), getattr(second, name),
                                           equal_nan=True), name)

    def test_memory_scale_is_linear_in_triangles(self):
        snapshot, queries = grid_case(rows=23, columns=23, query_count=256)
        triangles = int(snapshot.triangles.shape[0])
        self.assertEqual(triangles, 2 * 22 * 22)
        forbidden = (queries.positions.shape[0] * triangles * 3
                     * np.dtype(np.float64).itemsize)
        tracemalloc.start()
        result = map_points(snapshot, queries)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertEqual(result.status.shape[0], 256)
        self.assertLess(peak, forbidden // 4,
                        f"peak={peak} B, forbidden Q*T*3={forbidden} B")
        print(f"[memory] T={triangles} Q=256 peak={peak / 1024.0:.1f} KiB, "
              f"forbidden Q*T*3={forbidden / 1048576.0:.2f} MiB")
        for index in range(0, 256, 37):
            assert_row_consistent(self, snapshot, queries, result, index)


class TestExactOracle(unittest.TestCase):
    def test_selection_and_values_match_exact_arithmetic(self):
        # Independent Fraction oracle over dyadic cases, run with strict
        # tie_atol = tie_rtol = 0 so the oracle's tie rule and the frozen one
        # are the same rule. This is what pins "selection uses the reported
        # point's own distance standard": the identity, the distance and the
        # weights all have to agree with exact arithmetic, including the
        # shared-edge case where two constructions must tie bitwise.
        square = list(SQUARE)
        sliver = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1e-10, 0.0)]
        cases = (
            (square, SQUARE_TRIANGLES, [5, 2], (0.25, 0.5, 0.25)),
            (square, SQUARE_TRIANGLES, [5, 2], (0.5, 0.5, 0.3)),
            (square, SQUARE_TRIANGLES, [5, 2], (0.0, 0.0, 0.25)),
            (square, SQUARE_TRIANGLES, [5, 2], (-0.5, 0.5, 0.2)),
            (square, SQUARE_TRIANGLES, [5, 2], (1.6, 0.4, 0.75)),
            (sliver, [(0, 1, 2)], [7], (0.75, 2.5e-11, 0.5)),
        )
        config = MappingConfig(tie_atol=0.0, tie_rtol=0.0)
        for positions, triangles, ids, query in cases:
            snapshot = make_snapshot(positions, triangles, triangle_ids=ids)
            queries = make_queries([0], [query])
            result = map_points(snapshot, queries, config)
            exact = exact_mapping(positions, triangles, ids, query)
            self.assertIsNotNone(exact, query)
            squared, identity, point, weights, feature = exact
            self.assertEqual(int(result.triangle_ids[0]), identity, query)
            self.assertEqual(int(result.feature[0]), int(feature), query)
            self.assertAlmostEqual(float(result.distance[0]) ** 2, float(squared),
                                   places=12, msg=str(query))
            np.testing.assert_allclose(result.closest_point[0],
                                       [float(value) for value in point], atol=1e-15,
                                       err_msg=str(query))
            np.testing.assert_allclose(result.barycentric[0],
                                       [float(value) for value in weights], atol=1e-12,
                                       err_msg=str(query))


if __name__ == "__main__":
    unittest.main(verbosity=2)
