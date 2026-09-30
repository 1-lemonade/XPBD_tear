"""NumPy float64 closest-point queries on immutable cloth snapshots.

Queries are independent of cloth vertices. The mapper processes one query at
a time across all triangles, using O(Q*T) work, O(T) temporary storage and
O(Q) result storage. Degenerate triangles are skipped; invalid numerical
results raise errors. Thin triangles use compensated float64 arithmetic.

The global minimum defines the configured tie window; the smallest triangle
identity inside that window wins. The selected candidate's actual distance
determines threshold acceptance. Normals follow triangle winding, and local
normal offsets are not signed distance fields. Inputs and outputs retain
their snapshot version and use model length units.

Run tools.measure_coupling to measure read-only queries and paired trajectories.
"""

from __future__ import annotations

import numpy as np

from .contracts import (
    ClothSnapshot,
    MappingConfig,
    MappingFeature,
    MappingResult,
    MappingStatus,
    QueryPoints,
)


def _dot_cross(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Elementwise ``dot(cross(u, v), w)`` without materializing the cross."""
    return ((u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]) * w[:, 0]
            + (u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]) * w[:, 1]
            + (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) * w[:, 2])


# --- error-free transformations -------------------------------------------------
#
# ``cross(ab, ac)`` subtracts products of nearly parallel edge components.  For a
# triangle that is large but thin, the subtraction cancels most of both
# magnitudes, so each cross component loses absolute accuracy even though its
# operands are accurate: with edges of length 2000 and a height of 1e-6 the
# components are ~2e-4 and the naive subtraction leaves ~1e-11 of absolute
# error, i.e. ~1e-7 relative.  ``n2 = |n|^2`` inherits that, and because the
# signed-area weights are ``numerator / n2`` the weight -- and through it the
# reconstructed closest point -- inherits it verbatim.  The same cancellation
# affects the weight numerators, where products of order |edge|^2 cancel down to
# the area scale.
#
# Dekker's ``two_product`` and Knuth's ``two_sum`` are exact: for finite inputs
# ``head + tail`` equals the exact product/sum, with no rounding at all.  They
# are used only for the triangle-level tables and the two weight numerators, so
# the work stays O(T) per call and O(Q*T) overall, all in float64, and every
# reported value is still an ordinary float64 (the extra precision only decides
# which float64 is correct).

_SPLITTER = 134217729.0  # 2**27 + 1, exact in float64


def _two_product(x: np.ndarray, y: np.ndarray):
    """Exact elementwise product as ``(head, tail)`` with ``head + tail == x * y``."""
    with np.errstate(over="ignore", invalid="ignore"):
        product = x * y
        x_split = _SPLITTER * x
        x_high = x_split - (x_split - x)
        x_low = x - x_high
        y_split = _SPLITTER * y
        y_high = y_split - (y_split - y)
        y_low = y - y_high
        error = ((x_high * y_high - product) + x_high * y_low + x_low * y_high) + x_low * y_low
    return product, error


def _two_sum(x: np.ndarray, y: np.ndarray):
    """Exact elementwise sum as ``(head, tail)`` with ``head + tail == x + y``."""
    with np.errstate(over="ignore", invalid="ignore"):
        total = x + y
        y_virtual = total - x
        x_virtual = total - y_virtual
        error = (x - x_virtual) + (y - y_virtual)
    return total, error


def _cross_error_free(ab: np.ndarray, ac: np.ndarray):
    """Exact ``cross(ab, ac)`` per triangle as three ``(head, tail)`` pairs."""
    first = _two_product(ab[:, 1], ac[:, 2])
    other = _two_product(ab[:, 2], ac[:, 1])
    x_head, x_error = _two_sum(first[0], -other[0])
    components = [(x_head, x_error + (first[1] - other[1]))]
    first = _two_product(ab[:, 2], ac[:, 0])
    other = _two_product(ab[:, 0], ac[:, 2])
    y_head, y_error = _two_sum(first[0], -other[0])
    components.append((y_head, y_error + (first[1] - other[1])))
    first = _two_product(ab[:, 0], ac[:, 1])
    other = _two_product(ab[:, 1], ac[:, 0])
    z_head, z_error = _two_sum(first[0], -other[0])
    components.append((z_head, z_error + (first[1] - other[1])))
    return components


def _square_sum(components) -> np.ndarray:
    """Compensated ``sum(component ** 2)`` over ``(head, tail)`` pairs."""
    total = np.zeros_like(components[0][0])
    error = np.zeros_like(total)
    for head, tail in components:
        product_head, product_tail = _two_product(head, head)
        total, delta = _two_sum(total, product_head)
        error = error + (delta + product_tail + (2.0 * head) * tail)
    return total + error


def _dot_cross_error_free(u: np.ndarray, v: np.ndarray, w_components) -> np.ndarray:
    """Compensated ``dot(cross(u, v), w)`` per triangle."""
    total = np.zeros(u.shape[0], dtype=np.float64)
    error = np.zeros_like(total)
    for index, (i, j, k) in enumerate(((0, 1, 2), (1, 2, 0), (2, 0, 1))):
        first = _two_product(u[:, j], v[:, k])
        other = _two_product(u[:, k], v[:, j])
        cross_head, cross_error = _two_sum(first[0], -other[0])
        cross_error = cross_error + (first[1] - other[1])
        head, tail = w_components[index]
        product_head, product_tail = _two_product(cross_head, head)
        total, delta = _two_sum(total, product_head)
        error = error + (delta + product_tail + cross_error * head + cross_head * tail)
    return total + error


def _first_bad_row(value: np.ndarray) -> int:
    """Row of the first non-finite entry of a ``(T,)`` or ``(T, 3)`` array."""
    return int(np.argwhere(~np.isfinite(value))[0][0])


def _finite_rows(value: np.ndarray) -> np.ndarray:
    """Per-row finiteness of a ``(T,)`` or ``(T, 3)`` array."""
    finite = np.isfinite(value)
    return finite.all(axis=-1) if finite.ndim > 1 else finite


class _Surface:
    """Per-call triangle tables; rebuilt for every snapshot, never cached."""

    __slots__ = (
        "count", "a", "b", "c", "ab", "ac", "bc", "n", "n2", "eab2", "eac2",
        "ebc2", "valid", "usable", "missing_status", "triangle_ids", "vertex_ids",
        "n_components", "sliver",
    )

    def __init__(self, snapshot: ClothSnapshot, config: MappingConfig):
        triangles = snapshot.triangles
        positions = snapshot.positions
        self.count = int(triangles.shape[0])
        self.triangle_ids = snapshot.triangle_ids
        self.vertex_ids = snapshot.vertex_ids[triangles]
        if self.count == 0:
            empty3 = np.zeros((0, 3), dtype=np.float64)
            empty1 = np.zeros(0, dtype=np.float64)
            self.a = self.b = self.c = empty3
            self.ab = self.ac = self.bc = self.n = empty3
            self.n2 = self.eab2 = self.eac2 = self.ebc2 = empty1
            self.valid = np.zeros(0, dtype=bool)
            self.usable = False
            self.missing_status = int(MappingStatus.EMPTY_SURFACE)
            self.n_components = ()
            self.sliver = np.zeros(0, dtype=bool)
            return
        self.a = positions[triangles[:, 0]]
        self.b = positions[triangles[:, 1]]
        self.c = positions[triangles[:, 2]]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self.ab = self.b - self.a
            self.ac = self.c - self.a
            self.bc = self.c - self.b
            self.n_components = _cross_error_free(self.ab, self.ac)
            # ``head + tail`` is the correctly rounded cross product: the head
            # alone drops the low-order part, which for a thin triangle is the
            # same order as the component itself.
            self.n = np.stack([component[0] + component[1] for component in self.n_components],
                              axis=1)
            n2_exact = _square_sum(self.n_components)
            self.eab2 = (self.ab * self.ab).sum(axis=1)
            self.eac2 = (self.ac * self.ac).sum(axis=1)
            self.ebc2 = (self.bc * self.bc).sum(axis=1)
            longest = np.maximum(np.maximum(self.eab2, self.eac2), self.ebc2)
            threshold = (float(config.area_rtol) * longest) ** 2
            # The error-free transformations cost several extra flops per
            # triangle and per query.  They are only needed where the naive
            # arithmetic actually loses digits: the cross-product components are
            # computed by a cancelling subtraction, and the larger they are
            # relative to the edge lengths, the smaller the relative rounding
            # error.  ``|cross| <= |ab| * |ac|``, so this ratio is at most 1 and
            # exceeds 0.1 for any triangle whose height is a tenth of its
            # longest edge.  Triangles below it -- the slivers -- keep the
            # compensated values; the rest keep the cheap float64 ones, which
            # already agree with the compensated result to machine precision.
            with np.errstate(invalid="ignore", divide="ignore"):
                sliver = np.sqrt(np.maximum(n2_exact, 0.0)) <= 0.1 * np.sqrt(
                    np.maximum(self.eab2 * self.eac2, 0.0))
            self.sliver = sliver
            self.n2 = np.where(sliver, n2_exact, (self.n * self.n).sum(axis=1))
        # The area threshold is a derived quantity too: an area_rtol large
        # enough to overflow it must be diagnosed, not silently filter every
        # triangle into DEGENERATE_ONLY.
        self._check_derived(snapshot, (
            ("ab", self.ab), ("ac", self.ac), ("bc", self.bc),
            ("cross product", np.stack([component[0] + component[1]
                                        for component in self.n_components]
                                       + [component[1] for component in self.n_components],
                                       axis=1)),
            ("squared normal", self.n2), ("squared edge a-b", self.eab2),
            ("squared edge a-c", self.eac2), ("squared edge b-c", self.ebc2),
            ("area threshold", threshold),
        ))
        # Frozen predicate: L2 == 0 is already covered because then n2 == 0 and
        # the threshold is 0.
        self.valid = self.n2 > threshold
        self.usable = bool(self.valid.any())
        self.missing_status = int(MappingStatus.DEGENERATE_ONLY)

    def _check_derived(self, snapshot: ClothSnapshot, values) -> None:
        for name, value in values:
            if value.size and not np.isfinite(value).all():
                row = _first_bad_row(value)
                raise ValueError(
                    f"non-finite derived {name} for triangle_id="
                    f"{int(snapshot.triangle_ids[row])}: surface coordinates, edge lengths, "
                    "area_rtol or their products overflow float64"
                )

    def _require_finite(self, values, query_id: int) -> None:
        """Diagnose non-finite derived quantities of every valid triangle."""
        for name, value in values:
            bad = self.valid & ~_finite_rows(value)
            if bad.any():
                row = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"non-finite derived {name} for query_id={query_id}, triangle_id="
                    f"{int(self.triangle_ids[row])}: query and surface coordinates "
                    "overflow float64"
                )

    def map_query(self, point: np.ndarray, query_id: int, config: MappingConfig):
        """Map one query; every triangle shares one distance construction."""
        a, b, c = self.a, self.b, self.c
        ab, ac, bc = self.ab, self.ac, self.bc
        n, n2, valid = self.n, self.n2, self.valid
        n_components = self.n_components
        sliver = self.sliver
        count = self.count
        tie_atol = float(config.tie_atol)
        tie_rtol = float(config.tie_rtol)
        max_distance = None if config.max_distance is None else float(config.max_distance)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ap = point - a
            q_ab = ab[:, 0] * ap[:, 0] + ab[:, 1] * ap[:, 1] + ab[:, 2] * ap[:, 2]
            q_ac = ac[:, 0] * ap[:, 0] + ac[:, 1] * ap[:, 1] + ac[:, 2] * ap[:, 2]
            bp = ap - ab
            q_bc = bc[:, 0] * bp[:, 0] + bc[:, 1] * bp[:, 1] + bc[:, 2] * bp[:, 2]
            dn = n[:, 0] * ap[:, 0] + n[:, 1] * ap[:, 1] + n[:, 2] * ap[:, 2]
            # The weight numerators are the same triple product as ``_dot_cross``
            # but evaluated with error-free transformations, because their terms
            # are of order |edge|^2 while the result is of order |edge| * height.
            v_num = _dot_cross(ap, ac, n)
            w_num = _dot_cross(ab, ap, n)
            if sliver.any():
                v_num = np.where(sliver, _dot_cross_error_free(ap, ac, n_components), v_num)
                w_num = np.where(sliver, _dot_cross_error_free(ab, ap, n_components), w_num)
            v = v_num / n2
            w = w_num / n2
            u = 1.0 - v - w
            t = np.clip(q_ab / self.eab2, 0.0, 1.0)
            s = np.clip(q_ac / self.eac2, 0.0, 1.0)
            r = np.clip(q_bc / self.ebc2, 0.0, 1.0)
            # Raw candidate metrics. Every candidate of every valid triangle is
            # diagnosed, so a finite sibling can not hide an overflow.
            d2_face = dn * dn / n2
            diff = ap - t[:, None] * ab
            d2_ab = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
            diff = ap - s[:, None] * ac
            d2_ac = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
            diff = bp - r[:, None] * bc
            d2_bc = diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2
            self._require_finite((
                ("query-to-vertex offset", ap),
                ("dot(ab, query - a)", q_ab),
                ("dot(ac, query - a)", q_ac),
                ("dot(bc, query - b)", q_bc),
                ("dot(normal, query - a)", dn),
                ("face barycentric weight of b", v),
                ("face barycentric weight of c", w),
                ("face barycentric weight of a", u),
                ("edge a-b parameter", t),
                ("edge a-c parameter", s),
                ("edge b-c parameter", r),
                ("face candidate metric", d2_face),
                ("edge a-b candidate metric", d2_ab),
                ("edge a-c candidate metric", d2_ac),
                ("edge b-c candidate metric", d2_bc),
            ), query_id)

            # Exact closed-region classification from the signed-area weights.
            zero_count = ((u == 0.0).astype(np.int8) + (v == 0.0).astype(np.int8)
                          + (w == 0.0).astype(np.int8))
            inside = valid & (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
            face_feature = np.where(zero_count >= 2, int(MappingFeature.VERTEX),
                                    np.where(zero_count == 1, int(MappingFeature.EDGE),
                                             int(MappingFeature.FACE))).astype(np.int8)

            # Clamped boundary candidates in a fixed order, so exact ties
            # resolve identically; edges that tie share their endpoint.
            zeros = np.zeros(count, dtype=np.float64)
            edge_u, edge_v, edge_w = 1.0 - t, t, zeros
            edge_feature = np.where((t <= 0.0) | (t >= 1.0), int(MappingFeature.VERTEX),
                                    int(MappingFeature.EDGE)).astype(np.int8)
            edge_metric = np.where(valid, d2_ab, np.inf)
            take = d2_ac < edge_metric
            edge_metric = np.where(take, d2_ac, edge_metric)
            edge_u = np.where(take, 1.0 - s, edge_u)
            edge_v = np.where(take, zeros, edge_v)
            edge_w = np.where(take, s, edge_w)
            edge_feature = np.where(
                take, np.where((s <= 0.0) | (s >= 1.0), int(MappingFeature.VERTEX),
                               int(MappingFeature.EDGE)).astype(np.int8), edge_feature)
            take = d2_bc < edge_metric
            edge_u = np.where(take, zeros, edge_u)
            edge_v = np.where(take, 1.0 - r, edge_v)
            edge_w = np.where(take, r, edge_w)
            edge_feature = np.where(
                take, np.where((r <= 0.0) | (r >= 1.0), int(MappingFeature.VERTEX),
                               int(MappingFeature.EDGE)).astype(np.int8), edge_feature)

            # A region-valid in-plane projection is the unique closest point of
            # its triangle and wins outright; boundary candidates are only used
            # when a signed-area weight is negative.
            weight_u = np.where(inside, u, edge_u)
            weight_v = np.where(inside, v, edge_v)
            weight_w = np.where(inside, w, edge_w)
            feature = np.where(inside, face_feature, edge_feature)
            closest_all = weight_u[:, None] * a + weight_v[:, None] * b + weight_w[:, None] * c
            offset = point - closest_all
            distance_all = np.sqrt(offset[:, 0] ** 2 + offset[:, 1] ** 2 + offset[:, 2] ** 2)
            distance_all = np.where(valid, distance_all, np.inf)
            self._require_finite((
                ("candidate closest point", closest_all),
                ("candidate distance", distance_all),
            ), query_id)

            nearest_row = int(np.argmin(distance_all))
            d_min = float(distance_all[nearest_row])
            if not np.isfinite(d_min):
                raise ValueError(
                    f"non-finite candidate distance for query_id={query_id}, triangle_id="
                    f"{int(self.triangle_ids[nearest_row])}: surface coordinates overflow float64"
                )
            window = d_min + tie_atol + tie_rtol * d_min
            if not np.isfinite(window):
                raise ValueError(
                    f"non-finite tie window for query_id={query_id}, triangle_id="
                    f"{int(self.triangle_ids[nearest_row])}: d_min={d_min!r}, "
                    f"tie_atol={tie_atol!r}, tie_rtol={tie_rtol!r} overflow float64"
                )
            tie = np.flatnonzero(distance_all <= window)
            row = int(tie[np.argmin(self.triangle_ids[tie])])

            # Reported values come from the same arrays that were compared.
            weights = (float(weight_u[row]), float(weight_v[row]), float(weight_w[row]))
            closest = closest_all[row].copy()
            distance = float(distance_all[row])
            normal = n[row] / np.sqrt(n2[row])
            normal_offset = float(offset[row][0] * normal[0] + offset[row][1] * normal[1]
                                  + offset[row][2] * normal[2])
        if max_distance is not None and distance > max_distance:
            status = int(MappingStatus.OUTSIDE_THRESHOLD)
        else:
            status = int(MappingStatus.HIT)
        return (
            int(self.triangle_ids[row]),
            self.vertex_ids[row],
            np.asarray(weights, dtype=np.float64),
            closest,
            distance,
            normal,
            normal_offset,
            status,
            int(feature[row]),
        )


def map_points(
    snapshot: ClothSnapshot,
    queries: QueryPoints,
    config: MappingConfig | None = None,
) -> MappingResult:
    """Map every query to the closest point of ``snapshot`` with float64 NumPy.

    Results keep the input query order and snapshot version. Rows without a
    candidate use ``-1`` identities, ``NONE`` feature and NaN floating fields.
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("map_points requires a ClothSnapshot")
    if not isinstance(queries, QueryPoints):
        raise TypeError("map_points requires QueryPoints")
    if config is None:
        config = MappingConfig()
    elif not isinstance(config, MappingConfig):
        raise TypeError("config must be a MappingConfig or None")

    count = int(queries.positions.shape[0])
    triangle_ids = np.full(count, -1, dtype=np.int64)
    vertex_ids = np.full((count, 3), -1, dtype=np.int64)
    barycentric = np.full((count, 3), np.nan, dtype=np.float64)
    closest_point = np.full((count, 3), np.nan, dtype=np.float64)
    distance = np.full(count, np.nan, dtype=np.float64)
    normal = np.full((count, 3), np.nan, dtype=np.float64)
    normal_offset = np.full(count, np.nan, dtype=np.float64)
    feature = np.full(count, int(MappingFeature.NONE), dtype=np.int8)

    if count == 0:
        # No query can observe any surface quantity, so the per-triangle tables
        # are not derived at all.  That keeps an empty batch well defined for a
        # snapshot whose coordinates are legal but whose derived products would
        # overflow: the contract diagnoses overflow only where a value is
        # actually produced.  The input objects and config were validated above.
        status = np.full(0, int(MappingStatus.EMPTY_SURFACE if snapshot.triangles.shape[0] == 0
                                else MappingStatus.DEGENERATE_ONLY), dtype=np.int8)
    else:
        surface = _Surface(snapshot, config)
        status = np.full(count, surface.missing_status, dtype=np.int8)
        if surface.usable:
            for index in range(count):
                (
                    triangle_id,
                    vertex_id_row,
                    weights,
                    point,
                    row_distance,
                    normal_row,
                    row_offset,
                    row_status,
                    row_feature,
                ) = surface.map_query(queries.positions[index], int(queries.query_ids[index]), config)
                triangle_ids[index] = triangle_id
                vertex_ids[index] = vertex_id_row
                barycentric[index] = weights
                closest_point[index] = point
                distance[index] = row_distance
                normal[index] = normal_row
                normal_offset[index] = row_offset
                status[index] = row_status
                feature[index] = row_feature

    return MappingResult(
        query_ids=queries.query_ids,
        epoch=snapshot.epoch,
        step_index=snapshot.step_index,
        topology_version=snapshot.topology_version,
        triangle_ids=triangle_ids,
        vertex_ids=vertex_ids,
        barycentric=barycentric,
        closest_point=closest_point,
        distance=distance,
        normal=normal,
        normal_offset=normal_offset,
        status=status,
        feature=feature,
    )
