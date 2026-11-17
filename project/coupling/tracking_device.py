"""CPU f64 broad phase and surface velocity for the optional P20 tracker.

The local result is computed by P20 itself. A certificate only eliminates
omitted triangles whose current world AABB is strictly beyond the entire tie
window. Unsupported numeric ranges and inconclusive bounds go to full P20.
"""

import numpy as np
import taichi as ti

from ..cloth.taichi_runtime import initialize_taichi


@ti.kernel
def _certify(positions: ti.types.ndarray(), triangles: ti.types.ndarray(),
             queries: ti.types.ndarray(), limits: ti.types.ndarray(),
             offsets: ti.types.ndarray(), local: ti.types.ndarray(),
             boxes: ti.types.ndarray(), eligible: ti.types.ndarray(),
             result: ti.types.ndarray()):
    # The two outer loops are ordered kernel stages. Geometry is derived once
    # per current snapshot batch, including when topology stayed the same.
    for face in range(triangles.shape[0]):
        a = ti.Vector([positions[triangles[face, 0], k] for k in ti.static(range(3))])
        b = ti.Vector([positions[triangles[face, 1], k] for k in ti.static(range(3))])
        c = ti.Vector([positions[triangles[face, 2], k] for k in ti.static(range(3))])
        for k in ti.static(range(3)):
            boxes[face, 0, k] = ti.min(a[k], b[k], c[k])
            boxes[face, 1, k] = ti.max(a[k], b[k], c[k])
        ab, ac = b - a, c - a
        ab2, ac2 = ab.norm_sqr(), ac.norm_sqr()
        # P20 diagnoses even losing triangles' derived quantities. Admit only
        # a well-conditioned range on which those intermediates stay finite.
        eligible[face] = (ti.min(ab2, ac2, (b - c).norm_sqr()) >= 1e-12
                          and ab.cross(ac).norm_sqr() >= 0.04 * ab2 * ac2)

    for q in range(queries.shape[0]):
        safe = 0
        cursor = offsets[q]
        if limits[q] >= 0 and cursor < offsets[q + 1]:
            safe = 1
            for face in range(triangles.shape[0]):
                member = 0
                if cursor < offsets[q + 1]:
                    if face == local[cursor]:
                        member = 1
                        cursor += 1
                if not member:
                    gap = 0.0
                    for k in ti.static(range(3)):
                        gap = ti.max(gap, boxes[face, 0, k] - queries[q, k],
                                     queries[q, k] - boxes[face, 1, k])
                    if not eligible[face] or gap <= limits[q]:
                        safe = 0
                        break
        result[q] = safe


def certify_local_batch(snapshot, query_positions, local_rows, local_distances, config):
    """Return one certificate flag per local P20 result, in input query order.

    Rows are current triangle *slots* in strictly increasing order. The
    selected local distance is an upper bound on the local minimum, including
    when P20's smallest-stable-ID tie winner is farther than that minimum.
    NaN denotes no usable local result and cannot certify a row.
    """
    initialize_taichi("cpu", "f64", log=False)
    points = np.ascontiguousarray(query_positions, dtype=np.float64)
    distances = np.asarray(local_distances, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("query_positions must be finite (Q,3)")
    q = len(points)
    t = len(snapshot.triangles)
    if distances.shape != (q,) or len(local_rows) != q:
        raise ValueError("local distances and face rows must align with the query batch")
    result = np.zeros(q, dtype=np.int32)
    if q == 0 or t == 0:
        return result.astype(bool)
    chunks = []
    offsets = [0]
    for raw in local_rows:
        rows = np.asarray(raw)
        if (rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu")
                or np.any(rows < 0) or np.any(rows >= t) or np.any(rows[1:] <= rows[:-1])):
            raise ValueError("local face rows must be sorted unique active triangle slots")
        chunks.append(rows.astype(np.int64, copy=False))
        offsets.append(offsets[-1] + rows.size)
    coordinate_scale = np.max(np.abs(snapshot.positions), initial=0.0)
    if not offsets[-1] or coordinate_scale > 1e8 or config.area_rtol > 1.0:
        return result.astype(bool)
    with np.errstate(over="ignore", invalid="ignore"):
        windows = distances + config.tie_atol + config.tie_rtol * distances
        point_scale = np.max(np.abs(points), axis=1)
        ok = ((distances >= 0) & np.isfinite(windows) & (windows <= 1e9)
              & (point_scale <= 1e8))
        # P20 evaluates floats at a three-term barycentric reconstruction.
        # Nonnegative selected weights are (1-v-w,v,w) or (1-t,t,0) up
        # to permutation. Weight-sum and point reconstruction, subtraction,
        # Euclidean norm, and AABB subtraction together use fewer than 64
        # unit-roundoff-scaled absolute errors at this common scale. A floor
        # of 1 also covers small/subnormal differences. Strict separation
        # leaves the original rounded distance and inclusive tie rule intact.
        scale = np.maximum(np.maximum(max(1.0, coordinate_scale), point_scale), windows)
        limits = np.where(ok, windows + 64 * np.finfo(np.float64).eps * scale, -1.0)
    if np.any(ok):
        # Taichi 1.7.4 passes C-contiguous buffers directly but attempts to
        # copy F-only buffers back, even for inputs. Snapshot arrays are
        # read-only and may use either layout. Never change their flags.
        _certify(np.ascontiguousarray(snapshot.positions), np.ascontiguousarray(snapshot.triangles), points, limits,
                 np.asarray(offsets, dtype=np.int64), np.concatenate(chunks),
                 np.empty((t, 2, 3), dtype=np.float64), np.empty(t, dtype=np.int32), result)
    return result.astype(bool)


@ti.kernel
def _interpolate(velocities: ti.types.ndarray(), triangles: ti.types.ndarray(),
                 rows: ti.types.ndarray(), weights: ti.types.ndarray(),
                 result: ti.types.ndarray()):
    for q in range(rows.shape[0]):
        face = rows[q]
        if face >= 0:
            for k in ti.static(range(3)):
                value = 0.0
                for j in ti.static(range(3)):
                    value += weights[q, j] * velocities[triangles[face, j], k]
                result[q, k] = value


def interpolate_surface_velocity(snapshot, triangle_rows, barycentric):
    """Interpolate current HIT velocities; row -1 retains the NaN sentinel."""
    initialize_taichi("cpu", "f64", log=False)
    rows = np.asarray(triangle_rows)
    weights = np.ascontiguousarray(barycentric, dtype=np.float64)
    if (rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu")
            or np.any(rows < -1) or np.any(rows >= len(snapshot.triangles))
            or weights.shape != (rows.size, 3)):
        raise ValueError("triangle slots and barycentric rows must align")
    hit = rows >= 0
    if not np.isfinite(weights[hit]).all():
        raise ValueError("mapped barycentric weights must be finite")
    result = np.full((rows.size, 3), np.nan)
    if np.any(hit):
        _interpolate(np.ascontiguousarray(snapshot.velocity), np.ascontiguousarray(snapshot.triangles),
                     np.ascontiguousarray(rows, dtype=np.int64),
                     weights, result)
        if not np.isfinite(result[hit]).all():
            raise ValueError("nonfinite interpolated surface velocity")
    return result
