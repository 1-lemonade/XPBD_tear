"""Reference hinges and Taichi signed-dihedral geometry.

For reference edge length L and adjacent reference areas A1, A2, w=L²/(A1+A2).
The caller supplies B [N m]; U=0.5*B*w*wrap(theta-theta0)², alpha=1/(B*w).
This defines a discrete coefficient, not a calibrated continuum plate modulus.
"""

from dataclasses import dataclass
import math

import numpy as np
import taichi as ti


@dataclass(frozen=True)
class DihedralConstraint:
    a: int
    b: int
    c: int
    d: int
    rest_angle: float
    reference_weight: float

    @property
    def vertices(self) -> tuple[int, int, int, int]:
        return (self.a, self.b, self.c, self.d)


def reference_hinge(points: np.ndarray, relative_tolerance: float) -> tuple[float, float]:
    """Precompute from immutable reference positions; no production projection."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (4, 3) or not np.isfinite(p).all():
        raise ValueError("dihedral reference requires four finite positions")
    with np.errstate(over="ignore", invalid="ignore"):
        local = p[1:] - p[0]
        scale = float(np.max(np.abs(local)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("degenerate dihedral reference scale")
    e, u, v = local / scale
    n1, n2 = np.cross(e, u), np.cross(v, e)
    length, area1, area2 = np.linalg.norm(e), np.linalg.norm(n1), np.linalg.norm(n2)
    if min(length, area1, area2) <= relative_tolerance:
        raise ValueError("degenerate dihedral reference edge/triangle")
    t, n1, n2 = e / length, n1 / area1, n2 / area2
    angle = math.atan2(float(t @ np.cross(n1, n2)), float(n1 @ n2))
    weight = 2.0 * length**2 / (area1 + area2)
    return angle, weight


def build_dihedral_constraints(mesh, relative_tolerance: float) -> list[DihedralConstraint]:
    reference, _ = mesh.reference_data()
    result = []
    for edge in mesh.edges:
        if not edge.active or edge.broken or len(edge.adjacent_triangles) != 2:
            continue
        a, b = sorted((edge.a, edge.b))
        opposite = [
            int(next(v for v in mesh.triangles[t] if v not in (a, b)))
            for t in edge.adjacent_triangles
        ]
        c, d = sorted(opposite)
        if len({a, b, c, d}) != 4:
            raise ValueError(f"dihedral edge {(a, b)} requires four distinct vertices")
        try:
            angle, weight = reference_hinge(reference[[a, b, c, d]], relative_tolerance)
        except ValueError as error:
            raise ValueError(f"dihedral edge {(a, b)}: {error}") from error
        result.append(DihedralConstraint(a, b, c, d, angle, weight))
    return result


@ti.func
def finite_scalar(x):
    return not (ti.math.isnan(x) or ti.math.isinf(x))


@ti.func
def wrapped_angle(value):
    return ti.atan2(ti.sin(value), ti.cos(value))


@ti.func
def dihedral_geometry(a, b, c, d, relative_tolerance):
    """Angle, d(theta)/d(a,b,c,d), valid. Flat nonzero-area hinges are valid.

    Normalize local coordinates before cross products. Thresholds apply to
    edge length / scale and doubled triangle area / scale², never to theta.
    """
    angle = 0.0
    gradient = ti.Matrix([
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    valid = 1
    scale = 0.0
    e, u, v = b - a, c - a, d - a
    for axis in ti.static(range(3)):
        if not (finite_scalar(a[axis]) and finite_scalar(e[axis])
                and finite_scalar(u[axis]) and finite_scalar(v[axis])):
            valid = 0
        scale = ti.max(scale, ti.abs(e[axis]), ti.abs(u[axis]), ti.abs(v[axis]))
    if scale <= 0.0 or not finite_scalar(scale):
        valid = 0
    if valid:
        e, u, v = e / scale, u / scale, v / scale
        n1, n2 = e.cross(u), v.cross(e)
        l2, a12, a22 = e.dot(e), n1.dot(n1), n2.dot(n2)
        eps2 = relative_tolerance * relative_tolerance
        if l2 <= eps2 or a12 <= eps2 or a22 <= eps2:
            valid = 0
        if valid:
            length = ti.sqrt(l2)
            t, h1, h2 = e / length, n1 / ti.sqrt(a12), n2 / ti.sqrt(a22)
            angle = ti.atan2(t.dot(h1.cross(h2)), h1.dot(h2))
            qc, qd = -length * n1 / a12 / scale, -length * n2 / a22 / scale
            uc, ud = u.dot(e) / l2, v.dot(e) / l2
            qa, qb = (uc - 1.0) * qc + (ud - 1.0) * qd, -uc * qc - ud * qd
            for axis in ti.static(range(3)):
                gradient[0, axis], gradient[1, axis] = qa[axis], qb[axis]
                gradient[2, axis], gradient[3, axis] = qc[axis], qd[axis]
            for i, j in ti.static(ti.ndrange(4, 3)):
                if not finite_scalar(gradient[i, j]):
                    valid = 0
    return angle, gradient, valid
