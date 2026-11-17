"""Point-triangle closest point (Ericson, Real-Time Collision Detection, 2005, §5.1.5).

closest_point(p, a, b, c, dtype) -> cp, bary, feature   (dtype: working precision)
  bary     barycentric weights of cp on (a, b, c), summing to 1
  feature  FACE (interior), EDGE0 + k for edge k, VERTEX0 + k for vertex k, with
           edge k joining local vertices k and (k + 1) % 3 (edge 0 = ab, 1 = bc, 2 = ca);
           the classification is the exact Voronoi-region test of the algorithm, no
           spatial tolerance (Phase 2.0 MappingFeature semantics: FACE, EDGE, VERTEX).
Degenerate (zero-area) triangles must be skipped by the caller (the face branch divides
by twice the area squared).
"""
import taichi as ti

FACE, EDGE0, VERTEX0 = 0, 1, 4


@ti.func
def closest_point(p, a, b, c, dtype: ti.template()):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    bp = p - b
    d3 = ab.dot(bp)
    d4 = ac.dot(bp)
    cq = p - c
    d5 = ab.dot(cq)
    d6 = ac.dot(cq)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    one = ti.cast(1.0, dtype)
    zero = ti.cast(0.0, dtype)
    w0, w1, w2 = one, zero, zero
    feat = VERTEX0
    if d1 <= 0 and d2 <= 0:                                  # vertex a
        w0, w1, w2 = one, zero, zero
        feat = VERTEX0
    elif d3 >= 0 and d4 <= d3:                               # vertex b
        w0, w1, w2 = zero, one, zero
        feat = VERTEX0 + 1
    elif vc <= 0 and d1 >= 0 and d3 <= 0:                    # edge ab
        v = d1 / (d1 - d3)
        w0, w1, w2 = one - v, v, zero
        feat = EDGE0
    elif d6 >= 0 and d5 <= d6:                               # vertex c
        w0, w1, w2 = zero, zero, one
        feat = VERTEX0 + 2
    elif vb <= 0 and d2 >= 0 and d6 <= 0:                    # edge ca
        w = d2 / (d2 - d6)
        w0, w1, w2 = one - w, zero, w
        feat = EDGE0 + 2
    elif va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:      # edge bc
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        w0, w1, w2 = zero, one - w, w
        feat = EDGE0 + 1
    else:                                                    # face interior
        denom = one / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        w0, w1, w2 = one - v - w, v, w
        feat = FACE
    cp = w0 * a + w1 * b + w2 * c
    return cp, ti.Vector([w0, w1, w2]), feat
