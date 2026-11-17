"""Areal contact traction from a prescribed closed body onto the cloth.

The main load path is a full O(T) triangle integration: each current triangle
is sampled at three barycentric points of area weight A/3, the closed-body
signed distance/normal/velocity are evaluated at the actual sample point, and
the resulting impulse is scattered to the triangle vertices with the standard
barycentric weights.  The sparse ``map_points`` query path is an audit and a
future MPM boundary input; it never carries the main load.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np

from .contracts import ClothSnapshot, MappingResult, QueryPoints, _array, _count, _integer, _real
from .loads import LoadBatch, SourceIdentity
from .provider_contracts import ProviderBoundarySamples

# Three barycentric permutations of (2/3, 1/6, 1/6): equal area weight, and they
# cover the triangle interior symmetrically for a linear pressure field.
_BARYCENTRIC = np.array(((2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
                         (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
                         (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)), dtype=np.float64)
_SAMPLE_WEIGHT = 1.0 / 3.0
_MEASURES = ("unweighted_queries", "body_reference_area", "cloth_current_area")


def _signed_real(value, name):
    """Finite real scalar that may be negative (signed physical work)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, eq=False)
class TractionResult:
    """One aggregated per-substep load plus its audit quantities.

    ``force``/``torque``/``reaction_*`` are world-frame SI values; ``torque``
    is taken about the provider's current centre. ``work`` is *signed* energy
    [J] for the substep (``power * h``), so a retracting body can do negative
    work on the cloth.
    """

    batch: LoadBatch
    force: np.ndarray
    torque: np.ndarray
    reaction_force: np.ndarray
    reaction_torque: np.ndarray
    work: float
    max_penetration: float
    contact_area: float
    current_area: float
    active_samples: int
    integrated_samples: int
    sample_rule: str

    def __post_init__(self):
        if not isinstance(self.batch, LoadBatch):
            raise TypeError("batch must be LoadBatch")
        for name in ("force", "torque", "reaction_force", "reaction_torque"):
            value = _array(getattr(self, name), name, np.float64, (3,), finite=False)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "work", _signed_real(self.work, "work"))
        if not np.isfinite(self.work):
            raise ValueError("work must be finite")
        for name in ("max_penetration", "contact_area", "current_area"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        for name in ("active_samples", "integrated_samples"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.active_samples > self.integrated_samples:
            raise ValueError("active samples cannot exceed integrated samples")
        if self.active_samples and self.contact_area > self.current_area + 1.0e-12 * max(1.0, self.current_area):
            raise ValueError("contact area cannot exceed the current cloth area")


def contact_pressure(phi, cloth_velocity, body_velocity, outward_normal, stiffness, damping):
    """Non-adhesive normal pressure ``max(0, K*delta + c*max(0, -v_rel))``.

    ``phi`` is the closed-body signed distance (negative inside), the normal is
    the body outward normal, so the pressure is positive while the cloth is
    inside and while the cloth closes on the body.  Separating motion cannot
    pull the cloth back: the damping term is clamped to positive values only.
    ``K = c = 0`` is the legal no-contact control and returns exactly zero
    pressure.  Elementwise-finite inputs are not sufficient on their own, so
    every intermediate and the returned pressure must also stay finite.
    """
    phi_array = np.asarray(phi, dtype=np.float64)
    if phi_array.ndim != 1 or not np.isfinite(phi_array).all():
        raise ValueError("phi must be a finite vector")
    count = phi_array.shape[0]
    velocity = _array(cloth_velocity, "cloth_velocity", np.float64, (count, 3))
    body = _array(body_velocity, "body_velocity", np.float64, (count, 3))
    normal = _array(outward_normal, "outward_normal", np.float64, (count, 3))
    if count:
        lengths = np.linalg.norm(normal, axis=1)
        if not np.allclose(lengths, 1.0, rtol=0.0, atol=1.0e-9):
            raise ValueError("outward normal must be unit length")
    stiffness = _real(stiffness, "stiffness")
    damping = _real(damping, "damping")
    delta = np.maximum(0.0, -phi_array)
    with np.errstate(over="ignore", invalid="ignore"):
        relative = np.einsum("ij,ij->i", velocity - body, normal)
        closing = np.maximum(0.0, -relative)
        pressure = np.where(delta > 0.0, stiffness * delta + damping * closing, 0.0)
    if not (np.isfinite(relative).all() and np.isfinite(pressure).all()):
        raise ValueError("contact pressure is not finite; check penetration/stiffness overflow")
    pressure.setflags(write=False)
    return pressure


def contact_stability_limit(mass, area, stiffness, damping, beta: float = 0.2):
    """Local explicit-contact pre-screen: ``h <= beta*sqrt(m/(K A))`` etc.

    Returns the minimum of the active limits. ``K = 0`` and/or ``c = 0`` only
    removes the corresponding restriction; ``K = c = 0`` is the legal no-contact
    control and returns ``inf`` (no contact limit at all, never clamped to a
    finite number).  Finite inputs do not guarantee a finite limit, so
    intermediate products, quotients and the result are all checked.
    It is a reported pre-screen, not a proof of stability for the constrained
    system, and callers must not clamp impulses to hide divergence.
    """
    mass = _real(mass, "mass", positive=True)
    area = _real(area, "area", positive=True)
    stiffness = _real(stiffness, "stiffness")
    damping = _real(damping, "damping")
    beta = _real(beta, "beta", positive=True)
    limits = []
    if stiffness > 0.0:
        with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
            denominator = stiffness * area
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("stiffness*area overflows or vanishes")
            ratio = mass / denominator
            limit = beta * float(np.sqrt(ratio))
        if not (np.isfinite(ratio) and np.isfinite(limit)):
            raise ValueError("stiffness stability limit is not finite")
        limits.append(limit)
    if damping > 0.0:
        with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
            denominator = damping * area
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("damping*area overflows or vanishes")
            limit = beta * mass / denominator
        if not np.isfinite(limit):
            raise ValueError("damping stability limit is not finite")
        limits.append(limit)
    if not limits:
        return float("inf")   # K == c == 0: the contact path imposes no h limit
    return float(min(limits))


def _triangle_data(snapshot: ClothSnapshot):
    """Current sample points and areas [m^2], rejecting invalid contact geometry.

    A zero-area (3D-degenerate) triangle has no areal measure, so the contact
    path refuses it instead of silently zeroing its contribution.  Finite input
    rows are not sufficient: coordinate differences and cross products can still
    overflow to inf.
    """
    triangles = snapshot.triangles
    points = snapshot.positions[triangles]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.cross(first, second)
        double_area = np.linalg.norm(cross, axis=1)
        area = 0.5 * double_area
    if not (np.isfinite(points).all() and np.isfinite(cross).all()
            and np.isfinite(double_area).all()):
        raise ValueError("triangle geometry is not finite (coordinate/cross-product overflow)")
    if np.any(area <= 0.0):
        raise ValueError("contact path rejects zero-area (3D-degenerate) triangles")
    return points, area


def _provider_center(provider, time):
    """Torque origin: the prescribed body centre in world coordinates."""
    kinematics = getattr(provider, "kinematics", None)
    if not callable(kinematics):
        raise TypeError("torque about the body centre requires provider.kinematics(time)")
    returned = kinematics(time)
    if not isinstance(returned, (tuple, list)) or len(returned) != 2:
        raise TypeError("provider.kinematics(time) must return (centre, velocity)")
    center = _array(returned[0], "provider centre", np.float64, (3,))
    if not np.isfinite(center).all():
        raise ValueError("provider centre must be finite")
    return center


def _integrate(snapshot, provider, stiffness, damping):
    points, area = _triangle_data(snapshot)
    origin = _provider_center(provider, snapshot.time)
    sample_positions = np.einsum("kb,tkc->tbc", _BARYCENTRIC, points)
    flat = sample_positions.reshape(-1, 3)
    phi, normal, body_velocity = provider.evaluate_surface(flat, snapshot.time)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1)
    normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    body_velocity = np.asarray(body_velocity, dtype=np.float64).reshape(-1, 3)

    vertex_velocity = np.einsum("kb,tkc->tbc", _BARYCENTRIC, snapshot.velocity[snapshot.triangles])
    vertex_velocity = vertex_velocity.reshape(-1, 3)
    pressure = contact_pressure(phi, vertex_velocity, body_velocity, normal, stiffness, damping)

    point_area = np.repeat(area, 3) * _SAMPLE_WEIGHT
    point_force = normal * (pressure * point_area)[:, None]
    point_torque = np.cross(flat - origin, point_force)
    if not (np.isfinite(point_area).all() and np.isfinite(point_force).all()):
        raise ValueError("contact sample load is not finite")

    vertex_force = np.zeros((snapshot.positions.shape[0], 3), dtype=np.float64)
    vertex_torque = np.zeros_like(vertex_force)
    # Barycentric scatter: ``weights`` is ordered [v0,v1,v2] for every sample,
    # so the target rows must repeat the same interleaving (tile, not repeat).
    weights = np.tile(_BARYCENTRIC, (area.shape[0], 1))
    scattered_force = (point_force[:, None, :] * weights[:, :, None]).reshape(-1, 3)
    scattered_torque = (point_torque[:, None, :] * weights[:, :, None]).reshape(-1, 3)
    sample_rows = np.tile(snapshot.triangles[:, None, :], (1, 3, 1)).ravel()
    np.add.at(vertex_force, sample_rows, scattered_force)
    np.add.at(vertex_torque, sample_rows, scattered_torque)
    with np.errstate(over="ignore", invalid="ignore"):
        force = vertex_force.sum(axis=0)
        torque = vertex_torque.sum(axis=0)
        power = float(np.einsum("ij,ij->", point_force, body_velocity))
        contact_area = float(point_area[pressure > 0.0].sum())
        current_area = float(area.sum(dtype=np.float64))
    if not (np.isfinite(force).all() and np.isfinite(torque).all() and np.isfinite(power)
            and np.isfinite(contact_area) and np.isfinite(current_area)):
        raise ValueError("integrated contact load is not finite")
    max_penetration = float(np.max(np.maximum(0.0, -phi))) if phi.size else 0.0
    active = int(np.count_nonzero(pressure > 0.0))
    return {
        "force": force,
        "torque": torque,
        "power": power,
        "max_penetration": max_penetration,
        "contact_area": contact_area,
        "current_area": current_area,
        "active_samples": active,
        "integrated_samples": int(3 * area.shape[0]),
        "impulse": vertex_force,
    }


def integrate_traction(snapshot, provider, *, stream_id, h, stiffness, damping, batch_id):
    """Full-triangle areal traction load for the next physical substep.

    ``work`` is signed energy [J]: the areal contact power ``Σ F_q·v_body`` [W]
    is multiplied by the actual substep ``h``, so a retracting body can do
    negative work on the cloth.  The reported torque is about the provider's
    current centre, and the batch carries the snapshot's stable vertex IDs.
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    step = _real(h, "h", positive=True)
    stiffness = _real(stiffness, "stiffness")
    damping = _real(damping, "damping")
    source = _validate_provider_source(provider)
    if provider.time != snapshot.time:
        raise ValueError("provider time must equal the snapshot time")
    t_end = snapshot.time + step
    if not np.isfinite(t_end) or t_end <= snapshot.time:
        raise ValueError("substep h does not advance the snapshot clock")

    audit = _integrate(snapshot, provider, stiffness, damping)
    force = audit["force"]
    torque = audit["torque"]
    impulse = audit["impulse"]
    active_rows = np.flatnonzero(np.linalg.norm(impulse, axis=1) > 0.0)
    batch = LoadBatch(
        stream_id=stream_id,
        source=source,
        cloth_version=snapshot.version,
        batch_id=batch_id,
        t_start=snapshot.time,
        t_end=t_end,
        vertex_ids=snapshot.vertex_ids[active_rows],
        impulse=impulse[active_rows] * step,
    )
    result = TractionResult(
        batch=batch,
        force=force,
        torque=torque,
        reaction_force=-force,
        reaction_torque=-torque,
        work=audit["power"] * step,
        max_penetration=audit["max_penetration"],
        contact_area=audit["contact_area"],
        current_area=audit["current_area"],
        active_samples=audit["active_samples"],
        integrated_samples=audit["integrated_samples"],
        sample_rule="triangle_3point_areal",
    )
    return result


def integrate_traction_refined(snapshot, provider, *, subdivisions: int, stiffness, damping):
    """Higher-order audit variant of the areal integral (not the default rule).

    Each triangle is split into ``subdivisions**2`` sub-triangles that each carry
    the same 3-point rule; ``subdivisions=1`` reproduces :func:`integrate_traction`
    exactly (identical samples, forces, torque origin and power).

    This entry point has no substep, so it deliberately reports instantaneous
    contact ``power`` [W] (``power_units`` is explicit) instead of claiming
    work [J]; :func:`integrate_traction` returns ``work = power * h``.
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    subdivisions = _integer(subdivisions, "subdivisions")
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least 1")
    if subdivisions & (subdivisions - 1):
        raise ValueError("subdivisions must be a power of two")
    stiffness = _real(stiffness, "stiffness")
    damping = _real(damping, "damping")
    _validate_provider_source(provider)
    if provider.time != snapshot.time:
        raise ValueError("provider time must equal the snapshot time")
    points, area = _triangle_data(snapshot)
    if subdivisions == 1:
        audit = _integrate(snapshot, provider, stiffness, damping)
        audit.pop("impulse", None)
        audit["sub_triangles"] = int(area.shape[0])
        audit["power_units"] = "W"
        return audit

    # Recursive midpoint subdivision: every child triangle has a valid
    # barycentric map and exactly one quarter of its parent's area.
    local = np.eye(3, dtype=np.float64)[None, :, :]
    level = 0
    while (1 << level) < subdivisions:
        a, b, c = local[:, 0], local[:, 1], local[:, 2]
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        local = np.concatenate([
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ], axis=0)
        level += 1

    triangles = snapshot.triangles
    positions = snapshot.positions
    velocities = snapshot.velocity
    origin = _provider_center(provider, snapshot.time)
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    power = 0.0
    max_penetration = 0.0
    contact_area = 0.0
    active = 0
    sub_area_factor = 1.0 / (subdivisions * subdivisions)
    for weights in local:
        # Each row of weights maps one child vertex to the parent vertices.
        sample_positions = np.einsum("bk,tkc->tbc", weights, points)
        flat = np.einsum("kb,tkc->tbc", _BARYCENTRIC, sample_positions).reshape(-1, 3)
        phi, normal, body_velocity = provider.evaluate_surface(flat, snapshot.time)
        phi = np.asarray(phi, dtype=np.float64).reshape(-1)
        normal = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
        body_velocity = np.asarray(body_velocity, dtype=np.float64).reshape(-1, 3)
        triangle_velocity = np.einsum("bk,tkc->tbc", weights, velocities[triangles])
        vertex_velocity = np.einsum("kb,tkc->tbc", _BARYCENTRIC, triangle_velocity).reshape(-1, 3)
        pressure = contact_pressure(phi, vertex_velocity, body_velocity, normal, stiffness, damping)
        sub_area = np.repeat(area, 3) * _SAMPLE_WEIGHT * sub_area_factor
        point_force = normal * (pressure * sub_area)[:, None]
        if not np.isfinite(point_force).all():
            raise ValueError("refined contact sample load is not finite")
        total_force += point_force.sum(axis=0)
        total_torque += np.cross(flat - origin, point_force).sum(axis=0)
        power += float(np.einsum("ij,ij->", point_force, body_velocity))
        max_penetration = max(max_penetration, float(np.max(np.maximum(0.0, -phi))) if phi.size else 0.0)
        contact_area += float(sub_area[pressure > 0.0].sum())
        active += int(np.count_nonzero(pressure > 0.0))
    if not (np.isfinite(total_force).all() and np.isfinite(total_torque).all()
            and np.isfinite(power) and np.isfinite(contact_area)):
        raise ValueError("refined contact integral is not finite")
    return {
        "force": total_force,
        "torque": total_torque,
        "power": power,
        "power_units": "W",
        "max_penetration": max_penetration,
        "contact_area": contact_area,
        "current_area": float(area.sum(dtype=np.float64)),
        "active_samples": active,
        "integrated_samples": int(3 * area.shape[0] * subdivisions * subdivisions),
        "sub_triangles": int(area.shape[0] * subdivisions * subdivisions),
        "sample_rule": "triangle_3point_areal_subdivided",
    }


def make_boundary_samples(snapshot, queries, mapping, provider, *, stream_id,
                          stiffness, damping,
                          sampling_measure: str = "unweighted_queries",
                          measure_weights=None):
    """Build the audit packet; the sparse query path never carries the load.

    The dispatch default is ``unweighted_queries``: per-row unit weights that
    describe query coverage only and must not be used to integrate a resultant
    force.  ``body_reference_area`` optionally reports the sphere's reference
    surface measure: the deterministic sphere lattice places one equal-area
    cell of ``4*pi*R**2 / Q`` per query, so those weights really do correspond
    to the Q samples (MISS rows keep their weight, never renormalised).  That is
    still the *body* reference measure, not a cloth area and never the main load
    integral (the full-triangle rule in :func:`integrate_traction` is).
    ``cloth_current_area`` has no Jacobian here and therefore requires explicit
    caller-supplied ``measure_weights`` [m^2].
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    if not isinstance(queries, QueryPoints):
        raise TypeError("queries must be QueryPoints")
    if not isinstance(mapping, MappingResult):
        raise TypeError("mapping must be MappingResult")
    source = _validate_provider_source(provider)
    if provider.time != snapshot.time:
        raise ValueError("provider time must equal the snapshot time")
    stiffness = _real(stiffness, "stiffness")
    damping = _real(damping, "damping")
    if sampling_measure not in _MEASURES:
        raise ValueError("unknown sampling measure")

    q = _count(queries.query_ids, "query_ids")
    if measure_weights is not None:
        weights = _array(measure_weights, "measure_weights", np.float64, (q,))
        if sampling_measure == "unweighted_queries" and np.any(weights != 1.0):
            raise ValueError("unweighted query measure uses unit weights, including MISS")
    elif sampling_measure == "unweighted_queries":
        weights = np.ones(q, dtype=np.float64)
    elif sampling_measure == "body_reference_area":
        radius = _real(getattr(getattr(provider, "config", None), "radius", None),
                       "provider radius", positive=True)
        total = 4.0 * np.pi * radius * radius
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("body reference area must be finite and positive")
        weights = np.full(q, total / q, dtype=np.float64) if q else np.empty(0, dtype=np.float64)
    else:
        raise ValueError("cloth_current_area requires explicit measure_weights/Jacobian [m^2]")

    q = weights.shape[0]
    # Validate the producing snapshot/query bindings before invoking the body.
    cloth_velocity = ProviderBoundarySamples.interpolated_velocity(snapshot, queries, mapping)
    hit = mapping.status == 0
    phi_field = np.full(q, np.nan, dtype=np.float64)
    pressure = np.full(q, np.nan, dtype=np.float64)
    outward = np.full((q, 3), np.nan, dtype=np.float64)
    body = np.full((q, 3), np.nan, dtype=np.float64)
    if np.any(hit):
        # Non-HIT rows have no physical binding, even if a candidate cp exists.
        phi, normal, body_velocity = provider.evaluate_surface(mapping.closest_point[hit], snapshot.time)
        phi = np.asarray(phi, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        body_velocity = np.asarray(body_velocity, dtype=np.float64)
        hit_pressure = contact_pressure(phi, cloth_velocity[hit], body_velocity, normal, stiffness, damping)
        phi_field[hit], pressure[hit] = phi, hit_pressure
        outward[hit], body[hit] = normal, body_velocity
    active = hit & (phi_field < 0.0)
    cloth = np.where(hit[:, None], cloth_velocity, np.nan)

    return ProviderBoundarySamples(
        stream_id=stream_id,
        source=source,
        snapshot=snapshot,
        queries=queries,
        mapping=mapping,
        sample_time=snapshot.time,
        cloth_velocity=cloth,
        phi=phi_field,
        outward_normal=outward,
        body_velocity=body,
        pressure=pressure,
        active=active,
        sampling_measure=sampling_measure,
        measure_weights=weights,
    )


def _validate_provider_source(provider) -> SourceIdentity:
    """Only a prescribed provider may drive the load path; return its source."""
    source = getattr(provider, "source", None)
    if not isinstance(source, SourceIdentity):
        raise TypeError("provider must declare a SourceIdentity")
    if source.kind != "prescribed":
        raise ValueError("only a prescribed provider may drive the load path")
    return source


__all__ = ["TractionResult", "contact_pressure", "contact_stability_limit",
           "integrate_traction", "integrate_traction_refined", "make_boundary_samples"]
