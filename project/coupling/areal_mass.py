"""Fixed areal triangle mass with post-tear redistribution.

Mass belongs to the reference surface, not to the appended vertex rows: each
reference triangle keeps ``M_t = areal_density * A_t`` forever, and a vertex
owns one third of every incident triangle.  A tear duplicates a vertex without
duplicating reference area, so redistribution must not grow the total mass.
"""

from __future__ import annotations

import numpy as np

from .contracts import ClothSnapshot, _array, _count, _integer, _real
from .provider_contracts import ParticleMassState, ReferenceState
from .traction import contact_stability_limit as _scalar_stability_limit

__all__ = ["create_reference", "redistribute_mass", "vertex_areas",
           "contact_stability_limit", "stability_limit_from_state"]


def contact_stability_limit(mass, area, stiffness, damping, beta: float = 0.2):
    """Local explicit-contact pre-screen, re-exported from the traction model."""
    return _scalar_stability_limit(mass, area, stiffness, damping, beta)


def _triangle_areas(snapshot: ClothSnapshot) -> np.ndarray:
    points = snapshot.positions[snapshot.triangles]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _reference_triangle_areas(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Rest areas from the *reference* positions, never the stretched state."""
    points = positions[triangles]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    with np.errstate(over="ignore", invalid="ignore"):
        double_area = np.linalg.norm(np.cross(first, second), axis=1)
        area = 0.5 * double_area
    if not (np.isfinite(points).all() and np.isfinite(area).all()):
        raise ValueError("reference triangle geometry is not finite")
    if np.any(area <= 0.0):
        raise ValueError("reference triangles must have positive rest area")
    return area


def create_reference(snapshot, reference_positions, root_vertex_ids, boundary, *,
                     stream_id, areal_density):
    """Freeze per-triangle rest mass from the *reference* triangle areas.

    ``reference_positions`` is the immutable rest geometry (prestrain-free and
    tear-free), so a currently stretched or already torn active state cannot
    change the fixed triangle masses.
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    density = _real(areal_density, "areal_density", positive=True)
    n = _count(snapshot.vertex_ids, "vertex_ids")
    positions = _array(reference_positions, "reference_positions", np.float64, (n, 3))
    # Roots may repeat: every duplicate of a torn vertex keeps its original root.
    roots = _array(root_vertex_ids, "root_vertex_ids", np.int64, (n,))
    boundary_array = _array(boundary, "boundary", bool, (n,))
    index = {int(v): i for i, v in enumerate(snapshot.vertex_ids)}
    for root in roots:
        if int(root) not in index:
            raise ValueError("reference root ID is missing from the active vertex table")
        if int(roots[index[int(root)]]) != int(root):
            raise ValueError("a reference root must be self-rooted")
    for i, root in enumerate(snapshot.vertex_ids):
        j = index[int(roots[i])]
        if boundary_array[i] != boundary_array[j]:
            raise ValueError("duplicate must preserve boundary membership")
        if not np.array_equal(positions[i], positions[j]):
            raise ValueError("duplicate must preserve its reference position")
    areas = _reference_triangle_areas(positions, snapshot.triangles)
    triangle_mass = density * areas
    return ReferenceState(
        stream_id=stream_id,
        epoch=snapshot.epoch,
        topology_version=snapshot.topology_version,
        vertex_ids=snapshot.vertex_ids,
        root_vertex_ids=roots,
        reference_positions=positions,
        boundary=boundary_array,
        triangle_ids=snapshot.triangle_ids,
        triangle_mass=triangle_mass,
    )


def vertex_areas(snapshot: ClothSnapshot) -> np.ndarray:
    """One third of every incident triangle area, per active vertex row."""
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    n = _count(snapshot.vertex_ids, "vertex_ids")
    areas = _triangle_areas(snapshot)
    per_vertex = np.zeros(n, dtype=np.float64)
    np.add.at(per_vertex, snapshot.triangles.reshape(-1),
              np.repeat(areas / 3.0, 3))
    return per_vertex


def redistribute_mass(snapshot, reference):
    """Host mass/inverse-mass view; torn vertices share the fixed reference mass.

    Boundary membership comes from ``reference.boundary``, not from the active
    snapshot: a freshly duplicated boundary vertex is appended unpinned and must
    stay blocked (a relaxed duplicate must never free a boundary child).
    """
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    if not isinstance(reference, ReferenceState):
        raise TypeError("reference must be ReferenceState")
    reference.validate_for(snapshot, stream_id=reference.stream_id)
    n = _count(snapshot.vertex_ids, "vertex_ids")
    mass = np.zeros(n, dtype=np.float64)
    rows = snapshot.triangles.reshape(-1)
    np.add.at(mass, rows, np.repeat(reference.triangle_mass / 3.0, 3))
    if not np.isfinite(mass).all() or np.any(mass <= 0.0):
        raise ValueError("every active vertex needs positive incident reference mass")
    blocked = reference.boundary
    with np.errstate(over="ignore", divide="ignore"):
        inverse_mass = np.where(blocked, 0.0, 1.0 / mass)
    return ParticleMassState(
        stream_id=reference.stream_id,
        cloth_version=snapshot.version,
        vertex_ids=snapshot.vertex_ids,
        mass=mass,
        inverse_mass=inverse_mass,
        pinned=blocked,
    )


def stability_limit_from_state(snapshot, reference, *, stiffness, damping, beta: float = 0.2):
    """Minimum local contact limit over active vertices (never clamped)."""
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("snapshot must be ClothSnapshot")
    if not isinstance(reference, ReferenceState):
        raise TypeError("reference must be ReferenceState")
    reference.validate_for(snapshot, stream_id=reference.stream_id)
    mass = redistribute_mass(snapshot, reference).mass
    areas = vertex_areas(snapshot)
    limits = np.array([
        contact_stability_limit(mass[i], areas[i], stiffness, damping, beta)
        for i in range(mass.shape[0]) if areas[i] > 0.0
    ])
    if limits.size == 0:
        raise ValueError("no active vertex has incident area")
    return float(limits.min())
