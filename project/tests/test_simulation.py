"""Executable checks for cloth simulation and tearing."""

from __future__ import annotations

import argparse

import numpy as np

from project.cloth.failure import AlwaysFalseFailure
from project.main import ClothSimulation, DemoConfig
from project.cloth.taichi_runtime import initialize_taichi


def run(arch: str = "cpu", precision: str = "f32") -> None:
    initialize_taichi(arch, precision)
    interpolation = ClothSimulation(DemoConfig(resolution_x=4, resolution_y=6, arch=arch, precision=precision))
    bottom_id = interpolation.bottom_ids[0]
    initial_y = interpolation.particles.position[bottom_id, 1]
    targets = {
        constraint.particle: constraint.target.copy()
        for constraint in interpolation.solver.pin_constraints
    }
    targets[bottom_id][1] -= 0.04
    seen_y: list[float] = []
    upload = interpolation.solver._update_pin_targets

    def record_targets(values):
        seen_y.append(float(values[bottom_id][1]))
        upload(values)

    interpolation.solver._update_pin_targets = record_targets
    interpolation.solver.step(interpolation.config.dt, iterations=1, substeps=4, pin_targets=targets)
    assert np.allclose(seen_y, initial_y - np.array([0.01, 0.02, 0.03, 0.04])), "pin motion skipped substeps"

    config = DemoConfig(
        resolution_x=8,
        resolution_y=10,
        critical_strain=0.05,
        pull_speed=1.0,
        substeps=3,
        arch=arch,
        precision=precision,
    )
    simulation = ClothSimulation(config)
    observed_topology_version = simulation.mesh.topology_version
    observed_recolor_count = simulation.solver.coloring_recompute_count
    first_cut_checked = False
    for _ in range(80):
        simulation.step()
        assert simulation.finite(), "non-finite particle state"
        if simulation.fracture_log and not first_cut_checked:
            assert any(
                edge.broken and (not edge.active or len(edge.adjacent_triangles) == 1)
                for edge in simulation.mesh.edges
            ), "first fracture did not cut its shared edge"
            first_cut_checked = True
        if simulation.mesh.topology_version != observed_topology_version:
            assert simulation.solver.topology_version == simulation.mesh.topology_version
            assert simulation.solver.coloring_version == simulation.mesh.topology_version
            assert simulation.solver.coloring_recompute_count == observed_recolor_count + 1
            observed_topology_version = simulation.mesh.topology_version
            observed_recolor_count = simulation.solver.coloring_recompute_count
    assert len(simulation.fracture_log) > 0, "expected strain-driven fracture"
    assert simulation.particle_count > 8 * 10, "expected vertex duplication"
    assert simulation.disconnected_components() > 1, "expected disconnected triangle regions"
    assert simulation.solver.coloring_version == simulation.mesh.topology_version
    for color in simulation.solver.stretch_colors + simulation.solver.bend_colors:
        used_vertices: set[int] = set()
        for a, b in color:
            assert a not in used_vertices and b not in used_vertices, "a color contains a shared vertex"
            used_vertices.update((a, b))
    assert all(
        0 <= constraint.a < simulation.particle_count and 0 <= constraint.b < simulation.particle_count
        for constraint in simulation.solver.stretch_constraints + simulation.solver.bend_constraints
    )
    assert all(
        abs(edge.rest_length - simulation.mesh.rest_distance(edge.a, edge.b)) < 1e-10
        for edge in simulation.mesh.edges
    ), "topology rebuild changed material rest lengths"

    simulation.set_failure_model(AlwaysFalseFailure())
    fracture_count = len(simulation.fracture_log)
    for _ in range(20):
        simulation.step()
    assert len(simulation.fracture_log) == fracture_count, "failure model swap was ignored"
    print("all simulation checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32")
    args = parser.parse_args()
    run(args.arch, args.precision)
