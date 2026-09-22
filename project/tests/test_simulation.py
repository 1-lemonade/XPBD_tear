"""Small executable checks for the Phase 1 acceptance-critical behaviors."""

from __future__ import annotations

from project.cloth.failure import AlwaysFalseFailure
from project.main import ClothSimulation, DemoConfig


def run() -> None:
    config = DemoConfig(
        resolution_x=8,
        resolution_y=10,
        critical_strain=0.05,
        pull_speed=1.0,
        substeps=3,
    )
    simulation = ClothSimulation(config)
    for _ in range(80):
        simulation.step()
        assert simulation.finite(), "non-finite particle state"
    assert len(simulation.fracture_log) > 0, "expected strain-driven fracture"
    assert simulation.particle_count > 8 * 10, "expected vertex duplication"
    assert simulation.disconnected_components() > 1, "expected disconnected triangle regions"
    assert all(
        0 <= constraint.a < simulation.particle_count and 0 <= constraint.b < simulation.particle_count
        for constraint in simulation.solver.stretch_constraints + simulation.solver.bend_constraints
    )

    simulation.set_failure_model(AlwaysFalseFailure())
    fracture_count = len(simulation.fracture_log)
    for _ in range(20):
        simulation.step()
    assert len(simulation.fracture_log) == fracture_count, "failure model swap was ignored"
    print("all simulation checks passed")


if __name__ == "__main__":
    run()

