"""XPBD material, constraint construction, and compliant solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import TriangleMesh
from .particles import ParticleSoA


@dataclass
class ClothMaterial:
    # The demo is tuned for visible, controlled stretching rather than an
    # almost-rigid sheet.  A softer stretch response also keeps fracture
    # energy from turning into a large post-tear velocity spike.
    stretch_compliance: float = 1.0e-6
    bend_compliance: float = 2.0e-5
    pin_compliance: float = 0.0
    damping: float = 0.96
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)


@dataclass
class DistanceConstraint:
    a: int
    b: int
    rest_length: float
    compliance: float
    kind: str = "stretch"


@dataclass
class PinConstraint:
    particle: int
    target: np.ndarray
    compliance: float


class XPBDSolver:
    r"""Small-step XPBD solver independent of fracture/topology policy.

    The position correction is the standard XPBD update:

        \( \Delta\lambda = (-C - \alpha\lambda) / (\sum_i w_i\|\nabla_i C\|^2 + \alpha) \),
        \( \alpha = c / h^2 \).

    Lambdas are reset at the beginning of every substep, as is customary for
    a small-step implementation.
    """

    def __init__(self, particles: ParticleSoA, mesh: TriangleMesh, material: ClothMaterial | None = None):
        self.particles = particles
        self.mesh = mesh
        self.material = material or ClothMaterial()
        self.stretch_constraints: list[DistanceConstraint] = []
        self.bend_constraints: list[DistanceConstraint] = []
        self.pin_constraints: list[PinConstraint] = []
        self.rebuild_constraints()

    def rebuild_constraints(self) -> None:
        self.stretch_constraints = [
            DistanceConstraint(edge.a, edge.b, edge.rest_length, self.material.stretch_compliance, "stretch")
            for edge in self.mesh.edges
            if edge.active
        ]
        bend: list[DistanceConstraint] = []
        for edge in self.mesh.edges:
            if not edge.active or len(edge.adjacent_triangles) != 2:
                continue
            first = self.mesh.triangles[edge.adjacent_triangles[0]]
            second = self.mesh.triangles[edge.adjacent_triangles[1]]
            opposite_a = int(next(v for v in first if v not in (edge.a, edge.b)))
            opposite_b = int(next(v for v in second if v not in (edge.a, edge.b)))
            rest = float(np.linalg.norm(self.particles.position[opposite_a] - self.particles.position[opposite_b]))
            bend.append(DistanceConstraint(opposite_a, opposite_b, rest, self.material.bend_compliance, "bend"))
        self.bend_constraints = bend

    def set_pins(self, targets: dict[int, np.ndarray]) -> None:
        self.pin_constraints = [
            PinConstraint(pid, np.asarray(target, dtype=np.float64).copy(), self.material.pin_compliance)
            for pid, target in sorted(targets.items())
        ]

    def update_material(self, material: ClothMaterial) -> None:
        self.material = material
        for constraint in self.stretch_constraints:
            constraint.compliance = material.stretch_compliance
        for constraint in self.bend_constraints:
            constraint.compliance = material.bend_compliance
        for constraint in self.pin_constraints:
            constraint.compliance = material.pin_compliance

    def step(self, dt: float, iterations: int = 8, substeps: int = 4, pin_targets: dict[int, np.ndarray] | None = None) -> None:
        substeps = max(1, int(substeps))
        h = dt / substeps
        for _ in range(substeps):
            self.substep(h, iterations, pin_targets)

    def substep(self, dt: float, iterations: int = 8, pin_targets: dict[int, np.ndarray] | None = None) -> None:
        p = self.particles
        p.previous_position[:] = p.position
        gravity = np.asarray(self.material.gravity, dtype=np.float64)
        dynamic = p.inverse_mass > 0.0
        # Semi-implicit prediction: \( \mathbf{x}^{*} = \mathbf{x}^{n} + h\mathbf{v}^{n} + h^{2}\mathbf{g} \).
        p.predicted_position[:] = p.position
        p.predicted_position[dynamic] += p.velocity[dynamic] * dt + gravity * (dt * dt)

        if pin_targets is not None:
            for constraint in self.pin_constraints:
                if constraint.particle in pin_targets:
                    constraint.target = np.asarray(pin_targets[constraint.particle], dtype=np.float64)

        stretch_lambdas = np.zeros(len(self.stretch_constraints), dtype=np.float64)
        bend_lambdas = np.zeros(len(self.bend_constraints), dtype=np.float64)
        pin_lambdas = np.zeros((len(self.pin_constraints), 3), dtype=np.float64)
        for _ in range(max(1, int(iterations))):
            for i, constraint in enumerate(self.stretch_constraints):
                self._solve_distance(constraint, dt, stretch_lambdas, i)
            for i, constraint in enumerate(self.bend_constraints):
                self._solve_distance(constraint, dt, bend_lambdas, i)
            for i, constraint in enumerate(self.pin_constraints):
                self._solve_pin(constraint, dt, pin_lambdas[i])

        p.position[:] = p.predicted_position
        # Velocity reconstruction: \( \mathbf{v}^{n+1} = (\mathbf{x}^{n+1} - \mathbf{x}^{n}) / h \).
        p.velocity[:] = (p.position - p.previous_position) / max(dt, 1e-12)
        p.velocity *= self.material.damping
        for constraint in self.pin_constraints:
            p.position[constraint.particle] = constraint.target
            p.predicted_position[constraint.particle] = constraint.target
            p.velocity[constraint.particle] = 0.0

    def _solve_distance(self, c: DistanceConstraint, dt: float, lambdas: np.ndarray, index: int) -> None:
        p = self.particles
        x1 = p.predicted_position[c.a]
        x2 = p.predicted_position[c.b]
        delta = x1 - x2
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            return
        # Distance constraint: \( C(\mathbf{x}) = \|\mathbf{x}_a - \mathbf{x}_b\| - L_0 \).
        gradient = delta / length
        # Constraint gradient: \( \nabla C = (\mathbf{x}_a - \mathbf{x}_b) / \|\mathbf{x}_a - \mathbf{x}_b\| \).
        value = length - c.rest_length
        w1 = p.inverse_mass[c.a]
        w2 = p.inverse_mass[c.b]
        # XPBD compliance: \( \alpha = c / h^2 \), where c is compliance.
        alpha = c.compliance / max(dt * dt, 1e-16)
        denominator = w1 + w2 + alpha
        if denominator <= 1e-16:
            return
        # Multiplier update: \( \Delta\lambda = (-C - \alpha\lambda) / (\sum_i w_i\|\nabla_i C\|^2 + \alpha) \).
        delta_lambda = (-value - alpha * lambdas[index]) / denominator
        lambdas[index] += delta_lambda
        # Position correction: \( \Delta\mathbf{x}_i = w_i\Delta\lambda\nabla_i C \).
        p.predicted_position[c.a] += w1 * delta_lambda * gradient
        p.predicted_position[c.b] -= w2 * delta_lambda * gradient

    def _solve_pin(self, c: PinConstraint, dt: float, lambdas: np.ndarray) -> None:
        p = self.particles
        weight = p.inverse_mass[c.particle]
        # Pin constraint: \( C(\mathbf{x}) = \mathbf{x}_i - \mathbf{x}_{target} \), solved component-wise.
        alpha = c.compliance / max(dt * dt, 1e-16)
        denominator = weight + alpha
        if denominator <= 1e-16:
            p.predicted_position[c.particle] = c.target
            return
        correction = (-p.predicted_position[c.particle] + c.target - alpha * lambdas) / denominator
        lambdas += correction
        p.predicted_position[c.particle] += weight * correction
