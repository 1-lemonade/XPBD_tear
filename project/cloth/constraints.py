"""Taichi-backed XPBD material, constraints, and solver."""

from dataclasses import dataclass
import math

import numpy as np
import taichi as ti

from .mesh import TriangleMesh
from .particles import ParticleSoA
from .taichi_runtime import initialize_taichi


@dataclass
class ClothMaterial:
    # The demo is tuned for visible, controlled stretching rather than an
    # almost-rigid sheet. A softer stretch response avoids post-tear spikes.
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


@ti.data_oriented
class XPBDSolver:
    r"""Small-step XPBD solver independent of fracture/topology policy.

    The host owns the mutable mesh. Particle state and active constraint data
    are mirrored into fixed-capacity Taichi fields. Within each substep the
    solver processes stretch colors in order, then bend colors in order, then
    pins. Constraints in one color share no particles and therefore project
    in parallel without atomics; colors and the constraint type passes remain
    sequential Gauss-Seidel-style passes. The changed projection order may
    move the exact tear frame while preserving the compliant XPBD update.
    """

    def __init__(
        self,
        particles: ParticleSoA,
        mesh: TriangleMesh,
        material: ClothMaterial | None = None,
        *,
        capacity_headroom: float = 4.0,
        arch: str = "cpu",
        precision: str = "f32",
    ):
        self.backend = initialize_taichi(arch, precision, log=False)
        self.particles = particles
        self.mesh = mesh
        self.material = material or ClothMaterial()
        self.arch = arch
        self.precision = precision
        self.capacity_headroom = float(capacity_headroom)
        if self.capacity_headroom < 1.0:
            raise ValueError("capacity_headroom must be at least 1.0")

        self.stretch_constraints: list[DistanceConstraint] = []
        self.bend_constraints: list[DistanceConstraint] = []
        self.pin_constraints: list[PinConstraint] = []

        scalar_type = ti.f32 if precision == "f32" else ti.f64
        state_dtype = np.float32 if precision == "f32" else np.float64
        self.vertex_capacity = max(1, math.ceil(particles.count * self.capacity_headroom))
        initial_stretch_count = sum(edge.active for edge in mesh.edges)
        initial_bend_count = sum(edge.active and len(edge.adjacent_triangles) == 2 for edge in mesh.edges)
        initial_pin_count = max(1, int(np.count_nonzero(particles.pinned)))
        self.stretch_capacity = max(1, math.ceil(initial_stretch_count * self.capacity_headroom))
        self.bend_capacity = max(1, math.ceil(initial_bend_count * self.capacity_headroom))
        self.pin_capacity = max(1, math.ceil(initial_pin_count * self.capacity_headroom))
        self._state_dtype = state_dtype

        # Slots: x, previous x, predicted x, and velocity. This permits one
        # synchronized field readback after all substeps in a frame.
        self.state = ti.Vector.field(12, dtype=scalar_type, shape=self.vertex_capacity)
        self.inverse_mass_field = ti.field(dtype=scalar_type, shape=self.vertex_capacity)
        self.active_particle_count = ti.field(dtype=ti.i32, shape=())

        self.stretch_a = ti.field(dtype=ti.i32, shape=self.stretch_capacity)
        self.stretch_b = ti.field(dtype=ti.i32, shape=self.stretch_capacity)
        self.stretch_rest = ti.field(dtype=scalar_type, shape=self.stretch_capacity)
        self.stretch_lambda = ti.field(dtype=scalar_type, shape=self.stretch_capacity)
        self.active_stretch_count = ti.field(dtype=ti.i32, shape=())
        self.stretch_compliance_value = ti.field(dtype=scalar_type, shape=())

        self.bend_a = ti.field(dtype=ti.i32, shape=self.bend_capacity)
        self.bend_b = ti.field(dtype=ti.i32, shape=self.bend_capacity)
        self.bend_rest = ti.field(dtype=scalar_type, shape=self.bend_capacity)
        self.bend_lambda = ti.field(dtype=scalar_type, shape=self.bend_capacity)
        self.active_bend_count = ti.field(dtype=ti.i32, shape=())
        self.bend_compliance_value = ti.field(dtype=scalar_type, shape=())

        self.pin_particle = ti.field(dtype=ti.i32, shape=self.pin_capacity)
        self.pin_target = ti.Vector.field(3, dtype=scalar_type, shape=self.pin_capacity)
        self.pin_lambda = ti.Vector.field(3, dtype=scalar_type, shape=self.pin_capacity)
        self.pin_index_for_particle = ti.field(dtype=ti.i32, shape=self.vertex_capacity)
        self.active_pin_count = ti.field(dtype=ti.i32, shape=())
        self.pin_compliance_value = ti.field(dtype=scalar_type, shape=())
        self.dt_value = ti.field(dtype=scalar_type, shape=())
        self.gravity_value = ti.Vector.field(3, dtype=scalar_type, shape=())
        self.damping_value = ti.field(dtype=scalar_type, shape=())

        self.stretch_color_groups: list[tuple[int, int]] = []
        self.bend_color_groups: list[tuple[int, int]] = []
        self.stretch_colors: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.bend_colors: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.topology_version = -1
        self.coloring_version = -1
        self.repack_count = 0
        self.coloring_recompute_count = 0

        self.stretch_compliance_value[None] = self.material.stretch_compliance
        self.bend_compliance_value[None] = self.material.bend_compliance
        self.pin_compliance_value[None] = self.material.pin_compliance
        self.gravity_value[None] = np.asarray(self.material.gravity, dtype=self._state_dtype)
        self.damping_value[None] = self.material.damping
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
            rest = self.mesh.rest_distance(opposite_a, opposite_b)
            bend.append(DistanceConstraint(opposite_a, opposite_b, rest, self.material.bend_compliance, "bend"))
        self.bend_constraints = bend
        self._repack_after_topology_change()

    def set_pins(self, targets: dict[int, np.ndarray]) -> None:
        self.pin_constraints = [
            PinConstraint(pid, np.asarray(target, dtype=np.float64).copy(), self.material.pin_compliance)
            for pid, target in sorted(targets.items())
        ]
        self._upload_pin_targets()

    def update_material(self, material: ClothMaterial) -> None:
        self.material = material
        for constraint in self.stretch_constraints:
            constraint.compliance = material.stretch_compliance
        for constraint in self.bend_constraints:
            constraint.compliance = material.bend_compliance
        for constraint in self.pin_constraints:
            constraint.compliance = material.pin_compliance
        self.stretch_compliance_value[None] = material.stretch_compliance
        self.bend_compliance_value[None] = material.bend_compliance
        self.pin_compliance_value[None] = material.pin_compliance
        self.gravity_value[None] = np.asarray(material.gravity, dtype=self._state_dtype)
        self.damping_value[None] = material.damping

    def step(self, dt: float, iterations: int = 8, substeps: int = 4, pin_targets: dict[int, np.ndarray] | None = None) -> None:
        self._ensure_topology_current()
        substeps = max(1, int(substeps))
        h = dt / substeps
        # Move kinematic boundaries at the substep rate. Applying the whole
        # frame displacement in the first substep creates a spurious impulse.
        start_targets = {constraint.particle: constraint.target.copy() for constraint in self.pin_constraints}
        final_targets = {
            pid: np.asarray(pin_targets.get(pid, start), dtype=np.float64)
            if pin_targets is not None else start
            for pid, start in start_targets.items()
        }
        for substep_index in range(substeps):
            fraction = (substep_index + 1) / substeps
            targets = {
                pid: start + fraction * (final_targets[pid] - start)
                for pid, start in start_targets.items()
            }
            self._update_pin_targets(targets)
            self._run_substep(h, iterations)
        self._sync_particle_state()

    def substep(self, dt: float, iterations: int = 8, pin_targets: dict[int, np.ndarray] | None = None) -> None:
        self._ensure_topology_current()
        self._update_pin_targets(pin_targets)
        self._run_substep(dt, iterations)
        self._sync_particle_state()

    def _ensure_topology_current(self) -> None:
        if self.topology_version != getattr(self.mesh, "topology_version", 0):
            self.rebuild_constraints()

    def _repack_after_topology_change(self) -> None:
        """Repack indices and recolor both constraint sets at one mesh version."""
        topology_version = getattr(self.mesh, "topology_version", 0)
        self._check_capacity("vertices", self.particles.count, self.vertex_capacity)

        self.stretch_colors = self._color_constraints(self.stretch_constraints)
        self.bend_colors = self._color_constraints(self.bend_constraints)
        self.stretch_color_groups = self._upload_distance_constraints(
            self.stretch_constraints,
            self.stretch_colors,
            self.stretch_capacity,
            self.stretch_a,
            self.stretch_b,
            self.stretch_rest,
            self.active_stretch_count,
            "stretch constraints",
        )
        self.bend_color_groups = self._upload_distance_constraints(
            self.bend_constraints,
            self.bend_colors,
            self.bend_capacity,
            self.bend_a,
            self.bend_b,
            self.bend_rest,
            self.active_bend_count,
            "bend constraints",
        )

        count = self.particles.count
        state = np.zeros((self.vertex_capacity, 12), dtype=self._state_dtype)
        state[:count, 0:3] = self.particles.position
        state[:count, 3:6] = self.particles.previous_position
        state[:count, 6:9] = self.particles.predicted_position
        state[:count, 9:12] = self.particles.velocity
        inverse_mass = np.zeros(self.vertex_capacity, dtype=self._state_dtype)
        inverse_mass[:count] = self.particles.inverse_mass
        self.state.from_numpy(state)
        self.inverse_mass_field.from_numpy(inverse_mass)
        self.active_particle_count[None] = count
        self._upload_pin_targets()

        self.topology_version = topology_version
        self.coloring_version = topology_version
        self.repack_count += 1
        self.coloring_recompute_count += 1

    def _upload_distance_constraints(
        self,
        constraints: list[DistanceConstraint],
        colors: tuple[tuple[tuple[int, int], ...], ...],
        capacity: int,
        field_a,
        field_b,
        field_rest,
        active_count_field,
        label: str,
    ) -> list[tuple[int, int]]:
        self._check_capacity(label, len(constraints), capacity)
        ordered: list[DistanceConstraint] = []
        groups: list[tuple[int, int]] = []
        constraint_by_pair = {(c.a, c.b): c for c in constraints}
        for color in colors:
            start = len(ordered)
            ordered.extend(constraint_by_pair[pair] for pair in color)
            groups.append((start, len(ordered)))

        a = np.zeros(capacity, dtype=np.int32)
        b = np.zeros(capacity, dtype=np.int32)
        rest = np.zeros(capacity, dtype=self._state_dtype)
        for index, constraint in enumerate(ordered):
            a[index], b[index], rest[index] = constraint.a, constraint.b, constraint.rest_length
        field_a.from_numpy(a)
        field_b.from_numpy(b)
        field_rest.from_numpy(rest)
        active_count_field[None] = len(ordered)
        return groups

    @staticmethod
    def _color_constraints(constraints: list[DistanceConstraint]) -> tuple[tuple[tuple[int, int], ...], ...]:
        """Greedy graph coloring; every color contains vertex-disjoint pairs."""
        colors: list[list[tuple[int, int]]] = []
        occupied: list[set[int]] = []
        for constraint in constraints:
            pair = (constraint.a, constraint.b)
            vertices = {constraint.a, constraint.b}
            for color_index, used in enumerate(occupied):
                if vertices.isdisjoint(used):
                    colors[color_index].append(pair)
                    used.update(vertices)
                    break
            else:
                colors.append([pair])
                occupied.append(set(vertices))
        return tuple(tuple(color) for color in colors)

    @staticmethod
    def _check_capacity(label: str, count: int, capacity: int) -> None:
        if count > capacity:
            raise RuntimeError(
                f"{label} exceeded its preallocated capacity ({count} > {capacity}); "
                "increase --capacity-headroom and restart the run"
            )

    def _upload_pin_targets(self, rebuild_particle_index: bool = True) -> None:
        self._check_capacity("pin constraints", len(self.pin_constraints), self.pin_capacity)
        targets = np.zeros((self.pin_capacity, 3), dtype=self._state_dtype)
        particle_ids = np.zeros(self.pin_capacity, dtype=np.int32) if rebuild_particle_index else None
        pin_index = np.full(self.vertex_capacity, -1, dtype=np.int32) if rebuild_particle_index else None
        for index, constraint in enumerate(self.pin_constraints):
            targets[index] = constraint.target
            if rebuild_particle_index:
                particle_ids[index] = constraint.particle
                pin_index[constraint.particle] = index
        if rebuild_particle_index:
            self.pin_particle.from_numpy(particle_ids)
            self.pin_index_for_particle.from_numpy(pin_index)
        self.pin_target.from_numpy(targets)
        self.active_pin_count[None] = len(self.pin_constraints)

    def _update_pin_targets(self, pin_targets: dict[int, np.ndarray] | None) -> None:
        if pin_targets is not None:
            for constraint in self.pin_constraints:
                if constraint.particle in pin_targets:
                    constraint.target = np.asarray(pin_targets[constraint.particle], dtype=np.float64).copy()
        if self.pin_constraints:
            self._upload_pin_targets(rebuild_particle_index=False)

    def _sync_particle_state(self) -> None:
        count = self.particles.count
        state = self.state.to_numpy()[:count]
        self.particles.position[:] = state[:, 0:3]
        self.particles.previous_position[:] = state[:, 3:6]
        self.particles.predicted_position[:] = state[:, 6:9]
        self.particles.velocity[:] = state[:, 9:12]

    def _run_substep(self, dt: float, iterations: int) -> None:
        self.dt_value[None] = dt
        self._predict()
        self._clear_lambdas()
        for _ in range(max(1, int(iterations))):
            for start, end in self.stretch_color_groups:
                self._solve_distance_color(start, end, 0)
            for start, end in self.bend_color_groups:
                self._solve_distance_color(start, end, 1)
            self._solve_pins()
        self._finish_substep()

    @ti.kernel
    def _predict(self):
        for i in range(self.active_particle_count[None]):
            for axis in ti.static(range(3)):
                self.state[i][3 + axis] = self.state[i][axis]
            if self.inverse_mass_field[i] > 0.0:
                self.state[i][6] = self.state[i][0] + self.state[i][9] * self.dt_value[None] + self.gravity_value[None][0] * self.dt_value[None] * self.dt_value[None]
                self.state[i][7] = self.state[i][1] + self.state[i][10] * self.dt_value[None] + self.gravity_value[None][1] * self.dt_value[None] * self.dt_value[None]
                self.state[i][8] = self.state[i][2] + self.state[i][11] * self.dt_value[None] + self.gravity_value[None][2] * self.dt_value[None] * self.dt_value[None]
            else:
                for axis in ti.static(range(3)):
                    self.state[i][6 + axis] = self.state[i][axis]

    @ti.kernel
    def _clear_lambdas(self):
        for i in range(self.active_stretch_count[None]):
            self.stretch_lambda[i] = 0.0
        for i in range(self.active_bend_count[None]):
            self.bend_lambda[i] = 0.0
        for i in range(self.active_pin_count[None]):
            self.pin_lambda[i][0] = 0.0
            self.pin_lambda[i][1] = 0.0
            self.pin_lambda[i][2] = 0.0

    @ti.kernel
    def _solve_distance_color(self, start: ti.i32, end: ti.i32, kind: ti.i32):
        for index in range(start, end):
            a = 0
            b = 0
            rest = 0.0
            compliance = 0.0
            multiplier = 0.0
            if kind == 0:
                a = self.stretch_a[index]
                b = self.stretch_b[index]
                rest = self.stretch_rest[index]
                compliance = self.stretch_compliance_value[None]
                multiplier = self.stretch_lambda[index]
            else:
                a = self.bend_a[index]
                b = self.bend_b[index]
                rest = self.bend_rest[index]
                compliance = self.bend_compliance_value[None]
                multiplier = self.bend_lambda[index]

            dx = self.state[a][6] - self.state[b][6]
            dy = self.state[a][7] - self.state[b][7]
            dz = self.state[a][8] - self.state[b][8]
            length = ti.sqrt(dx * dx + dy * dy + dz * dz)
            if length > 1.0e-12:
                w1 = self.inverse_mass_field[a]
                w2 = self.inverse_mass_field[b]
                alpha = compliance / ti.max(self.dt_value[None] * self.dt_value[None], 1.0e-16)
                denominator = w1 + w2 + alpha
                if denominator > 1.0e-16:
                    value = length - rest
                    delta_lambda = (-value - alpha * multiplier) / denominator
                    if kind == 0:
                        self.stretch_lambda[index] += delta_lambda
                    else:
                        self.bend_lambda[index] += delta_lambda
                    scale = delta_lambda / length
                    self.state[a][6] += w1 * scale * dx
                    self.state[a][7] += w1 * scale * dy
                    self.state[a][8] += w1 * scale * dz
                    self.state[b][6] -= w2 * scale * dx
                    self.state[b][7] -= w2 * scale * dy
                    self.state[b][8] -= w2 * scale * dz

    @ti.kernel
    def _solve_pins(self):
        for index in range(self.active_pin_count[None]):
            particle = self.pin_particle[index]
            weight = self.inverse_mass_field[particle]
            alpha = self.pin_compliance_value[None] / ti.max(self.dt_value[None] * self.dt_value[None], 1.0e-16)
            denominator = weight + alpha
            for axis in ti.static(range(3)):
                if denominator <= 1.0e-16:
                    self.state[particle][6 + axis] = self.pin_target[index][axis]
                else:
                    correction = (
                        -self.state[particle][6 + axis]
                        + self.pin_target[index][axis]
                        - alpha * self.pin_lambda[index][axis]
                    ) / denominator
                    self.pin_lambda[index][axis] += correction
                    self.state[particle][6 + axis] += weight * correction

    @ti.kernel
    def _finish_substep(self):
        for i in range(self.active_particle_count[None]):
            pin_index = self.pin_index_for_particle[i]
            for axis in ti.static(range(3)):
                if pin_index >= 0:
                    target = self.pin_target[pin_index][axis]
                    self.state[i][axis] = target
                    self.state[i][6 + axis] = target
                    self.state[i][9 + axis] = 0.0
                else:
                    self.state[i][axis] = self.state[i][6 + axis]
                    if self.inverse_mass_field[i] > 0.0:
                        self.state[i][9 + axis] = (self.state[i][6 + axis] - self.state[i][3 + axis]) / ti.max(self.dt_value[None], 1.0e-12)
                        self.state[i][9 + axis] *= self.damping_value[None]
                    else:
                        self.state[i][9 + axis] = 0.0
