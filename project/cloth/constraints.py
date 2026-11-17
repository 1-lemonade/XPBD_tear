"""Taichi-backed XPBD material, constraints, and solver."""

from dataclasses import dataclass
import math
import time

import numpy as np
import taichi as ti

from .mesh import TriangleMesh
from .particles import ParticleSoA
from .taichi_runtime import initialize_taichi
from ..solver_convergence import convergence_policy, stop_reason, validate_convergence_record
from ..solver_convergence import (reference_record_from_checks, reference_stop_iteration,
                                  validate_reference_policy, validate_reference_record)
from .bending import (DihedralConstraint, build_dihedral_constraints,
                      dihedral_geometry, finite_scalar, wrapped_angle)


@dataclass
class ClothMaterial:
    # The demo is tuned for visible, controlled stretching rather than an
    # almost-rigid sheet. A softer stretch response avoids post-tear spikes.
    stretch_compliance: float = 1.0e-6
    bend_compliance: float = 2.0e-5
    pin_compliance: float = 0.0
    damping: float = 0.96
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    # Appended to preserve positional arguments. Distance compliance is m/N;
    # the opt-in dihedral coefficient B is N m and must be supplied explicitly.
    bend_model: str = "distance"
    bend_stiffness: float | None = None


@dataclass
class DistanceConstraint:
    a: int
    b: int
    rest_length: float
    compliance: float
    kind: str = "stretch"

    @property
    def vertices(self) -> tuple[int, int]:
        return (self.a, self.b)


@dataclass
class PinConstraint:
    particle: int
    target: np.ndarray
    compliance: float


class ReferenceAborted(RuntimeError):
    """A reference window was abandoned uncommitted at the caller's request."""


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
        self._validate_bend_material(self.material, precision)
        self._bend_model = self.material.bend_model
        self._bend_failed: str | None = None
        self._bend_geometry_tolerance = 1.0e-6 if precision == "f32" else 1.0e-12
        self._bend_branch_tolerance = 1.0e-5 if precision == "f32" else 1.0e-10
        self.arch = arch
        self.precision = precision
        self.capacity_headroom = float(capacity_headroom)
        if self.capacity_headroom < 1.0:
            raise ValueError("capacity_headroom must be at least 1.0")

        self.stretch_constraints: list[DistanceConstraint] = []
        self.bend_constraints: list[DistanceConstraint | DihedralConstraint] = []
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
        # Keep the default distance path's memory footprint and raw schema.
        if self._bend_model == "dihedral":
            self.bend_c = ti.field(dtype=ti.i32, shape=self.bend_capacity)
            self.bend_d = ti.field(dtype=ti.i32, shape=self.bend_capacity)
            self.bend_weight = ti.field(dtype=scalar_type, shape=self.bend_capacity)
            self.bend_angle = ti.field(dtype=scalar_type, shape=self.bend_capacity)
            self.bend_error = ti.field(dtype=scalar_type, shape=self.bend_capacity)
            self.bend_energy = ti.field(dtype=scalar_type, shape=self.bend_capacity)
            self.bend_stiffness_value = ti.field(dtype=scalar_type, shape=())
            self.bend_failure_slot = ti.field(dtype=ti.i32, shape=())
            self.bend_failure_code = ti.field(dtype=ti.i32, shape=self.bend_capacity)
            self.bend_stiffness_value[None] = self.material.bend_stiffness

        self.pin_particle = ti.field(dtype=ti.i32, shape=self.pin_capacity)
        self.pin_target = ti.Vector.field(3, dtype=scalar_type, shape=self.pin_capacity)
        self.pin_lambda = ti.Vector.field(3, dtype=scalar_type, shape=self.pin_capacity)
        self.pin_index_for_particle = ti.field(dtype=ti.i32, shape=self.vertex_capacity)
        self.active_pin_count = ti.field(dtype=ti.i32, shape=())
        self.pin_compliance_value = ti.field(dtype=scalar_type, shape=())
        self.dt_value = ti.field(dtype=scalar_type, shape=())
        self.gravity_value = ti.Vector.field(3, dtype=scalar_type, shape=())
        self.damping_value = ti.field(dtype=scalar_type, shape=())
        self.relaxation_value = ti.field(dtype=scalar_type, shape=())
        self.relaxation_value[None] = 1.
        self.convergence_residuals = ti.Vector.field(2, dtype=scalar_type, shape=())
        self.convergence_invalid = ti.field(dtype=ti.i32, shape=())
        self.last_convergence = None
        # Optional heavy-load diagnostics: opt-in and lazily allocated.
        self.last_reference = None
        self.last_reference_timing = None
        self._fixed_tolerance = 0.0
        self._fixed_tolerance_field = None
        self._reference_state = None
        self._reference_slots = None

        self.stretch_color_groups: list[tuple[int, int]] = []
        self.bend_color_groups: list[tuple[int, int]] = []
        self.stretch_colors: tuple[tuple[tuple[int, int], ...], ...] = ()
        self.bend_colors: tuple[tuple[tuple[int, ...], ...], ...] = ()
        self.topology_version = -1
        self.coloring_version = -1
        self.repack_count = 0
        self.coloring_recompute_count = 0
        self._diagnostic_stamp = None
        self._bridge_mass_generation = None
        self._bridge_mass_field = None
        self._balance_buffers = None
        self._balance_completed = False

        self.stretch_compliance_value[None] = self.material.stretch_compliance
        self.bend_compliance_value[None] = self.material.bend_compliance
        self.pin_compliance_value[None] = self.material.pin_compliance
        self.gravity_value[None] = np.asarray(self.material.gravity, dtype=self._state_dtype)
        self.damping_value[None] = self.material.damping
        self.rebuild_constraints()

    def rebuild_constraints(self) -> None:
        self._check_bend_health()
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
        self.bend_constraints = (
            build_dihedral_constraints(self.mesh, self._bend_geometry_tolerance)
            if self._bend_model == "dihedral" else bend
        )
        self._repack_after_topology_change()

    def set_pins(self, targets: dict[int, np.ndarray]) -> None:
        self._diagnostic_stamp = None
        self._bridge_mass_generation = None
        self.pin_constraints = [
            PinConstraint(pid, np.asarray(target, dtype=np.float64).copy(), self.material.pin_compliance)
            for pid, target in sorted(targets.items())
        ]
        self._upload_pin_targets()

    def update_material(self, material: ClothMaterial) -> None:
        self._check_bend_health()
        self._validate_bend_material(material, self.precision)
        if material.bend_model != self._bend_model:
            raise ValueError("changing bend_model requires a new solver")
        if self._bend_model == "dihedral":
            self._validate_hinge_stiffness(material.bend_stiffness)
        self._diagnostic_stamp = None
        self.material = material
        for constraint in self.stretch_constraints:
            constraint.compliance = material.stretch_compliance
        if self._bend_model == "distance":
            for constraint in self.bend_constraints:
                constraint.compliance = material.bend_compliance
        else:
            self.bend_stiffness_value[None] = material.bend_stiffness
        for constraint in self.pin_constraints:
            constraint.compliance = material.pin_compliance
        self.stretch_compliance_value[None] = material.stretch_compliance
        self.bend_compliance_value[None] = material.bend_compliance
        self.pin_compliance_value[None] = material.pin_compliance
        self.gravity_value[None] = np.asarray(material.gravity, dtype=self._state_dtype)
        self.damping_value[None] = material.damping

    def step(self, dt: float, iterations: int = 8, substeps: int = 4, pin_targets: dict[int, np.ndarray] | None = None) -> None:
        self._check_bend_health()
        self._diagnostic_stamp = None
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
        self._mark_completed_substep(h)

    def substep(self, dt: float, iterations: int = 8, pin_targets: dict[int, np.ndarray] | None = None,
                *, convergence: bool = False, measure_balance: bool = False) -> None:
        self._check_bend_health()
        self._diagnostic_stamp = None
        self._ensure_topology_current()
        if type(measure_balance) is not bool:
            raise ValueError("measure_balance must be bool")
        if measure_balance:
            self._prepare_projection_balance(dt, pin_targets)
        self._update_pin_targets(pin_targets)
        self._run_substep(dt, iterations, convergence=convergence, measure_balance=measure_balance)
        self._sync_particle_state()
        self._mark_completed_substep(dt)
        self._balance_completed = measure_balance

    def _prepare_projection_balance(self, dt, pin_targets):
        if self.backend != "cpu" or self.precision != "f64" or dt not in (0.001, 0.0005):
            raise ValueError("projection balance requires CPU f64 and a 1 ms or 0.5 ms substep")
        if self.material.pin_compliance != 0:
            raise ValueError("projection balance requires hard pins")
        device = self.read_device_particles()
        for pin in self.pin_constraints:
            target = pin.target if pin_targets is None else pin_targets.get(pin.particle, pin.target)
            if (not np.array_equal(target, device["positions"][pin.particle])
                    or np.any(device["velocity"][pin.particle] != 0)):
                raise ValueError("projection balance requires stationary hard pins")
        if self._balance_buffers is None:
            self._balance_buffers = ti.Vector.field(3, dtype=ti.f64, shape=(5, self.vertex_capacity))
        self._balance_buffers.fill(0)

    def read_projection_balance(self) -> dict[str, np.ndarray]:
        """Per-active-row projection J, world-origin moment of J, and damping J.

        Both endpoints (all four for a hinge) contribute before inverse-mass
        masking. Moments use each endpoint's predicted position at projection.
        Damping uses the physical device mass and the actual velocity update.
        """
        n = self._bridge_count()
        if (not self._balance_completed or self._diagnostic_stamp is None or
                self._diagnostic_stamp[:2] != (self.repack_count, self.coloring_version)):
            raise RuntimeError("projection balance requires a completed measured substep")
        buffers = self._balance_buffers.to_numpy()[:, :n]
        if not np.isfinite(buffers).all():
            raise ValueError("nonfinite measured projection balance")
        keys = ("stretch_impulse", "bend_impulse", "stretch_angular_impulse",
                "bend_angular_impulse", "damping_impulse")
        result = {name: buffers[i].copy() for i, name in enumerate(keys)}
        for value in result.values():
            value.setflags(write=False)
        return result

    # ------------------------------------------------------------------
    # Optional heavy-load reference diagnostics.
    # Diagnostic and opt-in: substep()/_run_substep() above are not changed.

    def set_fixed_constraint_tolerance(self, value) -> None:
        """Rounding allowance for fully pinned constraints in convergence measurement.

        Zero (the default) keeps the pre-existing exact kernel. A positive value
        switches measurement to a variant that rejects a fully pinned stretch
        constraint only when |length - rest| > value * rest, and a fully pinned
        hinge only when |c| > value [rad]. It is a rounding allowance, not a strain
        budget: values above 1e-12 are refused.
        """
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0.0 <= value <= 1.0e-12):
            raise ValueError("fixed constraint tolerance must be a finite number in [0, 1e-12]")
        if value > 0.0 and self._fixed_tolerance_field is None:
            scalar_type = ti.f32 if self.precision == "f32" else ti.f64
            self._fixed_tolerance_field = ti.field(dtype=scalar_type, shape=())
            self._fixed_counts = ti.Vector.field(2, dtype=ti.i32, shape=())
            self._fixed_maxima = ti.Vector.field(2, dtype=scalar_type, shape=())
        if self._fixed_tolerance_field is not None:
            self._fixed_tolerance_field[None] = value
        self._fixed_tolerance = float(value)

    @property
    def fixed_constraint_tolerance(self) -> float:
        """The rounding allowance in force; zero means the pre-existing exact rule."""
        return self._fixed_tolerance

    def read_fixed_constraint_statistics(self) -> dict:
        """Counts and largest constraint values of fully pinned rows at the last tolerant measurement."""
        if self._fixed_tolerance <= 0.0:
            raise RuntimeError("fixed constraint statistics exist only with a positive tolerance")
        counts, maxima = self._fixed_counts[None], self._fixed_maxima[None]
        return dict(tolerance=self._fixed_tolerance, stretch_count=int(counts[0]), bend_count=int(counts[1]),
                    stretch_max_relative=float(maxima[0]), bend_max_abs_rad=float(maxima[1]))

    def _raise_bend_failure(self) -> None:
        slot = int(self.bend_failure_slot[None])
        if slot:
            code = int(self.bend_failure_code[slot - 1])
            self._bend_failed = (f"dihedral failure at device hinge slot {slot - 1}, code {code} "
                                 "(1=geometry, 2=angle branch, 3=numeric update); "
                                 "state may be partial; construct a new solver to recover")
            raise RuntimeError(self._bend_failed)

    def _capture_reference_state(self, slot: int) -> None:
        # Same two checks the existing entry makes when it stops at this round.
        self._check_dihedral_commit()
        self._raise_bend_failure()
        self._copy_reference_state(slot)

    def substep_reference(self, dt: float, *, policy: dict, should_abort=None) -> None:
        """One same-h reference window: a single predict/clear/commit, bounded extra projection.

        Projection order, material and omega are those of substep(..., convergence=True,
        measure_balance=True). The window keeps the budget checkpoint (first target
        round, or the checkpoint round when the target comes later or never), the
        first-target state when it is later than the checkpoint, and commits the
        state at the end of the tail or at the budget. See solver_convergence for
        the round accounting; gates on the result are the caller's business.
        ``should_abort`` is polled after residual checks; a true value abandons the
        window uncommitted and leaves this solver unusable.
        """
        self._check_bend_health()
        self._diagnostic_stamp = None
        self._ensure_topology_current()
        policy = validate_reference_policy(policy, self.precision)
        if should_abort is not None and not callable(should_abort):
            raise TypeError("should_abort must be callable")
        if (self.backend != "cpu" or self._bend_model != "dihedral"
                or not math.isfinite(self.material.stretch_compliance) or self.material.stretch_compliance < 0):
            raise ValueError("the reference window requires CPU dihedral and nonnegative stretch compliance")
        device_dt = float(self._state_dtype(dt))
        if not math.isfinite(device_dt) or device_dt <= 0 or device_dt**2 < 1e-16:
            raise ValueError("reference duration must be representable with h squared >= 1e-16")
        if not self.stretch_color_groups and not self.bend_color_groups:
            raise ValueError("the reference window requires active constraints")
        if not np.any(self.particles.inverse_mass > 0):
            raise ValueError("the reference window requires a free particle")
        # CPU f64, a supported measured step, hard stationary pins; zeroes the five balance buffers.
        self._prepare_projection_balance(dt, None)
        if self._reference_state is None:
            self._reference_state = ti.Vector.field(9, dtype=ti.f64, shape=(2, self.vertex_capacity))
            self._reference_balance = ti.Vector.field(3, dtype=ti.f64, shape=(2, 4, self.vertex_capacity))
            self._reference_stretch_lambda = ti.field(dtype=ti.f64, shape=(2, self.stretch_capacity))
            self._reference_bend_lambda = ti.field(dtype=ti.f64, shape=(2, self.bend_capacity))
        self._update_pin_targets(None)
        self._balance_completed = False
        self.last_convergence = None
        self.last_reference = None
        self.last_reference_timing = None
        self._reference_slots = None
        started = time.perf_counter()
        measure_s = capture_s = 0.0
        self.relaxation_value[None] = policy["omega"]
        self.bend_failure_slot[None] = 0
        self.dt_value[None] = dt
        self._predict()
        self._clear_lambdas()
        checks, first, slots = [], None, {}
        stop_at, iteration = policy["budget"], 0
        while iteration < stop_at:
            iteration += 1
            for start, end in self.stretch_color_groups:
                self._solve_distance_color(start, end, 0, True)
            for start, end in self.bend_color_groups:
                self._solve_dihedral_color(start, end, True)
            self._solve_pins()
            if iteration % policy["check_every"]:
                continue
            clock = time.perf_counter()
            self._measure_convergence()
            if self.convergence_invalid[None]:
                self._bend_failed = "invalid geometry/numerics or strained fixed constraint in convergence measurement"
                raise RuntimeError(self._bend_failed)
            self._raise_bend_failure()
            residual = self.convergence_residuals[None]
            checks.append(dict(iteration=iteration, stretch_residual=float(residual[0]),
                               bend_residual=float(residual[1])))
            measure_s += time.perf_counter() - clock
            clock = time.perf_counter()
            if first is None and stop_reason(checks, policy) is not None:
                first = iteration
                stop_at = reference_stop_iteration(first, policy)
                if first > policy["checkpoint"]:
                    self._capture_reference_state(1)
                    slots["first_target"] = (1, iteration)
            if "checkpoint" not in slots and (first is not None or iteration == policy["checkpoint"]):
                self._capture_reference_state(0)
                slots["checkpoint"] = (0, iteration)
                if first is not None:
                    slots["first_target"] = (0, iteration)
            capture_s += time.perf_counter() - clock
            if iteration < stop_at and should_abort is not None and should_abort():
                self._bend_failed = ("reference window aborted before commit; state is partial; "
                                     "construct a new solver to recover")
                raise ReferenceAborted(self._bend_failed)
        self._check_dihedral_commit()
        self._raise_bend_failure()
        self._finish_substep(True)
        record = validate_reference_record(reference_record_from_checks(checks, policy), self.precision, policy)
        if (slots["checkpoint"][1] != record["checkpoint"]["iteration"]
                or ("first_target" in slots) != (first is not None)
                or (first is not None and slots["first_target"][1] != first)):
            self._bend_failed = "reference window captured states at the wrong rounds"
            raise RuntimeError(self._bend_failed)
        self._sync_particle_state()
        self._mark_completed_substep(dt)
        self._balance_completed = True
        self._reference_slots = slots
        self.last_reference = record
        total = time.perf_counter() - started
        self.last_reference_timing = dict(total_s=total, measurement_s=measure_s, capture_s=capture_s,
                                          projection_and_commit_s=total - measure_s - capture_s)

    def read_reference_states(self) -> dict:
        """Candidate states kept by the last reference window, as owned f64 arrays.

        ``checkpoint`` is the budget checkpoint; ``first_target`` is the state at
        the first target round (the same object when that round is not later than
        the checkpoint, None when the target was never reached). Each holds the
        candidate positions, velocity and damping impulse, the four projection
        accumulations as they were at that round, and both multiplier arrays.
        """
        n = self._bridge_count()
        if (self.last_reference is None or not self._balance_completed or self._diagnostic_stamp is None
                or self._diagnostic_stamp[:2] != (self.repack_count, self.coloring_version)):
            raise RuntimeError("reference states require a completed reference window")
        stretch_count, bend_count = int(self.active_stretch_count[None]), int(self.active_bend_count[None])
        state, balance = self._reference_state.to_numpy(), self._reference_balance.to_numpy()
        stretch = self._reference_stretch_lambda.to_numpy()
        bend = self._reference_bend_lambda.to_numpy()
        names = ("stretch_impulse", "bend_impulse", "stretch_angular_impulse", "bend_angular_impulse")
        cache, result = {}, {}
        for key in ("checkpoint", "first_target"):
            if key not in self._reference_slots:
                result[key] = None
                continue
            slot, iteration = self._reference_slots[key]
            if slot not in cache:
                item = dict(iteration=iteration, positions=state[slot, :n, 0:3].copy(),
                            velocity=state[slot, :n, 3:6].copy(), damping_impulse=state[slot, :n, 6:9].copy(),
                            stretch_lambda=stretch[slot, :stretch_count].copy(),
                            bend_lambda=bend[slot, :bend_count].copy())
                item.update({name: balance[slot, kind, :n].copy() for kind, name in enumerate(names)})
                for value in item.values():
                    if isinstance(value, np.ndarray):
                        if not np.isfinite(value).all():
                            raise ValueError("nonfinite reference state")
                        value.setflags(write=False)
                cache[slot] = item
            result[key] = cache[slot]
        return result

    def _ensure_topology_current(self) -> None:
        if self.topology_version != getattr(self.mesh, "topology_version", 0):
            self.rebuild_constraints()

    def _repack_after_topology_change(self) -> None:
        """Repack indices and recolor both constraint sets at one mesh version."""
        self._diagnostic_stamp = None
        self._bridge_mass_generation = None
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
        if self._bend_model == "dihedral":
            self.bend_color_groups = self._upload_dihedral_constraints()
        else:
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
    def _color_constraints(constraints) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Stable greedy coloring of all vertices, including pinned vertices."""
        colors: list[list[tuple[int, ...]]] = []
        occupied: list[set[int]] = []
        for constraint in constraints:
            pair = constraint.vertices
            vertices = set(pair)
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
    def _validate_bend_material(material, precision) -> None:
        if material.bend_model not in ("distance", "dihedral"):
            raise ValueError("bend_model must be distance or dihedral")
        if material.bend_model == "distance":
            if material.bend_stiffness is not None:
                raise ValueError("bend_stiffness requires bend_model='dihedral'")
        else:
            b = material.bend_stiffness
            limit = np.finfo(np.float32 if precision == "f32" else np.float64)
            if b is None or not math.isfinite(float(b)) or not limit.tiny <= b <= limit.max:
                raise ValueError("dihedral bend_stiffness must be finite, positive and representable")

    def _validate_hinge_stiffness(self, stiffness) -> None:
        weights = np.array([c.reference_weight for c in self.bend_constraints], dtype=np.float64)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            k = (float(stiffness) * weights).astype(self._state_dtype)
            alpha = (1.0 / k).astype(self._state_dtype)
        if (not np.isfinite(k).all() or np.any(k <= 0)
                or not np.isfinite(alpha).all() or np.any(alpha <= 0)):
            raise ValueError("dihedral stiffness/compliance is not representable for reference hinges")

    def _upload_dihedral_constraints(self) -> list[tuple[int, int]]:
        self._check_capacity("bend constraints", len(self.bend_constraints), self.bend_capacity)
        self._validate_hinge_stiffness(self.material.bend_stiffness)
        lookup = {c.vertices: c for c in self.bend_constraints}
        ordered, groups = [], []
        for color in self.bend_colors:
            start = len(ordered)
            ordered.extend(lookup[vertices] for vertices in color)
            groups.append((start, len(ordered)))
        rows = np.zeros((self.bend_capacity, 4), dtype=np.int32)
        rest = np.zeros(self.bend_capacity, dtype=self._state_dtype)
        weight = np.zeros_like(rest)
        for i, c in enumerate(ordered):
            rows[i], rest[i], weight[i] = c.vertices, c.rest_angle, c.reference_weight
        for j, field in enumerate((self.bend_a, self.bend_b, self.bend_c, self.bend_d)):
            field.from_numpy(rows[:, j].copy())
        self.bend_rest.from_numpy(rest)
        self.bend_weight.from_numpy(weight)
        self.active_bend_count[None] = len(ordered)
        return groups

    def _check_bend_health(self) -> None:
        if self._bend_failed is not None:
            raise RuntimeError(self._bend_failed)

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

    def _run_substep(self, dt: float, iterations: int, *, convergence: bool = False,
                     measure_balance: bool = False) -> None:
        self._diagnostic_stamp = None
        self._balance_completed = False
        self.last_convergence = None
        self._check_bend_health()
        if type(convergence) is not bool:
            raise ValueError("convergence must be bool")
        policy = convergence_policy(self.precision) if convergence else None
        if convergence:
            if (self.backend != "cpu" or self._bend_model != "dihedral"
                    or self.material.pin_compliance != 0
                    or not math.isfinite(self.material.stretch_compliance) or self.material.stretch_compliance < 0):
                raise ValueError("convergence requires CPU dihedral, hard pins and nonnegative stretch compliance")
            if type(iterations) is not int or not policy["minimum"] <= iterations <= policy["maximum"]:
                raise ValueError("convergence iteration cap must be an integer in 8..96")
            device_dt = float(self._state_dtype(dt))
            if not math.isfinite(device_dt) or device_dt <= 0 or device_dt**2 < 1e-16:
                raise ValueError("convergence duration must be representable with h squared >= 1e-16")
            if not self.stretch_color_groups and not self.bend_color_groups:
                raise ValueError("convergence requires active constraints")
            if not np.any(self.particles.inverse_mass > 0):
                raise ValueError("convergence requires a free particle")
        self.relaxation_value[None] = policy["omega"] if policy else 1.
        if self._bend_model == "dihedral":
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("dihedral substep duration must be finite and positive")
            self.bend_failure_slot[None] = 0
        self.dt_value[None] = dt
        self._predict()
        self._clear_lambdas()
        checks = []
        for iteration in range(1, max(1, int(iterations))+1):
            for start, end in self.stretch_color_groups:
                self._solve_distance_color(start, end, 0, measure_balance)
            for start, end in self.bend_color_groups:
                if self._bend_model == "dihedral":
                    self._solve_dihedral_color(start, end, measure_balance)
                else:
                    self._solve_distance_color(start, end, 1, measure_balance)
            self._solve_pins()
            if convergence and (iteration % policy["check_every"] == 0 or iteration == iterations):
                self._measure_convergence()
                if self.convergence_invalid[None]:
                    self._bend_failed = "invalid geometry/numerics or strained fixed constraint in convergence measurement"
                    raise RuntimeError(self._bend_failed)
                residual = self.convergence_residuals[None]
                checks.append(dict(iteration=iteration, stretch_residual=float(residual[0]),
                                   bend_residual=float(residual[1])))
                if stop_reason(checks, policy) is not None:
                    break
        if self._bend_model == "dihedral":
            self._check_dihedral_commit()
            slot = int(self.bend_failure_slot[None])
            if slot:
                code = int(self.bend_failure_code[slot - 1])
                self._bend_failed = (f"dihedral failure at device hinge slot {slot - 1}, code {code} "
                                     "(1=geometry, 2=angle branch, 3=numeric update); "
                                     "state may be partial; construct a new solver to recover")
                raise RuntimeError(self._bend_failed)
        self._finish_substep(measure_balance)
        if convergence:
            self.last_convergence = validate_convergence_record(
                dict(schema="xpbd-convergence-result-v1", iterations=iteration,
                     stop_reason=stop_reason(checks, policy) or "iteration_cap", checks=checks),
                self.precision, iterations)

    def _mark_completed_substep(self, h: float) -> None:
        self._diagnostic_stamp = (self.repack_count, self.coloring_version, float(h))

    def _bridge_count(self) -> int:
        self._check_bend_health()
        count = self.particles.count
        self._check_capacity("vertices", count, self.vertex_capacity)
        if (self.mesh.topology_version != self.topology_version
                or self.coloring_version != self.topology_version
                or self.active_particle_count[None] != count):
            raise RuntimeError("host bridge requires a rebuilt, aligned topology")
        return count

    @staticmethod
    def _host_real_array(value, shape, name):
        raw = np.asarray(value)
        if raw.shape != shape or raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a real array of shape {shape}")
        array = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array

    def upload_mass_state(self, mass, inverse_mass, pinned) -> None:
        """Opt-in host bridge, with a device mass mirror for independent audits.

        No kernel consumes the mirror; physical mass and inverse mass are both
        uploaded in the solver's actual precision, including physical pin mass.
        Pin constraints must already match the new topology before this call.
        """
        n = self._bridge_count()
        mass = self._host_real_array(mass, (n,), "mass")
        inv = self._host_real_array(inverse_mass, (n,), "inverse_mass")
        pin = np.asarray(pinned)
        if pin.shape != (n,) or pin.dtype.kind != "b":
            raise ValueError("pinned must be a boolean active-row array")
        if np.any(mass <= 0) or np.any(inv < 0):
            raise ValueError("mass must be positive; inverse_mass nonnegative")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            expected = np.where(pin, 0., 1. / mass)
        if not np.isfinite(expected).all() or not np.allclose(inv, expected, rtol=1e-12, atol=0):
            raise ValueError("inverse_mass must be zero for pins and 1/mass otherwise")
        actual_pin = self.pin_index_for_particle.to_numpy()[:n] >= 0
        if not np.array_equal(pin, actual_pin) or not np.array_equal(pin, self.particles.pinned):
            raise ValueError("pinned flags disagree with active device pin constraints")
        masses = np.zeros(self.vertex_capacity, dtype=self._state_dtype)
        inverses = np.zeros_like(masses)
        with np.errstate(over="ignore", under="ignore"):
            masses[:n], inverses[:n] = mass, inv
        if (not np.isfinite(masses).all() or not np.isfinite(inverses).all()
                or np.any(masses[:n] <= 0) or np.any(inverses[:n][~pin] <= 0)):
            raise ValueError("mass/inverse_mass is not representable in device precision")
        self._diagnostic_stamp = None
        self._bridge_mass_generation = None
        if self._bridge_mass_field is None:
            scalar_type = ti.f32 if self.precision == "f32" else ti.f64
            self._bridge_mass_field = ti.field(dtype=scalar_type, shape=self.vertex_capacity)
        self._bridge_mass_field.from_numpy(masses)
        self.inverse_mass_field.from_numpy(inverses)
        self.particles.mass[:] = mass
        self.particles.inverse_mass[:] = inv
        self._bridge_mass_generation = (self.repack_count, n)

    def apply_impulses(self, rows, impulse) -> None:
        """Add J*device_inverse_mass to device velocity, without repacking.

        Rows are active device slots resolved by the caller from stable IDs.
        The caller owns the once-only load ledger and aborts on upload failure.
        """
        n = self._bridge_count()
        if self._bridge_mass_generation != (self.repack_count, n):
            raise RuntimeError("upload aligned mass state before applying impulses")
        rows = np.asarray(rows)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
                or np.any(rows >= n) or len(np.unique(rows)) != len(rows)):
            raise ValueError("impulse rows must be unique active integer slots")
        rows = rows.astype(np.int64)
        impulse = self._host_real_array(impulse, (len(rows), 3), "impulse")
        state = self.state.to_numpy()
        inv = self.inverse_mass_field.to_numpy()[:n].astype(np.float64)
        if not np.isfinite(state[:n]).all() or not np.isfinite(inv).all():
            raise ValueError("nonfinite device state before impulse upload")
        velocity = state[rows, 9:12].astype(np.float64) + impulse * inv[rows, None]
        with np.errstate(over="ignore"):
            converted = velocity.astype(self._state_dtype)
        if not np.isfinite(converted).all():
            raise ValueError("impulse would overflow device velocity")
        self._diagnostic_stamp = None
        state[rows, 9:12] = converted
        self.state.from_numpy(state)
        self.particles.velocity[:] = state[:n, 9:12]

    def read_device_particles(self) -> dict[str, np.ndarray]:
        """Read active device buffers, convert to f64, and return owned arrays."""
        n = self._bridge_count()
        if self._bridge_mass_generation != (self.repack_count, n):
            raise RuntimeError("device mass mirror is stale or not initialized")
        state = self.state.to_numpy()[:n].astype(np.float64)
        result = dict(positions=state[:, :3].copy(), velocity=state[:, 9:12].copy(),
                      mass=self._bridge_mass_field.to_numpy()[:n].astype(np.float64),
                      inverse_mass=self.inverse_mass_field.to_numpy()[:n].astype(np.float64),
                      pinned=self.pin_index_for_particle.to_numpy()[:n] >= 0)
        for value in result.values():
            if not np.isfinite(value).all():
                raise ValueError("nonfinite device readback")
            value.setflags(write=False)
        return result

    def synchronize_host_state(self) -> None:
        """Read back all state slots after initialization or a topology upload."""
        self._bridge_count()
        self._sync_particle_state()

    def constraint_diagnostics(self, kind: str, vertex_ids=None) -> dict:
        """Export active device slots after a successful public solve.

        This raw array dictionary has no coupling types or driver clock. The
        driver adds its stream, completed version and time before packaging.
        """
        n = self._bridge_count()
        stamp = self._diagnostic_stamp
        if stamp is None or stamp[:2] != (self.repack_count, self.coloring_version):
            raise RuntimeError("diagnostics require a completed solve since the last mutation/repack")
        ids = np.arange(n, dtype=np.int64) if vertex_ids is None else np.asarray(vertex_ids)
        if (ids.shape != (n,) or ids.dtype.kind not in "iu" or np.any(ids < 0)
                or np.any(ids > np.iinfo(np.int64).max) or len(np.unique(ids)) != n):
            raise ValueError("vertex_ids must be unique nonnegative int64 identities")
        ids = ids.astype(np.int64)
        if kind in ("stretch", "bend"):
            count = int(getattr(self, f"active_{kind}_count")[None])
            a = getattr(self, f"{kind}_a").to_numpy()[:count]
            b = getattr(self, f"{kind}_b").to_numpy()[:count]
            rows = np.column_stack((a, b))
            if kind == "bend" and self._bend_model == "dihedral":
                rows = np.column_stack((a, b, self.bend_c.to_numpy()[:count],
                                        self.bend_d.to_numpy()[:count]))
            rest = getattr(self, f"{kind}_rest").to_numpy()[:count].astype(np.float64)
            lambdas = getattr(self, f"{kind}_lambda").to_numpy()[:count].astype(np.float64)
            groups = getattr(self, f"{kind}_color_groups")
            offsets = np.array([0] + [end for _, end in groups], dtype=np.int64)
        elif kind == "pin":
            count = int(self.active_pin_count[None])
            rows = self.pin_particle.to_numpy()[:count, None]
            rest = self.pin_target.to_numpy()[:count].astype(np.float64)
            lambdas = self.pin_lambda.to_numpy()[:count].astype(np.float64)
            offsets = np.array([0, count] if count else [0], dtype=np.int64)
        else:
            raise ValueError("unknown constraint kind")
        if np.any(rows < 0) or np.any(rows >= n):
            raise RuntimeError("device constraint references an inactive vertex")
        result = dict(kind=kind, endpoint_ids=ids[rows], rest=rest, lambdas=lambdas,
                      color_offsets=offsets, h=stamp[2], precision=self.precision,
                      backend=self.arch, coloring_version=self.coloring_version,
                      repack_count=self.repack_count, active_count=count,
                      export_stage="post_solve_pre_tear")
        if kind == "bend" and self._bend_model == "dihedral":
            weights = self.bend_weight.to_numpy()[:count].astype(np.float64)
            k_device = self._state_dtype(self.bend_stiffness_value[None]) * weights.astype(self._state_dtype)
            k = k_device.astype(np.float64)
            result.update(model="dihedral", angle_units="rad", stiffness_units="N m",
                          compliance_units="1/(N m)", reference_weights=weights,
                          stiffness=k, compliance=(1.0 / k_device).astype(np.float64),
                          angles=self.bend_angle.to_numpy()[:count].astype(np.float64),
                          errors=self.bend_error.to_numpy()[:count].astype(np.float64),
                          energy=self.bend_energy.to_numpy()[:count].astype(np.float64),
                          geometry_stage="final_pin_targets_before_commit",
                          lambda_stage="last_bend_projection")
        for value in result.values():
            if isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite device constraint diagnostic")
                value.setflags(write=False)
        return result

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
            if ti.static(self._bend_model == "dihedral"):
                self.bend_failure_code[i] = 0
        for i in range(self.active_pin_count[None]):
            self.pin_lambda[i][0] = 0.0
            self.pin_lambda[i][1] = 0.0
            self.pin_lambda[i][2] = 0.0

    @ti.func
    def _record_projection(self, kind, vertex, impulse):
        position = self._bend_position(vertex, False)
        self._balance_buffers[kind, vertex] += impulse
        self._balance_buffers[2 + kind, vertex] += position.cross(impulse)

    def _solve_dihedral_color(self, start, end, measure_balance=False):
        self._project_dihedral_color(start, end, measure_balance)

    @ti.kernel
    def _project_dihedral_color(self, start: ti.i32, end: ti.i32, measure_balance: ti.template()):
        for index in range(start, end):
            ids = ti.Vector([self.bend_a[index], self.bend_b[index],
                             self.bend_c[index], self.bend_d[index]])
            angle, gradient, valid = dihedral_geometry(
                self._bend_position(ids[0], False), self._bend_position(ids[1], False),
                self._bend_position(ids[2], False), self._bend_position(ids[3], False),
                self._bend_geometry_tolerance)
            value = wrapped_angle(angle - self.bend_rest[index])
            code = 0
            if not valid:
                code = 1
            elif ti.abs(value) >= math.pi - self._bend_branch_tolerance:
                code = 2
            else:
                h2 = self.dt_value[None] * self.dt_value[None]
                stiffness = self.bend_stiffness_value[None] * self.bend_weight[index]
                alpha = (1.0 / stiffness) / h2
                denominator = alpha
                for i, j in ti.static(ti.ndrange(4, 3)):
                    denominator += self.inverse_mass_field[ids[i]] * gradient[i, j]**2
                if (not finite_scalar(alpha) or alpha <= 0.0 or not finite_scalar(h2) or h2 <= 0.0
                        or not finite_scalar(denominator) or denominator <= 0.0):
                    code = 3
                else:
                    delta_lambda = (-value - alpha * self.bend_lambda[index]) / denominator
                    if self.relaxation_value[None] != 1.0:
                        delta_lambda *= self.relaxation_value[None]
                    multiplier = self.bend_lambda[index] + delta_lambda
                    update = gradient * delta_lambda
                    if not finite_scalar(multiplier):
                        code = 3
                    for i, j in ti.static(ti.ndrange(4, 3)):
                        update[i, j] *= self.inverse_mass_field[ids[i]]
                        if not finite_scalar(self.state[ids[i]][6 + j] + update[i, j]):
                            code = 3
                    # Validate the whole four-point update before writing any of it.
                    if code == 0:
                        self.bend_lambda[index] = multiplier
                        if ti.static(measure_balance):
                            for i in ti.static(range(4)):
                                impulse = ti.Vector([gradient[i, j] for j in ti.static(range(3))])
                                self._record_projection(1, ids[i], impulse * delta_lambda / self.dt_value[None])
                        for i, j in ti.static(ti.ndrange(4, 3)):
                            self.state[ids[i]][6 + j] += update[i, j]
            if code:
                self.bend_failure_code[index] = code
                ti.atomic_max(self.bend_failure_slot[None], index + 1)

    @ti.func
    def _bend_position(self, vertex, committing: ti.template()):
        position = ti.Vector([self.state[vertex][6], self.state[vertex][7], self.state[vertex][8]])
        if ti.static(committing):
            # _finish_substep commits pin targets even for compliant pins.
            pin = self.pin_index_for_particle[vertex]
            if pin >= 0:
                position = self.pin_target[pin]
        return position

    @ti.kernel
    def _check_dihedral_commit(self):
        for index in range(self.active_bend_count[None]):
            angle, gradient, valid = dihedral_geometry(
                self._bend_position(self.bend_a[index], True), self._bend_position(self.bend_b[index], True),
                self._bend_position(self.bend_c[index], True), self._bend_position(self.bend_d[index], True),
                self._bend_geometry_tolerance)
            value = wrapped_angle(angle - self.bend_rest[index])
            energy = 0.5 * self.bend_stiffness_value[None] * self.bend_weight[index] * value * value
            code = 0
            if not valid:
                code = 1
            elif ti.abs(value) >= math.pi - self._bend_branch_tolerance:
                code = 2
            elif not finite_scalar(energy):
                code = 3
            if code:
                self.bend_failure_code[index] = code
                ti.atomic_max(self.bend_failure_slot[None], index + 1)
            self.bend_angle[index] = angle
            self.bend_error[index] = value
            self.bend_energy[index] = energy

    def _solve_distance_color(self, start, end, kind, measure_balance=False):
        self._project_distance_color(start, end, kind, measure_balance)

    @ti.kernel
    def _project_distance_color(self, start: ti.i32, end: ti.i32, kind: ti.i32, measure_balance: ti.template()):
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
                    if self.relaxation_value[None] != 1.0:
                        delta_lambda *= self.relaxation_value[None]
                    if kind == 0:
                        self.stretch_lambda[index] += delta_lambda
                    else:
                        self.bend_lambda[index] += delta_lambda
                    scale = delta_lambda / length
                    if ti.static(measure_balance):
                        impulse = scale * ti.Vector([dx, dy, dz]) / self.dt_value[None]
                        self._record_projection(kind, a, impulse)
                        self._record_projection(kind, b, -impulse)
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

    def _measure_convergence(self):
        # The exact kernel below is the pre-existing one, body unchanged. The tolerant
        # variant runs only after set_fixed_constraint_tolerance(value > 0).
        if self._fixed_tolerance > 0.0:
            self._measure_convergence_tolerant()
        else:
            self._measure_convergence_exact()

    @ti.kernel
    def _measure_convergence_exact(self):
        self.convergence_residuals[None] = ti.Vector([0.0, 0.0])
        self.convergence_invalid[None] = 0
        h2 = self.dt_value[None] * self.dt_value[None]
        for index in range(self.active_stretch_count[None]):
            a, b = self.stretch_a[index], self.stretch_b[index]
            delta = self._bend_position(a, False) - self._bend_position(b, False)
            length = delta.norm()
            rest = self.stretch_rest[index]
            if length <= 1.0e-12 or rest <= 1.0e-12 or not finite_scalar(length):
                ti.atomic_max(self.convergence_invalid[None], 1)
            else:
                # This experimental mode excludes kinematically strained,
                # fully fixed constraints: lambda alone can cancel C there.
                if self.inverse_mass_field[a] + self.inverse_mass_field[b] == 0.0 and length != rest:
                    ti.atomic_max(self.convergence_invalid[None], 1)
                r = ti.abs(length-rest+self.stretch_compliance_value[None]/h2*self.stretch_lambda[index])/rest
                if not finite_scalar(r):
                    ti.atomic_max(self.convergence_invalid[None], 1)
                else:
                    ti.atomic_max(self.convergence_residuals[None][0], r)
        if ti.static(self._bend_model == "dihedral"):
            for index in range(self.active_bend_count[None]):
                angle, gradient, valid = dihedral_geometry(
                    self._bend_position(self.bend_a[index], False), self._bend_position(self.bend_b[index], False),
                    self._bend_position(self.bend_c[index], False), self._bend_position(self.bend_d[index], False),
                    self._bend_geometry_tolerance)
                c = wrapped_angle(angle-self.bend_rest[index])
                r = ti.abs(c + self.bend_lambda[index]/(self.bend_stiffness_value[None]*self.bend_weight[index]*h2))
                movable = (self.inverse_mass_field[self.bend_a[index]] + self.inverse_mass_field[self.bend_b[index]]
                           + self.inverse_mass_field[self.bend_c[index]] + self.inverse_mass_field[self.bend_d[index]])
                if movable == 0.0 and c != 0.0:
                    ti.atomic_max(self.convergence_invalid[None], 1)
                if not valid or ti.abs(c) >= math.pi-self._bend_branch_tolerance or not finite_scalar(r):
                    ti.atomic_max(self.convergence_invalid[None], 1)
                else:
                    ti.atomic_max(self.convergence_residuals[None][1], r)

    @ti.kernel
    def _measure_convergence_tolerant(self):
        # Copy of _measure_convergence_exact with one difference: a fully pinned
        # constraint is rejected only when its constraint value exceeds a rounding
        # allowance (stretch: |length-rest| > tol*rest; bend: |c| > tol). Host rest
        # lengths and device lengths of pinned diagonals can differ in the last bit.
        self.convergence_residuals[None] = ti.Vector([0.0, 0.0])
        self.convergence_invalid[None] = 0
        self._fixed_counts[None] = ti.Vector([0, 0])
        self._fixed_maxima[None] = ti.Vector([0.0, 0.0])
        h2 = self.dt_value[None] * self.dt_value[None]
        for index in range(self.active_stretch_count[None]):
            a, b = self.stretch_a[index], self.stretch_b[index]
            delta = self._bend_position(a, False) - self._bend_position(b, False)
            length = delta.norm()
            rest = self.stretch_rest[index]
            if length <= 1.0e-12 or rest <= 1.0e-12 or not finite_scalar(length):
                ti.atomic_max(self.convergence_invalid[None], 1)
            else:
                if self.inverse_mass_field[a] + self.inverse_mass_field[b] == 0.0:
                    ti.atomic_add(self._fixed_counts[None][0], 1)
                    ti.atomic_max(self._fixed_maxima[None][0], ti.abs(length - rest) / rest)
                    if ti.abs(length - rest) > self._fixed_tolerance_field[None] * rest:
                        ti.atomic_max(self.convergence_invalid[None], 1)
                r = ti.abs(length-rest+self.stretch_compliance_value[None]/h2*self.stretch_lambda[index])/rest
                if not finite_scalar(r):
                    ti.atomic_max(self.convergence_invalid[None], 1)
                else:
                    ti.atomic_max(self.convergence_residuals[None][0], r)
        if ti.static(self._bend_model == "dihedral"):
            for index in range(self.active_bend_count[None]):
                angle, gradient, valid = dihedral_geometry(
                    self._bend_position(self.bend_a[index], False), self._bend_position(self.bend_b[index], False),
                    self._bend_position(self.bend_c[index], False), self._bend_position(self.bend_d[index], False),
                    self._bend_geometry_tolerance)
                c = wrapped_angle(angle-self.bend_rest[index])
                r = ti.abs(c + self.bend_lambda[index]/(self.bend_stiffness_value[None]*self.bend_weight[index]*h2))
                movable = (self.inverse_mass_field[self.bend_a[index]] + self.inverse_mass_field[self.bend_b[index]]
                           + self.inverse_mass_field[self.bend_c[index]] + self.inverse_mass_field[self.bend_d[index]])
                if movable == 0.0:
                    ti.atomic_add(self._fixed_counts[None][1], 1)
                    ti.atomic_max(self._fixed_maxima[None][1], ti.abs(c))
                    if ti.abs(c) > self._fixed_tolerance_field[None]:
                        ti.atomic_max(self.convergence_invalid[None], 1)
                if not valid or ti.abs(c) >= math.pi-self._bend_branch_tolerance or not finite_scalar(r):
                    ti.atomic_max(self.convergence_invalid[None], 1)
                else:
                    ti.atomic_max(self.convergence_residuals[None][1], r)

    @ti.kernel
    def _copy_reference_state(self, slot: ti.i32):
        # Candidate state of the current projection round, without committing it. The
        # velocity and damping expressions repeat _commit_substep (divide, then damp).
        for i in range(self.active_particle_count[None]):
            pin_index = self.pin_index_for_particle[i]
            for axis in ti.static(range(3)):
                self._reference_state[slot, i][3 + axis] = 0.0
                self._reference_state[slot, i][6 + axis] = 0.0
                if pin_index >= 0:
                    self._reference_state[slot, i][axis] = self.pin_target[pin_index][axis]
                else:
                    self._reference_state[slot, i][axis] = self.state[i][6 + axis]
                    if self.inverse_mass_field[i] > 0.0:
                        self._reference_state[slot, i][3 + axis] = (self.state[i][6 + axis] - self.state[i][3 + axis]) / ti.max(self.dt_value[None], 1.0e-12)
                        before = self._reference_state[slot, i][3 + axis]
                        self._reference_state[slot, i][3 + axis] *= self.damping_value[None]
                        self._reference_state[slot, i][6 + axis] = self._bridge_mass_field[i] * (
                            self._reference_state[slot, i][3 + axis] - before)
            for kind in ti.static(range(4)):
                self._reference_balance[slot, kind, i] = self._balance_buffers[kind, i]
        for i in range(self.active_stretch_count[None]):
            self._reference_stretch_lambda[slot, i] = self.stretch_lambda[i]
        for i in range(self.active_bend_count[None]):
            self._reference_bend_lambda[slot, i] = self.bend_lambda[i]

    def _finish_substep(self, measure_balance=False):
        self._commit_substep(measure_balance)

    @ti.kernel
    def _commit_substep(self, measure_balance: ti.template()):
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
                        if ti.static(measure_balance):
                            before = self.state[i][9 + axis]
                            self.state[i][9 + axis] *= self.damping_value[None]
                            self._balance_buffers[4, i][axis] = self._bridge_mass_field[i] * (
                                self.state[i][9 + axis] - before)
                        else:
                            self.state[i][9 + axis] *= self.damping_value[None]
                    else:
                        self.state[i][9 + axis] = 0.0
