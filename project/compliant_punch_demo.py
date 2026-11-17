"""Independent SI cloth driver: prescribed compliant proxy, MPM backend empty.

Defaults are pilot inputs, not a calibrated or accepted tearing configuration.
No contact/fracture logic is imported by the cloth solver.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from .cloth.constraints import ClothMaterial, XPBDSolver
from .cloth.failure import StrainThresholdFailure
from .cloth.mesh import TriangleMesh
from .cloth.particles import ParticleSoA
from .cloth.topology import TopologyManager
from .coupling.contracts import ClothSnapshot, MappingConfig
from .coupling.loads import LoadLedger, SnapshotClock
from .coupling.provider_contracts import ConstraintDiagnostics, ReferenceState
from .coupling.providers import MotionConfig, PrescribedSphere
from .coupling.traction import contact_stability_limit, integrate_traction, make_boundary_samples
from .coupling.areal_mass import create_reference, redistribute_mass
from .coupling.mapping import map_points


LABEL = "XPBD tearing + prescribed compliant proxy; MPM backend empty"


@dataclass(frozen=True)
class PunchConfig:
    resolution_x: int = 128
    resolution_y: int = 160
    width: float = .8
    height: float = 1.
    areal_density: float = .2
    prestrain: float = .005
    dt: float = 1/60
    substeps: int = 8
    iterations: int = 10
    stretch_compliance: float = 1e-4
    bend_compliance: float = 2e-3
    gamma: float = 1.
    gravity: tuple = (0., 0., -9.81)
    stiffness: float = 1000.
    contact_damping: float = 0.
    stability_beta: float = .2
    critical_strain: float = .05
    contact_enabled: bool = True
    tearing_enabled: bool = True
    query_count: int = 64
    query_max_distance: float = .05
    output_stride: int = 3
    arch: str = "cpu"
    precision: str = "f32"
    capacity_headroom: float = 4.

    def __post_init__(self):
        for name in ("contact_enabled", "tearing_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name, minimum in (("resolution_x", 2), ("resolution_y", 2), ("substeps", 1),
                              ("iterations", 1), ("query_count", 0), ("output_stride", 1)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("width", "height", "areal_density", "dt", "critical_strain", "stability_beta",
                     "query_max_distance", "capacity_headroom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("prestrain", "gamma", "stiffness", "contact_damping", "stretch_compliance", "bend_compliance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.arch not in ("cpu", "cuda") or self.precision not in ("f32", "f64"):
            raise ValueError("unsupported explicit backend/precision")
        if self.capacity_headroom < 1 or np.shape(self.gravity) != (3,) or not np.isfinite(self.gravity).all():
            raise ValueError("invalid capacity or gravity")


class CompliantPunchSimulation:
    def __init__(self, config: PunchConfig | None = None, motion: MotionConfig | None = None):
        self.config = config or PunchConfig()
        c = self.config
        self.stream_id = str(uuid4())
        self.epoch, self.step_index, self.outer_index = 0, 0, 0
        self.time, self.last_h = 0., c.dt/c.substeps
        self.failed = False
        self.clock, self.ledger = SnapshotClock(self.stream_id), LoadLedger(self.stream_id)
        self.costs = {key: 0. for key in ("setup", "snapshot", "mass_upload", "load",
                      "load_validation_upload", "solver", "diagnostics", "mapping",
                      "boundary_packet", "tear_rebuild", "recording", "stability")}
        self.events, self.load_history = [], []
        self.last_diagnostics = None
        self.last_samples = None
        self.work = 0.
        self.minimum_stability_margin = float("inf")
        start = perf_counter()
        x = np.linspace(-c.width/2, c.width/2, c.resolution_x)
        y = np.linspace(-c.height/2, c.height/2, c.resolution_y)
        xx, yy = np.meshgrid(x, y)
        rest = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        a = (np.arange(c.resolution_y-1)[:, None]*c.resolution_x + np.arange(c.resolution_x-1)).ravel()
        triangles = np.stack((np.column_stack((a, a+1, a+c.resolution_x)),
                              np.column_stack((a+1, a+c.resolution_x+1, a+c.resolution_x))), axis=1).reshape(-1, 3)
        self.original_boundary = ((xx == x[0]) | (xx == x[-1]) | (yy == y[0]) | (yy == y[-1])).ravel()
        self.particles = ParticleSoA(rest * (1+c.prestrain))
        self.particles.pin(np.flatnonzero(self.original_boundary))
        self.mesh = TriangleMesh(triangles, rest)
        raw = self._host_snapshot()
        self.reference = create_reference(raw, rest, np.arange(len(rest)), self.original_boundary,
                                          stream_id=self.stream_id, areal_density=c.areal_density)
        self._assign_mass(redistribute_mass(raw, self.reference))
        self.initial_mass = self.particles.mass.sum(dtype=np.float64)
        self.solver = XPBDSolver(self.particles, self.mesh, self._material(self.last_h),
                                 arch=c.arch, precision=c.precision, capacity_headroom=c.capacity_headroom)
        self.solver.set_pins(self._pin_targets())
        self._upload_mass()
        self.solver.synchronize_host_state()
        self.topology = TopologyManager(self.particles, self.mesh)
        self.failure_model = StrainThresholdFailure(c.critical_strain)
        self.provider = PrescribedSphere(config=motion or MotionConfig())
        self.costs["setup"] = perf_counter()-start-self.costs["mass_upload"]
        self._publish()

    def _material(self, h):
        c = self.config
        return ClothMaterial(stretch_compliance=c.stretch_compliance, bend_compliance=c.bend_compliance,
                             pin_compliance=0., damping=float(np.exp(-c.gamma*h)), gravity=c.gravity)

    def _host_snapshot(self):
        p = self.particles
        return ClothSnapshot(p.position, self.mesh.triangles, np.arange(p.count),
                             np.arange(self.mesh.triangle_count), p.velocity, p.inverse_mass, p.pinned,
                             self.epoch, self.step_index, self.mesh.topology_version, self.time, self.last_h)

    def _publish(self):
        start = perf_counter()
        if self.failed or self.solver.topology_version != self.mesh.topology_version:
            raise RuntimeError("cannot publish a failed or unreconciled state")
        current = self._host_snapshot()
        self.clock.publish(current)
        self.current = current
        self.costs["snapshot"] += perf_counter()-start

    def _assign_mass(self, mass):
        self.particles.mass[:] = mass.mass
        self.particles.inverse_mass[:] = mass.inverse_mass
        self.particles.pinned[:] = mass.pinned

    def _upload_mass(self):
        start = perf_counter()
        p = self.particles
        self.solver.upload_mass_state(p.mass, p.inverse_mass, p.pinned)
        self.costs["mass_upload"] += perf_counter()-start

    def _pin_targets(self):
        return {int(i): self.reference.reference_positions[i]*(1+self.config.prestrain)
                for i in np.flatnonzero(self.reference.boundary)}

    def _check_stability(self, h):
        c = self.config
        if not c.contact_enabled or (c.stiffness == 0 and c.contact_damping == 0):
            return
        start = perf_counter()
        xyz = self.current.positions[self.current.triangles]
        area = .5*np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=1)
        if not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("current cloth has nonfinite or zero-area triangles")
        vertex_area = np.zeros(len(self.current.positions))
        np.add.at(vertex_area, self.current.triangles.ravel(), np.repeat(area/3, 3))
        if np.any(vertex_area <= 0):
            raise ValueError("current cloth has isolated vertices")
        ratio = self.particles.mass/vertex_area
        # Both shared limits increase monotonically with m/A. Evaluate the
        # actual weakest vertex through the single authoritative formula.
        weakest = int(np.argmin(ratio))
        limit = contact_stability_limit(self.particles.mass[weakest], vertex_area[weakest],
                                        c.stiffness, c.contact_damping, c.stability_beta)
        self.minimum_stability_margin = min(self.minimum_stability_margin, limit/h)
        self.costs["stability"] += perf_counter()-start
        if h > limit:
            raise ValueError(f"explicit contact pre-screen failed: h={h:g} > limit={limit:g}; increase substeps")

    def substep(self, h=None, *, diagnostics=False):
        if self.failed:
            raise RuntimeError("driver failed; start a new simulation")
        if isinstance(h, (bool, np.bool_)):
            raise ValueError("substep h must be a real duration")
        h = self.config.dt/self.config.substeps if h is None else float(h)
        if not np.isfinite(h) or h <= 0 or self.time+h <= self.time:
            raise ValueError("substep h must advance finite time")
        try:
            self._check_stability(h)
            c = self.config
            start = perf_counter()
            traction = integrate_traction(self.current, self.provider, stream_id=self.stream_id, h=h,
                         stiffness=c.stiffness if c.contact_enabled else 0.,
                         damping=c.contact_damping if c.contact_enabled else 0.,
                         batch_id=f"{self.epoch}:{self.step_index}")
            self.costs["load"] += perf_counter()-start
            start = perf_counter()
            self.ledger.consume(traction.batch, self.current, source=self.provider.source, h=h,
                                capacity=self.solver.vertex_capacity, apply=self.solver.apply_impulses)
            self.costs["load_validation_upload"] += perf_counter()-start
            start = perf_counter()
            self.solver.update_material(self._material(h))
            self.solver.substep(h, c.iterations)
            self.costs["solver"] += perf_counter()-start
            self.provider.advance(self.time, h)
            self.time += h
            self.last_h = h
            self.step_index += 1
            self.last_diagnostics = None
            self.last_samples = None
            self._publish()
            if diagnostics:
                self._diagnostics()
            self.work += float(traction.work)
            self.load_history.append(dict(step_index=self.step_index, t_start=traction.batch.t_start,
                 t_end=traction.batch.t_end, cloth_version=list(traction.batch.cloth_version),
                 batch_id=traction.batch.batch_id, source_id=traction.batch.source.source_id,
                 source_kind=traction.batch.source.kind, source_version=traction.batch.source.version,
                 force=np.asarray(traction.force).tolist(),
                 torque=np.asarray(traction.torque).tolist(), work=float(traction.work),
                 reaction_force=np.asarray(traction.reaction_force).tolist(),
                 reaction_torque=np.asarray(traction.reaction_torque).tolist(),
                 contact_area=float(traction.contact_area), current_area=float(traction.current_area),
                 max_penetration=float(traction.max_penetration)))
            return traction
        except Exception:
            self.failed = True
            self.last_samples = self.last_diagnostics = None
            raise

    def _diagnostics(self):
        start = perf_counter()
        self.last_diagnostics = {
            kind: ConstraintDiagnostics(stream_id=self.stream_id, cloth_version=self.current.version,
                                        time=self.time, **self.solver.constraint_diagnostics(kind, self.current.vertex_ids))
            for kind in ("stretch", "bend", "pin")}
        for diagnostic in self.last_diagnostics.values():
            diagnostic.validate_for(self.current, self.stream_id, repack_count=self.solver.repack_count)
        self.costs["diagnostics"] += perf_counter()-start

    def update_queries(self):
        if self.failed:
            raise RuntimeError("cannot query failed driver")
        self.last_samples = None
        start = perf_counter()
        queries = self.provider.sample_boundary(self.time, self.config.query_count)
        mapping = map_points(self.current, queries, MappingConfig(max_distance=self.config.query_max_distance))
        self.costs["mapping"] += perf_counter()-start
        start = perf_counter()
        c = self.config
        samples = make_boundary_samples(self.current, queries, mapping, self.provider,
                  stream_id=self.stream_id, stiffness=c.stiffness if c.contact_enabled else 0.,
                  damping=c.contact_damping if c.contact_enabled else 0.)
        samples.validate_for(self.current, stream_id=self.stream_id, source=self.provider.source)
        self.provider.accept_cloth_samples(samples)
        self.last_samples = samples
        self.costs["boundary_packet"] += perf_counter()-start
        return samples

    def _fracture_one_edge(self):
        candidates = []
        for edge in self.mesh.edges:
            if edge.active and len(edge.adjacent_triangles) == 2:
                edge.current_strain = edge.strain(self.particles.position)
                if self.failure_model.should_break(edge):
                    candidates.append((edge.current_strain, edge.id, edge))
        if not candidates:
            return None
        strain, edge_id, edge = max(candidates, key=lambda item: item[0])
        before, diagnostics = self.current, self.last_diagnostics
        old_ref = self.reference
        old_mass = self.particles.mass.copy()
        momentum_before = np.sum(old_mass[:, None]*self.particles.velocity, axis=0, dtype=np.float64)
        nested_cost_before = self.costs["mass_upload"] + self.costs["snapshot"]
        start = perf_counter()
        split = self.topology.split_vertex(edge.a, edge_id, gap=0.)
        if not split:
            return None
        ref, roots = self.mesh.reference_data()
        self.reference = ReferenceState(self.stream_id, self.epoch, self.mesh.topology_version,
                         np.arange(self.particles.count), roots, ref, self.original_boundary[roots],
                         old_ref.triangle_ids, old_ref.triangle_mass)
        self.reference.validate_successor(old_ref)
        mass = redistribute_mass(self._host_snapshot(), self.reference)
        self._assign_mass(mass)
        mass.validate_for(self._host_snapshot(), self.stream_id,
                          capacity=self.solver.vertex_capacity, reference=self.reference)
        momentum_after = np.sum(self.particles.mass[:, None]*self.particles.velocity, axis=0, dtype=np.float64)
        momentum_error = np.linalg.norm(momentum_after-momentum_before)
        if (abs(self.particles.mass.sum()-self.initial_mass)/self.initial_mass > 1e-12
                or momentum_error/max(np.linalg.norm(momentum_before), 1e-3) > 1e-12):
            raise RuntimeError("tear mass/momentum conservation failed")
        self.solver.rebuild_constraints()
        self.solver.set_pins(self._pin_targets())
        self._upload_mass()
        self.solver.synchronize_host_state()
        self.last_diagnostics = self.last_samples = None
        self._publish()
        nested_cost = self.costs["mass_upload"] + self.costs["snapshot"] - nested_cost_before
        self.costs["tear_rebuild"] += perf_counter()-start-nested_cost
        self.update_queries()
        event = dict(time=self.time, step_index=self.step_index, outer_index=self.outer_index,
                     edge=edge_id, strain=strain, before_version=list(before.version),
                     after_version=list(self.current.version), new_vertices=self.particles.count-len(before.positions),
                     momentum_error=float(momentum_error), gap=0.)
        self.events.append(event)
        return before, diagnostics, event

    def step(self):
        if self.failed:
            raise RuntimeError("driver failed; start a new simulation")
        try:
            for i in range(self.config.substeps):
                self.substep(diagnostics=i == self.config.substeps-1)
            self.outer_index += 1
            return self._fracture_one_edge() if self.config.tearing_enabled else None
        except Exception:
            self.failed = True
            self.last_samples = self.last_diagnostics = None
            raise


def _json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def run(config: PunchConfig, outer_steps: int, output: Path):
    if outer_steps < 1:
        raise ValueError("outer_steps must be positive")
    output.mkdir(parents=True, exist_ok=False)
    simulation = CompliantPunchSimulation(config)
    entries = []
    frames = 0

    def record(snapshot, diagnostics, event=None, samples=None):
        nonlocal frames
        start = perf_counter()
        payload = {f.name: getattr(snapshot, f.name) for f in fields(snapshot)}
        payload.update(stream_id=simulation.stream_id, source_id=simulation.provider.source.source_id,
                       source_kind=simulation.provider.source.kind, source_version=simulation.provider.source.version,
                       coordinate_frame="world", units="SI", label=LABEL)
        payload["mass"] = simulation.particles.mass.copy() if event != "pre_tear" else pre_mass.copy()
        payload["center"], payload["body_velocity"] = simulation.provider.kinematics(snapshot.time)
        if diagnostics:
            for kind, diagnostic in diagnostics.items():
                for field in fields(diagnostic):
                    payload[f"diagnostic_{kind}_{field.name}"] = getattr(diagnostic, field.name)
        if samples is not None:
            payload["query_positions"] = samples.queries.positions
            for field in fields(samples.mapping):
                payload[f"mapping_{field.name}"] = getattr(samples.mapping, field.name)
            for key in ("cloth_velocity", "phi", "outward_normal", "body_velocity", "pressure", "active", "measure_weights"):
                payload[f"provider_{key}"] = getattr(samples, key)
            payload["sampling_measure"] = samples.sampling_measure
        name = f"state_{len(entries):05d}.npz"
        np.savez_compressed(output/name, **payload)
        entries.append(dict(file=name, frame_index=frames if event is None else None,
                            event=event, outer_index=simulation.outer_index,
                            step_index=snapshot.step_index, epoch=snapshot.epoch,
                            topology_version=snapshot.topology_version, actual_time=snapshot.time,
                            scheduled_time=simulation.outer_index*config.dt if event is None else None,
                            h=snapshot.dt, outer_dt=config.dt, source_version=simulation.provider.source.version,
                            diagnostics_available=diagnostics is not None))
        frames += event is None
        simulation.costs["recording"] += perf_counter()-start

    _json_write(output/"config.json", dict(label=LABEL, config=asdict(config), motion=asdict(simulation.provider.config),
                                          provider_type="PrescribedSphere", representation_seed=simulation.provider.seed))
    np.savez_compressed(output/"representation.npz", positions=simulation.provider.representation_points(32768),
                        reference_center=simulation.provider.kinematics(0.)[0])
    simulation.update_queries()
    record(simulation.current, None, samples=simulation.last_samples)
    for _ in range(outer_steps):
        pre_mass = simulation.particles.mass.copy()
        tear = simulation.step()
        if tear:
            before, diagnostic, _ = tear
            record(before, diagnostic, event="pre_tear")
            record(simulation.current, None, event="post_tear", samples=simulation.last_samples)
        if simulation.outer_index % config.output_stride == 0:
            if simulation.last_samples is None:
                simulation.update_queries()
            record(simulation.current, simulation.last_diagnostics, samples=simulation.last_samples)
    _json_write(output/"manifest.json", dict(label=LABEL, frames=entries,
                output_stride_outer_steps=config.output_stride, fps=1/(config.dt*config.output_stride),
                physical_duration=simulation.time, playback_duration=frames*config.dt*config.output_stride))
    source = [Path(__file__), *Path(__file__).parent.joinpath("coupling").glob("*.py"),
              *Path(__file__).parent.joinpath("cloth").glob("*.py")]
    _json_write(output/"summary.json", dict(label=LABEL, status="integration_only_not_physical_acceptance",
                 source_mode="prescribed", units="SI", motion=asdict(simulation.provider.config),
                 representation_count=32768, representation_seed=simulation.provider.seed, uniform_frames=frames,
                 config=asdict(config), actual_backend=simulation.solver.backend,
                 vertices=simulation.particles.count, triangles=simulation.mesh.triangle_count,
                 steps=simulation.step_index, time=simulation.time, events=simulation.events,
                 mass=float(simulation.particles.mass.sum()), contact_work=simulation.work,
                 stability_margin=None if not np.isfinite(simulation.minimum_stability_margin) else simulation.minimum_stability_margin,
                 global_balance="not_closed_support_reaction_unmeasured", costs_seconds=simulation.costs,
                 cost_notes={"solver": "includes first-use JIT and built-in end-substep state readback",
                             "load_validation_upload": "ledger validation plus bridge host/device transfer",
                             "mapping": "full surface CPU queries; no GPU geometry mapping",
                             "rendering": "not performed; this command records raw states only"},
                 hashes={str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}))
    _json_write(output/"loads.json", simulation.load_history)
    return simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--no-contact", action="store_true")
    parser.add_argument("--no-tearing", action="store_true")
    args = parser.parse_args()
    result = run(PunchConfig(arch=args.arch, precision=args.precision, substeps=args.substeps,
                            contact_enabled=not args.no_contact, tearing_enabled=not args.no_tearing),
                 args.outer_steps, args.output)
    print(f"{LABEL}; completed {result.step_index} substeps, {result.time:.6f} s")
