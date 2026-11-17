"""Synthetic reference-area traction on B1 cloth, with physical-step archives.

SI engineering example, not a calibrated material or fluid-pressure model.
The existing XPBD solver, membrane, bend, and fracture criterion are reused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import uuid

import numpy as np

from .main import DemoConfig, make_grid
from .cloth.constraints import ClothMaterial, XPBDSolver
from .solver_convergence import convergence_policy
from .support_balance import SCHEMA as BALANCE_SCHEMA, build_balance
from .cloth.failure import StrainThresholdFailure
from .cloth.topology import TopologyManager
from .coupling.areal_mass import create_reference, redistribute_mass
from .coupling.contracts import ClothSnapshot
from .coupling.loads import LoadBatch, LoadLedger, SnapshotClock, SourceIdentity
from .coupling.provider_contracts import ReferenceState
from .cloth_archive import ArchiveWriter, ClothRecord, SCHEMA, json_text, sha256, tear_balance
from .traction_schedule import normalize_schedule, integrate_schedule, validate_schedule_coverage, load_schedule_file


@dataclass(frozen=True)
class LoadedClothConfig:
    resolution: int = 4
    width: float = 1.0
    height: float = 1.4
    steps: int = 6
    h: float = 1 / 240
    iterations: int = 4
    areal_density: float = .5       # kg/m^2 of reference surface
    traction: tuple = (0., 0., 10.) # N/m^2, fixed world vector per reference area
    traction_schedule: tuple | None = None  # (time [s], dimensionless nonnegative scale)
    bend_stiffness: float = .05    # B [N m], explicit engineering example
    stretch_compliance: float = 1e-6
    damping: float = 1.0           # velocity multiplier: 1 means no damping
    gravity: tuple = (0., 0., 0.)
    precision: str = "f64"
    arch: str = "cpu"
    capacity_headroom: float = 4.
    prescribed_cut_after_step: int | None = None
    cut_edge: tuple[int, int] | None = None
    critical_strain: float | None = None
    convergence: bool = False  # opt-in CPU engineering residual stopping/SOR
    support_balance: bool = False  # measure each projection on the static support

    def __post_init__(self):
        if type(self.support_balance) is not bool or (self.support_balance and
                (self.arch != "cpu" or self.precision != "f64" or self.h != .001 or not self.convergence)):
            raise ValueError("support_balance requires CPU f64, h=0.001 and convergence")
        if type(self.convergence) is not bool:
            raise ValueError("convergence must be bool")
        if self.convergence and (self.arch != "cpu" or type(self.iterations) is not int or not 8 <= self.iterations <= 96):
            raise ValueError("convergence requires CPU and iteration cap 8..96")
        for name, minimum in (("resolution", 3), ("steps", 1), ("iterations", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("width", "height", "h", "areal_density", "bend_stiffness", "capacity_headroom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive finite")
        if not 0 <= self.damping <= 1 or not np.isfinite(self.stretch_compliance) or self.stretch_compliance < 0:
            raise ValueError("invalid damping/compliance")
        for name in ("traction", "gravity"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite world vector")
            object.__setattr__(self, name, tuple(float(v) for v in value))
        if self.traction_schedule is not None:
            schedule = normalize_schedule(self.traction_schedule)
            validate_schedule_coverage(schedule, self.steps * self.h)
            object.__setattr__(self, "traction_schedule", schedule)
        if self.precision not in ("f32", "f64") or self.arch not in ("cpu", "cuda"):
            raise ValueError("invalid backend/precision")
        if self.prescribed_cut_after_step is not None:
            if (type(self.prescribed_cut_after_step) is not int
                    or not 1 <= self.prescribed_cut_after_step <= self.steps or self.critical_strain is not None):
                raise ValueError("prescribed cut must be in run and exclusive of strain mode")
            if (self.cut_edge is None or len(self.cut_edge) != 2 or
                    any(type(v) is not int or v < 0 for v in self.cut_edge) or self.cut_edge[0] == self.cut_edge[1]):
                raise ValueError("prescribed cut needs two distinct vertex IDs")
            object.__setattr__(self, "cut_edge", tuple(self.cut_edge))
        elif self.cut_edge is not None:
            raise ValueError("cut edge needs a prescribed cut step")
        if self.critical_strain is not None and (not np.isfinite(self.critical_strain) or self.critical_strain <= 0):
            raise ValueError("critical strain must be positive finite")


def _momentum(state):
    return (state["mass"][:, None] * state["velocity"]).sum(axis=0).tolist()


class LoadedClothDriver:
    def __init__(self, config=None, *, max_natural_events=None, stream_id=None):
        self.config = c = config or LoadedClothConfig()
        if max_natural_events is not None and (type(max_natural_events) is not int
                or max_natural_events < 1 or c.critical_strain is None):
            raise ValueError("max_natural_events requires a positive integer and natural mode")
        self.max_natural_events = max_natural_events
        self.event_count = 0
        self._decision = None
        self._decision_consumed = True
        self.stream_id = str(uuid.uuid4()) if stream_id is None else stream_id
        self.source = SourceIdentity("reference-area-traction", "prescribed", 0)
        self.clock, self.ledger = SnapshotClock(self.stream_id), LoadLedger(self.stream_id)
        self.time, self.step_index, self.epoch = 0., 0, 0
        # Initial P20 snapshots require a positive nominal dt. After a solve
        # this field describes the completed substep, including topology commits.
        self.last_h = c.h
        self.failed, self.stage, self.cut_roots = False, "initial", set()
        self.particles, self.mesh, _, _ = make_grid(DemoConfig(width=c.width, height=c.height,
                                        resolution_x=c.resolution, resolution_y=c.resolution))
        self.original_boundary = self.particles.pinned.copy()
        rest, roots = self.mesh.reference_data()
        self.reference = create_reference(self._host_snapshot(), rest, roots, self.original_boundary,
                                          stream_id=self.stream_id, areal_density=c.areal_density)
        self._assign_mass()
        self.solver = XPBDSolver(self.particles, self.mesh,
            ClothMaterial(stretch_compliance=c.stretch_compliance, bend_model="dihedral",
                          bend_stiffness=c.bend_stiffness, pin_compliance=0.,
                          damping=c.damping, gravity=c.gravity),
            arch=c.arch, precision=c.precision, capacity_headroom=c.capacity_headroom)
        self._sync_boundary_mass()
        self.topology = TopologyManager(self.particles, self.mesh)
        self.failure_model = StrainThresholdFailure(c.critical_strain) if c.critical_strain is not None else None
        self._publish()
        if c.cut_edge is not None:
            edge = self.mesh.edge_for_vertices(*c.cut_edge)
            if edge is None or len(edge.adjacent_triangles) != 2 or edge.broken:
                raise ValueError("prescribed cut must select an intact interior edge")

    def _host_snapshot(self, device=None):
        p = self.particles
        d = device or dict(positions=p.position, velocity=p.velocity,
                           inverse_mass=p.inverse_mass, pinned=p.pinned)
        return ClothSnapshot(d["positions"], self.mesh.triangles, np.arange(p.count),
                             np.arange(self.mesh.triangle_count), d["velocity"], d["inverse_mass"], d["pinned"],
                             self.epoch, self.step_index, self.mesh.topology_version, self.time, self.last_h)

    def _assign_mass(self):
        mass = redistribute_mass(self._host_snapshot(), self.reference)
        self.particles.mass[:] = mass.mass
        self.particles.inverse_mass[:] = mass.inverse_mass
        self.particles.pinned[:] = mass.pinned

    def _sync_boundary_mass(self):
        self.solver.set_pins({int(i): self.reference.reference_positions[i]
                              for i in np.flatnonzero(self.reference.boundary)})
        p = self.particles
        self.solver.upload_mass_state(p.mass, p.inverse_mass, p.pinned)
        self.solver.synchronize_host_state()

    def _publish(self):
        if self.failed or self.solver.topology_version != self.mesh.topology_version:
            raise RuntimeError("cannot publish failed/unreconciled state")
        self.current = self._host_snapshot(self.solver.read_device_particles())
        self.clock.publish(self.current)

    def make_load(self):
        if self.failed:
            raise RuntimeError("driver failed; start a new stream")
        # Fixed triangle masses / rho, scattered through CURRENT incidence.
        area = redistribute_mass(self.current, self.reference).mass / self.config.areal_density
        t_start, t_end = self.time, self.time + self.config.h
        if self.config.traction_schedule is None:
            impulse = area[:, None] * np.asarray(self.config.traction) * self.config.h
        else:
            integral = integrate_schedule(self.config.traction_schedule, t_start, t_end)
            impulse = area[:, None] * np.asarray(self.config.traction) * integral
        batch = LoadBatch(self.stream_id, self.source, self.current.version,
                          f"{self.epoch}:{self.step_index}", t_start, t_end,
                          self.current.vertex_ids, impulse)
        return batch, area

    def record(self, *, load=None, extra=None, event=None, balance=None, tear_before=None):
        if self.failed:
            raise RuntimeError("cannot record failed driver")
        d = self.solver.read_device_particles()
        arrays = dict(d, triangles=self.mesh.triangles, vertex_ids=self.current.vertex_ids,
                      triangle_ids=self.current.triangle_ids, root_vertex_ids=self.reference.root_vertex_ids,
                      reference_positions=self.reference.reference_positions, triangle_mass=self.reference.triangle_mass,
                      edge_vertex_ids=np.array([(e.a, e.b) for e in self.mesh.edges], dtype=np.int64).reshape(-1, 2),
                      edge_active=np.array([e.active for e in self.mesh.edges], dtype=bool),
                      edge_broken=np.array([e.broken for e in self.mesh.edges], dtype=bool),
                      cut_root_edges=np.array(sorted(self.cut_roots), dtype=np.int64).reshape(-1, 2))
        diagnostics = None
        if self.stage == "post_solve":
            diagnostics = {}
            for kind in ("stretch", "bend", "pin"):
                raw = self.solver.constraint_diagnostics(kind, self.current.vertex_ids)
                diagnostics[kind] = {k: v for k, v in raw.items() if not isinstance(v, np.ndarray)}
                diagnostics[kind].update(version=list(self.current.version), time=self.time)
                arrays.update({f"diag_{kind}_{k}": v for k, v in raw.items() if isinstance(v, np.ndarray)})
        arrays.update(extra or {})
        metadata = dict(schema=SCHEMA, stream_id=self.stream_id, stage=self.stage,
                version=list(self.current.version), time=self.time, h=self.current.dt,
                repack_count=self.solver.repack_count, load=load, event=event, diagnostics=diagnostics)
        if balance is not None:
            metadata["support_balance"] = balance
        if tear_before is not None:
            event["balance_jump"] = tear_balance(tear_before, d)
        if self.failure_model is not None:
            metadata["fracture_decision"] = self._decision if self.stage == "post_solve" else None
        if self.config.convergence and self.stage == "post_solve":
            metadata["solver_convergence"] = self.solver.last_convergence
        return ClothRecord(metadata, arrays)

    def _natural_decision(self):
        """Freeze selection before mutation; all eligible edges remain observable."""
        candidates = []
        for edge in self.mesh.edges:
            if not edge.active or edge.broken or len(edge.adjacent_triangles) != 2:
                continue
            roots = tuple(sorted(int(self.reference.root_vertex_ids[v]) for v in edge.key))
            if roots[0] == roots[1] or not np.isfinite(edge.rest_length) or edge.rest_length <= 1e-12:
                raise ValueError("ambiguous root pair or undefined reference edge")
            if roots in self.cut_roots:
                continue
            strain = float(edge.strain(self.particles.position))
            if not np.isfinite(strain):
                raise ValueError("nonfinite natural candidate strain")
            edge.current_strain = strain
            candidates.append(dict(edge=list(edge.key), root_edge=list(roots), strain=strain,
                                   exceeds_threshold=bool(self.failure_model.should_break(edge))))
        candidates.sort(key=lambda item: item["edge"])
        ranked, seen = [], set()
        for item in sorted(candidates, key=lambda v: (-v["strain"], v["root_edge"], v["edge"])):
            root = tuple(item["root_edge"])
            if item["exceeds_threshold"] and root not in seen:
                ranked.append(item["edge"])
                seen.add(root)
        exhausted = self.max_natural_events is not None and self.event_count >= self.max_natural_events
        would_select = ranked[0] if ranked else None
        selected = would_select if not exhausted else None
        return dict(schema="xpbd-fracture-decision-v2", version=list(self.current.version),
                    event_count_before=self.event_count, budget_exhausted=exhausted,
                    candidates=candidates, ranked_edges=ranked, would_select=would_select,
                    selected_edge=selected, reason=("no_candidate" if not ranked else
                    "event_budget_exhausted" if exhausted else "selected"))

    def step(self):
        if self.failed or self.step_index >= self.config.steps:
            raise RuntimeError("driver failed or reached configured end")
        if not self._decision_consumed:
            raise RuntimeError("consume the previous natural decision before advancing")
        try:
            batch, area = self.make_load()
            schedule_fields = {}
            if self.config.traction_schedule is not None:
                integral = integrate_schedule(self.config.traction_schedule, batch.t_start, batch.t_end)
                mean = integral / self.config.h
                if not np.isfinite(mean):
                    raise ValueError("mean traction scale must remain finite")
                schedule_fields = dict(traction_scale_integral_s=integral, mean_scale=mean)
            before = self.solver.read_device_particles()
            self.ledger.consume(batch, self.current, source=self.source, h=self.config.h,
                                capacity=self.solver.vertex_capacity, apply=self.solver.apply_impulses)
            uploaded = self.solver.read_device_particles()
            # The second argument is ITERATIONS, never another impulse.
            self.solver.substep(self.config.h, self.config.iterations, convergence=self.config.convergence,
                                measure_balance=self.config.support_balance)
            self.time += self.config.h
            self.step_index += 1
            self.stage = "post_solve"
            self._publish()
            if self.failure_model is not None:
                self._decision = self._natural_decision()
                self._decision_consumed = False
            solved = self.solver.read_device_particles()
            pin = before["pinned"]
            audit = dict(external_total=batch.impulse.sum(axis=0).tolist(),
                         external_free=batch.impulse[~pin].sum(axis=0).tolist(),
                         external_pinned=batch.impulse[pin].sum(axis=0).tolist(),
                         pinned_external_cancellation=(-batch.impulse[pin].sum(axis=0)).tolist(),
                         momentum_before=_momentum(before), momentum_uploaded=_momentum(uploaded),
                         momentum_solved=_momentum(solved),
                         support_reaction=BALANCE_SCHEMA if self.config.support_balance else "unmeasured")
            self.last_upload = dict(before=before, uploaded=uploaded)
            load = dict(version=list(batch.cloth_version), batch_id=batch.batch_id,
                        t_start=batch.t_start, t_end=batch.t_end, audit=audit)
            load.update(schedule_fields)
            extra = dict(load_vertex_ids=batch.vertex_ids, load_impulse=batch.impulse,
                         load_reference_area=area, load_velocity_before=before["velocity"],
                         load_velocity_uploaded=uploaded["velocity"])
            balance = None
            if self.config.support_balance:
                balance, measured = build_balance(
                    dict(before, vertex_ids=batch.vertex_ids), dict(solved, vertex_ids=self.current.vertex_ids),
                    batch.impulse, self.solver.read_projection_balance(),
                    h=self.config.h, gravity=self.config.gravity, damping=self.config.damping,
                    start_version=batch.cloth_version, end_version=self.current.version,
                    t_start=batch.t_start, t_end=batch.t_end)
                extra.update(measured)
            return self.record(load=load, extra=extra, balance=balance)
        except Exception:
            self.failed = True
            raise

    def tear_if_requested(self):
        if self.failed:
            raise RuntimeError("driver failed; start a new stream")
        if self.stage != "post_solve":
            return None
        c = self.config
        try:
            edge, kind = None, None
            if c.prescribed_cut_after_step == self.step_index:
                edge, kind = self.mesh.edge_for_vertices(*c.cut_edge), "prescribed_cut"
                if edge is None or edge.broken:
                    raise ValueError("prescribed edge unavailable")
            elif self.failure_model:
                if self._decision_consumed:
                    return None
                self._decision_consumed = True
                selected = self._decision["selected_edge"]
                if selected is not None:
                    edge, kind = self.mesh.edge_for_vertices(*selected), "strain_threshold"
                    if edge is None:
                        raise RuntimeError("frozen natural candidate is unavailable")
            if edge is None:
                return None
            source_stop_reason = self.solver.last_convergence["stop_reason"] if c.convergence else None
            before, old_ref = self.current, self.reference
            tear_before = self.solver.read_device_particles() if c.support_balance else None
            endpoints = [edge.a, edge.b]
            cut_root = tuple(sorted(int(old_ref.root_vertex_ids[i]) for i in endpoints))
            if cut_root[0] == cut_root[1] or cut_root in self.cut_roots:
                raise ValueError("ambiguous or repeated root-pair cut")
            edge.current_strain = edge.strain(self.particles.position)
            strain = float(edge.current_strain)
            if not self.topology.split_vertex(edge.a, edge.id, gap=0.):
                raise RuntimeError("requested cut made no topology event")
            rest, roots = self.mesh.reference_data()
            self.reference = ReferenceState(self.stream_id, self.epoch, self.mesh.topology_version,
                np.arange(self.particles.count), roots, rest, self.original_boundary[roots],
                old_ref.triangle_ids, old_ref.triangle_mass)
            self.reference.validate_successor(old_ref)
            self._assign_mass()
            self.solver.rebuild_constraints()
            self._sync_boundary_mass()
            self.cut_roots.add(cut_root)
            self.event_count += 1
            self.stage = "post_tear"
            self._publish()
            n = len(before.vertex_ids)
            event = dict(kind=kind, gap=0., cut_vertex_ids=endpoints, strain=strain,
                         before_version=list(before.version), after_version=list(self.current.version),
                         step=self.step_index, time=self.time, new_vertex_ids=list(range(n, self.particles.count)),
                         parent_vertex_ids=[parent for _, parent in self.topology.last_split_parents])
            if [new for new, _ in self.topology.last_split_parents] != event["new_vertex_ids"]:
                raise RuntimeError("topology did not report each immediate parent")
            if self.failure_model is not None:
                event.update(schema="xpbd-cloth-event-v2", sequence=self.event_count,
                             cut_root_edge=list(cut_root))
            if c.convergence:
                event["solver_stop_reason"] = source_stop_reason
            return self.record(event=event, tear_before=tear_before)
        except Exception:
            self.failed = True
            raise

    def run_metadata(self):
        root = Path(__file__).resolve().parent.parent
        paths = ["project/loaded_cloth.py", "project/cloth_archive.py", "project/solver_convergence.py", "project/support_balance.py", "project/traction_schedule.py", "project/main.py",
                 "project/cloth/constraints.py", "project/cloth/bending.py", "project/cloth/topology.py",
                 "project/cloth/mesh.py", "project/cloth/particles.py", "project/cloth/failure.py",
                 "project/coupling/loads.py", "project/coupling/contracts.py",
                 "project/coupling/areal_mass.py", "project/coupling/provider_contracts.py"]
        config = asdict(self.config)
        if self.config.traction_schedule is None:
            config.pop("traction_schedule")
        metadata = dict(schema=SCHEMA, stream_id=self.stream_id, source=asdict(self.source),
                    backend=self.solver.arch, precision=self.solver.precision, config=config,
                    steps=self.config.steps, units="SI", coordinate_frame="world",
                    load_model=("reference_area_fixed_direction" if self.config.traction_schedule is None
                                else "reference_area_piecewise_linear"),
                    support_reaction=BALANCE_SCHEMA if self.config.support_balance else "unmeasured",
                    parameter_status="engineering_example_not_calibrated",
                    source_sha256={p: sha256(root / p) for p in paths})
        if self.failure_model is not None:
            metadata["fracture_policy"] = dict(schema="xpbd-fracture-policy-v2",
                order="strain_desc_root_asc_vertex_asc", root_cut="once_per_root_pair",
                max_events=self.max_natural_events)
        if self.config.convergence:
            metadata["solver_convergence_policy"] = convergence_policy(self.config.precision)
        return metadata


def run(config, output, *, max_natural_events=None):
    driver = LoadedClothDriver(config, max_natural_events=max_natural_events)
    writer = ArchiveWriter(output, driver.run_metadata())
    try:
        writer.append(driver.record())
        for _ in range(config.steps):
            writer.append(driver.step())
            tear = driver.tear_if_requested()
            if tear is not None:
                writer.append(tear)
        writer.complete()
    except Exception as error:
        driver.failed = True
        writer.fail(error)
        raise
    return writer.manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="new archive directory")
    p.add_argument("--resolution", type=int, default=4)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--h", type=float, default=1/240)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--convergence", action="store_true", help="CPU residual stopping with SOR; --iterations is cap 8..96")
    p.add_argument("--support-balance", action="store_true", help="measure static support impulses (CPU f64, h=0.001, --convergence)")
    p.add_argument("--bend-stiffness", type=float, required=True, help="B [N m], engineering parameter")
    p.add_argument("--areal-density", type=float, default=.5)
    p.add_argument("--traction", nargs=3, type=float, default=(0., 0., 10.), metavar=("TX", "TY", "TZ"))
    p.add_argument("--traction-schedule", help="JSON [[time_s, nonnegative_scale], ...], covering the run")
    p.add_argument("--precision", choices=("f32", "f64"), default="f64")
    p.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--prescribed-cut-after-step", type=int)
    p.add_argument("--cut-edge", nargs=2, type=int)
    p.add_argument("--critical-strain", type=float)
    p.add_argument("--max-natural-events", type=int, help="optional total budget; tail decisions stay recorded")
    args = vars(p.parse_args(argv))
    output = args.pop("output")
    max_natural_events = args.pop("max_natural_events")
    if args["traction_schedule"] is not None:
        args["traction_schedule"] = load_schedule_file(args["traction_schedule"])
    manifest = run(LoadedClothConfig(**args), output, max_natural_events=max_natural_events)
    print(json_text(dict(status=manifest["status"], records=len(manifest["records"]), output=str(output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
