"""Numerical measurement CLI for XPBD_tear.

Records an auditable snapshot of one backend/precision run: the complete
demo config and material, the actually selected Taichi backend, package
versions, source SHA256 digests, per-frame correctness assertions (finite
state; solver repack/coloring versions consistent with the mesh), the
fracture log, final topology counts, a final-state NPZ, the diagnostics
topology report, and a nested timing breakdown.

Run from the repository root, one process per backend/precision:

    conda run -n XPBD_tear python -m tools.measure_simulation \
        --arch cpu --precision f32 --frames 120 --resolution 8 \
        --critical-strain 0.25 --pull-speed 0.18 \
        --output-dir artifacts/measure/cpu_f32

This tool writes numerical artifacts only. Diagnostic PNGs and GIFs are
produced separately by the existing visualization entry points and are
deliberately excluded from the measured loop.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

import taichi as ti

from project.cloth.constraints import ClothMaterial, XPBDSolver
from project.cloth.taichi_runtime import initialize_taichi
from project.cloth.topology import TopologyManager
from project.main import ClothSimulation, DemoConfig
from project.viz.diagnostics import DiagnosticsRecorder

REPO_ROOT = Path(__file__).resolve().parents[1]

TIMING_METHOD_NOTES = {
    "clock": "time.perf_counter() wall clock on the host process.",
    "nested_timers": (
        "Regions nest: complete_step contains solver_step plus the Python fracture scan; "
        "solver_step contains substeps and pin-target uploads; fracture_scan contains splitting and repacking; substeps contain "
        "kernel launches. Do not add these numbers together; use them as a breakdown."
    ),
    "clock_granularity": "Complete-step timing separately records its trailing ti.sync; solver_step includes its state readback.",
    "gpu_async": (
        "Kernel launches are asynchronous on CUDA. Inner regions (substeps, repacks, pin "
        "uploads) measure host elapsed time, potentially including implicit synchronization, not isolated device kernel time. "
        "Device work is captured by the ti.sync() call inside complete_step, which is why "
        "the outer numbers matter for the CUDA backend."
    ),
    "no_per_kernel_sync": (
        "No per-kernel ti.sync() is inserted: synchronizing each kernel would serialize "
        "the pipeline and distort the measured behaviour."
    ),
    "readback": "complete_step and solver_step include the solver's Taichi-to-NumPy state readback.",
    "excluded": (
        "Construction, Taichi initialization, validation checks, NPZ/JSON writing and "
        "topology reporting are outside the measured loop. First-step JIT remains included in totals; steady statistics omit the first five frames."
    ),
    "validation_outside_timers": (
        "Per-frame finite checks, version checks and the diagnostics recorder run outside all "
        "timers, so they cost wall time but never enter the reported step times; the recorder "
        "is the dominant add-on cost because it recomputes every active edge strain."
    ),
}

TIMER_REGIONS = {
    "complete_step": "Wrapped ClothSimulation.step excluding external trailing sync; see complete_step_with_sync_seconds for synchronized total",
    "solver_step": "XPBDSolver.step; includes state readback",
    "substeps": "XPBDSolver._run_substep per substep; host launch cost",
    "pin_target_upload": "XPBDSolver._upload_pin_targets",
    "topology_repack": "XPBDSolver._repack_after_topology_change",
    "state_readback": "XPBDSolver._sync_particle_state",
    "fracture_scan": "ClothSimulation._fracture_one_edge (Python edge scan)",
    "topology_split": "TopologyManager.split_vertex (only on fracture frames)",
}

SOURCE_SUFFIXES = (".py", ".yml", ".yaml", ".txt")
# (timer region, owner class name, method name)
_FILES = (
    ("complete_step", "ClothSimulation", "step"),
    ("solver_step", "XPBDSolver", "step"),
    ("substeps", "XPBDSolver", "_run_substep"),
    ("pin_target_upload", "XPBDSolver", "_upload_pin_targets"),
    ("topology_repack", "XPBDSolver", "_repack_after_topology_change"),
    ("state_readback", "XPBDSolver", "_sync_particle_state"),
    ("fracture_scan", "ClothSimulation", "_fracture_one_edge"),
    ("topology_split", "TopologyManager", "split_vertex"),
)


class TimerRegistry:
    """Nested wall-clock accumulators for a fixed set of named regions."""

    def __init__(self, names: list[str]) -> None:
        self._totals = {name: 0.0 for name in names}
        self._counts = {name: 0 for name in names}

    def add(self, name: str, seconds: float) -> None:
        self._totals[name] += float(seconds)
        self._counts[name] += 1

    def count(self, name: str) -> int:
        return self._counts[name]

    def total(self, name: str) -> float:
        return self._totals[name]

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_seconds": self._totals[name],
                "calls": self._counts[name],
                "mean_seconds_per_call": (
                    self._totals[name] / self._counts[name] if self._counts[name] else 0.0
                ),
            }
            for name in self._totals
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_hashes() -> dict[str, str]:
    """SHA256 of every project source file plus the environment files."""
    hashes: dict[str, str] = {}
    for path in sorted((REPO_ROOT / "project").rglob("*")):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES and "__pycache__" not in path.parts:
            hashes[path.relative_to(REPO_ROOT).as_posix()] = _sha256_file(path)
    for name in ("environment.yml", "requirements.txt"):
        path = REPO_ROOT / name
        if path.is_file():
            hashes[name] = _sha256_file(path)
    return hashes


def _relative_or_absolute(path: Path) -> str:
    """Report paths relative to the repository root when they live inside it."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


@contextlib.contextmanager
def timed_methods(registry: TimerRegistry):
    """Patch the measurement targets for the duration of the measured loop.

    Each entry of ``patches`` is ``(class, method_name, original_function)``;
    the wrapper itself is installed on the class and the original is restored
    in ``finally`` even if the measured loop raises.
    """
    targets = {
        "ClothSimulation": ClothSimulation,
        "XPBDSolver": XPBDSolver,
        "TopologyManager": TopologyManager,
    }
    patches: list[tuple[type, str, Callable[..., Any]]] = []
    try:
        for region, class_name, method_name in _FILES:
            target = targets[class_name]
            original = getattr(target, method_name)

            def make_wrapper(fn: Callable[..., Any], timer_region: str):
                def wrapper(self, *args, **kwargs):
                    started = time.perf_counter()
                    try:
                        return fn(self, *args, **kwargs)
                    finally:
                        registry.add(timer_region, time.perf_counter() - started)

                wrapper.__name__ = getattr(fn, "__name__", "wrapped")
                wrapper.__doc__ = getattr(fn, "__doc__", None)
                wrapper.__wrapped__ = fn
                return wrapper

            setattr(target, method_name, make_wrapper(original, region))
            patches.append((target, method_name, original))
        yield registry
    finally:
        for target, method_name, original in reversed(patches):
            setattr(target, method_name, original)


def _material_payload(material: ClothMaterial, config: DemoConfig) -> dict[str, Any]:
    return {
        "source": "project.cloth.constraints.ClothMaterial",
        "config_scenario": {
            "resolution_x": config.resolution_x,
            "resolution_y": config.resolution_y,
            "dt": config.dt,
            "iterations": config.iterations,
            "substeps": config.substeps,
            "critical_strain": config.critical_strain,
            "pull_speed": config.pull_speed,
            "tearing_enabled": config.tearing_enabled,
            "particle_mass": config.particle_mass,
        },
        "material": {
            "stretch_compliance": material.stretch_compliance,
            "bend_compliance": material.bend_compliance,
            "pin_compliance": material.pin_compliance,
            "damping": material.damping,
            "gravity": list(material.gravity),
        },
    }


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    registry = TimerRegistry(list(TIMER_REGIONS))

    setup_started = time.perf_counter()
    backend = initialize_taichi(args.arch, args.precision, log=True)
    config = DemoConfig(
        resolution_x=args.resolution,
        resolution_y=max(6, int(round(args.resolution * 1.4))),
        critical_strain=args.critical_strain,
        pull_speed=args.pull_speed,
        arch=args.arch,
        precision=args.precision,
        capacity_headroom=4.0,
    )
    simulation = ClothSimulation(config)
    recorder = DiagnosticsRecorder()
    recorder.record(simulation)
    setup_seconds = time.perf_counter() - setup_started

    source_hashes = collect_source_hashes()

    per_frame: list[dict[str, Any]] = []
    version_mismatch_errors: list[str] = []
    finite_failures: list[int] = []
    first_fracture: dict[str, Any] | None = None
    previous_topology_version = int(simulation.mesh.topology_version)
    previous_repack_count = int(simulation.solver.repack_count)
    previous_coloring_count = int(simulation.solver.coloring_recompute_count)

    def sync_and_record(frame_index: int, started: float) -> float:
        sync_started = time.perf_counter()
        ti.sync()
        complete_step = time.perf_counter() - started
        per_frame.append(
            {
                "frame": frame_index,
                "complete_step_with_sync_seconds": complete_step,
                "host_only_seconds": sync_started - started,
                "ti_sync_seconds": time.perf_counter() - sync_started,
            }
        )
        return complete_step

    def check_consistency(frame_number: int) -> None:
        mesh_version = int(simulation.mesh.topology_version)
        solver = simulation.solver
        if solver.topology_version != mesh_version:
            version_mismatch_errors.append(
                f"after frame {frame_number}: solver.topology_version={solver.topology_version} != mesh={mesh_version}"
            )
        if solver.coloring_version != mesh_version:
            version_mismatch_errors.append(
                f"after frame {frame_number}: solver.coloring_version={solver.coloring_version} != mesh={mesh_version}"
            )

    try:
        with timed_methods(registry):
            for frame_index in range(args.frames):
                frame_number = frame_index + 1
                started = time.perf_counter()
                simulation.step()
                sync_and_record(frame_index, started)

                if not simulation.finite():
                    finite_failures.append(frame_index)
                    raise AssertionError(f"non-finite particle state after frame {frame_number}")

                mesh_version = int(simulation.mesh.topology_version)
                if mesh_version != previous_topology_version:
                    check_consistency(frame_number)
                    if int(simulation.solver.repack_count) != previous_repack_count + 1:
                        version_mismatch_errors.append(
                            f"after frame {frame_number}: repack_count={simulation.solver.repack_count} "
                            f"expected {previous_repack_count + 1}"
                        )
                    if int(simulation.solver.coloring_recompute_count) != previous_coloring_count + 1:
                        version_mismatch_errors.append(
                            f"after frame {frame_number}: coloring_recompute_count="
                            f"{simulation.solver.coloring_recompute_count} expected {previous_coloring_count + 1}"
                        )
                    previous_topology_version = mesh_version
                    previous_repack_count = int(simulation.solver.repack_count)
                    previous_coloring_count = int(simulation.solver.coloring_recompute_count)
                else:
                    check_consistency(frame_number)

                if first_fracture is None and simulation.fracture_log:
                    first_fracture = dict(simulation.fracture_log[0])
                    first_fracture["frame"] = frame_number
                    first_fracture["time_seconds"] = float(simulation.time)

                recorder.record(simulation)
    finally:
        with contextlib.suppress(Exception):
            ti.sync()

    if version_mismatch_errors:
        raise AssertionError("topology/coloring consistency violations: " + "; ".join(version_mismatch_errors))

    if not per_frame:
        raise AssertionError("no frames were measured")

    complete_times = np.asarray([row["complete_step_with_sync_seconds"] for row in per_frame], dtype=np.float64)
    host_only_times = np.asarray([row["host_only_seconds"] for row in per_frame], dtype=np.float64)
    sync_times = np.asarray([row["ti_sync_seconds"] for row in per_frame], dtype=np.float64)
    warmup_frames = min(5, len(per_frame) - 1) if len(per_frame) > 1 else 0
    steady = complete_times[warmup_frames:] if len(complete_times) > warmup_frames else complete_times

    state = simulation.particles
    edge_a = np.fromiter((edge.a for edge in simulation.mesh.edges), dtype=np.int64)
    edge_b = np.fromiter((edge.b for edge in simulation.mesh.edges), dtype=np.int64)
    edge_rest = np.fromiter((edge.rest_length for edge in simulation.mesh.edges), dtype=np.float64)
    edge_active = np.fromiter((edge.active for edge in simulation.mesh.edges), dtype=bool)
    edge_broken = np.fromiter((edge.broken for edge in simulation.mesh.edges), dtype=bool)
    strains = np.abs(
        np.linalg.norm(state.position[edge_a] - state.position[edge_b], axis=1) / edge_rest - 1.0
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "final_state.npz"
    # main() already refused to reuse an output directory holding this tool's
    # outputs; this is the second guard for the same promise in case the NPZ
    # was left behind by a run that failed before writing its JSON.
    if npz_path.exists() and not args.force:
        raise FileExistsError(f"{npz_path} already exists; rerun with --force or a fresh --output-dir")
    np.savez_compressed(
        npz_path,
        positions=state.position,
        velocity=state.velocity,
        triangles=simulation.mesh.triangles,
        mass=state.mass,
        inverse_mass=state.inverse_mass,
        pinned=state.pinned,
        edge_a=edge_a,
        edge_b=edge_b,
        edge_rest_length=edge_rest,
        edge_active=edge_active,
        edge_broken=edge_broken,
    )

    timing_summary = registry.summary()
    complete_total = float(complete_times.sum())
    accounted = (
        timing_summary["solver_step"]["total_seconds"]
        + timing_summary["fracture_scan"]["total_seconds"]
    )
    timing_summary["complete_step"]["unaccounted_seconds"] = complete_total - accounted
    actual_arch = str(ti.lang.impl.current_cfg().arch)
    expected_arch = ti.cpu if args.arch == "cpu" else ti.cuda
    if ti.lang.impl.current_cfg().arch != expected_arch:
        raise AssertionError(f"Taichi selected {actual_arch} while {expected_arch} was requested")

    result: dict[str, Any] = {
        "tool": {
            "name": "tools.measure_simulation",
            "version": 1,
            "repo_root": str(REPO_ROOT),
            "argv": sys.argv,
            "command": "python -m tools.measure_simulation " + " ".join(sys.argv[1:]),
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "taichi": ti.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
            "conda_env": "XPBD_tear",
        },
        "backend": {
            "arch_requested": args.arch,
            "arch_selected": backend,
            "actual_arch_enum": actual_arch,
            "precision_requested": args.precision,
        },
        "config": {
            "frames": args.frames,
            "warmup_frames": warmup_frames,
            "measured_frames": len(per_frame),
            "resolution": args.resolution,
            "demo_config": asdict(config),
            **_material_payload(simulation.solver.material, config),
        },
        "source_sha256": source_hashes,
        "timing": {
            "method": TIMING_METHOD_NOTES,
            "region_descriptions": TIMER_REGIONS,
            "per_frame_fields": {
                "frame": "0-based measured frame index",
                "complete_step_with_sync_seconds": "ClothSimulation.step() plus an explicit ti.sync()",
                "host_only_seconds": "same interval without the trailing ti.sync()",
                "ti_sync_seconds": "the trailing ti.sync() call itself",
            },
            "setup_seconds_including_taichi_init_and_first_jit": setup_seconds,
            "first_frame_seconds_including_jit": float(complete_times[0]),
            "complete_step": {
                "total_seconds": complete_total,
                "mean_seconds": float(complete_times.mean()),
                "median_seconds": float(np.median(complete_times)),
                "p95_seconds": float(np.percentile(complete_times, 95)),
                "min_seconds": float(complete_times.min()),
                "max_seconds": float(complete_times.max()),
            },
            "steady_complete_step": {
                "warmup_frames": warmup_frames,
                "mean_seconds": float(steady.mean()),
                "median_seconds": float(np.median(steady)),
                "p95_seconds": float(np.percentile(steady, 95)),
            },
            "host_only_mean_seconds": float(host_only_times.mean()),
            "ti_sync_mean_seconds": float(sync_times.mean()),
            "regions": timing_summary,
        },
        "per_frame": per_frame,
        "integrity": {
            "finite_all_frames": not finite_failures,
            "finite_failure_frames": finite_failures,
            "topology_consistency_errors": version_mismatch_errors,
            "topology_version": int(simulation.mesh.topology_version),
            "solver_topology_version": int(simulation.solver.topology_version),
            "coloring_version": int(simulation.solver.coloring_version),
            "repack_count": int(simulation.solver.repack_count),
            "coloring_recompute_count": int(simulation.solver.coloring_recompute_count),
        },
        "fracture": {
            "events": len(simulation.fracture_log),
            "log": list(simulation.fracture_log),
            "first_fracture": first_fracture,
            "last_fracture": dict(simulation.fracture_log[-1]) if simulation.fracture_log else None,
        },
        "topology": {
            "particle_count": int(simulation.particle_count),
            "initial_particle_count": int(args.resolution * config.resolution_y),
            "components": int(simulation.disconnected_components()),
            "active_edge_count": int(simulation.active_edge_count),
            "broken_edge_count": int(np.count_nonzero(edge_broken)),
            "maximum_edge_strain": float(strains.max()) if strains.size else 0.0,
            "maximum_edge_strain_active_only": float(strains[edge_active].max()) if edge_active.any() else 0.0,
            "inverted_triangles": int(simulation.inverted_triangles()) if hasattr(simulation, "inverted_triangles") else None,
            "diagnostics_report": recorder.topology_check(),
        },
        "artifacts": {
            "json": None,
            "npz": _relative_or_absolute(npz_path),
            "npz_arrays": ["positions", "velocity", "triangles", "mass", "inverse_mass", "pinned",
                           "edge_a", "edge_b", "edge_rest_length", "edge_active", "edge_broken"],
        },
        "notes": (
            "Numerical measurement only: no diagnostics PNG or GIF is produced here. "
            "Run the visualization entry points separately so image/GIF encoding time never "
            "enters the numerical timings."
        ),
    }
    if not recorder.topology_check()["passed"]:
        raise AssertionError("diagnostics topology/geometry check failed")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.measure_simulation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu", help="Taichi backend (no silent fallback)")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32", help="Taichi field precision")
    parser.add_argument("--frames", type=int, default=120, help="measured frames after construction")
    parser.add_argument("--resolution", type=int, default=8, help="horizontal grid resolution")
    parser.add_argument("--critical-strain", type=float, default=0.25, dest="critical_strain")
    parser.add_argument("--pull-speed", type=float, default=0.18, dest="pull_speed")
    parser.add_argument("--output-dir", type=Path, required=True, help="required output directory for this run")
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow an output directory that already contains this tool's JSON/NPZ output (never needed for a fresh directory)",
    )
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.resolution < 4:
        parser.error("--resolution must be at least 4")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    json_path = output_dir / "measurement.json"
    args.output_dir = output_dir

    # Never silently overwrite earlier measurement output: the point of this
    # tool is to preserve per-revision evidence, so an existing result needs an
    # explicit --force and a deliberate decision.
    if not args.force and output_dir.exists():
        existing = sorted(
            path.name
            for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".npz"}
        )
        if existing:
            print(
                f"refusing to reuse {output_dir}: it already holds {', '.join(existing)}; "
                "pass --force to replace them or choose a fresh --output-dir",
                file=sys.stderr,
            )
            return 2

    try:
        result = run_measurement(args)
    except Exception:
        traceback.print_exc()
        return 1

    # The JSON is written last and in one shot so a failing run cannot leave a
    # half-written report behind. Neither the NPZ nor this write is inside the
    # measured loop.
    result["artifacts"]["json"] = _relative_or_absolute(json_path)
    try:
        json_text = json.dumps(result, indent=2)
    except (TypeError, ValueError) as error:
        print(f"failed to serialise measurement JSON: {error}", file=sys.stderr)
        return 1
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text, encoding="utf-8")

    print(json.dumps(result["backend"], indent=2))
    print(json.dumps({
        "json": result["artifacts"]["json"],
        "npz": result["artifacts"]["npz"],
        "fracture_events": result["fracture"]["events"],
        "first_fracture": result["fracture"]["first_fracture"],
        "particle_count": result["topology"]["particle_count"],
        "components": result["topology"]["components"],
        "topology_report": result["topology"]["diagnostics_report"],
        "steady_complete_step": result["timing"]["steady_complete_step"],
        "regions": {
            name: payload["total_seconds"] for name, payload in result["timing"]["regions"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
