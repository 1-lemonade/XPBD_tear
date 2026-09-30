"""Audit read-only surface queries and simulation equivalence."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc

import numpy as np


def source_hashes():
    paths = sorted(Path("project").rglob("*.py")) + [
        Path("tools/measure_coupling.py"), Path("environment.yml"), Path("requirements.txt")
    ]
    return {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "median_seconds": float(np.median(values)),
            "min_seconds": float(np.min(values)), "max_seconds": float(np.max(values))}


def state_record(sim):
    return {
        "positions": sim.particles.position.copy(),
        "velocity": sim.particles.velocity.copy(),
        "triangles": sim.mesh.triangles.copy(),
        "edges": [(e.id, e.a, e.b, tuple(e.adjacent_triangles), e.active, e.broken)
                  for e in sim.mesh.edges],
        "fracture_log": [dict(event) for event in sim.fracture_log],
        "versions": (sim.mesh.topology_version, sim.solver.topology_version,
                     sim.solver.coloring_version, sim.solver.coloring_recompute_count),
        "time": sim.time,
    }


def compare_trajectories(reference, queried):
    """Compare independent processes with fixed tolerances and discrete events."""
    assert len(reference) == len(queried) == 120
    errors = {"position": 0.0, "velocity": 0.0, "strain": 0.0, "time": 0.0}
    for frame, (a, b) in enumerate(zip(reference, queried), 1):
        for key, name, tolerance in (("positions", "position", 1e-5),
                                     ("velocity", "velocity", 5e-4)):
            av, bv = np.asarray(a[key]), np.asarray(b[key])
            assert av.shape == bv.shape, (frame, key, "shape")
            error = float(np.max(np.abs(av - bv)))
            errors[name] = max(errors[name], error)
            assert error <= tolerance, (frame, key, error)
        assert np.array_equal(a["triangles"], b["triangles"]), (frame, "triangles")
        for key in ("edges", "versions"):
            # Normalize tuples/NumPy values to the same representation as the
            # CLI's JSON payload; also support direct state_record callers.
            assert json.dumps(a[key], default=json_default) == json.dumps(b[key], default=json_default), (frame, key)
        errors["time"] = max(errors["time"], abs(a["time"] - b["time"]))
        assert errors["time"] <= 1e-10
        assert len(a["fracture_log"]) == len(b["fracture_log"]), frame
        for ea, eb in zip(a["fracture_log"], b["fracture_log"]):
            assert ea["edge"] == eb["edge"], frame
            assert abs(ea["time"] - eb["time"]) <= 1e-10, frame
            error = abs(ea["strain"] - eb["strain"])
            errors["strain"] = max(errors["strain"], error)
            assert error <= 1e-5, (frame, "strain", error)
    return errors


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def check_architecture():
    checked = []
    for path in sorted(Path("project").rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        if "cloth" in path.parts:
            assert not any("viz" in name.split(".") for name in imports), path
        if path.name == "constraints.py":
            assert not any(set(name.split(".")) & {"fracture", "coupling", "mpm"} for name in imports), path
        if path.as_posix() in ("project/coupling/contracts.py", "project/coupling/mapping.py"):
            allowed = set(sys.stdlib_module_names) | {"numpy", "contracts"}
            assert all(name.split(".")[0] in allowed for name in imports), (path, imports)
        checked.append(path.as_posix())
    return {"passed": True, "checked_files": checked}


def run(args):
    import taichi as ti
    from project.main import ClothSimulation, DemoConfig
    from project.cloth.taichi_runtime import initialize_taichi
    from project.coupling import QueryPoints, QueryOnlyCoupling, MappingConfig
    from project.coupling.mapping import map_points
    from project.tests.test_coupling import live_state_bytes

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"choose a new evidence directory: {output}")
    output.mkdir(parents=True)
    initial_hashes = source_hashes()
    actual = initialize_taichi(args.arch, args.precision)
    config = DemoConfig(resolution_x=8, resolution_y=11, dt=1/60, substeps=8,
                        iterations=10, critical_strain=.25, pull_speed=.18,
                        arch=args.arch, precision=args.precision)
    sim = ClothSimulation(config)
    rng = np.random.default_rng(240930)
    positions = rng.uniform([-.25, -.3, -.2], [1.25, 1.5, .2], (128, 3))
    queries = QueryPoints(np.arange(128, dtype=np.int64) * 13 - 100, positions)
    mapping_config = MappingConfig()
    snapshot_times, mapping_times, audit_times, step_times, raw_step_times = [], [], [], [], []
    trajectories, mappings, counts = [], [], []
    snapshot_method = sim.snapshot

    def timed_snapshot():
        start = time.perf_counter()
        snapshot = snapshot_method()
        snapshot_times.append(time.perf_counter() - start)
        return snapshot

    sim.snapshot = timed_snapshot

    class AuditedQuery(QueryOnlyCoupling):
        def exchange(self, snapshot, time_value, dt):
            audit_start = time.perf_counter()
            before = live_state_bytes(sim)
            audit = time.perf_counter() - audit_start
            start = time.perf_counter()
            super().exchange(snapshot, time_value, dt)
            mapping_times.append(time.perf_counter() - start)
            audit_start = time.perf_counter()
            assert before == live_state_bytes(sim), "exchange changed live host/device state"
            audit_times.append(audit + time.perf_counter() - audit_start)
            self.last_snapshot = snapshot

    if args.mode == "query":
        sim.coupling = AuditedQuery(queries, mapping_config)
    old = snapshot_method()
    old_bytes = {name: value.tobytes() for name, value in vars(old).items() if isinstance(value, np.ndarray)}
    events = 0
    for frame in range(1, 121):
        start = time.perf_counter()
        sim.step()
        ti.sync()
        elapsed = time.perf_counter() - start
        if args.mode == "query":
            assert len(audit_times) == len(mapping_times) == frame
            assert sim.coupling.last_result.step_index == frame
        raw_step_times.append(elapsed)
        step_times.append(elapsed - (audit_times[-1] if args.mode == "query" else 0))
        assert sim.step_index == frame and sim.epoch == 1
        assert sim.finite()
        assert sim.mesh.topology_version == sim.solver.topology_version == sim.solver.coloring_version
        trajectories.append(state_record(sim))
        counts.append({"N": sim.particle_count, "T": sim.mesh.triangle_count, "Q": 128 if args.mode == "query" else 0})
        if args.mode == "query":
            result, snap = sim.coupling.last_result, sim.coupling.last_snapshot
            assert result.version == snap.version == (sim.epoch, frame, sim.mesh.topology_version)
            np.testing.assert_array_equal(result.query_ids, queries.query_ids)
            # Simulation IDs equal rows here. The analytic tests cover ID permutations.
            np.testing.assert_array_equal(result.vertex_ids, snap.vertex_ids[snap.triangles[result.triangle_ids]])
            reconstructed = np.einsum("qi,qij->qj", result.barycentric, snap.positions[snap.triangles[result.triangle_ids]])
            np.testing.assert_allclose(reconstructed, result.closest_point, atol=1e-9, rtol=1e-9)
            assert np.isfinite(result.distance).all()
            if len(sim.fracture_log) > events or frame in (1, 120):
                mappings.append({"frame": frame, "version": result.version,
                                 "new_fracture": len(sim.fracture_log) > events,
                                 **{name: value for name, value in vars(result).items() if isinstance(value, np.ndarray)}})
        events = len(sim.fracture_log)
    for name, value in old_bytes.items():
        assert getattr(old, name).tobytes() == value, "old snapshot changed"
    assert events == 15 and sim.particle_count == 98, (events, sim.particle_count)
    large_times = []
    peak_mapping_bytes = None
    if args.mode == "query":
        snap = snapshot_method()
        large = QueryPoints(np.arange(1024, dtype=np.int64) * 17,
                            rng.uniform([-.25, -.3, -.2], [1.25, 1.5, .2], (1024, 3)))
        previous = None
        for _ in range(5):
            before = live_state_bytes(sim)
            start = time.perf_counter()
            result = map_points(snap, large, mapping_config)
            large_times.append(time.perf_counter() - start)
            assert before == live_state_bytes(sim)
            if previous is not None:
                for name, value in vars(result).items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_array_equal(value, getattr(previous, name))
            previous = result
        # Separate memory observation so tracemalloc cannot distort timed runs.
        before = live_state_bytes(sim)
        tracemalloc.start()
        memory_result = map_points(snap, large, mapping_config)
        _, peak_mapping_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert before == live_state_bytes(sim)
        del memory_result
    assert source_hashes() == initial_hashes, "source changed during this measurement"
    payload = {
        "passed": True, "architecture": check_architecture(),
        "mode": args.mode, "backend": actual, "actual_arch": str(ti.lang.impl.current_cfg().arch),
        "precision": args.precision, "environment": {"python": sys.version, "numpy": np.__version__,
            "taichi": ti.__version__, "platform": platform.platform(), "processor": platform.processor()},
        "config": asdict(config), "material": asdict(sim.solver.material), "mapping_config": asdict(mapping_config),
        "source_hashes": initial_hashes, "trajectory": trajectories, "mapping_frames": mappings,
        "queries": asdict(queries) if args.mode == "query" else None,
        "query_generator": {"seed": 240930, "bounds": [[-.25, -.3, -.2], [1.25, 1.5, .2]]},
        "counts": counts, "fracture_log": sim.fracture_log, "final_vertices": sim.particle_count,
        "first_fracture_frame": next(i+1 for i, state in enumerate(trajectories) if state["fracture_log"]),
        "snapshot_seconds": snapshot_times, "complete_step_seconds": step_times,
        "raw_complete_step_with_audit_seconds": raw_step_times,
        "mapping_seconds": mapping_times, "audit_seconds": audit_times,
        "timing_summary": {"snapshot": stats(snapshot_times), "complete_step": stats(step_times),
            "complete_step_steady_after_5": stats(step_times[5:])},
        "memory_policy": {"description": "NumPy host mapper; O(T) temporary storage, no Q*T*3 array",
                          "query_chunk_size": 1, "triangle_chunk_size": sim.mesh.triangle_count,
                          "measured_peak_bytes_q1024": peak_mapping_bytes,
                          "memory_run_separate_from_timing": True},
        "timing_method": "Host perf_counter; complete step includes trailing ti.sync and first-step JIT. "
            "Zero-write audit durations subtracted; raw durations also retained. Nested snapshot/mapping timers "
            "must not be added to complete step. Mapping includes surface table construction and all diagnostics. "
            "Synchronous exchange finishes the current audit before step returns; frame-count assertions verify this. "
            "Validation may perturb cache/scheduling. Copy includes contract validation.",
        "units": "Model length/time units, not calibrated SI. Timings in wall seconds.",
    }
    if large_times:
        payload["large_query_seconds"] = large_times
        payload["timing_summary"].update(mapping_q128=stats(mapping_times), mapping_q1024=stats(large_times))
    (output / "measurement.json").write_text(json.dumps(payload, default=json_default, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "mode": args.mode, "backend": actual, "precision": args.precision,
                      "events": events, "final_vertices": sim.particle_count,
                      "timing": payload["timing_summary"]}, indent=2))


def compare(args):
    root = Path(args.output_dir)
    comparisons = {}
    baseline_hashes = None
    for backend, precision in (("cpu", "f32"), ("cuda", "f32"), ("cpu", "f64")):
        pair = [json.loads((root / f"{backend}_{precision}_{mode}" / "measurement.json").read_text(encoding="utf-8"))
                for mode in ("null", "query")]
        for item in pair:
            assert item["passed"] and item["backend"] == backend and item["precision"] == precision
            assert item["source_hashes"] == source_hashes(), "source changed since measurements"
            if baseline_hashes is None:
                baseline_hashes = item["source_hashes"]
            assert baseline_hashes == item["source_hashes"]
        comparisons[f"{backend}_{precision}"] = compare_trajectories(pair[0]["trajectory"], pair[1]["trajectory"])
    output = root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({"passed": True, "max_errors": comparisons, "source_hashes": baseline_hashes}, indent=2), encoding="utf-8")
    print(output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32")
    parser.add_argument("--mode", choices=("null", "query", "compare"), default="query")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    compare(args) if args.mode == "compare" else run(args)
