"""Reproducible 100x-vertex XPBD cloth run and synchronized backend benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

from .cloth.taichi_runtime import initialize_taichi
from .main import ClothSimulation, DemoConfig


def _statistics(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_seconds": float(values.mean()),
        "median_seconds": float(np.median(values)),
        "p95_seconds": float(np.percentile(values, 95)),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
    }


def _render_grid(simulation: ClothSimulation, output: Path, backend: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nx, ny = simulation.config.resolution_x, simulation.config.resolution_y
    points = simulation.particles.position[: nx * ny, :2].reshape(ny, nx, 2)
    fig, ax = plt.subplots(figsize=(6, 8), dpi=130, constrained_layout=True)
    for row in sorted(set(range(0, ny, 10)) | {ny - 1}):
        ax.plot(points[row, :, 0], points[row, :, 1], color="#386eb2", linewidth=0.65)
    for column in sorted(set(range(0, nx, 10)) | {nx - 1}):
        ax.plot(points[:, column, 0], points[:, column, 1], color="#386eb2", linewidth=0.65)
    ax.set(xlabel="x (simulation units)", ylabel="y (simulation units)")
    ax.set_title(f"{simulation.particle_count:,}-vertex cloth · {backend} · t={simulation.time:.2f} s")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    fig.savefig(output)
    plt.close(fig)


def run(
    arch: str,
    frames: int,
    warmup: int,
    output_dir: Path,
    *,
    scale: int,
    tearing: bool,
    critical_strain: float,
    pull_speed: float | None,
    render: bool,
) -> dict[str, object]:
    if frames < 2 or not 0 <= warmup < frames:
        raise ValueError("frames must be at least 2 and warmup must be in [0, frames)")
    if scale not in (1, 10):
        raise ValueError("scale must be 1 or 10")
    if pull_speed is None:
        pull_speed = 0.18 / scale
    if critical_strain <= 0.0 or pull_speed < 0.0:
        raise ValueError("critical_strain must be positive and pull_speed must be nonnegative")
    backend = initialize_taichi(arch, "f32")
    config = DemoConfig(
        resolution_x=18 * scale,
        resolution_y=25 * scale,
        critical_strain=critical_strain,
        pull_speed=pull_speed,
        tearing_enabled=tearing,
        arch=arch,
        precision="f32",
        particle_mass=1.0 / (scale * scale),  # Keep total cloth mass at 450.
    )
    setup_started = time.perf_counter()
    simulation = ClothSimulation(config)
    setup_seconds = time.perf_counter() - setup_started
    solver_times: list[float] = []
    frame_times: list[float] = []
    original_solver_step = simulation.solver.step

    def timed_solver_step(*args, **kwargs):
        started = time.perf_counter()
        result = original_solver_step(*args, **kwargs)
        ti.sync()
        solver_times.append(time.perf_counter() - started)
        return result

    simulation.solver.step = timed_solver_step
    finite_all_frames = True
    maximum_inverted_triangles = 0
    for frame in range(frames):
        started = time.perf_counter()
        simulation.step()
        ti.sync()  # CUDA launches are asynchronous; include completed work.
        frame_times.append(time.perf_counter() - started)
        finite_all_frames = finite_all_frames and simulation.finite()
        xy_frame = simulation.particles.position[:, :2]
        faces_frame = simulation.mesh.triangles
        first_frame = xy_frame[faces_frame[:, 1]] - xy_frame[faces_frame[:, 0]]
        second_frame = xy_frame[faces_frame[:, 2]] - xy_frame[faces_frame[:, 0]]
        signed_frame = first_frame[:, 0] * second_frame[:, 1] - first_frame[:, 1] * second_frame[:, 0]
        maximum_inverted_triangles = max(maximum_inverted_triangles, int(np.count_nonzero(signed_frame <= 0.0)))
        if (frame + 1) % 30 == 0 or frame + 1 == frames:
            print(f"{arch} {'full' if tearing else 'solver'}: {frame + 1}/{frames} frames", flush=True)

    xy = simulation.particles.position[:, :2]
    triangles = simulation.mesh.triangles
    first = xy[triangles[:, 1]] - xy[triangles[:, 0]]
    second = xy[triangles[:, 2]] - xy[triangles[:, 0]]
    signed_area_twice = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    edge_a = np.fromiter((edge.a for edge in simulation.mesh.edges), dtype=np.int32)
    edge_b = np.fromiter((edge.b for edge in simulation.mesh.edges), dtype=np.int32)
    rest = np.fromiter((edge.rest_length for edge in simulation.mesh.edges), dtype=np.float64)
    strains = np.linalg.norm(simulation.particles.position[edge_a] - simulation.particles.position[edge_b], axis=1) / rest - 1.0
    top_error = max(float(np.linalg.norm(simulation.particles.position[pid] - target)) for pid, target in simulation.top_targets.items())
    bottom_error = max(
        float(np.linalg.norm(simulation.particles.position[pid] - (start + np.array((0.0, -simulation.pull_offset, 0.0)))))
        for pid, start in simulation.bottom_start.items()
    )
    mode = "full" if tearing else "solver"
    steady_frames = frame_times[warmup:]
    steady_solver = solver_times[warmup:]
    result: dict[str, object] = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "taichi": ti.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "scenario": {
            "arch_requested": arch,
            "arch_selected": backend,
            "mode": mode,
            "scale": scale,
            "resolution_x": config.resolution_x,
            "resolution_y": config.resolution_y,
            "frames": frames,
            "warmup_frames": warmup,
            "measured_frames": frames - warmup,
            "dt": config.dt,
            "iterations": config.iterations,
            "substeps": config.substeps,
            "critical_strain": critical_strain,
            "pull_speed": pull_speed,
            "particle_mass": config.particle_mass,
            "total_particle_mass": float(simulation.particles.mass.sum()),
            "tearing_enabled": tearing,
        },
        "mesh": {
            "initial_vertices": config.resolution_x * config.resolution_y,
            "final_vertices": simulation.particle_count,
            "triangles": simulation.mesh.triangle_count,
            "initial_stretch_constraints": len(simulation.solver.stretch_constraints),
            "initial_bend_constraints": len(simulation.solver.bend_constraints),
        },
        "timing": {
            "setup_seconds_excluding_taichi_init": setup_seconds,
            "first_frame_seconds_including_jit": frame_times[0],
            "steady_full_frame": _statistics(steady_frames),
            "steady_solver_step": _statistics(steady_solver),
            "steady_non_solver_mean_seconds": float(np.mean(np.asarray(steady_frames) - np.asarray(steady_solver))),
        },
        "validation": {
            "finite_all_frames": finite_all_frames,
            "maximum_inverted_triangles": maximum_inverted_triangles,
            "minimum_signed_area_twice": float(signed_area_twice.min()),
            "maximum_edge_strain": float(strains.max()),
            "edge_strain_below_threshold": bool(float(strains.max()) < critical_strain),
            "maximum_top_pin_error": top_error,
            "maximum_bottom_pin_error": bottom_error,
            "fracture_events": len(simulation.fracture_log),
            "topology_version_matches_solver": simulation.mesh.topology_version == simulation.solver.topology_version,
        },
    }
    result["validation"]["passed"] = bool(
        result["validation"]["finite_all_frames"]
        and result["validation"]["maximum_inverted_triangles"] == 0
        and result["validation"]["edge_strain_below_threshold"]
        and result["validation"]["fracture_events"] == 0
        and top_error < 1.0e-4
        and bottom_error < 1.0e-4
        and result["validation"]["topology_version_matches_solver"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{'large' if scale == 10 else 'small'}_demo_{arch}_{mode}"
    (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / f"{stem}_positions.npz", positions=simulation.particles.position)
    if render:
        _render_grid(simulation, output_dir / f"{stem}.png", backend)
    print(json.dumps(result, indent=2), flush=True)
    if not result["validation"]["passed"]:
        raise AssertionError("large demo validation failed")
    return result


def run_tear_probe(arch: str, max_frames: int, output_dir: Path, *, critical_strain: float = 0.1) -> dict[str, object]:
    """Stop after the first natural fracture on the 45,000-vertex cloth."""
    if critical_strain <= 0.0 or max_frames < 1:
        raise ValueError("probe threshold and max_frames must be positive")
    backend = initialize_taichi(arch, "f32")
    simulation = ClothSimulation(
        DemoConfig(
            resolution_x=180,
            resolution_y=250,
            particle_mass=0.01,
            critical_strain=critical_strain,
            pull_speed=0.018,
            arch=arch,
        )
    )
    elapsed = 0.0
    for frame in range(1, max_frames + 1):
        started = time.perf_counter()
        simulation.step()
        ti.sync()
        elapsed += time.perf_counter() - started
        if not simulation.finite():
            raise AssertionError("large tear probe produced non-finite state")
        if simulation.fracture_log:
            break
    else:
        raise AssertionError(f"no fracture in {max_frames} large-demo frames")

    xy = simulation.particles.position[:, :2]
    faces = simulation.mesh.triangles
    first = xy[faces[:, 1]] - xy[faces[:, 0]]
    second = xy[faces[:, 2]] - xy[faces[:, 0]]
    signed_area_twice = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    cut_edges = [edge for edge in simulation.mesh.edges if edge.broken]
    result: dict[str, object] = {
        "arch_requested": arch,
        "arch_selected": backend,
        "resolution_x": 180,
        "resolution_y": 250,
        "initial_vertices": 45000,
        "critical_strain": critical_strain,
        "pull_speed": 0.018,
        "frames_to_first_fracture": frame,
        "elapsed_seconds_including_jit": elapsed,
        "first_fracture": simulation.fracture_log[0],
        "final_vertices": simulation.particle_count,
        "cut_edges": len(cut_edges),
        "cut_shared_edge_observed": any(not edge.active or len(edge.adjacent_triangles) == 1 for edge in cut_edges),
        "inverted_triangles": int(np.count_nonzero(signed_area_twice <= 0.0)),
        "finite": simulation.finite(),
        "topology_version_matches_solver": simulation.mesh.topology_version == simulation.solver.topology_version,
        "coloring_version_matches_mesh": simulation.mesh.topology_version == simulation.solver.coloring_version,
        "rest_lengths_preserved": all(
            abs(edge.rest_length - simulation.mesh.rest_distance(edge.a, edge.b)) < 1.0e-10
            for edge in simulation.mesh.edges
        ),
    }
    result["passed"] = bool(
        result["cut_shared_edge_observed"]
        and result["inverted_triangles"] == 0
        and result["finite"]
        and result["topology_version_matches_solver"]
        and result["coloring_version_matches_mesh"]
        and result["rest_lengths_preserved"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"large_demo_{arch}_tear_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise AssertionError("large tear probe failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--scale", type=int, choices=(1, 10), default=10)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/large_demo"))
    parser.add_argument("--no-tearing", action="store_true", help="measure the solver path without the fracture scan")
    parser.add_argument("--critical-strain", type=float, default=0.25)
    parser.add_argument("--pull-speed", type=float, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--tear-probe", action="store_true", help="validate the first natural large-mesh fracture")
    parser.add_argument("--probe-critical-strain", type=float, default=0.1)
    args = parser.parse_args()
    if args.tear_probe:
        if args.scale != 10:
            parser.error("--tear-probe requires --scale 10")
        run_tear_probe(args.arch, args.frames, args.output_dir, critical_strain=args.probe_critical_strain)
        return
    run(
        args.arch,
        args.frames,
        args.warmup,
        args.output_dir,
        scale=args.scale,
        tearing=not args.no_tearing,
        critical_strain=args.critical_strain,
        pull_speed=args.pull_speed,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
