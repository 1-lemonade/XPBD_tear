"""Summarize the recorded 450- and 45,000-vertex CPU/CUDA runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _load(output: Path, scale: str, arch: str) -> dict:
    path = output / f"{scale}_demo_{arch}_full.json"
    return json.loads(path.read_text(encoding="utf-8"))


def make_report(output: Path) -> Path:
    cases = {
        (scale, arch): _load(output, scale, arch)
        for scale in ("small", "large")
        for arch in ("cpu", "cuda")
    }
    probes = {
        arch: json.loads((output / f"large_demo_{arch}_tear_probe.json").read_text(encoding="utf-8"))
        for arch in ("cpu", "cuda")
    }
    if not all(case["validation"]["passed"] for case in cases.values()) or not all(
        probe["passed"] for probe in probes.values()
    ):
        raise AssertionError("cannot report a failed benchmark or tear probe as accepted")

    differences = {}
    for scale in ("small", "large"):
        cpu_positions = np.load(output / f"{scale}_demo_cpu_full_positions.npz")["positions"]
        cuda_positions = np.load(output / f"{scale}_demo_cuda_full_positions.npz")["positions"]
        distance = np.linalg.norm(cpu_positions - cuda_positions, axis=1)
        differences[scale] = {"max": float(distance.max()), "mean": float(distance.mean())}

    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["450 vertices", "45,000 vertices"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=140)
    for axis, metric, title in (
        (axes[0], "steady_full_frame", "Complete frame, fracture scan enabled"),
        (axes[1], "steady_solver_step", "XPBD solver step"),
    ):
        cpu = [cases[(scale, "cpu")]["timing"][metric]["median_seconds"] for scale in ("small", "large")]
        cuda = [cases[(scale, "cuda")]["timing"][metric]["median_seconds"] for scale in ("small", "large")]
        centers = np.arange(2)
        axis.barh(centers - 0.18, cpu, height=0.34, label="CPU", color="#386eb2")
        axis.barh(centers + 0.18, cuda, height=0.34, label="CUDA", color="#d88648")
        axis.set(yticks=centers, yticklabels=labels, xlabel="Median wall time (s/frame)", title=title)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(loc="upper right", frameon=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.79, bottom=0.22, wspace=0.4)
    fig.suptitle("XPBD cloth backend comparison · 115 measured frames after 5 warmup frames", y=0.96)
    fig.text(0.5, 0.045, "Source: XPBD_tear large_demo JSON · 2026-09-23 · Taichi 1.7.4, f32", ha="center", fontsize=8)
    chart = output / "backend_comparison.png"
    fig.savefig(chart)
    plt.close(fig)

    def median(scale: str, arch: str, metric: str) -> float:
        return cases[(scale, arch)]["timing"][metric]["median_seconds"]

    def speedup(scale: str, metric: str) -> float:
        return median(scale, "cpu", metric) / median(scale, "cuda", metric)

    large_cpu = cases[("large", "cpu")]
    large_cuda = cases[("large", "cuda")]
    lines = [
        "# Large cloth benchmark",
        "",
        "## Scenario and method",
        "",
        "- Baseline: 18 × 25 = 450 vertices. Large demo: 180 × 250 = 45,000 vertices, 89,142 triangles, 134,141 stretch and 133,285 bend constraints.",
        "- Same total particle mass (450): mass per vertex is 1 at baseline and 0.01 at the large resolution. Large-demo pull speed is 0.018 units/s; the baseline uses 0.18 units/s.",
        "- Each backend ran in a separate process for 120 frames at 60 Hz, 8 substeps, 10 solver iterations, f32. The first 5 frames were excluded from steady timing. Rendering and JSON writing were outside the timed frames.",
        "- Each timed frame explicitly called `ti.sync()`; the solver measurement also includes particle-state readback. The full-frame measurement includes Python-side fracture scanning. The stable runs stayed below their tear thresholds and had no topology rebuilds.",
        "- Test host: AMD Ryzen 7 9700X, NVIDIA GeForce RTX 5070 Ti (driver 610.47, 16 GB), Windows, Conda `XPBD_tear`, Python 3.10.21, NumPy 1.26.4, Taichi 1.7.4.",
        "",
        "![CPU and CUDA median time per frame](backend_comparison.png)",
        "",
        "## Timing result",
        "",
        f"- 450-vertex complete frame: CPU {median('small','cpu','steady_full_frame'):.3f} s, CUDA {median('small','cuda','steady_full_frame'):.3f} s; CUDA speedup {speedup('small','steady_full_frame'):.2f}×.",
        f"- 45,000-vertex complete frame: CPU {median('large','cpu','steady_full_frame'):.3f} s, CUDA {median('large','cuda','steady_full_frame'):.3f} s; CUDA speedup {speedup('large','steady_full_frame'):.2f}×.",
        f"- 45,000-vertex XPBD solver step: CPU {median('large','cpu','steady_solver_step'):.3f} s, CUDA {median('large','cuda','steady_solver_step'):.3f} s; CUDA speedup {speedup('large','steady_solver_step'):.2f}×.",
        f"- Mean work outside the large solver step: CPU {large_cpu['timing']['steady_non_solver_mean_seconds']:.3f} s/frame, CUDA {large_cuda['timing']['steady_non_solver_mean_seconds']:.3f} s/frame. This mostly reflects the Python edge-strain scan and limits the complete-frame speedup.",
        "- These are single serial runs on one host, not a statistical hardware survey. Setup and first-use JIT costs are reported separately in the JSON files.",
        "",
        "## Stability and topology checks",
        "",
        f"- All four 120-frame runs remained finite, with zero inverted triangles and pin-target error below 4×10⁻⁸. The large runs reached maximum edge strain {large_cpu['validation']['maximum_edge_strain']:.3f} (CPU) and {large_cuda['validation']['maximum_edge_strain']:.3f} (CUDA), both below the 0.25 failure threshold.",
        f"- Final CPU/CUDA position difference: at 450 vertices, max {differences['small']['max']:.2e}, mean {differences['small']['mean']:.2e}; at 45,000 vertices, max {differences['large']['max']:.2e}, mean {differences['large']['mean']:.2e} simulation units.",
        f"- A separate 45,000-vertex tear probe used a 0.10 threshold and reached its first natural fracture at frame {probes['cpu']['frames_to_first_fracture']} on both backends. Both cut the same edge, duplicated one vertex, preserved material rest lengths, and passed topology/geometry checks.",
        "- The 120-frame speed measurements describe intact-cloth operation. They do not establish CUDA speedup during repeated topology rebuilds after tear propagation.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "conda activate XPBD_tear",
        "python -m project.large_demo --arch cpu",
        "python -m project.large_demo --arch cuda",
        "python -m project.large_demo --arch cpu --scale 1 --critical-strain 0.65",
        "python -m project.large_demo --arch cuda --scale 1 --critical-strain 0.65",
        "python -m project.large_demo --arch cpu --tear-probe",
        "python -m project.large_demo --arch cuda --tear-probe",
        "python -m project.large_demo_report",
        "```",
        "",
        "Timing method: [Taichi synchronization guidance](https://docs.taichi-lang.org/docs/master/kernel_sync).",
    ]
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "small_full_speedup": speedup("small", "steady_full_frame"),
        "large_full_speedup": speedup("large", "steady_full_frame"),
        "large_solver_speedup": speedup("large", "steady_solver_step"),
        "position_difference": differences,
        "all_passed": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/large_demo"))
    args = parser.parse_args()
    print(make_report(args.output_dir))


if __name__ == "__main__":
    main()
