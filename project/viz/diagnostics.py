"""Headless matplotlib diagnostics for completed or in-progress runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .style import FIGURE_DPI, FIGURE_SIZE, PALETTE, STRAIN_CMAP, apply_plot_style, style_axes


class DiagnosticsRecorder:
    """Collect numerical observables without coupling them to the solver."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.max_strain: list[float] = []
        self.mean_strain: list[float] = []
        self.edge_strains: list[dict[str, float]] = []
        self.active_constraints: list[int] = []
        self.fracture_events: list[int] = []
        self.components: list[int] = []
        self.cut_edges: list[int] = []
        self.particle_counts: list[int] = []
        self.inverted_triangles: list[int] = []

    def record(self, simulation: Any) -> None:
        strains = {
            f"{edge.a}-{edge.b}": edge.strain(simulation.particles.position)
            for edge in simulation.mesh.edges
            if edge.active
        }
        values = np.asarray(list(strains.values()), dtype=np.float64)
        self.times.append(float(simulation.time))
        self.max_strain.append(float(values.max()) if values.size else 0.0)
        self.mean_strain.append(float(values.mean()) if values.size else 0.0)
        self.edge_strains.append(strains)
        self.active_constraints.append(
            len(simulation.solver.stretch_constraints)
            + len(simulation.solver.bend_constraints)
            + len(simulation.solver.pin_constraints)
        )
        self.fracture_events.append(len(simulation.fracture_log))
        self.components.append(int(simulation.disconnected_components()))
        self.cut_edges.append(sum(edge.broken for edge in simulation.mesh.edges))
        self.particle_counts.append(int(simulation.particle_count))
        xy = simulation.particles.position[:, :2]
        triangles = simulation.mesh.triangles
        first = xy[triangles[:, 1]] - xy[triangles[:, 0]]
        second = xy[triangles[:, 2]] - xy[triangles[:, 0]]
        signed_area_twice = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        self.inverted_triangles.append(int(np.count_nonzero(signed_area_twice <= 0.0)))

    def topology_check(self) -> dict[str, int | bool]:
        initial = self.components[0] if self.components else 0
        maximum = max(self.components, default=initial)
        fractures = self.fracture_events[-1] if self.fracture_events else 0
        cut_edges = max(self.cut_edges, default=0)
        duplicated_vertices = max(self.particle_counts, default=0) - (self.particle_counts[0] if self.particle_counts else 0)
        geometric_split = cut_edges > 0 and duplicated_vertices > 0
        max_inverted = max(self.inverted_triangles, default=0)
        return {
            "initial_components": initial,
            "maximum_components": maximum,
            "fracture_events": fractures,
            "cut_edges": cut_edges,
            "duplicated_vertices": duplicated_vertices,
            "geometric_split_observed": geometric_split,
            "disconnected_components_observed": maximum > initial,
            "maximum_inverted_triangles": max_inverted,
            "passed": bool((fractures == 0 or geometric_split) and max_inverted == 0),
        }


def write_diagnostics(recorder: DiagnosticsRecorder, output_dir: str | Path) -> dict[str, Path]:
    """Write PNG plots and a machine-readable topology check."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Keep headless font/cache writes inside the requested artifact directory;
    # restricted Windows accounts may not be able to write the user profile.
    mpl_config = output / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_plot_style(plt)

    times = np.asarray(recorder.times)

    strain_path = output / "strain_over_time.png"
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True)
    axes[0].plot(times, recorder.max_strain, label="maximum", color=PALETTE["blue"], linewidth=2.2)
    axes[0].plot(times, recorder.mean_strain, label="mean", color=PALETTE["teal"], linewidth=1.8)
    axes[0].set_title("Active-edge strain")
    axes[0].set(xlabel="time (s)", ylabel="engineering strain")
    axes[0].legend(frameon=False, ncols=2)
    style_axes(axes[0])
    keys = sorted({key for sample in recorder.edge_strains for key in sample})
    if keys and recorder.edge_strains:
        heat = np.full((len(keys), len(recorder.edge_strains)), np.nan)
        key_index = {key: i for i, key in enumerate(keys)}
        for column, sample in enumerate(recorder.edge_strains):
            for key, value in sample.items():
                heat[key_index[key], column] = value
        x_start, x_end = (float(times[0]), float(times[-1]))
        if x_start == x_end:
            x_start -= 0.5
            x_end += 0.5
        image = axes[1].imshow(
            heat,
            aspect="auto",
            origin="lower",
            extent=(x_start, x_end, 0, len(keys)),
            interpolation="nearest",
            cmap=STRAIN_CMAP,
        )
        fig.colorbar(image, ax=axes[1], label="engineering strain", shrink=0.9)
    axes[1].set_title("Edge strain history")
    axes[1].set(xlabel="time (s)", ylabel="active edge index")
    style_axes(axes[1])
    fig.savefig(strain_path, dpi=FIGURE_DPI)
    plt.close(fig)

    events_path = output / "fracture_events.png"
    fig, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True)
    axes[0].plot(times, recorder.fracture_events, color=PALETTE["red"], linewidth=2.2, label="tears")
    axes[0].set_title("Fracture events")
    axes[0].set(xlabel="time (s)", ylabel="fracture events")
    axes[0].legend(frameon=False)
    style_axes(axes[0])
    axes[1].plot(times, recorder.active_constraints, color=PALETTE["purple"], linewidth=2.2, label="constraints")
    axes[1].set_title("Active constraints")
    axes[1].set(xlabel="time (s)", ylabel="active constraints")
    axes[1].legend(frameon=False)
    style_axes(axes[1])
    fig.savefig(events_path, dpi=FIGURE_DPI)
    plt.close(fig)

    topology_path = output / "topology_check.png"
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True)
    ax.plot(times, recorder.components, color=PALETTE["teal"], linewidth=2.2, label="components")
    ax.set(xlabel="time (s)", ylabel="triangle components", title="Topology connectivity")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(topology_path, dpi=FIGURE_DPI)
    plt.close(fig)

    check_path = output / "topology_check.json"
    check_path.write_text(json.dumps(recorder.topology_check(), indent=2), encoding="utf-8")
    return {
        "strain": strain_path,
        "events": events_path,
        "topology": topology_path,
        "topology_check": check_path,
    }
