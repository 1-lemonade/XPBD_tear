"""Headless matplotlib diagnostics for completed or in-progress runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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

    def topology_check(self) -> dict[str, int | bool]:
        initial = self.components[0] if self.components else 0
        maximum = max(self.components, default=initial)
        fractures = self.fracture_events[-1] if self.fracture_events else 0
        return {
            "initial_components": initial,
            "maximum_components": maximum,
            "fracture_events": fractures,
            "disconnected_components_observed": maximum > initial,
            "passed": bool(fractures == 0 or maximum > initial),
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

    times = np.asarray(recorder.times)

    strain_path = output / "strain_over_time.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(times, recorder.max_strain, label="maximum active-edge strain")
    axes[0].plot(times, recorder.mean_strain, label="mean active-edge strain")
    axes[0].set(xlabel="time (s)", ylabel="engineering strain")
    axes[0].legend()
    keys = sorted({key for sample in recorder.edge_strains for key in sample})
    if keys and recorder.edge_strains:
        heat = np.full((len(keys), len(recorder.edge_strains)), np.nan)
        key_index = {key: i for i, key in enumerate(keys)}
        for column, sample in enumerate(recorder.edge_strains):
            for key, value in sample.items():
                heat[key_index[key], column] = value
        image = axes[1].imshow(
            heat,
            aspect="auto",
            origin="lower",
            extent=(times[0], times[-1], 0, len(keys)),
            interpolation="nearest",
        )
        fig.colorbar(image, ax=axes[1], label="engineering strain")
    axes[1].set(xlabel="time (s)", ylabel="active edge index")
    fig.savefig(strain_path, dpi=140)
    plt.close(fig)

    events_path = output / "fracture_events.png"
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(times, recorder.fracture_events, color="tab:red")
    axes[0].set(xlabel="time (s)", ylabel="fracture events")
    axes[1].plot(times, recorder.active_constraints, color="tab:purple")
    axes[1].set(xlabel="time (s)", ylabel="active constraints")
    fig.savefig(events_path, dpi=140)
    plt.close(fig)

    topology_path = output / "topology_check.png"
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(times, recorder.components, color="tab:green")
    ax.set(xlabel="time (s)", ylabel="triangle components", title="Topology connectivity")
    fig.savefig(topology_path, dpi=140)
    plt.close(fig)

    check_path = output / "topology_check.json"
    check_path.write_text(json.dumps(recorder.topology_check(), indent=2), encoding="utf-8")
    return {
        "strain": strain_path,
        "events": events_path,
        "topology": topology_path,
        "topology_check": check_path,
    }
