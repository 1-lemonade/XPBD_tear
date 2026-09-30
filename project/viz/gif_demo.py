"""GIF export for the cloth demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .style import FIGURE_DPI, FIGURE_SIZE, PALETTE, STRAIN_CMAP, apply_plot_style, style_axes


def write_gif(simulation: Any, output_path: str | Path, frames: int = 60, fps: int = 12) -> Path:
    """Advance a simulation and write a short 2D cloth-tearing GIF.

    This visualization-only path uses the existing ``ClothSimulation.step``
    method and does not alter solver or topology behavior.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = output.parent / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import Normalize, to_rgb
    from PIL import Image

    apply_plot_style(plt)

    image_frames: list[Image.Image] = []
    total_frames = max(1, int(frames))
    frame_rate = max(1, int(fps))
    width = float(simulation.config.width)
    height = float(simulation.config.height)
    for frame_index in range(total_frames):
        if frame_index:
            simulation.step()

        positions = np.asarray(simulation.particles.position, dtype=np.float64)
        triangles = np.asarray(simulation.mesh.triangles, dtype=np.int64)
        polygons = [positions[face, :2] for face in triangles]
        broken_edges = [edge for edge in simulation.mesh.edges if edge.broken]
        broken_segments = [
            positions[[edge.a, edge.b], :2]
            for edge in broken_edges
        ]
        fracture_strains = np.asarray(
            [edge.fracture_strain for edge in broken_edges],
            dtype=np.float64,
        )

        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True)
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=(*to_rgb(PALETTE["blue"]), 0.76),
                edgecolors=(*to_rgb(PALETTE["navy"]), 0.72),
                linewidths=0.5,
            )
        )
        strain_scale = max(float(simulation.config.critical_strain) * 2.0, 0.1)
        norm = Normalize(vmin=0.0, vmax=strain_scale)
        if broken_segments:
            fracture_lines = LineCollection(
                broken_segments,
                cmap=STRAIN_CMAP,
                norm=norm,
                linewidths=2.5,
            )
            fracture_lines.set_array(fracture_strains)
            ax.add_collection(fracture_lines)
        else:
            fracture_lines = ScalarMappable(norm=norm, cmap=STRAIN_CMAP)
        fig.colorbar(fracture_lines, ax=ax, label="strain at tear", shrink=0.9, pad=0.025)

        margin = 0.08
        lower_y = min(-simulation.pull_offset - margin, -margin)
        ax.set_xlim(-margin, width + margin)
        ax.set_ylim(lower_y, height + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"XPBD cloth tearing  |  t={simulation.time:.2f}s  |  fractures={len(simulation.fracture_log)}"
        )
        style_axes(ax)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_frames.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(fig)

    image_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=image_frames[1:],
        duration=max(1, int(round(1000 / frame_rate))),
        loop=0,
        optimize=False,
    )
    return output
