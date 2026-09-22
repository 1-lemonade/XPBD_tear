"""GIF export for the Phase 1 cloth demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


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
    from matplotlib.collections import LineCollection, PolyCollection
    from PIL import Image

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
        broken_segments = [
            positions[[edge.a, edge.b], :2]
            for edge in simulation.mesh.edges
            if edge.broken
        ]

        fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=100)
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=(0.22, 0.52, 0.92, 0.72),
                edgecolors=(0.05, 0.12, 0.35, 0.75),
                linewidths=0.45,
            )
        )
        if broken_segments:
            ax.add_collection(LineCollection(broken_segments, colors="crimson", linewidths=2.0))

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
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))
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
