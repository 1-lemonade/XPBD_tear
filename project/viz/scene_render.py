"""Minimal pyrender scene export with a matplotlib 3D fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np

from .style import FIGURE_DPI, FIGURE_SIZE, PALETTE, STRAIN_CMAP, apply_plot_style, style_axes


def render_scene(
    simulation: Any,
    output_path: str | Path,
    width: int = 900,
    height: int = 700,
    postprocess_effects: dict[str, bool] | None = None,
) -> str:
    """Render the current cloth to PNG and return the backend used.

    The deterministic matplotlib export is the default. ``pyrender`` can be
    enabled with ``XPBD_USE_PYRENDER=1`` when a known-good offscreen OpenGL
    context is available. This avoids hanging headless Windows runs during
    driver/context initialization while preserving an inspectable artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config = path.parent / ".matplotlib"
    mpl_config.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config.resolve()))
    vertices = np.asarray(simulation.particles.position, dtype=np.float64)
    faces = np.asarray(simulation.mesh.triangles, dtype=np.int32)
    use_pyrender = os.environ.get("XPBD_USE_PYRENDER", "0") == "1"
    effect_names = ("fog", "ao", "edges", "tonemap")
    effects = {
        name: bool((postprocess_effects or {}).get(name, False))
        or os.environ.get(f"XPBD_POST_{name.upper()}", "0") == "1"
        for name in effect_names
    }
    if any(effects.values()) and not use_pyrender:
        print("pyrender post-processing requested; set XPBD_USE_PYRENDER=1 to render a depth buffer")
    if not use_pyrender:
        _render_matplotlib_fallback(simulation, path)
        return "matplotlib-3d-fallback"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        apply_plot_style(plt)
        import pyrender
        import trimesh

        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.15, 0.5, 0.9, 1.0), metallicFactor=0.0, roughnessFactor=0.8
        )
        scene = pyrender.Scene(bg_color=(0.03, 0.04, 0.07, 1.0), ambient_light=(0.25, 0.25, 0.25))
        scene.add(pyrender.Mesh.from_trimesh(surface, material=material, smooth=False))
        center = vertices.mean(axis=0)
        x_span = max(float(np.ptp(vertices[:, 0])), 1.0e-3)
        y_span = max(float(np.ptp(vertices[:, 1])), 1.0e-3)
        span = max(x_span, y_span, 1.0)
        camera = pyrender.OrthographicCamera(xmag=x_span * 0.62, ymag=y_span * 0.62)
        pose = np.eye(4)
        pose[:3, 3] = center + np.array((0.0, 0.0, span * 2.5))
        scene.add(camera, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            color, depth = renderer.render(scene)
        finally:
            renderer.delete()
        active_effects = {name: value for name, value in effects.items() if value}
        if active_effects:
            from .postprocess import apply_postprocessing

            color = apply_postprocessing(color, depth, **active_effects)
        plt.imsave(path, color)
        suffix = f" + {', '.join(active_effects)}" if active_effects else ""
        return "pyrender-offscreen" + suffix
    except Exception as error:
        _render_matplotlib_fallback(simulation, path, error)
        return "matplotlib-3d-fallback"


def _render_matplotlib_fallback(simulation: Any, path: Path, error: Exception | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    apply_plot_style(plt)

    vertices = np.asarray(simulation.particles.position)
    faces = np.asarray(simulation.mesh.triangles)
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PALETTE["panel"])
    polygons = [vertices[face] for face in faces]
    ax.add_collection3d(
        Poly3DCollection(
            polygons,
            alpha=0.78,
            facecolor=PALETTE["blue"],
            edgecolor=PALETTE["navy"],
            linewidth=0.35,
        )
    )
    broken_edges = [edge for edge in simulation.mesh.edges if edge.broken]
    if broken_edges:
        strains = np.asarray([edge.fracture_strain for edge in broken_edges], dtype=np.float64)
        upper = max(float(strains.max()), 1.0e-6)
        lines = Line3DCollection(
            [vertices[[edge.a, edge.b]] for edge in broken_edges],
            cmap=STRAIN_CMAP,
            norm=Normalize(vmin=0.0, vmax=upper),
            linewidth=2.7,
        )
        lines.set_array(strains)
        ax.add_collection3d(lines)
        fig.colorbar(lines, ax=ax, shrink=0.62, pad=0.08, label="strain at tear")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("XPBD cloth tearing · Matplotlib fallback")
    if error is not None:
        ax.text2D(0.02, 0.02, str(error).splitlines()[0][:100], transform=ax.transAxes, color=PALETTE["muted"], fontsize=8)
    if len(vertices):
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        for setter, low, high in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]), (ax.set_zlim, lo[2] - 0.1, hi[2] + 0.1)):
            if high - low < 1e-6:
                low, high = low - 0.5, high + 0.5
            setter(low, high)
    style_axes(ax)
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
