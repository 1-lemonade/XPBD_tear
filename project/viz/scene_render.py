"""Minimal pyrender scene export with a matplotlib 3D fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import numpy as np


def render_scene(simulation: Any, output_path: str | Path, width: int = 900, height: int = 700) -> str:
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
    try:
        if not use_pyrender:
            raise RuntimeError("pyrender offscreen disabled; set XPBD_USE_PYRENDER=1 to enable")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyrender
        import trimesh

        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.15, 0.5, 0.9, 1.0), metallicFactor=0.0, roughnessFactor=0.8
        )
        scene = pyrender.Scene(bg_color=(0.03, 0.04, 0.07, 1.0), ambient_light=(0.25, 0.25, 0.25))
        scene.add(pyrender.Mesh.from_trimesh(surface, material=material, smooth=False))
        center = vertices.mean(axis=0)
        span = max(float(np.ptp(vertices[:, 0])), float(np.ptp(vertices[:, 1])), 1.0)
        camera = pyrender.OrthographicCamera(xmag=span * 1.15, ymag=span * 1.15)
        pose = np.eye(4)
        pose[:3, 3] = center + np.array((0.0, 0.0, span * 2.5))
        scene.add(camera, pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        color, _ = renderer.render(scene)
        renderer.delete()
        plt.imsave(path, color)
        return "pyrender-offscreen"
    except Exception as error:
        _render_matplotlib_fallback(simulation, path, error)
        return "matplotlib-3d-fallback"


def _render_matplotlib_fallback(simulation: Any, path: Path, error: Exception) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = np.asarray(simulation.particles.position)
    faces = np.asarray(simulation.mesh.triangles)
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    polygons = [vertices[face] for face in faces]
    ax.add_collection3d(Poly3DCollection(polygons, alpha=0.72, facecolor="cornflowerblue", edgecolor="navy"))
    for edge in simulation.mesh.edges:
        if edge.broken:
            segment = vertices[[edge.a, edge.b]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="crimson", linewidth=2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("XPBD cloth tearing (matplotlib fallback)\n" + str(error).splitlines()[0][:100])
    if len(vertices):
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        for setter, low, high in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]), (ax.set_zlim, lo[2] - 0.1, hi[2] + 0.1)):
            if high - low < 1e-6:
                low, high = low - 0.5, high + 0.5
            setter(low, high)
    fig.savefig(path, dpi=140)
    plt.close(fig)
