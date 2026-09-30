"""Capture actual pyrender depth buffers and independently exercise each effect."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--view-tilt', type=float, default=0.0, help='Rigid view-only X rotation in degrees; never changes physics')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str((output / '.matplotlib').resolve())
    os.environ['XPBD_USE_PYRENDER'] = '1'
    for name in ('FOG', 'AO', 'EDGES', 'TONEMAP'):
        os.environ.pop('XPBD_POST_' + name, None)
    from project.main import ClothSimulation, DemoConfig
    from project.viz.scene_render import render_scene
    from project.viz.postprocess import apply_postprocessing
    from PIL import Image
    import pyrender
    import OpenGL.platform

    simulation = ClothSimulation(DemoConfig(resolution_x=8, resolution_y=11))
    for _ in range(args.frames):
        simulation.step()
    if args.view_tilt:
        from types import SimpleNamespace
        theta = np.deg2rad(args.view_tilt)
        rotation = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        view_positions = simulation.particles.position @ rotation.T
        simulation = SimpleNamespace(particles=SimpleNamespace(position=view_positions), mesh=simulation.mesh)
    captured = {}
    original = pyrender.OffscreenRenderer.render

    def capture(renderer, *positional, **keywords):
        color, depth = original(renderer, *positional, **keywords)
        captured['color'], captured['depth'] = color.copy(), depth.copy()
        return color, depth

    pyrender.OffscreenRenderer.render = capture
    started = time.perf_counter()
    try:
        backend = render_scene(simulation, output / 'pyrender_off.png')
    finally:
        pyrender.OffscreenRenderer.render = original
    result = {'backend': backend, 'opengl_platform': type(OpenGL.platform.PLATFORM).__name__,
              'render_seconds': time.perf_counter() - started, 'frames': args.frames,
              'real_depth_captured': bool(captured), 'view_only_x_rotation_degrees': args.view_tilt, 'effects': {}}
    if captured:
        color, depth = captured['color'], captured['depth']
        valid = np.isfinite(depth) & (depth > 0)
        result['positive_depth_pixels'] = int(valid.sum())
        result['depth_min'] = float(depth[valid].min()) if valid.any() else None
        result['depth_max'] = float(depth[valid].max()) if valid.any() else None
        np.savez_compressed(output / 'real_buffers.npz', color=color, depth=depth)
        assert np.array_equal(apply_postprocessing(color, depth), color)
        for effect in ('fog', 'ao', 'edges', 'tonemap', 'all'):
            options = {name: effect == 'all' or name == effect for name in ('fog', 'ao', 'edges', 'tonemap')}
            processed = apply_postprocessing(color, depth, **options)
            Image.fromarray(processed).save(output / f'pyrender_{effect}.png')
            result['effects'][effect] = {
                'changed_pixels': int(np.any(processed != color, axis=2).sum()),
                'finite': bool(np.isfinite(processed).all()), 'options': options}
        result['wired_backend'] = render_scene(simulation, output / 'pyrender_wired_all.png',
                                               postprocess_effects={name: True for name in ('fog', 'ao', 'edges', 'tonemap')})
    (output / 'renderer.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not captured or backend != 'pyrender-offscreen':
        raise SystemExit('Real pyrender rendering unavailable; inspect fallback and report limitation.')


if __name__ == '__main__':
    main()
