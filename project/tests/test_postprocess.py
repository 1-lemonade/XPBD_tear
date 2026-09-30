"""Checks for optional, depth-based pyrender post-processing effects."""

from __future__ import annotations

import numpy as np

from project.viz.postprocess import apply_postprocessing


def run() -> None:
    size = 9
    color = np.full((size, size, 4), 128, dtype=np.uint8)
    color[:, :, 3] = 255
    depth = np.ones((size, size), dtype=np.float32)
    depth[3:6, 3:6] = 1.5
    depth[:, 6:] = np.linspace(1.0, 2.2, size, dtype=np.float32)[:, None]
    depth[0, :] = 0.0

    unchanged = apply_postprocessing(color, depth)
    assert np.array_equal(unchanged, color), "disabled post-processing changed the image"

    fogged = apply_postprocessing(color, depth, fog=True)
    assert not np.array_equal(fogged[4, 7, :3], color[4, 7, :3]), "fog did not use depth"
    assert np.array_equal(fogged[0, 4, :3], color[0, 4, :3]), "fog changed background pixels"

    ao = apply_postprocessing(color, depth, ao=True)
    assert int(ao[3, 4, 0]) < int(color[3, 4, 0]), "ambient occlusion did not darken recessed depth"

    highlighted = apply_postprocessing(color, depth, edges=True)
    assert not np.array_equal(highlighted[:, :, :3], color[:, :, :3]), "depth edges were not highlighted"

    flat_depth = np.ones((size, size), dtype=np.float32)
    flat_depth[:, 0] = 0.0
    flat_edges = apply_postprocessing(color, flat_depth, edges=True)
    assert not np.array_equal(flat_edges[4, 1, :3], color[4, 1, :3]), "valid-depth boundaries were not highlighted"

    mapped = apply_postprocessing(color, depth, tonemap=True)
    assert not np.array_equal(mapped[:, :, :3], color[:, :, :3]), "tone mapping had no effect"

    combined = apply_postprocessing(color, depth, fog=True, ao=True, edges=True, tonemap=True)
    assert combined.shape == color.shape and combined.dtype == np.uint8
    assert np.array_equal(combined[:, :, 3], color[:, :, 3]), "post-processing changed alpha"

    for invalid_color, invalid_depth in ((color[:, :, :2], depth), (color, depth[:-1])):
        try:
            apply_postprocessing(invalid_color, invalid_depth, fog=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid buffer dimensions were accepted")
    print("all post-processing checks passed")


if __name__ == "__main__":
    run()
