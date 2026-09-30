"""Optional depth-buffer effects for pyrender scene output."""

from __future__ import annotations

import numpy as np


def apply_postprocessing(
    color: np.ndarray,
    depth: np.ndarray,
    *,
    fog: bool = False,
    ao: bool = False,
    edges: bool = False,
    tonemap: bool = False,
) -> np.ndarray:
    """Apply independent, opt-in depth cues and return an RGB/RGBA uint8 image.

    Depth effects operate only on finite, positive depth samples. The order is
    ambient occlusion, fog, depth-edge highlighting, then tone mapping.
    """
    source = np.asarray(color)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("color buffer must have shape (height, width, 3 or 4)")
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.shape != source.shape[:2]:
        raise ValueError("depth buffer dimensions must match the color buffer")
    if not (fog or ao or edges or tonemap):
        return source

    color_scale = 255.0 if source.dtype == np.uint8 or float(np.nanmax(source)) > 1.0 else 1.0
    rgb = np.clip(source[:, :, :3].astype(np.float32) / color_scale, 0.0, 1.0)
    alpha = source[:, :, 3:4].copy() if source.shape[2] == 4 else None
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    depth_range = _depth_range(depth_array, valid)

    if depth_range is not None and ao:
        rgb *= _ambient_occlusion(depth_array, valid, depth_range)[:, :, None]
    if depth_range is not None and fog:
        near, far = depth_range
        normalized = np.clip((depth_array - near) / max(far - near, 1.0e-6), 0.0, 1.0)
        amount = 0.32 * np.clip((normalized - 0.08) / 0.92, 0.0, 1.0)
        fog_color = np.asarray((0.78, 0.84, 0.91), dtype=np.float32)
        rgb[valid] = rgb[valid] * (1.0 - amount[valid, None]) + fog_color * amount[valid, None]
    if edges:
        edge_mask = _depth_edges(depth_array, valid, depth_range)
        edge_color = np.asarray((1.0, 0.39, 0.11), dtype=np.float32)
        rgb[edge_mask] = rgb[edge_mask] * 0.24 + edge_color * 0.76
    if tonemap:
        rgb = np.clip((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), 0.0, 1.0)

    output = np.round(rgb * 255.0).astype(np.uint8)
    if alpha is not None:
        if alpha.dtype != np.uint8:
            alpha_scale = 255.0 if float(np.nanmax(alpha)) > 1.0 else 1.0
            alpha = np.round(np.clip(alpha.astype(np.float32) / alpha_scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        output = np.concatenate((output, alpha), axis=2)
    return output


def _depth_range(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float] | None:
    values = depth[valid]
    if not values.size:
        return None
    low, high = np.percentile(values, (2.0, 98.0))
    if high - low < 1.0e-6:
        low, high = float(values.min()), float(values.max())
    if high - low < 1.0e-6:
        return None
    return float(low), float(high)


def _ambient_occlusion(depth: np.ndarray, valid: np.ndarray, depth_range: tuple[float, float]) -> np.ndarray:
    """Darken locally recessed pixels using closer valid neighbors."""
    near, far = depth_range
    scale = max((far - near) * 0.08, 1.0e-5)
    occlusion = np.zeros(depth.shape, dtype=np.float32)
    sample_count = np.zeros(depth.shape, dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        neighbor_depth = np.roll(depth, (dy, dx), axis=(0, 1))
        neighbor_valid = np.roll(valid, (dy, dx), axis=(0, 1))
        sample_valid = valid & neighbor_valid
        if dy > 0:
            sample_valid[:dy, :] = False
        elif dy < 0:
            sample_valid[dy:, :] = False
        if dx > 0:
            sample_valid[:, :dx] = False
        elif dx < 0:
            sample_valid[:, dx:] = False
        occlusion += np.maximum(depth - neighbor_depth - scale * 0.02, 0.0) * sample_valid
        sample_count += sample_valid
    factor = np.ones(depth.shape, dtype=np.float32)
    observed = sample_count > 0
    strength = np.clip(occlusion / np.maximum(sample_count * scale, 1.0e-6), 0.0, 1.0)
    factor[observed] = 1.0 - 0.42 * strength[observed]
    factor[~valid] = 1.0
    return factor


def _depth_edges(
    depth: np.ndarray,
    valid: np.ndarray,
    depth_range: tuple[float, float] | None,
) -> np.ndarray:
    """Find strong depth jumps and foreground/background boundaries."""
    threshold = max((depth_range[1] - depth_range[0]) * 0.018, 1.0e-5) if depth_range else float("inf")
    edge = np.zeros(depth.shape, dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        neighbor_depth = np.roll(depth, (dy, dx), axis=(0, 1))
        neighbor_valid = np.roll(valid, (dy, dx), axis=(0, 1))
        edge |= valid & neighbor_valid & (np.abs(depth - neighbor_depth) > threshold)
        edge |= valid ^ neighbor_valid
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return edge
