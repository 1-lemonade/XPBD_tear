"""Display-only NumPy to Taichi-field bridge for GGUI Canvas.triangles."""
from __future__ import annotations

import numpy as np
import taichi as ti


class CanvasMesh:
    """Reuse field buffers; unused indices draw only degenerate triangles."""

    def __init__(self):
        self.vertex_capacity = 0
        self.index_capacity = 0

    def draw(self, canvas, vertices: np.ndarray, indices: np.ndarray, colors: np.ndarray) -> None:
        vertex_count = len(vertices)
        flat_indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        if vertex_count > self.vertex_capacity:
            self.vertex_capacity = 1 << (vertex_count - 1).bit_length()
            self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.vertex_capacity)
            self.colors = ti.Vector.field(3, dtype=ti.f32, shape=self.vertex_capacity)
        if len(flat_indices) > self.index_capacity:
            # Round up in complete triangles, so padded zero indices are valid.
            self.index_capacity = 3 * (1 << (max(1, len(flat_indices) // 3) - 1).bit_length())
            self.indices = ti.field(dtype=ti.i32, shape=self.index_capacity)
        vertex_buffer = np.zeros((self.vertex_capacity, 3), dtype=np.float32)
        color_buffer = np.zeros_like(vertex_buffer)
        index_buffer = np.zeros(self.index_capacity, dtype=np.int32)
        vertex_buffer[:vertex_count] = vertices[:, :3]
        color_buffer[:vertex_count] = colors[:, :3]
        index_buffer[:len(flat_indices)] = flat_indices
        self.vertices.from_numpy(vertex_buffer)
        self.colors.from_numpy(color_buffer)
        self.indices.from_numpy(index_buffer)
        canvas.triangles(self.vertices, indices=self.indices, per_vertex_color=self.colors)
