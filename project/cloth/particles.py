"""Struct-of-arrays particle storage used by the cloth solver."""

from __future__ import annotations

import numpy as np


class ParticleSoA:
    """Contiguous particle buffers with stable integer particle identifiers."""

    def __init__(self, positions: np.ndarray, masses: np.ndarray | None = None):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (particle_count, 3)")
        count = positions.shape[0]
        if masses is None:
            masses = np.ones(count, dtype=np.float64)
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != (count,):
            raise ValueError("masses must have shape (particle_count,)")

        self.position = positions.copy()
        self.previous_position = positions.copy()
        self.predicted_position = positions.copy()
        self.velocity = np.zeros_like(self.position)
        self.mass = masses.copy()
        self.inverse_mass = np.where(masses > 0.0, 1.0 / masses, 0.0)
        self.pinned = np.zeros(count, dtype=bool)

    @property
    def count(self) -> int:
        return int(self.position.shape[0])

    def append(self, position: np.ndarray, mass: float | None = None, source: int | None = None) -> int:
        """Append a particle, copying state from ``source`` when supplied."""
        if source is not None:
            if not 0 <= source < self.count:
                raise IndexError(source)
            position = self.position[source].copy()
            if mass is None:
                mass = float(self.mass[source])
        if mass is None:
            mass = 1.0
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError("position must have shape (3,)")
        self.position = np.vstack((self.position, position))
        self.previous_position = np.vstack((self.previous_position, position))
        self.predicted_position = np.vstack((self.predicted_position, position))
        self.velocity = np.vstack((self.velocity, np.zeros((1, 3), dtype=np.float64)))
        self.mass = np.append(self.mass, mass)
        self.inverse_mass = np.append(self.inverse_mass, 1.0 / mass if mass > 0 else 0.0)
        self.pinned = np.append(self.pinned, False)
        return self.count - 1

    def pin(self, particle_ids: list[int] | np.ndarray) -> None:
        ids = np.asarray(particle_ids, dtype=np.int64)
        self.pinned[ids] = True
        self.inverse_mass[ids] = 0.0

    def set_pin_state(self, particle_id: int, is_pinned: bool) -> None:
        self.pinned[particle_id] = is_pinned
        self.inverse_mass[particle_id] = 0.0 if is_pinned else 1.0 / self.mass[particle_id]

    def clone_state(self, particle_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.position[particle_id].copy(), self.velocity[particle_id].copy()

