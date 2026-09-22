"""Future coupling seam; intentionally contains no MPM implementation."""

from __future__ import annotations

from typing import Any


class CouplingInterface:
    def exchange(self, cloth_state: Any, time: float, dt: float) -> None:
        """Exchange state with a future subsystem."""
        raise NotImplementedError


class NullCoupling(CouplingInterface):
    def exchange(self, cloth_state: Any, time: float, dt: float) -> None:
        del cloth_state, time, dt

