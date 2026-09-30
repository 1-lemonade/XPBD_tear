"""State exchange interface and no-op implementation."""

from __future__ import annotations

from typing import Any


class CouplingInterface:
    def exchange(self, cloth_state: Any, time: float, dt: float) -> None:
        """Receive simulation state at a completed step."""
        raise NotImplementedError


class NullCoupling(CouplingInterface):
    def exchange(self, cloth_state: Any, time: float, dt: float) -> None:
        del cloth_state, time, dt
