"""State exchange interface and no-op implementation."""

from __future__ import annotations

from .contracts import ClothSnapshot


def validate_exchange(snapshot: ClothSnapshot, time: float, dt: float) -> None:
    """Use stored time/dt, never reconstruct accumulated time as step * dt."""
    if not isinstance(snapshot, ClothSnapshot):
        raise TypeError("exchange requires ClothSnapshot")
    if time != snapshot.time or dt != snapshot.dt:
        raise ValueError("exchange time/dt must equal snapshot time/dt")


class CouplingInterface:
    def exchange(self, snapshot: ClothSnapshot, time: float, dt: float) -> None:
        """Receive simulation state at a completed step."""
        raise NotImplementedError

    def reset(self) -> None:
        """Discard component results/caches before a simulation reset."""


class NullCoupling(CouplingInterface):
    def exchange(self, snapshot: ClothSnapshot, time: float, dt: float) -> None:
        validate_exchange(snapshot, time, dt)
