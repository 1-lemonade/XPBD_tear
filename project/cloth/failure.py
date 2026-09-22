"""Swappable, solver-independent fracture criteria."""

from __future__ import annotations

from dataclasses import dataclass

from .mesh import Edge


class FailureModel:
    """Interface for deciding whether an edge should fracture."""

    def should_break(self, edge: Edge) -> bool:
        raise NotImplementedError


@dataclass
class StrainThresholdFailure(FailureModel):
    critical_strain: float = 0.35

    def should_break(self, edge: Edge) -> bool:
        return edge.current_strain > self.critical_strain


class AlwaysFalseFailure(FailureModel):
    def should_break(self, edge: Edge) -> bool:
        del edge
        return False
