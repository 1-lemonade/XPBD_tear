"""Explicitly attached, read-only surface queries after a complete step."""

from .contracts import ClothSnapshot, QueryPoints, MappingConfig, MappingResult
from .interface import CouplingInterface, validate_exchange


class QueryOnlyCoupling(CouplingInterface):
    def __init__(self, queries: QueryPoints, config: MappingConfig | None = None):
        if not isinstance(queries, QueryPoints):
            raise TypeError("queries must be QueryPoints")
        if config is not None and not isinstance(config, MappingConfig):
            raise TypeError("config must be MappingConfig")
        self.queries = QueryPoints(queries.query_ids, queries.positions)
        self.config = config if config is not None else MappingConfig()
        self.last_result: MappingResult | None = None

    def exchange(self, snapshot: ClothSnapshot, time: float, dt: float) -> None:
        from .mapping import map_points

        # Failed exchanges must not leave a previous result looking current.
        self.last_result = None
        validate_exchange(snapshot, time, dt)
        self.last_result = map_points(snapshot, self.queries, self.config)

    def reset(self) -> None:
        self.last_result = None
