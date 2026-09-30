"""Owned, read-only world-space data for completed cloth states.

Versions are scoped to a simulation object. Compare the entire version tuple:
geometry moves without a topology rebuild, and reset starts another epoch.
Array immutability is an API guarantee, not a security boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import numbers

import numpy as np


def _array(value, name, dtype, shape, *, finite=True):
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    target = np.dtype(dtype)
    if target.kind == "i":
        if raw.dtype.kind not in "iu" and raw.size:
            raise ValueError(f"{name} must contain integers, not {raw.dtype}")
        limits = np.iinfo(target)
        if raw.size and (np.any(raw < limits.min) or np.any(raw > limits.max)):
            raise ValueError(f"{name} exceeds {target} range")
    elif target.kind == "b":
        if raw.dtype.kind != "b" and raw.size:
            raise ValueError(f"{name} must contain booleans")
    elif raw.dtype.kind not in "fiu" and raw.size:
        raise ValueError(f"{name} must contain real numbers")
    try:
        array = np.array(raw, dtype=target, copy=True)
    except (TypeError, OverflowError) as error:
        raise ValueError(f"invalid {name}") from error
    if target.kind == "f" and finite and not np.isfinite(array).all():
        index = tuple(np.argwhere(~np.isfinite(array))[0])
        raise ValueError(f"{name}{index} must be finite")
    array.setflags(write=False)
    return array


def _count(value, name, columns=None):
    raw = np.asarray(value)
    if raw.ndim != (1 if columns is None else 2) or (
        columns is not None and raw.shape[1] != columns
    ):
        raise ValueError(f"{name} must have shape (Q,)" if columns is None
                         else f"{name} must have shape (N,{columns})")
    return raw.shape[0]


def _ids(value, name, count, *, nonnegative=False):
    array = _array(value, name, np.int64, (count,))
    if len(np.unique(array)) != count:
        raise ValueError(f"{name} must be unique")
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


class _Versioned:
    @property
    def version(self):
        return (self.epoch, self.step_index, self.topology_version)

    def _validate_version(self):
        for name in ("epoch", "step_index", "topology_version"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True, eq=False)
class ClothSnapshot(_Versioned):
    """Triangles index position rows; identity arrays need not be contiguous.

    Length/time units are those of the model, not necessarily SI. No capacity
    tails, rendering normalization or mutable live-state aliases are exposed.
    """

    positions: np.ndarray
    triangles: np.ndarray
    vertex_ids: np.ndarray
    triangle_ids: np.ndarray
    velocity: np.ndarray
    inverse_mass: np.ndarray
    pinned: np.ndarray
    epoch: int
    step_index: int
    topology_version: int
    time: float
    dt: float

    def __post_init__(self):
        self._validate_version()
        n, t = _count(self.positions, "positions", 3), _count(self.triangles, "triangles", 3)
        arrays = {
            "positions": _array(self.positions, "positions", np.float64, (n, 3)),
            "triangles": _array(self.triangles, "triangles", np.int64, (t, 3)),
            "vertex_ids": _ids(self.vertex_ids, "vertex_ids", n, nonnegative=True),
            "triangle_ids": _ids(self.triangle_ids, "triangle_ids", t, nonnegative=True),
            "velocity": _array(self.velocity, "velocity", np.float64, (n, 3)),
            "inverse_mass": _array(self.inverse_mass, "inverse_mass", np.float64, (n,)),
            "pinned": _array(self.pinned, "pinned", bool, (n,)),
        }
        if np.any(arrays["triangles"] < 0) or np.any(arrays["triangles"] >= n):
            raise ValueError("triangles contain an out-of-range position row index")
        if np.any(arrays["inverse_mass"] < 0):
            raise ValueError("inverse_mass must be nonnegative")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "time", _real(self.time, "time"))
        object.__setattr__(self, "dt", _real(self.dt, "dt", positive=True))


@dataclass(frozen=True, eq=False)
class QueryPoints:
    query_ids: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        q = _count(self.positions, "positions", 3)
        object.__setattr__(self, "query_ids", _ids(self.query_ids, "query_ids", q))
        object.__setattr__(self, "positions", _array(self.positions, "positions", np.float64, (q, 3)))


class MappingStatus(IntEnum):
    HIT = 0
    OUTSIDE_THRESHOLD = 1
    EMPTY_SURFACE = 2
    DEGENERATE_ONLY = 3


class MappingFeature(IntEnum):
    NONE = -1
    FACE = 0
    EDGE = 1
    VERTEX = 2


@dataclass(frozen=True)
class MappingConfig:
    area_rtol: float = 1e-12
    tie_atol: float = 1e-10
    tie_rtol: float = 1e-12
    max_distance: float | None = None

    def __post_init__(self):
        for name in ("area_rtol", "tie_atol", "tie_rtol", "max_distance"):
            value = getattr(self, name)
            if name == "max_distance" and value is None:
                continue
            object.__setattr__(self, name, _real(value, name, positive=name == "area_rtol"))


@dataclass(frozen=True, eq=False)
class MappingResult(_Versioned):
    """Per-query results; Q=0 intentionally carries no surface diagnosis.

    Keep the producing snapshot to resolve IDs and recover time/dt. Only HIT
    is a binding. Outside-threshold rows retain the selected candidate; its
    actual distance, not the pre-tie global minimum, determines acceptance.
    """

    query_ids: np.ndarray
    epoch: int
    step_index: int
    topology_version: int
    triangle_ids: np.ndarray
    vertex_ids: np.ndarray
    barycentric: np.ndarray
    closest_point: np.ndarray
    distance: np.ndarray
    normal: np.ndarray
    normal_offset: np.ndarray
    status: np.ndarray
    feature: np.ndarray

    def __post_init__(self):
        self._validate_version()
        q = _count(self.query_ids, "query_ids")
        object.__setattr__(self, "query_ids", _ids(self.query_ids, "query_ids", q))
        for name, shape, dtype in (
            ("triangle_ids", (q,), np.int64), ("vertex_ids", (q, 3), np.int64),
            ("barycentric", (q, 3), np.float64), ("closest_point", (q, 3), np.float64),
            ("distance", (q,), np.float64), ("normal", (q, 3), np.float64),
            ("normal_offset", (q,), np.float64), ("status", (q,), np.int8),
            ("feature", (q,), np.int8),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), name, dtype, shape, finite=False))
        if not np.isin(self.status, list(MappingStatus)).all():
            raise ValueError("unknown mapping status")
        candidate = self.status <= MappingStatus.OUTSIDE_THRESHOLD
        missing = ~candidate
        if (np.any(self.triangle_ids[candidate] < 0) or np.any(self.vertex_ids[candidate] < 0)
                or not np.isin(self.feature[candidate], [0, 1, 2]).all()):
            raise ValueError("candidate rows require valid IDs and a geometric feature")
        if (np.any(self.triangle_ids[missing] != -1) or np.any(self.vertex_ids[missing] != -1)
                or np.any(self.feature[missing] != MappingFeature.NONE)):
            raise ValueError("missing rows require -1 IDs and NONE feature")
        for name in ("barycentric", "closest_point", "distance", "normal", "normal_offset"):
            value = getattr(self, name)
            if not np.isfinite(value[candidate]).all() or not np.isnan(value[missing]).all():
                raise ValueError(f"{name}: candidates must be finite; missing rows must be NaN")
        weights = self.barycentric[candidate]
        if (np.any(self.distance[candidate] < 0) or np.any(weights < -1e-9)
                or np.any(weights > 1 + 1e-9)
                or np.any(np.abs(weights.sum(axis=1) - 1) > 1e-9)
                or np.any(np.abs(np.linalg.norm(self.normal[candidate], axis=1) - 1) > 1e-9)):
            raise ValueError("invalid candidate distance, barycentric weights or unit normal")
