"""MPM-side mirror of XPBD's window exchange `xpbd-exchange-v1` (P2-U2; host data, no kernels).

Field names and checks follow XPBD's project/coupling/loads.py (SourceIdentity, LoadBatch) and
exchange.py (WindowOffer, ImpulseWindow), read-only reference (SHA-256 pinned in the tests);
the MPM process never imports XPBD code. In P2-U3 the transport converts these objects to and
from XPBD's at the process boundary, field by field.

Conventions kept from XPBD:
  SourceIdentity(source_id, kind in {prescribed, real_backend}, version >= 0)
  LoadBatch: material -> cloth impulse per stable vertex ID for one window, (N, 3) f64 N s,
    coordinate_frame "world" = XPBD's native world frame (G_native = R^T G_mpm_world), units SI,
    t_start = S_k.time, t_end = S_k.time + h (XPBD's clock, never the MPM substep clock)
  WindowOffer: h / particle_dt must be an integer ratio of the shortest decimal forms
    (Fraction(str(h)) / Fraction(str(particle_dt))), substeps = that ratio
The surface checks (finite, IDs, manifold, winding) happen when the MPM endpoint reads the
snapshot (LiveSurface.ingest); XPBD's validate_surface runs on its side.
"""
import math
import numbers
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

SCHEMA = "xpbd-exchange-v1"
KINDS = ("prescribed", "real_backend")


def _text(v, name):
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return v


def _int(v, name):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Integral) or v < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {v!r}")
    return int(v)


def _real(v, name, positive=False):
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise ValueError(f"{name} must be a real scalar, got {v!r}")
    v = float(v)
    if not math.isfinite(v) or v < 0 or (positive and v == 0):
        raise ValueError(f"{name} must be finite and {'> 0' if positive else '>= 0'}, got {v!r}")
    return v


def _frozen(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    kind: str
    version: int

    def __post_init__(self):
        _text(self.source_id, "source_id")
        if self.kind not in KINDS:
            raise ValueError(f"source kind must be one of {KINDS}, got {self.kind!r}")
        object.__setattr__(self, "version", _int(self.version, "source version"))

    def as_dict(self):
        return dict(source_id=self.source_id, kind=self.kind, version=self.version)


@dataclass(frozen=True, eq=False)
class LoadBatch:
    stream_id: str
    source: SourceIdentity
    cloth_version: tuple
    batch_id: str
    t_start: float
    t_end: float
    vertex_ids: np.ndarray
    impulse: np.ndarray
    coordinate_frame: str = "world"
    units: str = "SI"

    def __post_init__(self):
        _text(self.stream_id, "stream_id")
        _text(self.batch_id, "batch_id")
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        cv = tuple(self.cloth_version) if isinstance(self.cloth_version, (tuple, list)) else None
        if cv is None or len(cv) != 3:
            raise ValueError("cloth_version must be (epoch, step_index, topology_version)")
        object.__setattr__(self, "cloth_version", tuple(_int(v, "cloth_version") for v in cv))
        for name in ("t_start", "t_end"):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if not self.t_end > self.t_start:
            raise ValueError("load interval must advance representable time")
        if self.coordinate_frame != "world" or self.units != "SI":
            raise ValueError("loads require the (XPBD native) world frame and SI (impulse N s)")
        ids = np.asarray(self.vertex_ids)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError("vertex_ids must be a 1-D integer array")
        ids = ids.astype(np.int64)
        if (ids < 0).any() or len(np.unique(ids)) != len(ids):
            raise ValueError("vertex_ids must be unique and nonnegative")
        imp = np.asarray(self.impulse, np.float64)
        if imp.shape != (len(ids), 3) or not np.isfinite(imp).all():
            raise ValueError(f"impulse must be a finite ({len(ids)}, 3) array")
        object.__setattr__(self, "vertex_ids", _frozen(ids))
        object.__setattr__(self, "impulse", _frozen(imp))


def substeps_of(h, particle_dt) -> int:
    """XPBD's rule: the shortest decimal forms of h and particle_dt have an integer ratio."""
    h = _real(h, "h", positive=True)
    dt = _real(particle_dt, "particle_dt", positive=True)
    ratio = Fraction(str(h)) / Fraction(str(dt))
    if ratio.denominator != 1 or ratio.numerator > 2 ** 53:
        raise ValueError(f"h = {h!r} is not an integer number of particle ticks {dt!r}")
    return int(ratio.numerator)


@dataclass(frozen=True, eq=False)
class WindowOffer:
    schema: str
    stream_id: str
    source: SourceIdentity
    window_index: int
    snapshot: object
    particle_dt: float
    h: float
    cut_edges: frozenset = frozenset()
    substeps: int = field(init=False)

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.source, SourceIdentity):
            raise ValueError("unsupported window schema or source")
        _text(self.stream_id, "stream_id")
        _int(self.window_index, "window_index")
        object.__setattr__(self, "substeps", substeps_of(self.h, self.particle_dt))
        object.__setattr__(self, "cut_edges", frozenset(self.cut_edges))

    def geometry_at_tick(self, i):
        """Frozen-velocity predictor positions (XPBD semantics), for checks."""
        i = _int(i, "tick")
        if i >= self.substeps:
            raise ValueError("tick must be inside the half-open exchange window")
        return np.asarray(self.snapshot.positions) + (i * self.particle_dt) * np.asarray(self.snapshot.velocity)


@dataclass(frozen=True)
class ImpulseWindow:
    schema: str
    window_index: int
    batch: LoadBatch

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.batch, LoadBatch):
            raise ValueError("unsupported impulse window schema or payload")
        _int(self.window_index, "window_index")
