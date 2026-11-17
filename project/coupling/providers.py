"""Prescribed soft-body boundary providers for the compliant-punch driver.

Everything here is external input, not a solved soft body: motion is an
analytic function of time, and the empty MPM slot raises for every physical
method. Geometry uses world coordinates and SI units. The 32768-point volume
representation is display/interface sampling only and is never advanced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from .contracts import QueryPoints, _count, _integer, _real
from .loads import SourceIdentity, _text
from .provider_contracts import BoundaryProvider, ProviderBoundarySamples


@dataclass(frozen=True)
class MotionConfig:
    """Analytic punch motion: smooth load, hold, smooth retract, then rest.

    The displacement uses a quintic smoothstep, so centre velocity and
    acceleration are continuous at every phase boundary and exactly zero
    during the final rest. Durations are seconds and the stroke is metres.
    """

    radius: float = 0.12
    center_start: tuple[float, float, float] = (0.0, 0.0, 0.22)
    stroke: float = 0.16
    load_duration: float = 8.0 / 3.0
    hold_duration: float = 1.0 / 3.0
    retract_duration: float = 8.0 / 3.0

    def __post_init__(self):
        for name in ("radius", "stroke", "load_duration", "hold_duration", "retract_duration"):
            object.__setattr__(self, name, _real(getattr(self, name), name, positive=True))
        center = np.asarray(self.center_start, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center_start must be three finite world coordinates")
        object.__setattr__(self, "center_start", tuple(float(v) for v in center))

    @property
    def sim_duration(self) -> float:
        return self.load_duration + self.hold_duration + self.retract_duration


def _smoothstep(x):
    """Quintic smoothstep ``x**3 * (10 - 15x + 6x**2)`` on [0, 1].

    S(0)=0, S(1)=1 and the first two derivatives vanish at both ends, so the
    prescribed motion is C2 across every phase boundary.
    """
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def _smoothstep_d(x):
    """d/dx of :func:`_smoothstep`; zero at x=0 and x=1."""
    if x >= 1.0:
        return 0.0
    return 30.0 * x * x * (1.0 - x) ** 2


def _smoothstep_dd(x):
    """Second derivative of :func:`_smoothstep`."""
    if x >= 1.0:
        return 0.0
    return 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x)


def motion_displacement(config: MotionConfig, time: float):
    """Return ``(z_offset, zdot, zddot)`` of the punch centre along -z."""
    time = _real(time, "time")
    load_end = config.load_duration
    hold_end = load_end + config.hold_duration
    retract_end = hold_end + config.retract_duration
    if time < load_end:
        duration = config.load_duration
        x = time / duration
        return (config.stroke * _smoothstep(x),
                config.stroke * _smoothstep_d(x) / duration,
                config.stroke * _smoothstep_dd(x) / (duration * duration))
    if time < hold_end:
        return config.stroke, 0.0, 0.0
    if time < retract_end:
        duration = config.retract_duration
        x = (time - hold_end) / duration
        return (config.stroke * (1.0 - _smoothstep(x)),
                -config.stroke * _smoothstep_d(x) / duration,
                -config.stroke * _smoothstep_dd(x) / (duration * duration))
    return 0.0, 0.0, 0.0


class PrescribedSphere:
    """Closed analytic sphere with externally prescribed motion.

    This is a *boundary provider*, not a material model: it reports the signed
    distance function ``phi = |p - c(t)| - R`` (negative inside), the outward
    unit normal and the prescribed surface velocity. ``source.version`` is a
    provider revision counter advanced only by :meth:`advance`.
    """

    def __init__(self, config: MotionConfig | None = None, source_id: str = "punch", seed: int = 2026):
        self.config = config if config is not None else MotionConfig()
        if not isinstance(self.config, MotionConfig):
            raise TypeError("config must be MotionConfig")
        self.seed = _integer(seed, "seed")
        self._source_id = _text(source_id, "source_id")
        self.time = 0.0
        self.version = 0
        self.last_advance: tuple[float, float] | None = None
        self.last_samples: ProviderBoundarySamples | None = None
        self._stream_id: str | None = None
        self._last_cloth_version: tuple[int, int, int] | None = None
        self._unit_cache: dict[int, np.ndarray] = {}

    @property
    def source(self) -> SourceIdentity:
        return SourceIdentity(self._source_id, "prescribed", self.version)

    # -- geometry ---------------------------------------------------------
    def kinematics(self, time: float):
        """Pure function: world centre and centre velocity at ``time``."""
        offset, zdot, _ = motion_displacement(self.config, time)
        start = np.asarray(self.config.center_start, dtype=np.float64)
        center = start + np.array((0.0, 0.0, -offset))
        velocity = np.array((0.0, 0.0, -zdot))
        return center, velocity

    def _unit_sphere_points(self, count: int) -> np.ndarray:
        count = _integer(count, "count")
        cached = self._unit_cache.get(count)
        if cached is not None:
            return cached
        if count == 0:
            points = np.empty((0, 3), dtype=np.float64)
        else:
            index = np.arange(count, dtype=np.float64)
            z = 1.0 - 2.0 * (index + 0.5) / count
            radius = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
            golden = math.pi * (3.0 - math.sqrt(5.0))
            phi = index * golden
            points = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
            norms = np.linalg.norm(points, axis=1)
            if np.any(norms <= 0.0):
                raise RuntimeError("degenerate unit sphere sample")
            points = points / norms[:, None]
        points.setflags(write=False)
        self._unit_cache[count] = points
        return points

    def sample_boundary(self, time: float, count: int) -> QueryPoints:
        """Deterministic sphere-surface queries in world coordinates."""
        self._require_time(time)
        center, _ = self.kinematics(time)
        points = center + self.config.radius * self._unit_sphere_points(count)
        ids = np.arange(_count(points, "positions", 3), dtype=np.int64)
        return QueryPoints(ids, points)

    def evaluate_surface(self, positions, time: float):
        """Return ``(phi, outward_normal, body_velocity)`` at world positions."""
        self._require_time(time)
        array = np.asarray(positions, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("positions must have shape (Q,3)")
        if not np.isfinite(array).all():
            raise ValueError("positions must be finite")
        center, center_velocity = self.kinematics(time)
        delta = array - center
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 0.0):
            raise ValueError("outward normal is undefined at the sphere centre")
        phi = distance - self.config.radius
        normal = delta / distance[:, None]
        velocity = np.broadcast_to(center_velocity, array.shape).copy()
        for value in (phi, normal, velocity):
            value.setflags(write=False)
        return phi, normal, velocity

    # -- clock ------------------------------------------------------------
    def _require_time(self, time: float) -> float:
        value = _real(time, "time")
        if value != self.time:
            raise ValueError("provider time must equal the current prescribed state")
        return value

    def advance(self, t_start: float, h: float) -> None:
        """Advance the prescribed clock by exactly ``h`` and bump the revision.

        Rejects a step that does not move the representable clock or that
        overflows it, and clears ``last_samples``: samples accepted at the old
        time/revision can never describe the new prescribed state.
        """
        self._require_time(t_start)
        step = _real(h, "h", positive=True)
        new_time = self.time + step
        if not np.isfinite(new_time):
            raise ValueError("provider time advance overflows the representable clock")
        if new_time <= self.time:
            raise ValueError("provider time advance does not move the clock")
        self.time = new_time
        self.version += 1
        self.last_advance = (t_start, step)
        self.last_samples = None

    def accept_cloth_samples(self, samples: ProviderBoundarySamples) -> None:
        """Accept cloth samples of the current revision, bound to one stream.

        The first *accepted* packet binds the run stream; foreign streams are
        rejected, as are stale cloth versions and provider revisions or sample
        times that do not match the current prescribed state. Re-querying the
        same completed cloth state is allowed (the version may repeat), but a
        newer provider revision never legitimises *older* cloth data. A rejected
        packet has no side effect: it can neither bind the stream nor advance
        the accepted cloth revision.
        """
        if not isinstance(samples, ProviderBoundarySamples):
            raise TypeError("samples must be ProviderBoundarySamples")
        stream_id = self._stream_id if self._stream_id is not None else samples.stream_id
        samples.validate_for(samples.snapshot, stream_id=stream_id, source=self.source)
        if samples.sample_time != self.time:
            raise ValueError("cloth samples are not at the current prescribed time")
        cloth_version = samples.snapshot.version
        if self._last_cloth_version is not None and cloth_version < self._last_cloth_version:
            raise ValueError("cloth samples contain an older cloth revision")
        # commit only after every check passed
        self._stream_id = stream_id
        self._last_cloth_version = cloth_version
        self.last_samples = samples

    # -- representation ---------------------------------------------------
    def representation_points(self, count: int = 32768) -> np.ndarray:
        """Exactly ``count`` fixed-seed volume samples (display/interface only).

        Directions are the deterministic equal-area sphere lattice and radii
        are ``R * U**(1/3)`` with ``U`` drawn from a seeded generator, which is
        a uniform sampling of the ball interior. Every requested point is
        returned: this is sampling, not an MPM particle body, and it is never
        advanced by the provider.
        """
        count = _integer(count, "count")
        if count == 0:
            return np.empty((0, 3), dtype=np.float64)
        directions = self._unit_sphere_points(count)
        generator = random.Random(self.seed)
        unit_radius = np.array([generator.random() ** (1.0 / 3.0) for _ in range(count)],
                               dtype=np.float64)
        center, _ = self.kinematics(self.time)
        points = center + (self.config.radius * unit_radius)[:, None] * directions
        points.setflags(write=False)
        return points


class EmptyMPMProvider:
    """Explicit empty MPM slot: provenance is declared, physics is not."""

    PHYSICAL_METHODS = ("sample_boundary", "evaluate_surface", "accept_cloth_samples", "advance", "step")

    def __init__(self, source_id: str = "mpm-empty"):
        self._source_id = _text(source_id, "source_id")
        self.last_samples: ProviderBoundarySamples | None = None

    @property
    def source(self) -> SourceIdentity:
        return SourceIdentity(self._source_id, "real_backend", 0)

    def _unimplemented(self, name: str):
        raise NotImplementedError(
            f"empty MPM slot: {name} requires a real material-point backend "
            "(particle state, contact loads and time integration)"
        )

    def sample_boundary(self, time: float, count: int) -> QueryPoints:
        self._unimplemented("sample_boundary")

    def evaluate_surface(self, positions, time: float):
        self._unimplemented("evaluate_surface")

    def accept_cloth_samples(self, samples: ProviderBoundarySamples) -> None:
        self._unimplemented("accept_cloth_samples")

    def advance(self, t_start: float, h: float) -> None:
        self._unimplemented("advance")

    def step(self, h: float) -> None:
        self._unimplemented("step")


__all__ = ["MotionConfig", "PrescribedSphere", "EmptyMPMProvider", "motion_displacement",
           "BoundaryProvider"]
