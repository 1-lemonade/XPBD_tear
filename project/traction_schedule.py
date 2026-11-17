"""Pure-Python scheduling for reference-area, fixed-direction traction.

Node times are seconds and scales are nonnegative, dimensionless binary64
values. Integration returns seconds; multiplying it by a reference area and
the base traction gives a window impulse. This module has no solver/device
dependency and does not advance, round, or reconstruct the solver's clock.

Only an out-of-domain window end may be clamped: the inclusive upper allowance
is exactly 16 successive ``nextafter(last_time, +inf)`` operations. Starts,
interior knots, and values already inside the domain are never snapped. The
original load timestamps remain the caller's responsibility. This tolerance
does not guarantee coverage of arbitrarily long repeated-addition clocks.

Extreme inputs that produce nonfinite intermediates or a zero result for a
strictly positive interpolated scale/segment area are rejected. No claim of
exact real arithmetic or support for every finite binary64 extreme is made.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
import math
from numbers import Real
from pathlib import Path
import sys


MAX_NODES = 1024
MAX_FILE_BYTES = 65536
_END_ULPS = 16

Schedule = tuple[tuple[float, float], ...]


def _nonnegative_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number, not bool/string")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be representable as a finite number") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def normalize_schedule(nodes) -> Schedule:
    """Validate 2..MAX_NODES list/tuple pairs and detach them into float tuples.

    The first node must be at zero; all subsequent times must remain strictly
    increasing after conversion to binary64. None is not a schedule: callers
    keep their existing constant-traction path separate.
    """
    if not isinstance(nodes, (list, tuple)) or not 2 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"schedule must contain 2..{MAX_NODES} nodes")
    normalized = []
    previous_time = None
    for index, node in enumerate(nodes):
        if not isinstance(node, (list, tuple)) or len(node) != 2:
            raise ValueError(f"schedule node {index} must be [time_s, scale]")
        time = _nonnegative_number(node[0], f"node {index} time")
        scale = _nonnegative_number(node[1], f"node {index} scale")
        if index == 0:
            if time != 0:
                raise ValueError("schedule must start at time zero")
            time = 0.0
        elif time <= previous_time:
            raise ValueError("schedule times must be strictly increasing")
        normalized.append((time, scale))
        previous_time = time
    return tuple(normalized)


def _covered_end(last_time: float, end_time: float) -> float:
    if end_time <= last_time:
        return end_time
    allowed_end = last_time
    for _ in range(_END_ULPS):
        allowed_end = math.nextafter(allowed_end, math.inf)
    if end_time > allowed_end:
        raise ValueError("schedule does not cover the requested end time")
    return last_time


def validate_schedule_coverage(schedule, end_time) -> None:
    """Require coverage through a finite nonnegative endpoint, with 16 ULP slack.

    The endpoint is supplied by the caller. This function does not replace an
    actual repeated-addition endpoint with ``steps * h`` or create a new clock.
    Individual integration windows are still checked against their own times.
    """
    nodes = normalize_schedule(schedule)
    end = _nonnegative_number(end_time, "coverage end time")
    _covered_end(nodes[-1][0], end)


def _finite_sum(values, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise ValueError(f"{name} overflow") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must remain finite")
    return result


def _interpolate(left, right, time: float) -> float:
    if time == left[0]:
        return left[1]
    if time == right[0]:
        return right[1]
    if left[1] == right[1]:
        return left[1]
    fraction = (time - left[0]) / (right[0] - left[0])
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid interpolation fraction")
    # Use a convex combination of nonnegative scales. Do not silently lose a
    # positive interior scale through underflow at extreme binary64 magnitudes.
    terms = (left[1] * (1.0 - fraction), right[1] * fraction)
    if not all(math.isfinite(value) for value in terms):
        raise ValueError("scale interpolation must remain finite")
    result = _finite_sum(terms, "interpolated scale")
    if result == 0 and (left[1] > 0 or right[1] > 0):
        raise ValueError("positive interpolated scale underflow")
    return result


def integrate_schedule(normalized, t_start, t_end) -> float:
    """Integrate one positive-length window, splitting at every internal knot.

    Revalidation is bounded by MAX_NODES and keeps this public helper safe for
    archive data as well as already-normalized driver configuration. Knot
    lookup uses bisect; only knots strictly inside this window are integrated.
    A zero-amplitude interval is valid and returns zero. A zero-duration window
    is invalid, including one collapsed by the permitted end clamp.
    """
    nodes = normalize_schedule(normalized)
    start = _nonnegative_number(t_start, "window start")
    end = _nonnegative_number(t_end, "window end")
    if end <= start:
        raise ValueError("integration window must have positive length")
    if start >= nodes[-1][0]:
        raise ValueError("window start must be strictly before the schedule end")
    end = _covered_end(nodes[-1][0], end)
    if end <= start:
        raise ValueError("clamped integration window must have positive length")

    times = tuple(node[0] for node in nodes)
    first = bisect_right(times, start)
    last = bisect_left(times, end)
    points = (start, *times[first:last], end)
    areas = []
    for left_time, right_time in zip(points[:-1], points[1:]):
        index = bisect_right(times, left_time) - 1
        left_scale = _interpolate(nodes[index], nodes[index + 1], left_time)
        right_scale = _interpolate(nodes[index], nodes[index + 1], right_time)
        high = max(left_scale, right_scale)
        # Sum before halving for small scales so equal subnormals survive;
        # halve first only when the unscaled sum could overflow. A tiny addend
        # lost by halving in the latter case is below the large term's ULP.
        if high <= sys.float_info.max * 0.5:
            average = _finite_sum((left_scale, right_scale), "trapezoid scale sum") * 0.5
        else:
            average = _finite_sum((left_scale * 0.5, right_scale * 0.5), "trapezoid mean scale")
        if high > 0 and average == 0:
            raise ValueError("positive trapezoid mean scale underflow")
        area = (right_time - left_time) * average
        if not math.isfinite(area):
            raise ValueError("trapezoid area must remain finite")
        if high > 0 and area == 0:
            raise ValueError("positive trapezoid area underflow")
        areas.append(area)
    return _finite_sum(areas, "schedule integral")


def _invalid_json_constant(value):
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def load_schedule_file(path) -> Schedule:
    """Read a UTF-8 JSON node list, bounded before decoding or parsing.

    At most MAX_FILE_BYTES + 1 bytes are read; the extra byte detects oversize
    input without relying on a potentially stale stat size. I/O errors remain
    I/O errors, while invalid data raises ValueError. No file content executes.
    """
    with Path(path).open("rb") as source:
        data = source.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"schedule file exceeds {MAX_FILE_BYTES} bytes")
    try:
        nodes = json.loads(data.decode("utf-8"), parse_constant=_invalid_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("schedule file must contain a UTF-8 JSON node list") from error
    return normalize_schedule(nodes)
