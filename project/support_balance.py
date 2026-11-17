"""World-frame impulse and angular impulse for one measured static-support window.

The solver supplies endpoint impulses, including virtual impulses on pinned
endpoints. A support cancels those impulses and external loading at that pin.
The angular time correction is diagnostic and never enters the support force.
"""
from __future__ import annotations

import math

import numpy as np


SCHEMA = "xpbd-support-balance-v1"
RAW_KEYS = ("stretch_impulse", "bend_impulse", "stretch_angular_impulse",
            "bend_angular_impulse", "damping_impulse")
SUPPORT_KEYS = ("support_vertex_ids", "support_positions", "support_impulse")


def _array(value, name, shape, dtype):
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected finite {dtype} with shape {shape}")
    return array


def _version(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer))
                   or v < 0 for v in value)):
        raise ValueError("support balance requires three nonnegative version integers")
    return [int(v) for v in value]


def _real(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)):
        raise ValueError(f"{name}: expected a finite real number")
    return float(value)


def _state(data):
    ids = np.asarray(data["vertex_ids"])
    if (ids.ndim != 1 or len(ids) == 0 or ids.dtype != np.int64
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)):
        raise ValueError("invalid support balance vertex IDs")
    n = len(ids)
    mass = _array(data["mass"], "mass", (n,), np.float64)
    if np.any(mass <= 0):
        raise ValueError("support balance needs positive vertex masses")
    pinned = _array(data["pinned"], "pinned", (n,), np.bool_)
    if "inverse_mass" in data:
        inverse = _array(data["inverse_mass"], "inverse_mass", (n,), np.float64)
        if np.any(inverse[pinned] != 0) or np.any(inverse[~pinned] <= 0):
            raise ValueError("support balance requires zero inverse mass only at hard pins")
    return (ids, pinned, mass,
            _array(data["positions"], "positions", (n, 3), np.float64),
            _array(data["velocity"], "velocity", (n, 3), np.float64))


def _roundoff(actual, expected, scale, name):
    # This check relates two representations of the same endpoint operation.
    # It is distinct from a scenario's physical balance acceptance budget.
    tolerance = 128 * np.finfo(np.float64).eps * scale + 1e-16
    if not np.isfinite(expected).all() or np.any(np.abs(actual - expected) > tolerance):
        raise ValueError(f"inconsistent {name}")


def build_balance(before, solved, impulse, raw, *, h, gravity, damping,
                  start_version, end_version, t_start, t_end):
    """Reduce a real projection/damping trace; no support is inferred from ΔP."""
    h = _real(h, "h")
    t_start, t_end = _real(t_start, "t_start"), _real(t_end, "t_end")
    damping = _real(damping, "damping")
    start, end = _version(start_version), _version(end_version)
    if (h not in (0.001, 0.0005) or t_start < 0 or t_end != t_start + h or t_end <= t_start
            or end != [start[0], start[1] + 1, start[2]]):
        raise ValueError("invalid measured 1 ms/0.5 ms solve window/version")
    gravity = np.asarray(gravity)
    if (gravity.shape != (3,) or gravity.dtype.kind not in "iuf" or not np.isfinite(gravity).all()
            or not 0 <= damping <= 1):
        raise ValueError("invalid gravity or damping in support balance")
    ids, pin, mass, x0, v0 = _state(before)
    end_ids, end_pin, end_mass, x1, v1 = _state(solved)
    if not all(np.array_equal(a, b) for a, b in ((ids, end_ids), (pin, end_pin), (mass, end_mass))):
        raise ValueError("support balance window changed IDs, boundary, or mass")
    if (not np.array_equal(x0[pin], x1[pin]) or np.any(v0[pin]) or np.any(v1[pin])):
        raise ValueError("support balance requires stationary hard pins")
    shape = (len(ids), 3)
    load = _array(impulse, "load_impulse", shape, np.float64)
    if set(raw) != set(RAW_KEYS):
        raise ValueError("incomplete or unknown support measurement fields")
    arrays = {key: _array(raw[key], key, shape, np.float64) for key in RAW_KEYS}

    weight = mass[:, None] * gravity * h
    damp = arrays["damping_impulse"]
    pre_damp = (x1 - x0) / h
    velocity_scale = np.abs(pre_damp) + np.abs(v1)
    _roundoff(v1, damping * pre_damp, velocity_scale, "reconstructed velocity")
    _roundoff(damp, mass[:, None] * (v1 - pre_damp),
              mass[:, None] * velocity_scale, "measured damping impulse")
    if np.any(damp[pin]):
        raise ValueError("stationary pins cannot receive a damping impulse")

    stretch, bend = arrays["stretch_impulse"], arrays["bend_impulse"]
    support = -(load[pin] + weight[pin] + stretch[pin] + bend[pin])
    arrays.update(support_vertex_ids=ids[pin], support_positions=x0[pin], support_impulse=support)
    p0, p1 = mass[:, None] * v0, mass[:, None] * v1
    vectors = dict(origin=np.zeros(3), P_before=p0.sum(axis=0), P_after=p1.sum(axis=0),
                   L_before=np.cross(x0, p0).sum(axis=0), L_after=np.cross(x1, p1).sum(axis=0))
    for name, positions, values in (("load", x0, load), ("gravity", x0, weight),
                                    ("damping", x1, damp), ("support", x0[pin], support)):
        vectors[f"J_{name}"] = values.sum(axis=0)
        vectors[f"K_{name}"] = np.cross(positions, values).sum(axis=0)
    vectors["mean_support_force"] = vectors["J_support"] / h

    for name, values in (("load", load), ("gravity", weight), ("stretch", stretch), ("bend", bend)):
        vectors[f"J_support_{name}"] = -values[pin].sum(axis=0)
        vectors[f"K_support_{name}"] = -np.cross(x0[pin], values[pin]).sum(axis=0)
    for name, values in (("stretch", stretch), ("bend", bend)):
        instantaneous = arrays[f"{name}_angular_impulse"]
        at_start = np.cross(x0, values)
        # Fixed endpoints have exactly one lever arm through every projection.
        pin_scale = (np.max(np.abs(x0[pin]), axis=1, keepdims=True)
                     * np.max(np.abs(values[pin]), axis=1, keepdims=True)
                     + np.abs(instantaneous[pin]))
        _roundoff(instantaneous[pin], at_start[pin], pin_scale, f"{name} pin moment")
        vectors[f"J_{name}"] = values.sum(axis=0)
        vectors[f"K_{name}"] = instantaneous.sum(axis=0)
        vectors[f"K_{name}_time_correction"] = (at_start - instantaneous).sum(axis=0)

    processes = ("load", "gravity", "support", "damping")
    vectors["linear_residual"] = vectors["P_after"] - vectors["P_before"] - sum(
        vectors[f"J_{name}"] for name in processes)
    raw_residual = vectors["L_after"] - vectors["L_before"] - sum(
        vectors[f"K_{name}"] for name in processes)
    vectors["angular_residual_raw"] = raw_residual
    vectors["angular_residual_corrected"] = (
        raw_residual - vectors["K_stretch_time_correction"] - vectors["K_bend_time_correction"])
    if not all(np.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("nonfinite support balance aggregate")
    balance = dict(schema=SCHEMA, start_version=start, end_version=end, t_start=t_start, t_end=t_end, h=h)
    balance.update({key: value.tolist() for key, value in vectors.items()})
    return balance, arrays


def _check_metadata(actual, expected):
    if type(actual) is not dict or set(actual) != set(expected):
        raise ValueError("incomplete or unknown support balance metadata")
    for name, target in expected.items():
        value = actual[name]
        if name in ("start_version", "end_version"):
            valid = type(value) is list and _version(value) == target
        elif isinstance(target, list):
            valid = (type(value) is list and len(value) == 3
                     and all(type(v) in (float, int) and math.isfinite(v) for v in value)
                     and np.array_equal(value, target))
        elif isinstance(target, str):
            valid = type(value) is str and value == target
        else:
            valid = _real(value, name) == target
        if not valid:
            raise ValueError(f"support balance field differs from stored measurements: {name}")


def validate_balance(record, previous, config):
    """Check the solve window and reduction against archived states and raw data."""
    m, a = record.metadata, record.arrays
    if previous is None or m["stage"] != "post_solve":
        raise ValueError("support balance requires a solved positive-length window")
    pm = previous.metadata
    if (m["stream_id"] != pm["stream_id"] or config["precision"] != "f64"
            or config["arch"] != "cpu" or config["h"] != m["h"]):
        raise ValueError("support balance configuration or stream mismatch")
    load = m["load"]
    if (not isinstance(load, dict) or _version(load["version"]) != _version(pm["version"])
            or load["t_start"] != pm["time"] or load["t_end"] != m["time"]
            or not np.array_equal(a["load_vertex_ids"], previous.arrays["vertex_ids"])):
        raise ValueError("support balance differs from its load window or vertex order")
    used = set(RAW_KEYS + SUPPORT_KEYS)
    if not used.issubset(a):
        raise ValueError("missing support balance measurement arrays")
    expected, arrays = build_balance(
        previous.arrays, a, a["load_impulse"], {key: a[key] for key in RAW_KEYS},
        h=m["h"], gravity=config["gravity"], damping=config["damping"],
        start_version=pm["version"], end_version=m["version"], t_start=pm["time"], t_end=m["time"])
    for key in SUPPORT_KEYS:
        got = _array(a[key], key, arrays[key].shape, arrays[key].dtype)
        if not np.array_equal(got, arrays[key]):
            raise ValueError(f"{key} differs from static pin measurements")
    _check_metadata(m.get("support_balance"), expected)
    return used
