"""Versioned cloth state archives. Pure NumPy; never imports or runs a solver.

The v1 format records a single fixed-h run, including same-time topology events.
Hashes detect file changes, not physical correctness. Read-only arrays are an
API convenience, not a security boundary. This is not a restart checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np

from .coupling.contracts import ClothSnapshot
from .coupling.loads import LoadBatch, SnapshotClock, SourceIdentity
from .coupling.provider_contracts import ReferenceState
from .traction_schedule import normalize_schedule, integrate_schedule, validate_schedule_coverage
from .solver_convergence import convergence_policy, validate_convergence_record
from .support_balance import SCHEMA as BALANCE_SCHEMA, validate_balance

SCHEMA = "xpbd-cloth-run-v1"
STAGES = ("initial", "post_solve", "post_tear")
STATE_KEYS = {"positions", "velocity", "mass", "inverse_mass", "pinned", "triangles",
              "vertex_ids", "triangle_ids", "reference_positions", "root_vertex_ids",
              "triangle_mass", "edge_vertex_ids", "edge_active", "edge_broken", "cut_root_edges"}
LOAD_KEYS = {"load_vertex_ids", "load_impulse", "load_reference_area",
             "load_velocity_before", "load_velocity_uploaded"}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_read(text):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, parse_constant=invalid)


@dataclass(frozen=True)
class ClothRecord:
    metadata: dict
    arrays: dict[str, np.ndarray]

    def __post_init__(self):
        # Enforce JSON scalar metadata and detached owned arrays at the API edge.
        metadata = _json_read(json_text(self.metadata))
        arrays = {}
        for name, value in self.arrays.items():
            a = np.array(value, copy=True)
            if a.dtype.kind not in "biuf" or not np.isfinite(a).all():
                raise ValueError(f"{name}: finite numeric/bool arrays required")
            a.setflags(write=False)
            arrays[name] = a
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "arrays", arrays)


@dataclass(frozen=True)
class ClothRun:
    metadata: dict
    records: tuple[ClothRecord, ...]


def snapshot(record):
    a, m = record.arrays, record.metadata
    return ClothSnapshot(a["positions"], a["triangles"], a["vertex_ids"], a["triangle_ids"],
                         a["velocity"], a["inverse_mass"], a["pinned"],
                         *m["version"], m["time"], m["h"])


def reference(record, stream_id):
    a, m = record.arrays, record.metadata
    return ReferenceState(stream_id, m["version"][0], m["version"][2], a["vertex_ids"],
                          a["root_vertex_ids"], a["reference_positions"], a["pinned"],
                          a["triangle_ids"], a["triangle_mass"])


def _array(a, name, shape, kind="f"):
    v = a[name]
    if v.shape != shape or v.dtype.kind not in kind or not np.isfinite(v).all():
        raise ValueError(f"{name}: invalid shape/dtype/values")
    return v


def _close(actual, expected, tol, name):
    if not np.allclose(actual, expected, rtol=tol, atol=tol * 1e-5):
        raise ValueError(f"{name}: inconsistent values")


def _event_real(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name}: finite real scalar required")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name}: finite real scalar required") from error
    if not np.isfinite(value):
        raise ValueError(f"{name}: finite real scalar required")
    return value


def _event_edge(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v < 0 for v in value) or value[0] == value[1]):
        raise ValueError(f"{name}: two distinct nonnegative integer IDs required")
    return tuple(sorted(value))


def _fracture_config(config, steps):
    """Mirror the loaded-cloth modes without importing its numerical driver."""
    step = config.get("prescribed_cut_after_step")
    edge = config.get("cut_edge")
    critical = config.get("critical_strain")
    if step is None:
        if edge is not None:
            raise ValueError("cut_edge requires a prescribed cut step")
    else:
        if type(step) is not int or not 1 <= step <= steps:
            raise ValueError("prescribed cut step must be an integer within the run")
        if critical is not None:
            raise ValueError("prescribed and strain modes are mutually exclusive")
        edge = _event_edge(edge, "config.cut_edge")
    if critical is not None:
        critical = _event_real(critical, "critical_strain")
        if critical <= 0:
            raise ValueError("critical_strain must be positive")
    return step, edge, critical


def _validate_tear_cause(event, previous, mode):
    """Check internal event evidence, not authenticity or event completeness.

    No maximum-candidate/tie-order check is made. The driver archives both
    precision modes as the same float64 host geometry used by Edge.strain.
    """
    if not isinstance(event, dict) or event.get("kind") not in ("prescribed_cut", "strain_threshold"):
        raise ValueError("invalid topology event kind")
    edge = _event_edge(event.get("cut_vertex_ids"), "event.cut_vertex_ids")
    if type(event.get("step")) is not int or event["step"] != previous.metadata["version"][1]:
        raise ValueError("event.step must match the solved integer step")
    for name in ("new_vertex_ids", "parent_vertex_ids"):
        ids = event.get(name)
        if not isinstance(ids, list) or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError(f"event.{name}: nonnegative integer IDs required")
    recorded = _event_real(event.get("strain"), "event.strain")
    p = previous.arrays
    old_index = {int(v): i for i, v in enumerate(p["vertex_ids"])}
    edge_rows = {tuple(sorted(map(int, pair))): i for i, pair in enumerate(p["edge_vertex_ids"])}
    if edge not in edge_rows or any(v not in old_index for v in edge):
        raise ValueError("cut edge missing from previous topology")
    i, j = (old_index[v] for v in edge)
    row = edge_rows[edge]
    incident = sum(i in face and j in face for face in p["triangles"])
    if not p["edge_active"][row] or p["edge_broken"][row] or incident != 2:
        raise ValueError("cut requires an active intact interior edge")
    # Explicit promotion also avoids dtype-dependent subtraction/norm rounding.
    x = np.asarray(p["positions"], dtype=np.float64)
    rest = np.asarray(p["reference_positions"], dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            length = float(np.linalg.norm(x[i] - x[j]))
            rest_length = float(np.linalg.norm(rest[i] - rest[j]))
            strain = 0. if rest_length <= 1e-12 else (length - rest_length) / rest_length
    except FloatingPointError as error:
        raise ValueError("nonfinite cut geometry") from error
    if not np.isfinite([length, rest_length, strain]).all():
        raise ValueError("nonfinite cut geometry")
    if not np.isclose(recorded, strain, rtol=1e-12, atol=1e-12):
        raise ValueError("recorded cut strain disagrees with solved geometry")
    step, configured_edge, critical = mode
    if event["kind"] == "prescribed_cut":
        if step != event["step"] or edge != configured_edge or critical is not None:
            raise ValueError("prescribed event disagrees with cut configuration")
    else:
        if step is not None or configured_edge is not None or critical is None:
            raise ValueError("strain event requires strain-only configuration")
        # Deliberately no tolerance in the fracture gate.
        if not strain > critical:
            raise ValueError("cut geometry does not strictly exceed critical_strain")


def _diag(record, run, tol):
    a, m = record.arrays, record.metadata
    enabled = run["config"].get("convergence", False)
    if type(enabled) is not bool:
        raise ValueError("invalid convergence flag")
    if enabled:
        if run["backend"] != "cpu" or run.get("solver_convergence_policy") != convergence_policy(run["precision"]):
            raise ValueError("invalid convergence run policy")
        if m["stage"] == "post_solve":
            validate_convergence_record(m.get("solver_convergence"), run["precision"], run["config"]["iterations"])
        elif m.get("solver_convergence") is not None:
            raise ValueError("convergence result only belongs to solved state")
    elif m.get("solver_convergence") is not None or run.get("solver_convergence_policy") is not None:
        raise ValueError("convergence evidence requires explicit opt-in policy")
    if m["stage"] != "post_solve":
        if m["diagnostics"] is not None or any(k.startswith("diag_") for k in a):
            raise ValueError("diagnostics unavailable outside post_solve")
        return set()
    if set(m["diagnostics"]) != {"stretch", "bend", "pin"}:
        raise ValueError("missing diagnostic kinds")
    used = set()
    for kind, d in m["diagnostics"].items():
        if (d["kind"] != kind or d["version"] != m["version"] or d["time"] != m["time"]
                or d["h"] != m["h"] or d["coloring_version"] != m["version"][2]
                or d["repack_count"] != m["repack_count"] or d["precision"] != run["precision"]
                or d["backend"] != run["backend"] or d["export_stage"] != "post_solve_pre_tear"):
            raise ValueError("diagnostic generation/version mismatch")
        n = d["active_count"]
        if type(n) is not int or n < 0:
            raise ValueError("invalid diagnostic count")
        width = {"pin": 1, "stretch": 2, "bend": 4}[kind]
        prefix = f"diag_{kind}_"
        names = ["endpoint_ids", "rest", "lambdas", "color_offsets"]
        if kind == "bend":
            names += ["reference_weights", "stiffness", "compliance", "angles", "errors", "energy"]
            if (d["model"] != "dihedral" or d["angle_units"] != "rad"
                    or d["stiffness_units"] != "N m" or d["compliance_units"] != "1/(N m)"
                    or d["geometry_stage"] != "final_pin_targets_before_commit"
                    or d["lambda_stage"] != "last_bend_projection"):
                raise ValueError("unsupported bend diagnostic semantics")
        ids = _array(a, prefix + "endpoint_ids", (n, width), "iu")
        if not np.isin(ids, a["vertex_ids"]).all() or any(len(set(row)) != width for row in ids):
            raise ValueError("unknown/repeated diagnostic endpoints")
        shape = (n, 3) if kind == "pin" else (n,)
        for name in ("rest", "lambdas"):
            _array(a, prefix + name, shape)
        off = a[prefix + "color_offsets"]
        if (off.ndim != 1 or off.dtype.kind not in "iu" or len(off) == 0
                or off[0] != 0 or off[-1] != n or np.any(np.diff(off) <= 0)):
            raise ValueError("invalid color partition")
        for start, end in zip(off[:-1], off[1:]):
            if len(np.unique(ids[start:end])) != (end - start) * width:
                raise ValueError("same-color diagnostic endpoints overlap")
        if kind == "bend":
            for name in names[4:]:
                _array(a, prefix + name, (n,))
            for name in ("reference_weights", "stiffness", "compliance"):
                if np.any(a[prefix + name] <= 0):
                    raise ValueError("invalid positive bend parameter")
            _close(a[prefix + "stiffness"], run["config"]["bend_stiffness"] *
                   a[prefix + "reference_weights"], tol, "bend stiffness")
            _close(a[prefix + "compliance"] * a[prefix + "stiffness"], 1, tol, "bend compliance")
            _close(a[prefix + "energy"], .5 * a[prefix + "stiffness"] *
                   a[prefix + "errors"] ** 2, tol, "bend energy")
        used.update(prefix + name for name in names)
    return used


def _natural_policy(run, mode):
    policy = run.get("fracture_policy")
    if policy is None:
        return None  # Legacy v1: event cause is checked, completeness is not certified.
    if (not isinstance(policy, dict) or set(policy) != {"schema", "order", "root_cut", "max_events"}
            or policy["schema"] != "xpbd-fracture-policy-v2"
            or policy["order"] != "strain_desc_root_asc_vertex_asc"
            or policy["root_cut"] != "once_per_root_pair" or mode[2] is None):
        raise ValueError("unsupported natural fracture policy")
    limit = policy["max_events"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("invalid natural event budget")
    return policy


def _validate_natural_decision(record, critical, policy, event_count):
    """Independent archive geometry/incidence oracle; no solver/selector imports."""
    a, m = record.arrays, record.metadata
    decision = m.get("fracture_decision")
    if not isinstance(decision, dict):
        raise ValueError("v2 requires every solved candidate decision")
    index = {int(v): i for i, v in enumerate(a["vertex_ids"])}
    cuts = {tuple(map(int, pair)) for pair in a["cut_root_edges"]}
    candidates = []
    for edge, active, broken in zip(a["edge_vertex_ids"], a["edge_active"], a["edge_broken"]):
        pair = tuple(sorted(map(int, edge)))
        i, j = (index[v] for v in pair)
        incidence = sum(i in face and j in face for face in a["triangles"])
        if not active or broken or incidence != 2:
            continue
        roots = tuple(sorted((int(a["root_vertex_ids"][i]), int(a["root_vertex_ids"][j]))))
        rest = float(np.linalg.norm(a["reference_positions"][i] - a["reference_positions"][j]))
        if roots[0] == roots[1] or not np.isfinite(rest) or rest <= 1e-12:
            raise ValueError("ambiguous root pair or undefined reference edge")
        if roots in cuts:
            continue
        length = float(np.linalg.norm(a["positions"][i] - a["positions"][j]))
        strain = (length - rest) / rest
        if not np.isfinite(strain):
            raise ValueError("nonfinite candidate geometry")
        candidates.append(dict(edge=list(pair), root_edge=list(roots), strain=strain,
                               exceeds_threshold=bool(strain > critical)))
    candidates.sort(key=lambda item: item["edge"])
    # Group by both roots; a common endpoint root alone never suppresses an edge.
    representatives = {}
    for item in candidates:
        if item["exceeds_threshold"]:
            root = tuple(item["root_edge"])
            key = (-item["strain"], item["edge"])
            if root not in representatives or key < (-representatives[root]["strain"], representatives[root]["edge"]):
                representatives[root] = item
    ranked = [v["edge"] for v in sorted(representatives.values(),
              key=lambda item: (-item["strain"], item["root_edge"], item["edge"]))]
    exhausted = policy["max_events"] is not None and event_count >= policy["max_events"]
    expected = dict(schema="xpbd-fracture-decision-v2", version=m["version"],
        event_count_before=event_count, budget_exhausted=exhausted,
        candidates=candidates, ranked_edges=ranked, would_select=ranked[0] if ranked else None,
        selected_edge=ranked[0] if ranked and not exhausted else None,
        reason="no_candidate" if not ranked else "event_budget_exhausted" if exhausted else "selected")
    if set(decision) != set(expected) or type(decision["event_count_before"]) is not int:
        raise ValueError("invalid natural decision fields")
    if type(decision["budget_exhausted"]) is not bool:
        raise ValueError("invalid natural budget state")
    if (not isinstance(decision["candidates"], list)
            or len(decision["candidates"]) != len(candidates)):
        raise ValueError("candidate list incomplete")
    for actual, geometry in zip(decision["candidates"], candidates):
        if not isinstance(actual, dict) or set(actual) != set(geometry):
            raise ValueError("invalid candidate fields")
        strain = _event_real(actual["strain"], "candidate.strain")
        if not np.isclose(strain, geometry["strain"], rtol=1e-12, atol=1e-12):
            raise ValueError("candidate strain disagrees with geometry")
        if type(actual["exceeds_threshold"]) is not bool or actual["exceeds_threshold"] != (strain > critical):
            raise ValueError("candidate strict gate disagrees with exact stored strain")
        for key in ("edge", "root_edge", "exceeds_threshold"):
            if actual[key] != geometry[key]:
                raise ValueError("candidate identity or strict geometry gate mismatch")
    # Compare the ordering of recorded floats as well as independent geometry.
    seen, stored_rank = set(), []
    for item in sorted(decision["candidates"], key=lambda v: (-v["strain"], v["root_edge"], v["edge"])):
        root = tuple(item["root_edge"])
        if item["exceeds_threshold"] and root not in seen:
            stored_rank.append(item["edge"])
            seen.add(root)
    if stored_rank != ranked:
        raise ValueError("candidate ranking is numerically ambiguous or inconsistent")
    for key in set(expected) - {"candidates"}:
        if decision[key] != expected[key]:
            raise ValueError(f"natural decision mismatch: {key}")


def _validate_natural_transition(previous, record, event_count, tol):
    p, a = previous.arrays, record.arrays
    pm, m, event = previous.metadata, record.metadata, record.metadata["event"]
    if (event.get("schema") != "xpbd-cloth-event-v2" or type(event.get("sequence")) is not int
            or event["sequence"] != event_count + 1
            or sorted(event["cut_vertex_ids"]) != pm["fracture_decision"]["selected_edge"]):
        raise ValueError("natural event sequence/selection mismatch")
    if (m["version"] != [pm["version"][0], pm["version"][1], pm["version"][2] + 1]
            or m["time"] != pm["time"] or m["repack_count"] != pm["repack_count"] + 1):
        raise ValueError("natural event must commit exactly one same-time rebuild")
    old_index = {int(v): i for i, v in enumerate(p["vertex_ids"])}
    new_index = {int(v): i for i, v in enumerate(a["vertex_ids"])}
    root_edge = sorted(int(p["root_vertex_ids"][old_index[v]]) for v in event["cut_vertex_ids"])
    if event.get("cut_root_edge") != root_edge:
        raise ValueError("natural event root identity mismatch")
    parents = dict(zip(event["new_vertex_ids"], event["parent_vertex_ids"]))
    if len(parents) != len(event["new_vertex_ids"]):
        raise ValueError("duplicate new stable identity")
    # Validate immediate sources from actual triangle rewrites, even when several
    # pre-event vertices already share a root and happen to have equal velocity.
    for before_face, after_face in zip(p["triangles"], a["triangles"]):
        before_ids = p["vertex_ids"][before_face]
        after_ids = a["vertex_ids"][after_face]
        if [parents.get(int(v), int(v)) for v in after_ids] != before_ids.tolist():
            raise ValueError("triangle rewrite disagrees with immediate parent")
    for old_id, old_row in old_index.items():
        group = [new_index[old_id]] + [new_index[new] for new, parent in parents.items() if parent == old_id]
        _close(a["mass"][group].sum(), p["mass"][old_row], tol, "immediate-parent mass")
        _close((a["mass"][group, None] * a["velocity"][group]).sum(axis=0),
               p["mass"][old_row] * p["velocity"][old_row], tol, "immediate-parent momentum")
    for new, parent in parents.items():
        for name in ("reference_positions", "pinned"):
            if not np.array_equal(a[name][new_index[new]], p[name][old_index[parent]]):
                raise ValueError("immediate-parent reference/pin inheritance mismatch")
    broken = {tuple(sorted(map(int, edge))) for edge, flag in zip(p["edge_vertex_ids"], p["edge_broken"]) if flag}
    broken.add(tuple(sorted(event["cut_vertex_ids"])))
    for edge, active, flag in zip(a["edge_vertex_ids"], a["edge_active"], a["edge_broken"]):
        key = tuple(sorted(map(int, edge)))
        rows = [new_index[v] for v in key]
        incident = sum(all(v in face for v in rows) for face in a["triangles"])
        expected_broken = key in broken
        if flag != expected_broken or active != (not expected_broken or incident == 1):
            raise ValueError("actual cut edge flags disagree with topology commit")


def tear_balance(before, after):
    """Zero-time topology commit. These are endpoint states, with no load window."""
    jump = {"schema": "xpbd-tear-balance-v1"}
    for suffix, state in (("before", before), ("after", after)):
        mass = state["mass"]
        momentum = mass[:, None] * state["velocity"]
        jump["mass_" + suffix] = float(mass.sum())
        jump["P_" + suffix] = momentum.sum(axis=0).tolist()
        jump["L_" + suffix] = np.cross(state["positions"], momentum).sum(axis=0).tolist()
    return jump


def validate_records(run, records, *, complete=True):
    """Validate the recorded contract, not solver equilibrium or material truth."""
    measured = run["config"].get("support_balance", False)
    if type(measured) is not bool or (measured and
            (run["precision"] != "f64" or run["backend"] != "cpu" or run["config"]["h"] != .001
             or run["config"].get("precision") != "f64" or run["config"].get("arch") != "cpu"
             or run["config"].get("convergence") is not True)):
        raise ValueError("invalid support balance configuration")
    reaction = BALANCE_SCHEMA if measured else "unmeasured"
    if (run["schema"] != SCHEMA or run["units"] != "SI" or run["coordinate_frame"] != "world"
            or run["precision"] not in ("f32", "f64") or run["backend"] not in ("cpu", "cuda")
            or run["load_model"] not in ("reference_area_fixed_direction", "reference_area_piecewise_linear")
            or run["support_reaction"] != reaction):
        raise ValueError("unsupported run schema/semantics")
    if type(run["steps"]) is not int or run["steps"] < 1 or not records:
        raise ValueError("invalid run length")
    fracture_mode = _fracture_config(run["config"], run["steps"])
    policy = _natural_policy(run, fracture_mode)
    event_count = solve_count = 0
    schedule = run["config"].get("traction_schedule")
    if run["load_model"] == "reference_area_fixed_direction":
        if schedule is not None:
            raise ValueError("constant load model cannot contain a schedule")
    else:
        schedule = normalize_schedule(schedule)
        validate_schedule_coverage(schedule, run["steps"] * run["config"]["h"])
    source = SourceIdentity(**run["source"])
    clock, seen_batches = SnapshotClock(run["stream_id"]), set()
    tol = 3e-6 if run["precision"] == "f32" else 3e-12
    previous = previous_ref = None
    for index, record in enumerate(records):
        a, m = record.arrays, record.metadata
        if ("support_balance" in m) != (measured and m["stage"] == "post_solve"):
            raise ValueError("support balance belongs to each measured solve window only")
        if (m["schema"] != SCHEMA or m["stream_id"] != run["stream_id"] or m["stage"] not in STAGES
                or m["h"] != run["config"]["h"] or not np.isfinite(m["h"]) or m["h"] <= 0
                or type(m["repack_count"]) is not int or m["repack_count"] < 1):
            raise ValueError("invalid record metadata")
        s = snapshot(record)
        clock.publish(s)
        r = reference(record, run["stream_id"])
        r.validate_for(s, run["stream_id"])
        n, t = len(s.vertex_ids), len(s.triangle_ids)
        mass = _array(a, "mass", (n,))
        if np.any(mass <= 0):
            raise ValueError("invalid mass")
        expected_mass = np.zeros(n)
        np.add.at(expected_mass, s.triangles.ravel(), np.repeat(a["triangle_mass"] / 3, 3))
        _close(mass, expected_mass, tol, "reference mass distribution")
        _close(s.inverse_mass, np.where(s.pinned, 0, 1 / mass), tol, "inverse mass")
        edges = a["edge_vertex_ids"]
        _array(a, "edge_vertex_ids", (len(edges), 2), "iu")
        edge_set = {tuple(sorted(pair)) for pair in edges}
        expected_edges = {tuple(sorted((int(s.vertex_ids[tri[i]]), int(s.vertex_ids[tri[j]]))))
                          for tri in s.triangles for i, j in ((0, 1), (1, 2), (2, 0))}
        if edge_set != expected_edges or len(edge_set) != len(edges):
            raise ValueError("edge table does not match triangles")
        for name in ("edge_active", "edge_broken"):
            _array(a, name, (len(edges),), "b")
        cuts = a["cut_root_edges"]
        _array(a, "cut_root_edges", (len(cuts), 2), "iu")
        if not np.isin(cuts, r.root_vertex_ids).all() or any(x >= y for x, y in cuts):
            raise ValueError("invalid root cut table")
        if policy is not None and len({tuple(pair) for pair in cuts}) != len(cuts):
            raise ValueError("duplicate root cut table entry")
        if policy is None:
            if m.get("fracture_decision") is not None or (isinstance(m.get("event"), dict) and "schema" in m["event"]):
                raise ValueError("v2 evidence requires its run-level fracture policy")
        elif m["stage"] == "post_solve":
            _validate_natural_decision(record, fracture_mode[2], policy, event_count)
        elif m.get("fracture_decision") is not None:
            raise ValueError("candidate decision only belongs to solved state")
        used = STATE_KEYS | _diag(record, run, tol)
        if index == 0:
            if m["stage"] != "initial" or s.step_index != 0 or m["load"] is not None or m["event"] is not None:
                raise ValueError("run must begin with initial state")
            if policy is not None and (len(cuts) or np.any(a["edge_broken"]) or not np.all(a["edge_active"]) or s.time != 0.):
                raise ValueError("v2 natural run must begin intact at time zero")
        else:
            p, pm, ps = previous.arrays, previous.metadata, snapshot(previous)
            if policy is not None and pm["stage"] == "post_solve":
                selected = pm["fracture_decision"]["selected_edge"] is not None
                if selected != (m["stage"] == "post_tear"):
                    raise ValueError("missing or unexpected natural event after decision")
            if s.epoch != ps.epoch:
                raise ValueError("one archive cannot change epoch")
            if m["stage"] == "post_solve":
                solve_count += 1
                if policy is not None and (s.step_index != ps.step_index + 1 or s.time != ps.time + s.dt):
                    raise ValueError("natural run has missing/repeated physical step")
                for name in STATE_KEYS - {"positions", "velocity"}:
                    if not np.array_equal(a[name], p[name]):
                        raise ValueError(f"solve changed topology/reference/mass: {name}")
                if m["event"] is not None or m["repack_count"] != pm["repack_count"]:
                    raise ValueError("unexpected solve topology event")
                load = m["load"]
                batch = LoadBatch(run["stream_id"], source, tuple(load["version"]), load["batch_id"],
                                  load["t_start"], load["t_end"], a["load_vertex_ids"], a["load_impulse"])
                batch.validate_for(ps, stream_id=run["stream_id"], source=source, h=s.dt, capacity=n)
                if batch.batch_id in seen_batches or not np.array_equal(batch.vertex_ids, s.vertex_ids):
                    raise ValueError("duplicate batch or incomplete load table")
                seen_batches.add(batch.batch_id)
                area = _array(a, "load_reference_area", (n,))
                _close(area, expected_mass / run["config"]["areal_density"], 1e-12, "load reference area")
                if schedule is None:
                    _close(batch.impulse, area[:, None] * np.asarray(run["config"]["traction"]) * s.dt,
                           1e-12, "reference traction impulse")
                else:
                    # Shared integral gives consistency; independent hand-integrated
                    # test oracles establish the schedule quadrature correctness.
                    integral = integrate_schedule(schedule, batch.t_start, batch.t_end)
                    for key, expected in (("traction_scale_integral_s", integral),
                                          ("mean_scale", integral / s.dt)):
                        value = load[key]
                        if (type(value) not in (int, float) or not np.isfinite(value) or value < 0):
                            raise ValueError(f"{key}: finite nonnegative numeric scalar required")
                        _close(value, expected, 1e-12, key)
                    _close(batch.impulse, area[:, None] * np.asarray(run["config"]["traction"]) * integral,
                           1e-12, "scheduled reference traction impulse")
                vb = _array(a, "load_velocity_before", (n, 3))
                vu = _array(a, "load_velocity_uploaded", (n, 3))
                _close(vb, p["velocity"], tol, "preload velocity")
                _close(vu, vb + batch.impulse * p["inverse_mass"][:, None], tol, "single impulse application")
                audit = load["audit"]
                for key, value in (("external_total", batch.impulse.sum(axis=0)),
                                   ("external_free", batch.impulse[~s.pinned].sum(axis=0)),
                                   ("external_pinned", batch.impulse[s.pinned].sum(axis=0)),
                                   ("pinned_external_cancellation", -batch.impulse[s.pinned].sum(axis=0)),
                                   ("momentum_before", (mass[:, None] * vb).sum(axis=0)),
                                   ("momentum_uploaded", (mass[:, None] * vu).sum(axis=0)),
                                   ("momentum_solved", (mass[:, None] * s.velocity).sum(axis=0))):
                    _close(audit[key], value, tol, key)
                if audit["support_reaction"] != reaction:
                    raise ValueError("support reaction status differs from run policy")
                if measured:
                    used |= validate_balance(record, previous, run["config"])
                used |= LOAD_KEYS
            elif m["stage"] == "post_tear":
                if pm["stage"] != "post_solve" or m["load"] is not None or m["repack_count"] <= pm["repack_count"]:
                    raise ValueError("tear must follow solved state without consuming load")
                r.validate_successor(previous_ref)
                event = m["event"]
                if measured:
                    if event.get("balance_jump") != tear_balance(p, a):
                        raise ValueError("invalid zero-time mass/momentum/angular momentum record")
                elif "balance_jump" in event:
                    raise ValueError("tear balance requires a measured run")
                if run["config"].get("convergence", False):
                    if event.get("solver_stop_reason") != pm["solver_convergence"]["stop_reason"]:
                        raise ValueError("event convergence status differs from its source solve")
                elif "solver_stop_reason" in event:
                    raise ValueError("event convergence status requires opt-in policy")
                _validate_tear_cause(event, previous, fracture_mode)
                if (event["kind"] not in ("prescribed_cut", "strain_threshold") or event["gap"] != 0
                        or event["before_version"] != pm["version"] or event["after_version"] != m["version"]
                        or event["time"] != m["time"] or event["step"] != s.step_index):
                    raise ValueError("invalid topology event identity")
                old_index = {int(v): i for i, v in enumerate(ps.vertex_ids)}
                edge = event["cut_vertex_ids"]
                if tuple(sorted(edge)) not in {tuple(sorted(e)) for e in p["edge_vertex_ids"]}:
                    raise ValueError("cut edge missing from previous topology")
                cut_root = tuple(sorted(p["root_vertex_ids"][[old_index[v] for v in edge]]))
                old_cuts = {tuple(e) for e in p["cut_root_edges"]}
                if cut_root in old_cuts or {tuple(e) for e in cuts} != old_cuts | {cut_root}:
                    raise ValueError("cut transition missing or duplicated")
                if not np.array_equal(r.root_vertex_ids[s.triangles], previous_ref.root_vertex_ids[ps.triangles]):
                    raise ValueError("tear changed reference faces")
                old_n = len(ps.vertex_ids)
                for name in ("positions", "velocity"):
                    _close(a[name][:old_n], p[name], tol, "existing tear state")
                if event["new_vertex_ids"] != s.vertex_ids[old_n:].tolist():
                    raise ValueError("invalid new vertex map")
                parents = event["parent_vertex_ids"]
                if len(parents) != n - old_n or any(v not in edge for v in parents):
                    raise ValueError("invalid parent map")
                for row, parent in zip(range(old_n, n), parents):
                    j = old_index[parent]
                    if r.root_vertex_ids[row] != previous_ref.root_vertex_ids[j]:
                        raise ValueError("invalid parent provenance")
                    for name in ("positions", "velocity"):
                        _close(a[name][row], p[name][j], tol, "duplicate inheritance")
                for root in np.unique(r.root_vertex_ids):
                    _close(mass[r.root_vertex_ids == root].sum(),
                           p["mass"][previous_ref.root_vertex_ids == root].sum(), tol, "root mass conservation")
                _close((mass[:, None] * s.velocity).sum(axis=0),
                       (p["mass"][:, None] * ps.velocity).sum(axis=0), tol, "tear momentum")
                if policy is not None:
                    _validate_natural_transition(previous, record, event_count, tol)
                event_count += 1
            else:
                raise ValueError("initial state repeated")
        if set(a) != used:
            raise ValueError(f"unexpected/missing record arrays: {set(a) ^ used}")
        previous, previous_ref = record, r
    if complete and snapshot(records[-1]).step_index != run["steps"]:
        raise ValueError("completed run has missing physical steps")
    if complete and policy is not None:
        if solve_count != run["steps"]:
            raise ValueError("completed natural run has missing decisions")
        last = records[-1].metadata
        if last["stage"] == "post_solve" and last["fracture_decision"]["selected_edge"] is not None:
            raise ValueError("completed natural run is missing its final selected event")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plain_file(root, name):
    if not isinstance(name, str) or re.fullmatch(r"record_[0-9]{6}\.npz", name) is None:
        raise ValueError("unsafe archive filename")
    path = root / name
    st = path.lstat()
    if (not stat.S_ISREG(st.st_mode) or path.is_symlink()
            or getattr(st, "st_file_attributes", 0) & 0x400 or path.resolve().parent != root.resolve()):
        raise ValueError("archive member must be a plain local file")
    return path


def _read_records(root, manifest):
    records = []
    for index, item in enumerate(manifest["records"]):
        if item["index"] != index or item["file"] != f"record_{index:06d}.npz":
            raise ValueError("invalid record sequence")
        path = _plain_file(root, item["file"])
        if sha256(path) != item["sha256"]:
            raise ValueError("record hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            metadata = _json_read(str(data["metadata"].item()))
            arrays = {k: data[k] for k in data.files if k != "metadata"}
        if any(item[k] != metadata[k] for k in ("stage", "version", "time", "h")):
            raise ValueError("manifest/record metadata mismatch")
        records.append(ClothRecord(metadata, arrays))
    validate_records(manifest["run"], records)
    return tuple(records)


def load_run(directory):
    """Read only complete archives; no recovery, writes, device or solver access."""
    root = Path(directory)
    manifest = _json_read((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
        raise ValueError("archive is not complete or has an unknown schema")
    return ClothRun(deepcopy(manifest["run"]), _read_records(root, manifest))


class ArchiveWriter:
    """Single writer; a failure is terminal and completed records stay on disk."""
    def __init__(self, directory, run_metadata):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = dict(schema=SCHEMA, status="recording", run=deepcopy(run_metadata), records=[])
        _atomic_json(self.root / "manifest.json", self.manifest)

    def append(self, record):
        if self.manifest["status"] != "recording":
            raise RuntimeError("archive writer is closed")
        try:
            index = len(self.manifest["records"])
            name = f"record_{index:06d}.npz"
            temporary = self.root / (name + ".tmp")
            with temporary.open("xb") as output:
                np.savez_compressed(output, metadata=np.asarray(json_text(record.metadata)), **record.arrays)
            os.replace(temporary, self.root / name)
            item = {k: record.metadata[k] for k in ("stage", "version", "time", "h")}
            item.update(index=index, file=name, sha256=sha256(self.root / name))
            self.manifest["records"].append(item)
            _atomic_json(self.root / "manifest.json", self.manifest)
        except Exception as error:
            self.fail(error)
            raise

    def complete(self):
        if self.manifest["status"] != "recording":
            raise RuntimeError("archive writer is closed")
        try:
            _read_records(self.root, self.manifest)
            self.manifest["status"] = "complete"
            _atomic_json(self.root / "manifest.json", self.manifest)
        except Exception as error:
            self.fail(error)
            raise

    def fail(self, error):
        self.manifest["status"] = "failed"
        self.manifest["failure"] = dict(type=type(error).__name__, message=str(error))
        _atomic_json(self.root / "manifest.json", self.manifest)
