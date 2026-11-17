"""Read and write a finite synthetic exchange trace. No restart or provenance claim."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .contracts import ClothSnapshot, MappingConfig, MappingFeature, MappingStatus, QueryPoints, _ids
from .exchange import SCHEMA, WindowOffer, validate_surface
from .loads import LoadBatch, SourceIdentity
from .mapping import _Surface, map_points
from .tracking_topology import SurfaceTopology, canonical_cut_edges
from ..solver_convergence import validate_convergence_record
from ..support_balance import RAW_KEYS, SUPPORT_KEYS, build_balance

ARCHIVE = "xpbd-exchange-archive-v1"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _snapshot_dict(snapshot):
    return {f.name: _plain(getattr(snapshot, f.name)) for f in fields(snapshot)}


def make_record(offer, message, trace, receipt, end_snapshot):
    if message.schema != offer.schema or message.window_index != offer.window_index:
        raise ValueError("archive message differs from its negotiated window")
    return _plain(dict(schema=offer.schema, window_index=offer.window_index,
                       stream_id=offer.stream_id, source=asdict(offer.source),
                       h=offer.h, particle_dt=offer.particle_dt, substeps=offer.substeps,
                       start=_snapshot_dict(offer.snapshot), end=_snapshot_dict(end_snapshot),
                       cut_edges=sorted(offer.cut_edges), batch=asdict(message.batch),
                       peer=trace, cloth=receipt))


def _near(actual, expected, name):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.allclose(left, right, rtol=1e-12, atol=1e-12)):
        raise ValueError(f"inconsistent {name}")


def _check_peer(peer, offer, dense):
    if peer["schema"] != "xpbd-synthetic-peer-trace-v1" or peer["window_index"] != offer.window_index:
        raise ValueError("unexpected synthetic peer or window")
    ids = _ids(peer["particle_ids"], "particle_ids", len(peer["mass"]))
    mass = np.asarray(peer["mass"], dtype=float)
    force = np.asarray(peer["force_on_cloth_N"], dtype=float)
    x = np.asarray(peer["positions_before"], dtype=float)
    v = np.asarray(peer["velocity_before"], dtype=float)
    n = len(ids)
    if (mass.shape != (n,) or np.any(mass <= 0) or force.shape != (n, 3)
            or x.shape != force.shape or v.shape != force.shape or len(peer["ticks"]) != offer.substeps):
        raise ValueError("incomplete material ledger")
    vertex_rows = {int(v_id): row for row, v_id in enumerate(offer.snapshot.vertex_ids)}
    config = MappingConfig(**peer["mapping_config"])
    topology = SurfaceTopology(offer.snapshot, offer.cut_edges)
    distributed = np.zeros_like(dense)
    for i, tick in enumerate(peer["ticks"]):
        if tick["index"] != i or tick["tau"] != offer.sample_time(i):
            raise ValueError("missing, shifted or replayed material tick")
        _near(tick["query_positions"], x, "material query positions")
        eligible = np.asarray(tick["eligible"])
        status = np.asarray(tick["status"])
        if (eligible.shape != (n,) or eligible.dtype.kind != "b" or status.shape != (n,)
                or np.any(status[eligible] != 0)):
            raise ValueError("invalid material eligibility")
        geometry = offer.geometry_at_tick(i)
        mapped = map_points(geometry, QueryPoints(ids, x), config)
        for key in ("status", "triangle_ids", "vertex_ids"):
            if not np.array_equal(tick[key], getattr(mapped, key)):
                raise ValueError(f"material {key} differs from the frozen surface query")
        for key in ("barycentric", "closest_point"):
            stored = np.asarray(tick[key], dtype=float)
            reference = getattr(mapped, key)
            if stored.shape != reference.shape or not np.allclose(stored, reference, atol=1e-12,
                                                                  rtol=1e-12, equal_nan=True):
                raise ValueError(f"material {key} differs from the frozen surface query")
        qualified = np.zeros(n, dtype=bool)
        hit = mapped.status == MappingStatus.HIT
        velocity = np.full((n, 3), np.nan)
        valid_faces = _Surface(geometry, config).valid if np.any(hit & (mapped.feature == MappingFeature.EDGE)) else None
        for row in np.flatnonzero(hit):
            face = topology.row_for_id[int(mapped.triangle_ids[row])]
            weights = mapped.barycentric[row]
            kind = topology.classify(face, mapped.feature[row], weights, valid_faces)
            qualified[row] = kind in ("face", "interior_edge")
            velocity[row] = weights @ geometry.velocity[geometry.triangles[face]]
        stored_velocity = np.asarray(tick["surface_velocity"], dtype=float)
        if (not np.array_equal(eligible, qualified) or stored_velocity.shape != velocity.shape
                or not np.allclose(stored_velocity, velocity, atol=1e-12, rtol=1e-12, equal_nan=True)):
            raise ValueError("material eligibility or velocity differs from frozen geometry")
        j = np.where(eligible[:, None], force * offer.particle_dt, 0.)
        _near(tick["point_impulse_on_cloth"], j, "per-point impulse")
        moment = np.zeros(3)
        for row in np.flatnonzero(eligible):
            vids = mapped.vertex_ids[row]
            weights = mapped.barycentric[row]
            _near(weights.sum(), 1., "barycentric partition")
            slots = [vertex_rows[vid] for vid in vids]
            point = weights @ geometry.positions[slots]
            _near(tick["closest_point"][row], point, "predicted application point")
            moment += np.cross(point, j[row])
            distributed[slots] += weights[:, None] * j[row]
        _near(tick["cloth_angular_impulse"], moment, "frozen surface angular impulse")
        v = v - j / mass[:, None]
        x = x + v * offer.particle_dt
        _near(tick["particle_velocity_after"], v, "tick action/reaction")
        _near(tick["particle_positions_after"], x, "tick material position")
    _near(peer["positions_after"], x, "material end position")
    _near(peer["velocity_after"], v, "material end velocity")
    _near(distributed, dense, "per-vertex scatter")
    _near(peer["cloth_impulse"], dense.sum(axis=0), "material load on cloth")
    dp = (mass[:, None] * (v - np.asarray(peer["velocity_before"]))).sum(axis=0)
    _near(peer["peer_impulse"], dp, "material momentum change")
    _near(dp, -dense.sum(axis=0), "material action/reaction")


def _state(raw, *, snapshot=None):
    state = {key: np.asarray(raw[key]) for key in
             ("positions", "velocity", "pinned", "vertex_ids", "inverse_mass", "mass")}
    n = len(state["vertex_ids"])
    state["vertex_ids"] = _ids(state["vertex_ids"], "device vertex_ids", n, nonnegative=True)
    if state["pinned"].dtype.kind != "b" or state["pinned"].shape != (n,):
        raise ValueError("invalid device hard pins")
    for key in ("positions", "velocity", "inverse_mass", "mass"):
        a = state[key]
        if (a.shape != ((n, 3) if key in ("positions", "velocity") else (n,))
                or a.dtype.kind not in "fi" or not np.isfinite(a).all()):
            raise ValueError(f"invalid device {key}")
    if np.any(state["mass"] <= 0):
        raise ValueError("device mass must be positive")
    _near(state["inverse_mass"], np.where(state["pinned"], 0., 1 / state["mass"]), "device inertia")
    if snapshot is not None:
        for key in ("positions", "velocity", "inverse_mass", "vertex_ids", "pinned"):
            if not np.array_equal(state[key], getattr(snapshot, key)):
                raise ValueError(f"device {key} differs from publication")
    return state


def _check_split(event, offer, end, cuts, start_roots, end_roots, solved, committed):
    start = offer.snapshot
    old_rows = {int(vid): row for row, vid in enumerate(start.vertex_ids)}
    old_faces = start.vertex_ids[start.triangles]
    added = [int(vid) for vid in end.vertex_ids if vid not in old_rows]
    parents = event["parent_vertex_ids"]
    edge = event["cut_vertex_ids"]
    if (event["kind"] != "prescribed_cut" or event["gap"] != 0 or event["step"] != end.step_index
            or event["new_vertex_ids"] != added or len(parents) != len(added)
            or len(edge) != 2 or len(set(edge)) != 2 or not set(parents).issubset(edge)
            or sum(set(edge).issubset(face) for face in old_faces) != 2
            or not set(old_rows).issubset(end.vertex_ids.tolist())):
        raise ValueError("invalid split endpoints, new IDs or immediate parents")
    inherited = dict(zip(added, parents))
    parent_rows = [old_rows[inherited.get(vid, vid)] for vid in end.vertex_ids]
    roots = np.asarray(start_roots)[parent_rows]
    if not np.array_equal(end_roots, roots) or not np.array_equal(start.triangle_ids, end.triangle_ids):
        raise ValueError("split changed material face or root identity")
    if not np.array_equal(start.vertex_ids[parent_rows][end.triangles], old_faces):
        raise ValueError("split changed the parent of a triangle corner")
    for key in ("positions", "velocity", "pinned"):
        if not np.array_equal(solved[key][parent_rows], committed[key]):
            raise ValueError(f"zero-gap split changed inherited {key}")
    restored = np.zeros(len(start.vertex_ids))
    np.add.at(restored, parent_rows, committed["mass"])
    _near(restored, solved["mass"], "mass inherited from each parent")
    root_for_id = dict(zip(start.vertex_ids, start_roots))
    cut_roots = {tuple(sorted(root_for_id[v] for v in pair)) for pair in (*offer.cut_edges, edge)}
    current_edges = set()
    for a, b, c in end.triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if tuple(sorted((roots[u], roots[v]))) in cut_roots:
                current_edges.add(tuple(sorted((end.vertex_ids[u], end.vertex_ids[v]))))
    if current_edges != cuts:
        raise ValueError("published crack edges differ from the inherited cut roots")


def _check_cloth(c, offer, batch, dense, end, start_roots):
    start = offer.snapshot
    convergence = validate_convergence_record(c["convergence"], "f64", 96)
    solved_version = [start.epoch, start.step_index + 1, start.topology_version]
    if (c["schema"] != "xpbd-exchange-cloth-v1" or c["window_index"] != offer.window_index
            or c["stream_id"] != offer.stream_id
            or c["source"] != asdict(offer.source) or c["batch_id"] != batch.batch_id
            or c["previous_dt"] != start.dt or c["h"] != offer.h
            or c["t_start"] != start.time or c["t_end"] != batch.t_end
            or c["start_version"] != list(start.version) or c["solved_version"] != solved_version
            or c["committed_version"] != list(end.version) or end.epoch != start.epoch
            or end.step_index != start.step_index + 1 or end.time != batch.t_end or end.dt != offer.h):
        raise ValueError("cloth receipt differs from the exchange clock or source")
    if (not np.array_equal(c["vertex_ids"], start.vertex_ids)
            or not np.array_equal(c["batch_vertex_ids"], batch.vertex_ids)):
        raise ValueError("cloth receipt stable IDs differ from input")
    _near(c["batch_impulse"], batch.impulse, "batch receipt")
    _near(c["impulse"], dense, "device impulse ledger")
    before = _state(c["before"], snapshot=start)
    uploaded = _state(c["uploaded"])
    solved = _state(c["solved"])
    committed = _state(c["committed"], snapshot=end)
    for key in ("mass", "vertex_ids", "inverse_mass", "pinned"):
        if not all(np.array_equal(before[key], state[key]) for state in (uploaded, solved)):
            raise ValueError(f"device {key} changed during solve")
    _near(uploaded["positions"], before["positions"], "upload positions")
    _near(uploaded["velocity"], before["velocity"] + dense * before["inverse_mass"][:, None],
          "device velocity after upload")
    cuts = canonical_cut_edges(c["cut_edges"])
    validate_surface(end, cuts)
    for name in ("triangles", "triangle_ids"):
        if not np.array_equal(c[name], getattr(end, name)):
            raise ValueError(f"committed {name} differs from publication")
    if c["event"] is None:
        if (end.topology_version != start.topology_version or cuts != offer.cut_edges
                or not np.array_equal(start_roots, c["root_vertex_ids"])
                or not np.array_equal(start.triangles, end.triangles)
                or not np.array_equal(start.triangle_ids, end.triangle_ids)
                or any(not np.array_equal(v, committed[k]) for k, v in solved.items())):
            raise ValueError("topology or committed state changed without an event")
    else:
        e = c["event"]
        if (e["before_version"] != solved_version or e["after_version"] != list(end.version)
                or e["solver_stop_reason"] != convergence["stop_reason"]
                or e["time"] != end.time or end.topology_version <= start.topology_version):
            raise ValueError("invalid zero-time topology event")
        _check_split(e, offer, end, cuts, start_roots, c["root_vertex_ids"], solved, committed)
    if offer.h == .001:
        raw = c["balance_arrays"]
        metadata, arrays = build_balance(before, solved, dense,
            {key: np.asarray(raw[key], dtype=float) for key in RAW_KEYS}, h=offer.h,
            gravity=(0., 0., 0.), damping=1., start_version=start.version,
            end_version=solved_version, t_start=start.time, t_end=end.time)
        if c["balance"] != metadata:
            raise ValueError("U9 support ledger differs from endpoint measurements")
        for key in SUPPORT_KEYS:
            if not np.array_equal(raw[key], arrays[key]):
                raise ValueError("U9 support arrays differ from endpoint measurements")
    elif c["balance"] is not None or c["balance_arrays"] is not None:
        raise ValueError("U9 measurements outside the 1 ms contract")


def _validate(records):
    if not isinstance(records, list) or not records:
        raise ValueError("exchange archive needs at least one window")
    previous = None
    batch_ids = set()
    for index, r in enumerate(records):
        start, end = ClothSnapshot(**r["start"]), ClothSnapshot(**r["end"])
        source = SourceIdentity(**r["source"])
        if source.kind != "prescribed":
            raise ValueError("a synthetic exchange trace requires a prescribed source")
        offer = WindowOffer(r["schema"], r["stream_id"], source, r["window_index"], start,
                            r["particle_dt"], r["h"], r["cut_edges"])
        if offer.window_index != index or offer.substeps != r["substeps"]:
            raise ValueError("exchange archive has a missing, replayed or mistimed window")
        if previous is None:
            if start.time != 0 or start.step_index != 0:
                raise ValueError("archive must start at the beginning of a stream")
        else:
            if (previous["end"] != r["start"] or r["stream_id"] != previous["stream_id"]
                    or r["source"] != previous["source"] or previous["cloth"]["cut_edges"] != r["cut_edges"]
                    or any(previous["peer"][key] != r["peer"][key] for key in ("particle_ids", "mass"))
                    or previous["peer"]["positions_after"] != r["peer"]["positions_before"]
                    or previous["peer"]["velocity_after"] != r["peer"]["velocity_before"]):
                raise ValueError("discontinuous exchange publication or material ledger")
        b = r["batch"]
        batch = LoadBatch(**dict(b, source=SourceIdentity(**b["source"]), cloth_version=tuple(b["cloth_version"])))
        rows = batch.validate_for(start, stream_id=offer.stream_id, source=source,
                                  h=offer.h, capacity=len(start.vertex_ids))
        if batch.batch_id in batch_ids:
            raise ValueError("replayed batch in exchange archive")
        batch_ids.add(batch.batch_id)
        dense = np.zeros_like(start.velocity)
        dense[rows] = batch.impulse
        _check_peer(r["peer"], offer, dense)
        roots = start.vertex_ids if previous is None else previous["cloth"]["root_vertex_ids"]
        _check_cloth(r["cloth"], offer, batch, dense, end, roots)
        previous = r


def write_archive(path, records):
    """Publish a complete checked file without overwriting an earlier run."""
    path = Path(path)
    _validate(records)
    data = json.dumps({"schema": ARCHIVE, "windows": records}, allow_nan=False,
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        name = temp.name
    try:
        Path(name).write_bytes(data)
        os.link(name, path)
    finally:
        Path(name).unlink()


def read_archive(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")
    document = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if document["schema"] != ARCHIVE:
        raise ValueError("unsupported exchange archive")
    records = document["windows"]
    _validate(records)
    return records

