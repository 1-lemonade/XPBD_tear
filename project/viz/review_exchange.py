"""Adapt windows already validated by exchange_archive.read_archive for offline review."""
from __future__ import annotations

from math import hypot

from ..coupling.exchange_archive import ARCHIVE
from .review_cloth import face_components, support_points


def _frame(index, stage, state, surface, roots, parents, cuts, version, time, h,
           cumulative, *, exchange=None, event=None, balance=None, arrays=None):
    vertex_ids = state["vertex_ids"]
    return dict(
        index=index, stage=stage, version=version, time=time, h=h, event=event,
        positions=state["positions"], triangles=surface["triangles"],
        triangle_ids=surface["triangle_ids"], vertex_ids=vertex_ids,
        root_vertex_ids=roots, parent_vertex_ids=[parents.get(v) for v in vertex_ids],
        pinned=state["pinned"], cut_edges=cuts,
        components=face_components([[vertex_ids[i] for i in face] for face in surface["triangles"]],
                                   surface["triangle_ids"], inactive_edges=cuts),
        support=balance, support_points=support_points(balance, arrays),
        exchange=exchange, cumulative_input_impulse_norm_Ns=hypot(*cumulative),
        max_displacement_m=None, bend_energy_J=None,
    )


def _parameter_values(windows, key):
    values = list(dict.fromkeys(record[key] for record in windows))
    return values[0] if len(values) == 1 else values


def exchange_case(windows):
    """Keep the actual solve and any zero-time cut in their recorded order."""
    first = windows[0]
    start = first["start"]
    roots = start["vertex_ids"]
    parents = {}
    cumulative = [0., 0., 0.]
    frames = [_frame(0, "initial", start, start, roots, parents, first["cut_edges"],
                     first["cloth"]["start_version"], start["time"], start["dt"], cumulative)]
    for record in windows:
        cloth, peer = record["cloth"], record["peer"]
        start = record["start"]
        event = cloth["event"]
        window = {key: cloth[key] for key in
                  ("window_index", "stream_id", "source", "batch_id", "t_start", "t_end",
                   "h", "previous_dt", "start_version", "solved_version", "committed_version")}
        window.update(
            particle_dt=record["particle_dt"], substeps=record["substeps"],
            iterations=cloth["convergence"]["iterations"], stop_reason=cloth["convergence"]["stop_reason"],
        )
        cumulative = [old + delta for old, delta in zip(cumulative, peer["cloth_impulse"])]
        measured = (record["h"] == .001 and cloth["balance"] is not None
                    and cloth["balance_arrays"] is not None)
        frames.append(_frame(
            len(frames), "post_solve", cloth["solved"], start, roots, parents, record["cut_edges"],
            cloth["solved_version"], cloth["t_end"], record["h"], cumulative,
            exchange=dict(window, G=peer["cloth_impulse"], peer_impulse=peer["peer_impulse"],
                          is_committed=event is None),
            balance=cloth["balance"] if measured else None, arrays=cloth["balance_arrays"] if measured else None,
        ))
        roots = cloth["root_vertex_ids"]
        if event is not None:
            parents.update(zip(event["new_vertex_ids"], event["parent_vertex_ids"]))
            # The window ended at the solve. Commit changes topology at the same time.
            frames.append(_frame(
                len(frames), "post_tear", cloth["committed"], record["end"], roots, parents,
                cloth["cut_edges"], cloth["committed_version"], cloth["t_end"], record["h"], cumulative,
                exchange=dict(window, G=None, peer_impulse=None, is_committed=True), event=event,
            ))
    parameters = {
        "场景": "合成材料对端 · 同窗点载荷交换",
        "布步 h [s]": _parameter_values(windows, "h"),
        "材料子步 [s]": _parameter_values(windows, "particle_dt"),
        "物理步数 [步]": len(windows),
        "载荷来源": first["source"]["source_id"],
        "弯曲刚度 B [N·m]": None,
        "拉伸柔度 [m/N]": None,
        "撕裂应变阈值 [1]": None,
    }
    return dict(kind="exchange", archive_schema=ARCHIVE, stream_id=first["stream_id"],
                parameters=parameters, frames=frames)
