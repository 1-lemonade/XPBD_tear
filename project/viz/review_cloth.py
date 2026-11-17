"""Read-only views of validated cloth states and static-support windows."""
from collections import defaultdict

import numpy as np

from .cloth_replay import payload_from_run


def face_components(triangles, triangle_ids, *, inactive_edges=()):
    """Group stable face IDs connected across active edges with two incident faces.

    The face corners and the inactive edge endpoints must use the same ID space.
    A shared vertex alone does not connect faces.
    """
    excluded = {tuple(sorted(edge)) for edge in inactive_edges}
    neighbors = {int(face): [] for face in triangle_ids}
    incident = defaultdict(list)
    for face, (a, b, c) in zip(triangle_ids, triangles):
        for ends in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted(ends))
            if edge not in excluded:
                incident[edge].append(int(face))
    for faces in incident.values():
        if len(faces) == 2:
            first, second = faces
            neighbors[first].append(second)
            neighbors[second].append(first)
    groups = []
    visited = set()
    for face in sorted(neighbors):
        if face in visited:
            continue
        pending = [face]
        visited.add(face)
        group = []
        while pending:
            current = pending.pop()
            group.append(current)
            for adjacent in neighbors[current]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    pending.append(adjacent)
        groups.append(sorted(group))
    return groups


def support_points(balance, arrays):
    """Return measured pin impulses and derived moments/mean forces in world axes."""
    if balance is None:
        return None
    positions = np.asarray(arrays["support_positions"], dtype=float).reshape(-1, 3)
    impulse = np.asarray(arrays["support_impulse"], dtype=float).reshape(-1, 3)
    moment = np.cross(positions - balance["origin"], impulse)
    force = impulse / balance["h"]
    return [
        dict(vertex_id=int(vertex), position=x.tolist(), J_support=j.tolist(),
             K_support=k.tolist(), mean_support_force=f.tolist())
        for vertex, x, j, k, f in zip(arrays["support_vertex_ids"], positions, impulse, moment, force)
    ]


def _cloth_support_points(balance, arrays, gravity):
    points = support_points(balance, arrays)
    # The validator checks the load and solve use the same stable ID ordering.
    # A support cancels the external and constraint impulses applied to its pin.
    rows = {int(vertex): row for row, vertex in enumerate(arrays["vertex_ids"])}
    gravity = np.asarray(gravity)
    for point in points:
        row = rows[point["vertex_id"]]
        point.update(
            J_support_load=(-arrays["load_impulse"][row]).tolist(),
            J_support_gravity=(-arrays["mass"][row] * gravity * balance["h"]).tolist(),
            J_support_stretch=(-arrays["stretch_impulse"][row]).tolist(),
            J_support_bend=(-arrays["bend_impulse"][row]).tolist(),
        )
    return points


def _parameters(metadata):
    config = metadata["config"]
    convergence = metadata.get("solver_convergence_policy", {})
    if config.get("critical_strain") is not None:
        fracture = "自然撕裂"
    elif config.get("prescribed_cut_after_step") is not None:
        fracture = "规定切割"
    else:
        fracture = "无切割"
    return {
        "场景": "独立布料 · " + fracture,
        "计算后端 / 精度": [metadata["backend"], metadata["precision"]],
        "布步 h [s]": config["h"],
        "材料子步 [s]": None,
        "物理步数 [步]": metadata["steps"],
        "每边顶点数 [个]": config.get("resolution"),
        "布宽 [m]": config.get("width"),
        "布高 [m]": config.get("height"),
        "面密度 [kg/m²]": config.get("areal_density"),
        "弯曲刚度 B [N·m]": config.get("bend_stiffness"),
        "拉伸柔度 [m/N]": config.get("stretch_compliance"),
        "撕裂应变阈值 [1]": config.get("critical_strain"),
        "自然事件预算 [个]": metadata.get("fracture_policy", {}).get("max_events"),
        "载荷来源": metadata["source"]["source_id"],
        "参考牵引 [N/m²]": config.get("traction"),
        "牵引时程 [s, 倍率]": config.get("traction_schedule"),
        "重力加速度 [m/s²]": config.get("gravity"),
        "速度阻尼倍率 [1]": config.get("damping"),
        "固定迭代数或停止上限 [次]": config.get("iterations"),
        "迭代方式": "SOR 残差停止" if convergence else "固定迭代",
        "SOR 松弛系数 [1]": convergence.get("omega"),
        "材料参数状态": metadata.get("parameter_status"),
    }


def cloth_case(run, fingerprint):
    """Adapt a complete run already read by cloth_archive.load_run."""
    frames = payload_from_run(run, fingerprint)["frames"]
    parents = {}
    for record, frame in zip(run.records, frames):
        arrays, meta = record.arrays, record.metadata
        event = frame["event"]
        if event is not None:
            parents.update(zip(event["new_vertex_ids"], event["parent_vertex_ids"]))
        balance = meta.get("support_balance")
        faces = arrays["vertex_ids"][arrays["triangles"]]
        inactive = arrays["edge_vertex_ids"][~arrays["edge_active"]]
        frame.update(
            triangle_ids=arrays["triangle_ids"].tolist(),
            parent_vertex_ids=[parents.get(vertex) for vertex in frame["vertex_ids"]],
            components=face_components(faces, arrays["triangle_ids"], inactive_edges=inactive),
            edge_active=arrays["edge_active"].tolist(),
            edge_broken=arrays["edge_broken"].tolist(),
            support=balance,
            support_points=(_cloth_support_points(balance, arrays, run.metadata["config"]["gravity"])
                            if balance is not None else None),
            exchange=None,
        )
    return dict(kind="cloth", archive_schema=run.metadata["schema"], stream_id=run.metadata["stream_id"],
                parameters=_parameters(run.metadata), frames=frames)
