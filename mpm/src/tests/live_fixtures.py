"""P2-U1 test helpers: snapshots built from a stub surface or from the XPBD reference mesh, live
surface specs, a tear (slit) of the reference mesh, and the scene interface used by the D21
IR scan (tests/ir_global_tmp.py: SCENE_DEFAULTS, initial_positions, prepare), with the
geometry of scenes/configs/stub_static_load.json read through a live surface."""
import copy
import os
import pathlib
import sys

import numpy as np

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / "scenes"))
from mpm.contact import SurfaceSnapshot, sheet, xpbd_mesh  # noqa: E402
import stub_contact  # noqa: E402

SCENE_DEFAULTS = stub_contact.SCENE_DEFAULTS
SHEET = dict(type="sheet", center=[0.5, 0.3, 0.5], size=[0.8, 0.8], cells=[16, 16], normal_axis="y",
             normal_sign=1, pinned="rim", id_base=0, motion={"type": "fixed"})      # stub_static_load
# the published mesh directory: $MPM_PUBLISHED_MESH (a host test package, P2-U4), else in the shared directory
PUBLISHED_MESH = pathlib.Path(os.environ.get("MPM_PUBLISHED_MESH") or WORK.parent / "data" / xpbd_mesh.PUBLISHED["name"])


def live_spec(n_vertices, n_triangles, transform="identity", extra=(0, 0)):
    return dict(type="live", capacity=dict(vertices=n_vertices + extra[0], triangles=n_triangles + extra[1]),
                transform=transform)


def snapshot(positions, triangles, vertex_ids, triangle_ids, velocity=None, pinned=None, step=0, time=0.0,
             epoch=1, topo=0, dt=1e-3):
    N = len(positions)
    return SurfaceSnapshot(positions=np.asarray(positions, np.float64), triangles=np.asarray(triangles, np.int64),
                           vertex_ids=np.asarray(vertex_ids, np.int64),
                           triangle_ids=np.asarray(triangle_ids, np.int64),
                           velocity=np.zeros((N, 3)) if velocity is None else np.asarray(velocity, np.float64),
                           pinned=np.zeros(N, bool) if pinned is None else np.asarray(pinned, bool),
                           epoch=epoch, step_index=step, topology_version=topo, time=time, dt=dt)


def snapshot_of_stub(S, step, time, epoch=1, topo=0):
    """World-frame snapshot (for an identity live surface) equal to the stub's current x_s and
    v_s, with its triangles, IDs and pinned mask."""
    return snapshot(S.x_s.to_numpy(), S.T, S.vertex_ids, S.triangle_ids, S.v_s.to_numpy().astype(np.float64),
                    S.pinned_np.astype(bool), step=step, time=time, epoch=epoch, topo=topo)


def reference_mesh(published=True) -> dict:
    """The XPBD 52 x 52 reference mesh: the published mesh.npz (SHA-256 bound) when it is in
    the shared directory, else the documented formula."""
    if published and (PUBLISHED_MESH / "mesh.npz").exists():
        src = xpbd_mesh.load(PUBLISHED_MESH, xpbd_mesh.PUBLISHED["mesh_sha256"], xpbd_mesh.PUBLISHED["manifest_sha256"])
    else:
        src = xpbd_mesh.reconstructed()
    return dict(src["arrays"], binding=src["binding"], sha256=src["sha256"])


def reference_snapshot(a, positions=None, velocity=None, pinned="pinned_two_rows", **kw):
    return snapshot(a["positions"] if positions is None else positions, a["triangles"], a["vertex_ids"],
                    a["triangle_ids"], velocity, a[pinned] if isinstance(pinned, str) else pinned, **kw)


def slit(a, column=25, rows=(10, 40), id_offset=100000):
    """Tear the reference mesh along `column` between `rows` (exclusive ends): the vertices
    strictly inside the slit are duplicated (new stable IDs id + id_offset, appended rows) and
    the faces on the +x side use the copies, as XPBD does on a tear. Returns (positions,
    triangles, vertex_ids, triangle_ids, pinned_two_rows, duplicated original IDs)."""
    X, T = a["positions"].copy(), a["triangles"].copy()
    ids = a["vertex_ids"].copy()
    r, c = np.divmod(ids, xpbd_mesh.COLS)
    cut = np.flatnonzero((c == column) & (r > rows[0]) & (r < rows[1]))
    new_rows = len(X) + np.arange(len(cut))
    remap = np.arange(len(X))
    remap[cut] = new_rows
    xc = X[:, 0][T].mean(1)
    right = xc > X[cut[0], 0]
    T[right] = remap[T[right]]
    X = np.vstack([X, X[cut]])
    pin = np.concatenate([a["pinned_two_rows"], a["pinned_two_rows"][cut]])
    return X, T, np.concatenate([ids, ids[cut] + id_offset]), a["triangle_ids"].copy(), pin, ids[cut]


# ---------------------------------------------------------------- IR scan scene (ir_global_tmp.py)
def initial_positions(cfg):
    c = copy.deepcopy(cfg)
    c["contact"]["surface"] = dict(SHEET)
    return stub_contact.initial_positions(c)


def prepare(sim):
    X, T = sheet(SHEET)
    sim.contact.surface.ingest(snapshot(X, T, np.arange(len(X)), np.arange(len(T))))
