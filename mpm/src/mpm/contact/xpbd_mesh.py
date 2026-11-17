"""XPBD coupling reference mesh `xpbd-coupling-reference-mesh-v1` (P1-U5; NumPy, init and
I/O only, no kernels).

The XPBD side (read-only reference geometry)
exported the coupling candidate cloth as `mesh.npz` + `manifest.json` in the shared
integration directory: 0.8 x 0.8 m, 52 x 52 vertices, areal density 0.5 kg/m^2, in the
XPBD native frame `xpbd-native-xy-v1` (plane z = 0, normal +Z). Its manifest documents the
construction; `reference_arrays()` rebuilds the same arrays from that documented formula
so that a file can be checked independently of XPBD code:

  id = row * 52 + column (row, column = 0..51); position (0.8 column / 51, 0.8 (1 - row / 51), 0)
  cell (r, c): a = 52 r + c, b = a + 1, q = a + 52, d = q + 1; faces (a, q, b), (b, q, d)
  (triangle IDs 2 (51 r + c) and 2 (51 r + c) + 1; diagonal b-q; normal +Z)
  triangle mass rho_A A_t; vertex mass sum over incident faces of M_t / 3
  supports: pinned_two_rows (rows 0 and 51, 104 vertices), pinned_corner36 (3 x 3 per corner)

The rigid map to the exchange world (+Y up) proposed by XPBD and accepted in the MPM P1-U5
reference transform: X = C + R (x - c), R = [[1,0,0],[0,0,1],[0,-1,0]],
c = (0.4, 0.4, 0), C = (0.5, 0.3, 0.5); vectors only rotate (G_native = R^T G_exchange).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "xpbd-coupling-reference-mesh-v1"
FRAME = "xpbd-native-xy-v1"
ROWS = COLS = 52
WIDTH = HEIGHT = 0.8
AREAL_DENSITY = 0.5
FIELDS = dict(positions=("float64", (2704, 3)), triangles=("int64", (5202, 3)), vertex_ids=("int64", (2704,)),
              root_vertex_ids=("int64", (2704,)), triangle_ids=("int64", (5202,)),
              pinned_two_rows=("bool", (2704,)), pinned_corner36=("bool", (2704,)),
              mass=("float64", (2704,)), triangle_mass=("float64", (5202,)))
SUPPORTS = ("pinned_two_rows", "pinned_corner36")
# values reported by XPBD for the published package cloth52-v1-20261109 
PUBLISHED = dict(name="cloth52-v1-20261109",
                 mesh_sha256="e6b155ccf4bb7f18ad4ef5be058c291194815c7996af343f6161498dc4cfe198",
                 manifest_sha256="45a469a9c1d68655b9cacf0431cc153706bc3683b25d0ba6cac3e8a90bd5dca5",
                 mesh_bytes=345594)
CENTER_FACE = dict(triangle_id=2600, vertex_ids=(1325, 1377, 1326), barycentric=(0.0, 0.5, 0.5))


def proposed_transform() -> dict:
    return dict(R=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], c=[0.4, 0.4, 0.0], C=[0.5, 0.3, 0.5],
                formula="X = C + R @ (x - c)", impulse_formula="G_native = transpose(R) @ G_exchange")


class RigidMap:
    """native <-> exchange world: X = C + R (x - c); vectors rotate only."""

    def __init__(self, R, c, C):
        self.R = np.asarray(R, np.float64)
        self.c = np.asarray(c, np.float64)
        self.C = np.asarray(C, np.float64)
        if self.R.shape != (3, 3) or not np.allclose(self.R.T @ self.R, np.eye(3), rtol=0, atol=1e-14) \
                or not np.isclose(np.linalg.det(self.R), 1.0, rtol=0, atol=1e-14):
            raise ValueError("transform R must be a proper rotation (R^T R = I, det R = +1)")

    def to_world(self, x):
        return (np.asarray(x, np.float64) - self.c) @ self.R.T + self.C

    def to_native(self, X):
        return (np.asarray(X, np.float64) - self.C) @ self.R + self.c

    def vec_to_world(self, v):
        return np.asarray(v, np.float64) @ self.R.T

    def vec_to_native(self, V):
        return np.asarray(V, np.float64) @ self.R

    def torque_to_native(self, K_world, J_world):
        """Angular impulse about the native origin from one about the world origin:
        sum x_n x G_n = c x J_n + R^T (K_w - C x J_w) (x_n = c + R^T (X - C), G_n = R^T G_w)."""
        K_world, J_world = np.asarray(K_world, np.float64), np.asarray(J_world, np.float64)
        J_n = self.vec_to_native(J_world)
        return np.cross(self.c, J_n) + self.vec_to_native(K_world - np.cross(self.C, J_world))

    def describe(self) -> dict:
        return dict(R=self.R.tolist(), c=self.c.tolist(), C=self.C.tolist(), formula="X = C + R @ (x - c)",
                    impulse_formula="G_native = transpose(R) @ G_exchange")


def reference_arrays() -> dict:
    """The candidate mesh rebuilt from the formula documented in the XPBD manifest."""
    ids = np.arange(ROWS * COLS, dtype=np.int64)
    rows, columns = np.divmod(ids, COLS)
    positions = np.column_stack((WIDTH * columns / (COLS - 1), HEIGHT * (1 - rows / (ROWS - 1)),
                                 np.zeros(ROWS * COLS)))
    faces = []
    for r in range(ROWS - 1):
        for c in range(COLS - 1):
            a = r * COLS + c
            faces.extend(((a, a + COLS, a + 1), (a + 1, a + COLS, a + COLS + 1)))
    triangles = np.asarray(faces, np.int64)
    corners = positions[triangles]
    area = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2] / 2
    triangle_mass = AREAL_DENSITY * area
    mass = np.zeros(ROWS * COLS)
    np.add.at(mass, triangles.ravel(), np.repeat(triangle_mass / 3, 3))
    return dict(positions=positions, triangles=triangles, vertex_ids=ids.copy(), root_vertex_ids=ids.copy(),
                triangle_ids=np.arange(len(triangles), dtype=np.int64),
                pinned_two_rows=(rows == 0) | (rows == ROWS - 1),
                pinned_corner36=((rows < 3) | (rows > ROWS - 4)) & ((columns < 3) | (columns > COLS - 4)),
                mass=mass, triangle_mass=triangle_mass)


def check_arrays(arrays: dict) -> dict:
    """Independent check of a candidate mesh against the documented construction; raises
    ValueError on the first mismatch. Positions, IDs, faces and masks must be equal bit for
    bit, masses within 1e-12 relative (they are sums)."""
    missing = set(FIELDS) - set(arrays)
    if missing:
        raise ValueError(f"mesh: missing arrays {sorted(missing)}")
    for name, (dtype, shape) in FIELDS.items():
        a = arrays[name]
        if a.dtype != np.dtype(dtype) or a.shape != shape or a.dtype.hasobject or not a.dtype.isnative:
            raise ValueError(f"mesh: {name} has {a.dtype}{a.shape}, expected native {dtype}{shape}")
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise ValueError(f"mesh: non-finite values in {name}")
    ref = reference_arrays()
    for name in ("positions", "triangles", "vertex_ids", "root_vertex_ids", "triangle_ids") + SUPPORTS:
        if not np.array_equal(arrays[name], ref[name]):
            raise ValueError(f"mesh: {name} differs from the documented construction")
    for name in ("mass", "triangle_mass"):
        if not np.allclose(arrays[name], ref[name], rtol=1e-12, atol=0):
            raise ValueError(f"mesh: {name} differs from the documented construction")
    T, X = arrays["triangles"], arrays["positions"]
    n = np.cross(X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]])
    if not (np.all(n[:, :2] == 0) and np.all(n[:, 2] > 0)):
        raise ValueError("mesh: winding is not +Z everywhere")
    e = np.sort(np.concatenate((T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]])), axis=1)
    _, inc = np.unique(e, axis=0, return_counts=True)
    tid = CENTER_FACE["triangle_id"]
    centre = np.asarray(CENTER_FACE["barycentric"]) @ X[T[tid]]
    out = dict(vertices=int(len(X)), triangles=int(len(T)), edges=int(len(inc)), boundary_edges=int((inc == 1).sum()),
               internal_edges=int((inc == 2).sum()), nonmanifold_edges=int((inc > 2).sum()),
               supports={k: int(arrays[k].sum()) for k in SUPPORTS}, total_mass_kg=float(arrays["mass"].sum()),
               reference_area_m2=float(n[:, 2].sum() / 2),
               center_face=dict(CENTER_FACE, triangle=T[tid].tolist(), position=centre.tolist()))
    if out["nonmanifold_edges"] or out["edges"] != 7905 or out["boundary_edges"] != 204:
        raise ValueError(f"mesh: unexpected edge structure {out}")
    if tuple(T[tid].tolist()) != CENTER_FACE["vertex_ids"] or not np.allclose(centre, [0.4, 0.4, 0.0], atol=1e-15):
        raise ValueError("mesh: centre reference face differs")
    return out


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, expect_sha256=None, expect_manifest_sha256=None) -> dict:
    """Read mesh.npz (allow_pickle=False) and, when present, the manifest.json next to it.
    Checks: the file's SHA-256 against the manifest's file record and against expect_sha256
    (if given), the manifest's own SHA-256 against expect_manifest_sha256 (if given), the
    schema and frame of the manifest, and the arrays (check_arrays). binding = "file" only
    with a valid manifest (a package must cite both digests); a bare mesh.npz gives
    "file-without-manifest", which is never publishable ."""
    path = Path(path)
    if path.is_dir():
        path = path / "mesh.npz"
    digest = sha256(path)
    if expect_sha256 is not None and digest != expect_sha256:
        raise ValueError(f"mesh: SHA-256 {digest} differs from the expected {expect_sha256}")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    man_path = path.parent / "manifest.json"
    manifest, man_digest = None, None
    if man_path.exists():
        man_digest = sha256(man_path)
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("frame_id") != FRAME or manifest.get("complete") is not True:
            raise ValueError("mesh manifest: schema, frame_id or complete flag differs")
        rec = manifest.get("file", {})
        if rec.get("sha256") != digest or rec.get("bytes") != path.stat().st_size:
            raise ValueError("mesh manifest: file record does not match mesh.npz")
    if expect_manifest_sha256 is not None and man_digest != expect_manifest_sha256:
        raise ValueError(f"mesh manifest: SHA-256 {man_digest} differs from the expected {expect_manifest_sha256}")
    checks = check_arrays(arrays)
    return dict(arrays=arrays, path=str(path), sha256=digest, bytes=path.stat().st_size, manifest=manifest,
                manifest_sha256=man_digest, checks=checks,
                binding="file" if manifest is not None else "file-without-manifest")


def reconstructed() -> dict:
    """The formula rebuild, for development where the published file is not available. Its
    output is never publishable: there is no file hash to bind a replay package to."""
    arrays = reference_arrays()
    return dict(arrays=arrays, path=None, sha256=None, bytes=None, manifest=None, manifest_sha256=None,
                checks=check_arrays(arrays), binding="formula-reconstruction")
