"""F3 one-way replay package for the XPBD heavy-load check (P1-U5; host I/O only, no kernels).

Format: the XPBD draft `xpbd-f3-replay-proposal-v1` (read-only input contract): one directory per run, `batches.npz` + `manifest.json`, all
windows of the run in one file, one logical LoadBatch per window. The producer writes the
NPZ, checks it by reading it back, and writes the manifest last; a directory without
`complete: true` must not be consumed, a published directory is never overwritten (a
revision is a new run_id).

ReplayRecorder is a contact-block period hook (ContactBlock.period_hooks): after every
settled window it copies the window's per-vertex G (exchange world, f64) and the window
deltas of the block's cumulative totals. package() turns the record into the arrays and
manifest in the XPBD native frame (G_native = R^T G_world).
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

SCHEMA = "xpbd-f3-replay-proposal-v1"
NPZ = "batches.npz"
MANIFEST = "manifest.json"
SOURCE = dict(source_id="mpm-couple.stub-contact.f3-n64", kind="real_backend", version=1)
CLOCK_RULE = "t = t_ns / 1e9 (IEEE-754 binary64, correctly rounded; t_ns = k * H_ns exact integers)"


def window_time(ns: int) -> float:
    return ns / 1e9


class ReplayRecorder:
    """Period hook: per window k the batch (window_index, version, clock, G per vertex in the
    world frame) and the deltas of the block's cumulative JG (material momentum lost to the
    contact), KG (its angular impulse about the world origin) and JP. Stops recording after
    `windows` windows (later windows raise: the run must be exactly as long as the package)."""

    def __init__(self, block, windows: int):
        S = block.surface
        self.W, self.V = int(windows), S.nv
        self.G = np.zeros((self.W, self.V, 3))
        self.J_material = np.zeros((self.W, 3))
        self.K_material = np.zeros((self.W, 3))
        self.J_P = np.zeros((self.W, 3))
        self.t_ns = np.zeros((self.W, 2), np.int64)
        self.t_solver = np.zeros((self.W, 2))
        self.version = np.zeros((self.W, 3), np.int64)
        # per window: constrained and penetrated particles (last substep), new misses and new
        # overflows in the window 
        self.counts = np.zeros((self.W, 4), np.int64)
        self.k = 0
        self._prev = [np.zeros(3), np.zeros(3), np.zeros(3)]
        self._prev_counts = (0, 0)
        block.period_hooks.append(self)

    def __call__(self, block):
        k = self.k
        if k >= self.W:
            raise RuntimeError(f"replay recorder: window {k} beyond the planned {self.W}")
        b = block.surface.last_batch()
        if b["window_index"] != k or b["t_start_ns"] is None:
            raise RuntimeError(f"replay recorder: batch {b['window_index']} / clock {b['t_start_ns']} at window {k}")
        self.G[k] = b["impulse_G"]
        self.t_ns[k] = (b["t_start_ns"], b["t_end_ns"])
        self.t_solver[k] = b["t_solver"]
        self.version[k] = b["cloth_version"]
        cum = [np.asarray(f[None].to_numpy(), np.float64) for f in (block.JG, block.KG, block.JP)]
        self.J_material[k], self.K_material[k], self.J_P[k] = (c - p for c, p in zip(cum, self._prev))
        self._prev = cum
        missed, overflow = int(block.missed[None]), int(block.cand_overflow[None] + block.bin_overflow[None])
        self.counts[k] = (block.n_active[None], block.n_pen[None], missed - self._prev_counts[0],
                          overflow - self._prev_counts[1])
        self._prev_counts = (missed, overflow)
        self.k += 1

    @property
    def complete(self) -> bool:
        return self.k == self.W


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(work: Path, extra=()) -> dict:
    """SHA-256 of every file the replay depends on, relative to Work_place (the project has
    no version control; the exact source identity is recorded)."""
    files = sorted(work.glob("mpm/**/*.py")) + [work / p for p in extra]
    return {str(f.relative_to(work)).replace("\\", "/"): sha256(f) for f in files if f.is_file()}


def package(rec: ReplayRecorder, surface, *, run_id: str, producer: dict, load_history: dict) -> tuple:
    """(arrays, manifest without the file record). Native frame through surface.rigid."""
    if not rec.complete:
        raise RuntimeError(f"replay: {rec.k} of {rec.W} windows recorded")
    if surface.spec["motion"]["type"] != "fixed":
        raise RuntimeError("replay: the stub must be fixed (K_to_cloth_ref_native uses the reference positions)")
    rigid, src, native = surface.rigid, surface.source, surface.native
    W, V = rec.W, rec.V
    H_ns = int(rec.t_ns[0, 1] - rec.t_ns[0, 0])
    k = np.arange(W, dtype=np.int64)
    if not (np.array_equal(rec.t_ns[:, 0], k * H_ns) and np.array_equal(rec.t_ns[:, 1], (k + 1) * H_ns)):
        raise RuntimeError("replay: window clock is not k * H_ns")
    if not np.array_equal(rec.version, np.column_stack((np.ones(W, np.int64), k, np.zeros(W, np.int64)))):
        raise RuntimeError("replay: stub versions are not (1, k, 0)")
    G = rigid.vec_to_native(rec.G)                               # [W, V, 3]
    x_ref = native["positions"]
    J_to_cloth = G.sum(1)
    J_material = rigid.vec_to_native(rec.J_material)
    K_to_cloth = np.cross(x_ref[None], G).sum(1)
    K_material = np.stack([rigid.torque_to_native(Kw, Jw) for Kw, Jw in zip(rec.K_material, rec.J_material)])
    masks = {name: native[name] for name in ("pinned_two_rows", "pinned_corner36")}
    arrays = dict(window_index=k, t_start_ns=rec.t_ns[:, 0].copy(), t_end_ns=rec.t_ns[:, 1].copy(),
                  vertex_ids=np.asarray(surface.vertex_ids, np.int64).copy(), impulse=G,
                  valid=np.ones((W, V), bool), J_material=J_material, J_to_cloth=J_to_cloth,
                  K_to_cloth_ref_native=K_to_cloth, K_material_native=K_material,
                  J_P=rigid.vec_to_native(rec.J_P), contact_counts=rec.counts.copy(),
                  **{f"G_on_{name}": G[:, m].sum(1) for name, m in masks.items()})
    for name, a in arrays.items():
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise RuntimeError(f"replay: non-finite values in {name}")
    stream = f"mpm-f3-replay:{run_id}"
    batches = []
    for i in range(W):
        ts, te = int(rec.t_ns[i, 0]), int(rec.t_ns[i, 1])
        batches.append(dict(window_index=i, batch_id=f"{stream}/w{i:05d}", stream_id=stream, source=SOURCE,
                            cloth_version=[int(v) for v in rec.version[i]], t_start=window_time(ts),
                            t_end=window_time(te), t_start_hex=window_time(ts).hex(), t_end_hex=window_time(te).hex(),
                            t_solver=[float(v) for v in rec.t_solver[i]], coordinate_frame="world", units="SI"))
    m = native["mass"]
    free = {name: ~mk for name, mk in masks.items()}
    dv = np.linalg.norm(G, axis=2) / m[None]                    # |G_i| / m_i [m/s], every vertex
    stats = dict(
        J_total_native=J_to_cloth.sum(0).tolist(),
        load_ratio_first_last=[float(-J_material[0, 2] / producer["Mg_H"]), float(-J_material[-1, 2] / producer["Mg_H"])],
        load_ratio_min_max=[float((-J_material[:, 2] / producer["Mg_H"]).min()),
                            float((-J_material[:, 2] / producer["Mg_H"]).max())],
        vertices_loaded_max=int((np.abs(G).sum(2) > 0).sum(1).max()),
        G_vertex_max_Ns=float(np.linalg.norm(G, axis=2).max()),
        dv_free_max_m_s={name: float(dv[:, f].max()) for name, f in free.items()},
        split_rel_max=float((np.linalg.norm(J_to_cloth - J_material, axis=1)
                             / np.maximum(np.linalg.norm(J_material, axis=1), 1e-300)).max()),
        K_lever_rel_max=float((np.linalg.norm(K_material - K_to_cloth, axis=1)
                               / np.maximum(np.linalg.norm(K_material, axis=1), 1e-300)).max()),
        G_on_pinned_total_native={f"G_on_{name}": arrays[f"G_on_{name}"].sum(0).tolist() for name in masks},
        counts_max=dict(zip(("active", "penetrated", "missed_new", "overflow_new"), rec.counts.max(0).tolist())),
        counts_sum=dict(zip(("missed", "overflow"), rec.counts[:, 2:].sum(0).tolist())))
    mesh = dict(binding=src["binding"], mesh_sha256=src["sha256"], mesh_manifest_sha256=src["manifest_sha256"],
                mesh_bytes=src["bytes"], path_at_production=src["path"], vertices=int(V),
                triangles=int(len(surface.T)), topology="one untorn topology for every window",
                checks=src["checks"])
    fields = {name: dict(dtype=str(a.dtype), shape=list(a.shape)) for name, a in arrays.items()}
    units = dict(t_start_ns="ns", t_end_ns="ns", impulse="N s", J_material="N s", J_to_cloth="N s",
                 K_to_cloth_ref_native="N m s", K_material_native="N m s", G_on_pinned_two_rows="N s",
                 G_on_pinned_corner36="N s", J_P="N s", contact_counts="count")
    for name in fields:
        fields[name]["units"] = units.get(name, "1")
    publishable = src["binding"] == "file" and src["sha256"] is not None and src["manifest_sha256"] is not None
    manifest = dict(
        schema=SCHEMA, complete=publishable, run_id=run_id, window_count=W, H_ns=H_ns, time_origin_ns=0,
        publishable=publishable,
        not_publishable_reason=None if publishable else (
            "mesh rebuilt from the documented formula, not read from the published mesh.npz (development run)"
            if src["binding"] == "formula-reconstruction" else
            "mesh.npz read without its manifest.json: the package could not cite both mesh digests"),
        mesh=mesh, coordinate_frame="world", frame_id="xpbd-native-xy-v1", units="SI",
        quantity="material_to_cloth_impulse",
        transform_status=src["transform_status"],
        transform=dict(rigid.describe(), status=src["transform_status"],
                       agreement=dict(proposed_by="not asserted by this candidate",
                                      accepted_by="not asserted by this candidate",
                                      user_signoff="pending")),
        source=SOURCE, source_stream_id=stream, producer=producer, load_history=load_history,
        clock_encoding=dict(canonical="integer nanoseconds t_start_ns / t_end_ns", float_rule=CLOCK_RULE,
                            float_fields="batches[k].t_start / t_end (and *_hex)",
                            solver_clock="batches[k].t_solver: MPM's accumulated substep clock, for audit only"),
        fields=fields,
        semantics=dict(
            impulse="G: realised impulse material -> cloth per stable vertex and window, N s, XPBD native world "
                    "frame; includes pinned vertices and explicit zeros; not divided by mass, not a force, "
                    "no support reaction R",
            valid="true: computed completely for that window and vertex",
            J_material="momentum the material lost to the contact in the window (grid measure, MPM sink), native",
            J_to_cloth="sum over vertices of impulse; differs from J_material by the barycentric split round-off",
            K_to_cloth_ref_native="sum_i x_ref,i x impulse_i about the native origin, x_ref = mesh positions "
                                  "(the rigid stub does not move, so window-start positions equal x_ref)",
            K_material_native="sum_g x_g x dJ_g of the realised grid impulse about the native origin; "
                              "K_material - K_to_cloth = sum_p (x_p - cp_p) x G_p, the contact-layer lever arm "
                              "(tangential part of G only); diagnostic, not a pass condition",
            G_on_pinned="sum of impulse over each support mask of the mesh (G_on_pinned, formerly J_absorbed)",
            J_P="particle-side target impulse (P measure) of the window, native; equals J_material with "
                "correction C, alpha = 1 (D28)",
            contact_counts="per window [constrained, penetrated] in its last substep and [new misses, new "
                           "overflows] during the window (int64; zero misses and overflows expected)",
            sign="material -> cloth (the cloth receives +impulse; the material lost it)"),
        batches=batches, summary=stats)
    return arrays, manifest


def write(out_dir, arrays: dict, manifest: dict) -> dict:
    """Create out_dir (must not exist), write the NPZ (exclusive), read it back and compare,
    then write the manifest (exclusive, last)."""
    out = Path(out_dir)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"replay package {out} exists: a revision needs a new run_id")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"replay parent {out.parent} must exist")
    out.mkdir()
    path = out / NPZ
    with path.open("xb") as fh:
        np.savez(fh, **arrays)
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(arrays):
            raise RuntimeError("replay: NPZ member set differs after writing")
        for name, a in arrays.items():
            b = z[name]
            if b.dtype != a.dtype or b.shape != a.shape or not np.array_equal(b, a):
                raise RuntimeError(f"replay: NPZ roundtrip differs for {name}")
    manifest = dict(manifest, file=dict(name=NPZ, bytes=path.stat().st_size, sha256=sha256(path)))
    with (out / MANIFEST).open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, allow_nan=False, indent=1)
        fh.write("\n")
    return dict(dir=str(out), npz_sha256=manifest["file"]["sha256"], npz_bytes=manifest["file"]["bytes"],
                manifest_sha256=sha256(out / MANIFEST), complete=manifest["complete"])


def runtime_record() -> dict:
    import taichi as ti
    return dict(platform=platform.platform(), python=sys.version.split()[0], numpy=np.__version__,
                taichi=".".join(str(v) for v in ti.__version__))
