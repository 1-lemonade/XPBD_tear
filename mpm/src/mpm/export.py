"""Run directory writer, export contract v1 (; a superset of v0).

<out_root>/<name>/: config.json (resolved input, loadable again with --config),
run_info.json (module descriptions, derived quantities, backend, command line,
warnings, and the `export` manifest that names the contract and lists every file,
frame field and diag column), particles.npz (per-particle constants), diag.csv (one
row per frame, appended as the run goes), frames/NNNNN.npz (only when frames are
exported), summary.json, profile.json (only with runtime.kernel_profiler); scenes may
add figures. An existing non-empty run directory is never overwritten unless asked.
"""
import json
import math
import pathlib

import numpy as np

CONTRACT = "mpm-particle-v1"

# written by the scene frame loop (scenes/_cli.py), part of every diag row
TIMING_COLUMNS = {
    "time_sim_s": ("s", "wall time advancing this frame's substeps (ti.sync included)"),
    "time_diag_s": ("s", "wall time of this row's diagnostics"),
    "time_io_s": ("s", "wall time exporting this frame (0 when frames are off)"),
}

# U4 writes these; names and meaning are fixed now, readers must tolerate their absence.
# Sign convention as for every sink: impulse delivered BY the material TO the cloth;
# all impulses are cumulative from the start and never reset (per exchange period H:
# difference of two rows; contact_epoch counts completed periods).
RESERVED = dict(
    diag_columns={
        "contact_JP_x": ("N s", "contact impulse material -> cloth, particle-side measure (P), cumulative"),
        "contact_JP_y": ("N s", "same, y"), "contact_JP_z": ("N s", "same, z"),
        "contact_JG_x": ("N s", "contact impulse material -> cloth, grid measure (G), cumulative; "
                                "also the momentum sink 'contact'"),
        "contact_JG_y": ("N s", "same, y"), "contact_JG_z": ("N s", "same, z"),
        "contact_Jabs_x": ("N s", "part of contact_JG taken by fixed cloth nodes (G_on_pinned; J_absorbed in "
                                  "), cumulative"),
        "contact_Jabs_y": ("N s", "same, y"), "contact_Jabs_z": ("N s", "same, z"),
        "contact_epoch": ("count", "exchange periods H completed"),
        "contact_active": ("count", "particles constrained by the contact block in the last substep"),
        "contact_candidates": ("count", "particles in the candidate table"),
        "contact_overflow": ("count", "candidate table overflows since start"),
        "contact_ambiguous": ("count", "particles with an ambiguous nearest face in the last substep"),
        "contact_pen_max": ("m", "largest penetration depth in the last substep (recorded only)"),
    },
    frame_field_prefix="contact_",
    frame_fields={
        "contact_state": ("int8", ["N"], "1", "0 none, 1 candidate, 2 constrained"),
        "contact_face": ("int32", ["N"], "1", "nearest cloth face, -1 none"),
        "contact_dist": ("float32", ["N"], "m", "signed distance to that face"),
    },
    dirs=["cloth/"],
)


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))          # shortest round-trip form; nan / inf spelled as such


def frame_fields(sim) -> dict:
    fields = {
        "t": ("float64", [], "s", "frame time"),
        "step": ("int64", [], "count", "substeps done"),
        "x": ("float32", ["N", 3], "m", "position"),
        "v": ("float32", ["N", 3], "m/s", "velocity"),
        "flag": ("int8", ["N"], "1", "0 cone tip, 1 elastic, 2 plastic"),
    }
    if "q" in sim.model.state_members():
        fields["q"] = ("float32", ["N"], "1", "accumulated plastic measure of the constitutive model")
    if sim.contact is not None:
        fields.update(CONTACT_FRAME_FIELDS)
        if hasattr(sim.contact.surface, "cloth_state"):    # live surface (P2-U2): rows are window-local
            fields["contact_face"] = ("int32", ["N"], "1", "row of the nearest cloth triangle in the same-numbered "
                                      "cloth/NNNNN.npz triangles (live surface; stable ID: its triangle_ids), -1 none")
        if "contact_side" in sim.contact.state_members():
            fields["contact_side"] = ("int8", ["N"], "1", "remembered side of the surface (side = tracked): +1, -1, 0 none")
    return fields


# U4, written when the run has a contact block (the reserved names of v1)
CONTACT_FRAME_FIELDS = {
    "contact_state": ("int8", ["N"], "1", "0 none, 1 candidate, 2 constrained (last substep)"),
    "contact_face": ("int32", ["N"], "1", "row of the nearest cloth triangle in cloth/mesh.npz, -1 none"),
    "contact_dist": ("float32", ["N"], "m", "signed distance (1 - 2z) r to that triangle (gap = dist - r0); NaN if none"),
}
CLOTH_MESH = {
    "vertex_ids": ("int64", ["V"], "1", "stable vertex IDs (need not equal rows)"),
    "triangle_ids": ("int64", ["T"], "1", "stable triangle IDs"),
    "triangles": ("int32", ["T", 3], "1", "vertex rows; winding gives the normal"),
    "x0": ("float64", ["V", 3], "m", "initial vertex positions"),
    "pinned": ("int8", ["V"], "1", "1 = pinned vertex (its G share is G_on_pinned, formerly J_absorbed)"),
    "degenerate": ("int8", ["T"], "1", "1 = zero-area triangle, skipped by the contact block"),
}
# live surface (P2-U2): rows are window-local, so every cloth frame also carries them
CLOTH_FRAME_LIVE = {
    "vertex_ids": ("int64", ["V"], "1", "stable vertex IDs of the frame's rows (live surface)"),
    "triangle_ids": ("int64", ["T"], "1", "stable triangle IDs of the frame's rows (live surface)"),
    "triangles": ("int32", ["T", 3], "1", "vertex rows of this frame (live surface)"),
}
CLOTH_FRAME = {
    "t": ("float64", [], "s", "frame time"),
    "version": ("int64", [3], "1", "(epoch, step_index, topology_version) of the stub snapshot"),
    "x": ("float64", ["V", 3], "m", "vertex positions at t (snapshot + linear prediction)"),
    "v": ("float32", ["V", 3], "m/s", "vertex velocities"),
    "JP": ("float64", ["V", 3], "N s", "cumulative P-measure impulse per vertex, settled periods plus the open one"),
    "JG": ("float64", ["V", 3], "N s", "cumulative G-measure impulse attributed per vertex, same"),
}


def manifest(sim, diag_columns: dict, frames: bool, profile: bool) -> dict:
    """Self-description of a run's files for readers that never import the solver."""
    spec = lambda d: {k: dict(dtype=t, shape=s, unit=u, meaning=m) for k, (t, s, u, m) in d.items()}
    sinks = sim.sink_columns()
    files = {
        "config.json": "resolved input configuration (loadable with --config)",
        "run_info.json": "module descriptions, derived quantities, runtime, command line, warnings, this manifest",
        "particles.npz": "per-particle constants",
        "diag.csv": "one row per frame; columns in diag_columns; read by header name",
        "summary.json": "run results",
    }
    if frames:
        files["frames/NNNNN.npz"] = "per-frame particle state; fields in frame_fields"
    if profile:
        files["profile.json"] = "kernel profiler records"
    cloth = None
    if sim.contact is not None:
        files["cloth/mesh.npz"] = "stub cloth surface: IDs, triangles, initial positions, pinned flags"
        if frames:
            files["cloth/NNNNN.npz"] = "stub cloth state per frame; fields in cloth.frame_fields"
        live = hasattr(sim.contact.surface, "cloth_state")
        frame_spec = dict(CLOTH_FRAME)
        if live:
            files["cloth/mesh.npz"] = ("live cloth surface: IDs, triangles and positions of the first snapshot "
                                       "(rows are window-local; each cloth frame carries its own)")
            if frames:
                files["cloth/NNNNN.npz"] = ("live cloth state per frame: the active rows of the current snapshot, "
                                            "with their stable IDs and triangles; fields in cloth.frame_fields")
            frame_spec.update(CLOTH_FRAME_LIVE)
            frame_spec["version"] = ("int64", [3], "1", "(epoch, step_index, topology_version) of the current "
                                     "snapshot (live surface)")
            frame_spec["x"] = ("float64", ["V", 3], "m", "active vertex positions at t (snapshot + frozen-velocity "
                               "prediction, live surface)")
        cloth = dict(mesh_fields=spec(CLOTH_MESH), frame_fields=spec(frame_spec) if frames else {},
                     note="impulses are material -> cloth, cumulative; per exchange period H use "
                          "the difference of two states that both lie on period ends (contact_epoch)")
    return dict(
        contract=CONTRACT, n_particles=sim.n, files=files, frames_written=bool(frames),
        precision=sim.cfg["precision"],
        frame_precision_note="frames store x, v, q as float32 for every run precision; quantities "
                             "recomputed from frames are good to float32 only, diag.csv is exact",
        particles=spec({
            "m": ("float64", ["N"], "kg", "mass"),
            "V0": ("float64", ["N"], "m^3", "rest volume"),
            "mat_id": ("int32", ["N"], "1", "row of the constitutive parameter table"),
        }),
        frame_fields=spec(frame_fields(sim)),
        diag_columns=[dict(name=k, unit=u, meaning=m) for k, (u, m) in diag_columns.items()],
        momentum_sinks=sinks,
        cloth=cloth,
        balance=dict(
            residual="r = (P - P_ref) - M (gdt - gdt_ref) + sum_s (J_s - J_s_ref); ref = first diag row",
            P=["px", "py", "pz"], M="mass", gdt=["gdt_x", "gdt_y", "gdt_z"], p_abs="p_abs",
            residual_columns=["bal_px", "bal_py", "bal_pz"],
            scale="bal_scale = M |gdt - gdt_ref| + sum_s |J_s - J_s_ref| + p_abs",
            relative="bal_rel = |r| / bal_scale (0 when r = 0)"),
        reserved=dict(
            diag_columns=[dict(name=k, unit=u, meaning=m) for k, (u, m) in RESERVED["diag_columns"].items()],
            frame_field_prefix=RESERVED["frame_field_prefix"],
            frame_fields=spec(RESERVED["frame_fields"]),
            dirs=RESERVED["dirs"]),
    )


class RunWriter:
    def __init__(self, out_root, name: str, export_frames: bool, overwrite: bool = False):
        self.dir = pathlib.Path(out_root) / name
        if self.dir.exists() and any(self.dir.iterdir()) and not overwrite:
            raise FileExistsError(f"run directory {self.dir} is not empty (keeps earlier runs safe; pass --overwrite)")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.export_frames = bool(export_frames)
        if self.export_frames:
            (self.dir / "frames").mkdir(exist_ok=True)
        self._diag = None
        self._cols = None

    def write_json(self, fname: str, obj):
        """Strict JSON: NaN / inf become null."""
        (self.dir / fname).write_text(json.dumps(_json_safe(obj), indent=2, ensure_ascii=False, default=str,
                                                 allow_nan=False))

    def particles(self, sim):
        """Per-particle constants, written once."""
        P = sim.P
        np.savez_compressed(self.dir / "particles.npz", m=P.m.to_numpy().astype(np.float64),
                            V0=P.V0.to_numpy().astype(np.float64), mat_id=P.mat_id.to_numpy().astype(np.int32))

    def frame(self, sim, index: int, t: float):
        """Particle half of the export contract: t, step, x, v, flag, q."""
        if not self.export_frames:
            return
        P = sim.P
        data = dict(t=np.float64(t), step=np.int64(sim.step[None]), x=P.x.to_numpy().astype(np.float32),
                    v=P.v.to_numpy().astype(np.float32), flag=P.flag.to_numpy().astype(np.int8))
        if "q" in sim.model.state_members():
            data["q"] = P.q.to_numpy().astype(np.float32)
        if sim.contact is not None:
            face = P.contact_face.to_numpy().astype(np.int32)
            dist = P.contact_dist.to_numpy().astype(np.float32)
            dist[face < 0] = np.nan
            data.update(contact_state=P.contact_state.to_numpy().astype(np.int8), contact_face=face, contact_dist=dist)
            if "contact_side" in sim.contact.state_members():
                data["contact_side"] = P.contact_side.to_numpy().astype(np.int8)
            S = sim.contact.surface
            if hasattr(S, "cloth_state"):          # live surface (P2-U2): active rows, with their IDs and triangles
                np.savez_compressed(self.dir / "cloth" / f"{index:05d}.npz", t=np.float64(t), **S.cloth_state(t))
            else:
                np.savez_compressed(self.dir / "cloth" / f"{index:05d}.npz", t=np.float64(t),
                                    version=np.asarray(S.version(), np.int64), x=S.positions_at(t),
                                    v=S.v_s.to_numpy().astype(np.float32),
                                    JP=S.JP_cum.to_numpy() + S.JP_per.to_numpy(), JG=S.JG_cum.to_numpy() + S.JG_per.to_numpy())
        np.savez_compressed(self.dir / "frames" / f"{index:05d}.npz", **data)

    def cloth_mesh(self, sim):
        """cloth/mesh.npz, written once when the run has a contact block."""
        if sim.contact is None:
            return
        (self.dir / "cloth").mkdir(exist_ok=True)
        np.savez_compressed(self.dir / "cloth" / "mesh.npz", **sim.contact.surface.mesh_arrays())

    def open_diag(self, columns):
        self._cols = list(columns)
        self._diag = open(self.dir / "diag.csv", "w")
        self._diag.write(",".join(self._cols) + "\n")
        self._diag.flush()

    def diag_row(self, row: dict):
        """Append one row; 'momentum' is split into px, py, pz. Every column must be present."""
        r = dict(row)
        if "momentum" in r:
            r["px"], r["py"], r["pz"] = r.pop("momentum")
        missing = [k for k in self._cols if k not in r]
        if missing:
            raise KeyError(f"diag row lacks columns {missing}")
        self._diag.write(",".join(_fmt(r[k]) for k in self._cols) + "\n")
        self._diag.flush()

    def close(self):
        if self._diag is not None:
            self._diag.close()
            self._diag = None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
