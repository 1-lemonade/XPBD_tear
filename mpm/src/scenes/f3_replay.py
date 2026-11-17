"""F3 static load on the XPBD coupling reference mesh (rigid stub, one-way): replay package
for the XPBD heavy-load / convergence check (P1-U5; D29).

    python scenes/f3_replay.py [--config scenes/configs/f3_replay_n64.json] [--mesh DIR|mesh.npz]
        [--mesh-sha256 HEX] [--mesh-manifest-sha256 HEX] [--replay-root DIR] [--run-id ID]
        [--set ...] [--arch cpu|cuda] [--out DIR]

The stub is the XPBD 52 x 52 candidate mesh (mpm/contact/xpbd_mesh.py), placed in the F3
position by the rigid map X = C + R (x - c); it stays fixed and records the realised
material -> cloth impulse G per stable vertex, one batch per exchange window H = 1 ms
(5 substeps of 0.2 ms at n64). The sand block (0.4 x 0.1 x 0.4 m, ppc 8, D-P phi 30) starts
at rest directly on the sheet (bottom_gap 0) under constant gravity; no ramp.

--mesh       the published mesh directory (mesh.npz + manifest.json): the package cites both
             SHA-256 and complete = true. A bare mesh.npz without its manifest, or no --mesh
             (the mesh is then rebuilt from the documented formula), gives a development
             package (complete = false, publishable = false; the XPBD consumer must refuse it).
scene.H_ns   the exchange window in ns (default 1 ms); exchange_substeps * dt must equal it.
--replay-root  parent directory of the package f3-n64-<run_id>/ (default: the run directory).

Outputs: the usual run directory (contract v1 + contact columns + cloth/), summary.json with a
`replay` section, summary.png, and the package (batches.npz, manifest.json). The package
format is the XPBD draft xpbd-f3-replay-proposal-v1 (mpm/replay.py). A fixed-surface
recording carries no feedback from a deforming cloth: it is a solver load test on the XPBD
side, not a coupling-stability test.
"""
import datetime
import json
import os
import pathlib
import uuid

import numpy as np

import _cli
from mpm import Solver, replay
from mpm.diagnostics import nanmax
from mpm.export import CONTRACT
from sand_column import sample_box

SCENE_DEFAULTS = dict(
    name="f3_replay",
    block=dict(center_xz=[0.5, 0.5], size=[0.4, 0.1, 0.4], bottom_gap=0.0),
    windows=300,                   # exchange windows in the package (0 .. 0.3 s at H = 1 ms)
    H_ns=1_000_000,                # required exchange window H = exchange_substeps * dt [ns]
    windows_per_frame=10,          # diag row / frame export every this many windows
    seed=0, export_frames=True,
    vmax_hint=1.0,
)


def sheet_plane(sim):
    """World plane height and footprint of the (horizontal) mesh stub."""
    X = sim.contact.surface.X0
    if np.ptp(X[:, 1]) > 1e-12:
        raise ValueError("f3_replay expects a horizontal mesh in the world (+Y normal)")
    return dict(y=float(X[0, 1]), lo=(float(X[:, 0].min()), float(X[:, 2].min())),
                hi=(float(X[:, 0].max()), float(X[:, 2].max())))


def initial_positions(cfg, y_sheet):
    sc, gc = cfg["scene"], cfg["grid"]
    dx = gc["domain"] / gc["n"]
    b = sc["block"]
    (cx, cz), (sx, sy, sz) = b["center_xz"], b["size"]
    y0 = y_sheet + b["bottom_gap"]
    lo, hi = (cx - sx / 2, y0, cz - sz / 2), (cx + sx / 2, y0 + sy, cz + sz / 2)
    return sample_box(lo, hi, dx, cfg["particles"]["ppc"], np.random.default_rng(sc["seed"])), (lo, hi)


def plot(rec, man, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(pathlib.Path(man["dir"]) / replay.NPZ, allow_pickle=False) as z:
        G, Jm, Gp2, Gp36 = z["impulse"], z["J_material"], z["G_on_pinned_two_rows"], z["G_on_pinned_corner36"]
    MgH, H = man["Mg_H"], man["H"]
    t = (np.arange(len(Jm)) + 0.5) * H
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(t, -Jm[:, 2] / MgH, label="-J_material,z / (M g H) (native z = up)")
    ax[0].plot(t, -Gp2[:, 2] / MgH, label="G_on_pinned two rows / (M g H)")
    ax[0].plot(t, -Gp36[:, 2] / MgH, label="G_on_pinned corner 3x3 / (M g H)")
    ax[0].axhline(1.0, color="k", ls=":")
    ax[0].set_xlabel("window centre [s]"); ax[0].legend(fontsize=8); ax[0].set_title("load per window")
    ax[1].plot(t, np.linalg.norm(G, axis=2).max(1), label="max_i |G_i|")
    ax[1].set_xlabel("window centre [s]"); ax[1].set_ylabel("N s"); ax[1].set_title("largest vertex impulse")
    k = int(np.argmax(-Jm[:, 2]))
    Gz = -G[k, :, 2].reshape(52, 52)
    im = ax[2].imshow(Gz, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax[2], label="-G_z [N s]")
    ax[2].set_title(f"window {k}: -G_z per vertex (row 0 at top)")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = _cli.parser("f3_replay_n64", __doc__.splitlines()[0])
    ap.add_argument("--mesh", help="published mesh.npz or its directory (omit: formula rebuild, not publishable)")
    ap.add_argument("--mesh-sha256", help="expected SHA-256 of mesh.npz (checked before use)")
    ap.add_argument("--mesh-manifest-sha256", help="expected SHA-256 of the mesh manifest.json (checked before use)")
    ap.add_argument("--replay-root", help="parent of the package directory (default: the run directory)")
    ap.add_argument("--run-id", help="package run_id (default: UTC time + random suffix)")
    args = ap.parse_args()
    cfg, rt = _cli.start(args, SCENE_DEFAULTS)
    surf = cfg["contact"]["surface"]
    if surf.get("type") != "mesh":
        raise SystemExit("f3_replay needs contact.surface.type = mesh")
    if args.mesh:
        surf["path"] = str(pathlib.Path(args.mesh).resolve())
    if args.mesh_sha256:
        surf["expect_sha256"] = args.mesh_sha256
    if args.mesh_manifest_sha256:
        surf["expect_manifest_sha256"] = args.mesh_manifest_sha256
    if (surf.get("motion") or {}).get("type", "fixed") != "fixed":
        raise SystemExit("f3_replay records a fixed stub only (contact.surface.motion.type = fixed)")
    sc = cfg["scene"]
    W, wpf = int(sc["windows"]), int(sc["windows_per_frame"])
    if W < 1 or wpf < 1 or W % wpf:
        raise SystemExit("scene.windows must be a positive multiple of scene.windows_per_frame")

    from mpm.contact import xpbd_mesh            # the sheet height is the transform's C_y
    tf = dict(xpbd_mesh.proposed_transform(), **(surf.get("transform") or {}))
    pos, box = initial_positions(cfg, float(tf["C"][1]))
    sim, build_s = _cli.timed(Solver, cfg, pos)
    c = sim.contact
    sh = sheet_plane(sim)
    if abs(sh["y"] - float(tf["C"][1])) > 1e-15:
        raise SystemExit("sheet height differs from the transform")
    H_ns = c.window_ns()
    if H_ns is None or H_ns != int(sc["H_ns"]):
        raise SystemExit(f"exchange window must be H = scene.H_ns = {sc['H_ns']} ns exactly, got {H_ns} ns "
                         f"({c.n_sub} x dt {sim.dt_host})")
    H = H_ns / 1e9
    rec = replay.ReplayRecorder(c, W)

    writer = _cli.open_run(args, cfg)
    warns = sim.report(vmax_hint=sc["vmax_hint"])
    diag, d0, timing = _cli.prepare(sim, writer, cfg, rt, args, warns, build_s)
    fps = 1.0 / (wpf * H)
    rows, status, loop = _cli.run_frames(sim, writer, diag, W * H, fps)
    timing.update(loop)
    post_start = _cli.time.perf_counter()
    if status == "ok" and not rec.complete:
        status = f"stopped: {rec.k} of {W} windows recorded"

    M = float(d0["mass"])
    g = float(np.linalg.norm(sim.gravity[None].to_numpy()))
    last = rows[-1]
    run_id = args.run_id or (datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "-" + uuid.uuid4().hex[:6])
    work = pathlib.Path(__file__).resolve().parents[1]
    config_path = writer.dir / "config.json"
    derived = sim.derived()
    producer = dict(
        project="MPM_Couple", unit="P1-U5", scene="scenes/f3_replay.py", run_dir=str(writer.dir),
        runtime=dict(replay.runtime_record(), actual_arch=rt["actual_arch"], requested_arch=args.arch,
                     cpu_threads=rt["cpu_threads"], cpu_threads_meaning="cap on Taichi CPU threads; 0 or null = no cap",
                     logical_cpus=os.cpu_count(), precision=cfg["precision"]),
        source_sha256=replay.source_hashes(work, ("scenes/f3_replay.py", "scenes/_cli.py", "scenes/sand_column.py")),
        config_sha256=replay.sha256(config_path), config_file="config.json in the run directory",
        grid=dict(n=cfg["grid"]["n"], dx=derived["dx"], domain=cfg["grid"]["domain"]),
        particles=dict(count=int(sim.n), ppc=derived["ppc"], total_mass_kg=M, particle_mass_kg=derived["p_mass_per_material"][0],
                       block_box_m=dict(lo=list(box[0]), hi=list(box[1])),
                       sampling=f"jittered grid, ppc {cfg['particles']['ppc']}, seed {sc['seed']}",
                       note="the target scene mass of about 35.9 kg is the n48 discretisation of the same block; "
                            "the actual mass of this run is total_mass_kg"),
        material=cfg["constitutive"], gravity=cfg["gravity"], boundaries=cfg["boundaries"],
        time=dict(dt=sim.dt_host, substeps_per_window=c.n_sub, H_ns=H_ns, substeps_total=int(sim.step[None]),
                  every_window=f"{c.n_sub} substeps of dt (constant)"),
        contact=dict(params=c.params(), correction=c.correction, side=c.side_mode, exchange_substeps=c.n_sub,
                     pinned_mask_in_stub=surf["pinned"], surface=c.surface.describe()),
        Mg_H=M * g * H)
    load_history = dict(
        initial="sand block at rest directly on the sheet plane (bottom_gap 0; the lowest samples lie a quarter "
                "of the sample spacing or more above it), zero velocity and stress",
        loading="constant gravity (0, -9.81, 0) m/s^2 in the exchange world from t = 0 (native -Z); no ramp",
        window_range=f"[0, {W} H) with H = {H_ns} ns", cloth="rigid fixed stub; no feedback from cloth deformation")
    out = dict(status=status)
    if status == "ok":
        arrays, manifest = replay.package(rec, c.surface, run_id=run_id, producer=producer, load_history=load_history)
        root = pathlib.Path(args.replay_root).resolve() if args.replay_root else writer.dir
        out = replay.write(root / f"f3-n64-{run_id}", arrays, manifest)
        out.update(status="ok", Mg_H=producer["Mg_H"], H=H, summary=manifest["summary"])
    guard = sim.guard.readout()
    with_grid = rows[1:]
    summary = dict(status=status, **{k: timing[k] for k in ("wall_s", "first_frame_s", "substeps", "steps_per_frame",
                                                          "steady_ms_per_substep")}, n=sim.n, contract=CONTRACT,
                   actual_arch=rt["actual_arch"], cpu_threads=rt["cpu_threads"], timing=timing,
                   mass=M, mass_drift_rel=(last["mass"] - d0["mass"]) / d0["mass"],
                   contact=dict(params=c.params(), exchange_substeps=c.n_sub, correction=c.correction,
                                windows=last["contact_epoch"], penetrated_max_run=last["contact_penetrated_max_run"],
                                pen_max_run=last["contact_pen_max_run"], missed=last["contact_missed"],
                                overflow=last["contact_overflow"], JG_split_rel_final=last["contact_JG_split_rel"],
                                surface=c.surface.describe()),
                   balance=dict(bal_rel_max=nanmax(r["bal_rel"] for r in with_grid),
                                bal_abs_max=[nanmax(abs(r[f"bal_p{cc}"]) for r in rows) for cc in "xyz"],
                                clock=sim.clock()),
                   cfl_max=max(r["cfl"] for r in rows), nonfinite=guard, replay=out)
    _cli.close_timing(timing, post_start)
    writer.write_json("summary.json", summary)
    _cli.finish_profiles(cfg, sim, writer)
    print(json.dumps({k: summary[k] for k in ("status", "wall_s", "n", "mass", "balance", "replay")}, indent=2,
                     default=str))
    if status == "ok":
        plot(rec, out, writer.dir / "summary.png")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
