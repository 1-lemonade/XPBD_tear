"""Sand on a live cloth surface driven through the MPM endpoint by prescribed snapshots
(P2-U2; MPM process only, no XPBD process, not a coupling run).

    python scenes/live_endpoint.py [--config scenes/configs/live_endpoint_f3.json] [--mesh DIR|mesh.npz]
        [--mesh-sha256 HEX] [--mesh-manifest-sha256 HEX] [--set ...] [--arch cpu|cuda] [--out DIR]

The cloth is the XPBD 52 x 52 reference mesh (mpm/contact/xpbd_mesh.py; --mesh: the published
file, else the documented formula) in its native frame, offered every window by
coupling.PrescribedPeer (XPBD KinematicSurfaceBridge semantics: whole surface pinned, one
constant velocity for the stream, next state = positions + h v) and read by
coupling.MPMEndpoint into a live surface (contact.surface.type = live, proposed transform). The
endpoint registers the real MPM source; the peer expects exactly that source.

scene.sheet_velocity   world velocity of the sheet [m/s] (0: fixed); native = R^T v
scene.block            sand block (as f3_replay: bottom_gap above the sheet plane y = C_y)
scene.windows, H_ns    windows and window length (H_ns must equal exchange_substeps * dt)
scene.windows_per_frame  diag row / frame export / balance check every this many windows
scene.stop             overrides of the pre-registered stop rules (coupling.STOP_DEFAULTS)
scene.shuffle_rows     null, or a seed: the peer permutes rows every window (same topology)
scene.stream_id        null: live-endpoint-<run id>

Outputs: the usual run directory (export contract v1; live cloth frames carry their rows' IDs and
triangles), summary.json (status, per-window load ratio -sum G_y / (M g H), contact onset,
stop state, provenance summary, archive check) and endpoint/ (coupling.archive: batches,
first snapshot, ledger, provenance, manifest anchored to this run directory). Exit code 0 when
every window was delivered, 4 on an endpoint stop, 1 on another early stop.
"""
import json
import pathlib

import numpy as np

import _cli
from mpm import Solver
from mpm.contact import xpbd_mesh
from mpm.contact.live import resolve_transform
from mpm.diagnostics import nanmax
from mpm.export import CONTRACT
import f3_replay
import sand_column
from f3_replay import initial_positions
from coupling import EndpointStop, MPMEndpoint, PrescribedPeer, archive

SCENE_DEFAULTS = dict(
    name="live_endpoint",
    block=dict(center_xz=[0.5, 0.5], size=[0.4, 0.1, 0.4], bottom_gap=0.0),
    windows=300, H_ns=1_000_000, windows_per_frame=10, seed=0, export_frames=True, vmax_hint=1.0,
    sheet_velocity=[0.0, 0.0, 0.0], stop={}, shuffle_rows=None, stream_id=None,
)


def main():
    ap = _cli.parser("live_endpoint_f3", __doc__.splitlines()[0])
    ap.add_argument("--mesh", help="published mesh.npz or its directory (omit: formula rebuild)")
    ap.add_argument("--mesh-sha256")
    ap.add_argument("--mesh-manifest-sha256")
    args = ap.parse_args()
    cfg, rt = _cli.start(args, SCENE_DEFAULTS)
    sc, surf = cfg["scene"], cfg["contact"]["surface"]
    if surf.get("type") != "live":
        raise SystemExit("live_endpoint needs contact.surface.type = live")
    W, wpf = int(sc["windows"]), int(sc["windows_per_frame"])
    if W < 1 or wpf < 1:
        raise SystemExit("scene.windows and scene.windows_per_frame must be positive")
    src = (xpbd_mesh.load(args.mesh, args.mesh_sha256, args.mesh_manifest_sha256) if args.mesh
           else xpbd_mesh.reconstructed())
    a = src["arrays"]
    tf = resolve_transform(surf.get("transform"))
    rigid = xpbd_mesh.RigidMap(tf["R"], tf["c"], tf["C"])
    y_sheet = float(rigid.to_world(a["positions"][:1])[0, 1])
    pos, box = initial_positions(cfg, y_sheet)
    sim, build_s = _cli.timed(Solver, cfg, pos)
    ep = MPMEndpoint(sim, scene=sc["name"], cfg=cfg, stop=sc["stop"],
                     scene_files=[__file__, _cli.__file__, f3_replay.__file__, sand_column.__file__])
    if ep.H_ns != int(sc["H_ns"]):
        raise SystemExit(f"exchange window must be scene.H_ns = {sc['H_ns']} ns, got {ep.H_ns} ns")
    v_native = rigid.vec_to_native(np.asarray(sc["sheet_velocity"], np.float64))
    stream = sc["stream_id"] or f"live-endpoint-{ep.run_id[:12]}"
    peer = PrescribedPeer(a["positions"], a["triangles"], a["vertex_ids"], a["triangle_ids"], v_native, h=ep.H,
                          stream_id=stream, expected_source=ep.source, shuffle_rows=sc["shuffle_rows"])
    status = "ok"
    try:
        ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))      # the first snapshot precedes the export
    except EndpointStop as e:
        raise SystemExit(f"first offer rejected: {e}")
    writer = _cli.open_run(args, cfg)
    warns = sim.report(vmax_hint=sc["vmax_hint"])
    diag, d0, timing = _cli.prepare(sim, writer, cfg, rt, args, warns, build_s)
    t0 = _cli.time.perf_counter()

    def frame_row(index):
        a_ = _cli.time.perf_counter()
        d = diag()
        b_ = _cli.time.perf_counter()
        writer.frame(sim, index, d["t"])
        c_ = _cli.time.perf_counter()
        d.update(time_sim_s=0.0, time_diag_s=b_ - a_, time_io_s=c_ - b_)
        writer.diag_row(d)
        return d

    rows = [frame_row(0)]
    for k in range(W):
        try:
            msg = ep.advance()
            try:
                receipt = peer.consume(msg)
            except ValueError as e:                              # the peer refused the batch: both sides stop
                ep.reject(str(e))
            ep.acknowledge(receipt)
            if (k + 1) % wpf == 0:
                rows.append(frame_row((k + 1) // wpf))
                ep.check_frame(rows[-1])
                if rows[-1].get("contact_overflow", 0) > 0:
                    ep.halt("S2 table overflow at a frame (scene check)")
            if k + 1 < W:
                ep.receive(peer.open_window(h=ep.H, particle_dt=ep.dt))
        except EndpointStop as e:
            status = f"stopped: {e.reason}"
            break
    writer.close()
    timing.update(wall_s=_cli.time.perf_counter() - t0, windows_delivered=ep.k, substeps=int(sim.step[None]))
    post_start = _cli.time.perf_counter()

    arch = archive.write(writer.dir / "endpoint", ep, run_dir=writer.dir)
    check = archive.verify(writer.dir / "endpoint", work=pathlib.Path(__file__).resolve().parents[1])
    M = float(d0["mass"])
    g = float(np.linalg.norm(sim.gravity[None].to_numpy()))
    MgH = M * g * ep.H
    led = [r for r in ep.ledger if "sumG_y" in r]               # a window stopped before its substeps has no sums
    load = [-r["sumG_y"] / MgH for r in led]                      # world y = native z
    onset = next((r["window"] for r in led if r["sumG_x"] or r["sumG_y"] or r["sumG_z"]), None)
    vel = [[r["momentum"][i] / r["mass"] for i in range(3)] for r in rows]
    summary = dict(
        status=status, windows=W, windows_delivered=ep.k, n=sim.n, mass=M, Mg_H=MgH, contract=CONTRACT,
        actual_arch=rt["actual_arch"], cpu_threads=rt["cpu_threads"], timing=timing,
        sheet_velocity_world=sc["sheet_velocity"], sheet_velocity_native=v_native.tolist(),
        mesh=dict(binding=src["binding"], sha256=src["sha256"], manifest_sha256=src["manifest_sha256"]),
        load_ratio=load, contact_onset_window=onset, frame_mean_velocity=vel,
        stop=ep.stopped, warnings=ep.warnings, provenance=dict(source=ep.source.as_dict(),
                                                              code_digest=ep.registration["code_digest"],
                                                              mpm_run_id=ep.run_id, stream_id=ep.stream_id,
                                                              chain_final=ep.chain),
        receipts=len(peer.receipts), archive=dict(arch, verify=check),
        balance=dict(bal_rel_max=nanmax(r["bal_rel"] for r in rows[1:]) if len(rows) > 1 else None),
        stop_rules=ep.registration["stop"])
    _cli.close_timing(timing, post_start)
    writer.write_json("summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "windows_delivered", "mass", "contact_onset_window",
                                              "balance", "provenance")}, indent=2, default=str))
    if status == "ok":
        return 0
    return 4 if status.startswith("stopped: S") else 1


if __name__ == "__main__":
    raise SystemExit(main())
