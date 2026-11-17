"""Sand on the XPBD registered kinematic bridge served by a separate XPBD process (P2-U3; MPM driver + XPBD service, two local processes; not the U4 host joint run).

    python -B scenes/coupled_u13.py [--config scenes/configs/coupled_u13.json] [--set ...] [--threads N] [--out DIR]

The XPBD service (coupling.xpbd_service, spawned) imports only XPBD's public entry
create_registered_kinematic_adapter and the exchange contract classes from scene.xpbd.root
(null: $MPM_XPBD_WORK, else the enclosing XPBD candidate; files pinned by SHA-256,
scene.xpbd.expected_sha256 null = coupling.driver.PINNED_XPBD), builds the all-pinned snapshot
from the published mesh bound by both digests (scene.xpbd.snapshot) moving with
scene.sheet_velocity (world; native = R^T v), and serves OPEN / PEER / IMPULSE. The MPM endpoint
(handshake=True) reads every offer, advances the window, delivers the batch and acknowledges
XPBD's receipt (coupling.driver.CoupledDriver); any failure stops both sides, nothing is retried.

scene.xpbd.snapshot   mesh ("file"), mesh_path / mesh_sha256 / manifest_sha256 (null: the published
                      cloth52-v1-20261109 and its digests), epoch, topology_version , previous_dt (null: h)
scene.transport       timeouts (s: hello register open peer impulse stop idle), max_frame_bytes
scene.stream_id       null: a fresh u3-<uuid4> per run

The top level imports only the standard library and coupling.link / coupling.wire (NumPy):
multiprocessing spawn re-runs this file's top level in the service process (as __mp_main__),
which must not load Taichi or mpm (the service refuses to start if it does).

Outputs: the run directory (export contract v1, live cloth frames), endpoint/ (coupling.archive),
coupling/coupled.json (driver record: both stops, per-window timing), coupling/xpbd_service/
(hello.json, bound.json, receipts.jsonl, final.json), summary.json (status, stops, reconcile,
archive check). Exit code 0 when every window was acknowledged, 4 on a joint stop, 1 otherwise.
"""
import pathlib
import sys

WORK = pathlib.Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
from coupling.link import DEFAULT_TIMEOUTS  # noqa: E402  (standard library + NumPy, no Taichi, no mpm)
from coupling.wire import MAX_FRAME  # noqa: E402

SCENE_DEFAULTS = dict(
    name="coupled_u13",
    block=dict(center_xz=[0.5, 0.5], size=[0.4, 0.1, 0.4], bottom_gap=0.0),
    windows=30, H_ns=1_000_000, windows_per_frame=10, seed=0, export_frames=True, vmax_hint=1.0,
    sheet_velocity=[0.0, 0.0, 0.0], stop={}, stream_id=None,
    xpbd=dict(root=None, expected_sha256=None,
              snapshot=dict(mesh="file", mesh_path=None, mesh_sha256=None, manifest_sha256=None, epoch=1,
                            topology_version=None, previous_dt=None)),
    transport=dict(timeouts=dict(DEFAULT_TIMEOUTS), max_frame_bytes=MAX_FRAME),
)


def main():
    import json
    import time

    import numpy as np

    import _cli
    import f3_replay
    import sand_column
    from mpm import Solver
    from mpm.contact import xpbd_mesh
    from mpm.contact.live import resolve_transform
    from mpm.diagnostics import nanmax
    from mpm.export import CONTRACT
    from coupling import MPMEndpoint, archive
    from coupling.driver import PUBLISHED_MESH, CoupledDriver, reconcile, snapshot_spec

    ap = _cli.parser("coupled_u13", __doc__.splitlines()[0])
    args = ap.parse_args()
    cfg, rt = _cli.start(args, SCENE_DEFAULTS)
    sc, surf = cfg["scene"], cfg["contact"]["surface"]
    if surf.get("type") != "live":
        raise SystemExit("coupled_u13 needs contact.surface.type = live")
    W, wpf = int(sc["windows"]), int(sc["windows_per_frame"])
    if W < 1 or wpf < 1:
        raise SystemExit("scene.windows and scene.windows_per_frame must be positive")
    xs = sc["xpbd"]["snapshot"]
    if xs["mesh"] != "file":
        raise SystemExit("coupled_u13 binds the published mesh file (scene.xpbd.snapshot.mesh = file)")
    mesh_path = xs["mesh_path"] or str(PUBLISHED_MESH["path"])
    sha, msha = xs["mesh_sha256"] or PUBLISHED_MESH["mesh_sha256"], xs["manifest_sha256"] or PUBLISHED_MESH["manifest_sha256"]
    src = xpbd_mesh.load(mesh_path, sha, msha)                 # MPM's own copy: sand placement and the expectation
    a = src["arrays"]
    tf = resolve_transform(surf.get("transform"))
    rigid = xpbd_mesh.RigidMap(tf["R"], tf["c"], tf["C"])
    y_sheet = float(rigid.to_world(a["positions"][:1])[0, 1])
    pos, box = f3_replay.initial_positions(cfg, y_sheet)
    sim, build_s = _cli.timed(Solver, cfg, pos)
    ep = MPMEndpoint(sim, scene=sc["name"], cfg=cfg, stop=sc["stop"], handshake=True,
                     scene_files=[__file__, _cli.__file__, f3_replay.__file__, sand_column.__file__])
    if ep.H_ns != int(sc["H_ns"]):
        raise SystemExit(f"exchange window must be scene.H_ns = {sc['H_ns']} ns, got {ep.H_ns} ns")
    spec = snapshot_spec(ep, velocity_world=sc["sheet_velocity"], mesh="file", mesh_path=mesh_path, mesh_sha256=sha,
                         manifest_sha256=msha, epoch=xs["epoch"], topology_version=xs["topology_version"],
                         previous_dt=xs["previous_dt"])
    writer = _cli.open_run(args, cfg)
    st = dict(rows=[])

    def frame_row(index):
        a_ = time.perf_counter()
        d = st["diag"]()
        b_ = time.perf_counter()
        writer.frame(sim, index, d["t"])
        c_ = time.perf_counter()
        d.update(time_sim_s=0.0, time_diag_s=b_ - a_, time_io_s=c_ - b_)
        writer.diag_row(d)
        return d

    def on_received(k):
        if k == 0:                                             # the first snapshot precedes the export
            warns = sim.report(vmax_hint=sc["vmax_hint"])
            st["diag"], st["d0"], st["timing"] = _cli.prepare(sim, writer, cfg, rt, args, warns, build_s)
            st["rows"].append(frame_row(0))
            st["t0"] = time.perf_counter()

    def on_window(k):
        if (k + 1) % wpf == 0:
            st["rows"].append(frame_row((k + 1) // wpf))
            ep.check_frame(st["rows"][-1])
            if st["rows"][-1].get("contact_overflow", 0) > 0:
                ep.halt("S2 table overflow at a frame (scene check)")

    drv = CoupledDriver(ep, snapshot=spec, xpbd_root=sc["xpbd"]["root"], expected_sha256=sc["xpbd"]["expected_sha256"],
                        log_dir=writer.dir / "coupling", timeouts=sc["transport"]["timeouts"],
                        max_frame_bytes=sc["transport"]["max_frame_bytes"], stream_id=sc["stream_id"], expect_mesh=a)
    result = drv.run(W, on_received=on_received, on_window=on_window)
    writer.close()
    post_start = time.perf_counter()
    timing = st.get("timing", dict(build_s=build_s))
    timing.update(wall_s=time.perf_counter() - st.get("t0", post_start), windows_acknowledged=ep.acknowledged(),
                  substeps=int(sim.step[None]))
    arch = check = rec = None
    if ep.snapshot0 is not None:                               # a handshake failure before the first read: no archive
        arch = archive.write(writer.dir / "endpoint", ep, run_dir=writer.dir)
        check = archive.verify(writer.dir / "endpoint", work=WORK)
        rec = reconcile(writer.dir / "endpoint", writer.dir / "coupling" / "xpbd_service")
    led = [r for r in ep.ledger if "sumG_y" in r]
    MgH = None
    if "d0" in st:
        MgH = float(st["d0"]["mass"]) * float(np.linalg.norm(sim.gravity[None].to_numpy())) * ep.H
    rows = st["rows"]
    summary = dict(
        status=result["status"], windows=W, windows_acknowledged=ep.acknowledged(), n=sim.n, contract=CONTRACT,
        actual_arch=rt["actual_arch"], cpu_threads=rt["cpu_threads"], timing=timing,
        sheet_velocity_world=sc["sheet_velocity"], snapshot_spec=spec,
        mesh=dict(binding=src["binding"], sha256=src["sha256"], manifest_sha256=src["manifest_sha256"]),
        load_ratio=None if MgH is None else [-r["sumG_y"] / MgH for r in led],
        contact_onset_window=next((r["window"] for r in led if r["sumG_x"] or r["sumG_y"] or r["sumG_z"]), None),
        frame_mean_velocity=[[r["momentum"][i] / r["mass"] for i in range(3)] for r in rows],
        mpm_stop=ep.stopped, xpbd_stop=result["xpbd_stop"], awaiting_receipt=ep.awaiting_receipt,
        stream_id=result["stream_id"], registration_digest=ep.registration_digest,
        xpbd_hello=dict((result["hello"] or {}), xpbd_files=(result["hello"] or {}).get("xpbd_files")),
        provenance=dict(source=ep.source.as_dict(), code_digest=ep.registration["code_digest"], mpm_run_id=ep.run_id,
                        chain_final=ep.chain),
        archive=None if arch is None else dict(arch, verify=check), reconcile=rec,
        balance=dict(bal_rel_max=nanmax(r["bal_rel"] for r in rows[1:]) if len(rows) > 1 else None),
        window_wall_s=dict(median=float(np.median([w["window_s"] for w in result["windows"]])) if result["windows"]
                           else None))
    _cli.close_timing(timing, post_start)
    writer.write_json("summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "windows_acknowledged", "stream_id", "reconcile")}, indent=2,
                     default=str))
    if result["status"] == "ok":
        return 0
    return 4 if result["status"].startswith("stopped: S") else 1


if __name__ == "__main__":
    raise SystemExit(main())
