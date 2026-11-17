"""P2-U4 host package: judge one joint case against the locked run baseline (tools/u4_cases.json).

    python -B tools/u4_check.py --run <case run dir> --case <name> --cases tools/u4_cases.json --out <check.json>
                                --scene-exit <code> [--mpm-pid PID] [--xpbd-root DIR] [--rehearsal]

Reads the scene's run directory (summary.json, coupling/coupled.json, coupling/xpbd_service/, endpoint/), then:
reconcile row by row; the replay into the loop on this machine (path A: the archive's sealed inputs through the
transport into a fresh XPBD service process; path B: the same inputs consumed by the same XPBD adapter in this
process, keeping every snapshot); the common gates and the case gates of the baseline. Writes check.json (gates,
per-window series, timing) and exits 0 when every gate passed, 2 when a gate failed, 1 when the check itself broke.
--rehearsal (the package rehearsal with shortened runs only): the case gates are listed, not evaluated.

The top level imports only the standard library: the transport replay spawns the service process, which re-runs
this file's top level (multiprocessing spawn) and must not load Taichi or mpm.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import traceback

WORK = pathlib.Path(__file__).resolve().parents[1]

RECEIPT_BOUNDARY = dict(kinematic=True, free_dof_count=0, load_received=True, load_applied_to_geometry=False,
                        support_balance=None, drive_work=None, reaction_measurement="not_measured")
REFERENCE_R = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def matches_public_transform(tf):
    """Check the fixed example transform, not a personal approval or signature."""
    if not isinstance(tf, dict) or set(tf) != {"R", "c", "C", "status"}:
        return False
    try:
        numbers = [x for row in tf["R"] for x in row] + list(tf["c"]) + list(tf["C"])
    except (TypeError, KeyError):
        return False
    return (all(type(x) in (int, float) and math.isfinite(x) for x in numbers)
            and tf["R"] == REFERENCE_R and tf["c"] == [0.4, 0.4, 0.0]
            and tf["C"] == [0.5, 0.3, 0.5] and tf["status"] == "fixed-transform-v1")


def _gate(results, name, fn):
    try:
        ok, detail = fn()
        results[name] = dict(passed=bool(ok), **detail)
    except Exception as e:  # noqa: BLE001  (a gate that cannot be evaluated has failed)
        results[name] = dict(passed=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-exit", type=int, required=True)
    ap.add_argument("--mpm-pid", type=int)
    ap.add_argument("--xpbd-root")
    ap.add_argument("--rehearsal", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(WORK))
    import numpy as np
    from coupling import archive
    from coupling.driver import (PINNED_XPBD, _inputs_from_archive, first_mismatch, reconcile, run_offline,
                                 run_transport)

    run = pathlib.Path(args.run)
    base = json.loads(pathlib.Path(args.cases).read_text(encoding="utf-8"))
    case = next(c for c in base["cases"] if c["name"] == args.case)
    W = int(case["windows"])
    v_world = np.asarray(case["velocity_world"], np.float64)
    import hashlib
    out = dict(case=case["name"], baseline=base["baseline"], run_dir=str(run), rehearsal=bool(args.rehearsal),
               judged_with={n: hashlib.sha256((WORK / "tools" / n).read_bytes()).hexdigest()
                            for n in ("u4_check.py", "u4_cases.json")},
               scene_exit=args.scene_exit, gates={}, case_gates={}, series={}, timing={}, records={})
    G = out["gates"]

    def load(p):
        return json.loads((run / p).read_text(encoding="utf-8"))
    summary = load("summary.json") if (run / "summary.json").exists() else None
    coupled = load("coupling/coupled.json") if (run / "coupling" / "coupled.json").exists() else None
    if summary is not None:
        W = int(summary["windows"])                            # a rehearsal shortens the run (checked below)
    out["windows"] = W
    arch_dir, svc_dir = run / "endpoint", run / "coupling" / "xpbd_service"
    a = archive.read(arch_dir) if (arch_dir / "manifest.json").exists() else None
    man = None if a is None else a["manifest"]
    ledger = [] if a is None else a["ledger"]

    def f(row, key):
        return float(row[key]) if row.get(key) not in (None, "") else math.nan

    # ------------------------------------------------------------------ common gates
    _gate(G, "run_status", lambda: (
        args.scene_exit == 0 and summary["status"] == "ok" and summary["windows_acknowledged"] == W
        and (args.rehearsal or W == int(case["windows"])),
        dict(status=summary["status"], windows=W, requested=int(case["windows"]),
             acknowledged=summary["windows_acknowledged"], scene_exit=args.scene_exit)))
    xs = (coupled or {}).get("xpbd_stop") or {}
    _gate(G, "xpbd_clean_stop", lambda: (
        xs.get("bye") is not None and xs.get("exitcode") == 0 and not xs.get("terminated") and not xs.get("late_frames"),
        dict(exitcode=xs.get("exitcode"), terminated=xs.get("terminated"), late_frames=len(xs.get("late_frames") or []),
             bye_reason=(xs.get("bye") or {}).get("reason"))))
    rc = {}

    def g_reconcile():
        rc.update(reconcile(arch_dir, svc_dir))
        return (rc["consistent"] and rc["extra_xpbd_windows"] == 0 and rc["log_condition"] == "final"
                and rc["xpbd_committed"] == W and str(rc["xpbd_final_reason"]).startswith("STOP"), dict(rc))
    _gate(G, "reconcile", g_reconcile)

    def g_archive():
        v = archive.verify(arch_dir, work=WORK)
        out["records"]["archive_verify"] = v
        return (v["complete"] and v["acknowledged"] == W and v.get("code_unchanged") is True
                and v.get("anchor_ok") is True, dict(complete=v["complete"], acknowledged=v["acknowledged"],
                                                     code_unchanged=v.get("code_unchanged"), anchor_ok=v.get("anchor_ok"),
                                                     stop_reason=v.get("stop_reason")))
    _gate(G, "archive", g_archive)

    inp = A = B = None
    try:
        inp = _inputs_from_archive(arch_dir)
    except Exception as e:  # noqa: BLE001
        out["records"]["replay_inputs_error"] = f"{type(e).__name__}: {e}"
    timeouts = (coupled or {}).get("transport", {}).get("timeouts")

    def g_transport():
        nonlocal A
        A = run_transport(inp, xpbd_root=args.xpbd_root, log_dir=run.parent.parent / "replay" / case["name"] / "transport",
                          timeouts=timeouts)
        offers = [r["offer_digest"] for r in coupled["windows"]]
        st = A["stop"]
        ok = (A["registration_digest"] == summary["registration_digest"] and A["offer_digests"] == offers
              and A["receipt_digests"] == man["receipt_digests"][:W] and len(A["receipt_digests"]) == W
              and A["snapshot0_equal"] is True and st.get("bye") is not None and st.get("exitcode") == 0)
        return ok, dict(windows=len(A["receipt_digests"]), registration_equal=A["registration_digest"] == summary["registration_digest"],
                        offers_equal=A["offer_digests"] == offers,
                        receipts_equal=A["receipt_digests"] == man["receipt_digests"][:W],
                        snapshot0_equal=A["snapshot0_equal"], service_exit=st.get("exitcode"),
                        environment=A.get("environment"))
    _gate(G, "replay_transport", g_transport)

    def g_offline():
        nonlocal B
        B = run_offline(inp, xpbd_root=args.xpbd_root, keep_snapshots=True)
        mism = first_mismatch(A, B) if A is not None else "path A missing"
        ok = (B["receipt_digests"] == man["receipt_digests"][:W] and A is not None
              and B["receipt_canonical"] == A["receipt_canonical"] and mism is None and len(B["committed"]) == W
              and B["registration_digest"] == summary["registration_digest"])
        return ok, dict(receipts_equal_run=B["receipt_digests"] == man["receipt_digests"][:W],
                        canonical_equal_transport=A is not None and B["receipt_canonical"] == A["receipt_canonical"],
                        committed_snapshots=len(B["committed"]), first_mismatch=mism)
    _gate(G, "replay_offline", g_offline)

    def g_provenance():
        hello = coupled["hello"]
        files = hello.get("xpbd_files") or {}
        core = ("project/coupling/registered_kinematic.py", "project/coupling/contracts.py",
                "project/coupling/loads.py", "project/coupling/exchange.py")
        # every pinned file on disk (inputs.json of the package run; kinematic_surface.py is pinned, not imported)
        # and every pinned file the service imported, the four core ones among them 
        inputs_path = run.parent.parent / "inputs.json"
        disk = json.loads(inputs_path.read_text(encoding="utf-8")).get("xpbd_files") if inputs_path.exists() else None
        pins_ok = (disk == PINNED_XPBD and all(k in files for k in core)
                   and all(files[k] == v for k, v in PINNED_XPBD.items() if k in files))
        rt = a["provenance"]["registration"]["runtime"]
        src = a["provenance"]["registration"]["source"]
        regs = dict(endpoint=summary["registration_digest"], bound=(coupled.get("bound") or {}).get("registration_digest"),
                    transport=None if A is None else A["registration_digest"],
                    offline=None if B is None else B["registration_digest"])
        checks = dict(pins=pins_ok, no_forbidden=hello.get("forbidden_modules_at_start") == []
                      and hello.get("forbidden_modules_after_import") == [], no_fault=hello.get("fault") == {},
                      no_bytecode=bool(hello.get("dont_write_bytecode")),
                      same_python_numpy=(hello.get("python"), hello.get("numpy")) == (rt["python"], rt["numpy"]),
                      two_processes=args.mpm_pid is None or hello.get("pid") != args.mpm_pid,
                      real_backend=src.get("kind") == "real_backend",
                      stream_u4=str(summary["stream_id"]).startswith("u4-"),
                      registration_equal=len(set(regs.values())) == 1)
        return all(checks.values()), dict(checks=checks, pinned_on_disk=disk, service_pid=hello.get("pid"), mpm_pid=args.mpm_pid,
                                          xpbd_root=hello.get("xpbd_root"), xpbd_files=files, registration=regs,
                                          source=src, stream_id=summary["stream_id"], platform=hello.get("platform"))
    _gate(G, "provenance", g_provenance)

    def g_boundary():
        motion = "fixed" if not v_world.any() else "uniform_constant_translation"
        bad = []
        for k, text in enumerate(B["receipt_canonical"]):
            r = json.loads(text)
            want = dict(RECEIPT_BOUNDARY, motion=motion)
            diff = {key: r.get(key, "(missing)") for key, val in want.items()
                    if key not in r or r[key] != val or type(r[key]) is not type(val)}
            topo = summary["snapshot_spec"]["topology_version"]
            for key in ("start_version", "committed_version"):
                v = r.get(key)
                if not (type(v) is list and len(v) == 3 and type(v[2]) is int and v[2] == topo):
                    diff[key] = v
            if diff:
                bad.append(dict(window=k, differs=diff))
        first = json.loads(B["receipt_canonical"][0])
        flags = {key: first.get(key) for key in ("source_authenticity", "authenticated", "external_process_observed",
                                                  "external_files_verified", "config_verified", "real_coupling",
                                                  "device_solve", "candidate_only")}
        return not bad and len(B["receipt_canonical"]) == W, dict(motion=motion, windows_bad=bad[:5],
                                                                   recorded_flags_window_0=flags)
    _gate(G, "u13_boundary", g_boundary)

    def g_motion():
        S = B["snapshots"]
        tf_cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))["contact"]["surface"]["transform"]
        R = np.asarray(tf_cfg["R"], np.float64)                    # the run's own transform 
        tf = summary["snapshot_spec"]
        v_native = np.asarray(tf["velocity_native"], np.float64)
        h = float(inp["h"])
        x0 = S[0]["positions"]
        dev = dt_bad = 0.0
        problems = []
        for k, s in enumerate(S):
            if s["epoch"] != S[0]["epoch"] or s["topology_version"] != S[0]["topology_version"] or s["step_index"] != k:
                problems.append(dict(window=k, version=[s["epoch"], s["step_index"], s["topology_version"]]))
            if not (s["velocity"] == v_native).all():
                problems.append(dict(window=k, velocity="not R^T v_world on every vertex"))
            dev = max(dev, float(np.abs(s["positions"] - x0 - k * h * v_native).max()))
            dt_bad = max(dt_bad, abs(s["time"] - k * h))
            if s["dt"] != h:
                problems.append(dict(window=k, dt=s["dt"]))
        world_ok = bool((R @ v_native == v_world).all())
        initial_ok = S[0]["topology_version"] == tf["topology_version"] and S[0]["step_index"] == 0
        reference_matches = matches_public_transform(tf_cfg)
        ok = (not problems and dev <= 1e-12 and dt_bad <= 1e-12 and world_ok and len(S) == W + 1 and initial_ok
              and reference_matches)
        return ok, dict(snapshots=len(S), max_position_deviation_m=dev, max_time_deviation_s=dt_bad,
                        R_v_native_equals_v_world=world_ok, v_native=v_native.tolist(),
                        topology_version=S[0]["topology_version"], initial_topology_version=tf["topology_version"],
                        transform_reference_matches=reference_matches, transform_status=tf_cfg.get("status"), problems=problems[:5],
                        sheet_displacement_world_m=(R @ (S[-1]["positions"][0] - x0[0])).tolist())
    _gate(G, "u13_motion", g_motion)

    def g_frame():
        imp = a["arrays"]["impulse"]
        bad = []
        for k, row in enumerate(ledger[:W]):
            w = np.array([f(row, "sumG_x"), f(row, "sumG_y"), f(row, "sumG_z")])
            n = np.array([f(row, "sumG_native_x"), f(row, "sumG_native_y"), f(row, "sumG_native_z")])
            arch = imp[k].sum(0)
            # the three sums differ only in the reduction order: a component is compared to 1e-12 of the window's
            # impulse magnitude (a small, cancelling component is not held to 1e-12 of itself)
            tol = 1e-12 * max(float(np.linalg.norm(w)), float(np.linalg.norm(n))) + 1e-18
            if not (np.allclose(w, [n[0], n[2], -n[1]], rtol=0, atol=tol) and np.allclose(arch, n, rtol=0, atol=tol)):
                bad.append(dict(window=k, world=w.tolist(), native=n.tolist(), archive=arch.tolist()))
        frame, units = man.get("coordinate_frame"), man.get("units")
        return frame == "world (XPBD native)" and units == "SI" and not bad and len(ledger) >= W, dict(
            coordinate_frame=frame, units=units, windows_bad=bad[:5], impulse_unit="N s")
    _gate(G, "impulse_frame", g_frame)
    _gate(G, "mpm_balance", lambda: (
        summary["balance"]["bal_rel_max"] is not None and summary["balance"]["bal_rel_max"] <= 1e-10,
        dict(bal_rel_max=summary["balance"]["bal_rel_max"])))

    # ------------------------------------------------------------------ per-window series and timing (records)
    try:
        L = [x for x in (summary.get("load_ratio") or [])]
        wpf = int(json.loads((run / "config.json").read_text(encoding="utf-8"))["scene"]["windows_per_frame"])
        vx_rows = summary.get("frame_mean_velocity") or []
        out["series"] = dict(
            load_ratio=L, sumG_world=[[f(r, "sumG_x"), f(r, "sumG_y"), f(r, "sumG_z")] for r in ledger[:W]],
            n_active=[f(r, "n_active") for r in ledger[:W]], n_pen_active=[f(r, "n_pen_active") for r in ledger[:W]],
            pen_max=[f(r, "pen_max") for r in ledger[:W]], missed_new=[f(r, "missed_new") for r in ledger[:W]],
            frame_end_window=[j * wpf - 1 for j in range(len(vx_rows))], frame_mean_velocity=vx_rows,
            window_s=[r["window_s"] for r in (coupled or {}).get("windows", [])],
            xpbd_consume_s=[r.get("xpbd_consume_s") for r in (coupled or {}).get("windows", [])])
        ws = np.asarray(out["series"]["window_s"][1:], np.float64)
        out["timing"] = dict(first_window_s=(out["series"]["window_s"] or [None])[0],
                             median_s=float(np.median(ws)) if ws.size else None,
                             p95_s=float(np.percentile(ws, 95)) if ws.size else None,
                             max_s=float(ws.max()) if ws.size else None,
                             run_wall_s=(coupled or {}).get("wall_s"), cpu_threads=summary.get("cpu_threads"),
                             actual_arch=summary.get("actual_arch"), n=summary.get("n"))
        out["records"]["contact_onset_window"] = summary.get("contact_onset_window")
    except Exception as e:  # noqa: BLE001
        out["records"]["series_error"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ case gates (pre-registered)
    C = out["case_gates"]
    gates = case.get("gates", {})
    if args.rehearsal:
        C.update({k: dict(passed=None, note="rehearsal: listed, not evaluated") for k in gates})
    else:
        S = out["series"]
        Gy = np.array([g[1] for g in S.get("sumG_world", [])])
        Gx = np.array([g[0] for g in S.get("sumG_world", [])])
        Lr = np.asarray(S.get("load_ratio", []), np.float64)
        for name, spec in gates.items():
            if name == "sumGy_negative_windows":
                lo, hi = spec
                _gate(C, name, lambda lo=lo, hi=hi: (len(Gy) >= hi and bool((Gy[lo:hi] < 0).all()),
                                                     dict(windows=[lo, hi], max_sumG_y=float(Gy[lo:hi].max()))))
            elif name == "onset_range":
                def g_onset(spec=spec):
                    nz = [bool(np.asarray(x).any()) for x in a["arrays"]["impulse"][:W]]
                    onset = nz.index(True) if True in nz else None
                    return (onset is not None and spec[0] <= onset <= spec[1] and not any(nz[:onset]),
                            dict(onset=onset, range=spec))
                _gate(C, name, g_onset)
            elif name == "sumGy_after_onset_negative":
                def g_after():
                    nz = [bool(np.asarray(x).any()) for x in a["arrays"]["impulse"][:W]]
                    onset = nz.index(True)
                    return float(Gy[onset:].sum()) < 0, dict(onset=onset, sum_after=float(Gy[onset:].sum()))
                _gate(C, name, g_after)
            elif name == "load_ratio_absmax":
                (lo, hi), mx = spec["windows"], spec["max"]
                _gate(C, name, lambda lo=lo, hi=hi, mx=mx: (len(Lr) >= hi and float(np.abs(Lr[lo:hi]).max()) < mx,
                                                            dict(windows=[lo, hi], absmax=float(np.abs(Lr[lo:hi]).max()), max=mx)))
            elif name == "load_ratio_mean":
                (lo, hi), (a_, b_) = spec["windows"], spec["range"]
                _gate(C, name, lambda lo=lo, hi=hi, a_=a_, b_=b_: (
                    len(Lr) >= hi and a_ <= float(Lr[lo:hi].mean()) <= b_,
                    dict(windows=[lo, hi], mean=float(Lr[lo:hi].mean()), range=[a_, b_])))
            elif name == "sumGx_total_negative":
                _gate(C, name, lambda: (len(Gx) == W and float(Gx.sum()) < 0, dict(sum=float(Gx.sum()))))
            elif name == "vx_mean":
                def g_vx(spec=spec):
                    lo, hi = spec["windows"]
                    a_, b_ = spec["range"]
                    ends = S["frame_end_window"]
                    vals = [S["frame_mean_velocity"][j][0] for j in range(1, len(ends)) if lo <= ends[j] < hi]
                    m = float(np.mean(vals))
                    low_ok = m > a_ if spec.get("low_exclusive") else m >= a_
                    return bool(vals) and low_ok and m <= b_, dict(windows=[lo, hi], frames=len(vals), mean=m,
                                                                   range=[a_, b_], low_exclusive=bool(spec.get("low_exclusive")))
                _gate(C, name, g_vx)
            else:
                C[name] = dict(passed=False, error=f"unknown gate {name!r}")
    passed = all(g["passed"] for g in G.values()) and all(g["passed"] is not False for g in C.values())
    out["passed"] = bool(passed)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    failed = [k for k, g in {**G, **C}.items() if g["passed"] is False]
    print(json.dumps(dict(case=case["name"], passed=passed, failed=failed), default=str))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
