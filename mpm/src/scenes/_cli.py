"""Shared scene command line and frame loop .

Every scene accepts:
  --config FILE        scene configuration (default scenes/configs/<scene>.json)
  --set KEY=VALUE      override any config entry, repeatable (VALUE parsed as JSON if possible)
  --arch cpu|cuda      Taichi backend; the backend actually used is printed and recorded,
                       and a cuda request that does not start on cuda exits with code 2
  --out DIR            output root; the run is written to DIR/<name>/ (default Work_place/runs)
  --name NAME          run name (default scene.name from the config)
  --threads N          CPU thread cap (default runtime.cpu_threads from the config; 0 = no cap)
  --no-frames          do not export frames (no frames/ directory is created)
  --overwrite          allow writing into an existing non-empty run directory

Timing (U2), a hierarchy (all wall-clock seconds):
  total_s     = import_s + init_s + build_s + prepare_s + wall_s + post_s (+ small glue)
  import_s      taichi and mpm imports in this module
  init_s        ti.init
  build_s       solver construction (fields, initial kernels)
  prepare_s     diagnostics set-up incl. diag_first_s (first call, compiles its kernel),
                particles.npz, config.json, run_info.json
  wall_s        frame loop = sim_s + diag_s + io_s (+ loop glue); sim_s contains
                first_substep_s (first substep call: kernel compilation or offline-cache
                load, plus one substep); ti.sync() once per frame and after that call
  post_s        scene analysis up to writing summary.json (plots afterwards are not counted)
Optional, from the config: runtime.kernel_profiler (per-task kernel times -> profile.json),
runtime.compile_profile (Taichi's scoped profiler printed at the end: compile phases),
runtime.offline_cache (null = Taichi default).
"""
import argparse
import pathlib
import re
import sys
import time

T_START = time.perf_counter()
WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))
import taichi as ti  # noqa: E402

from mpm import BackendMismatch, Diagnostics, RunWriter, init_taichi, load_config  # noqa: E402
from mpm.export import TIMING_COLUMNS, manifest  # noqa: E402

IMPORT_S = time.perf_counter() - T_START


def parser(scene: str, description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(WORK / "scenes" / "configs" / f"{scene}.json"))
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--arch", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out", default=str(WORK / "runs"))
    ap.add_argument("--name")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def start(args, scene_defaults: dict):
    """Config -> Taichi runtime. Exits 2 on a backend mismatch."""
    cfg = load_config(args.config, args.overrides, scene_defaults)
    if args.threads is not None:
        cfg["runtime"]["cpu_threads"] = args.threads
    rc = cfg["runtime"]
    extra = {}
    if rc["kernel_profiler"]:
        extra["kernel_profiler"] = True
    if rc["offline_cache"] is not None:
        extra["offline_cache"] = bool(rc["offline_cache"])
    t0 = time.perf_counter()
    try:
        rt = init_taichi(args.arch, cfg["precision"], rc["cpu_threads"], **extra)
    except BackendMismatch as e:
        print(f"[runtime] ERROR: {e}; refusing a silent fallback", file=sys.stderr)
        sys.exit(2)
    rt["init_s"] = time.perf_counter() - t0
    return cfg, rt


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def open_run(args, cfg):
    """Run directory <out>/<name>/; call after the solver is built, so an invalid
    configuration never leaves an empty run directory. Exits 3 if it would overwrite."""
    name = args.name or cfg["scene"]["name"]
    try:
        return RunWriter(args.out, name, cfg["scene"]["export_frames"] and not args.no_frames, args.overwrite)
    except FileExistsError as e:
        print(f"[run] ERROR: {e}", file=sys.stderr)
        sys.exit(3)


def prepare(sim, writer, cfg, rt, args, warns, build_s):
    """Diagnostics (first call compiles and takes the balance reference), particles.npz,
    the diag.csv header, config.json and run_info.json with the export manifest."""
    t0 = time.perf_counter()
    diag = Diagnostics(sim)
    d0, diag_first_s = timed(diag)
    columns = dict(diag.columns(), **TIMING_COLUMNS)
    writer.particles(sim)
    writer.cloth_mesh(sim)
    writer.open_diag(columns)
    writer.write_json("config.json", cfg)
    writer.write_json("run_info.json", dict(
        resolved=sim.describe(), runtime=rt, cli=vars(args), argv=sys.argv, warnings=warns,
        export=manifest(sim, columns, writer.export_frames, bool(cfg["runtime"]["kernel_profiler"]))))
    return diag, d0, dict(import_s=IMPORT_S, init_s=rt["init_s"], build_s=build_s,
                          prepare_s=time.perf_counter() - t0, diag_first_s=diag_first_s)


def close_timing(timing: dict, post_start: float) -> dict:
    """post_s and total_s; call right before writing summary.json."""
    now = time.perf_counter()
    timing["post_s"] = now - post_start
    timing["total_s"] = now - T_START
    return timing


def run_frames(sim, writer, diag, t_end: float, fps: int):
    """Advance frame by frame; a diag row every frame (written as it comes), frames
    exported if enabled. Stops at the end of the first frame with a non-finite value
    (diagnostics or the G2P guard). wall_s / first_frame_s / steady_ms_per_substep keep
    their 2026-10-30 meaning (whole loop; first frame includes JIT compilation). With a
    contact block the first substep also runs the period start (bins, candidate table)."""
    spf = max(1, round(1.0 / (fps * float(sim.dt[None]))))
    n_frames = int(round(t_end * fps))
    rows, status, first = [], "ok", None
    seg = dict(first_substep_s=None, sim_s=0.0, diag_s=0.0, io_s=0.0)
    t0 = time.perf_counter()
    for f in range(n_frames + 1):
        a = time.perf_counter()
        if f == 1:
            sim.substep()
            ti.sync()
            seg["first_substep_s"] = time.perf_counter() - a
            sim.advance(spf - 1)
        elif f > 1:
            sim.advance(spf)
        ti.sync()
        b = time.perf_counter()
        d = diag()                      # t = in-kernel clock
        c = time.perf_counter()
        writer.frame(sim, f, d["t"])
        e = time.perf_counter()
        d.update(time_sim_s=b - a, time_diag_s=c - b, time_io_s=e - c)
        writer.diag_row(d)
        rows.append(d)
        seg["sim_s"] += b - a
        seg["diag_s"] += c - b
        seg["io_s"] += e - c
        if f == 1:
            first = time.perf_counter() - t0
        if d["nonfinite"] > 0 or d["nonfinite_events"] > 0:
            status = "stopped: non-finite"
            break
        if d.get("contact_overflow", 0) > 0 or d.get("contact_bin_overflow", 0) > 0:
            status = "stopped: contact table overflow"     # never continue with a truncated table (U4)
            break
    wall = time.perf_counter() - t0
    writer.close()
    done = rows[-1]["step"]
    steady = (wall - first) / ((len(rows) - 2) * spf) * 1e3 if first is not None and len(rows) > 2 else None
    steady_sim = (sum(r["time_sim_s"] for r in rows[2:]) / (rows[-1]["step"] - rows[1]["step"]) * 1e3
                  if len(rows) > 2 else None)
    timing = dict(wall_s=wall, first_frame_s=first, substeps=done, steps_per_frame=spf,
                  steady_ms_per_substep=steady, **seg, steady_sim_ms_per_substep=steady_sim)
    return rows, status, timing


def finish_profiles(cfg, sim, writer):
    """profile.json (runtime.kernel_profiler) and the scoped compile profile (runtime.compile_profile)."""
    if cfg["runtime"]["kernel_profiler"]:
        writer.write_json("profile.json", kernel_profile(sim))
    if cfg["runtime"]["compile_profile"]:
        print("[timing] Taichi scoped profiler (host-side compile phases, all kernels):", flush=True)
        ti.profiler.print_scoped_profiler_info()


TASK = re.compile(r"^(?P<kernel>.+?)_c\d+_\d+_kernel_(?P<idx>\d+)_(?P<kind>[a-z_]+)$")


def kernel_profile(sim) -> dict:
    """Per offloaded task totals; the substep kernel's parallel loops are labelled with
    sim.stages() in order (raw names only if their number does not match)."""
    from taichi.profiler.kernel_profiler import get_default_kernel_profiler
    kp = get_default_kernel_profiler()
    kp._update_records()                       # Taichi 1.7.4 has no public per-record query
    agg = {}
    for r in kp._traced_records:
        a = agg.setdefault(r.name, dict(count=0, total_ms=0.0, min_ms=float("inf"), max_ms=0.0))
        a["count"] += 1
        a["total_ms"] += r.kernel_time
        a["min_ms"] = min(a["min_ms"], r.kernel_time)
        a["max_ms"] = max(a["max_ms"], r.kernel_time)
    tasks = []
    for name, a in agg.items():
        m = TASK.match(name)
        tasks.append(dict(name=name, kernel=m["kernel"] if m else name, index=int(m["idx"]) if m else -1,
                          kind=m["kind"] if m else "", **a, mean_ms=a["total_ms"] / a["count"]))
    sub = sorted((t for t in tasks if t["kernel"] == sim.SUBSTEP_KERNEL), key=lambda t: t["index"])
    loops = [t for t in sub if t["kind"] != "serial"]
    stages = {}
    labels = sim.stages()
    if len(loops) == len(labels):
        for label, t in zip(labels, loops):
            stages[label] = dict(task=t["name"], mean_ms=t["mean_ms"], total_ms=t["total_ms"], count=t["count"])
        stages["serial"] = dict(total_ms=sum(t["total_ms"] for t in sub if t["kind"] == "serial"),
                                mean_ms=sum(t["mean_ms"] for t in sub if t["kind"] == "serial"))
    note = ("kernel profiler synchronises every task: absolute times are inflated, use the shares; "
            "the substep loops are labelled by order") if stages else "substep tasks did not match sim.stages()"
    return dict(tasks=sorted(tasks, key=lambda t: -t["total_ms"]), substep_stages=stages, note=note)
