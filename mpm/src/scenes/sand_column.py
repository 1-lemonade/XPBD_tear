"""Dry-sand column collapse on the ground (stage-1 smoke scene, F2 at scene level).

    python scenes/sand_column.py [--config ...] [--set grid.n=32 --set scene.t_end=0.3] [--arch cpu|cuda] [--out DIR]

Writes <out>/<name>/ per export contract v1 (): config.json,
run_info.json, particles.npz, diag.csv, frames/*.npz (if enabled), summary.json,
summary.png, profile.json (runtime.kernel_profiler). Analysis is the 2026-10-30 one
(runout, final height, mid slope from the radial max-height profile).
"""
import json
import sys

import numpy as np

import _cli
from mpm import Solver
from mpm.diagnostics import nanmax
from mpm.export import CONTRACT

SCENE_DEFAULTS = dict(
    name="sand_column",
    column=dict(lo_xz=[0.4, 0.4], hi_xz=[0.6, 0.6], height=0.4),   # base on the ground plane y = buffer_cells*dx
    t_end=1.5, fps=30, seed=0, export_frames=True,
    vmax_hint=3.0,                 # velocity used for the start-up CFL estimate only
)


def sample_box(lo, hi, dx, ppc, rng):
    """Jittered-grid sampling, ppc particles per cell."""
    k = round(ppc ** (1 / 3))
    assert k ** 3 == ppc, "use a cubic ppc (1, 8, 27) for jittered sampling"
    h = dx / k
    axes = [np.arange(lo[d] + h / 2, hi[d], h) for d in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g + (rng.random(g.shape) - 0.5) * 0.5 * h


def initial_positions(cfg):
    sc, gc = cfg["scene"], cfg["grid"]
    dx = gc["domain"] / gc["n"]
    floor = gc["buffer_cells"] * dx
    col = sc["column"]
    lo = (col["lo_xz"][0], floor, col["lo_xz"][1])
    hi = (col["hi_xz"][0], floor + col["height"], col["hi_xz"][1])
    return sample_box(lo, hi, dx, cfg["particles"]["ppc"], np.random.default_rng(sc["seed"])), floor


def analyse(x, cfg, floor):
    col = cfg["scene"]["column"]
    cx = 0.5 * (col["lo_xz"][0] + col["hi_xz"][0])
    cz = 0.5 * (col["lo_xz"][1] + col["hi_xz"][1])
    r = np.hypot(x[:, 0] - cx, x[:, 2] - cz)
    y = x[:, 1] - floor
    bins = np.linspace(0, 0.45, 46)
    idx = np.digitize(r, bins)
    prof = np.array([[0.5 * (bins[i - 1] + bins[i]), y[idx == i].max()] for i in range(1, len(bins)) if (idx == i).sum() > 20])
    runout = float(np.percentile(r, 99.5))
    h_final = float(np.percentile(y, 99.5))
    mid = prof[(prof[:, 1] > 0.15 * h_final) & (prof[:, 1] < 0.85 * h_final)]
    slope = float(np.degrees(np.arctan(-np.polyfit(mid[:, 0], mid[:, 1], 1)[0]))) if len(mid) >= 3 else float("nan")
    return dict(runout_r=runout, height_final=h_final, mid_slope_deg=slope,
                initial_height=col["height"], initial_half_width=0.5 * (col["hi_xz"][0] - col["lo_xz"][0])), prof, y


def plot(sim, x, y, prof, rows, slope, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col_c = 0.5 * sim.grid.domain
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    sel = np.abs(x[:, 2] - col_c) < 0.03
    fl = sim.P.flag.to_numpy()[sel]
    for v, c, lab in ((0, "tab:red", "tip"), (1, "tab:blue", "elastic"), (2, "tab:orange", "plastic")):
        s = fl == v
        ax[0].scatter(x[sel][s, 0], y[sel][s], s=1, c=c, label=lab)
    ax[0].set_aspect("equal"); ax[0].set_title("final slice |z-0.5|<0.03, by flag"); ax[0].legend(markerscale=6)
    ax[1].plot(prof[:, 0], prof[:, 1], "k.-"); ax[1].set_title(f"radial max-height profile, mid slope ~ {slope:.1f} deg")
    ax[1].set_xlabel("r [m]"); ax[1].set_ylabel("h [m]")
    T = [r_["t"] for r_ in rows]
    ax[2].plot(T, [r_["kinetic"] for r_ in rows], label="KE [J]")
    ax[2].plot(T, [r_["frac_plastic"] for r_ in rows], label="frac plastic")
    ax[2].plot(T, [r_["cfl"] for r_ in rows], label="CFL")
    ax[2].set_yscale("log"); ax[2].legend(); ax[2].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = _cli.parser("sand_column", __doc__.splitlines()[0]).parse_args()
    cfg, rt = _cli.start(args, SCENE_DEFAULTS)
    sc = cfg["scene"]
    pos, floor = initial_positions(cfg)
    sim, build_s = _cli.timed(Solver, cfg, pos)
    writer = _cli.open_run(args, cfg)
    warns = sim.report(vmax_hint=sc["vmax_hint"])
    diag, d0, timing = _cli.prepare(sim, writer, cfg, rt, args, warns, build_s)

    rows, status, loop = _cli.run_frames(sim, writer, diag, sc["t_end"], sc["fps"])
    timing.update(loop)
    post_start = _cli.time.perf_counter()

    x = sim.P.x.to_numpy().astype(np.float64)
    geo, prof, y = analyse(x, cfg, floor) if status == "ok" else (dict(), None, None)
    mat0 = sim.model.material_info(0)
    guard = sim.guard.readout()
    with_grid = rows[1:]
    summary = dict(status=status, **{k: timing[k] for k in ("wall_s", "first_frame_s", "substeps", "steps_per_frame",
                                                          "steady_ms_per_substep")}, n=sim.n,
                   mass_drift_rel=(rows[-1]["mass"] - d0["mass"]) / d0["mass"],
                   kinetic_final=rows[-1]["kinetic"], kinetic_peak=max(r_["kinetic"] for r_ in rows),
                   cfl_max=max(r_["cfl"] for r_ in rows), inverted_max=max(r_["inverted"] for r_ in rows),
                   clamped_total=rows[-1]["clamped_total"], **geo,
                   friction_angle_range_deg=mat0.get("friction_angle_range_deg"),
                   actual_arch=rt["actual_arch"], cpu_threads=rt["cpu_threads"],
                   boundary_readout=sim.bset.readout(),
                   contract=CONTRACT, timing=timing,
                   balance=dict(bal_rel_max=nanmax(r_["bal_rel"] for r_ in with_grid),
                                bal_abs_max=[nanmax(abs(r_[f"bal_p{c}"]) for r_ in rows) for c in "xyz"],
                                grid_mass_rel_max=nanmax(abs(r_["grid_mass_rel"]) for r_ in with_grid),
                                g2p_rel_max=nanmax(r_["g2p_rel"] for r_ in with_grid),
                                reference=diag.balance_ref(), clock=sim.clock()),
                   nonfinite=guard)
    _cli.close_timing(timing, post_start)
    writer.write_json("summary.json", summary)
    _cli.finish_profiles(cfg, sim, writer)
    print(json.dumps({k: v for k, v in summary.items() if k != "boundary_readout"}, indent=2, default=str))
    if status == "ok":
        plot(sim, x, y, prof, rows, geo["mid_slope_deg"], writer.dir / "summary.png")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
