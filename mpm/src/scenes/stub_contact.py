"""Sand on a stub cloth sheet (P1-U4 fixtures F3 static load and F4 drop impact; round 2:
two-sided tracking checks, pressing from below and draining through a pre-cut hole).

    python scenes/stub_contact.py --config scenes/configs/stub_static_load.json [--set ...] [--arch cpu|cuda] [--out DIR]
    python scenes/stub_contact.py --config scenes/configs/stub_drop_impact.json
    python scenes/stub_contact.py --config scenes/configs/stub_press_below.json
    python scenes/stub_contact.py --config scenes/configs/stub_hole_drain.json

A box of dry sand starts `bottom_gap` above a fixed (or translating) stub sheet held by
the contact block (mpm/contact/). Output per export contract v1 plus the U4 contact
columns, frame fields and cloth/ files (). summary.json adds
  load     over the window [t1, t2]: mean force F = -dJ_y / dt of the P and G measures
           and of J_absorbed, each over M g (static load F3: G -> 1 when the sand rests
           only on the sheet; this follows from the momentum balance, it is not a contact
           stiffness), the P/G ratio, the floor/wall sinks over M g, the spread of the
           frame-mean G force (std over the window), the mean contact_gain, and the mass
           fraction still on the sheet (above its plane, inside its footprint) at t2
  impact   (F4) particles on the illegal side (contact_penetrated: max over substeps),
           deepest of them, largest legal-side overlap, peak mean force over a frame over
           M g (signed, largest magnitude; > 0 = pushed down), the centre-of-mass velocity at impact (most negative) and the largest
           rebound after it, and from the frames an independent count of particles below
           the sheet plane inside its footprint (pass-through), per frame max and final
  crossing (needs frames; independent of the contact block's own side logic) particles
           whose side of the sheet plane changes between two frames, classified by where
           their straight path meets the plane: through the sheet = illegal (unique
           particles; those within r0 + d_hat of a hole edge also counted apart), inside a
           hole or outside the sheet = legal; drained = mass fraction below the sheet plane
           at the end; with side = tracked the final count of remembered sides +1 / -1
           (frame sampling is coarse: a path that bends between two frames is classified
           by its chord)
  rest     kinetic energy final/peak, final vmax
This is a qualitative fixture: no calibration is implied.
"""
import json

import numpy as np

import _cli
from mpm import Solver
from mpm.diagnostics import nanmax
from mpm.export import CONTRACT
from sand_column import sample_box

SCENE_DEFAULTS = dict(
    name="stub_contact",
    block=dict(center_xz=[0.5, 0.5], size=[0.4, 0.1, 0.4], bottom_gap=0.0),   # above the sheet plane [m]
    t_end=1.0, fps=30, seed=0, export_frames=True,
    vmax_hint=2.0,                 # velocity used for the start-up CFL estimate only
    window=[0.6, 1.0],             # load averaging window [s]
)


def sheet_geometry(cfg):
    s = cfg["contact"]["surface"]
    a = {"x": 0, "y": 1, "z": 2}[s["normal_axis"]]
    if a != 1:
        raise ValueError("stub_contact expects a horizontal sheet (normal_axis y; normal_sign +-1)")
    c = s["center"]
    return dict(y=float(c[1]), lo=(c[0] - s["size"][0] / 2, c[2] - s["size"][1] / 2),
                hi=(c[0] + s["size"][0] / 2, c[2] + s["size"][1] / 2))


def initial_positions(cfg):
    sc, gc = cfg["scene"], cfg["grid"]
    dx = gc["domain"] / gc["n"]
    sh = sheet_geometry(cfg)
    b = sc["block"]
    (cx, cz), (sx, sy, sz) = b["center_xz"], b["size"]
    y0 = sh["y"] + b["bottom_gap"]
    lo = (cx - sx / 2, y0, cz - sz / 2)
    hi = (cx + sx / 2, y0 + sy, cz + sz / 2)
    return sample_box(lo, hi, dx, cfg["particles"]["ppc"], np.random.default_rng(sc["seed"])), sh


def at(rows, t):
    """Row whose time is closest to t."""
    return min(rows, key=lambda r: abs(r["t"] - t))


def vec(row, prefix):
    return np.array([row[f"{prefix}{c}"] for c in "xyz"])


def load_summary(rows, sim, window):
    M = rows[0]["mass"]
    g = float(np.linalg.norm(sim.gravity[None].to_numpy()))
    r1, r2 = at(rows, window[0]), at(rows, window[1])
    dt = r2["t"] - r1["t"]
    out = dict(window=[r1["t"], r2["t"]], Mg=M * g)
    for key, prefix in (("P", "contact_JP_"), ("G", "contact_JG_"), ("abs", "contact_Jabs_")):
        F = (vec(r2, prefix) - vec(r1, prefix)) / dt
        out[f"F_{key}"] = F.tolist()
        out[f"ratio_{key}"] = float(-F[1] / (M * g))
    out["ratio_P_over_G"] = out["ratio_P"] / out["ratio_G"] if out["ratio_G"] != 0 else None
    other = {}
    for label, cols in sim.sink_columns().items():
        if label == "contact":
            continue
        dJ = np.array([r2[c] - r1[c] for c in cols]) / dt
        other[label] = float(-dJ[1] / (M * g))
    out["other_sinks_ratio_y"] = other
    i1, i2 = rows.index(r1), rows.index(r2)
    T = np.array([r["t"] for r in rows[i1:i2 + 1]])
    J = np.array([r["contact_JG_y"] for r in rows[i1:i2 + 1]])
    if len(T) > 2:
        out["G_frame_force_std_over_Mg"] = float(np.std(-np.diff(J) / np.diff(T)) / (M * g))
    gains = [r["contact_gain"] for r in rows[i1:i2 + 1] if np.isfinite(r["contact_gain"])]
    out["gain_mean"] = float(np.mean(gains)) if gains else None
    return out


def on_sheet_fraction(sim, sh):
    x = sim.P.x.to_numpy().astype(np.float64)
    m = sim.P.m.to_numpy().astype(np.float64)
    inside = (x[:, 0] > sh["lo"][0]) & (x[:, 0] < sh["hi"][0]) & (x[:, 2] > sh["lo"][1]) & (x[:, 2] < sh["hi"][1])
    return float(m[inside & (x[:, 1] > sh["y"])].sum() / m.sum())


def pass_through(writer, sh, r0, n_frames):
    """From the exported frames: particles strictly below the sheet plane inside its footprint."""
    counts = []
    for f in range(n_frames):
        path = writer.dir / "frames" / f"{f:05d}.npz"
        if not path.exists():
            break
        x = np.load(path)["x"].astype(np.float64)
        inside = (x[:, 0] > sh["lo"][0]) & (x[:, 0] < sh["hi"][0]) & (x[:, 2] > sh["lo"][1]) & (x[:, 2] < sh["hi"][1])
        counts.append(int((inside & (x[:, 1] < sh["y"])).sum()))
    return counts


def crossings(writer, cfg, sim, n_frames):
    """Frame-based side changes relative to the (horizontal, translating) sheet plane. A
    change between two frames is classified by where the straight path between the two
    positions meets the plane: inside a hole or outside the sheet = legal; on the sheet =
    illegal (a crossing through a face), reported apart when within r0 + d_hat of a hole
    edge ."""
    S = sim.contact.surface
    x0c = S.X0.mean(0)
    surf = cfg["contact"]["surface"]
    p = sim.contact.params()
    edge = p["r0"] + p["d_hat"]
    holes = [((h["center"][0], h["center"][1]), (h["size"][0] / 2, h["size"][1] / 2)) for h in (surf.get("holes") or [])]
    half = (surf["size"][0] / 2, surf["size"][1] / 2)
    prev = None
    illegal, near, per_frame, legal = set(), set(), [], 0
    m = sim.P.m.to_numpy().astype(np.float64)
    drained = None
    for f in range(n_frames):
        fp, cp = writer.dir / "frames" / f"{f:05d}.npz", writer.dir / "cloth" / f"{f:05d}.npz"
        if not fp.exists() or not cp.exists():
            break
        x = np.load(fp)["x"].astype(np.float64)
        off = np.load(cp)["x"].mean(0) - x0c
        yc = x0c[1] + off[1]
        side = np.where(x[:, 1] >= yc, 1, -1)
        if prev is not None:
            xp, sp, ycp = prev
            ch = np.flatnonzero(side != sp)
            a = np.clip((0.5 * (yc + ycp) - xp[ch, 1]) / (x[ch, 1] - xp[ch, 1]), 0.0, 1.0)
            u = xp[ch, 0] + a * (x[ch, 0] - xp[ch, 0]) - off[0]
            w = xp[ch, 2] + a * (x[ch, 2] - xp[ch, 2]) - off[2]
            on_sheet = (np.abs(u - x0c[0]) < half[0]) & (np.abs(w - x0c[2]) < half[1])
            in_hole = np.zeros(len(ch), bool)
            by_hole = np.zeros(len(ch), bool)
            for (hu, hw), (gu, gw) in holes:
                in_hole |= (np.abs(u - hu) < gu) & (np.abs(w - hw) < gw)
                by_hole |= (np.abs(u - hu) < gu + edge) & (np.abs(w - hw) < gw + edge)
            bad = on_sheet & ~in_hole
            illegal.update(ch[bad].tolist())
            near.update(ch[bad & by_hole].tolist())
            per_frame.append(int(bad.sum()))
            legal += int((~bad).sum())
        prev = (x, side, yc)
        drained = float(m[side < 0].sum() / m.sum())
    out = dict(illegal_particles=len(illegal), illegal_near_hole_edge=len(near),
               illegal_max_per_frame=max(per_frame) if per_frame else 0,
               legal_crossings=legal, drained_final=drained, hole_edge_band=edge)
    if "contact_side" in sim.contact.state_members():
        sd = sim.P.contact_side.to_numpy()
        out["side_final"] = {"+1": int((sd == 1).sum()), "-1": int((sd == -1).sum()), "0": int((sd == 0).sum())}
    return out


def plot(sim, rows, sh, load, through, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = sim.P.x.to_numpy().astype(np.float64)
    st = sim.P.contact_state.to_numpy()
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    sel = np.abs(x[:, 2] - 0.5 * (sh["lo"][1] + sh["hi"][1])) < 0.02
    for v, c, lab in ((0, "tab:gray", "none"), (1, "tab:blue", "candidate"), (2, "tab:red", "constrained")):
        s = sel & (st == v)
        ax[0].scatter(x[s, 0], x[s, 1], s=1, c=c, label=lab)
    ax[0].plot([sh["lo"][0], sh["hi"][0]], [sh["y"]] * 2, "k-", lw=1)
    ax[0].set_aspect("equal", adjustable="datalim"); ax[0].set_title("final slice |z - z_c| < 0.02, contact state")
    ax[0].legend(markerscale=6, fontsize=8, loc="upper right")
    T = np.array([r["t"] for r in rows])
    M, g = rows[0]["mass"], load["Mg"] / rows[0]["mass"]
    for prefix, lab in (("contact_JP_", "P"), ("contact_JG_", "G"), ("contact_Jabs_", "abs")):
        J = np.array([r[f"{prefix}y"] for r in rows])
        ax[1].plot(T, -J / (M * g), label=f"-J_{lab},y/(Mg)")
    ax[1].plot(T, T, "k:", label="t (full support)")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("[s]"); ax[1].legend(); ax[1].set_title("cumulative impulse / Mg")
    Tm = 0.5 * (T[1:] + T[:-1])
    for prefix, lab in (("contact_JP_", "P"), ("contact_JG_", "G")):
        J = np.array([r[f"{prefix}y"] for r in rows])
        ax[2].plot(Tm, -np.diff(J) / np.diff(T) / (M * g), label=lab)
    ax[2].axhline(1.0, color="k", ls=":")
    ax[2].set_xlabel("t [s]"); ax[2].set_title("frame-mean force / Mg"); ax[2].legend()
    ax[3].plot(T, [r["contact_penetrated"] for r in rows], label="illegal side, last substep")
    ax[3].plot(T, [r["contact_active"] for r in rows], label="constrained")
    if through:
        ax[3].plot(T[:len(through)], through, label="below sheet (frames)")
    ax[3].set_yscale("symlog"); ax[3].set_xlabel("t [s]"); ax[3].legend(); ax[3].set_title("counts")
    fig.tight_layout(); fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = _cli.parser("stub_static_load", __doc__.splitlines()[0]).parse_args()
    cfg, rt = _cli.start(args, SCENE_DEFAULTS)
    if not cfg["contact"]:
        raise SystemExit("stub_contact needs a contact section in the config")
    sc = cfg["scene"]
    pos, sh = initial_positions(cfg)
    sim, build_s = _cli.timed(Solver, cfg, pos)
    writer = _cli.open_run(args, cfg)
    warns = sim.report(vmax_hint=sc["vmax_hint"])
    diag, d0, timing = _cli.prepare(sim, writer, cfg, rt, args, warns, build_s)

    rows, status, loop = _cli.run_frames(sim, writer, diag, sc["t_end"], sc["fps"])
    timing.update(loop)
    post_start = _cli.time.perf_counter()

    c = sim.contact
    load = load_summary(rows, sim, sc["window"]) if status == "ok" else {}
    if load:
        load["mass_on_sheet_final"] = on_sheet_fraction(sim, sh)
    through = pass_through(writer, sh, c.params()["r0"], len(rows)) if writer.export_frames else []
    M, g = rows[0]["mass"], float(np.linalg.norm(sim.gravity[None].to_numpy()))
    T = np.array([r["t"] for r in rows])
    JGy = np.array([r["contact_JG_y"] for r in rows])
    fy = -np.diff(JGy) / np.diff(T) / (M * g)            # > 0: the material pushes the sheet down
    peak = float(fy[np.argmax(np.abs(fy))]) if len(rows) > 1 else None
    last = rows[-1]
    vy = np.array([r["py"] if "py" in r else r["momentum"][1] for r in rows]) / M
    i_imp = int(np.argmin(vy))
    impact = dict(penetrated_max_run=last["contact_penetrated_max_run"], pen_max_run=last["contact_pen_max_run"],
                  overlap_max_run=last["contact_overlap_max_run"],
                  com_vy_impact=float(vy[i_imp]), com_vy_rebound_max=float(vy[i_imp:].max()),
                  t_impact_frame=float(rows[i_imp]["t"]),
                  penetrated_final=last["contact_penetrated"], peak_frame_force_G_over_Mg=peak,
                  pass_through_max=max(through) if through else None, pass_through_final=through[-1] if through else None,
                  missed=last["contact_missed"], overflow=last["contact_overflow"],
                  ambiguous_final=last["contact_ambiguous"])
    crossing = crossings(writer, cfg, sim, len(rows)) if writer.export_frames else None
    with_grid = rows[1:]
    guard = sim.guard.readout()
    summary = dict(status=status, **{k: timing[k] for k in ("wall_s", "first_frame_s", "substeps", "steps_per_frame",
                                                          "steady_ms_per_substep")}, n=sim.n,
                   mass_drift_rel=(last["mass"] - d0["mass"]) / d0["mass"],
                   kinetic_final=last["kinetic"], kinetic_peak=max(r["kinetic"] for r in rows),
                   cfl_max=max(r["cfl"] for r in rows), inverted_max=max(r["inverted"] for r in rows),
                   clamped_total=last["clamped_total"],
                   friction_angle_range_deg=sim.model.material_info(0).get("friction_angle_range_deg"),
                   actual_arch=rt["actual_arch"], cpu_threads=rt["cpu_threads"],
                   boundary_readout=sim.bset.readout(), contract=CONTRACT, timing=timing,
                   contact=dict(params=c.params(), exchange_substeps=c.n_sub, correction=c.correction,
                                epochs=last["contact_epoch"], candidates_final=last["contact_candidates"],
                                rq_final=last["contact_rq"], JG_split_rel_final=last["contact_JG_split_rel"],
                                last_batch=dict((k, v) for k, v in c.surface.last_batch().items()
                                                if k in ("cloth_version", "t_start", "t_end"))),
                   load=load, impact=impact, crossing=crossing, side=c.side_mode,
                   rest=dict(ke_final_over_peak=last["kinetic"] / max(r["kinetic"] for r in rows)
                             if max(r["kinetic"] for r in rows) > 0 else None, vmax_final=last["vmax"]),
                   balance=dict(bal_rel_max=nanmax(r["bal_rel"] for r in with_grid),
                                bal_abs_max=[nanmax(abs(r[f"bal_p{cc}"]) for r in rows) for cc in "xyz"],
                                grid_mass_rel_max=nanmax(abs(r["grid_mass_rel"]) for r in with_grid),
                                g2p_rel_max=nanmax(r["g2p_rel"] for r in with_grid),
                                reference=diag.balance_ref(), clock=sim.clock()),
                   nonfinite=guard)
    _cli.close_timing(timing, post_start)
    writer.write_json("summary.json", summary)
    _cli.finish_profiles(cfg, sim, writer)
    print(json.dumps({k: summary[k] for k in ("status", "wall_s", "n", "load", "impact", "crossing", "rest", "balance")},
                     indent=2, default=str))
    if status == "ok":
        plot(sim, rows, sh, load, through, writer.dir / "summary.png")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
