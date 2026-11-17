"""Drucker-Prager dry sand: Hencky elasticity + return mapping in log-strain space
(Klar et al. 2016), optional volume correction (Tampubolon et al. 2017). No cohesion (D1).

Parameters live in a Taichi table indexed by the particle's mat_id, so they can be
changed at run time without recompiling. The kernel formulas are those of the
2026-10-30 implementation (F1/F2 evidence) with constants replaced by table reads;
the derived constant k_vol = (3 lam + 2 mu) / (2 mu) is computed in Python double,
which is exactly what the old code's compile-time folding produced.

Hardening on the cone tip (option tip_dq, structural): Klar 2016 sets dq = |eps|_F for
a tip projection, eps the trial log strain. With the volume correction the projected
strain is e = eps + vc/3; "corrected" (default, the 2026-10-30 formula) uses |e|_F,
"elastic" uses |eps|_F. They differ only with volume_correction on: a particle resting
with vc > 0 is projected to the tip every substep and "corrected" adds |vc|/sqrt(3)
to q each time, so q grows with the number of substeps while nothing moves (P1-U3).
"""
import math

import taichi as ti

from ..config import check_keys

FLAG_TIP, FLAG_ELASTIC, FLAG_PLASTIC = 0, 1, 2

# Klar et al. 2016 sand parameters, checked against the paper (P1-U3, D5): Table 3 rows
# Butterfly / Raking / Shovel (rho, E, nu, h0/h1/h2/h3 = 35/9/0.2/10, angles in degrees).
# The paper notes real sand has E = 3.537e7 Pa; 3.537e5 is its chosen value.
# Scene configs use constant phi = 30 deg instead (h0 = 30, h1 = h3 = 0; D5, P1-U3); this
# hardening set stays for the tests and as a configurable option.
KLAR2016_SAND_RECALLED = dict(E=3.537e5, nu=0.3, rho=2200.0, h0=35.0, h1=9.0, h2=0.2, h3=10.0)


@ti.data_oriented
class DruckerPragerSand:
    name = "dp_sand"
    PARAM_KEYS = ("E", "nu", "rho", "h0", "h1", "h2", "h3", "sigma_min")
    PARAM_DEFAULTS = dict(sigma_min=1e-4)          # floor on singular values before log (D4)
    OPTION_DEFAULTS = dict(volume_correction=True, tip_dq="corrected")
    TIP_DQ = ("corrected", "elastic")

    def __init__(self, materials, dtype, **options):
        check_keys("constitutive.options", options, self.OPTION_DEFAULTS)
        opts = dict(self.OPTION_DEFAULTS, **options)
        if opts["tip_dq"] not in self.TIP_DQ:
            raise ValueError(f"constitutive.options.tip_dq must be one of {list(self.TIP_DQ)}, got {opts['tip_dq']!r}")
        if not materials:
            raise ValueError("constitutive.materials is empty")
        self.dtype = dtype
        self.vol_corr = bool(opts["volume_correction"])     # structural (ti.static)
        self.tip_dq = opts["tip_dq"]                         # structural (ti.static)
        self.options = opts
        self.n_mat = len(materials)
        row = ti.types.struct(E=dtype, nu=dtype, mu=dtype, lam=dtype, k_vol=dtype, rho=dtype,
                              h0=dtype, h1=dtype, h2=dtype, h3=dtype, sigma_min=dtype)
        self.table = row.field(shape=self.n_mat)
        self._rows = [None] * self.n_mat
        for mid, spec in enumerate(materials):
            spec = dict(spec)
            label = spec.pop("name", f"mat{mid}")
            check_keys(f"constitutive.materials[{mid}]", spec, self.PARAM_KEYS)
            missing = [k for k in self.PARAM_KEYS if k not in spec and k not in self.PARAM_DEFAULTS]
            if missing:
                raise KeyError(f"constitutive.materials[{mid}] missing {missing}")
            self._rows[mid] = dict(self.PARAM_DEFAULTS, name=label)
            self.set_material(mid, **spec)

    # ------------------------------------------------------------ python side
    def state_members(self) -> dict:
        return dict(q=self.dtype,        # accumulated plastic measure (hardening variable)
                    vc=self.dtype,       # volume-correction log strain
                    alpha=self.dtype)    # alpha(q) after the last update (diagnostics only)

    def set_material(self, mid: int, **kw):
        """Run-time parameter change: writes the table row, no recompilation."""
        check_keys(f"material {mid}", kw, self.PARAM_KEYS)
        row = self._rows[mid]
        row.update(kw)
        E, nu = float(row["E"]), float(row["nu"])
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        row.update(mu=mu, lam=lam, k_vol=(3.0 * lam + 2.0 * mu) / (2.0 * mu))
        for k in ("E", "nu", "mu", "lam", "k_vol", "rho", "h0", "h1", "h2", "h3", "sigma_min"):
            getattr(self.table, k)[mid] = float(row[k])

    def material_info(self, mid: int) -> dict:
        r = dict(self._rows[mid])
        r["wave_speed"] = math.sqrt((r["lam"] + 2 * r["mu"]) / r["rho"])
        curve = self.friction_angle_curve(mid)
        vals = [v for v in (curve["phi_q0"], curve["phi_inf"], curve["phi_max"]) if v is not None]
        # Range of phi(q) over q >= 0; [h0 - h3, h0] misses the
        # maximum of the curve: 48.3 deg at q = 6.1 for 35/9/0.2/10)
        r["friction_angle_range_deg"] = [min(vals), max(vals) if curve["phi_inf"] is not None else None]
        r["friction_angle_curve"] = curve
        return r

    def friction_angle_curve(self, mid: int = 0) -> dict:
        """phi(q) = h0 + (h1 q - h3) exp(-h2 q) [deg] (Klar 2016 eq. 30): value at q = 0, limit
        for q -> inf (None if unbounded), and its maximum (None if the curve is monotone)."""
        h0, h1, h2, h3 = (float(self._rows[mid][k]) for k in ("h0", "h1", "h2", "h3"))
        out = dict(phi_q0=h0 - h3, phi_inf=h0 if h2 > 0 else (h0 - h3 if h1 == 0 else None),
                   q_at_max=None, phi_max=None)
        if h1 > 0 and h2 > 0:
            q = (h1 + h2 * h3) / (h1 * h2)
            out.update(q_at_max=q, phi_max=h0 + (h1 * q - h3) * math.exp(-h2 * q))
        return out

    def alpha_py(self, q: float, mid: int = 0) -> float:
        r = self._rows[mid]
        phi = math.radians(r["h0"] + (r["h1"] * q - r["h3"]) * math.exp(-r["h2"] * q))
        s = math.sin(phi)
        return math.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)

    def describe(self) -> dict:
        return dict(model=self.name, options=self.options,
                    materials=[self.material_info(i) for i in range(self.n_mat)])

    # ------------------------------------------------------------ taichi side
    @ti.func
    def _alpha(self, q, mid):
        phi = (self.table.h0[mid] + (self.table.h1[mid] * q - self.table.h3[mid]) * ti.exp(-self.table.h2[mid] * q)) * (math.pi / 180.0)
        s = ti.sin(phi)
        return ti.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)

    @ti.func
    def _tau_from_log(self, U, e, mid):
        """Kirchhoff stress tau = P F^T = U diag(2 mu e + lam tr e) U^T (Hencky)."""
        d = 2.0 * self.table.mu[mid] * e + self.table.lam[mid] * e.sum()
        return U @ ti.Matrix([[d[0], 0, 0], [0, d[1], 0], [0, 0, d[2]]]) @ U.transpose()

    @ti.func
    def _log_sv(self, F, mid):
        U, sig, V = ti.svd(F)
        smin = self.table.sigma_min[mid]
        # D4: max() also turns a NaN singular value into sigma_min (F stays finite, the guard
        # cannot see it). Letting NaN through works but changes the fast-math code of the
        # update (ulp-level, P1-U3), so it is not done here.
        e = ti.Vector([ti.log(ti.max(sig[i, i], smin)) for i in ti.static(range(3))])
        return U, e, V

    @ti.func
    def init_state(self, P: ti.template(), p):
        P.q[p] = 0.0
        P.vc[p] = 0.0
        P.alpha[p] = self._alpha(ti.cast(0.0, self.dtype), P.mat_id[p])

    @ti.func
    def stress(self, P: ti.template(), p, F):
        """Stateless: tau from log F only. vc is already contained in F after update()
        (F_new = U exp(H) V^T), so adding vc here would count it twice."""
        mid = P.mat_id[p]
        U, e, V = self._log_sv(F, mid)
        return self._tau_from_log(U, e, mid)

    @ti.func
    def update(self, P: ti.template(), p, F_trial):
        """Return mapping. Reads/writes this model's state members of particle p;
        returns (F_new, tau, flag). Callers never touch q, vc, alpha."""
        mid = P.mat_id[p]
        q = P.q[p]
        vc = P.vc[p]
        U, eps, V = self._log_sv(F_trial, mid)
        e = eps
        if ti.static(self.vol_corr):
            e = eps + vc / 3.0
        alpha = self._alpha(q, mid)

        tr = e.sum()
        e_hat = e - tr / 3.0
        e_hat_norm = e_hat.norm()
        H = e
        dq = 0.0 * q
        flag = FLAG_ELASTIC
        if tr > 0.0:
            # case II: volumetric expansion -> project to cone apex.
            # (pure hydrostatic compression has e_hat = 0, dgamma < 0 -> elastic, no division; D9)
            H = e * 0.0
            dq = e.norm()
            if ti.static(self.tip_dq == "elastic"):
                dq = eps.norm()
            flag = FLAG_TIP
        else:
            dgamma = e_hat_norm + self.table.k_vol[mid] * tr * alpha
            if dgamma > 0.0:
                H = e - dgamma * e_hat / e_hat_norm
                dq = dgamma
                flag = FLAG_PLASTIC

        expH = ti.exp(H)
        F_new = U @ ti.Matrix([[expH[0], 0, 0], [0, expH[1], 0], [0, 0, expH[2]]]) @ V.transpose()
        vc_new = vc
        if ti.static(self.vol_corr):
            # vc += log det F_trial - log det F_new, the expansion the projection discarded
            # (tr e at the tip, 0 otherwise). Tampubolon 2017 eq. (26) prints the opposite sign,
            # which diverges for a particle at rest; its text describes this one (P1-U3).
            vc_new = vc + eps.sum() - H.sum()
        q_new = q + dq
        P.q[p] = q_new
        P.vc[p] = vc_new
        P.alpha[p] = self._alpha(q_new, mid)
        tau = self._tau_from_log(U, H, mid)
        return F_new, tau, flag
