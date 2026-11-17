"""MPM endpoint of the window exchange (P2-U2; python orchestration, no kernels). Design:
interface specification .

    ep = MPMEndpoint(sim, scene=..., cfg=...)     # registers the real MPM source
    msg = ep.step(offer)                          # = ep.receive(offer); ep.advance()
    receipt = peer.consume(msg); ep.acknowledge(receipt)    # or ep.reject(reason) when the peer refuses

Phases: idle -> receive(offer) -> received -> advance() -> delivered (k + 1, the batch is out, its
receipt awaited; P2-U3) -> acknowledge(receipt) -> idle; a stop from anywhere -> stopped (final).
k = windows delivered = index of the next window. A stop in the delivered phase leaves that
window delivered but unacknowledged (awaiting_receipt: the peer's commit is unknown).

receive(offer): checks only, then LiveSurface.ingest (which itself checks before changing
anything): idle phase at a window boundary, schema, stream (bound by the first offer), source
(the offer must name this endpoint's registered source, the load source the peer expects),
window_index == k, h == H = n_sub dt exactly (shortest decimal of h times 1e9 == H_ns),
particle_dt == dt, h / particle_dt == n_sub (XPBD's rule), S_0.time == 0 at solver time 0,
S_k.time == the last batch's t_end exactly and |S_k.time - k H| <= time_skew, an explicit
bin_capacity large enough for the parked triangles . Then S4 (cloth off its frozen-velocity prediction) before any substep.

advance(): drives the window with the call sequence of Solver.substep (contact.begin_period,
n_sub x Solver._substep, contact.end_period; bit for bit the same, test E1) so that S2 (table
overflow) and S3 (missed contacts), known once the period's candidate table is built, stop
before any substep; then the batch (t_start = S_k.time, t_end = S_k.time + h, every active
vertex by stable ID, native frame), the window's measures, S1 S5 S6 S7 after the substeps (the
window is computed but not delivered). S5 counts, in the window's last substep, the constrained
particles (contact_state 2: r <= r0 + d_hat) and those of them on the illegal side
(contact_dist < 0); both from the particle fields, so the ratio is <= 1. S7 uses the endpoint's
own Diagnostics every balance_every windows (default 1).

Fail-stop: any rejection or stop rule ends the endpoint (EndpointStop, final); a rejected offer
leaves solver and surface untouched; recovery is a new endpoint, stream and solver.

Source and provenance: SourceIdentity("mpm-couple.<scene>", "real_backend", version), version =
the first 8 hex digits of the code digest: SHA-256 of the sorted (path, SHA-256) list of
mpm/**/*.py, coupling/**/*.py and the scene files (the project has no version control). The
registration adds the config digest, runtime, run id, epoch and, at the first offer, the peer
binding and the first snapshot. Every window adds a ledger row (substeps and clock before and
after, realised contact-sink increment against the batch sum, counters, ingest measures, a
SHA-256 of the particle positions and velocities after the window) and extends a SHA-256
chain over batch and state digests (archive.verify recomputes it). This is an auditable,
self-consistent record of one process, not a proof against a forger inside that process; the
evidence that the loads come from a real MPM run across processes belongs to P2-U3 / U4.
"""
import datetime
import hashlib
import json
import math
import pathlib
import platform
import sys
import time
import uuid
from fractions import Fraction

import numpy as np
import taichi as ti

from mpm import Diagnostics
from mpm.contact.live import LiveSurface
from . import wire
from .contracts import SCHEMA, ImpulseWindow, LoadBatch, SourceIdentity, substeps_of
from .prescribed import receipt_digest as receipt_content_digest

WORK = pathlib.Path(__file__).resolve().parents[1]
STOP_DEFAULTS = dict(missed_windows=3, pen_ratio=0.005, pen_windows=5, pen_min_active=20, dev_warn_fraction=0.5,
                     impulse_rel=1e-9, bal_rel=1e-10, time_skew=1e-9, bin_per_triangle=64, balance_every=1)
DIGEST_RULE = ("batch_digest = SHA-256(vertex_ids int64 LE bytes | impulse float64 LE bytes, row-major (V, 3) | "
               "utf-8 repr((window_index, batch_id, (epoch, step_index, topology_version), t_start.hex(), "
               "t_end.hex()))); state_digest = SHA-256(x float64 | v float64 bytes of all particles after the "
               "window, working precision cast to float64); chain_0 = SHA-256('mpm-endpoint-v1|<run_id>|"
               "<code_digest>'), chain_k = SHA-256(ascii(chain_{k-1} + batch_digest_k + state_digest_k)); "
               "all digests lowercase hex")


class EndpointStop(RuntimeError):
    def __init__(self, reason, window):
        super().__init__(f"window {window}: {reason}")
        self.reason, self.window = reason, window


def sha256_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def code_hashes(work=WORK, extra=()) -> dict:
    """{path relative to Work_place: SHA-256} of mpm/**/*.py, coupling/**/*.py and extra files."""
    work = pathlib.Path(work).resolve()
    files = list(work.glob("mpm/**/*.py")) + list(work.glob("coupling/**/*.py")) + [pathlib.Path(p) for p in extra]
    out = {}
    for f in files:
        f = f.resolve()
        try:
            rel = str(f.relative_to(work))
        except ValueError:
            rel = f.name
        out[rel.replace("\\", "/")] = sha256_file(f)
    return dict(sorted(out.items()))


def digest_of(hashes: dict) -> str:
    """Canonical: one 'path<TAB>sha256<LF>' line per file, sorted by path, UTF-8, SHA-256."""
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(hashes.items())).encode()).hexdigest()


def batch_digest(window_index, batch_id, cloth_version, t_start, t_end, vertex_ids, impulse) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertex_ids, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(impulse, dtype="<f8").tobytes())
    h.update(repr((int(window_index), str(batch_id), tuple(int(v) for v in cloth_version), float(t_start).hex(),
                   float(t_end).hex())).encode())
    return h.hexdigest()


def chain_next(chain: str, batch: str, state: str) -> str:
    return hashlib.sha256((chain + batch + state).encode("ascii")).hexdigest()


def chain_start(run_id: str, code_digest: str) -> str:
    return hashlib.sha256(f"mpm-endpoint-v1|{run_id}|{code_digest}".encode()).hexdigest()


class StopRules:
    """Pre-registered stop rules of the U2 local protocol ; python only.
    limit = r0 + d_hat [m]. ingest(): S4 at the read; period(): S2 S3 once the window's candidate
    table is built (before any substep); window(): S1 S5 S6 after the substeps; frame(): S1 S7."""

    def __init__(self, limit, **params):
        unknown = set(params) - set(STOP_DEFAULTS)
        if unknown:
            raise KeyError(f"stop rules: unknown parameters {sorted(unknown)}")
        self.p = dict(STOP_DEFAULTS, **params)
        self.limit = float(limit)
        self.missed_run = 0
        self.pen_run = 0

    def ingest(self, rep) -> tuple:
        dev, warns = rep.get("max_prediction_deviation"), []
        if dev is not None:
            if not dev <= self.limit:
                return f"S4 cloth off its frozen-velocity prediction by {dev:.3g} m > r0 + d_hat = {self.limit:.3g} m", warns
            if dev > self.p["dev_warn_fraction"] * self.limit:
                warns.append(f"S4 warning: prediction deviation {dev:.3g} m > {self.p['dev_warn_fraction']} (r0 + d_hat)")
        return None, warns

    def period(self, overflow_new, missed_new):
        if overflow_new > 0:
            return f"S2 candidate/bin table overflow (+{overflow_new}) when the window's table was built"
        self.missed_run = self.missed_run + 1 if missed_new > 0 else 0
        if self.missed_run >= self.p["missed_windows"]:
            return f"S3 missed contacts in {self.missed_run} consecutive windows"
        return None

    def window(self, row) -> tuple:
        p, warns = self.p, []
        if row["guard_new"] > 0 or not row["impulse_finite"]:
            return "S1 non-finite value in the window", warns
        if row["n_active"] >= p["pen_min_active"]:              # small denominators are not counted (no change)
            self.pen_run = self.pen_run + 1 if row["n_pen_active"] / row["n_active"] > p["pen_ratio"] else 0
        if self.pen_run >= p["pen_windows"]:
            return f"S5 penetrated > {p['pen_ratio']:.3%} of the constrained in {self.pen_run} windows", warns
        if row["pen_max_run"] > self.limit:
            return f"S5 penetration depth {row['pen_max_run']:.3g} m > r0 + d_hat", warns
        if not row["impulse_rel"] <= p["impulse_rel"]:
            return f"S6 batch sum differs from the contact sink by {row['impulse_rel']:.3g} (relative)", warns
        return None, warns

    def frame(self, d) -> str:
        if d.get("nonfinite", 0) > 0:
            return "S1 non-finite particles"
        if not d["bal_rel"] <= self.p["bal_rel"]:
            return f"S7 momentum balance {d['bal_rel']:.3g} > {self.p['bal_rel']:.3g}"
        return None


class MPMEndpoint:
    """P2-U3 changes : a delivered window waits for its
    receipt (phase "delivered": no next offer before acknowledge); the registration with peer = null
    is frozen at construction (registration_frozen: canonical JSON text, registration_digest:
    its SHA-256), the late peer binding is checked against the peer's binding, never folded into
    the frozen identity; acknowledge computes the receipt digest itself (canonical JSON rule,
    wire.DIGEST_RULE) from the receipt actually received and checks the receipt against the
    delivered batch. handshake=True (the two-process driver) makes the phase machine require
    confirm_registration before the first offer and confirm_peer before the first advance."""

    def __init__(self, sim, *, scene, cfg, run_id=None, scene_files=(), stop=None, work=WORK, handshake=False):
        c = sim.contact
        if c is None or not isinstance(c.surface, LiveSurface):
            raise ValueError("the MPM endpoint needs a contact block with a live surface (contact.surface.type = live)")
        H_ns = c.window_ns()
        if H_ns is None:
            raise ValueError("exchange window H = exchange_substeps * dt must be a whole number of ns (D30)")
        if sim._period_pos != 0 or int(sim.step[None]) != 0:
            raise ValueError("the endpoint needs a fresh solver at a window boundary")
        self.sim, self.block, self.surface = sim, c, c.surface
        self.n_sub, self.H_ns, self.dt = int(c.n_sub), int(H_ns), float(sim.dt_host)
        self.H = self.H_ns / 1e9
        hashes = code_hashes(work, scene_files)
        code_digest = digest_of(hashes)
        self.source = SourceIdentity(f"mpm-couple.{scene}", "real_backend", int(code_digest[:8], 16))
        self.run_id = run_id or uuid.uuid4().hex
        pr = c.params()
        self.rules = StopRules(pr["r0"] + pr["d_hat"], **(stop or {}))
        self.diag = Diagnostics(sim)                            # S7 balance; reference row at t = 0
        self.diag()
        self.registration = dict(
            schema="mpm-endpoint-registration-v1", source=self.source.as_dict(), scene=scene,
            mpm_run_id=self.run_id, mpm_epoch=1,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            code_digest=code_digest, code_files=hashes,
            config_digest=hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                                                    default=str).encode()).hexdigest(),
            runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                         taichi=".".join(str(v) for v in ti.__version__),
                         arch=str(ti.lang.impl.current_cfg().arch), cpu_threads=cfg["runtime"].get("cpu_threads"),
                         precision=cfg["precision"], platform=platform.platform()),
            window=dict(n_sub=self.n_sub, dt=self.dt, H_ns=self.H_ns),
            stop=dict(self.rules.p, r0_plus_d_hat=self.rules.limit), peer=None, digest_rule=DIGEST_RULE,
            evidence="auditable record of this process: per window solver substeps and clock, realised "
                     "contact-sink increment vs the batch sum, counters, ingest measures, particle-state digest, "
                     "batch digest and chain (archive.verify); not a proof against a forger in the process")
        # frozen identity (peer = null), never rewritten; registration["peer"] is filled at the first offer (U2)
        self.registration_frozen = wire.canonical_json(self.registration)
        self.registration_digest = hashlib.sha256(self.registration_frozen.encode("utf-8")).hexdigest()
        self.handshake = dict(required=bool(handshake), digest_rule=wire.DIGEST_RULE, registration_confirmed=None,
                              peer_confirmed=None)
        self.k = 0
        self.phase = "idle"
        self.stream_id = None
        self.stopped = None
        self.awaiting_receipt = None
        self.warnings = []
        self.ledger, self.records = [], []
        self.snapshot0 = None
        self.chain0 = self.chain = chain_start(self.run_id, code_digest)
        self._offer = None
        self._delivered = None
        self._last_t_end = None

    # ------------------------------------------------------------ state
    def provenance(self) -> dict:
        return dict(registration=self.registration, registration_frozen=self.registration_frozen,
                    registration_digest=self.registration_digest, handshake=self.handshake, windows_delivered=self.k,
                    windows_acknowledged=self.acknowledged(), awaiting_receipt=self.awaiting_receipt,
                    stopped=self.stopped, chain_start=self.chain0, chain_final=self.chain, warnings=list(self.warnings))

    def acknowledged(self) -> int:
        return sum(1 for r in self.records if r.get("acknowledged"))

    def _stop(self, reason, window, cause=None):
        if self.phase == "delivered":                          # stopped while waiting for the receipt: commit unknown
            self.awaiting_receipt = self.k - 1
        # read but not advanced (no record of window k yet): kept for the archive 
        pending = self.k if self.phase == "received" and not (self.records and
                                                              self.records[-1]["window_index"] == self.k) else None
        self.stopped = dict(reason=reason, window=int(window), windows_delivered=self.k,
                            windows_acknowledged=self.acknowledged(), awaiting_receipt=self.awaiting_receipt,
                            pending_window=pending)
        self.phase = "stopped"
        self._offer = None
        raise EndpointStop(reason, window) from cause

    def _alive(self):
        if self.stopped is not None:
            raise EndpointStop(f"endpoint stopped earlier ({self.stopped['reason']}); start a new endpoint, stream "
                               "and solver", self.k)

    # ------------------------------------------------------------ handshake (P2-U3)
    def confirm_registration(self, digest, info=None):
        """The peer's registration_digest must be the digest of the frozen registration (peer = null)."""
        self._alive()
        try:
            if self.phase != "idle" or self.k != 0 or self.handshake["registration_confirmed"] is not None:
                raise RuntimeError(f"registration handshake in phase {self.phase!r} at window {self.k} (once, before "
                                   "the first offer)")
            if digest != self.registration_digest:
                raise ValueError(f"peer registration digest {digest!r} differs from the frozen registration digest "
                                 f"{self.registration_digest}")
        except Exception as e:
            self._stop(f"S8 handshake: {type(e).__name__}: {e}", self.k, e)
        self.handshake["registration_confirmed"] = dict(digest=digest, **(info or {}))

    def confirm_peer(self, binding):
        """The peer's binding (six fields) must equal, as canonical JSON, the peer recorded at the first offer."""
        self._alive()
        try:
            if self.phase != "received" or self.k != 0 or self.handshake["peer_confirmed"] is not None:
                raise RuntimeError(f"peer binding check in phase {self.phase!r} at window {self.k} (once, after the "
                                   "first offer is read and before it is advanced)")
            mine, theirs = wire.canonical_json(self.registration["peer"]), wire.canonical_json(binding)
            if mine != theirs:
                raise ValueError(f"peer binding {theirs} differs from the recorded {mine}")
        except Exception as e:
            self._stop(f"S8 handshake: {type(e).__name__}: {e}", self.k, e)
        self.handshake["peer_confirmed"] = dict(binding=binding)

    # ------------------------------------------------------------ receive
    def _check_offer(self, offer):
        if self.phase != "idle":
            raise RuntimeError(f"offer in phase {self.phase!r} (a received window must be advanced and a delivered "
                               "window acknowledged first)")
        if self.handshake["required"] and self.handshake["registration_confirmed"] is None:
            raise RuntimeError("offer before the registration handshake (confirm_registration)")
        if self.sim._period_pos != 0:
            raise RuntimeError("the solver is not at a window boundary")
        if getattr(offer, "schema", None) != SCHEMA:
            raise ValueError(f"offer schema {getattr(offer, 'schema', None)!r}, expected {SCHEMA!r}")
        stream = getattr(offer, "stream_id", None)
        if self.stream_id is None:
            if not isinstance(stream, str) or not stream.strip():
                raise ValueError("offer stream_id must be a nonempty string")
        elif stream != self.stream_id:
            raise ValueError(f"offer stream {stream!r} is not the bound stream {self.stream_id!r}")
        src = getattr(offer, "source", None)
        got = (getattr(src, "source_id", None), getattr(src, "kind", None), getattr(src, "version", None))
        want = (self.source.source_id, self.source.kind, self.source.version)
        if got != want:
            raise ValueError(f"offer expects load source {got}, this endpoint is the registered {want} "
                             "(no relabelling: the peer must register this source)")
        if getattr(offer, "window_index", None) != self.k:
            raise ValueError(f"offer window_index {getattr(offer, 'window_index', None)!r}, expected {self.k}")
        h, pdt = float(offer.h), float(offer.particle_dt)
        if Fraction(str(h)) * 10 ** 9 != self.H_ns:
            raise ValueError(f"h = {h!r} s is not the MPM exchange window H = {self.H_ns} ns (H = h)")
        if pdt != self.dt:
            raise ValueError(f"particle_dt = {pdt!r} is not the MPM substep dt = {self.dt!r}")
        if substeps_of(h, pdt) != self.n_sub:
            raise ValueError(f"h / particle_dt = {substeps_of(h, pdt)} is not exchange_substeps = {self.n_sub}")
        t = float(offer.snapshot.time)
        if self.k == 0:
            if t != 0.0 or float(self.sim.time[None]) != 0.0 or int(offer.snapshot.step_index) != 0:
                raise ValueError(f"the stream must start at snapshot step 0, time 0 and solver time 0 (got step "
                                 f"{offer.snapshot.step_index!r}, time {t!r}, solver {float(self.sim.time[None])!r})")
        elif t != self._last_t_end:
            raise ValueError(f"snapshot time {t!r} is not the last batch's t_end {self._last_t_end!r}")
        elif float(offer.snapshot.dt) != h:                    # the previous completed window's h (interface specification;
            raise ValueError(f"snapshot dt {float(offer.snapshot.dt)!r} is not the previous window's h {h!r}")
            # window 0's dt is a construction convention, checked by the two-process driver against BOUND)
        if abs(t - self.k * self.H) > self.rules.p["time_skew"]:
            raise ValueError(f"snapshot time {t!r} drifted from k H = {self.k * self.H!r}")
        if self.k == 0 and self.block.spec.get("bin_capacity") is not None:
            T = len(np.asarray(offer.snapshot.triangles))
            need = (self.surface.T_cap - T) + self.rules.p["bin_per_triangle"] * T
            if self.block.bin_cap < need:
                raise ValueError(f"contact.bin_capacity = {self.block.bin_cap} < {need} = parked triangles "
                                 f"({self.surface.T_cap - T}) + {self.rules.p['bin_per_triangle']} per active one")

    def receive(self, offer) -> dict:
        """Check the offer and read its snapshot; no solver step. Returns the ingest report."""
        self._alive()
        try:
            self._check_offer(offer)
            rep = self.surface.ingest(offer.snapshot)
        except Exception as e:                                 # nothing changed: checks precede every write
            code = "S1" if "non-finite" in str(e) else "S8"    # a non-finite snapshot is S1 
            self._stop(f"{code} rejected the offer for window {self.k}: {type(e).__name__}: {e}", self.k, e)
        if self.k == 0:
            self.stream_id = offer.stream_id
            S = offer.snapshot
            self.snapshot0 = {n: np.array(getattr(S, n)) for n in ("positions", "velocity", "triangles", "vertex_ids",
                                                                   "triangle_ids", "pinned")}
            self.snapshot0.update(version=np.array([S.epoch, S.step_index, S.topology_version], np.int64),
                                  time=np.float64(S.time), dt=np.float64(S.dt))
            self.registration["peer"] = dict(stream_id=offer.stream_id, expected_source=self.source.as_dict(),
                                             snapshot_epoch=int(S.epoch), topology_version=int(S.topology_version),
                                             n_vertices=int(len(np.asarray(S.vertex_ids))),
                                             n_triangles=int(len(np.asarray(S.triangles))))
        reason, warns = self.rules.ingest(rep)                 # S4 before any substep on this geometry
        self.warnings += [f"window {self.k}: {w}" for w in warns]
        if reason is not None:
            self._stop(reason, self.k)
        self._offer, self._ingest, self.phase = offer, rep, "received"
        return rep

    # ------------------------------------------------------------ advance
    def _counters(self):
        c, sim = self.block, self.sim
        return dict(step=int(sim.step[None]), time=float(sim.time[None]),
                    JG=np.asarray(c.JG[None].to_numpy(), np.float64), KG=np.asarray(c.KG[None].to_numpy(), np.float64),
                    JP=np.asarray(c.JP[None].to_numpy(), np.float64), guard=int(sim.guard.events[None]),
                    missed=int(c.missed[None]), overflow=int(c.cand_overflow[None]) + int(c.bin_overflow[None]))

    def _state_digest(self) -> str:
        P = self.sim.P
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(P.x.to_numpy(), dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(P.v.to_numpy(), dtype="<f8").tobytes())
        return h.hexdigest()

    def advance(self) -> ImpulseWindow:
        """The received window: candidate table, S2/S3, n_sub substeps, batch, measures, S1/S5/S6/S7."""
        self._alive()
        k = self.k
        if self.phase != "received":
            self._stop(f"S8 advance in phase {self.phase!r} (receive an offer first)", k)
        if self.handshake["required"] and k == 0 and self.handshake["peer_confirmed"] is None:
            self._stop("S8 handshake: advance of the first window before the peer binding check (confirm_peer)", k)
        offer, sim, c = self._offer, self.sim, self.block
        pre = self._counters()
        t0 = time.perf_counter()
        c.begin_period()                                       # = Solver.substep at period position 0
        mid_missed = int(c.missed[None]) - pre["missed"]
        mid_over = int(c.cand_overflow[None]) + int(c.bin_overflow[None]) - pre["overflow"]
        reason = self.rules.period(mid_over, mid_missed)
        if reason is not None:                                 # stopped before any substep of this window
            self.ledger.append(dict(window=k, stop=reason, delivered=False, step_before=pre["step"],
                                    step_after=int(sim.step[None]), missed_new=mid_missed, overflow_new=mid_over,
                                    batch_digest=None, state_digest=None, chain=self.chain, receipt_digest=None))
            self._stop(reason, k)
        for _ in range(self.n_sub):
            sim._substep()
            sim._period_pos += 1
        sim._period_pos = 0
        c.end_period()
        post = self._counters()
        wall = time.perf_counter() - t0
        S = offer.snapshot
        version = tuple(int(v) for v in (S.epoch, S.step_index, S.topology_version))
        b = self.surface.last_batch()
        try:
            if post["step"] - pre["step"] != self.n_sub:
                raise RuntimeError(f"the window did not advance exactly {self.n_sub} substeps")
            if b is None or b["window_index"] != k or tuple(b["cloth_version"]) != version:
                raise RuntimeError("the settled batch does not belong to this window and snapshot")
        except Exception as e:
            self._stop(f"S8 {e}", k, e)
        t_start = float(S.time)
        t_end = t_start + float(offer.h)
        batch_id = f"{offer.stream_id}/w{k:05d}"
        G_w, G_n = b["impulse_G"], b["impulse_G_native"]
        finite = bool(np.isfinite(G_n).all() and np.isfinite(G_w).all())
        dJG, dKG, dJP = post["JG"] - pre["JG"], post["KG"] - pre["KG"], post["JP"] - pre["JP"]
        sumG = G_w.sum(0)
        nJ = float(np.linalg.norm(dJG))
        rel = float(np.linalg.norm(sumG - dJG)) / nJ if nJ > 0 else (0.0 if not sumG.any() else math.inf)
        state = sim.P.contact_state.to_numpy()
        dist = sim.P.contact_dist.to_numpy()
        con = state == 2
        rep = self._ingest
        digest = batch_digest(k, batch_id, version, t_start, t_end, b["vertex_ids"], G_n)
        sdig = self._state_digest()
        chain = chain_next(self.chain, digest, sdig)
        row = dict(window=k, epoch=version[0], step_index=version[1], topology_version=version[2],
                   snapshot_time=t_start, t_end=t_end, t_start_ns=b["t_start_ns"], t_end_ns=b["t_end_ns"],
                   step_before=pre["step"], step_after=post["step"], time_before=pre["time"], time_after=post["time"],
                   n_vertices=len(b["vertex_ids"]),
                   dJG_x=dJG[0], dJG_y=dJG[1], dJG_z=dJG[2], sumG_x=sumG[0], sumG_y=sumG[1], sumG_z=sumG[2],
                   sumG_native_x=G_n[:, 0].sum(), sumG_native_y=G_n[:, 1].sum(), sumG_native_z=G_n[:, 2].sum(),
                   dJP_x=dJP[0], dJP_y=dJP[1], dJP_z=dJP[2], dKG_x=dKG[0], dKG_y=dKG[1], dKG_z=dKG[2],
                   G_vertex_max=float(np.linalg.norm(G_w, axis=1).max()) if finite else math.inf,
                   n_active=int(con.sum()), n_pen_active=int((con & (dist < 0)).sum()),
                   n_active_block=int(c.n_active[None]), n_pen_candidates=int(c.n_pen[None]),
                   n_pen_max_run=int(c.n_pen_max_run[None]), n_ambig=int(c.n_ambig[None]),
                   pen_max=float(c.pen_max[None]), pen_max_run=float(c.pen_max_run[None]),
                   missed_new=post["missed"] - pre["missed"], overflow_new=post["overflow"] - pre["overflow"],
                   guard_new=post["guard"] - pre["guard"], impulse_finite=finite, impulse_rel=rel, bal_rel=None,
                   max_step=rep["max_step"], max_prediction_deviation=rep["max_prediction_deviation"],
                   rebuilt=rep["rebuilt"], reordered=rep["reordered"], wall_advance_s=wall,
                   batch_digest=digest, state_digest=sdig, chain=chain, receipt_digest=None,
                   receipt_content_digest=None, delivered=False, acknowledged=False, peer_rejected=None, stop=None,
                   warnings="")
        reason, warns = self.rules.window(row)
        if reason is None and (k + 1) % int(self.rules.p["balance_every"]) == 0:
            d = self.diag()
            row["bal_rel"] = float(d["bal_rel"])
            reason = self.rules.frame(d)
        row.update(stop=reason, warnings="; ".join(warns), delivered=reason is None)
        self.warnings += [f"window {k}: {w}" for w in warns]
        rig = self.surface.rigid
        self.records.append(dict(window_index=k, batch_id=batch_id, cloth_version=version, t_start=t_start,
                                 t_end=t_end, t_start_ns=b["t_start_ns"], t_end_ns=b["t_end_ns"],
                                 vertex_ids=np.asarray(b["vertex_ids"], np.int64).copy(),
                                 impulse=np.asarray(G_n, np.float64).copy(), J_material=rig.vec_to_native(dJG),
                                 J_to_cloth=G_n.sum(0), K_material=rig.torque_to_native(dKG, dJG),
                                 counts=np.array([row["n_active"], row["n_pen_active"], row["missed_new"],
                                                  row["overflow_new"]], np.int64),
                                 delivered=reason is None, acknowledged=False, digest=digest, state_digest=sdig))
        self.ledger.append(row)
        self.chain = chain
        if reason is not None:
            self._stop(reason, k)
        self.k += 1
        self._last_t_end = t_end
        batch = LoadBatch(stream_id=offer.stream_id, source=self.source, cloth_version=version, batch_id=batch_id,
                          t_start=t_start, t_end=t_end, vertex_ids=b["vertex_ids"], impulse=G_n)
        self._delivered_window = dict(h=float(offer.h), previous_dt=float(S.dt))
        self._offer, self._delivered, self.phase = None, batch, "delivered"      # waits for the receipt (P2-U3)
        return ImpulseWindow(SCHEMA, k, batch)

    def step(self, offer) -> ImpulseWindow:
        self.receive(offer)
        return self.advance()

    def _check_receipt(self, receipt, k):
        """The receipt actually received must describe the delivered batch (an echo of the loads by stable ID)."""
        b = self._delivered
        missing = [n for n in ("t_start", "t_end", "start_version", "committed_version", "vertex_ids", "impulse")
                   if n not in receipt]
        if missing:
            raise ValueError(f"receipt lacks {missing}")
        if type(receipt["t_start"]) is not float or type(receipt["t_end"]) is not float \
                or receipt["t_start"] != b.t_start or receipt["t_end"] != b.t_end:
            raise ValueError(f"receipt interval {receipt['t_start']!r}..{receipt['t_end']!r} is not the batch's "
                             f"{b.t_start!r}..{b.t_end!r}")
        e, s, tv = b.cloth_version
        sv, cv = receipt["start_version"], receipt["committed_version"]
        if list(sv) != [e, s, tv]:
            raise ValueError(f"receipt start_version {sv!r} is not the batch's cloth_version {[e, s, tv]}")
        if not (isinstance(cv, list) and len(cv) == 3 and cv[0] == e and cv[1] == s + 1 and cv[2] >= tv):
            raise ValueError(f"receipt committed_version {cv!r} is not (epoch {e}, step {s + 1}, topology >= {tv})")
        ids = np.asarray(receipt["vertex_ids"])
        imp = np.asarray(receipt["impulse"], dtype=np.float64)
        bid = np.asarray(b.vertex_ids, np.int64)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or imp.shape != (len(ids), 3) or len(ids) != len(bid):
            raise ValueError("receipt vertex_ids / impulse shape differs from the delivered batch")
        o_r, o_b = np.argsort(ids, kind="stable"), np.argsort(bid, kind="stable")
        if not np.array_equal(ids[o_r].astype(np.int64), bid[o_b]):
            raise ValueError("receipt vertex IDs differ from the delivered batch")
        if np.ascontiguousarray(imp[o_r]).tobytes() != np.ascontiguousarray(np.asarray(b.impulse)[o_b]).tobytes():
            raise ValueError("receipt impulse is not the delivered batch, bit for bit by stable ID")
        reg, w = self.registration, self._delivered_window
        for key, want in (("stream_id", b.stream_id), ("batch_id", b.batch_id), ("source", b.source.as_dict()),
                          ("h", w["h"]), ("previous_dt", w["previous_dt"]), ("mpm_run_id", reg["mpm_run_id"]),
                          ("mpm_epoch", reg["mpm_epoch"]), ("code_digest", reg["code_digest"]),
                          ("config_digest", reg["config_digest"])):        # when present 
            if key in receipt and (receipt[key] != want or type(receipt[key]) is not type(want)):
                raise ValueError(f"receipt {key} {receipt[key]!r} is not {want!r}")
        if self.handshake["required"]:
            if receipt.get("registration_digest") != self.registration_digest:
                raise ValueError(f"receipt registration_digest {receipt.get('registration_digest')!r} is not the frozen "
                                 f"{self.registration_digest}")
            pc = self.handshake["peer_confirmed"]
            if pc is None or wire.canonical_json(receipt.get("registration_binding")) != wire.canonical_json(pc["binding"]):
                raise ValueError("receipt registration_binding differs from the confirmed peer binding")

    def acknowledge(self, receipt: dict):
        """The peer's receipt for the delivered window: computes its digest (canonical JSON, wire.DIGEST_RULE)
        and its content digest (by stable ID) from the receipt as received, checks it against the delivered
        batch, records both; then the next offer may come. Any mismatch stops (S8)."""
        self._alive()
        k = self.k - 1
        try:
            if self.phase != "delivered" or k < 0:
                raise RuntimeError(f"phase {self.phase!r}: no delivered window waits for a receipt")
            if type(receipt) is not dict:
                raise TypeError("a receipt must be a JSON object (dict)")
            if type(receipt.get("window_index")) is not int or receipt["window_index"] != k:   # no bool (S1)
                raise ValueError(f"receipt for window {receipt.get('window_index')!r}, expected {k}")
            digest = wire.json_digest(receipt)                 # JSON-native values only
            content = receipt_content_digest(receipt)
            self._check_receipt(receipt, k)
        except Exception as e:
            self._stop(f"S8 receipt for window {k}: {type(e).__name__}: {e}", max(k, 0), e)
        self.ledger[-1].update(receipt_digest=digest, receipt_content_digest=content, peer_rejected=False,
                               acknowledged=True)
        self.records[-1]["acknowledged"] = True
        self._delivered, self.phase = None, "idle"

    def reject(self, reason: str):
        """The peer refused the delivered window: stop (S8); the window stays in the ledger and the archive as
        not delivered (its batch is not re-fed). Outside the delivered phase it stops without marking windows."""
        self._alive()
        k = self.k - 1
        if self.phase == "delivered" and k >= 0:
            self.ledger[-1].update(peer_rejected=True, delivered=False)
            self.records[-1]["delivered"] = False
            self.phase = "rejected"
            self._stop(f"S8 the peer rejected window {k}: {reason}", k)
        self._stop(f"S8 the peer rejected (phase {self.phase!r}, no delivered window waits for a receipt): {reason}",
                   max(self.k, 0))

    def halt(self, reason: str):
        """A stop decided by the driver (e.g. a scene's frame check): final, like any other stop."""
        self._alive()
        self._stop(reason, self.k)

    def check_frame(self, d: dict):
        """A scene's frame diagnostics (S1 non-finite particles, S7 balance), between windows."""
        self._alive()
        reason = self.rules.frame(d)
        if reason is not None:
            self._stop(reason, self.k - 1)
