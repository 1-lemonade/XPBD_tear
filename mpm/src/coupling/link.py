"""MPM side of the process link to the XPBD service (P2-U3; host I/O, standard library + NumPy).

    link = XPBDLink(service_cfg, timeouts)
    hello = link.start()                     # spawn + HELLO (or PeerError from the service's ERROR)
    fr = link.request("OPEN", meta, expect="OFFER", timeout=...)   # WireTimeout / WireClosed / PeerError
    rec = link.stop(reason)                  # STOP, drain until BYE, join, terminate if needed

multiprocessing spawn context on every platform (Windows only has spawn), a duplex Pipe, the
service entry coupling.xpbd_service.main as the target. The service process is a daemon so it
cannot outlive the driver process.
"""
import multiprocessing as mp
import time

from . import wire
from . import xpbd_service

# Default timeouts: handshake 120 s, message 30 s, stop 5 s.
# grace; idle (the service waiting for the next request) must exceed MPM's longest window, kernel compile included
DEFAULT_TIMEOUTS = dict(hello=120.0, register=120.0, open=30.0, peer=30.0, impulse=30.0, send=30.0, stop=5.0, idle=300.0)


class PeerError(RuntimeError):
    """The service answered with ERROR."""

    def __init__(self, meta):
        super().__init__(f"XPBD service error at {meta.get('stage')}: {meta.get('error_type')}: {meta.get('message')}")
        self.meta = meta


class XPBDLink:
    def __init__(self, service_cfg, timeouts=None):
        self.cfg = dict(service_cfg)
        self.t = dict(DEFAULT_TIMEOUTS, **(timeouts or {}))
        unknown = set(self.t) - set(DEFAULT_TIMEOUTS)
        if unknown:
            raise KeyError(f"transport timeouts: unknown keys {sorted(unknown)}")
        self.cfg.setdefault("idle_timeout", self.t["idle"])
        self.cfg.setdefault("send_timeout", self.t["send"])
        self.send_stalled = None
        self.proc = None
        self.chan = None
        self.hello = None
        self.stopped = None

    def start(self) -> dict:
        ctx = mp.get_context("spawn")
        mine, theirs = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=xpbd_service.main, args=(theirs, self.cfg), name="xpbd-service", daemon=True)
        self.proc.start()
        theirs.close()
        self.chan = wire.Channel(mine, max_frame=int(self.cfg.get("max_frame_bytes", wire.MAX_FRAME)),
                                 recv_modes=wire.FROM_SERVICE)
        fr = self.chan.recv(self.t["hello"], modes=("HELLO", "ERROR"), on_stall=self._stalled)
        if fr.mode == "ERROR":
            self.hello = fr.meta.get("hello")
            raise PeerError(fr.meta)
        self.hello = fr.meta
        return fr.meta

    def _stalled(self):
        """Send or read watchdog: the service stopped reading, or a frame from it never completed; end it (its
        end closes, the blocked send or read fails)."""
        self.send_stalled = time.monotonic()
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()

    def request(self, mode, meta, arrays=None, *, expect, timeout):
        self.chan.send(mode, meta, arrays, timeout=self.t["send"], on_stall=self._stalled)
        fr = self.chan.recv(timeout, modes=(expect, "ERROR"), on_stall=self._stalled)
        if fr.mode == "ERROR":
            raise PeerError(fr.meta)
        return fr

    def stop(self, reason) -> dict:
        """Tell the service to stop, then keep reading: replies that arrive after the driver gave up on them
        (a late OFFER or RECEIPT; they hold the next seq) are recorded as late frames and drained until BYE,
        so a service blocked in send_bytes on a large frame is released . Then reap the
        process (terminate after the stop timeout). Never raises."""
        if self.stopped is not None:
            return self.stopped
        rec = dict(reason=str(reason), stop_sent=False, bye=None, late_frames=[], drain_error=None, exitcode=None,
                   terminated=False, wall_s=None)
        t0 = time.monotonic()
        if self.chan is not None and self.chan.broken is not None:
            rec["drain_error"] = f"channel out of step, no STOP/BYE exchange: {self.chan.broken}"
        elif self.chan is not None:
            try:
                self.chan.send("STOP", dict(reason=str(reason)[:2000]), timeout=self.t["send"], on_stall=self._stalled)
                rec["stop_sent"] = True
            except (wire.WireClosed, wire.WireError) as e:
                rec["drain_error"] = f"send STOP: {type(e).__name__}: {e}"
            deadline = time.monotonic() + self.t["stop"]
            while rec["stop_sent"]:
                left = deadline - time.monotonic()
                if left <= 0:
                    rec["drain_error"] = f"no BYE within {self.t['stop']} s"
                    break
                try:
                    fr = self.chan.recv(left, modes=wire.FROM_SERVICE, on_stall=self._stalled)
                except (wire.WireTimeout, wire.WireClosed, wire.WireError) as e:
                    rec["drain_error"] = f"{type(e).__name__}: {e}"
                    break
                if fr.mode == "BYE":
                    rec["bye"] = fr.meta
                    break
                late = dict(mode=fr.mode, seq=fr.seq)          # a reply that came after the driver gave up on it
                if fr.mode == "RECEIPT":
                    late.update(window_index=fr.meta["receipt"].get("window_index"),
                                receipt_digest=fr.meta.get("receipt_digest"))
                elif fr.mode == "ERROR":
                    late.update(stage=fr.meta.get("stage"), message=fr.meta.get("message"))
                rec["late_frames"].append(late)
        if self.chan is not None:
            self.chan.close()
        if self.proc is not None:
            self.proc.join(self.t["stop"])
            if self.proc.is_alive():
                self.proc.terminate()
                rec["terminated"] = True
                self.proc.join(5.0)
            rec["exitcode"] = self.proc.exitcode
        rec["wall_s"] = time.monotonic() - t0
        rec["send_watchdog_fired"] = self.send_stalled is not None
        if self.chan is not None:
            rec.update(frames_sent=self.chan.sent, frames_received=self.chan.received, bytes_sent=self.chan.bytes_sent,
                       bytes_received=self.chan.bytes_received)
        self.stopped = rec
        return rec
