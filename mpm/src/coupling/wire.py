"""Wire format `mpm-xpbd-wire-v1` between the MPM driver and the XPBD service process (P2-U3;
host I/O, standard library + NumPy only: importing this module loads neither Taichi nor mpm).

    frame   = b"MXW1" | uint32 big-endian header length L | header (L bytes) | payload
    header  = canonical JSON object with exactly the keys
              wire ("mpm-xpbd-wire-v1"), mode, seq, meta (object), arrays (list)
    arrays  = [{name, dtype in {"<f8", "<i8", "|b1"}, shape, nbytes, sha256}], the payload is
              their bytes back to back in this order (C order); nothing else is in the payload

One frame is one Connection.send_bytes / recv_bytes (bounded: recv_bytes(max_frame)). The
receiver checks the magic, the header length, strict JSON (duplicate keys and non-finite
numbers rejected), that the header is exactly its canonical encoding (re-encoded and compared
byte for byte, so e.g. an integer -0 cannot change silently), the exact key sets, the wire
version, the mode, seq == the next expected value of that direction (lost, repeated or
reordered frames are refused), at most MAX_ARRAYS arrays, each array's dtype, shape, length
(checked before any allocation) and SHA-256, that every byte of a "|b1" array is 0 or 1 , and that
the payload holds exactly the arrays. No pickle anywhere. Limits and the big-endian length follow XPBD's interface specification recommendation
(16 MiB frame, 1 MiB header, 16 arrays); a send has its own watchdog (Channel.send timeout).

Canonical JSON (the digest rule written in XPBD's registered entry, agreed for the registration
and the receipt digests): json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=True, allow_nan=False), JSON-native values only (dict with str keys, list, str,
int, finite float, bool, None; no tuple, no NumPy scalar), then UTF-8, SHA-256, lowercase hex.
Floats use CPython's JSON form (repr, shortest round trip); both sides are CPython, so values
round-trip exactly, -0.0 included.
"""
import hashlib
import json
import math
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

WIRE = "mpm-xpbd-wire-v1"
MAGIC = b"MXW1"
DTYPES = ("<f8", "<i8", "|b1")
HEADER_KEYS = frozenset(("wire", "mode", "seq", "meta", "arrays"))
ARRAY_KEYS = frozenset(("name", "dtype", "shape", "nbytes", "sha256"))
TO_SERVICE = ("REGISTER", "OPEN", "PEER", "IMPULSE", "STOP")
FROM_SERVICE = ("HELLO", "BOUND", "OFFER", "PEER_OK", "RECEIPT", "ERROR", "BYE")
MAX_FRAME = 16 * 2 ** 20
MAX_HEADER = 2 ** 20
MAX_ARRAYS = 16
DIGEST_RULE = ("SHA-256 (lowercase hex) of the UTF-8 bytes of json.dumps(value, sort_keys=True, separators=(',', ':'), "
               "ensure_ascii=True, allow_nan=False); value JSON-native only (dict with str keys, list, str, int, "
               "finite float, bool, None), floats in CPython's JSON form")


class WireError(ValueError):
    """A frame or JSON text that breaks the wire rules (protocol error, never retried)."""


class WireTimeout(TimeoutError):
    """No frame within the watchdog time."""


class WireClosed(ConnectionError):
    """The other end closed the connection or its process ended."""


# ------------------------------------------------------------------ JSON
def check_json_native(value, where="value"):
    t = type(value)
    if t is dict:
        for k, v in value.items():
            if type(k) is not str:
                raise WireError(f"{where}: JSON object keys must be str, got {k!r}")
            check_json_native(v, f"{where}.{k}")
    elif t is list:
        for i, v in enumerate(value):
            check_json_native(v, f"{where}[{i}]")
    elif t is float:
        if not math.isfinite(value):
            raise WireError(f"{where}: non-finite number {value!r}")
    elif value is None or t in (str, bool, int):
        pass
    else:
        raise WireError(f"{where}: {t.__name__} is not a JSON-native value")
    return value


def canonical_json(value) -> str:
    check_json_native(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def json_digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise WireError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _constant(name):
    raise WireError(f"non-finite JSON constant {name}")


def _float(text):
    v = float(text)
    if not math.isfinite(v):
        raise WireError(f"JSON number {text} overflows to {v}")
    return v


def strict_loads(text):
    """json.loads that refuses duplicate keys, NaN / Infinity and numbers overflowing to inf."""
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8")
        except UnicodeDecodeError as e:
            raise WireError(f"JSON text is not UTF-8: {e}") from e
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    except WireError:
        raise
    except (ValueError, TypeError, RecursionError) as e:
        raise WireError(f"invalid JSON: {e}") from e


def is_canonical(text: str) -> bool:
    """The text is exactly the canonical form of what it parses to."""
    return canonical_json(strict_loads(text)) == text


# ------------------------------------------------------------------ frames
@dataclass(frozen=True)
class Frame:
    mode: str
    seq: int
    meta: dict
    arrays: dict
    nbytes: int


def _sha(b) -> str:
    return hashlib.sha256(b).hexdigest()


def encode(mode, seq, meta, arrays=None) -> bytes:
    """Arrays must already have one of the wire dtypes (no silent cast)."""
    if mode not in TO_SERVICE + FROM_SERVICE:
        raise WireError(f"unknown mode {mode!r}")
    if type(seq) is not int or seq < 0:
        raise WireError("seq must be a nonnegative int")
    if type(meta) is not dict:
        raise WireError("meta must be a JSON object")
    entries, raws = [], []
    for name, a in (arrays or {}).items():
        a = np.asarray(a)
        if a.dtype.str not in DTYPES:
            raise WireError(f"array {name!r}: dtype {a.dtype.str} is not one of {DTYPES}")
        raw = np.ascontiguousarray(a).tobytes()
        if a.dtype.str == "|b1":
            _check_bool_bytes(name, raw)                       # never send a bool byte other than 0 / 1
        entries.append(dict(name=str(name), dtype=a.dtype.str, shape=[int(n) for n in a.shape], nbytes=len(raw),
                            sha256=_sha(raw)))
        raws.append(raw)
    if len(entries) > MAX_ARRAYS:
        raise WireError(f"{len(entries)} arrays exceed {MAX_ARRAYS}")
    head = canonical_json(dict(wire=WIRE, mode=mode, seq=seq, meta=meta, arrays=entries)).encode("ascii")
    if len(head) > MAX_HEADER:
        raise WireError(f"header of {len(head)} bytes exceeds {MAX_HEADER}")
    return MAGIC + struct.pack(">I", len(head)) + head + b"".join(raws)


def _check_bool_bytes(name, raw):
    """A "|b1" array is one byte per value, 0 or 1 only : NumPy would read any other
    byte as True, so a byte of 2 with a matching SHA-256 must still be refused (P2-U4 prefix fix A)."""
    u = np.frombuffer(raw, dtype=np.uint8)
    if u.size and int(u.max()) > 1:
        bad = int(np.flatnonzero(u > 1)[0])
        raise WireError(f"array {name!r}: |b1 byte {int(u[bad])} at element {bad} is not 0 or 1")


def decode(buf, *, expect_seq=None, modes=None) -> Frame:
    buf = bytes(buf)
    if len(buf) < 8 or buf[:4] != MAGIC:
        raise WireError("bad magic or truncated frame")
    (L,) = struct.unpack(">I", buf[4:8])
    if L > MAX_HEADER or 8 + L > len(buf):
        raise WireError(f"header length {L} does not fit the frame of {len(buf)} bytes")
    head = strict_loads(buf[8:8 + L])
    if type(head) is not dict or set(head) != HEADER_KEYS:
        raise WireError(f"header keys {sorted(head) if type(head) is dict else type(head).__name__} != {sorted(HEADER_KEYS)}")
    if canonical_json(head).encode("ascii") != buf[8:8 + L]:
        raise WireError("the header is not in canonical form")
    if head["wire"] != WIRE:
        raise WireError(f"wire version {head['wire']!r}, expected {WIRE!r}")
    mode, seq, meta, entries = head["mode"], head["seq"], head["meta"], head["arrays"]
    if mode not in TO_SERVICE + FROM_SERVICE or (modes is not None and mode not in modes):
        raise WireError(f"unexpected mode {mode!r} (expected one of {tuple(modes) if modes else 'the known modes'})")
    if type(seq) is not int or seq < 0:
        raise WireError("seq must be a nonnegative int")
    if expect_seq is not None and seq != expect_seq:
        raise WireError(f"seq {seq}, expected {expect_seq} (lost, repeated or reordered frame)")
    if type(meta) is not dict or type(entries) is not list:
        raise WireError("meta must be an object and arrays a list")
    if len(entries) > MAX_ARRAYS:
        raise WireError(f"{len(entries)} arrays exceed {MAX_ARRAYS}")
    payload = memoryview(buf)[8 + L:]
    arrays, off = {}, 0
    for e in entries:
        if type(e) is not dict or set(e) != ARRAY_KEYS:
            raise WireError(f"array entry keys differ from {sorted(ARRAY_KEYS)}")
        name, dt, shape, nb, sha = e["name"], e["dtype"], e["shape"], e["nbytes"], e["sha256"]
        if type(name) is not str or not name or name in arrays:
            raise WireError(f"array name {name!r} empty or repeated")
        if dt not in DTYPES:
            raise WireError(f"array {name!r}: dtype {dt!r} not allowed")
        if type(shape) is not list or any(type(n) is not int or n < 0 for n in shape):
            raise WireError(f"array {name!r}: bad shape {shape!r}")
        count = math.prod(shape)                               # Python ints: no overflow 
        if type(nb) is not int or nb != count * np.dtype(dt).itemsize:
            raise WireError(f"array {name!r}: nbytes {nb!r} differs from shape x itemsize")
        if off + nb > len(payload):
            raise WireError(f"array {name!r}: payload too short")
        raw = payload[off:off + nb]
        if _sha(raw) != sha:
            raise WireError(f"array {name!r}: SHA-256 differs")
        if dt == "|b1":
            _check_bool_bytes(name, raw)                       # P2-U4 prefix fix A: digest is no range check
        try:
            a = np.frombuffer(raw, dtype=np.dtype(dt), count=count).reshape(shape).copy()
        except ValueError as ex:
            raise WireError(f"array {name!r}: {ex}") from ex
        a.setflags(write=False)
        arrays[name] = a
        off += nb
    if off != len(payload):
        raise WireError(f"payload has {len(payload) - off} bytes beyond the declared arrays")
    return Frame(mode, seq, meta, arrays, len(buf))


class Channel:
    """One end of a duplex Connection: per-direction seq counters, watchdog, byte counts.
    A frame longer than max_frame is refused by recv_bytes after it has read only the length
    header, so the stream is out of step: the channel is then `broken`  and every later recv raises WireClosed; the owner stops without a STOP/BYE exchange."""

    def __init__(self, conn, *, max_frame=MAX_FRAME, recv_modes=None):
        self.conn = conn
        self.max_frame = int(max_frame)
        self.recv_modes = recv_modes
        self.sent = self.received = 0
        self.bytes_sent = self.bytes_received = 0
        self.last_sent_bytes = self.last_received_bytes = 0
        self.last_wait_s = None
        self.broken = None

    def send(self, mode, meta, arrays=None, *, timeout=None, on_stall=None):
        """timeout: send watchdog (a peer that stops reading blocks send_bytes once the pipe buffer is full):
        on expiry on_stall() runs (it must end the other side, e.g. terminate the process or close its end),
        which unblocks the send with an error; the send then raises WireClosed naming the watchdog."""
        buf = encode(mode, self.sent, meta, arrays)
        if len(buf) > self.max_frame:
            raise WireError(f"frame of {len(buf)} bytes exceeds max_frame_bytes = {self.max_frame}")
        fired = []
        timer = None
        if timeout is not None and on_stall is not None:
            def stall():
                fired.append(time.monotonic())
                on_stall()
            timer = threading.Timer(float(timeout), stall)
            timer.daemon = True
            timer.start()
        try:
            self.conn.send_bytes(buf)
        except (BrokenPipeError, ConnectionError, EOFError, OSError) as e:
            why = f"send watchdog: {mode} not taken within {timeout} s; " if fired else ""
            raise WireClosed(f"{why}send {mode}: {type(e).__name__}: {e}") from e
        finally:
            if timer is not None:
                timer.cancel()
        self.sent += 1
        self.bytes_sent += len(buf)
        self.last_sent_bytes = len(buf)
        return len(buf)

    def recv(self, timeout, modes=None, *, strict_seq=True, on_stall=None) -> Frame:
        """poll(timeout) waits for the start of a frame; once it has started, recv_bytes may still block on a
        frame that never completes (a half frame): with on_stall a read watchdog (the time left of `timeout`,
        at least 1 s) runs on_stall(), which must end the other side, and the read fails as WireClosed naming
        the watchdog ."""
        if self.broken is not None:
            raise WireClosed(f"the channel is out of step after: {self.broken}")
        t0 = time.monotonic()
        try:
            ready = self.conn.poll(timeout)
        except (ConnectionError, EOFError, OSError) as e:
            raise WireClosed(f"poll: {type(e).__name__}: {e}") from e
        if not ready:
            raise WireTimeout(f"no frame within {timeout} s (expected {tuple(modes) if modes else 'any'})")
        fired, timer = [], None
        if on_stall is not None:
            def stall():
                fired.append(time.monotonic())
                on_stall()
            timer = threading.Timer(max(1.0, float(timeout) - (time.monotonic() - t0)), stall)
            timer.daemon = True
            timer.start()
        try:
            buf = self.conn.recv_bytes(self.max_frame)
        except EOFError as e:
            why = "read watchdog: a frame did not complete in time; " if fired else ""
            raise WireClosed(f"{why}the other end closed the connection") from e
        except OSError as e:                                   # message longer than maxlength, or a broken pipe
            if fired:
                raise WireClosed(f"read watchdog: a frame did not complete in time; {type(e).__name__}: {e}") from e
            self.broken = f"recv: {type(e).__name__}: {e} (max_frame_bytes = {self.max_frame})"
            raise WireError(self.broken) from e
        finally:
            if timer is not None:
                timer.cancel()
        expect = self.received
        self.received += 1                                     # a frame was consumed, valid or not
        self.bytes_received += len(buf)
        self.last_received_bytes = len(buf)
        fr = decode(buf, expect_seq=expect if strict_seq else None, modes=modes or self.recv_modes)
        self.last_wait_s = time.monotonic() - t0
        return fr

    def close(self):
        try:
            self.conn.close()
        except OSError:
            pass
