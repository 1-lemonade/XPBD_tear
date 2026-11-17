"""Endpoint batch archive and MPM-side batch re-feed (P2-U2; host I/O only).

<dir>/batches.npz     per window that produced a batch: window_index, cloth_version (W, 3), t_start /
                      t_end (snapshot clock = the LoadBatch times), t_start_ns / t_end_ns (MPM window
                      clock), vertex_ids (W, V), impulse (W, V, 3) native frame N s, J_material /
                      J_to_cloth / K_material (native), counts (W, 4) [constrained, penetrated among them,
                      new misses, new overflows], delivered
<dir>/snapshot0.npz   the first snapshot (native positions, velocity, triangles, IDs, pinned, version,
                      time, dt): with the manifest's h, dt, stream and source it rebuilds the prescribed peer
<dir>/ledger.csv      the endpoint's per-window measures (header = column names; a window stopped before
                      its substeps has only its stop columns)
<dir>/provenance.json registration, chain start and final, stop state, warnings
<dir>/manifest.json   written last: format, complete (no stop, no pending or awaited window, every
                      window delivered and acknowledged), stop reason, sizes and SHA-256 of the other
                      files, stream, source, window, batch IDs, batch and state digests, receipt digests
                      (canonical JSON of the receipt as received, coupling.wire.DIGEST_RULE) and receipt
                      content digests (by stable ID, row-order independent), acknowledged flags, the
                      window awaiting its receipt at a stop, the frozen registration digest, digest
                      rules, run anchor (run directory, SHA-256 of its config.json and run_info.json)
P2-U3 additions (absent keys read as before): acknowledged, receipt_content_digests,
awaiting_receipt, registration_digest in the manifest; registration_frozen and handshake in the
provenance. refeed() sends the acknowledged windows (delivered ones when the archive has no
acknowledged flags) and compares receipt content digests.
One topology per archive: the topology_version and the vertex ID set are constant. verify()
recomputes file digests, batch digests and the chain (with the recorded state digests), and the
ledger sums; with work= it says whether the code digest still matches the sources. refeed()
sends the delivered batches to a fresh prescribed peer rebuilt from the archive (or a given
one) and checks every receipt digest. It re-feeds batches; it does not re-run MPM (that needs the
initial particle state and the solver).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import wire
from .contracts import SCHEMA, ImpulseWindow, LoadBatch, SourceIdentity
from .endpoint import DIGEST_RULE, batch_digest, chain_next, chain_start, code_hashes, digest_of, sha256_file
from .prescribed import PrescribedPeer, receipt_digest

FORMAT = "mpm-endpoint-batches-v1"
REVISION = "u3-ack-1"           # P2-U3 semantics : absent = U2 (receipt_digests are content digests)
U3_MANIFEST_KEYS = ("acknowledged", "receipt_content_digests", "awaiting_receipt", "registration_digest",
                    "receipt_digest_rule")
NPZ, SNAP0, LEDGER, PROV, MANIFEST = "batches.npz", "snapshot0.npz", "ledger.csv", "provenance.json", "manifest.json"


def write(out_dir, ep, run_dir=None) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"archive {out} exists")
    recs = ep.records
    if ep.snapshot0 is None:
        raise RuntimeError("archive: no snapshot was read (nothing to archive)")
    ids0 = np.sort(ep.snapshot0["vertex_ids"])
    tv0 = int(ep.snapshot0["version"][2])
    if any(r["cloth_version"][2] != tv0 or not np.array_equal(np.sort(r["vertex_ids"]), ids0) for r in recs):
        raise ValueError("archive: one topology per archive (topology_version and vertex IDs constant)")
    V = len(ids0)
    out.mkdir(parents=True)
    if not recs:                                               # stopped before the first window's substeps
        empty = dict(window_index=(0,), cloth_version=(0, 3), t_start=(0,), t_end=(0,), t_start_ns=(0,),
                     t_end_ns=(0,), vertex_ids=(0, V), impulse=(0, V, 3), J_material=(0, 3), J_to_cloth=(0, 3),
                     K_material=(0, 3), counts=(0, 4), delivered=(0,))
        types = dict(window_index=np.int64, cloth_version=np.int64, t_start_ns=np.int64, t_end_ns=np.int64,
                     vertex_ids=np.int64, counts=np.int64, delivered=bool)
        arr = {k: np.zeros(sh, types.get(k, np.float64)) for k, sh in empty.items()}
    else:
        arr = dict(window_index=np.array([r["window_index"] for r in recs], np.int64),
                   cloth_version=np.array([r["cloth_version"] for r in recs], np.int64),
                   t_start=np.array([r["t_start"] for r in recs], np.float64),
                   t_end=np.array([r["t_end"] for r in recs], np.float64),
                   t_start_ns=np.array([r["t_start_ns"] for r in recs], np.int64),
                   t_end_ns=np.array([r["t_end_ns"] for r in recs], np.int64),
                   vertex_ids=np.stack([r["vertex_ids"] for r in recs]).astype(np.int64),
                   impulse=np.stack([r["impulse"] for r in recs]).astype(np.float64),
                   J_material=np.stack([r["J_material"] for r in recs]), J_to_cloth=np.stack([r["J_to_cloth"] for r in recs]),
                   K_material=np.stack([r["K_material"] for r in recs]), counts=np.stack([r["counts"] for r in recs]),
                   delivered=np.array([r["delivered"] for r in recs], bool))
    for name, data in ((NPZ, arr), (SNAP0, ep.snapshot0)):
        with (out / name).open("xb") as fh:
            np.savez(fh, **data)
        with np.load(out / name, allow_pickle=False) as z:
            for k, a in data.items():
                a = np.asarray(a)
                if not (z[k].dtype == a.dtype and z[k].shape == a.shape and z[k].tobytes() == a.tobytes()):
                    raise RuntimeError(f"archive: {name} roundtrip differs for {k}")   # bytes: NaN batches too
    cols = list(dict.fromkeys(c for row in ep.ledger for c in row))
    with (out / LEDGER).open("x", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in ep.ledger:
            w.writerow(["" if c not in row or row[c] is None else
                        repr(float(row[c])) if isinstance(row[c], (float, np.floating)) else row[c] for c in cols])
    with (out / PROV).open("x", encoding="utf-8") as fh:
        json.dump(ep.provenance(), fh, indent=1, default=str)
    anchor = None
    if run_dir is not None:
        rd = Path(run_dir)
        anchor = dict(run_dir=str(rd), **{f"{n}_sha256": sha256_file(rd / n) for n in ("config.json", "run_info.json")
                                         if (rd / n).exists()})
    pending = ep.k if ep.phase == "received" else (ep.stopped or {}).get("pending_window")   # read, not advanced
    awaiting = ep.awaiting_receipt if ep.phase == "stopped" else (ep.k - 1 if ep.phase == "delivered" else None)
    acked = [bool(r.get("acknowledged")) for r in recs]
    complete = (ep.stopped is None and pending is None and awaiting is None and all(r["delivered"] for r in recs)
                and all(acked))
    man = dict(format=FORMAT, revision=REVISION, schema=SCHEMA, complete=complete, pending_window=pending,
               awaiting_receipt=awaiting,
               stop_reason=None if ep.stopped is None else ep.stopped["reason"], windows=len(recs),
               windows_delivered=ep.k, vertices=V, stream_id=ep.stream_id, source=ep.source.as_dict(), n_sub=ep.n_sub,
               dt=ep.dt, H_ns=ep.H_ns, h=ep.H, batch_ids=[r["batch_id"] for r in recs],
               batch_digests=[r["digest"] for r in recs], state_digests=[r["state_digest"] for r in recs],
               receipt_digests=[row.get("receipt_digest") for row in ep.ledger[:len(recs)]],
               receipt_content_digests=[row.get("receipt_content_digest") for row in ep.ledger[:len(recs)]],
               acknowledged=acked, registration_digest=ep.registration_digest, receipt_digest_rule=wire.DIGEST_RULE,
               delivered=[bool(r["delivered"]) for r in recs], chain_final=ep.chain, digest_rule=DIGEST_RULE,
               run_anchor=anchor, coordinate_frame="world (XPBD native)", units="SI",
               impulse="material -> cloth, N s, every active vertex by stable ID (pinned and zero rows included)",
               files={n: dict(bytes=(out / n).stat().st_size, sha256=sha256_file(out / n))
                      for n in (NPZ, SNAP0, LEDGER, PROV)})
    with (out / MANIFEST).open("x", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    return dict(dir=str(out), manifest_sha256=sha256_file(out / MANIFEST), windows=len(recs), vertices=V,
                complete=complete)


def read(dir_) -> dict:
    d = Path(dir_)
    man = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    with np.load(d / NPZ, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(d / SNAP0, allow_pickle=False) as z:
        snap0 = {k: z[k] for k in z.files}
    with (d / LEDGER).open(newline="") as fh:
        ledger = list(csv.DictReader(fh))
    return dict(manifest=man, arrays=arrays, snapshot0=snap0, ledger=ledger,
                provenance=json.loads((d / PROV).read_text(encoding="utf-8")))


def verify(dir_, work=None) -> dict:
    """Raises ValueError on any inconsistency; returns a report."""
    d = Path(dir_)
    a = read(d)
    man, arr, led, prov = a["manifest"], a["arrays"], a["ledger"], a["provenance"]
    for n, rec in man["files"].items():
        if sha256_file(d / n) != rec["sha256"]:
            raise ValueError(f"archive: {n} differs from its manifest digest")
    reg = prov["registration"]
    if digest_of(reg["code_files"]) != reg["code_digest"]:
        raise ValueError("archive: code digest does not follow from the file list")
    chain = chain_start(reg["mpm_run_id"], reg["code_digest"])
    if chain != prov["chain_start"]:
        raise ValueError("archive: chain start does not follow from the run id and code digest")
    W = man["windows"]
    if not np.array_equal(arr["window_index"], np.arange(W)):
        raise ValueError("archive: window indices are not 0 .. W-1")
    if W and (arr["t_start"][0] != 0.0 or not np.array_equal(arr["t_start"][1:], arr["t_end"][:-1])):
        raise ValueError("archive: window times do not start at 0 or do not join (t_start_k == t_end_k-1)")
    if list(map(bool, arr["delivered"])) != list(man["delivered"]):
        raise ValueError("archive: delivered flags differ between the manifest and the batches")
    u3 = man.get("revision") == REVISION
    if man.get("revision") not in (None, REVISION):
        raise ValueError(f"archive: unknown revision {man.get('revision')!r}")
    if u3:                                     # a U3 manifest must carry its U3 keys 
        missing = [k for k in U3_MANIFEST_KEYS if k not in man] + [k for k in ("registration_frozen", "handshake")
                                                                    if k not in prov]
        if missing:
            raise ValueError(f"archive: revision {REVISION} lacks {missing}")
    acked = man["acknowledged"] if u3 else None
    if man["complete"] != (man["stop_reason"] is None and man.get("pending_window") is None
                           and man.get("awaiting_receipt") is None and all(man["delivered"])
                           and (acked is None or all(acked))):
        raise ValueError("archive: complete flag is inconsistent with the stop, pending / awaited window, delivered "
                         "and acknowledged flags")
    if acked is not None:
        if len(acked) != W or any(a and not d for a, d in zip(acked, man["delivered"])):
            raise ValueError("archive: acknowledged flags do not fit the delivered windows")
        if hashlib.sha256(prov["registration_frozen"].encode("utf-8")).hexdigest() != man["registration_digest"]:
            raise ValueError("archive: the frozen registration does not give the registration digest")
        if not wire.is_canonical(prov["registration_frozen"]):              # re-encoded, byte for byte 
            raise ValueError("archive: the frozen registration is not in canonical form")
        frozen = wire.strict_loads(prov["registration_frozen"])
        if frozen.get("peer") is not None or dict(reg, peer=None) != frozen:     # differ only in the late peer (S9)
            raise ValueError("archive: the registration differs from the frozen registration beyond the peer binding")
    lim = float(reg["stop"]["impulse_rel"])
    n_sub = int(man["n_sub"])
    for k in range(W):
        row = led[k]
        if int(row["step_after"]) - int(row["step_before"]) != n_sub:
            raise ValueError(f"archive: window {k} did not advance {n_sub} substeps by the ledger")
        if (row["receipt_digest"] or None) != man["receipt_digests"][k]:
            raise ValueError(f"archive: receipt digest of window {k} differs between the ledger and the manifest")
        if acked is not None:
            if (row.get("receipt_content_digest") or None) != man["receipt_content_digests"][k]:
                raise ValueError(f"archive: receipt content digest of window {k} differs between ledger and manifest")
            if (row.get("acknowledged") == "True") != bool(acked[k]) or bool(acked[k]) != (man["receipt_digests"][k]
                                                                                           is not None):
                raise ValueError(f"archive: acknowledged flag of window {k} differs from the ledger or its receipt")
        if man["delivered"][k]:
            dJ = np.array([float(row[f"dJG_{c}"]) for c in "xyz"])
            sG = np.array([float(row[f"sumG_{c}"]) for c in "xyz"])
            nJ = np.linalg.norm(dJ)
            rel = np.linalg.norm(sG - dJ) / nJ if nJ > 0 else (0.0 if not sG.any() else np.inf)
            if not rel <= lim:
                raise ValueError(f"archive: delivered window {k} has batch sum vs contact sink {rel:.3g} > {lim:.3g}")
    for k in range(W):
        dg = batch_digest(arr["window_index"][k], man["batch_ids"][k], arr["cloth_version"][k], arr["t_start"][k],
                          arr["t_end"][k], arr["vertex_ids"][k], arr["impulse"][k])
        if dg != man["batch_digests"][k] or dg != led[k]["batch_digest"]:
            raise ValueError(f"archive: batch digest of window {k} does not match")
        if led[k]["state_digest"] != man["state_digests"][k]:
            raise ValueError(f"archive: state digest of window {k} does not match")
        chain = chain_next(chain, dg, led[k]["state_digest"])
        if chain != led[k]["chain"]:
            raise ValueError(f"archive: chain broken at window {k}")
        for c, i in zip("xyz", range(3)):
            a_, b_ = float(led[k][f"sumG_native_{c}"]), float(arr["impulse"][k][:, i].sum())
            if a_ != b_ and not (np.isnan(a_) and np.isnan(b_)):              # a non-finite (S1) batch keeps NaN
                raise ValueError(f"archive: ledger sum of window {k} differs from the batch")
    if chain != man["chain_final"] or chain != prov["chain_final"]:
        raise ValueError("archive: final chain differs")
    rep = dict(windows=man["windows"], delivered=int(sum(man["delivered"])), complete=man["complete"],
               acknowledged=None if acked is None else int(sum(acked)), awaiting_receipt=man.get("awaiting_receipt"),
               stop_reason=man["stop_reason"], pending_window=man.get("pending_window"), chain_final=chain,
               files_ok=True, anchor_ok=None,
               covers="file, batch and state digests, chain, window indices and times, substeps per window, batch "
                      "sum vs contact sink of delivered windows, ledger sums, receipt digests and content digests, "
                      "delivered / acknowledged / complete flags, frozen registration digest; not a proof against a "
                      "forger who rewrites every file consistently (the manifest has no outside anchor)")
    anchor = man.get("run_anchor")
    if anchor and Path(anchor["run_dir"]).is_dir():
        rd = Path(anchor["run_dir"])
        rep["anchor_ok"] = all(sha256_file(rd / n[:-7]) == v for n, v in anchor.items() if n.endswith("_sha256"))
    if work is not None:                                      # every registered file, scenes included 
        work = Path(work)
        now = code_hashes(work, [])
        for key in reg["code_files"]:
            if key not in now and (work / key).is_file():
                now[key] = sha256_file(work / key)
        rep["code_unchanged"] = digest_of(now) == reg["code_digest"]
    return rep


def peer_from(dir_, shuffle_rows=None) -> PrescribedPeer:
    """The prescribed peer of the archived stream, rebuilt from snapshot0 and the manifest."""
    a = read(dir_)
    s, man = a["snapshot0"], a["manifest"]
    v = s["velocity"]
    if not (v == v[0]).all() or not s["pinned"].all():
        raise ValueError("archive: the first snapshot is not a prescribed uniform-velocity surface")
    return PrescribedPeer(s["positions"], s["triangles"], s["vertex_ids"], s["triangle_ids"], v[0], h=man["h"],
                          stream_id=man["stream_id"], expected_source=SourceIdentity(**man["source"]),
                          epoch=int(s["version"][0]), topology_version=int(s["version"][2]), shuffle_rows=shuffle_rows)


def refeed(dir_, peer=None) -> list:
    """Send the acknowledged batches (delivered ones in an archive without acknowledged flags) to a fresh
    peer (rebuilt from the archive unless given); every receipt content digest (by stable ID, row-order
    independent) must equal the archived one."""
    a = read(dir_)
    man, arr = a["manifest"], a["arrays"]
    peer = peer_from(dir_) if peer is None else peer
    src = SourceIdentity(**man["source"])
    if man.get("revision") == REVISION and "acknowledged" not in man:
        raise ValueError(f"refeed: revision {REVISION} lacks the acknowledged flags")
    acked = man["acknowledged"] if man.get("revision") == REVISION else None
    out = []
    for k in range(man["windows"]):
        if not man["delivered"][k] or (acked is not None and not acked[k]):
            break
        offer = peer.open_window(h=man["h"], particle_dt=man["dt"])
        S = offer.snapshot
        if offer.window_index != k or tuple(arr["cloth_version"][k]) != (S.epoch, S.step_index, S.topology_version) \
                or float(arr["t_start"][k]) != float(S.time):
            raise ValueError(f"refeed: the peer's window {k} differs from the archive")
        batch = LoadBatch(stream_id=man["stream_id"], source=src,
                          cloth_version=tuple(int(v) for v in arr["cloth_version"][k]), batch_id=man["batch_ids"][k],
                          t_start=float(arr["t_start"][k]), t_end=float(arr["t_end"][k]),
                          vertex_ids=arr["vertex_ids"][k], impulse=arr["impulse"][k])
        receipt = peer.consume(ImpulseWindow(SCHEMA, k, batch))
        if acked is not None:
            if receipt_digest(receipt) != man["receipt_content_digests"][k]:
                raise ValueError(f"refeed: receipt content of window {k} differs from the archived run")
        elif man["receipt_digests"][k] is not None and receipt.get("digest") != man["receipt_digests"][k]:
            raise ValueError(f"refeed: receipt of window {k} differs from the archived run")
        out.append(receipt)
    return out
