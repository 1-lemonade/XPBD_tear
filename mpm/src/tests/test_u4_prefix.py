"""P2-U4 prefix fixes : the three counterexamples of
the U4 entry fact check  as
regression tests, plus the receipt-log conditions XPBD asked to tell apart .

  A  wire: a |b1 byte other than 0 / 1 is refused even when its SHA-256 matches 
  B  reconcile: row by row (window index consecutive from 0, batch id of that window, at most the one awaited
     extra row); missing log, missing final.json and a receipt-log write failure reported separately
  C  U13 fixed topology: an offer or a receipt whose topology_version is not the initial one stops the run;
     the endpoint itself keeps the general "not decreasing" rule (U2 / U3)

Two local processes, n24, <= 5 windows (A tier)."""
import copy
import dataclasses
import hashlib
import json
import shutil

import numpy as np
import pytest

import test_adapter as ta
from coupling import EndpointStop, archive, wire, xpbd_service
from coupling.driver import DriverError, check_fixed_topology, reconcile
import test_endpoint as te

needs_xpbd = ta.needs_xpbd


# ------------------------------------------------------------------ A: |b1 value range
def _bool_frame_with_byte(value):
    good = wire.encode("OFFER", 0, {}, dict(pinned=np.array([True, True, False])))
    L = int.from_bytes(good[4:8], "big")
    head = json.loads(good[8:8 + L])
    payload = bytearray(good[8 + L:])
    payload[1] = value
    head["arrays"][0]["sha256"] = hashlib.sha256(bytes(payload)).hexdigest()   # digest made consistent
    h = wire.canonical_json(head).encode("ascii")
    return wire.MAGIC + len(h).to_bytes(4, "big") + h + bytes(payload)


def test_a_bool_bytes_other_than_0_and_1_are_refused():
    ok = wire.decode(_bool_frame_with_byte(1))
    assert ok.arrays["pinned"].tolist() == [True, True, False]
    for value in (2, 255):                                  # U3 code: accepted, read as True 
        with pytest.raises(wire.WireError, match=r"\|b1 byte"):
            wire.decode(_bool_frame_with_byte(value))
    bad = np.array([1, 2, 0], np.uint8).view(np.bool_)      # the sender refuses such an array too
    with pytest.raises(wire.WireError, match=r"\|b1 byte 2"):
        wire.encode("OFFER", 0, {}, dict(pinned=bad))
    empty = wire.decode(wire.encode("OFFER", 0, {}, dict(pinned=np.zeros(0, bool))))
    assert empty.arrays["pinned"].shape == (0,)


# ------------------------------------------------------------------ B: reconcile row by row
@pytest.fixture(scope="module")
def timeout_run(tmp_path_factory):
    """A real two-process stop while waiting for a receipt: MPM acknowledged window 0, XPBD committed 0 and 1."""
    tmp = tmp_path_factory.mktemp("u4_timeout")
    sim, ep = ta.build_h()
    r, drv = ta.drive(ep, 5, tmp, **copy.deepcopy(ta.FAULTS["receipt_timeout"]))
    archive.write(tmp / "endpoint", ep)
    return tmp


def _variant(src, tmp, edit=None, drop=None):
    shutil.copytree(src / "endpoint", tmp / "endpoint")
    svc = tmp / "svc"
    shutil.copytree(src / "coupling" / "xpbd_service", svc)
    if drop:
        (svc / drop).unlink()
    if edit:
        lines = (svc / "receipts.jsonl").read_text().splitlines()
        (svc / "receipts.jsonl").write_text("".join(x + "\n" for x in edit([json.loads(x) for x in lines], lines)))
    return reconcile(tmp / "endpoint", svc)


def _rows(rows):
    return [wire.canonical_json(r) for r in rows]


def _set(i, **kw):
    def f(rows, _):
        rows[i].update(kw)
        return _rows(rows)
    return f


VARIANTS = {   # name: (edit, drop, consistent, issue pattern, log_condition)
    "extra_row_window_99": (_set(1, window_index=99), None, False, "row 1: window_index 99", "final"),
    "extra_row_window_bool": (_set(1, window_index=True), None, False, "row 1: window_index True", "final"),
    "rows_swapped": (lambda rows, _: _rows([rows[1], rows[0]]), None, False, "row 0: window_index 1", "final"),
    "prefix_batch_id": (_set(0, batch_id="other/w0"), None, False, "row 0: batch_id", "final"),
    "extra_row_batch_id": (_set(1, batch_id="other/w1"), None, False, "row 1: batch_id", "final"),
    "two_extra_rows": (lambda rows, _: _rows(rows + [dict(rows[1], window_index=2)]), None, False,
                       "MPM delivered only 2 windows", "final"),
    "unreadable_row": (lambda rows, lines: [lines[0], lines[1][:-1]], None, False, "not strict JSON", "final"),
    "no_final": (None, "final.json", True, None, "no_final"),
    "log_missing": (None, "receipts.jsonl", False, "no receipt log", "log_missing"),
}


@needs_xpbd
def test_b_reconcile_checks_every_row(timeout_run, tmp_path):
    base = _variant(timeout_run, tmp_path / "base")
    assert base["consistent"] and base["extra_xpbd_windows"] == 1 and base["awaiting_receipt"] == 1, base
    assert base["note"] and base["log_condition"] == "final" and base["issues"] == []
    for name, (edit, drop, consistent, pattern, condition) in VARIANTS.items():
        rc = _variant(timeout_run, tmp_path / name, edit, drop)
        assert rc["consistent"] is consistent and rc["log_condition"] == condition, (name, rc)
        if pattern is not None:
            assert any(pattern in x for x in rc["issues"]), (name, rc["issues"])
        if name.startswith("extra_row") or name == "rows_swapped":
            assert rc["note"] is None, (name, rc["note"])              # no claim about a wrong awaited row


@needs_xpbd
def test_b_final_count_must_match_the_log(timeout_run, tmp_path):
    shutil.copytree(timeout_run / "endpoint", tmp_path / "endpoint")
    svc = tmp_path / "svc"
    shutil.copytree(timeout_run / "coupling" / "xpbd_service", svc)
    final = json.loads((svc / "final.json").read_text())
    final["committed_windows"] = 3
    (svc / "final.json").write_text(json.dumps(final))
    rc = reconcile(tmp_path / "endpoint", svc)
    assert not rc["consistent"] and any("final.json says 3 committed" in x for x in rc["issues"]), rc


@needs_xpbd
def test_b_receipt_log_write_failure_is_its_own_condition(tmp_path):
    sim, ep = ta.build_h()
    r, drv = ta.drive(ep, 5, tmp_path, fault=dict(log_fail_at=1))
    rows, final = ta.service_log(tmp_path)
    assert ep.acknowledged() == 1 and ep.awaiting_receipt == 1, ep.stopped     # committed, receipt never sent
    assert "commit of window 1 is unknown" in ep.stopped["reason"], ep.stopped["reason"]
    assert final["committed_windows"] == 2 and len(rows) == 1 and final["log_errors"][0]["window_index"] == 1
    assert r["xpbd_stop"]["bye"]["log_errors"] == final["log_errors"]
    archive.write(tmp_path / "endpoint", ep)
    rc = reconcile(tmp_path / "endpoint", tmp_path / "coupling" / "xpbd_service")
    assert rc["log_condition"] == "log_write_failed" and not rc["consistent"], rc


# ------------------------------------------------------------------ C: U13 fixed topology
def test_c_check_fixed_topology():
    r = dict(start_version=[1, 3, 1], committed_version=[1, 4, 1])
    check_fixed_topology(r, 1)
    for bad in (dict(r, committed_version=[1, 4, 2]), dict(r, start_version=[1, 3, 0]),
                dict(r, committed_version=[1, 4, True]), dict(r, committed_version=(1, 4, 1)),
                {k: v for k, v in r.items() if k != "committed_version"}):
        with pytest.raises(DriverError, match="fixed U13 topology"):
            check_fixed_topology(bad, 1)


def test_c_the_endpoint_keeps_the_general_rule():
    """The endpoint accepts topology + 1 : the U13 restriction lives in
    the registered-bridge driver, not in the general endpoint."""
    sim, ep, peer, _ = te.build()
    rc = copy.deepcopy(peer.consume(ep.step(peer.open_window(h=ep.H, particle_dt=ep.dt))))
    rc["committed_version"] = [rc["committed_version"][0], rc["committed_version"][1], rc["committed_version"][2] + 1]
    ep.acknowledge(rc)
    assert ep.acknowledged() == 1 and ep.stopped is None


def _patched_request(drv, mode_name, k_target, mutate):
    orig = drv.link.request
    count = dict(n=0)

    def request(mode, meta, arrays=None, *, expect, timeout):
        fr = orig(mode, meta, arrays, expect=expect, timeout=timeout)
        if fr.mode == mode_name:
            k, count["n"] = count["n"], count["n"] + 1
            if k == k_target:
                return mutate(fr)
        return fr
    drv.link.request = request


def _bump_receipt(fr):
    m = copy.deepcopy(fr.meta)
    cv = m["receipt"]["committed_version"]
    m["receipt"]["committed_version"] = [cv[0], cv[1], cv[2] + 1]
    m["receipt_digest"] = wire.json_digest(m["receipt"])      # stated digest made consistent
    return dataclasses.replace(fr, meta=m)


def _bump_offer(fr):
    m = copy.deepcopy(fr.meta)
    m.pop("offer_digest")
    m["topology_version"] += 1
    m["offer_digest"] = xpbd_service.payload_digest(m, fr.arrays)
    return dataclasses.replace(fr, meta=m)


@needs_xpbd
@pytest.mark.parametrize("case", ["receipt", "offer"])
def test_c_topology_change_stops_the_u13_run(case, tmp_path):
    sim, ep = ta.build_h()
    spec, mesh = ta.spec_for(ep)
    drv = ta.CoupledDriver(ep, snapshot=spec, mesh_arrays=mesh, log_dir=tmp_path / "coupling", xpbd_root=ta.XPBD_WORK,
                           timeouts=ta.FAST, expect_mesh=ta.lf.reference_mesh())
    _patched_request(drv, "RECEIPT" if case == "receipt" else "OFFER", 1,
                     _bump_receipt if case == "receipt" else _bump_offer)
    r = drv.run(5)
    rows, final = ta.service_log(tmp_path)
    assert r["status"].startswith("stopped: S8") and "fixed U13 topology" in ep.stopped["reason"], ep.stopped
    assert ep.acknowledged() == 1 and r["xpbd_stop"]["exitcode"] == 0
    with pytest.raises(EndpointStop, match="stopped earlier"):
        ep.halt("again")
    archive.write(tmp_path / "endpoint", ep)
    rc = reconcile(tmp_path / "endpoint", tmp_path / "coupling" / "xpbd_service")
    assert rc["consistent"], rc
    if case == "receipt":                                     # XPBD committed window 1, MPM refused its receipt
        assert ep.awaiting_receipt == 1 and len(rows) == 2 and rc["extra_xpbd_windows"] == 1 and rc["note"]
        assert "did not acknowledge" in ep.stopped["reason"]
    else:                                                     # the offer is refused before MPM advances
        assert ep.awaiting_receipt is None and len(rows) == 1 and final["open_window"] == 1
        assert ep.k == 1 and int(sim.step[None]) == ep.n_sub


# ------------------------------------------------------------------ the fixed example transform
def test_config_may_state_the_transform_status_and_signoff():
    """The fixed example preserves its numerical transform without a personal approval record."""
    from mpm import load_config
    from mpm.contact.live import PROPOSED_STATUS, resolve_transform
    from mpm.contact.xpbd_mesh import proposed_transform
    import coupled_u13
    cfg = load_config(str(ta.WORK / "scenes" / "configs" / "coupled_u4.json"), [], coupled_u13.SCENE_DEFAULTS)
    t = resolve_transform(cfg["contact"]["surface"]["transform"])
    p = proposed_transform()
    assert (t["R"], t["c"], t["C"]) == (p["R"], p["c"], p["C"])             # fixed reference values
    assert t["status"] == "fixed-transform-v1" and t["user_signoff"] == "pending"
    assert "user_signoff" not in cfg["contact"]["surface"]["transform"]
    assert {k: resolve_transform(None)[k] for k in PROPOSED_STATUS} == PROPOSED_STATUS   # default unchanged
    assert resolve_transform(dict(user_signoff="x"))["status"] == PROPOSED_STATUS["status"]
    with pytest.raises(KeyError):
        resolve_transform(dict(signoff="x"))
