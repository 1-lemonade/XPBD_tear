"""Pre-registered CPU f64 SI exchange checks with a prescribed synthetic peer."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import traceback
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import taichi as ti

from ..coupling.contracts import MappingConfig, MappingFeature, MappingStatus, QueryPoints
from ..coupling.exchange import SCHEMA, ExchangeAdapter, WindowOffer
from ..coupling.exchange_archive import make_record, read_archive, write_archive
from ..coupling.exchange_cloth import ExchangeCloth
from ..coupling.loads import SourceIdentity
from ..coupling.mapping import map_points
from ..coupling.synthetic_peer import SyntheticPeer
from .test_mapping import make_snapshot

SOURCE = SourceIdentity("synthetic-point-forces", "prescribed", 1)
PARTICLE_DT = .0002
WINDOWS = (.001,.001,.001,.0006)
IDS = (17,-5,103)
MASS = np.array([.2,.3,.4])
FORCES = np.array([[.3,-.1,.8],[-.1,.25,.5],[.5,.2,1.]])
RADIUS = .025
ERROR_PEAKS = {}


def close(a, b, *, absolute=1e-12, relative=1e-12):
    np.testing.assert_allclose(a,b,rtol=relative,atol=absolute,equal_nan=True)
    errors = np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))
    finite = errors[np.isfinite(errors)]
    budget = f"atol={absolute:g},rtol={relative:g}"
    ERROR_PEAKS[budget] = max(ERROR_PEAKS.get(budget,0.),float(finite.max(initial=0.)))


def reject(rejections, name, call):
    try:
        call()
    except (ValueError, RuntimeError) as error:
        rejections[name] = str(error)
        return
    raise AssertionError(f"{name}: accepted invalid exchange")


def make_peer(s):
    weights = np.array([[.2,.3,.5], [.4,.35,.25]])
    points = (weights[:,:,None]*s.positions[s.triangles[[0,7]]]).sum(axis=1)
    positions = np.vstack((points + [0.,0.,.002], [2.,2.,.002]))
    return SyntheticPeer(SOURCE, IDS, positions, np.zeros((3,3)), MASS, FORCES, max_distance=RADIUS)


def state_momenta(data):
    m = np.asarray(data["mass"])
    p = m[:,None]*np.asarray(data["velocity"])
    return m.sum(), p.sum(axis=0), np.cross(data["positions"],p).sum(axis=0)


def audit_peer(offer, trace, message, initial_x, initial_v):
    """Recompute material ticks and loads independently from the actual start state."""
    s, h, dt = offer.snapshot, offer.h, offer.particle_dt
    x, v = initial_x.copy(), initial_v.copy()
    assert offer.substeps == (3 if h == .0006 else 5) == len(trace["ticks"])
    assert trace["window_index"] == offer.window_index
    np.testing.assert_array_equal(trace["particle_ids"], IDS)
    close(trace["positions_before"],x)
    close(trace["velocity_before"],v)
    close(trace["force_on_cloth_N"], FORCES)
    close(trace["mass"], MASS)
    index = {int(k):i for i,k in enumerate(s.vertex_ids)}
    impulse = np.zeros((len(index),3))
    peak_prediction_displacement = 0.
    for i,tick in enumerate(trace["ticks"]):
        tau = i*dt
        assert tick["index"] == i and tick["tau"] == tau and offer.sample_time(i) == tau
        assert 0 <= tau < h
        # Form the independent extrapolation explicitly, without using the offer's helper as oracle.
        expected_position = s.positions + s.velocity*tau
        predicted = replace(s, positions=expected_position)
        provided = offer.geometry_at_tick(i)
        np.testing.assert_array_equal(provided.positions, predicted.positions)
        assert provided.version == s.version and provided.time == s.time and provided.dt == s.dt
        for field in ("triangles","vertex_ids","triangle_ids","velocity"):
            np.testing.assert_array_equal(getattr(provided, field),getattr(s,field))
        peak_prediction_displacement = max(peak_prediction_displacement,
                                          float(np.abs(predicted.positions-s.positions).max()))
        mapping = map_points(predicted, QueryPoints(IDS,x), MappingConfig(max_distance=RADIUS))
        for name, values in (("status",mapping.status), ("triangle_ids",mapping.triangle_ids),
                             ("vertex_ids",mapping.vertex_ids)):
            np.testing.assert_array_equal(tick[name], values)
        close(np.asarray(tick["barycentric"], dtype=float),mapping.barycentric)
        # The two selected fixture points are strictly in face interiors; the third is out of range.
        np.testing.assert_array_equal(mapping.status,[MappingStatus.HIT,MappingStatus.HIT,MappingStatus.OUTSIDE_THRESHOLD])
        np.testing.assert_array_equal(mapping.feature[:2], [MappingFeature.FACE]*2)
        np.testing.assert_array_equal(tick["eligible"], [True,True,False])
        expected_j = FORCES*dt
        expected_j[2] = 0.
        close(tick["point_impulse_on_cloth"],expected_j)
        close(tick["query_positions"],x)
        surface_v = np.full((len(IDS),3),np.nan)
        nodal_tick = np.zeros_like(impulse)
        for p in range(2):
            rows = [index[int(k)] for k in mapping.vertex_ids[p]]
            weights = mapping.barycentric[p]
            surface_v[p] = weights @ s.velocity[rows]
            for row,w in zip(rows,weights):
                nodal_tick[row] += expected_j[p]*w
        impulse += nodal_tick
        close(tick["closest_point"],mapping.closest_point)
        point_moment = np.cross(mapping.closest_point[:2], expected_j[:2]).sum(axis=0)
        close(tick["cloth_angular_impulse"],point_moment)
        close(np.cross(predicted.positions,nodal_tick).sum(axis=0),point_moment)
        close(np.asarray(tick["surface_velocity"], dtype=float),surface_v)
        v -= expected_j / MASS[:,None]
        x += dt*v
        close(tick["particle_velocity_after"],v)
        close(tick["particle_positions_after"],x)
    close(trace["positions_after"],x)
    close(trace["velocity_after"],v)
    change = (MASS[:,None]*(v-initial_v)).sum(axis=0)
    close(trace["peer_impulse"],change)
    close(trace["cloth_impulse"],impulse.sum(axis=0))
    close(change+impulse.sum(axis=0),np.zeros(3))
    rows = [index[int(k)] for k in message.batch.vertex_ids]
    close(message.batch.impulse,impulse[rows])
    assert len(rows) == len(index) and len(set(rows)) == len(rows)
    return impulse,x,v,peak_prediction_displacement


def audit_receipt(receipt, offer, impulse, published):
    h, s = offer.h, offer.snapshot
    assert receipt["schema"] == "xpbd-exchange-cloth-v1"
    assert receipt["h"] == h and receipt["previous_dt"] == s.dt
    assert receipt["t_start"] == s.time and receipt["t_end"] == published.time == s.time+h
    assert list(receipt["start_version"]) == list(s.version)
    assert list(receipt["solved_version"]) == [s.epoch,s.step_index+1,s.topology_version]
    assert list(receipt["committed_version"]) == list(published.version)
    np.testing.assert_array_equal(receipt["vertex_ids"],s.vertex_ids)
    close(receipt["impulse"],impulse)
    begin, upload, solve, final = (receipt[k] for k in ("before","uploaded","solved","committed"))
    close(begin["velocity"],s.velocity)
    close(begin["positions"],s.positions)
    close(upload["positions"],s.positions)
    close(upload["velocity"],s.velocity+impulse*s.inverse_mass[:,None])
    close(np.asarray(upload["velocity"])[s.pinned],s.velocity[s.pinned])
    assert np.linalg.norm(impulse[s.pinned].sum(axis=0)) > 1e-8
    assert np.linalg.norm(impulse[~s.pinned].sum(axis=0)) > 1e-8
    np.testing.assert_array_equal(final["vertex_ids"], published.vertex_ids)
    close(final["positions"],published.positions)
    close(final["velocity"],published.velocity)
    free_dx = np.linalg.norm((np.asarray(solve["positions"])-s.positions)[~s.pinned],axis=1).max()
    assert free_dx > 1e-9
    dp = state_momenta(solve)[1]-state_momenta(begin)[1]
    if h == .001:
        b = receipt["balance"]
        close(b["J_support_load"], -impulse[s.pinned].sum(axis=0))
        close(b["J_load"],impulse.sum(axis=0))
        close(dp - (impulse.sum(axis=0)+b["J_support"]),np.zeros(3),absolute=1e-10,relative=0)
        support = np.asarray(receipt["balance_arrays"]["support_impulse"]).sum(axis=0)
        close(support,b["J_support"])
        assert np.linalg.norm(support) > 1e-9
    else:
        assert receipt["balance"] is None and receipt["balance_arrays"] is None
    m0,p0,l0 = state_momenta(solve)
    m1,p1,l1 = state_momenta(final)
    close(np.r_[m1-m0,p1-p0,l1-l0],np.zeros(7),absolute=1e-12,relative=0)
    assert receipt["convergence"]["stop_reason"] == "converged"
    return dict(verified_input=impulse.sum(axis=0).tolist(), pin_input=impulse[s.pinned].sum(axis=0).tolist(),
                cloth_momentum_change=dp.tolist(), max_free_displacement=float(free_dx), support_measured=(h == .001),
                start_version=list(s.version), end_version=list(published.version), stop=receipt["convergence"])


def invalid_messages(adapter, offer, message, rejections):
    batch = message.batch
    reject(rejections,"duplicate open offer",lambda: adapter.open_window(h=offer.h,particle_dt=PARTICLE_DT))
    for field, changes in (("stream",dict(stream_id="foreign")),
                           ("source",dict(source=replace(SOURCE,version=SOURCE.version+1))),
                           ("epoch",dict(cloth_version=(offer.snapshot.epoch+1,*batch.cloth_version[1:]))),
                           ("step",dict(cloth_version=(offer.snapshot.epoch,offer.snapshot.step_index+1,
                                                      offer.snapshot.topology_version))),
                           ("topology",dict(cloth_version=(*batch.cloth_version[:2],offer.snapshot.topology_version+1))),
                           ("start time",dict(t_start=np.nextafter(batch.t_start,np.inf))),
                           ("end time",dict(t_end=np.nextafter(batch.t_end,np.inf))),
                           ("stable ID",dict(vertex_ids=batch.vertex_ids+10000))):
        reject(rejections,field,lambda: adapter.consume(replace(message,batch=replace(batch,**changes))))
    reject(rejections,"foreign schema",lambda: adapter.consume(replace(message,schema="foreign-v1")))
    reject(rejections,"missing window",lambda: adapter.consume(replace(message,window_index=message.window_index+1)))


def archive_rejections(path, rejections):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    edits = (
        ("synthetic labelled real", 0, lambda w: w["source"].update(kind="real_backend")),
        ("peer reaction sign", 0, lambda w: w["peer"]["peer_impulse"].__setitem__(0,-w["peer"]["peer_impulse"][0])),
        ("uploaded velocity", 0, lambda w: w["cloth"]["uploaded"]["velocity"][0].__setitem__(0,.01)),
        ("end version",0,lambda w: w["end"].__setitem__("step_index",w["end"]["step_index"]+1)),
        ("missing peer tick",0,lambda w: w["peer"]["ticks"].pop()),
        ("foreign event parent",1,lambda w: w["cloth"]["event"]["parent_vertex_ids"].__setitem__(0,999)),
        ("outside misreported HIT",0,lambda w: w["peer"]["ticks"][0]["status"].__setitem__(2,0)),
    )
    for name,index,edit in edits:
        bad = deepcopy(envelope)
        edit(bad["windows"][index])
        with TemporaryDirectory(dir=path.parent) as temporary:
            trial = Path(temporary)/"invalid.json"
            trial.write_text(json.dumps(bad,allow_nan=False),encoding="utf-8")
            reject(rejections,"archive "+name,lambda: read_archive(trial))


def invalid_surfaces(rejections):
    vertices = [(0,0,0),(1,0,0),(0,1,0),(.5,-1,0),(1,1,0)]
    cases = (
        ("same-winding pair",vertices[:4],[(0,1,2),(0,1,3)],"winding"),
        ("three-face edge",vertices,[(0,1,2),(1,0,3),(0,1,4)],"nonmanifold"),
        ("disconnected vertex fan",vertices,[(0,1,2),(0,3,4)],"fan"),
    )
    for label,positions,triangles,reason in cases:
        s = make_snapshot(positions,triangles,epoch=0,step_index=0,time=0.,dt=.001)
        reject(rejections,label,lambda: WindowOffer(SCHEMA,"invalid-surface",SOURCE,0,s,PARTICLE_DT,.001))
        assert reason in rejections[label]


def main_windows(output, report):
    rejections = report["expected_rejections"]
    invalid_surfaces(rejections)
    bridge = ExchangeCloth(source=SOURCE, stream_id="u11-main", cut_after_window=2,cut_edge=(1,3))
    peer = make_peer(bridge.snapshot)
    adapter = ExchangeAdapter(bridge,SOURCE)
    for name,h in (("non-integral clock",.0007),("ULP wrong clock",np.nextafter(.001,np.inf))):
        reject(rejections,name,lambda: adapter.open_window(h=h,particle_dt=PARTICLE_DT))
    records, windows, history = [], [], []
    report["windows"] = windows
    for i,h in enumerate(WINDOWS):
        offer = adapter.open_window(h=h,particle_dt=PARTICLE_DT)
        assert offer.window_index == i and offer.h == h and offer.particle_dt == PARTICLE_DT
        assert offer.source == SOURCE and offer.stream_id == "u11-main"
        assert offer.snapshot.dt == .001
        if i == 2:
            assert tuple(offer.cut_edges) and offer.snapshot.topology_version > history[-1].batch.cloth_version[2]
        x0, v0 = peer.positions.copy(),peer.velocity.copy()
        if i == 2:
            old_snapshot = replace(offer.snapshot, topology_version=history[-1].batch.cloth_version[2])
            reject(rejections, "stale material-side opening version",
                   lambda: peer.prepare(replace(offer, snapshot=old_snapshot)))
        msg,trace = peer.prepare(offer)
        close(peer.positions,x0)
        close(peer.velocity,v0)
        impulse,x1,v1,peak = audit_peer(offer,trace,msg,x0,v0)
        if i == 0:
            invalid_messages(adapter,offer,msg,rejections)
        if i:
            reject(rejections,"replay window "+str(i-1),lambda: adapter.consume(history[-1]))
        if i == 1:
            reused = replace(msg.batch,batch_id=history[-1].batch.batch_id)
            reject(rejections,"reused batch ID on current window",lambda: adapter.consume(replace(msg,batch=reused)))
        if h != offer.snapshot.dt:
            wrong_end = offer.snapshot.time + offer.snapshot.dt
            reject(rejections,"previous dt substituted for next h",
                   lambda: adapter.consume(replace(msg,batch=replace(msg.batch,t_end=wrong_end))))
        if i == 2:
            old = history[-1].batch.cloth_version[2]
            tagged = replace(msg.batch,cloth_version=(*offer.snapshot.version[:2],old))
            reject(rejections,"explicit pre-tear topology",lambda: adapter.consume(replace(msg,batch=tagged)))
        receipt = adapter.consume(msg)
        history.append(msg)
        end = bridge.snapshot
        if i == 0:
            wrong_g = deepcopy(receipt)
            wrong_g["batch_impulse"][0][0] += .0001
            for label,attempt in (("missing cloth receipt",None),("wrong receipt G",wrong_g),
                                  ("wrong receipt window",dict(receipt,window_index=i+1)),
                                  ("wrong receipt version",dict(receipt,committed_version=list(offer.snapshot.version)))):
                reject(rejections,label,lambda: peer.commit(offer,trace,attempt))
                np.testing.assert_array_equal(peer.positions,x0)
                np.testing.assert_array_equal(peer.velocity,v0)
        peer.commit(offer,trace,receipt)
        close(peer.positions,x1)
        close(peer.velocity,v1)
        assert end.dt == h
        assert (receipt["event"] is not None) == (i == 1)
        if i == 1:
            assert set(receipt["event"]["cut_vertex_ids"]) == {1,3}
        if i > 0:
            assert peak > 1e-12, "moving frozen prediction was never exercised"
        result = audit_receipt(receipt,offer,impulse,end)
        result.update(index=i,h=h,previous_dt=offer.snapshot.dt,peer_ticks=offer.substeps,
                      peak_frozen_prediction_displacement=peak)
        windows.append(result)
        records.append(make_record(offer,msg,trace,receipt,end))
    reject(rejections,"reconsume last window",lambda: adapter.consume(history[-1]))
    reject(rejections,"double peer commit",lambda: peer.commit(offer,trace,receipt))
    assert sum(w["peer_ticks"] for w in windows) == 18
    path = output / "exchange.json"
    write_archive(path,records)
    readback = read_archive(path)
    assert len(readback) == len(WINDOWS)
    assert readback == records
    report["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    archive_rejections(path,rejections)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["archive_sha256"]


def faults(report):
    for fault in ("upload","rebuild"):
        bridge = ExchangeCloth(source=SOURCE, stream_id="u11-fault-"+fault,
                               cut_after_window=1 if fault == "rebuild" else None,cut_edge=(1,3))
        peer = make_peer(bridge.snapshot)
        adapter = ExchangeAdapter(bridge,SOURCE)
        offer = adapter.open_window(h=.001,particle_dt=PARTICLE_DT)
        x0, v0 = peer.positions.copy(),peer.velocity.copy()
        message,trace = peer.prepare(offer)
        solver = bridge._driver.solver
        calls = []
        if fault == "upload":
            target = "apply_impulses"
            actual_upload = solver.apply_impulses
            def fail(instance,*args,**kwargs):
                assert instance is solver
                actual_upload(*args,**kwargs)
                calls.append("after device upload")
                raise RuntimeError("injected failure after device upload")
        else:
            target = "rebuild_constraints"
            def fail(instance,*args,**kwargs):
                assert instance is solver
                calls.append("topology rebuild")
                raise RuntimeError("injected topology rebuild failure")
        with patch.object(type(solver),target,new=fail):
            reject(report["expected_rejections"],fault+" injection",lambda: adapter.consume(message))
        assert len(calls) == 1 and bridge.failed
        assert "injected" in report["expected_rejections"][fault+" injection"]
        close(peer.positions,x0)
        close(peer.velocity,v0)
        published_steps = bridge._driver.step_index
        assert published_steps == (1 if fault == "rebuild" else 0)
        reject(report["expected_rejections"],fault+" poisoned reopen",
               lambda: adapter.open_window(h=.001,particle_dt=PARTICLE_DT))
        reject(report["expected_rejections"],fault+" poisoned consume",lambda: adapter.consume(message))
        report.setdefault("faults",{})[fault] = dict(injection_point=calls[0],driver_step_index=published_steps,
                                                    peer_unchanged=True,terminal=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(command=[sys.executable,*sys.argv],python=sys.version,numpy=np.__version__,taichi=ti.__version__,
                  expected_rejections={},test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        main_windows(args.output,report)
        faults(report)
        actual = ti.lang.impl.current_cfg()
        assert actual.arch == ti.cpu and actual.default_fp == ti.f64
        report.update(status="passed",backend=str(actual.arch),dtype=str(actual.default_fp))
    except Exception as error:
        report.update(status="failed",error=str(error),traceback=traceback.format_exc())
    report["maximum_actual_errors"] = ERROR_PEAKS
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print({k:report.get(k) for k in ("status","backend","dtype","error")})
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
