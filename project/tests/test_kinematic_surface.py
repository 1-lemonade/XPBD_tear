"""Bounded host-only U13 candidate checks. No MPM, Taichi init or physics."""
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from ..coupling.contracts import ClothSnapshot
from ..coupling.exchange import SCHEMA, ExchangeAdapter, ImpulseWindow
from ..coupling.kinematic_surface import KinematicSurfaceBridge
from ..coupling.loads import LoadBatch, SourceIdentity

SOURCE = SourceIdentity("u13-local-synthetic-fixture", "prescribed", 1)


def surface(velocity=(0., 0., 0.)):
    return ClothSnapshot(np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),
                         np.array([[0,1,2],[0,2,3]]), np.array([17,3,44,8]),
                         np.array([51,20]), np.tile(velocity,(4,1)), np.zeros(4),
                         np.ones(4,dtype=bool), 2,0,7,0.,.002)


def setup(velocity=(0.,0.,0.), cls=KinematicSurfaceBridge, h=.001):
    b = cls(surface(velocity), source=SOURCE, stream_id="u13-candidate-test", h=h)
    return b, ExchangeAdapter(b,SOURCE)


def message(offer, **changes):
    order = np.array([2,0,3,1])
    values = dict(stream_id=offer.stream_id, source=offer.source,
                  cloth_version=offer.snapshot.version, batch_id=f"batch-{offer.window_index}",
                  t_start=offer.snapshot.time, t_end=offer.snapshot.time+offer.h,
                  vertex_ids=offer.snapshot.vertex_ids[order],
                  impulse=np.arange(12,dtype=float).reshape(4,3)[order]*.01)
    values.update(changes)
    return ImpulseWindow(SCHEMA,offer.window_index,LoadBatch(**values))


class Checks(unittest.TestCase):
    def test_fixed_three_windows_dense_permutation_and_null_reaction(self):
        b,a=setup(); old=a.snapshot
        for k in range(3):
            o=a.open_window(h=.001,particle_dt=.0002); r=a.consume(message(o))
            np.testing.assert_array_equal(a.snapshot.positions,old.positions)
            np.testing.assert_array_equal(r["impulse"],np.arange(12).reshape(4,3)*.01)
            self.assertEqual(a.snapshot.version,(2,k+1,7)); self.assertEqual(r["window_index"],k)
            self.assertIsNone(r["support_balance"]); self.assertIsNone(r["drive_work"])
            self.assertFalse(r["load_applied_to_geometry"] or r["authenticated"] or r["device_solve"])
        self.assertEqual(old.step_index,0)

    def test_translation_three_windows_and_frozen_tick_metadata(self):
        b,a=setup((.2,-.3,.5)); originals=[]
        for _ in range(3):
            o=a.open_window(h=.001,particle_dt=.0002); p=o.snapshot.positions.copy()
            originals.append((o.snapshot,p))
            for i in range(5):
                tick=o.geometry_at_tick(i)
                np.testing.assert_array_equal(tick.positions,p+(i*.0002)*o.snapshot.velocity)
                self.assertEqual(tick.time,o.snapshot.time); self.assertEqual(tick.version,o.snapshot.version)
            with self.assertRaises(ValueError): o.geometry_at_tick(5)
            a.consume(message(o)); np.testing.assert_array_equal(a.snapshot.positions,p+.001*o.snapshot.velocity)
            self.assertEqual(a.snapshot.dt,.001)
        for s,p in originals: np.testing.assert_array_equal(s.positions,p)

    def test_sparse_empty_unknown_and_zero_dense_retry(self):
        b,a=setup(); o=a.open_window(h=.001,particle_dt=.0002)
        for ids,imp in (([17,3,44],np.zeros((3,3))),([],np.zeros((0,3))),
                        ([17,3,44,99],np.zeros((4,3))),([17,3,44,8,99],np.zeros((5,3)))):
            with self.assertRaises(ValueError): a.consume(message(o,vertex_ids=np.array(ids,dtype=np.int64),impulse=imp))
            self.assertFalse(b.failed); self.assertIs(a.snapshot,o.snapshot)
        r=a.consume(message(o,impulse=np.zeros((4,3))))
        self.assertEqual(r["impulse"],np.zeros((4,3)).tolist())

    def test_identity_clock_and_window_rejections_retry(self):
        b,a=setup(); o=a.open_window(h=.001,particle_dt=.0002)
        bads=[dict(stream_id="foreign"),dict(source=replace(SOURCE,version=2)),
              dict(source=SourceIdentity("pretend-real","real_backend",1)),
              dict(cloth_version=(2,0,8)),dict(cloth_version=(3,0,7)),
              dict(t_start=.0001),dict(t_end=.002)]
        for change in bads:
            with self.assertRaises(ValueError): a.consume(message(o,**change))
            self.assertFalse(b.failed); self.assertIs(a.snapshot,o.snapshot)
        with self.assertRaises(ValueError): a.consume(replace(message(o),window_index=1))
        m=message(o); a.consume(m)
        with self.assertRaises(RuntimeError): a.consume(m)
        a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaises(ValueError): a.consume(m)

    def test_batch_shape_duplicate_and_nonfinite_rejected(self):
        _,a=setup(); o=a.open_window(h=.001,particle_dt=.0002)
        for change in [dict(vertex_ids=[17,3,44,44]),dict(impulse=np.zeros((3,3))),
                       dict(impulse=np.full((4,3),np.nan)),dict(impulse=np.full((4,3),np.inf)),
                       dict(vertex_ids=[17.,3.,44.,8.])]:
            with self.assertRaises(ValueError): message(o,**change)
        a.consume(message(o))

    def test_h_and_dt_are_explicit_and_fixed(self):
        b,a=setup()
        for h,dt in [(.0005,.0001),(.001,.0003)]:
            with self.assertRaises(ValueError): a.open_window(h=h,particle_dt=dt)
            self.assertFalse(b.failed)
        o=a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaises(RuntimeError): a.open_window(h=.001,particle_dt=.0002)
        self.assertEqual(o.snapshot.dt,.002)  # previous dt is not next h
        a.consume(message(o))

    def test_constructor_scope_rejections(self):
        s=surface(); nonuniform=s.velocity.copy(); nonuniform[1,0]=1
        for state in [replace(s,velocity=nonuniform),replace(s,pinned=np.zeros(4,dtype=bool)),
                      replace(s,inverse_mass=np.ones(4)),replace(s,step_index=1),replace(s,time=.001),
                      replace(s,positions=np.zeros((4,3)))]:
            with self.assertRaises(ValueError): KinematicSurfaceBridge(state,source=SOURCE,stream_id="s",h=.001)
        for source in [None,SourceIdentity("mpm","real_backend",1)]:
            with self.assertRaises(ValueError): KinematicSurfaceBridge(s,source=source,stream_id="s",h=.001)
        with self.assertRaises(ValueError): KinematicSurfaceBridge(s,source=SOURCE,stream_id="s",h=0)

    def test_owned_arrays_and_receipt_are_detached(self):
        s=surface((.2,0.,0.)); b=KinematicSurfaceBridge(s,source=SOURCE,stream_id="s",h=.001)
        self.assertFalse(np.shares_memory(s.positions,b.snapshot.positions))
        for value in (b.snapshot.positions,b.snapshot.velocity,b.snapshot.vertex_ids,b.snapshot.triangles):
            with self.assertRaises(ValueError): value.flat[0]=999
        a=ExchangeAdapter(b,SOURCE); o=a.open_window(h=.001,particle_dt=.0002)
        m=message(o); r=a.consume(m); r["impulse"][0][0]=999
        self.assertNotEqual(m.batch.impulse[0,0],999)
        np.testing.assert_array_equal(s.positions,surface().positions)

    def test_failure_after_publication_hook_poisoned_and_prefix_intact(self):
        class Broken(KinematicSurfaceBridge):
            def _commit(self,state):
                super()._commit(state)
                raise RuntimeError("injected post-assignment publication error")
        b,a=setup(cls=Broken); old=a.snapshot; p=old.positions.copy()
        o=a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaisesRegex(RuntimeError,"injected"): a.consume(message(o))
        self.assertTrue(b.failed and b._ledger.poisoned and a.failed)
        self.assertIs(b._snapshot,old); np.testing.assert_array_equal(old.positions,p)
        with self.assertRaises(RuntimeError): a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaises(RuntimeError): b.snapshot

    def test_reentrancy_and_duplicate_batch_id(self):
        class Reentrant(KinematicSurfaceBridge):
            def _commit(self,state): self.advance(self.pending,.001)
        b,a=setup(cls=Reentrant); o=a.open_window(h=.001,particle_dt=.0002); m=message(o); b.pending=m.batch
        with self.assertRaisesRegex(RuntimeError,"reentrant"): a.consume(m)
        self.assertTrue(b.failed)
        b,a=setup(); o=a.open_window(h=.001,particle_dt=.0002); a.consume(message(o))
        o=a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaisesRegex(ValueError,"already consumed"): a.consume(message(o,batch_id="batch-0"))
        self.assertFalse(b.failed); a.consume(message(o))

    def test_nonfinite_candidate_prevalidation_is_retryable(self):
        b,a=setup((1e308,0.,0.),h=10.); o=a.open_window(h=10.,particle_dt=10.)
        with self.assertRaises(ValueError): a.consume(message(o))
        self.assertFalse(b.failed); self.assertEqual(len(b._ledger._batch_ids),0)
        self.assertIs(a.snapshot,o.snapshot)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    log=io.StringIO(); suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
    root=Path(__file__).resolve().parents[2]
    files=["project/coupling/kinematic_surface.py","project/tests/test_kinematic_surface.py",
           "project/coupling/contracts.py","project/coupling/exchange.py","project/coupling/loads.py"]
    report=dict(scope="host-only synthetic candidate; no physics or MPM",tests_run=result.testsRun,
                success=result.wasSuccessful(),log=log.getvalue(),python=sys.version,numpy=np.__version__,
                forbidden_modules_loaded=[name for name in sys.modules if name=="taichi" or name=="mpm"],
                source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files})
    if report["forbidden_modules_loaded"]: report["success"]=False
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(log.getvalue()); print(json.dumps({k:report[k] for k in ("scope","tests_run","success","forbidden_modules_loaded")}))
    return 0 if report["success"] else 1


if __name__=="__main__": raise SystemExit(main())
