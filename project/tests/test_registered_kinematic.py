"""Bounded host-only registration tests; no MPM or Taichi import."""
import argparse
import copy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np

from .test_kinematic_surface import surface, message, Checks as CandidateChecks
from ..coupling.kinematic_surface import KinematicSurfaceBridge
from ..coupling.loads import SourceIdentity
from ..coupling.registered_kinematic import create_registered_kinematic_adapter as create
from ..coupling.registered_kinematic import _json, _code_digest


def registration():
    files = {"mpm/a.py": "a"*64, "coupling/b.py": "b"*64}
    raw = ("coupling/b.py\t" + "b"*64 + "\nmpm/a.py\t" + "a"*64 + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    return dict(schema="mpm-endpoint-registration-v1",
                source=dict(source_id="mpm-couple.test", kind="real_backend",
                            version=int(digest[:8],16)),
                code_digest=digest, code_files=files, config_digest="c"*64,
                mpm_run_id="fixture-not-a-real-MPM-run", mpm_epoch=1,
                window=dict(dt=.0002,n_sub=5,H_ns=1000000), peer=None,
                runtime=dict(python="test", nested=["unverified"]), evidence="fixture",
                extra=dict(authenticated=True, nested=[1,2]))


def setup(velocity=(0.,0.,0.), reg=None, **kwargs):
    return create(surface(velocity), registration=registration() if reg is None else reg,
                  stream_id="registered-test", h=.001, particle_dt=.0002, **kwargs)


class RegisteredChecks(unittest.TestCase):
    def test_fixed_and_translation_three_windows(self):
        for velocity in [(0.,0.,0.),(.2,-.3,.5)]:
            a=setup(velocity); original=a.snapshot.positions.copy()
            for k in range(3):
                o=a.open_window(h=.001,particle_dt=.0002)
                self.assertEqual(o.source.kind,"real_backend")
                self.assertEqual(o.source,SourceIdentity(**a.registration["source"]))
                self.assertEqual(o.substeps,5)
                for i in range(5):
                    np.testing.assert_array_equal(o.geometry_at_tick(i).positions,
                        o.snapshot.positions+i*.0002*o.snapshot.velocity)
                r=a.consume(message(o))
                np.testing.assert_array_equal(r["impulse"],np.arange(12).reshape(4,3)*.01)
                np.testing.assert_allclose(a.snapshot.positions,
                    original+(k+1)*.001*np.array(velocity),atol=1e-15,rtol=0)
                self.assertEqual(r["window_index"],k)
                self.assertEqual(r["previous_dt"],.002 if k==0 else .001)
                self.assertEqual(r["registration_digest"],a.registration_digest)
                self.assertEqual(r["registration_binding"],a.binding)
                self.assertEqual(r["mpm_run_id"],a.registration["mpm_run_id"])
                for key in ["authenticated","external_process_observed","external_files_verified",
                            "config_verified","device_solve","real_coupling","load_applied_to_geometry"]:
                    self.assertIs(r[key],False)
                self.assertIsNone(r["support_balance"]);self.assertIsNone(r["drive_work"])
                self.assertEqual(r["source_authenticity"],"unverified_real_backend_registration")
                self.assertEqual(r["mpm_epoch"],1)
                self.assertEqual(r["start_version"][0],2)

    def test_bound_dt_at_every_open(self):
        a=setup()
        for k in range(2):
            for dt in [.0005,.0001,.0003,True,0,float("nan")]:
                with self.assertRaises(ValueError):a.open_window(h=.001,particle_dt=dt)
                self.assertFalse(a.failed)
            o=a.open_window(h=.001,particle_dt=.0002);a.consume(message(o))

    def test_registration_deep_ownership_unknown_fields(self):
        r=registration();a=setup(reg=r);before=a.registration;digest=a.registration_digest
        r["code_files"]["mpm/a.py"]="0"*64
        r["runtime"]["nested"].append("changed");r["extra"]["nested"][0]=99
        r["peer"]={"bad":"binding"};r["source"]["kind"]="prescribed"
        output=a.registration;output["extra"]["nested"][1]=88
        binding=a.binding;binding["expected_source"]["source_id"]="bad"
        self.assertEqual(a.registration,before);self.assertEqual(a.registration_digest,digest)
        o=a.open_window(h=.001,particle_dt=.0002);self.assertFalse(a.consume(message(o))["authenticated"])

    def test_peer_null_and_exact_late_binding(self):
        a=setup();d=a.registration_digest
        self.assertTrue(a.validate_peer(a.binding))
        self.assertIsNone(a.registration["peer"]);self.assertEqual(a.registration_digest,d)
        r=registration();r["peer"]=a.binding;b=setup(reg=r)
        r["peer"]["expected_source"]["version"]=999
        self.assertEqual(b.binding,a.binding)
        for key in a.binding:
            wrong=a.binding;wrong[key]=None
            with self.assertRaises(ValueError):a.validate_peer(wrong)
        wrong=a.binding;wrong["snapshot_epoch"]=True
        with self.assertRaises(ValueError):a.validate_peer(wrong)
        for peer in [{}, "x", [], {"extra":1}, wrong]:
            r=registration();r["peer"]=peer
            with self.assertRaises(ValueError):setup(reg=r)

    def test_malformed_registration_rejected(self):
        for key in registration():
            if key in ("runtime","evidence","extra"):continue
            r=registration();del r[key]
            with self.assertRaises((TypeError,ValueError)):setup(reg=r)
        cases=[("schema","x"),("source",{}),("code_digest","A"*64),
               ("code_files",{}),("config_digest","bad"),("mpm_run_id"," "),
               ("mpm_epoch",0),("mpm_epoch",True),("window",{}),
               ("code_digest","0"*64)]
        for key,val in cases:
            r=registration();r[key]=val
            with self.assertRaises((TypeError,ValueError)):setup(reg=r)
        for key,val in [("kind","prescribed"),("version",True),("version",0)]:
            r=registration();r["source"][key]=val
            with self.assertRaises(ValueError):setup(reg=r)

    def test_paths_and_aggregate(self):
        for name in ["../a.py","/a.py","C:a.py","a//b.py","a/./b.py","a\\b.py","a\tb.py"]:
            r=registration();r["code_files"]={name:"a"*64}
            r["code_digest"]=_code_digest(r["code_files"]);r["source"]["version"]=int(r["code_digest"][:8],16)
            with self.assertRaises(ValueError):setup(reg=r)
        r=registration();r["code_files"]["mpm/a.py"]="0"*64
        with self.assertRaises(ValueError):setup(reg=r)

    def test_json_domain_and_canonical_known_bytes(self):
        self.assertEqual(_json({"é":1.25,"a":[True,None,2]}),
                         '{"a":[true,null,2],"\u005cu00e9":1.25}')
        files={"é.py":"a"*64,"a.py":"0"*64}
        expected=hashlib.sha256(("a.py\t"+"0"*64+"\né.py\t"+"a"*64+"\n").encode("utf-8")).hexdigest()
        self.assertEqual(_code_digest(files),expected)
        r=registration();a=setup(reg=r)
        reversed_r=dict(reversed(list(r.items())))
        self.assertEqual(setup(reg=reversed_r).registration_digest,a.registration_digest)
        for value in [float("nan"),float("inf"),("\ud800"),{1:"not-string"},(1,2)]:
            r=registration();r["extra"]=value
            with self.assertRaises(ValueError):setup(reg=r)
        for value in ['{"source":1,"source":2}', [], None]:
            with self.assertRaises(TypeError):setup(reg=value) if value is not None else create(
                surface(),registration=None,stream_id="x",h=.001,particle_dt=.0002)

    def test_leading_zero_digest_and_unicode_path_registration(self):
        r=registration();r['code_files']={'mpm/33.py':'a'*64}
        r['code_digest']='0564a7cc6b4600f6f322ced70d5da6c58d34b8d1dfd9c723e64f359244344a9c'
        r['source']['version']=90482636
        a=setup(reg=r);self.assertEqual(a.registration['source']['version'],int('0564a7cc',16))
        r=registration();r['code_files']={'mpm/é.py':'a'*64}
        r['code_digest']=hashlib.sha256(('mpm/é.py\t'+'a'*64+'\n').encode('utf-8')).hexdigest()
        r['source']['version']=int(r['code_digest'][:8],16)
        self.assertEqual(setup(reg=r).registration['code_files'],r['code_files'])
    def test_registered_clock_rejections(self):
        for key,value in [("dt",.0005),("n_sub",2),("n_sub",True),("H_ns",999999),
                          ("H_ns",1000000.0),("dt",float("inf"))]:
            r=registration();r["window"][key]=value
            with self.assertRaises(ValueError):setup(reg=r)
        for h,dt in [(.0005,.0002),(.0010000001,.00020000002),(.001,.0005)]:
            with self.assertRaises(ValueError):create(surface(),registration=registration(),
                stream_id="x",h=h,particle_dt=dt)

    def test_bad_source_stream_time_version_dense_retry(self):
        a=setup();o=a.open_window(h=.001,particle_dt=.0002)
        changes=[dict(stream_id="bad"),dict(source=replace(o.source,version=0)),
                 dict(source=replace(o.source,kind="prescribed")),dict(cloth_version=(9,0,7)),
                 dict(t_start=.0001),dict(t_end=.002),
                 dict(vertex_ids=np.array([17,3,44]),impulse=np.zeros((3,3))),
                 dict(vertex_ids=np.array([17,3,44,99])),dict(vertex_ids=np.array([17,3,44,44]))]
        for change in changes:
            with self.assertRaises(ValueError):a.consume(message(o,**change))
            self.assertFalse(a.failed);self.assertIs(a.snapshot,o.snapshot)
        with self.assertRaises(ValueError):a.consume(replace(message(o),window_index=9))
        receipt=a.consume(message(o,impulse=np.zeros((4,3))))
        self.assertEqual(receipt["impulse"],np.zeros((4,3)).tolist())

    def test_repeat_id_and_window_rejected(self):
        a=setup();o=a.open_window(h=.001,particle_dt=.0002);m=message(o);a.consume(m)
        with self.assertRaises(RuntimeError):a.consume(m)
        o=a.open_window(h=.001,particle_dt=.0002)
        with self.assertRaises(ValueError):a.consume(message(o,batch_id="batch-0"))
        a.consume(message(o))

    def test_commit_exception_and_interrupt_poison(self):
        for error in [RuntimeError("injected"),KeyboardInterrupt("injected")]:
            a=setup();old=a.snapshot;o=a.open_window(h=.001,particle_dt=.0002)
            b=a._bridge
            def broken(state):
                b._snapshot=state
                raise error
            b._commit=broken
            with self.assertRaises(type(error)):a.consume(message(o))
            self.assertTrue(a.failed);self.assertIs(b._snapshot,old)
            with self.assertRaises(RuntimeError):a.open_window(h=.001,particle_dt=.0002)

    def test_constructor_and_old_gate(self):
        source=SourceIdentity(**registration()["source"])
        with self.assertRaises(ValueError):KinematicSurfaceBridge(surface(),source=source,stream_id="x",h=.001)
        s=surface();v=s.velocity.copy();v[0,0]=1
        for state in [replace(s,step_index=1),replace(s,time=.001),replace(s,velocity=v),
                      replace(s,inverse_mass=np.ones(4)),replace(s,positions=np.zeros((4,3)))]:
            with self.assertRaises(ValueError):create(state,registration=registration(),
                stream_id="x",h=.001,particle_dt=.0002)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(RegisteredChecks),
                             unittest.defaultTestLoader.loadTestsFromTestCase(CandidateChecks)])
    log=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=log,verbosity=2,failfast=True).run(suite)
    root=Path(__file__).resolve().parents[2]
    files=["project/coupling/registered_kinematic.py","project/tests/test_registered_kinematic.py",
           "project/coupling/kinematic_surface.py","project/coupling/contracts.py",
           "project/coupling/exchange.py","project/coupling/loads.py"]
    report=dict(scope="host-only synthetic registration fixtures; no MPM process or authentication",
                tests_run=result.testsRun,success=result.wasSuccessful(),log=log.getvalue(),
                elapsed_seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,
                forbidden_modules_loaded=[n for n in sys.modules if n=="taichi" or n=="mpm"],
                source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files})
    report["success"] &= not report["forbidden_modules_loaded"]
    with args.output.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(log.getvalue());print(json.dumps({k:report[k] for k in ("tests_run","success","elapsed_seconds","forbidden_modules_loaded")}))
    return 0 if report["success"] else 1


if __name__=="__main__":raise SystemExit(main())
