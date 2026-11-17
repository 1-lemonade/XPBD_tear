"""Export deterministic host-only contract examples from synthetic fixtures.
No MPM, transport, physics or authentication. Run from work_place with XPBD_tear Python.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project.tests.test_registered_kinematic import registration
from project.tests.test_kinematic_surface import surface, message
from project.coupling.registered_kinematic import create_registered_kinematic_adapter

def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)

def vector(value):
    text = canonical(value)
    return dict(value=value, canonical_json=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    reg = registration()
    assert reg["peer"] is None
    cases = []
    for label, velocity in [("fixed", (0.,0.,0.)), ("translation", (.2,-.3,.5))]:
        snap = replace(surface(velocity), epoch=1, topology_version=0, dt=.001)
        stream = "u3-self-declared-fixture-" + label
        adapter = create_registered_kinematic_adapter(
            snap, registration=reg, stream_id=stream, h=.001, particle_dt=.0002)
        assert adapter.registration_digest == vector(reg)["sha256"]
        assert adapter.validate_peer(adapter.binding)
        case = dict(name=label, stream_id=stream, initial_snapshot=native(asdict(snap)),
                    binding=vector(adapter.binding), windows=[])
        for k in range(3):
            offer = adapter.open_window(h=.001, particle_dt=.0002)
            packet = message(offer)
            receipt = adapter.consume(packet)
            assert receipt["window_index"] == k
            assert not any(receipt[x] for x in ("authenticated","external_process_observed",
                       "external_files_verified","config_verified","device_solve","real_coupling"))
            case["windows"].append(dict(impulse_window=native(asdict(packet)),
                receipt=vector(receipt), committed_snapshot=native(asdict(adapter.snapshot))))
        cases.append(case)
    root = Path(__file__).resolve().parents[1]
    paths = ["tools/export_u3_contract_vectors.py",
             "project/tests/test_registered_kinematic.py","project/tests/test_kinematic_surface.py",
             "project/coupling/registered_kinematic.py","project/coupling/contracts.py",
             "project/coupling/exchange.py","project/coupling/loads.py"]
    out = dict(schema="xpbd-u3-host-contract-vectors-v1",
        scope="self-declared 4-vertex fixtures; not real MPM evidence, mesh52 or transport testing",
        h=.001, particle_dt=.0002, H_ns=1000000,
        registration=vector(reg), cases=cases,
        scalar_vectors=[vector({"z":-0.0,"a":0.0,"int":0,"float":1.0,"unicode":"é"})],
        source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        python=sys.version, numpy=np.__version__)
    assert not any(n in sys.modules for n in ("taichi","mpm"))
    encoded=(json.dumps(out,ensure_ascii=True,indent=2)+"\n").encode("utf-8")
    assert len(encoded)<250000
    with args.output.open("xb") as f:
        f.write(encoded)
    print(json.dumps(dict(cases=2,windows=6,bytes=len(encoded),
        elapsed_seconds=time.perf_counter()-start,sha256=hashlib.sha256(encoded).hexdigest(),
        forbidden_modules_loaded=[])))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
