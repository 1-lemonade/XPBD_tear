"""Taichi start-up: backend, CPU thread cap, precision; reports the backend actually used.

Requesting cuda and getting anything else is an error : no silent fallback.
"""
import platform
import sys

import numpy as np
import taichi as ti


class BackendMismatch(RuntimeError):
    pass


def init_taichi(arch: str = "cpu", precision: str = "f32", cpu_threads=None, **extra) -> dict:
    if arch not in ("cpu", "cuda"):
        raise ValueError(f"arch must be cpu or cuda, got {arch!r}")
    if precision not in ("f32", "f64"):
        raise ValueError(f"precision must be f32 or f64, got {precision!r}")
    kw = dict(arch=ti.cuda if arch == "cuda" else ti.cpu,
              default_fp=ti.f64 if precision == "f64" else ti.f32, log_level=ti.WARN)
    if cpu_threads:
        kw["cpu_max_num_threads"] = int(cpu_threads)
    kw.update(extra)
    ti.init(**kw)
    actual = str(ti.lang.impl.current_cfg().arch)
    info = dict(requested_arch=arch, actual_arch=actual, cpu_threads=cpu_threads, precision=precision,
                taichi=".".join(map(str, ti.__version__)), python=sys.version.split()[0],
                numpy=np.__version__, platform=platform.platform(), init_extra={k: str(v) for k, v in extra.items()})
    print(f"[runtime] requested={arch} actual={actual} cpu_threads={cpu_threads} precision={precision}")
    expected = ("cuda",) if arch == "cuda" else ("x64", "arm64")   # ti.cpu reports the host ISA
    if not any(k in actual.lower() for k in expected):
        raise BackendMismatch(f"requested {arch} but Taichi started on {actual}")
    return info
