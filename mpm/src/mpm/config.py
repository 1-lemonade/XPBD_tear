"""Run configuration: built-in defaults <- scene JSON file <- --set overrides.

A parameter change is a config change, never a code change .
Unknown keys are errors, so a misspelt key in a parameter study cannot be silently
ignored. List items (materials, boundaries) are checked by the module that owns them.
"""
import copy
import json

DEFAULTS = {
    "scene": {},                      # scene-specific; the scene supplies its own defaults
    "precision": "f32",               # f32 | f64, one switch for the whole run
    "runtime": {"cpu_threads": 3,     # Keep the default CPU thread count bounded.
                "kernel_profiler": False,   # per-task kernel times -> profile.json (U2)
                "compile_profile": False,   # print Taichi's scoped (compile) profiler at the end
                "offline_cache": None},     # None = Taichi default; true/false explicit
    "grid": {"domain": 1.0,           # cubic domain edge [m]
             "n": 48,                 # nodes per axis
             "buffer_cells": 3,       # >= quadratic stencil support; default wall offset
             "active_box": True},     # restrict grid loops to the particle bounding box
    "time": {"dt": 2e-4, "cfl_limit": 0.3},
    "gravity": [0.0, -9.81, 0.0],
    "particles": {"ppc": 8},
    "transfer": {"scheme": "mls_apic_quadratic", "tau_cache": True},
    "constitutive": {"model": "dp_sand", "options": {}, "materials": []},
    "boundaries": [],
    "contact": {},                    # P1-U4 contact block; {} = off (keys: mpm/contact/block.py)
}


def _merge(base, over, path=""):
    out = copy.deepcopy(base)
    for k, v in over.items():
        p = f"{path}.{k}" if path else k
        if k not in out:
            raise KeyError(f"unknown config key '{p}'")
        if isinstance(out[k], dict) and isinstance(v, dict) and out[k]:
            out[k] = _merge(out[k], v, p)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_override(cfg, assignment: str):
    """'grid.n=32', 'boundaries.0.mu=0.3', 'constitutive.materials=[...]' (value is JSON if it parses)."""
    if "=" not in assignment:
        raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
    key, text = assignment.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        where = ".".join(parts[:i + 1])
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f"unknown config key '{where}'")
    last = parts[-1]
    value = _parse_value(text)
    if isinstance(node, list):
        node[int(last)] = value
    elif isinstance(node, dict):
        # dict-valued sections only accept known keys; list items (dicts inside lists)
        # are checked by their owning module when the solver is built
        node[last] = value
    else:
        raise KeyError(f"cannot set '{key}': parent is not a section")
    return cfg


def load_config(path=None, overrides=(), scene_defaults=None) -> dict:
    base = copy.deepcopy(DEFAULTS)
    base["scene"] = copy.deepcopy(scene_defaults or {})
    cfg = base
    if path:
        with open(path) as fh:
            cfg = _merge(base, json.load(fh))
    for a in overrides:
        apply_override(cfg, a)
    _merge(base, cfg)   # re-check: an override may have introduced an unknown section key
    return cfg


def check_keys(where: str, spec: dict, allowed) -> None:
    unknown = sorted(set(spec) - set(allowed))
    if unknown:
        raise KeyError(f"{where}: unknown key(s) {unknown}; allowed {sorted(allowed)}")
