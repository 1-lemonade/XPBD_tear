"""Constitutive models. Each model is its own module with the same interface:

    state_members() -> dict          members registered into the particle schema
    set_material(mid, **params)      run-time parameter change (table row), no recompilation
    material_info(mid) -> dict       includes rho and wave_speed
    describe() -> dict
    init_state(P, p)        [ti.func]
    update(P, p, F_trial)   [ti.func] -> F_new, tau, flag   (reads/writes only its own state members)
    stress(P, p, F)         [ti.func] -> tau                 (recompute path)

Model choice is structural (one model per run); mat_id indexes that model's parameter table.
"""
from .dp_sand import DruckerPragerSand, KLAR2016_SAND_RECALLED, FLAG_TIP, FLAG_ELASTIC, FLAG_PLASTIC

MODELS = {DruckerPragerSand.name: DruckerPragerSand}


def build(spec: dict, dtype):
    from ..config import check_keys
    check_keys("constitutive", spec, ("model", "options", "materials"))
    if spec["model"] not in MODELS:
        raise KeyError(f"unknown constitutive model {spec['model']!r}; known {sorted(MODELS)}")
    return MODELS[spec["model"]](spec["materials"], dtype, **spec.get("options", {}))
