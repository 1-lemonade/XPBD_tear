"""Particle state as one Taichi struct field (SoA)

The solver owns the base members; other modules (constitutive state, later contact
and export) register their members before build(). Particle order never changes,
so the index is the stable particle ID (export contract v0).
"""
import taichi as ti


class ParticleSchema:
    def __init__(self, dtype):
        self.dtype = dtype
        vec3 = ti.types.vector(3, dtype)
        mat3 = ti.types.matrix(3, 3, dtype)
        self._members = {}
        self._built = False
        self.add("solver", dict(
            x=vec3, v=vec3,          # position [m], velocity [m/s]
            C=mat3,                  # APIC affine velocity
            F=mat3,                  # elastic deformation gradient
            tau=mat3,                # Kirchhoff stress cache (decision D2)
            m=dtype, V0=dtype,       # mass [kg], rest volume [m^3]
            mat_id=ti.i32,           # row of the constitutive parameter table
            flag=ti.i8,              # 0 tip, 1 elastic, 2 plastic (diagnostics, colouring)
        ))

    def add(self, owner: str, members: dict):
        if self._built:
            raise RuntimeError("particle schema already built; register members before build()")
        for name, t in members.items():
            elem = getattr(t, "dtype", t)
            if elem in (ti.f32, ti.f64) and elem != self.dtype:
                raise TypeError(f"particle member '{name}' from {owner} is {elem}, schema precision is {self.dtype}")
            if name in self._members:
                raise ValueError(f"particle member '{name}' from {owner} already registered by {self._members[name][1]}")
            self._members[name] = (t, owner)

    def build(self, n: int):
        self._built = True
        struct = ti.types.struct(**{k: t for k, (t, _) in self._members.items()})
        return struct.field(shape=n, layout=ti.Layout.SOA)

    def describe(self) -> dict:
        return {k: dict(owner=o, type=str(t)) for k, (t, o) in self._members.items()}
