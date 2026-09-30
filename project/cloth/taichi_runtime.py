"""Taichi initialization and requested-backend verification."""

from __future__ import annotations

try:
    import taichi as ti
except ImportError:  # pragma: no cover - setup error is reported at use time
    ti = None


_INITIALIZED: tuple[str, str, str] | None = None


def initialize_taichi(arch: str = "cpu", precision: str = "f32", *, log: bool = True) -> str:
    """Initialize Taichi once and verify it selected the requested backend.

    ``ti.gpu`` is intentionally not used: it may silently select a different
    backend. This solver supports an explicit CPU or CUDA request.
    """
    global _INITIALIZED
    if ti is None:
        raise RuntimeError("Taichi is not installed. Create the XPBD_tear environment from environment.yml first.")

    requested_arch = str(arch).lower()
    requested_precision = str(precision).lower()
    if requested_arch not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported Taichi backend {arch!r}; choose 'cpu' or 'cuda'")
    if requested_precision not in {"f32", "f64"}:
        raise ValueError(f"unsupported Taichi precision {precision!r}; choose 'f32' or 'f64'")

    expected_arch = ti.cpu if requested_arch == "cpu" else ti.cuda
    if _INITIALIZED is None:
        ti.init(
            arch=expected_arch,
            default_fp=ti.f32 if requested_precision == "f32" else ti.f64,
            offline_cache=False,
            log_level=ti.ERROR,
        )
        actual_arch = ti.lang.impl.current_cfg().arch
        if actual_arch != expected_arch:
            ti.reset()
            raise RuntimeError(
                f"Taichi selected {actual_arch} for requested {requested_arch}; refusing backend fallback"
            )
        actual_name = "cpu" if requested_arch == "cpu" else "cuda"
        _INITIALIZED = (requested_arch, requested_precision, actual_name)
    else:
        initialized_arch, initialized_precision, actual_name = _INITIALIZED
        if (requested_arch, requested_precision) != (initialized_arch, initialized_precision):
            raise RuntimeError(
                "Taichi is already initialized as "
                f"{initialized_arch}/{initialized_precision}; requested {requested_arch}/{requested_precision}. "
                "Run each backend or precision validation in a fresh process."
            )

    if log:
        actual_arch = ti.lang.impl.current_cfg().arch
        print(
            f"Taichi backend selected: {actual_name} "
            f"(requested={requested_arch}, actual_arch={actual_arch}, precision={requested_precision})"
        )
    return actual_name
