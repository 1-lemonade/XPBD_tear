import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pytest
import taichi as ti


@pytest.fixture(autouse=True)
def taichi_cpu():
    # Bound CPU threads for the test fixtures.
    ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=0, log_level=ti.WARN, cpu_max_num_threads=3)
    yield
