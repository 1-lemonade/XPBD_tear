"""Read-only coupling integration checks, run in one backend per process."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import pickle
import unittest

import numpy as np

from project.coupling import (ClothSnapshot, QueryPoints, MappingConfig,
                              MappingStatus, QueryOnlyCoupling, NullCoupling)
from project.coupling.interface import validate_exchange
from project.main import ClothSimulation, DemoConfig
from project.cloth.taichi_runtime import initialize_taichi


def live_state_bytes(simulation) -> bytes:
    """Include all host particle/mesh state and every allocated solver field.

    This intentionally observes capacity tails as well as active constraints;
    a mapper has no reason to mutate either. It is an oracle for zero writes,
    not for the correctness of the closest-point calculation.
    """
    solver = simulation.solver
    device = {name: value.to_numpy() for name, value in vars(solver).items()
              if hasattr(value, "to_numpy")}
    host = {name: value for name, value in vars(solver).items()
            if name not in device and name not in ("particles", "mesh") and not callable(value)}
    return pickle.dumps((
        vars(simulation.particles), vars(simulation.mesh),
        host, device, simulation.config, simulation.top_ids, simulation.bottom_ids,
        simulation.top_targets, simulation.bottom_start,
        simulation.time, simulation.pull_offset, simulation.fracture_log,
        simulation.epoch, simulation.step_index,
    ), protocol=5)


def array_fields(value):
    assert is_dataclass(value)
    return {field.name: getattr(value, field.name) for field in fields(value)
            if isinstance(getattr(value, field.name), np.ndarray)}


def assert_readonly_independent(case, value, sources=()):
    for name, array in array_fields(value).items():
        case.assertFalse(array.flags.writeable, name)
        for source in sources:
            case.assertFalse(np.shares_memory(array, source), name)
        if array.size:
            with case.assertRaises(ValueError, msg=name):
                array.flat[0] = array.flat[0]


def triangle_snapshot(**changes):
    values = dict(positions=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                  triangles=np.array([[0, 1, 2]]), vertex_ids=np.array([10, 30, 70]),
                  triangle_ids=np.array([91]), velocity=np.zeros((3, 3)),
                  inverse_mass=np.ones(3), pinned=np.zeros(3, dtype=bool),
                  epoch=1, step_index=0, topology_version=1, time=0., dt=1/60)
    values.update(changes)
    return ClothSnapshot(**values)


class ContractTests(unittest.TestCase):
    def test_snapshot_and_query_input_validation(self):
        bad = [dict(triangles=[[-1, 1, 2]]), dict(triangles=[[0, 1, 3]]),
               dict(triangles=[[0., 1., 2.]]), dict(vertex_ids=[1, 1, 2]),
               dict(vertex_ids=np.array([0, 1, 2**64-1], dtype=np.uint64)),
               dict(vertex_ids=[True, False, True]), dict(triangle_ids=[-1]),
               dict(inverse_mass=[1, -1, 1]), dict(positions=np.full((3, 3), np.nan)),
               dict(velocity=np.full((3, 3), np.inf)), dict(positions=[]),
               dict(epoch=True), dict(step_index=0.5), dict(dt=0), dict(time=-1)]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(ValueError):
                triangle_snapshot(**change)
        for ids, positions in (([1, 1], np.zeros((2, 3))), ([1.5], np.zeros((1, 3))),
                               ([True], np.zeros((1, 3))), ([0], [[np.inf, 0, 0]]),
                               ([0], [[0, 0]]), ([0], [])):
            with self.assertRaises(ValueError):
                QueryPoints(ids, positions)
        for name in ("area_rtol", "tie_atol", "tie_rtol", "max_distance"):
            for value in (-1, np.nan, np.inf, True, 10**1000):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    MappingConfig(**{name: value})
        with self.assertRaises(ValueError):
            MappingConfig(area_rtol=0)
        q = QueryPoints(np.array([], dtype=np.int64), np.empty((0, 3)))
        self.assertEqual(q.positions.shape, (0, 3))
        triangle_snapshot(triangles=[[0, 0, 1]])  # legal but degenerate

    def test_copy_and_time_validation(self):
        positions = np.zeros((1, 3))
        q = QueryPoints([-1], positions)
        positions[0, 0] = 123
        self.assertEqual(q.positions[0, 0], 0)
        assert_readonly_independent(self, q, [positions])
        time = 0.
        for _ in range(6):
            time += 1/60
        snap = triangle_snapshot(step_index=6, time=time)
        validate_exchange(snap, time, snap.dt)
        with self.assertRaises(ValueError):
            validate_exchange(snap, 6/60, snap.dt)
        with self.assertRaises(TypeError):
            validate_exchange(object(), time, snap.dt)
        with self.assertRaises(AttributeError):
            snap.epoch = 4


class IntegrationTests(unittest.TestCase):
    arch = "cpu"
    precision = "f32"

    def make_sim(self, **changes):
        values = dict(resolution_x=4, resolution_y=6, arch=self.arch, precision=self.precision)
        values.update(changes)
        return ClothSimulation(DemoConfig(**values))

    def test_isolation_zero_writes_and_reset(self):
        sim = self.make_sim(tearing_enabled=False)
        queries = QueryPoints([92, -1, 701], [[0.2, 0.3, .2], [.8, .7, -.1], [2, 2, 0]])
        coupling = QueryOnlyCoupling(queries)
        self.assertFalse(np.shares_memory(coupling.queries.positions, queries.positions))
        sim.coupling = coupling
        before = live_state_bytes(sim)
        snapshot = sim.snapshot()
        self.assertEqual(before, live_state_bytes(sim), "snapshot wrote live state")
        self.assertEqual(snapshot.version, (1, 0, sim.mesh.topology_version))
        np.testing.assert_array_equal(snapshot.positions, sim.particles.position)
        assert_readonly_independent(self, snapshot, list(vars(sim.particles).values()) + [sim.mesh.triangles])
        saved = {name: a.copy() for name, a in array_fields(snapshot).items()}
        coupling.exchange(snapshot, snapshot.time, snapshot.dt)
        self.assertEqual(before, live_state_bytes(sim), "exchange wrote live state")
        result = coupling.last_result
        self.assertEqual(len(result.query_ids), 3)
        np.testing.assert_array_equal(result.query_ids, queries.query_ids)
        assert_readonly_independent(self, result, list(array_fields(snapshot).values()) + list(array_fields(queries).values()))
        for step in range(1, 7):
            sim.step()
            self.assertEqual(sim.step_index, step)
            self.assertEqual(coupling.last_result.version, sim.snapshot().version)
            self.assertEqual(sim.mesh.topology_version, snapshot.topology_version)
        with self.assertRaises(ValueError):
            coupling.exchange(sim.snapshot(), -1, sim.config.dt)
        self.assertIsNone(coupling.last_result)
        sim.step()
        sim.reset()
        self.assertIsNone(coupling.last_result)
        self.assertIsInstance(sim.coupling, NullCoupling)
        self.assertEqual((sim.epoch, sim.step_index), (2, 0))
        for name, value in saved.items():
            np.testing.assert_array_equal(getattr(snapshot, name), value)

    def test_mismatch_never_repairs_solver(self):
        sim = self.make_sim()
        sim.mesh.rebuild_adjacency(sim.particles.position)
        before = live_state_bytes(sim)
        with self.assertRaisesRegex(RuntimeError, "topology mismatch"):
            sim.snapshot()
        self.assertEqual(before, live_state_bytes(sim))

    def test_zero_write_oracle_detects_host_schedule_mutation(self):
        sim = self.make_sim()
        before = live_state_bytes(sim)
        groups = sim.solver.stretch_color_groups
        sim.solver.stretch_color_groups = []
        self.assertNotEqual(before, live_state_bytes(sim))
        sim.solver.stretch_color_groups = groups
        self.assertEqual(before, live_state_bytes(sim))
        particle = sim.bottom_ids[0]
        original = sim.bottom_start[particle].copy()
        sim.bottom_start[particle][1] += .125
        self.assertNotEqual(before, live_state_bytes(sim))
        sim.bottom_start[particle][:] = original
        self.assertEqual(before, live_state_bytes(sim))

    def test_real_tears_and_disconnected_regions(self):
        sim = self.make_sim(resolution_x=8, resolution_y=10, critical_strain=.05,
                            pull_speed=1., substeps=3)
        queries = QueryPoints([800, 2, -7, 81, 99], [[.1, .5, .1], [.3, .6, -.1],
                                                   [.5, .7, 0], [.8, .9, 0], [2, 2, 2]])
        coupling = QueryOnlyCoupling(queries)
        sim.coupling = coupling
        old = sim.snapshot()
        saved = {name: a.copy() for name, a in array_fields(old).items()}
        events = 0
        for frame in range(1, 81):
            sim.step()
            snapshot = sim.snapshot()
            result = coupling.last_result
            self.assertEqual(result.version, snapshot.version)
            self.assertEqual(result.step_index, frame)
            for qi, tid in enumerate(result.triangle_ids):
                row = int(np.flatnonzero(snapshot.triangle_ids == tid)[0])
                vertex_rows = snapshot.triangles[row]
                np.testing.assert_array_equal(result.vertex_ids[qi], snapshot.vertex_ids[vertex_rows])
                np.testing.assert_allclose(result.barycentric[qi] @ snapshot.positions[vertex_rows],
                                           result.closest_point[qi], atol=1e-9, rtol=1e-9)
            if len(sim.fracture_log) > events:
                events = len(sim.fracture_log)
                before = live_state_bytes(sim)
                coupling.exchange(snapshot, snapshot.time, snapshot.dt)
                self.assertEqual(before, live_state_bytes(sim))
                self.assertEqual(snapshot.topology_version, sim.solver.topology_version)
        self.assertGreater(events, 0)
        self.assertGreater(sim.disconnected_components(), 1)
        print(f"low-threshold tears: backend={self.arch}/{self.precision}, "
              f"events={events}, vertices={sim.particle_count}, "
              f"components={sim.disconnected_components()}, version={sim.snapshot().version}")
        for name, value in saved.items():
            np.testing.assert_array_equal(getattr(old, name), value)

    def test_analytic_crack_and_reordered_topology(self):
        # A gap of width 2, separate vertex identities, no triangle spans it.
        from project.coupling.mapping import map_points
        positions = np.array([[-2, 0, 0], [-1, 0, 0], [-1, 1, 0],
                              [1, 0, 0], [2, 0, 0], [1, 1, 0]], dtype=float)
        kwargs = dict(positions=positions, triangles=[[0, 1, 2], [3, 4, 5]],
                      vertex_ids=[10, 11, 12, 20, 21, 22], triangle_ids=[90, 3],
                      velocity=np.zeros((6, 3)), inverse_mass=np.ones(6), pinned=np.zeros(6, bool))
        snap = triangle_snapshot(**kwargs)
        queries = QueryPoints([8, -1, 50], [[-.75, .25, .5], [.75, .25, .5], [0, .25, .5]])
        result = map_points(snap, queries)
        np.testing.assert_array_equal(result.triangle_ids, [90, 3, 3])
        np.testing.assert_allclose(result.closest_point, [[-1, .25, 0], [1, .25, 0], [1, .25, 0]], atol=1e-12)
        np.testing.assert_allclose(result.distance, [np.sqrt(.3125), np.sqrt(.3125), np.sqrt(1.25)], atol=1e-12)
        kwargs.update(triangles=[[3, 4, 5], [0, 1, 2]], triangle_ids=[3, 90], step_index=1, topology_version=2)
        new = triangle_snapshot(**kwargs)
        updated = map_points(new, queries)
        np.testing.assert_array_equal(updated.triangle_ids, result.triangle_ids)
        np.testing.assert_array_equal(updated.vertex_ids, result.vertex_ids)
        self.assertNotEqual(updated.version, result.version)
        for row in updated.vertex_ids:
            self.assertTrue(set(row) <= {10, 11, 12} or set(row) <= {20, 21, 22})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("f32", "f64"), default="f32")
    parser.add_argument("--contracts-only", action="store_true")
    args = parser.parse_args()
    IntegrationTests.arch, IntegrationTests.precision = args.arch, args.precision
    if args.contracts_only:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    else:
        initialize_taichi(args.arch, args.precision)
        suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
