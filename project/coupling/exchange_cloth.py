"""Native XY-cloth endpoint of the sequential, synthetic U11 exchange.

One call consumes an integrated world-frame impulse and advances one physical
XPBD substep. The loaded-cloth driver owns mass, pin and topology rebuilding.
Its intermediate solved publication is private to the endpoint during a cut.
"""

from dataclasses import asdict
import math

import numpy as np

from ..loaded_cloth import LoadedClothConfig, LoadedClothDriver
from ..support_balance import build_balance
from .contracts import _real
from .loads import SourceIdentity
from .tracking_topology import archive_cut_edges


def _device_receipt(state, ids):
    return dict(vertex_ids=ids.tolist(), **{key: value.tolist() for key, value in state.items()})


class ExchangeCloth:
    """One CPU f64, 3x3 physical endpoint. Calls are sequential, without resume."""

    def __init__(self, *, source, stream_id=None, cut_after_window=None, cut_edge=(1, 3)):
        if not isinstance(source, SourceIdentity):
            raise TypeError("exchange source must be SourceIdentity")
        if cut_after_window is not None and (type(cut_after_window) is not int or cut_after_window < 1):
            raise ValueError("cut_after_window must be a positive one-based window")
        # The outer session controls window count. The prescribed-cut schedule
        # is validated by the same driver that owns the actual topology change.
        config = LoadedClothConfig(resolution=3, h=.001, iterations=96, convergence=True,
                                   support_balance=True, traction=(0., 0., 0.),
                                   steps=cut_after_window or 1, prescribed_cut_after_step=cut_after_window,
                                   cut_edge=cut_edge if cut_after_window is not None else None)
        self._source = source
        self._driver = LoadedClothDriver(config, stream_id=stream_id)
        self._snapshot = self._driver.current
        self._cut_edges = frozenset()

    @property
    def source(self):
        return self._source

    @property
    def stream_id(self):
        return self._driver.stream_id

    @property
    def failed(self):
        return self._driver.failed

    @property
    def snapshot(self):
        if self.failed:
            raise RuntimeError("failed exchange endpoint cannot publish a snapshot")
        return self._snapshot

    @property
    def cut_edges(self):
        if self.failed:
            raise RuntimeError("failed exchange endpoint cannot publish cut edges")
        return self._cut_edges

    @staticmethod
    def validate_h(h):
        h = _real(h, "next XPBD h", positive=True)
        if h < 1.e-8 or h > math.sqrt(np.finfo(np.float64).max):
            raise ValueError("next XPBD h is outside the solver convergence clock range")
        return h

    def advance(self, batch, h):
        """Consume a LoadBatch against the frozen surface, then publish a receipt.

        Ledger validation precedes the upload callback and remains retryable.
        Any error after entering that callback invalidates this endpoint; the
        device or topology manager may already have applied a partial change.
        """
        driver = self._driver
        initial = self.snapshot
        solver = driver.solver
        h = self.validate_h(h)
        entered_upload = False
        before = uploaded = None
        dense = np.zeros_like(initial.velocity)

        def upload(rows, values):
            nonlocal entered_upload, before, uploaded
            entered_upload = True
            before = solver.read_device_particles()
            dense[rows] = values
            solver.apply_impulses(rows, values)
            uploaded = solver.read_device_particles()

        try:
            driver.ledger.consume(batch, initial, source=self._source, h=h,
                                  capacity=solver.vertex_capacity, apply=upload)
            measured = h == .001
            solver.substep(h, driver.config.iterations, convergence=True, measure_balance=measured)
            driver.last_h = h
            driver.time, driver.step_index = batch.t_end, initial.step_index + 1
            driver.stage = "post_solve"
            driver._publish()
            solved_surface = driver.current
            solved = solver.read_device_particles()
            convergence = solver.last_convergence
            balance = measurements = None
            if measured:
                balance, raw = build_balance(
                    dict(before, vertex_ids=initial.vertex_ids), dict(solved, vertex_ids=solved_surface.vertex_ids),
                    dense, solver.read_projection_balance(), h=h, gravity=driver.config.gravity,
                    damping=driver.config.damping, start_version=initial.version, end_version=solved_surface.version,
                    t_start=batch.t_start, t_end=batch.t_end)
                measurements = {name: data.tolist() for name, data in raw.items()}
            tear = driver.tear_if_requested()
            committed = driver.current
            edges = self._cut_edges if tear is None else archive_cut_edges(tear)
            from .exchange import validate_surface
            validate_surface(committed, edges)
            committed_state = solved if tear is None else solver.read_device_particles()
            receipt = dict(
                schema="xpbd-exchange-cloth-v1", stream_id=self.stream_id, source=asdict(self._source),
                batch_id=batch.batch_id, t_start=batch.t_start, t_end=batch.t_end, h=h, previous_dt=initial.dt,
                start_version=list(initial.version), solved_version=list(solved_surface.version),
                committed_version=list(committed.version), vertex_ids=initial.vertex_ids.tolist(), impulse=dense.tolist(),
                batch_vertex_ids=batch.vertex_ids.tolist(), batch_impulse=batch.impulse.tolist(),
                before=_device_receipt(before, initial.vertex_ids), uploaded=_device_receipt(uploaded, initial.vertex_ids),
                solved=_device_receipt(solved, solved_surface.vertex_ids),
                committed=_device_receipt(committed_state, committed.vertex_ids),
                triangle_ids=committed.triangle_ids.tolist(), triangles=committed.triangles.tolist(),
                root_vertex_ids=driver.reference.root_vertex_ids.tolist(),
                balance=balance, balance_arrays=measurements, convergence=convergence,
                event=None if tear is None else tear.metadata["event"], cut_edges=[list(pair) for pair in sorted(edges)])
            self._snapshot, self._cut_edges = committed, edges
            return receipt
        except Exception:
            if entered_upload:
                driver.failed = True
            raise
