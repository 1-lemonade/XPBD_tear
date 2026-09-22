"""Runnable XPBD cloth tearing demo and numerical smoke checks."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

try:
    import taichi as ti
except ImportError:  # pragma: no cover - gives a useful message on setup errors
    ti = None

if __package__:
    from .cloth.constraints import ClothMaterial, XPBDSolver
    from .cloth.failure import FailureModel, StrainThresholdFailure
    from .cloth.mesh import TriangleMesh
    from .cloth.particles import ParticleSoA
    from .cloth.topology import TopologyManager
    from .coupling.interface import NullCoupling
else:  # direct ``python main.py`` execution from project/
    from cloth.constraints import ClothMaterial, XPBDSolver
    from cloth.failure import FailureModel, StrainThresholdFailure
    from cloth.mesh import TriangleMesh
    from cloth.particles import ParticleSoA
    from cloth.topology import TopologyManager
    from coupling.interface import NullCoupling


@dataclass
class DemoConfig:
    width: float = 1.0
    height: float = 1.4
    resolution_x: int = 18
    resolution_y: int = 25
    dt: float = 1.0 / 60.0
    iterations: int = 10
    substeps: int = 8
    critical_strain: float = 0.15
    pull_speed: float = 0.18
    tearing_enabled: bool = True


def make_grid(config: DemoConfig) -> tuple[ParticleSoA, TriangleMesh, list[int], list[int]]:
    nx, ny = config.resolution_x, config.resolution_y
    positions = []
    for row in range(ny):
        y = config.height * (1.0 - row / (ny - 1))
        for column in range(nx):
            x = config.width * column / (nx - 1)
            positions.append((x, y, 0.0))
    positions_array = np.asarray(positions, dtype=np.float64)
    triangles = []
    for row in range(ny - 1):
        for column in range(nx - 1):
            a = row * nx + column
            b = a + 1
            c = a + nx
            d = c + 1
            triangles.extend(((a, c, b), (b, c, d)))
    particles = ParticleSoA(positions_array)
    mesh = TriangleMesh(np.asarray(triangles, dtype=np.int64), particles.position)
    top = list(range(nx))
    bottom = list(range((ny - 1) * nx, ny * nx))
    particles.pin(top)
    return particles, mesh, top, bottom


class ClothSimulation:
    def __init__(self, config: DemoConfig | None = None):
        self.config = config or DemoConfig()
        self.time = 0.0
        self.pull_offset = 0.0
        self.fracture_log: list[dict[str, float | int]] = []
        self.reset()

    def reset(self) -> None:
        self.particles, self.mesh, self.top_ids, self.bottom_ids = make_grid(self.config)
        material = ClothMaterial()
        self.solver = XPBDSolver(self.particles, self.mesh, material)
        self.top_targets = {pid: self.particles.position[pid].copy() for pid in self.top_ids}
        self.bottom_start = {pid: self.particles.position[pid].copy() for pid in self.bottom_ids}
        self._update_pin_constraints()
        self.topology = TopologyManager(self.particles, self.mesh)
        self.failure_model: FailureModel = StrainThresholdFailure(self.config.critical_strain)
        self.coupling = NullCoupling()
        self.time = 0.0
        self.pull_offset = 0.0
        self.fracture_log.clear()

    @property
    def particle_count(self) -> int:
        return self.particles.count

    @property
    def active_edge_count(self) -> int:
        return sum(edge.active for edge in self.mesh.edges)

    def set_failure_model(self, model: FailureModel) -> None:
        self.failure_model = model

    def update_parameters(self, stretch: float, bend: float, critical: float, substeps: int, tearing: bool) -> None:
        self.solver.material.stretch_compliance = max(0.0, float(stretch))
        self.solver.material.bend_compliance = max(0.0, float(bend))
        self.solver.update_material(self.solver.material)
        self.config.critical_strain = max(0.001, float(critical))
        self.config.substeps = max(1, int(substeps))
        self.config.tearing_enabled = bool(tearing)
        if isinstance(self.failure_model, StrainThresholdFailure):
            self.failure_model.critical_strain = self.config.critical_strain

    def step(self) -> None:
        # Pull grows continuously in displacement, never as an impulse.
        self.time += self.config.dt
        self.pull_offset += self.config.pull_speed * self.config.dt
        pin_targets = dict(self.top_targets)
        for pid, start in self.bottom_start.items():
            pin_targets[pid] = start + np.array((0.0, -self.pull_offset, 0.0))
        self.solver.step(self.config.dt, self.config.iterations, self.config.substeps, pin_targets)

        # This is deliberately outside XPBDSolver and runs only after a full
        # substep sequence; no topology mutates inside a constraint iteration.
        if self.config.tearing_enabled:
            self._fracture_one_edge()
        self.coupling.exchange(self.particles, self.time, self.config.dt)

    def _fracture_one_edge(self) -> None:
        candidates = []
        for edge in self.mesh.edges:
            if not edge.active or len(edge.adjacent_triangles) != 2:
                continue
            edge.current_strain = edge.strain(self.particles.position)
            if self.failure_model.should_break(edge):
                candidates.append((edge.current_strain, edge.id, edge))
        if not candidates:
            return
        strain, edge_id, edge = max(candidates, key=lambda item: item[0])
        split_ids = self.topology.split_vertex(edge.a, edge_id)
        if not split_ids:
            return
        self.solver.rebuild_constraints()
        # Preserve pin constraints after the topology rebuild and keep the
        # prescribed boundary state deterministic.
        self._update_pin_constraints()
        self.fracture_log.append({"time": self.time, "edge": edge_id, "strain": strain})

    def _update_pin_constraints(self) -> None:
        targets = dict(self.top_targets)
        for pid, start in self.bottom_start.items():
            targets[pid] = start.copy()
        self.solver.set_pins(targets)

    def finite(self) -> bool:
        return bool(
            np.isfinite(self.particles.position).all()
            and np.isfinite(self.particles.velocity).all()
            and np.isfinite(self.particles.predicted_position).all()
        )

    def disconnected_components(self) -> int:
        """Count triangle components across active shared edges."""
        if self.mesh.triangle_count == 0:
            return 0
        parent = list(range(self.mesh.triangle_count))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for edge in self.mesh.edges:
            if edge.active and len(edge.adjacent_triangles) == 2:
                union(edge.adjacent_triangles[0], edge.adjacent_triangles[1])
        return len({find(i) for i in range(self.mesh.triangle_count)})

    def render_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = self.particles.position.copy()
        x = 0.08 + positions[:, 0] / max(self.config.width, 1e-12) * 0.62
        y = 0.10 + np.clip(positions[:, 1] / max(self.config.height + self.pull_offset, 0.1), -0.05, 1.0) * 0.80
        vertices = np.column_stack((x, y, np.zeros(len(x))))
        colors = np.tile(np.array([[0.15, 0.55, 0.95, 1.0]], dtype=np.float32), (len(x), 1))
        for edge in self.mesh.edges:
            if edge.broken:
                colors[edge.a, :3] = (0.95, 0.18, 0.12)
                colors[edge.b, :3] = (0.95, 0.18, 0.12)
        return vertices.astype(np.float32), self.mesh.triangles.astype(np.int32), colors


def initialize_taichi() -> str:
    if ti is None:
        raise RuntimeError("Taichi is not installed. Create the XPBD_tear environment from environment.yml first.")
    try:
        ti.init(arch=ti.gpu, offline_cache=False, log_level=ti.ERROR)
        return "gpu"
    except Exception:
        ti.reset()
        ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
        return "cpu"


def run_smoke_test(frames: int = 1200) -> ClothSimulation:
    config = DemoConfig(tearing_enabled=True, critical_strain=0.65, substeps=8)
    simulation = ClothSimulation(config)
    for _ in range(frames):
        simulation.step()
        if not simulation.finite():
            raise AssertionError("cloth produced NaN or infinity")
    if simulation.particle_count <= 0 or simulation.active_edge_count <= 0:
        raise AssertionError("invalid cloth topology after smoke run")
    print(
        f"smoke ok: particles={simulation.particle_count}, "
        f"active_edges={simulation.active_edge_count}, fractures={len(simulation.fracture_log)}, "
        f"components={simulation.disconnected_components()}"
    )
    return simulation


def run_headless_artifacts(
    frames: int,
    output_dir: str,
    resolution: int,
    critical_strain: float,
    pull_speed: float,
    render: bool,
) -> None:
    """Run without Taichi windowing and write plots/topology/render artifacts."""
    if __package__:
        from .viz.diagnostics import DiagnosticsRecorder, write_diagnostics
        from .viz.scene_render import render_scene
    else:  # direct ``python main.py`` execution from project/
        from viz.diagnostics import DiagnosticsRecorder, write_diagnostics
        from viz.scene_render import render_scene

    config = DemoConfig(
        resolution_x=max(4, int(resolution)),
        resolution_y=max(6, int(round(max(4, int(resolution)) * 1.4))),
        critical_strain=float(critical_strain),
        pull_speed=float(pull_speed),
    )
    simulation = ClothSimulation(config)
    recorder = DiagnosticsRecorder()
    recorder.record(simulation)
    for _ in range(max(0, int(frames))):
        simulation.step()
        if not simulation.finite():
            raise AssertionError("cloth produced NaN or infinity")
        recorder.record(simulation)
    outputs = write_diagnostics(recorder, output_dir)
    backend = None
    if render:
        backend = render_scene(simulation, f"{output_dir}/cloth_final.png")
    print(f"diagnostics written to {output_dir}")
    print(f"topology check: {recorder.topology_check()}")
    if backend:
        print(f"scene renderer: {backend}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


def run_gif_demo(
    frames: int,
    output_path: str,
    resolution: int,
    critical_strain: float,
    pull_speed: float,
    fps: int,
) -> None:
    """Write a short animated visualization without changing the solver."""
    if __package__:
        from .viz.gif_demo import write_gif
    else:  # direct ``python main.py`` execution from project/
        from viz.gif_demo import write_gif

    resolution = max(4, int(resolution))
    config = DemoConfig(
        resolution_x=resolution,
        resolution_y=max(6, int(round(resolution * 1.4))),
        critical_strain=float(critical_strain),
        pull_speed=float(pull_speed),
    )
    simulation = ClothSimulation(config)
    path = write_gif(simulation, output_path, frames=max(1, int(frames)), fps=max(1, int(fps)))
    print(f"gif written: {path}")


def run_demo() -> None:
    initialize_taichi()
    simulation = ClothSimulation()
    window = ti.ui.Window("XPBD Cloth Tearing Demo", (1180, 760), vsync=True)
    canvas = window.get_canvas()
    paused = False
    single_step = False
    stretch = simulation.solver.material.stretch_compliance
    bend = simulation.solver.material.bend_compliance
    critical = simulation.config.critical_strain
    substeps = simulation.config.substeps
    tearing = simulation.config.tearing_enabled
    resolution = simulation.config.resolution_x
    while window.running:
        gui = window.get_gui()
        gui.begin("XPBD controls", 0.73, 0.03, 0.25, 0.52)
        gui.text(f"time: {simulation.time:.2f}s")
        gui.text(f"fractures: {len(simulation.fracture_log)}")
        gui.text(f"particles: {simulation.particle_count}")
        gui.text(f"components: {simulation.disconnected_components()}")
        stretch = gui.slider_float("stretch compliance", float(stretch), 1.0e-9, 2.0e-5)
        bend = gui.slider_float("bend compliance", float(bend), 1.0e-9, 5.0e-4)
        critical = gui.slider_float("critical strain", float(critical), 0.05, 1.5)
        substeps = int(gui.slider_int("substeps", int(substeps), 1, 12))
        resolution_new = int(gui.slider_int("mesh resolution", int(resolution), 8, 30))
        if resolution_new != resolution:
            resolution = resolution_new
            simulation.config.resolution_x = resolution
            simulation.config.resolution_y = max(10, int(round(resolution * 1.4)))
            simulation.reset()
        if gui.button("Tearing on / off"):
            tearing = not tearing
        if gui.button("Pause / resume"):
            paused = not paused
        if gui.button("Step"):
            single_step = True
        if gui.button("Reset"):
            simulation.reset()
            paused = False
        gui.text(f"tearing: {tearing}   paused: {paused}")
        gui.end()

        simulation.update_parameters(stretch, bend, critical, substeps, tearing)
        if not paused or single_step:
            simulation.step()
            single_step = False
        vertices, indices, colors = simulation.render_data()
        canvas.set_background_color((0.035, 0.045, 0.075))
        canvas.triangles(vertices, indices, per_vertex_color=colors)
        window.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="run a headless numerical stability/fracture test")
    parser.add_argument("--frames", type=int, default=1200, help="smoke-test frame count")
    parser.add_argument("--diagnostics", action="store_true", help="write headless matplotlib diagnostics")
    parser.add_argument("--render-scene", action="store_true", help="write a pyrender scene or 3D fallback PNG")
    parser.add_argument("--gif", action="store_true", help="write a short animated GIF demo")
    parser.add_argument("--gif-output", default="artifacts/xpbd_tearing_demo.gif")
    parser.add_argument("--gif-frames", type=int, default=60)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--output-dir", default="artifacts", help="directory for headless output")
    parser.add_argument("--resolution", type=int, default=18, help="horizontal grid resolution for headless runs")
    parser.add_argument("--critical-strain", type=float, default=0.15)
    parser.add_argument("--pull-speed", type=float, default=0.18)
    args = parser.parse_args(argv)
    if args.smoke_test:
        run_smoke_test(args.frames)
    elif args.gif:
        run_gif_demo(
            args.gif_frames,
            args.gif_output,
            args.resolution,
            args.critical_strain,
            args.pull_speed,
            args.gif_fps,
        )
    elif args.diagnostics or args.render_scene:
        run_headless_artifacts(
            args.frames,
            args.output_dir,
            args.resolution,
            args.critical_strain,
            args.pull_speed,
            args.render_scene,
        )
    else:
        run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
