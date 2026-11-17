"""Contact modules (P1-U4). One contact block per run, configured by the `contact` section
({} = off). Interface used by the solver (fixed; a replacement block keeps it):

    build(spec, grid, transfer, dtype, n_particles, ppc) -> block or None
    block.state_members() -> dict     particle members (registered before the field is built)
    block.bind(sim)                   solver fields the kernels use (P, dt, time, gravity)
    block.n_sub                       substeps per exchange period H
    block.begin_period() / end_period()   python, around each period (kernels inside)
    block.apply()          [ti.func]  top-level loops between the grid update and G2P
    block.STAGES                      labels of those loops, in order (kernel profiler)
    block.JG, block.label             realised grid impulse field = the momentum sink 'contact'
    block.readout() / describe() / set_params(**kw)

The block reaches the cloth only through its surface (surface.build_surface): type sheet /
mesh -> StubSurface (rigid, P1), type live -> LiveSurface (a completed cloth state read
every window by ingest(), P2-U1). Members the block uses: nv, nt, tri, vertex_at, side,
plane_side, degenerate, v_s, t_s, step_index, JP_per, JG_per, pinned, settle_window.
"""
from .block import ContactBlock, build, NONE, CANDIDATE, CONSTRAINED
from .geometry import closest_point, FACE, EDGE0, VERTEX0
from .surface import StubSurface, build_surface, sheet, side_data
from .live import LiveSurface, SurfaceSnapshot
