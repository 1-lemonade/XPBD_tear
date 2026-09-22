# Project Brief: XPBD Cloth Tearing Demo

## 1. What this project is

Build a research-grade demo that simulates **tensile tearing of cloth** using
**XPBD (Extended Position-Based Dynamics)**. A cloth mesh is fixed on one edge
and progressively stretched on another. As local strain exceeds a threshold,
the mesh should visibly tear: the affected vertex splits, adjacent triangles
are reassigned, and the constraint set is rebuilt — producing a real crack,
not just a "softened" patch of cloth.

This is **Phase 1** of a longer-term research direction (XPBD cloth ↔ MPM
soft-body coupling, inspired by SoftMAC's independent-dynamics-plus-coupling-
layer architecture). Phase 1 must stand alone as a complete, correct,
demonstrable result. It must **not** depend on MPM to be considered done.

## 2. Explicit non-goals for this phase

Do **not** implement, stub out physics for, or spend design effort on:
- MPM particles, grids, or any P2G/G2P transfer
- Any contact/coupling force computation
- Differentiability / autodiff / gradient graphs

The only obligation regarding future coupling is **architectural**: don't
paint the code into a corner. See Section 4.

## 3. Functional requirements

1. **Cloth representation**: particles + triangle mesh + edges, with
   adjacency information (vertex→triangles, vertex→edges).
2. **XPBD solver** implementing at minimum:
   - Distance (stretch) constraints
   - Bending constraints
   - Pin/fixed constraints (for the anchored edge)
   - Gravity and basic damping
   - Standard XPBD substep loop: predict → solve constraints (iterate) →
     update velocity
3. **Strain measurement**: per-edge engineering strain
   `ε = (l_current - l_rest) / l_rest`, recomputed each step (or substep).
4. **Fracture criterion**: a pluggable check, `ε > ε_crit ⇒ fracture`
   (see FailureModel in Section 4 — must be swappable, not hardcoded into
   the solver).
5. **Real tearing via vertex splitting**, not edge deletion alone:
   - Duplicate the shared vertex into two vertices
   - Reassign one side's triangles to the new vertex
   - Remove the broken constraint
   - Rebuild/regenerate local bending constraints around the split
   - Topology changes happen **between** substeps, never mid-iteration
6. **Driving scenario**: cloth pinned on one edge, opposite edge (or a
   grabbed vertex/region) displaced with a slowly increasing offset (not an
   instantaneous force) so the failure sequence is visible:
   elastic deformation → localized high strain → crack nucleation → crack
   propagation → full separation.
7. **Live-tunable parameters** via on-screen UI: stretch compliance, bend
   compliance, critical strain, substep count, tearing on/off toggle,
   reset/pause/step controls.
8. **Visualization**: real-time render of the mesh, showing the emerging
   tear geometrically (not just a color change).

## 4. Architectural requirements (non-negotiable, even though MPM is out of scope)

- **Separate three concerns** that are currently conflated in most PBD
  tearing tutorials:
  - `ClothMaterial` — material parameters
  - `FailureModel` — `should_break(edge) -> bool`, swappable strategy
  - `TopologyManager` — `split_vertex()`, `remove_edge()`, `rebuild_adjacency()`
- The XPBD solver must **never** reference fracture logic, MPM types, or
  coupling types directly.
- Add a `CouplingInterface` with a single `NullCoupling` implementation that
  does nothing. This is a stub for later — do not build anything behind it
  yet, just leave the seam.
- Prefer struct-of-arrays / contiguous buffers for particle data
  (`position[]`, `velocity[]`, `mass[]`, ...) over one-object-per-particle,
  since this will matter if/when the solver moves to GPU.

## 5. Acceptance criteria

The demo is **done** when all of the following hold:

- [ ] A cloth mesh hangs and settles under gravity with no explosion/NaNs
      over a multi-minute run (basic stability check).
- [ ] Applying a slow, increasing displacement to a pulled edge produces
      visibly increasing strain on the affected edges before any fracture.
- [ ] When `ε > ε_crit` on some edge, that edge's constraint is removed and
      the shared vertex is split — verified by: (a) a visible gap opens in
      the mesh at that location, not just an invisible constraint removal,
      and (b) the two resulting mesh regions are provably disconnected in
      the adjacency structure (not just visually).
- [ ] Tearing propagates naturally under continued displacement (neighboring
      edges fail in sequence as stress redistributes) without manual
      per-frame intervention.
- [ ] Changing `critical_strain`, `compliance`, or mesh resolution via the
      UI changes tearing behavior in the expected direction, live, without
      restart.
- [ ] The solver does not crash, deadlock, or produce invalid topology
      (e.g. dangling constraint references to deleted vertices) across at
      least a few dozen fracture events in one run.
- [ ] `FailureModel` can be swapped for a different criterion (e.g. a
      constant-always-false stub) without touching solver code, demonstrating
      the separation from Section 4 actually holds.
- [ ] `CouplingInterface` exists and is wired in as a no-op, without any
      MPM-specific code anywhere in the cloth/solver path.

## 6. Suggested stack

Python + Taichi (SNode-based dynamic fields handle the runtime topology
mutation better than hand-rolled GPU buffers at this stage; GPU execution and
future autodiff/MPM examples are already available in the Taichi ecosystem).
CPU backend first for correctness, GPU backend once stable. Revisit
C++/CUDA only if a concrete performance ceiling is hit later — do not
pre-optimize for that now.

## 7. Deliverable format

- Working code with a runnable entry point and a short README describing
  how to run it and what the UI controls do.
- Brief note (a few sentences) on which of the acceptance criteria above
  were validated and how (screenshot/recording/log is fine — no formal test
  suite required for Phase 1).
