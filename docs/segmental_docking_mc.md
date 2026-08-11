# Segmental GRAMM-style docking MC with Brownian blending

*Design proposal — recorded 2026-08-10. Author: Karim Ghabra.*

## The problem

We want the docking-based Monte Carlo methodology of Vakser et al. (PNAS 2022)
at "full atom" resolution — and specifically we want collagen molecules that
can **bend**. Bending makes the original GRAMM-docking MC method untenable as
stated: every new bent orientation of a collagen molecule would effectively be
a new molecule, requiring its own precomputed docking landscape. The
conformational space of a flexible chain makes that enumeration explode.

## The proposal: split the problem in two

### 1. Links as the rigid docking bodies

Break a single tropocollagen molecule into **links on a chain**. Each link is
the rigid body of the docking method. Link size is set by the empirical
persistence length, so each link is stiff enough to treat as rigid.

Precompute the GRAMM-style binding energies between **all unique pairs of
links**. For a hypothetical type-Z collagen with 5 links, the unique link-pair
mappings number 5+4+3+2+1 = 15, each providing a
**quaternion → binding energy** lookup used during simulation.

### 2. Superposition + the original MC move sampling

By the principle of superposition, the total intermolecular binding energy is
the sum over interacting link pairs. With that energy in hand, perform the
same Monte Carlo move sampling as the original method.

### Remaining energetic contributions

Binding energy alone is not sufficient. The Hamiltonian also needs:

- **bending energy** between consecutive links of a chain,
- **sterics** (excluded volume),
- **prevention of chains moving through each other** (topology protection).

### How molecules bend during simulation — two candidate schemes

(a) **Whole-molecule moves + injected Brownian behavior**: sample rigid-body
moves of entire collagen molecules as before, and inject Brownian dynamics of
the individual links between/around those moves.

(b) **Per-link moves**: each MC step moves one link, and the remaining links
of the chain respond accordingly (the chain constraint propagates the move).

Either way, this is arguably a **more rigorous and faithful reproduction of
the original GRAMM-docking paper under flexibility** than treating whole
flexible molecules as rigid bodies.

---

## Relation to the current codebase (implementation annotations)

*These notes are commentary on the proposal, not part of it.*

- **The link-pair lookups largely exist.** The registry tables
  U(Δ, φa, φb) are precomputed pairwise energies between locally-straight
  contact windows; restricting Δ to the window implied by a link pair (i, j)
  yields exactly the quaternion→energy lookup the proposal calls for, because
  for quasi-parallel links the relative quaternion reduces to
  (local stagger, facing angles, small tilt). The PMF harness is currently
  measuring these same lookups at all-atom MD resolution; the Δ-learned
  tables would become the "GRAMM computes" of this method — with a physical
  Hamiltonian behind them rather than a score analog.
- **Link count for type I**: L/Lp ≈ 290/60 → 5 links matches the worked
  example; n(n+1)/2 = 15 unique sequence-window pair tables. Finer links
  (more, shorter) trade table count (n(n+1)/2 growth) against rigidity error
  within a link (~(link/2Lp)²).
- **Scheme (a)** corresponds to the hybrid MC + BD-relaxation plan already
  scoped (composition of balance-preserving kernels); **scheme (b)** is the
  classic polymer-MC move set (single-link displacement with crankshaft /
  pivot responses), which needs the acceptance rule to account for the
  chain-response Jacobian — configurational-bias MC machinery.
- **Sterics/topology**: the existing capsule excluded-volume machinery and
  collision checks apply per-link directly.
- Natural home: this replaces the rigid-molecule v1 in the MC (docking) tab.

## As built (v0.3.0)

Implemented in `src/mcdock.{h,cpp}`. Where the build departs from the proposal,
it is because measurement said so:

- **One kernel, not n(n+1)/2 tables.** The proposal's per-link-pair lookups are
  the same function evaluated on different contour windows, so the build keeps
  a single merged kernel U(Δ, φa, φb) and passes each link pair its own contour
  offsets. Adding links costs nothing in table memory, so link count became a
  live UI slider instead of a compile-time commitment.
- **Contact integral instead of one quaternion→energy lookup per pair.** Nine
  arc-length samples per link pair, each with its own local stagger and facing
  angles. This is what makes bent and tilted contacts price correctly — and it
  follows from the tilt series, which found no universal angular falloff to
  put in a lookup: the angular physics is already in the stagger ramp.
- **Scheme (b) is primary, with rigid transport retained.** Per-link dock hops
  and pivots do the conformational work as proposed. Whole-molecule rigid moves
  are kept for transport only: center-of-mass diffusion built from link moves
  alone is Rouse-like (~N² sweeps to move one molecular length) and, unlike a
  rigid translation, has no σ²/6D mapping to give the MC clock a calibration.
  Scheme (a)'s BD-relax interleave is not implemented; the pivot move covers
  the chain relaxation it was meant to provide.
- **The chain-response Jacobian the notes worried about is avoided.** Dock hops
  drag the chain rigidly rather than deforming it, so internal bend energy is
  unchanged by construction and no configurational-bias weights are needed.
  The state-dependent transport width does carry an explicit Hastings ratio.
- **Added move not in the proposal: axial slide.** Translation along a link's
  own tangent, including ±D and ±2D jumps. Registry annealing in a dense gel
  turned out to be the binding constraint — dock hops are almost all rejected
  once the system percolates, because they sweep too much volume.

Verified: an isolated pair of 5-link molecules docks at −1.98 D with all five
link contacts agreeing (the rigid v1 parks at zero stagger instead), and the
discrete-WLC hinges reproduce ⟨cos θ⟩ = 0.37 against the exact 0.38.
