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
