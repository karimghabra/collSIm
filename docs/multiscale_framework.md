# Hierarchical multiscale framework

*Design — 2026-08-11. Status: rung contract, propagator and rung 2 skeleton built
and passing their gates; every closure measurement still to do.*

## The gap this fills

The molecular engine is as rigorous as we know how to make it and it reaches
microseconds. Fibrillogenesis takes 30–120 minutes. That is a factor of about
10¹⁰, and nothing in the last four versions closed it honestly:

- **BD (fast)** buys speed with a displacement clamp, a force clamp and
  attraction saturation. Those make its invariant distribution a property of the
  clamps rather than of a potential, so it cannot be the bottom of a
  coarse-graining hierarchy — you cannot integrate out degrees of freedom of a
  model whose distribution nobody can write down.
- **BD (rigorous)** fixes exactly that, and pays for it with a 5 ps step.
- **Docking MC** reached further, but its moves are moves. The one number that
  looked like a clock had to be recalibrated twice, and the correction that
  mattered was discovering that only the transport channel may feed it.
- **Weighted ensemble** is the one honest accelerator, and it is variance
  reduction, not acceleration: it redistributes effort, so it buys rare events
  at fixed timescale, not a longer timescale.
- **KMC** reached hours by making molecules into events — and its rates were
  knobs. It is deleted, and this document is what replaces it.

So: no more engines that trade rigour for reach at a fixed level of description.
Reach comes from **changing the level of description**, with every coarse
parameter measured at the level below.

## Where the time actually comes from

Not from taking more steps per second. From taking steps that are worth more
physical time, because the object being moved is intrinsically slower.

An overdamped step is limited by `D·k·dt ≲ 0.1`, where `k` is the stiffest
curvature in the energy and `D` the mobility of the coordinate it acts on.
Coarse-graining changes both, and the DOF count on top:

| rung | stiffest mode | D | D·k | usable dt | DOF per 300 nm of material |
|---|---|---|---|---|---|
| 1 molecule (beads) | bond, 260 kT/nm² | 0.143 nm²/ns | 37 /ns | 5 ps | 288 |
| 2 microfibril | stretch, 30 kT/nm² | 2×10⁻³ nm²/ns | 0.06 /ns | ~1–5 ns | 36 (for 5 molecules) |

Both numbers are measured, not projected: rung 2's self-test runs at **dt = 1 ns
with 95.7% acceptance**, and rung 1's 5 ps is the shipped `dtRig`. That is ~200×
on the step and ~40× on the DOF count per unit of material — about **10⁴ in cost
per unit physical time, per rung**. Two more rungs of similar ratio put hours
within reach of an overnight run.

**The honest caveat.** The coarse process is not the fine process sped up. It is
the fine process *projected*, and a projection is Markovian only when the
integrated-out degrees of freedom relax fast compared to the kept ones. Where
that separation is weak the true coarse dynamics carries a memory kernel
(Mori–Zwanzig) and our memoryless propagator is an approximation. We do not
assert the separation — we test it, in the overlap window, and that test is the
gate below.

## The rung contract

A rung is not "a coarser picture of collagen". It is a precise object:
coordinates `q = M(x)` of the rung below, plus

```
F(q) = -kT ln ∫ exp(-U(x)/kT) δ(M(x) - q) dx        free energy
D(q) = mobility of q, tied to the noise by the FDT   transport
```

`F` and `D` are **properties of the rung below**, not modelling choices. The
only legitimate way to get them is to measure them there. A rung whose `F` was
written down by hand is a phenomenological model wearing a multiscale costume.

Every rung supplies exactly five things (`src/scale/level.h`):

1. `energy(q)` — F, in kT
2. `gradient(q)` — the **exact** analytic gradient of `energy`
3. `diffusion()` — D per coordinate
4. collective moves, each with an honestly computable proposal ratio
5. `provenance()` — where every number came from

Items 3 and 5 are the ones people skip. Item 5 is enforced: each parameter
carries `Measured` / `Derived` / `Literature` / `Assumed`, and any rung holding
an `Assumed` value prints an `EXPLORATORY` banner into its own run log and into
the header of every file it writes. A number without a source is how a hack
survives contact with a reviewer.

## Metropolis–Hastings, and exactly what it buys

Every transition on every rung is Metropolis–Hastings accepted. Two channels,
and conflating them is the single most expensive mistake available here.

### Kinetic channel — MALA

```
μ  = q - D ∇F dt
q' = μ + √(2 D dt) ξ            accepted with min{1, e^{-ΔF} · T(q'→q)/T(q→q')}
```

- **Thermodynamics is exact at any dt.** The invariant measure is `exp(-F/kT)`
  exactly. This is what retires the clamps: rigorous BD needs no `dxMax`, no
  `fMax`, no adaptive-dt guard, because the Metropolis test does their job
  without deforming the distribution. Three hacks, one substitution.
- **Dynamics is exact only as dt → 0.** Every rejection is a deviation from the
  true Langevin path, so any rate read off a MALA run is biased at O(dt) and
  requires a dt-convergence series. This is measured, not asserted — see below.
- **The clock advances on rejected steps too.** A rejection means the system
  stayed put for dt, which is a physical outcome, not a wasted iteration.
- `dt` adaptation is legal **only during equilibration**. While dt moves, the
  kernel is not fixed and detailed balance means nothing. `Mala::freeze()` ends
  it; `MalaStats::adaptedThisRun` marks any run that isn't production-clean.
- Leimkuhler–Matthews noise must be **off** under MALA — it correlates
  consecutive draws, and the Metropolis ratio assumes independence. (The
  rigorous integrator already carries this warning.)

### Sampling channel — collective moves

Registry kinks, insertions, twist flips. These satisfy detailed balance with
respect to `exp(-F/kT)`, so they are **exact for thermodynamics and worthless
for kinetics**. They advance no time at all — the propagator refuses to add it.

To get the *rate* of such a transition, measure its barrier and use
`src/scale/kramers.h`, which integrates the exact overdamped mean-first-passage
formula (no parabolic fit, no transmission coefficient, no fitted prefactor).
That rate is what the rung above consumes as a Gillespie propensity. **That is
the hand-off that replaces the deleted KMC engine**: fine rung measures a
barrier, coarse rung fires an event, and nobody sets `k+` by hand.

Any move whose proposal ratio cannot be stated honestly does not belong in the
move set. The retired docking engine's dock hops had a deterministic placement
with no library reverse; we ended up shipping a switch to disable them. Not
again.

## Validation gates

Nothing is trusted because it is derived. Each rung passes four gates before it
is used, and the self-test (`tests/scale_selftest.cpp`, wired into CTest) runs
the first three on every build.

1. **Gradient gate.** `gradient()` vs central differences. Rung 2 currently
   agrees to 1.6×10⁻⁸ on a deliberately messy configuration.
2. **Equipartition gate.** With the substrate switched off, rung 2's twist and
   slip sectors are exactly harmonic in differences with unit Jacobian, so
   `⟨Δψ²⟩ = kT/k_twist` exactly — no small-fluctuation caveat. Measured: 0.03%
   and 0.35% off. Any clamp, cap or truncation anywhere in the propagator breaks
   this, which is what it is there to catch.
3. **Clock gate.** Free particle gives `MSD = 2Dt` (0.4% off) and acceptance
   exactly 1; enabling collective moves leaves the clock bit-identical.
4. **Kinetic gate — the one that licenses reading rates at all.** MALA's
   measured MFPT against the exact Smoluchowski answer, as a dt series:

   | dt | MFPT / exact |
   |---|---|
   | 0.008 | 1.152 ± 0.023 |
   | 0.004 | 1.110 ± 0.022 |
   | 0.002 | 1.070 ± 0.021 |
   | 0.001 | 1.016 ± 0.020 |

   Clean, monotone, one-signed O(dt) convergence. (The textbook parabolic
   Kramers formula is 10% off on the same profile — which is why `kramers.h`
   integrates the MFPT instead.) The error bars are printed because a dt series
   read without them is how the weighted-ensemble τ story went wrong.

5. **Overlap gate** (per rung, run once against real closures). Each rung must
   reproduce an observable of the rung below — both a static one (distribution)
   and a dynamic one (autocorrelation, MSD) — **inside a window both can
   reach**. No rung is trusted outside a validated overlap. This is the gate
   that detects a failed Markovian projection, and it is why the ladder is built
   bottom-up with each rung's reach overlapping the last.

## The ladder

| rung | object | coordinates | reach | status |
|---|---|---|---|---|
| 0 | all-atom segment pairs | Cartesian | ns | **done** — PMF campaign, 305 rows |
| 1 | tropocollagen molecule | 96 beads, twist | µs (ms with WE) | **done** — rigorous BD |
| 2 | microfibril (5-strand) | centerline, twist, strand slips | ms–s | **skeleton built** |
| 3 | fibril | rod centerline, twist, radius, plastic slip field | s–min | designed |
| 4 | fibril bundle / fascicle | fibril centerlines, crimp, matrix shear | min–**hours** | designed |

Rung 0 → 1 is already a working instance of this contract: measured segment-pair
forces, baseline-subtracted, kernel-regressed, blended into the tables the
engine reads. The ladder above is the same operation repeated.

### Rung 2 — the microfibril (`src/scale/microfibril.{h,cpp}`)

Coarse-graining stops at the microfibril, not the molecule, because it is the
smallest unit whose internal registry is already *resolved*: below it, D-stagger
is the thing being determined; above it, D-stagger is a property you carry. The
docking engine's mean cluster size already sits at ~5.3, so the object is one
the model produces rather than one we impose.

Coordinates, per node (one D-period of arc, 66.9 nm): centerline `r` (3), twist
`ψ` (1), and the axial slip of strands 1–4 against strand 0 (4) — **eight**.
Strand 0 deliberately has no slip coordinate: its axial position is already the
node's arc position, and giving it one would leave `F` flat along the sum of the
slips. A coordinate with no restoring force and no physical meaning is a gauge
freedom, and it silently poisons any mobility measurement made in it.

```
F = κ Σ (1 - cos θ)                        bend
  + ½ k_stretch Σ (|u| - D)²               stretch
  + ½ k_twist  Σ (Δψ - ψ₀D)²               twist, ψ₀ = the emergent supertwist
  + ½ k_j,m    Σ (Δδ)²                     shear lag — note the per-bond spring
  + Σ V(δ)                                 measured registry substrate, period D
```

Two things fell out of building it that were not in the plan, and both matter.

**This rung is a Frenkel–Kontorova chain.** A harmonic chain in a periodic
substrate is the canonical model of interfacial slip and dislocations. That
identification is not decoration — it dictates the move set. Slip cannot happen
as a step; it happens as a *kink* of finite core width, and the kink is the
object that moves. The barrier that sets creep is therefore the Peierls–Nabarro
barrier for kink glide, **not** the substrate amplitude. `slipProfile()` returns
the substrate, labelled explicitly as a strict upper bound, precisely so nobody
quotes it as the creep barrier.

**The shear-lag spring cannot be uniform, and finding that out was the useful
part.** With a uniform `EA/D` spring the cheapest possible full-D slip costs
~470 kT even spread over an optimal kink, and the kink channel accepted **zero**
of 20 000 proposals. That reads as "registry is frozen" while actually meaning
the model was wrong: a tropocollagen molecule is only ~4.46 D long, so every ~5
nodes the strand *ends*, and nothing covalent bridges the gap zone in an
uncrosslinked fibril. The coupling is per-bond:

- `k_gap = 0` — uncrosslinked. Slip localises at molecular termini and costs a
  few kT. **This is why uncrosslinked collagen creeps.**
- `k_gap > 0` — enzymatic crosslinks bridge the gap, load transfers
  elastically, slip is suppressed.

With that one change the kink channel accepts **35.8%** uncrosslinked and
**exactly 0%** crosslinked, over 20 000 attempts each. That contrast is the
rung's first prediction and it is asserted in the self-test, not merely printed.
It is also the mechanistic payoff for the crosslink work at rung 1: `k_gap` is
the elastic/viscoplastic switch, and it is a number rung 1 can measure.

This is the same failure shape as the v0.3.0 contact-integral aliasing bug — a
wrong discretisation of a physical mechanism presenting as an absence of the
mechanism. Worth watching for at every rung.

**Closure tasks** (nothing here is measured yet; the rung prints `EXPLORATORY`):

| id | parameter | measurement at rung 1 |
|---|---|---|
| L2-1 | κ_bend | constrained end-to-end sampling of a 5-mer → L_p |
| L2-2 | k_stretch | axial pull, through the entropic→enthalpic crossover |
| L2-3 | k_twist, ψ₀ | twist-restrained sampling; ψ₀ from the supertwist already observed emergently |
| L2-4 | k_shear, **k_gap** | differential axial restraint within a molecule, and across a telopeptide crosslink (`xlinkBuf`) |
| L2-5 | Peierls–Nabarro barrier | string/umbrella in rung 2's own coordinates, cross-checked against pulling one strand through its registry at rung 1 |
| L2-6 | non-integer gap period | 4.46 D rather than 5; the gap bond must migrate along the strand |
| L2-7 | D_trans, D_twist, D_slip | MSD of a free 5-mer, and of one strand's offset inside a held bundle |

`k_slip` depth and width are already `Measured` (PMF campaign), and their ratio
gives a substrate curvature of 2.67 kT/nm² against an independently measured
D-well stiffness of ~2 — an agreement, not a fit. That agreement is the only
reason the functional form is admissible, and if it ever degrades past ~2× the
well *shape* is wrong, not the depth.

### Rung 3 — the fibril

A viscoplastic elastic rod. Coordinates per node: centerline (3), twist (1),
radius/count (1), and a **plastic slip field** — the cross-section-averaged
accumulated slip. The constitutive content is the flow rule relating slip rate
to resolved shear stress, and it is *measured at rung 2* as the kink nucleation
and glide rate versus applied differential load. That is standard crystal
plasticity, with the flow rule measured instead of fitted, and it is where
collagen's stress relaxation and yield come from.

Growth (microfibril attachment, D-registry annealing across the cross-section)
enters as a Gillespie layer whose propensities come from rung 2 barriers via
`kramers.h`. This is the piece that structurally replaces the old KMC tab, and
the difference is that none of its rates are knobs.

### Rung 4 — the bundle

Fibrils plus interfibrillar matrix (proteoglycan, decorin, water). Coordinates:
fibril centerlines, crimp amplitude/wavelength, interfibrillar slip. The
relevant modes here are slow *by nature* — this is where minutes and hours stop
being a stretch and become the natural units.

## What is built

```
src/scale/level.h/.cpp    rung contract, gradient gate, provenance audit
src/scale/mala.h/.cpp     MALA + collective channel + clock accounting
src/scale/kramers.h/.cpp  exact overdamped MFPT, basin ratios, detailed balance
src/scale/rng.h           counter-based RNG, same family as the GPU kernels
src/scale/microfibril.*   rung 2
tests/scale_selftest.cpp  the gates; `ctest` target scale_selftest
```

No GL, no GLM, no dependencies — a rung is a free energy, a mobility and a
propagator, and none of those need a graphics context. The gates run in CI on a
machine with no GPU.

Two pieces of existing machinery carry straight up the ladder. **Weighted
ensemble** is propagator-agnostic: it needs only snapshot/restore, which
`SimState` already provides, so it applies at every rung — and the open ~2×
deficit will be far cheaper to root-cause at a coarse rung, where brute force is
affordable and the reference is not the bottleneck. **The environment controls**
(T, pH, ionic strength) enter every rung through the closures rather than
through duplicated parameterisations.

## Build order

1. **L2-7 mobilities and L2-1/2/3 elastic constants** — cheapest closures, and
   they unlock the overlap gate.
2. **Overlap gate 1↔2** in the 1–10 µs window (rung 1 reaches it with WE).
   Until this passes, rung 2 is a hypothesis.
3. **L2-4 k_gap, both with and without crosslinks** — the switch, and the first
   result worth publishing.
4. **L2-5 Peierls–Nabarro barrier** → first real creep rate.
5. Rung 3, and the Gillespie layer fed from rung 2 barriers.
6. Rung 4.

## Where hacks would get in, and the answer to each

| temptation | why it is tempting | the answer |
|---|---|---|
| fit `F` at a rung to reproduce an experiment | it works immediately | provenance marks it `Assumed`; the rung prints `EXPLORATORY` and can never be quoted as kinetics |
| count accepted collective moves per sweep and call it a rate | there is a number right there | the propagator adds no time on that channel, by construction |
| use the substrate barrier as the creep barrier | it is already computed | `slipProfile()` is documented and named as an upper bound |
| assume a coarse mobility from Stokes | it saves a measurement | admissible as `Literature`, but the overlap gate on MSD will catch it if it is wrong |
| skip the dt-convergence series | 95% acceptance "looks fine" | thermodynamics is exact at any dt; kinetics is not, and the gate measures the difference |
| widen a rung's reach past its validated overlap | that is the whole point of reach | no rung is trusted outside a validated overlap — extend the overlap first |
