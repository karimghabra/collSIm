# collagenSim

Real-time, GPU-resident simulation of **type I collagen fibril self-assembly**
with full-atom rendering of every tropocollagen molecule. C++20 / OpenGL 4.6
compute shaders; built for an RTX 5080 but any GL 4.6 GPU works.

No crosslinks — pure non-covalent self-assembly, driven by interaction profiles
computed from the real human COL1A1/COL1A2 sequences.

![close-up](assets/validation/close_0001.png)

## Install

Grab the latest installer or portable zip from the
[Releases page](https://github.com/karimghabra/collSIm/releases). Windows
x64, any OpenGL 4.6 GPU (built for an RTX 5080).

## Build & run

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\collagenSim.exe
```

Left-drag orbit, right/middle-drag pan, wheel zoom. Presets: **dilute**
(isotropic solution, purist slow nucleation), **quench** (dense aligned phase +
crowding agents — watch lateral condensation and D-registration), **dense**
(isotropic + crowding).

Useful CLI: `--preset 0|1|2 --nmol N --steps perFrame --colormode 0..4
--camdist nm --lod nm --hidden --frames N --every K --shot path/prefix`

## The model, honestly

**True all-atom MD of fibrillogenesis is impossible in real time** (assembly
takes minutes–hours; atomistic MD advances ~ns/day). This app separates
structure from dynamics:

1. **Structure (full-atom, data-derived).** `tools/build_tropocollagen.py`
   extracts the triple-helix screw operator from the 1.30 Å crystal structure
   of [(Pro-Pro-Gly)10]3 (PDB 1K6F) — reproducing its published 7/2 symmetry
   (measured −102.87°/residue ≡ +51.37°/triplet, rise 2.858 Å/residue) — and
   threads the real helical domains (auto-detected: α1 residues 179–1192,
   1014 aa; α2 analogous) through it. Side chains are placed from PDB Chemical
   Component Dictionary ideal coordinates with greedy χ1 clash relief;
   Y-position prolines become 4-hydroxyproline. Result: 19,460 heavy atoms per
   molecule, L = 290.4 nm, validated for backbone continuity (C–N =
   1.334 ± 0.017 Å across all junctions) and Gly-in-core packing.

2. **Interaction physics (sequence-derived).** Following Hulmes & Miller
   (1973), per-residue charge and centered Kyte–Doolittle hydrophobicity are
   projected onto the molecular axis and cross-correlated as a function of
   axial stagger between two molecules (`tools/interaction_profile.py`).
   The three deepest minima land at **1.00 D, 1.99 D, 3.00 D** with
   D = 234 residues = 66.9 nm — quarter-stagger emerges from the sequence, it
   is not scripted. Parallel and antiparallel channels are stored as 1D
   textures the GPU samples during force evaluation.

3. **Dynamics (coarse-grained, rigorous statistical mechanics).** Each
   molecule is a 96-bead semiflexible filament (discrete worm-like chain,
   default Lp = 60 nm) evolved by overdamped Langevin (Euler–Maruyama, PCG4D
   Gaussian noise, XPBD bond constraints). Inter-molecular forces act between
   closest points of segment pairs found via a GPU cell grid: soft-core
   excluded volume (~19 kT crossing barrier — filaments cannot pass through
   each other; a hard-overlap counter in the UI verifies 0), a contact
   envelope modulated by the sequence registry energy U(Δ) (radial +
   axial-sliding force from the texture gradient), optional isotropic
   depletion attraction ("crowding agents", cf. PEG in in-vitro
   fibrillogenesis), and soft box walls.

4. **Rendering.** Every near-LOD molecule draws all 19,460 heavy atoms as
   instanced sphere impostors: the straight atomic template (contour s, local
   cross-section x,y) is mapped in the vertex shader onto the deformed
   filament through interpolated bead positions and parallel-transported
   frames. Distant molecules render as capsule impostors. Color modes:
   element (CPK), residue class, chain (α1/α1/α2), molecule, and **D-phase**
   (hue = position mod D — stagger registration shows up as aligned banding).

### Knobs that trade realism for watchability (all exposed in the UI)

- **Kinetic speed-up ×N** divides the drag γ. In overdamped dynamics this is
  an exact relabeling of time (trajectories are identical); displayed sim
  time already accounts for it.
- **Depletion attraction** accelerates encounter kinetics the same way
  crowding agents do experimentally.
- Excluded volume is softened (19 kT barrier instead of hard-core) to allow
  larger timesteps; barrier ≫ kT so topology is preserved.
- Telopeptides are excluded (helical domain only), hydration/counterion
  structure is implicit in the two ε weights, and the molecule has no twist
  degree of freedom (registry is azimuth-independent in this model).

## Performance & stability guards

Rendering is capped (default 30 fps, adjustable) and decoupled from the
physics: an auto-pacer adapts BD steps-per-frame to the wall-clock frame
budget, and the GPU queue is fenced every frame so compute can never pile up
behind the compositor (the failure mode that can freeze a desktop / trip the
Windows TDR watchdog). Displayed positions pass through an adjustable
exponential smoothing buffer (render-only; the physics state is untouched).
LOD selection reads back only per-molecule centers (4 KB) computed on-GPU.

## Periodic boundaries

`--pbc` / UI checkbox: minimum-image forces applied to whole segments
(positions stay unwrapped so bonded terms never cross a seam), wrapped
neighbor grid, and whole-molecule recentering applied to physics and display
buffers together. With the fibril scenario the box length snaps to a whole
number of 5D lattice periods, producing an **effectively infinite fibril** —
the clean setup for studying emergent fibril twist and axial-compression
buckling without end effects.

## Environment controls (real-time, experiment-mapped)

- **Temperature (0–60 °C).** Scales thermal noise (kT) and the hydrophobic
  channel strength (entropic effect: ~+1.3%/K), reproducing cold
  depolymerization / warm assembly.
- **pH (2.5–11).** Per-residue protonation via Henderson–Hasselbalch (pKa:
  Asp 3.65, Glu 4.25, His 6.0, Lys 10.53, Arg 12.48) over the actual counts
  (94 D / 143 E / 18 H / 108 K / 163 R per trimer). The electrostatic registry
  tables are bilinear in charges, so they are recombined on the GPU from 15
  precomputed type-pair basis tables in milliseconds when the slider moves —
  including the net-charge (DC) term. Verified: pairs bind and register at
  pH 7.4; at pH 3.5 (net ≈ +81 e) they repel and disperse — the classic
  acid-stock → neutral-buffer fibrillogenesis trigger.
- **Concentration.** Set in mg/mL (shown as % w/v), converted through the
  computed trimer mass of 284.7 kDa and the box volume, matching turbidity
  assay conventions (defaults land near 0.1 % w/v).
- **Preassembled fibril scenario.** Hodge–Petruska quasi-hexagonal packing
  (5D file period, 0.54 D gaps, neighbor files staggered by (i+2j mod 5)·D),
  radius and length (up to several µm / 4000 molecules) adjustable. Use it to
  study fibril twist with the azimuthal registry, or reduce the box-z wall to
  compress axially and explore crimp-like buckling. Note: true tendon crimp
  is a µm-to-100 µm fascicle-scale phenomenon involving inter-fibrillar
  matrix; what this reaches is fibril-level twist/buckling mechanics.

## Measured interactions (v0.2.0)

The registry tables now carry a third channel: **Delta-learned corrections
from all-atom implicit-solvent MD** (`tools/pmf_harness.py` campaign, 305
baseline-corrected segment-pair mean-force measurements; merged by
`tools/pmf_merge.py` with bootstrap shrinkage and a conservative +/-6 kT/seg
cap). Measured well corrections at the three D-stagger families (-3.7, -6.0,
-6.0 kT per engine segment at 67.6/132.3/203.4 nm), radial envelope refit
from the gap scan (attR0 1.65 nm), and — per the tilt-series verdict — the
cos^2 alignment factor demoted to polarity gating (the stagger ramp of each
contact carries the physical angular attenuation). At the time this was
validated on the KMC and docking-MC engines (D-banding order 0.02 -> 0.99 and
0.69 -> 0.93 respectively), which retired the idealized funnel; both engines
were removed in v0.4.0, but the tables they validated are what both BD engines
now use. The `measured
correction` slider scales the channel; raw fits and uncertainties are in
`pmf_data/merge_report.txt`.

## Registry models (and what they taught us)

The engines use one registry model. Two coarser ancestors (a nonspecific
contact term, and a 1D funnel resolving stagger only) were removed in v0.4.0:
they were strictly less informed than the measured 2D tables and existed only
as history.

- **2D azimuthal (sequence-raw, default).** U(stagger, facingA, facingB)
  computed from every residue's true (axial, azimuthal) surface position,
  including real telopeptide sequences at the helix ends; each molecule
  carries a twist degree of freedom with rotational Langevin dynamics and
  torque from the azimuthal gradient of the table. No sharpening, no imposed
  wells. Result: robust bundling with **weak D-selection (~12% of aligned
  contacts within 3 nm of a D-multiple vs ~8% random)**. Honest finding: with
  telopeptides included, the deepest best-over-angles minimum sits at 1.01 D,
  but competing minima are within a few percent — mean-field surface
  correlations under-determine registration.
- **1D funnel (idealized).** Gaussian wells only at the D-multiples with
  sequence-derived positions (67.5 / 134.4 / 201.0 / 267.5 nm) and relative
  depths (−1.18 / −0.61 / −0.38 / −0.25), plus the zero-stagger charge
  repulsion. An explicit idealization standing in for dock-and-lock
  telopeptide specificity. Result: **strong D-selection (~40%)** and visually
  coherent D-banding in the D-phase color mode.
- **Off.** Nonspecific attraction only — control case; amorphous bundling.

This bracket mirrors the biology: pepsin-treated collagen (telopeptides
removed) forms poorly banded fibrils, and our continuum treatment of flexible
telopeptides (azimuth-averaged) dilutes exactly the docking specificity that
registers real fibrils. The clean next step for full rigor is explicit
telopeptide docking sites with stochastic bond kinetics (a discrete-site
KMC/BD hybrid) — architecture-compatible with the current force pass.

Also learned the hard way (all fixed, verified by GPU counters and pair unit
tests): pair-additive attraction needs per-segment saturation or dense phases
self-crush; timestep displacement must be clamped below the excluded-volume
shell or filaments tunnel; and a segment cell grid must use cell ≥ segment
length + cutoff or contacts silently vanish.

## Reaching rare events without giving up real kinetics (WE, `--advanced`)

The honest limit of everything below is that BD has the only trustworthy clock
and reaches microseconds, while fibrillation takes 30–120 minutes. The usual
temptations — bigger MC steps, force-biased proposals, softer potentials — all
buy reach by damaging the dynamics, and a gradient-following proposal cannot
cross a barrier to a *different* registry well anyway.

The **WE tab** (hidden unless the app is started with `--advanced`) takes the
other route. Weighted ensemble runs many walkers of
plain, unadjusted Brownian dynamics — the propagator is not modified in any
way — and redistributes only *computational effort*: walkers that climb a
progress coordinate are split, ones that fall back are merged, and each carries
a weight that keeps the ensemble statistically exact. It is variance reduction,
not acceleration, so the rates that come out are unbiased estimates of the true
BD rates. That is the entire reason to prefer it over a faster sampler.

- **Progress coordinate**: size of the largest connected cluster, *plus* how
  near the closest outside molecule has come to it. The continuous part matters
  — with integer cluster size alone every walker sits in the same bin, nothing
  distinguishes one about to gain a partner from one in empty solvent, and the
  estimator saturates at one event per τ.
- **Rate**: walkers reaching the target cluster have their weight harvested as
  flux and are recycled to a fresh dispersed state, so the steady-state flux is
  the nucleation rate directly (Hill relation). The first iterations are
  discarded as burn-in — flux before weight has spread up the coordinate is
  meaningless.
- **Validation**: `--wetest` measures the same rate twice on the same system,
  once by WE and once by brute-force BD, and prints the ratio. A rare-event
  estimator that has not been checked against brute force in a regime where
  brute force still works is not evidence of anything.

Both engines must be equalized carefully to compare: `restart()` applies a
1500-step soft-start ramp and `restore()` clears it, so a reference run left
mid-ramp is handicapped precisely where the events occur. Both now complete the
ramp before the clock starts.

The reactant basin also has to be *clean*: at 1 mg/mL roughly one equilibrated
start in eight already contains a trimer, and a single pre-nucleated state in
the recycling pool would manufacture flux out of nothing by re-crossing the
target the instant any walker landed on it. Both the pool and the brute-force
reference now reject such starts and say how many they threw away.

### Validation status: WE reads ~2× low, cause still unknown

Measured on 6 molecules at 0.49 mg/mL, target trimer, against a brute-force BD
mean first passage of ~8200 ns (genuinely rare — well over a hundred τ, so not
a resolution-floor artifact):

| τ | WE iterations | crossings | WE / brute force |
|---|---|---|---|
| 60 ns | 320 | 15 | 0.49× |
| 30 ns | 500 | 14 | 1.27× |
| 30 ns | 900 | 26 | 0.48× |

τ was the obvious suspect — flux is only harvested at iteration boundaries, so
a walker that crosses and falls back inside one τ is never counted — and the
middle row looked like confirmation. **The high-statistics row refutes it.**
Halving τ does not move the answer: WE sits at ~0.48× at both τ values once
enough crossings accumulate. The middle row was a fluctuation, not a trend.

That fluctuation is itself the second finding. Two runs at identical τ with the
same recycling pool differ by 2.6×, far outside the ±27% that counting
crossings and taking 1/√N suggests. **Poisson-on-crossing-counts is the wrong
error bar for WE**: crossings carry unequal weights, and a single heavy walker
dominates the flux sum, so the true variance is much larger than the count
implies. The error bars `--wetest` currently prints are optimistic and need
replacing with block averaging over iterations.

Remaining suspects for the ~2× deficit, in order: the recycling pool is 6 fixed
states rather than a sampled basin distribution (and the pre-nucleation filter
removes precisely the fastest-nucleating starts, which biases the reactant
ensemble slow); and merge lossiness at the low walker counts these runs settle
into (12–20 walkers across 8 bins).

Brute force reproduces itself across independent batches (1.23e-4, 1.20e-4,
1.21e-4 /ns), so the reference is solid and the discrepancy is on the WE side.
Until this is resolved, WE rates are a lower bound and are **not** wired into
anything downstream. Worth re-running against the rigorous engine, whose
equilibrium is exact -- the brute-force reference was itself measured on the
fast engine.

Remaining limitations: recycling draws from a small fixed pool of dispersed
starts rather than a sampled basin distribution, so the reactant state is not
fully decorrelated; and τ still sets a hard resolution floor (events faster
than one per τ are reported as 1/τ, and `--wetest` warns when the measured MFPT
falls within 5τ). Both push the estimate the same way — toward under-counting
flux — so with τ chosen too large a WE rate here is a lower bound, not a
two-sided estimate.

## Three engines, and what each one is for

All three tabs drive the same `Sim` on the same molecules, sharing the measured
interaction tables, the cell grid, the neighbour lists and the renderer. The
tab selects the *propagator*, so you can switch mid-run and watch the same
configuration evolve under any of them.

| Tab | Propagator | Samples `exp(-U/kT)`? | dt |
|---|---|---|---|
| **BD - fast** | Euler-Maruyama + XPBD projection + clamps | **no potential exists** | 0.3 ns effective |
| **BD - rigorous** | explicit `U`, harmonic bonds, no clamps, Metropolis-adjusted | **yes, exactly** | 5e-4 ns |
| **BD - constrained** | same force field, bonds rigid via SHAKE | not yet (no Fixman term) | 0.02 ns |

The fast engine is ~50x cheaper per simulated nanosecond than the rigorous one
and is the only one that reaches interesting assembly. The rigorous engine is
the reference that says what that costs. The constrained engine removes the
stiffest mode from the spectrum so dt can rise 40x, and is the more faithful
limit -- collagen's real axial stiffness is `EA/L ~ 675 kT/nm^2` (E ~ 5 GPa,
A = pi(0.75 nm)^2), i.e. 1.3% rms strain, *stiffer* than any spring worth
integrating. Softening bonds to buy dt is therefore not a legitimate move; the
rigid limit is.

### Why the fast engine has no potential energy function

Not "an approximate one" -- none. Four separate things make its force field
non-conservative, and each was added for stability rather than for physics:

| Element | Why it breaks |
|---|---|
| XPBD bond projection | a projection, not a force; a projected Euler-Maruyama step has no invariant Gibbs measure |
| `attMag` saturation | scales pair forces by a factor computed from the **previous** step, so the drift is not even a function of the current state |
| `fMax` force clamp | truncates the pair force |
| `dxMax` displacement clamp | clips the proposal; the noise is no longer Gaussian |

The displacement clamp is not a rare-event guard. At the shipped defaults the
per-axis thermal noise is 0.293 nm, so |dx| ~ sigma*chi_3 exceeds
`dxMax` = 0.35 with probability ~0.70 from thermal motion alone, before any
force contributes. Measured **74.2%** of steps (40 windows, bonded forces
only). For most of this project's life the clamp, not the force field, was
deciding where the beads went.

Measuring this correctly needs care: the force pass zeroes the stats buffer
when `totalSteps % 16 == 0`, so sampling off that boundary divides by 16 steps
while accumulating fewer -- and stepping 16 preserves the phase, so averaging
repeats the same wrong window rather than washing it out. `--equipart` aligns
to the boundary first.

### The rigorous engine: U written out

```
U = sum_bonds  1/2 kBond (|x_{i+1} - x_i| - a)^2      kBond from p.bondStrain
  + sum_angles kBend (1 - cos theta)                   kBend = Lp kT / a
  + sum_pairs  1/2 kRep (rRep - d)^2                   d = closest approach
  + sum_pairs  [env(d) - env(cutoff)] G(D, phiA, phiB)
  + sum_pairs  -epsDep [exp(-md^2/depR^2) - ...]
  + sum_beads  1/2 kWall max(|x_i| - L_i, 0)^2
  + sum_xlinks 1/2 kX (l - rest)^2
```

Three choices worth stating. The registry stagger and facing angles are
evaluated at **segment midpoints**, not at the closest-approach point: a
minimiser would make `U` depend on an inner optimisation whose gradient needs
`du/dx`, and that is also where the closest-point solve's clamping puts kinks.
Pair energies are **shifted to zero at the cutoff** -- free in force, but
without it a pair entering the neighbour list steps `U` by ~0.01 kT, invisible
in a trajectory and fatal to a Metropolis ratio. The energy reduction runs in
**double**: `U` reaches ~1e5 kT while the Metropolis test needs `dU` to
~0.01 kT, and differencing two float32 totals of that size leaves nothing.

`--gradtest` finite-differences `U` against the drift, one bead at a time
(a direction spread over all 3N coordinates displaces each bead by ~2 ulp of a
float32 coordinate and measures rounding instead of physics):

| model | median error | vs mean force |
|---|---|---|
| bonded + steric only | 1.5-4e-3 kT/nm | 0.01-0.02% (noise floor) |
| full, with registry | 1.9-3.6e-2 kT/nm | 0.1-0.2% |

The bonded gradients are exact. The registry term leaves a systematic ~0.2%,
which is the deliberately omitted `d(facing angle)/dx` and `d(gate)/dx`.
Metropolis-Hastings needs an exact `U` but tolerates an approximate drift, so
this costs acceptance rate, not correctness.

### MALA, and what acceptance actually means

Positions and twist are proposed jointly and accepted together; the whole test
runs on the GPU because a readback per step would stall the pipeline. Both
endpoints use the same neighbour list, or their energy difference is
meaningless.

Metropolis makes the sampled distribution exactly `exp(-U/kT)` at **any** dt.
So acceptance is not an accuracy readout -- it is a **kinetics** readout. A
rejection freezes the system for a step, which Brownian motion never does, so
the trajectory is only interpretable as dynamics while acceptance is near 1.

`--dtscan` at 12 molecules (1140 bonds), against the Gaussian-chain prediction
`erfc(sqrt(10 nBonds) a^1.5 / 2 sqrt 2)` with `a = kBond dt / gamma`:

| dtRig (ns) | dt/tau | accept | predicted |
|---|---|---|---|
| 5e-4 | 0.019 | 0.761 | 0.888 |
| 1e-3 | 0.038 | 0.616 | 0.690 |
| 2e-3 | 0.076 | 0.284 | 0.259 |
| 5e-3 | 0.191 | 0.000 | 0.000 |

Measurement tracks the closed form, so the sampler is sound. Note the
prediction must use the **chain** mode spectrum: a bead-spring chain has
stiffnesses `2k(1 - cos q)` spanning 0 to `4k`, and `<(2-2cos q)^3> = 20`.
Using the bare spring constant understates the variance twentyfold -- an error
that initially looked like a bug in the sampler and was not.

Acceptance falls as `N^(-1/3)`, so reaching 0.99 -- where the trajectory is
still dynamics -- needs dt ~1e-4 ns at 12 molecules and ~3e-5 ns at 400. That
is the honest cost, and it is why constrained bonds (which remove the stiffest
mode outright, worth ~15x) is the next thing worth building.

### The constrained engine: rigid bonds (`BD - constrained`)

The bond is the fastest mode by a wide margin, and it is the only reason dt has
to be so small:

| mode | k (kT/nm^2) | tau = gamma/k (ns) |
|---|---|---|
| **harmonic bond** | 268 | **0.026** |
| attraction well curvature | ~18 | 0.39 |
| soft core `kRep` | 15 | 0.47 |
| bending, transverse (`kBend/a^2`) | 2.1 | 3.3 |

Removing it raises the ceiling to the next mode at ~0.39 ns. This tab does that
with red-black SHAKE: `xpbd.comp` already solves one constraint exactly and
skips alternate bonds so no two touching constraints run together, which makes
iterating it Gauss-Seidel SHAKE. The difference from tab 1 is the sweep count.
**Tab 1 runs 4 sweeps and does not converge** -- its measured 0.052 nm bond
spread is residual projection error, not physics, which is why its `sd(l)` sits
14% below the analytic value while the rigorous engine matches it to 0.4%.

**Two pieces are missing, and both fail silently.** Constraining changes the
equilibrium measure: recovering the stiff-spring statistics needs the Fixman
potential `U_F = (kT/2) ln det(J J^T)`, where for a bead chain `J J^T` is
tridiagonal with 2 on the diagonal and `-cos theta_k` off it -- angle-dependent,
so not a constant that can be dropped. And Metropolis adjustment on a manifold
needs a tangent-space proposal plus a reverse-projection check
(Zappa/Holmes-Cerfon/Goodman 2018; Lelievre/Rousset/Stoltz 2019); omitting the
check breaks detailed balance.

Neither shows up as an instability or a visibly wrong trajectory. They show up
as slightly wrong angle statistics -- which is exactly what `--equipart` below
measures, and why this tab shipped unadjusted rather than unmeasured.

### Does either engine sample what it claims? (`--equipart`)

With pair interactions off, `U` is bonds + bending only. In bond-vector
coordinates the Jacobian is `prod l_i^2`, both terms separate, and each has a
closed-form equilibrium:

```
p(l)       ~ l^2 exp(-k (l-a)^2 / 2kT)   =>  <l> = a + 2 sigma^2 / a
p(cos t)   ~ exp(-kappa (1 - cos t))     =>  <cos t> = coth kappa - 1/kappa
```

The second is the Langevin function and depends on nothing but
`kappa = kBend/kT`. Both engines have the same `kBend`, so both must reproduce
it. 4 molecules, 2500 ns each, all quarters flat:

| engine | sd(l) | `<cos t>` | Lp (nm) | accept |
|---|---|---|---|---|
| **exact** | **0.06113** | **0.94906** | **58.5** | — |
| BD - fast | 0.05227 | 0.96147 | **77.8** | — |
| BD - rigorous, unadjusted | 0.06138 | 0.95157 | 61.6 | — |
| BD - rigorous + MALA | 0.06136 | 0.95194 | 62.1 | 0.744 |
| BD - constrained (SHAKE) | 0.00160 | 0.95180 | 61.9 | — |

**The fast engine is 33% too stiff.** Its molecules behave as though the
persistence length were 78 nm rather than the 60 nm configured, and its bond
spread is 14% low -- and that 0.052 nm spread is not a soft bond, it is
unconverged SHAKE, since tab 1 runs only 4 sweeps. Any fast-engine result that
depends on chain flexibility -- how readily a molecule conforms to a neighbour,
how much entropy it surrenders on binding -- carries that stiffening.

The rigorous and constrained engines land within 0.3% of the analytic
`<cos t>` (6% in Lp, which amplifies the error because `Lp = -a/ln<cos t>` and
the log is near zero). Their quarters are still drifting down -- MALA runs
0.95325 -> 0.95078 -- so the residual is mostly incomplete sampling rather than
bias.

**The constrained engine agrees with the flexible ones to within 0.0002 in
`<cos t>`**, which is below the scatter between the two flexible runs. So the
Fixman correction, at this bending stiffness, is smaller than this measurement
can resolve: rigid bonds reproduce the stiff-spring statistics here. That is
the empirical answer to whether the missing `U_F` term matters, and it is why
this tab is usable in practice despite being formally incomplete. It is a
statement about *this* `kappa`, not a general one -- a floppier chain would
need the term.

MALA and the unadjusted propagator agree closely here, which is the expected
result at dt = 1e-3: the discretisation bias is genuinely small at that step.
The point of the Metropolis correction is not that it beats the unadjusted run
at a good dt -- it is that it *guarantees* the agreement instead of requiring
you to assume it, and reports acceptance when the assumption fails.

**Two traps this test set, both worth knowing.** Molecules are laid down as
straight rods, and bending relaxes slowly enough that a 400 ns run reports an
equilibration artefact rather than a property of the engine; every run here is
pre-relaxed from a shared state and prints `<cos t>` per quarter so drift is
visible. And **nominal simulated time is not a valid budget for MALA**: a
rejected step advances the clock while the system stays put, so a
low-acceptance run can report thousands of nanoseconds having explored almost
nothing. Read the acceptance column before reading the row.

## Data provenance

- PDB **1K6F** (Berisio et al., Protein Sci. 2002) — triple-helix geometry
- UniProt **P02452 / P08123** — human COL1A1 / COL1A2 sequences
- PDB Chemical Component Dictionary — ideal residue coordinates (incl. HYP)
- Hulmes & Miller 1973; Orgel et al. 2006 — D-stagger interaction logic

## Layout

```
tools/     Python pipeline (structure + interaction profiles)
assets/    atoms.bin, profiles2d*.bin, correction2d.bin, tropocollagen.pdb
src/       C++ app (sim orchestration, renderer, UI)
shaders/   GLSL compute + render kernels
data/      downloaded source data (1K6F, FASTA, CCD)
```
