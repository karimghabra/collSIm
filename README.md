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
contact carries the physical angular attenuation). Result: KMC D-banding
order from sequence-raw tables rose 0.02 -> 0.99 and docking-MC 0.69 -> 0.93,
retiring the idealized funnel as the best-performing model. The `measured
correction` slider scales the channel; raw fits and uncertainties are in
`pmf_data/merge_report.txt`.

## Registry models (and what they taught us)

The app ships three switchable registry models (UI: physics → registry model):

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

## Simulation methods FAQ

**Why Brownian dynamics instead of kinetic Monte Carlo?** KMC needs a
discrete event catalog with computable rates; continuum diffusion + bending +
sliding of semiflexible filaments is exactly what you cannot discretize
without a lattice (which would destroy semiflexibility and smooth motion).
BD gives the same Boltzmann statistics with physically meaningful kinetics
and is embarrassingly GPU-parallel. A KMC-flavored layer (explicit bond
on/off events with Bell rates) can be added on top later without changing
the architecture.

## Data provenance

- PDB **1K6F** (Berisio et al., Protein Sci. 2002) — triple-helix geometry
- UniProt **P02452 / P08123** — human COL1A1 / COL1A2 sequences
- PDB Chemical Component Dictionary — ideal residue coordinates (incl. HYP)
- Hulmes & Miller 1973; Orgel et al. 2006 — D-stagger interaction logic

## Layout

```
tools/     Python pipeline (structure + interaction profiles)
assets/    atoms.bin, profiles.bin, tropocollagen.pdb, validation plots
src/       C++ app (sim orchestration, renderer, UI)
shaders/   GLSL compute + render kernels
data/      downloaded source data (1K6F, FASTA, CCD)
```
