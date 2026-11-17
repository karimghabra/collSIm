# Explicit-solvent free energies of collagen segment pairs (`aamd/`)

All-atom molecular dynamics in explicit water, used to measure what the coarse
models assume: how strongly two triple-helical segments attract at a
favourable and an unfavourable axial stagger, whether lysine succinylation at
pH 4.2 changes that, and what Ca2+ and sulfate do between them. Every number
here is set against the prediction of `modcoll` (the site model of
`docs/modified_collagen_sites.md` and the ion model of
`docs/modified_collagen_ions.md`) at the same geometry, with the same charges.

<!-- RESULTS -->

## What was simulated

### Segments and pairs

A segment is 30 consecutive helix residues of each of the three chains (ten
Gly-X-Y triplets, 8.6 nm), cut from the full-atom model that
`tools/build_tropocollagen.py` threads through the 1K6F screw operator (human
COL1A1 x2 + COL1A2, Hyp at Y positions). Each chain is capped with ACE and NME
built from the neighbouring residues' atoms, so no artificial termini carry
charge.

Segment A is helix residues 624-653. Its partner at a stagger of Delta
residues is residues 624 - Delta ... 653 - Delta, placed alongside so that the
residues facing each other are those of two whole molecules at stagger Delta:

| pair | partner residues | stagger |
|---|---|---|
| `1D` | 390-419 | 234 residues = 1 D = 66.9 nm, the native-favoured register |
| `05D` | 507-536 | 117 residues = 0.5 D, a disfavoured register |

The window was chosen with the coarse model before any simulation
(`aamd/select_segments.py`): over windows starting every six residues, the one
with the largest predicted contrast between 1 D and the off-D staggers 0.25,
0.5 and 0.75 D for native pairs at d = 1.65 nm. The coarse model gives a
30-residue pair at 1 D -2.6 kT (of which -2.0 kT hydrophobic) and at 0.5 D
-0.1 kT, so the selection is a pair where the model makes a definite,
testable prediction. The D-register of whole molecules is a sum over many such
pairs; one pair is a sample of it, not a substitute.

Chemistry and protonation (discrete states, chosen per site type so that the
number of charged groups is the Henderson-Hasselbalch expectation with the
recommended pKa set of `modcoll.electrostatics`, rounded; which residues of a
type carry the charge is a seeded draw, `aamd.coarse.discrete_charges`):

| chemistry | pH | Asp (3.9) | Glu (4.3) | succinyl (4.5) | Lys | His (6.5) | Arg |
|---|---|---|---|---|---|---|---|
| native | 7.4 | all charged | all charged | - | all charged | neutral (HIE) | charged |
| succinylated (every Lys) | 4.2 | 67 % charged | 44 % charged | 33 % charged | none left | charged | charged |

The same discrete pattern is given to the coarse model when it predicts the
same system, so the comparison is charge for charge.

### Force field and parameters

| component | choice | source |
|---|---|---|
| protein | AMBER ff14SB | Maier et al., J. Chem. Theory Comput. 11 (2015) 3696 |
| 4-hydroxyproline | HYP of ff14SB's amino12.lib | shipped with AmberTools 26 |
| acetyl-lysine | ALY of AmberTools' ff14SB_modAA (mod_amino.lib, frcmod.ff14SBmodAA) | Belfon & Simmerling, ff14SB-compatible non-canonical residues |
| succinyl-lysine (SLL, carboxylate; SLH, acid) | built here: ff14SB types, RESP charges | `aamd/params/`, below |
| water | TIP3P | Jorgensen et al. 1983, the ff14SB water |
| Na+, Cl- | Joung-Cheatham (TIP3P) | J. Phys. Chem. B 112 (2008) 9020 |
| Mg2+, Ca2+ | Li-Merz 12-6-4 (TIP3P), with the ion-carboxylate r^-4 term fitted to acetate binding | Li & Merz, J. Chem. Theory Comput. 10 (2014) 289; Jafari et al., J. Phys. Chem. B 128 (2024) 684 |
| sulfate | GAFF2 types, RESP charges | `aamd/params/`, below |

**Why these ions.** The ion report this work builds on (and the coarse ion
model's calibration) rests on one fact: fixed-charge divalent ions over-bind
carboxylate unless they are calibrated against solution data. Jafari et al.
(2024) did exactly that for ff14SB + TIP3P: the 12-6-4 model's C4 term between
the ion and the carboxylate oxygens was refitted, through the oxygen's
polarizability, to the experimental free energy of Mg2+ and Ca2+ binding to
acetate. With TIP3P their model gives -1.25 +- 0.07 (Mg2+) and -1.19 +- 0.16
kcal/mol (Ca2+) against the measured -1.33 and -1.27 (the plain 12-6-4 set
gives -3.91 for Mg2+, the 12-6 set -8.74 and -3.48). Those measured values are
the same association constants (log K 1.26 and 1.18) the coarse ion model is
fitted to, so the two models start from the same experimental anchor. Their
model keeps Mg2+ monodentate (99.95 %) and makes Ca2+ about half bidentate
(52 %), against 22-32 % bidentate Ca-carboxylate contacts in protein
structures.

The r^-4 term is added in OpenMM (`aamd.md.add_c4_term`) per atom pair:
C4(M, j) = C4_M(water) alpha_j / alpha_OW, with alpha_j from AmberTools'
`lj_1264_pol.dat` and C4_M(water) from parmed's TIP3P table (Mg 132.9, Ca 87.3
kcal/mol/A^4). The carboxylate oxygen (type O2) gets Jafari's polarizabilities,
0.125 A^3 with Mg2+ and 0.600 A^3 with Ca2+, which give C4 = 11.50 and 36.27
kcal/mol/A^4. A per-atom term is needed because in ff14SB the carbonyl O and
the carboxylate O2 share one Lennard-Jones type, so parmed's per-type table
(`add12_6_4`) cannot give the carboxylate alone the fitted value.
`tests/aamd/test_md.py` checks that every coefficient equals what parmed writes
once O2 has its own type, and that the OpenMM energy is the direct sum.

**Succinyl-lysine.** Atom types and bonded terms come from residues ff14SB
already has, so no bonded parameter is new: the lysine CB..CE and the
N-epsilon amide take the types of the AmberTools ALY (8C, N, H, C, O), the
CH2-CH2-COO(-) tail the ff94 glutamate types (CT, HC, C, O2) and its acid form
those of GLH (C, O, OH, HO). Charges: two-stage RESP (Bayly et al. 1993) on
ACE-X-NME in two backbone conformations (alpha -60/-45, beta -120/130), side
chain extended, HF/6-31G* electrostatic potential (psi4 1.11, density-fitted)
on Merz-Kollman shells, cap atoms fixed at their ff14SB charges and the
backbone N, H, C, O at the neutral-backbone values ALY also uses. The
geometries are ff14SB energy minima with those dihedrals restrained, not
HF/6-31G* optima as in the original AMBER procedure: on this machine a single
HF optimisation step took about ten minutes. The whole procedure was checked
by refitting the AmberTools ALY with it (results below).

**Sulfate.** Tetrahedron optimised at HF/6-31G*, RESP with the four oxygens
equivalent, GAFF2 Lennard-Jones and bonded terms (parmchk2). In these systems
sulfate meets only Na+ and the protein (never Mg2+ or Ca2+).

<!-- PARAMS -->

Versions: OpenMM 8.6.1 (CUDA, mixed precision), AmberTools 26.0, parmed
4.3.1, psi4 1.11, pymbar 4.2.0, mdtraj 1.11.1, numpy 2.4.6 (all conda-forge;
exact builds with checksums in `aamd/environment.lock.txt`).

### Systems

Each pair sits in a rectangular box of TIP3P water about 8.0 x 5.1 x 12.6 nm
(about 40,700 atoms): 1.25 nm of water beyond the solute in x and y at the
largest separation, 1.2 nm beyond the ends in z. NaCl 150 mM (counted per
litre of water) plus neutralising counterions in every box; the divalent
salts on top of that at 50 mM (CaCl2, MgCl2, Na2SO4: 12 ions in these boxes).

| system | pair | chemistry | pH | divalent | run |
|---|---|---|---|---|---|
| native-1D | 1D | native | 7.4 | - | PMF |
| native-05D | 05D | native | 7.4 | - | PMF |
| succ-1D | 1D | succinylated | 4.2 | - | PMF |
| succ-05D | 05D | succinylated | 4.2 | - | PMF |
| succ-1D-Ca | 1D | succinylated | 4.2 | 50 mM CaCl2 | PMF |
| succ-1D-SO4 | 1D | succinylated | 4.2 | 50 mM Na2SO4 | PMF |
| native-1D-Ca | 1D | native | 7.4 | 50 mM CaCl2 | 20 ns at contact |
| native-1D-Mg | 1D | native | 7.4 | 50 mM MgCl2 | 20 ns at contact |
| succ-1D-Mg | 1D | succinylated | 4.2 | 50 mM MgCl2 | 20 ns at contact |

### Restraints

The coordinate is the inter-axial separation d = x_B - x_A of the CA
centroids of the two segments. Held by harmonic terms (`aamd.md`):

- y_B - y_A = 0 and z_B - z_A = the built stagger offset (5000 kJ/mol/nm^2):
  the pair stays side by side along x at its stagger, so d is a
  one-dimensional coordinate with no rotational Jacobian, the quantity the
  coarse model's pair energy is a function of;
- each segment's axis parallel to z (the xy offset of its two end-triplet
  centroids, 1000 kJ/mol/nm^2);
- the helix ends intact: an elastic network of distance restraints
  (2000 kJ/mol/nm^2) between backbone N, CA, C atoms of the two terminal
  triplets at each end, pairs closer than 0.8 nm in the built helix. (An RMSD
  collective variable does the same job; it slowed the runs fourfold.)

Nothing restrains the rotation of either segment about its own axis, and
nothing restrains the middle six triplets.

### Sampling

Umbrella windows at d = 1.4, 1.5, ..., 2.8 nm (15 windows), k = 1000
kJ/mol/nm^2 (thermal width 0.05 nm), started from one pull (0.5 ns at
3.0 nm, then 2 ns from 3.0 to 1.15 nm). In the native 1 D pull the separation
stopped following the restraint at about 1.45 nm (the steric contact), so no
window was placed below 1.4 nm. Per window 0.5 ns equilibration, discarded,
and 4 ns production. NPT (Monte Carlo barostat, 1 bar), 298.15 K, Langevin
middle integrator at 1/ps, 4 fs steps with hydrogen masses repartitioned to
3 amu, PME with 0.9 nm cutoff.

**Budget.** One window on the RTX 3080 runs at 245 ns/day; three windows side
by side reach about 400 ns/day in total. A PMF costs 15 x 4.5 ns = 67.5 ns
plus a 2.5 ns pull, about 4 GPU-hours. The full matrix asked for (native and
succinylated, both staggers, each without and with Ca2+ and Mg2+, sulfate with
the succinylated pair) is 14 PMFs, about 2.7 GPU-days. Run instead, about 1.4
GPU-days: the four monovalent PMFs, ion PMFs for the succinylated 1 D pair
with Ca2+ and with sulfate, and 20 ns at contact for the remaining ion
conditions (bridge counts only).

### Analysis

`aamd.pmf`: each window's series subsampled to its statistical inefficiency;
MBAR (pymbar) with a histogram free-energy surface on 0.02 nm bins, WHAM as an
independent check; zero = mean over 2.55-2.80 nm; error = block standard error
over four consecutive blocks of every window; convergence = the PMF from the
first and from the second half of production, separately. Tested on synthetic
umbrella data drawn from a known profile (`tests/aamd/test_pmf.py`).

Bridges (`aamd.bridges`) use the contact and bridge definitions of
`modcoll/ions/bridges.py` with its thresholds read from `params.toml`; an
all-atom ion has no coordination sites, so "inner" is the first shell
(O within d_MO + w_r: 0.247 nm Mg, 0.298 nm Ca). `tests/aamd/test_bridges.py`
puts the same configuration through both codes and requires the same bridges.
The coarse count at matched geometry (`aamd.ions_compare`) runs modcoll's
grand-canonical sampler around the all-atom positions of every charged group
of a frame, with the helix cores as excluded cylinders (radius 0.5 nm, as in
modcoll's S3 rows), at the all-atom bulk concentration of the ion and the
box's monovalent ionic strength.

## Live checks while the campaign runs

Every job (pull, window, contact run) checks itself after each 0.5 ns chunk
and writes `status.json` (ns done, ns/day, ETA, last checkpoint, worst check)
in its directory; a failing check writes one line to `aamd/work/WARNINGS`, and
a hard failure stops the job (it is marked `failed.json` and not retried).
`aamd/aamd status` summarises every job, the WARNINGS tail, the overlap of
neighbouring windows, and with `--convergence` the half-against-half PMF of
the data so far. `aamd/aamd run` starts (or restarts) three workers; each takes
the next job that is ready and not locked by a live worker, so after a crash
or a reboot `aamd/aamd run` resumes every job from its last chunk.

<!-- CHECKS -->

## Reproduce

```
# environment (conda-forge), user-local
micromamba create -n md --file aamd/environment.lock.txt
export PATH=<env>/bin:$PATH
python tools/build_tropocollagen.py            # assets/tropocollagen.pdb
python -m aamd.params.derive                   # SLL, SLH, SO4 libraries + ALY check (minutes)
python -m pytest tests/aamd                    # analysis, parameters, checks
python -m aamd.build native-1D ...             # or let the workers build
aamd/aamd run 3                                # the campaign in aamd/jobs.txt
aamd/aamd status --convergence
python -m aamd.selftest native-1D 6            # show that the checks fire
python -m aamd.predict                         # coarse predictions
python -m aamd.compare                         # tables and figures
python -m aamd.ions_compare SYSTEM aamd/work/SYSTEM/<job>
```

Large outputs (topologies, trajectories, restart files) stay in the
git-ignored `aamd/work/`; the analysed results are in `aamd/results/`.
