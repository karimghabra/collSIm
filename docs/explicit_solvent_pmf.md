# Explicit-solvent free energies of collagen segment pairs (`aamd/`)

All-atom molecular dynamics in explicit water, used to measure what the coarse
models assume: how strongly two triple-helical segments attract at a
favourable and an unfavourable axial stagger, whether lysine succinylation at
pH 4.2 changes that, and what Ca2+ and sulfate do between them. Every number
here is set against the prediction of `modcoll` (the site model of
`docs/modified_collagen_sites.md` and the ion model of
`docs/modified_collagen_ions.md`) at the same geometry, with the same charges.

## Results

All free energies in kT at 298 K, per 30-residue overlap (8.6 nm of each
helix), zero at 2.55-2.80 nm. Errors are block standard errors (4 blocks);
"halves" is the difference between the PMFs from the first and the second half
of production at the minimum. Numbers: `aamd/results/comparison.md` and `.json`;
profiles: `aamd/results/fig_pmf_*.png`, `fig_summary.png`.

### Measured against predicted

| system | measured minimum | at d (nm) | halves | coarse at that d | coarse el / hy | residual |
|---|---|---|---|---|---|---|
| native-1D | -7.5 +- 0.6 | 1.31 | 0.5 | -5.7 | -1.0 / -4.7 | -1.8 |
| native-05D | -7.3 +- 1.1 | 1.27 | 0.5 | -0.2 | +0.1 / -0.2 | -7.2 |
| succ-1D | -9.3 +- 0.5 | 1.39 | 1.3 | -2.5 | +1.1 / -4.0 | -6.8 |
| succ-05D | -8.9 +- 1.5 | 1.33 | 0.0 | +1.0 | +1.1 / -0.2 | -9.9 |
| succ-1D-Ca | -8.6 +- 1.6 | 1.35 | **4.6** | -3.1 | +0.9 / -4.3 | -5.5 |
| succ-1D-SO4 | -13.0 +- 2.0 | 1.37 | 0.3 | -2.9 | +0.9 / -4.1 | -10.1 |

"Coarse" is `modcoll`'s site-model pair free energy with both helices free to
rotate (`aamd.coarse`, same segments, same protonation, same monovalent ionic
strength), read at the measured minimum; its own minimum lies at the inner edge
of its range (1.20 nm). WHAM agrees with MBAR to 0.25 kT or better at every
minimum.

| difference | measured | coarse |
|---|---|---|
| native: G(1D) - G(0.5D) | -0.2 +- 1.2 | -5.5 |
| succinylated: G(1D) - G(0.5D) | -0.4 +- 1.6 | -3.4 |
| 1D: G(succinylated, pH 4.2) - G(native, pH 7.4) | -1.8 +- 0.8 | +3.2 |
| succ-1D: G(Ca2+) - G(monovalent) | +0.7 +- 1.7 (unconverged) | -0.6 (screening only) |
| succ-1D: G(sulfate) - G(monovalent) | -3.7 +- 2.0 | -0.4 (screening only) |

### Where the coarse model is right, wrong, or untestable

- **Right: two helices attract at contact, by several kT per 30 residues, and
  native 1 D is about as deep as predicted** (-7.5 against -5.7).
- **Wrong: stagger selectivity.** The 1 D and 0.5 D pairs bind equally
  (-0.2 +- 1.2 native, -0.4 +- 1.6 succinylated), where the coarse model puts
  the whole of its attraction in 1 D. The coarse 1 D depth comes almost
  entirely from the hydrophobic term (-4.7 of -5.7), and that term has nothing
  to work with at 0.5 D (overlap -3 KD^2 against +66). What holds the 0.5 D pair
  is absent from the model.
- **Wrong sign: succinylation at pH 4.2 strengthens the 1 D pair** by
  1.8 +- 0.8 kT; the coarse model weakens it by 3.2 kT (the electrostatic part
  turns repulsive: net charges -6/0 against -5/+5, and the Lys hydrophobic
  weights are kept for the succinylated residues). At pH 4.2, 8 of the 11
  succinyl groups, 6 of 11 Glu and 2 of 6 Asp are protonated in these
  segments; the most frequent contacts at the succinylated interface are
  protonated succinyl with Glu, Hyp and Gln (each in contact in 95-97 % of frames,
  `aamd/results/contacts_succ-1D.json`). A neutral carboxylic acid next to a
  carboxylate or an amide is a hydrogen-bond pair; the coarse model sees only a
  smaller mean charge.
- **Range: about right; contact distance: wrong.** The measured attraction
  sets in at about 2.0 nm (0.5 nm surface separation for the model's 0.75 nm
  radius; beyond it every PMF is within 1.5 kT of zero) and is steepest
  between 1.4 and 1.7 nm. The coarse native 1 D profile has a similar range
  (-0.9 kT at 2.0 nm). But the minima lie at 1.27-1.39 nm, 0.1-0.2 nm inside
  the sum of the model's nominal radii (the side chains interdigitate), and at
  the model's contact distance of 1.65 nm the measured PMFs are -2 to -5 kT,
  25-60 % of their depth.
- **Untestable here: Ca2+.** The succ-1D-Ca PMF has not converged (the halves
  differ by 4.6 kT at the minimum after extension of the four slowest runs to
  20 ns; the second half is the deeper), so neither the sign nor the size of
  the Ca2+ effect is measured. Its bridges are (below).
- **Missing: sulfate.** 50 mM Na2SO4 deepens the succinylated 1 D pair by
  3.7 +- 2.0 kT (halves agree to 0.3 kT; the block error is large because the
  1.3-1.5 nm windows have inefficiencies near 1 ns). The coarse model changes
  only the screening (-0.4 kT), and neither model forms a sulfate bridge in
  these geometries (below), so the effect is not a bridge in modcoll's sense.

### The hydrophobic term

Fitting the hydrophobic strength alone to the four monovalent PMFs
(dG - E_el = -eps_h H) gives eps_h = 0.13 +- 0.09 kT/KD^2, against the
model's assumed 0.05. That number only rescales the 1 D depth: with a
register-independent constant added (dG - E_el = -eps_h H + c, weighted by the
block errors) the fit gives **eps_h = 0.00 +- 0.06 kT/KD^2 and c = -9 +- 3 kT**
(chi^2 27 on 2 degrees of freedom; most of the scatter is the uniform
2-3 kT by which the succinylated pairs are deeper than the native ones). The
stagger contrast leaves no room for an eps_h large enough to make up the
depth; the assumed 0.05 is at the edge of one standard error, so a
Kyte-Doolittle term of that size is allowed but not required by these data.

The contact composition says the same from the other side
(`aamd.contacts`, heavy atoms within 0.45 nm, `aamd/results/contacts_*.json`):

| system | residue pairs in contact per frame | C-C / C-polar / polar-polar atom contacts | Spearman rho(frequency, h_i h_j) | contacts on pairs the coarse model calls attractive |
|---|---|---|---|---|
| native-1D | 25.8 | 0.40 / 0.46 / 0.15 | 0.20 | 66 % |
| native-05D | 21.1 | 0.34 / 0.48 / 0.18 | 0.10 | 64 % |
| succ-1D | 27.9 | 0.28 / 0.52 / 0.20 | 0.06 | 59 % |
| succ-05D | 21.8 | 0.32 / 0.50 / 0.19 | -0.02 | 60 % |
| succ-1D-Ca | 31.5 | 0.30 / 0.50 / 0.20 | 0.01 | 62 % |
| succ-1D-SO4 | 27.1 | 0.33 / 0.49 / 0.17 | 0.08 | 60 % |

Which residue pairs touch is unrelated to the coarse hydrophobic weight
product (rho 0.2 at most), and a third or less of the atom contacts are
carbon-carbon. The interface is mixed: apolar, imino (Pro, Hyp), polar and
charged residues all contribute, with 20-26 water molecules hydrogen-bonded to
both segments at contact (water O within 0.35 nm of both). The 0.5 D pair
makes about 20 % fewer residue contacts than the 1 D pair and still binds as
strongly.

### Ion bridges and the contact calibration

`aamd.ions_compare`: all-atom bridges by modcoll's definitions, every 10 ps;
coarse bridges from modcoll's grand-canonical sampler around the all-atom
positions of the charged groups of eight frames per run, at the all-atom bulk
concentration.

| system, run (mean d) | ions in contact with protein (of 12) | all-atom bridges per frame (inner-outer / outer-outer) | coarse bridges per frame (inner-outer / outer-outer) |
|---|---|---|---|
| succ-1D-Ca, 1.3 nm (1.36) | 3.7 | 0.03 +- 0.02 (0.03 / 0.00) | 0.14 (0.03 / 0.11) |
| succ-1D-Ca, 1.4 nm (1.35) | 4.1 | 0.34 +- 0.19 (0.00 / 0.34) | 0.42 (0.21 / 0.20) |
| succ-1D-Ca, 1.5 nm (1.47) | 2.6 | 0.00 | 0.26 (0.14 / 0.12) |
| succ-1D-SO4, 1.3 / 1.4 / 1.5 nm | 2.5 / 2.4 / 3.6 | 0 | 0 |
| native-1D-Ca, 1.3 nm (1.35) | 1.2 | 0.03 +- 0.03 (0.00 / 0.03) | 0.13 (0.08 / 0.05) |
| native-1D-Mg, 1.3 nm (1.35) | 1.5 | 0.10 +- 0.04 (0.00 / 0.10) | 0.10 (0.02 / 0.08) |
| succ-1D-Mg, 1.4 nm (1.41) | 2.2 | 0.07 +- 0.03 (0.00 / 0.07) | 0.14 (0.05 / 0.09) |

The ion-only runs are 20 ns at the contact (the 1.4 nm centre for succ-1D-Mg,
whose pull came to rest there); the others are the PMF windows.

No inner-inner bridge (an ion directly bound to a carboxylate of each helix)
forms in either model, and the all-atom bridges are outer-outer (water
between the ion and both groups) except in succ-1D-Ca at 1.3 nm, where all of
its 0.03 per frame are inner-outer. Bridges are rare in the
all-atom runs: 0.03-0.34 per frame, each long-lived when present (hence the
block errors), none in 12 ns at 1.5 nm in succ-1D-Ca. Against the coarse
sampler at the same geometry:

- **Mg2+: right within the errors** (0.10 against 0.10 native, 0.07 against
  0.14 succinylated). The all-atom Mg2+ keeps its six waters for the whole
  run, as it should on 20 ns (Mg2+ water exchange takes microseconds), so
  these runs cannot test the coarse model's inner-sphere Mg2+ states; they
  test the outer-sphere counts, which agree.
- **Ca2+: the coarse sampler predicts more**, about twice the all-atom count
  over the four Ca2+ runs (0.24 against 0.10 per frame) and four to five times in
  the two 1.3 nm runs. The excess is in inner-outer bridges (Ca2+ inner-sphere
  on one helix, outer-sphere on the other: 0.03-0.21 coarse, 0.00-0.03
  all-atom). In the all-atom runs 3-4 Ca2+ ions are in contact with the
  succinylated pair (1 with the native pair), but a contacting ion rarely
  reaches across to the other helix.
- **Sulfate: right**, zero in both, for want of cationic partners near the
  succinylated interface.

So the ion-carboxylate contact calibration is not contradicted for Mg2+; for
Ca2+ it errs towards too many cross-helix bridges, by a factor whose own
uncertainty (a few long events per run) is about as large as itself. It argues
against making the Ca2+-carboxylate contact stronger and is consistent with
making the inner-sphere Ca2+ reach somewhat shorter; it does not support a
recalibration on its own. The divalent ions' effect on the free energy is a
separate matter: bridges at 0.1-0.3 per frame cannot carry the several kT that
sulfate adds, and for Ca2+ the PMF did not converge.

### Short range and hydration

The force between the helices was examined at sub-nanometre surface
separation for the repulsion that osmotic-stress experiments on collagen
attribute to hydration, and for whether the attraction is water-mediated:

| system | minimum at d (surface separation) | inner wall (F 2 kT above minimum) | inner decay length | mean F at surface separation 0.5-1.0 nm |
|---|---|---|---|---|
| native-1D | 1.31 nm (-0.19) | 1.25 nm | 0.014 nm | +0.15 |
| native-05D | 1.27 nm (-0.23) | 1.23 nm | - | +0.37 |
| succ-1D | 1.39 nm (-0.11) | 1.31 nm | 0.020 nm | -0.35 |
| succ-05D | 1.33 nm (-0.17) | 1.27 nm | 0.014 nm | -0.07 |
| succ-1D-Ca | 1.35 nm (-0.15) | 1.27 nm | 0.030 nm | +0.77 |
| succ-1D-SO4 | 1.37 nm (-0.13) | 1.25 nm | 0.066 nm | -0.77 |

- **No hydration repulsion is visible.** Inside the minimum the free energy
  rises over 0.02-0.07 nm, a steric wall, not an exponential with a decay
  length of the order of a water layer. Between 0.5 and 1.0 nm surface
  separation the PMFs are flat to within 0.8 kT, with no consistent sign, and
  no PMF has a barrier outside its error band on the way in (succ-1D-Ca has a
  +1.2 kT shoulder at 1.9 nm, inside two block errors, in an unconverged
  profile).
- **The attraction is water-mediated at contact, in the sense that water stays
  in the interface.** The number of bridging waters rises steadily from 0 at
  2.5 nm to 20-26 at the minimum and does not drop when the helices touch; the
  interface is never dry. This does not separate an entropic (water-release)
  from an enthalpic contribution; that needs the temperature dependence.
- What the coarse model calls the contact distance (1.65 nm) is on the
  attractive slope of every measured PMF, 3-8 kT above its minimum.

### Is a modified-collagen correction needed?

Yes, in the sense that the coarse model gets the sign of the succinylation
effect wrong. The residual (measured minus coarse) is -1.8 kT for native 1 D
and -6.8 kT for succinylated 1 D. A correction that leaves the 1 D/0.5 D
selectivity alone would have to add about -5 kT per 30 residues to the
succinylated pair at pH 4.2 relative to native: about -0.5 kT per succinyl
group in these segments, or about -0.6 kT per protonated succinyl if, as the
contacts suggest, the neutral acid carries it. A single pair of segments at a
single pH cannot say whether the term follows protonation, acyl-chain carbon
or both; it also cannot fix the larger problem, a register-independent
attraction of 7-9 kT per 30 residues that the model lacks in every system.

### Limits

- **Two segments, one sequence window.** One 30-residue window, chosen with
  the coarse model as the one with the largest 1 D preference; another window
  could rank the staggers differently. Each segment is a 30-residue piece of a
  300 nm molecule with its ends held, so twist, bending and end effects of the
  full molecule are absent.
- **One coordinate.** The PMF is along the inter-axial separation with the
  axes held parallel and the stagger fixed. The free energy of rotating either
  helix about its own axis is integrated over, but slowly: the residual
  disagreement between halves (up to 1.3 kT in the converged profiles, 4.6 kT
  for succ-1D-Ca) is that rotation and the interfacial side chains relaxing on
  a 1 ns scale.
- **Force field.** ff14SB/TIP3P, Hyp from AmberTools' amino12.lib,
  succinyl RESP charges fitted on two backbone conformations with the side
  chain extended, Li-Merz
  12-6-4 Ca2+/Mg2+ with a carboxylate-specific C4. ff14SB with TIP3P has been
  reported to overstate protein-protein attraction, so the absolute depths
  may be too large; differences between systems (the stagger and chemistry
  contrasts) are less exposed to that.
- **Fixed protonation**, assigned once (the Henderson-Hasselbalch number of
  charged groups per residue type with modcoll's pKa set, a seeded choice of
  which ones carry the charge);
  charge regulation at the interface (a carboxylate next to a carboxylate
  taking up a proton, the effect the coarse model also lacks) is absent.
- **Sampling.** Statistical inefficiencies of 0.3-1.6 ns at contact against
  contact-region windows of 12-20 ns: 6-23 independent samples in the slowest windows. The
  stagger and chemistry contrasts are several error bars from the coarse
  predictions; the ion effects are not.

### Next steps

- Run the same contrasts at one or two more sequence windows, including one
  the coarse model does not favour, to separate the register-independent
  attraction from window choice.
- Converge succ-1D-Ca (the four contact-region windows to 40 ns, or replica
  exchange between windows) before using it to calibrate anything.
- A temperature repeat of one monovalent PMF (e.g. at 310 K) would split the
  contact free energy into enthalpy and entropy, which is what bears on
  water-release (entropy-driven) assembly; not run here.
- Add to the coarse model a register-independent contact term, and a
  succinyl/acid term, fitted to these PMFs, and test whether the 1 D
  preference of the full molecule survives it.

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

**What came out** (`aamd/params/*_resp.json`):

| residue | net charge | RESP rrms | key charges (e) | ff14SB analogue |
|---|---|---|---|---|
| SLL | -1 | 0.023 | carboxylate C 0.843, O -0.835 x 2; amide N -0.653, H 0.320, C 0.703, O -0.592 | GLU CD 0.805, OE -0.819 |
| SLH | 0 | 0.100 | acid C 0.784, O= -0.625, O(H) -0.682, H 0.467 | GLH CD 0.680, OE1 -0.584, OE2 -0.651, HE2 0.464 |
| SO4 | -2 | 0.004 | S +1.365, O -0.841 x 4 (S-O 0.149 nm) | |

(The relative rms of a neutral residue is larger because its potential is
small.) **Check of the procedure**: the AmberTools ALY refitted with it differs
from the shipped charges by 0.066 e rms, 0.17 e at most. The large differences
are on the buried side-chain carbons (CB -0.07 against -0.23, CG 0.04 against
0.16, CE -0.04 against -0.21), the atoms RESP determines worst; the exposed
groups agree to 0.1 e (amide C 0.746 against 0.857, O -0.592 against -0.609, N
-0.616 against -0.703, H 0.334 against 0.379; acetyl methyl C -0.383 against
-0.478), and the charges summed over chemical groups to 0.07 e. Part of the
difference is the geometry step (ff14SB minima here); Belfon & Simmerling's
exact QM protocol was not available to repeat. This is the size of the
uncertainty to keep in mind for the succinyl charges.

Versions: OpenMM 8.6.1 (CUDA, mixed precision), AmberTools 26.0, parmed
4.3.1, psi4 1.11, pymbar 4.2.0, mdtraj 1.11.1, numpy 2.4.6 (all conda-forge;
exact builds with checksums in `aamd/environment.lock.txt`).

### Systems

Each pair sits in a rectangular box of TIP3P water, 7.97 x 5.06 x 12.57 nm
(about 40,700 atoms): for the native pairs tleap's box with 1.25 nm of water
beyond the solute in x and y at the largest separation and 1.2 nm beyond the
ends in z; the succinylated systems are given the same box (their side chains
are built fully extended, and tleap would otherwise make the box 37 % larger
around them), so that every system has the same volume and the same numbers
of ions. The native 0.5 D box came out 1 % smaller in each direction. NaCl 150 mM (counted per
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
stopped following the restraint at about 1.45 nm (the steric contact), so the
regular windows start at 1.4 nm. The 1.4 nm window then sat at its own centre
(mean 1.398 nm), i.e. the free-energy minimum was at or below the edge of the
sampled range, so every PMF gets one more window at 1.3 nm (`contact S 1.300`
in `aamd/jobs.txt`, started from the 1.4 nm state, same spring). Per window
0.5 ns equilibration, discarded, and 4 ns production. NPT (Monte Carlo
barostat, 1 bar), 298.15 K, Langevin middle integrator at 1/ps, 4 fs steps
with hydrogen masses repartitioned to 3 amu, PME with 0.9 nm cutoff.

**Contact-region extension.** At 4 ns the native 1 D windows from 1.3 to
1.9 nm had statistical inefficiencies of 160-590 ps (against 40-130 ps at
2.1-2.8 nm), and the PMFs from the first and second halves of production
differed by up to 4.7 kT at contact, with the windows at 1.4 and 1.5 nm
still drifting inwards. The slow variables are those the umbrella does not
act on: rotation of each helix about its own axis (left free, so that the
pair can find its best register of side chains) and the rearrangement of
interfacial side chains and water. Those seven runs are extended to 16 ns
(job lines `window S I 16`, which continue a finished window from its last
chunk). The other five PMFs get the same seven runs (1.3-1.9 nm) extended to
12 ns, and in succ-1D-Ca, whose halves still disagreed by 2 kT at 12 ns, the
four slowest (1.3, 1.4, 1.6, 1.7 nm) to 20 ns.

**Budget.** One window on the RTX 3080 runs at 245 ns/day; three windows side
by side reach about 400 ns/day in total (330-520 ns/day in production,
depending on what else shares the machine). The full matrix (native and
succinylated, both staggers, each without and with Ca2+ and Mg2+, sulfate with
the succinylated pair) would be 14 PMFs. Run instead: the four monovalent
PMFs, ion PMFs for the succinylated 1 D pair with Ca2+ and with sulfate, and
20 ns at contact for the three remaining ion conditions (bridge counts only).
Planned at 4 ns per window, about 1.4 GPU-days; with the contact-region
extensions the campaign simulated 890 ns of windows and contact runs plus
nine 2.5 ns pulls, about 2 days of the GPU running three jobs at a time.

### Analysis

`aamd.pmf`: each window's leading non-stationary part dropped (the start,
at most half the window, that maximises the number of independent samples;
Chodera, JCTC 12, 1799 (2016)); MBAR (pymbar) on every fifth 1 ps sample with
a histogram free-energy surface on 0.02 nm bins, WHAM as an independent check;
zero = mean over 2.55-2.80 nm; error = block standard error over four
consecutive blocks of every window; convergence = the PMF from the first and
from the second half of production, separately. The number of independent
samples per window (length / statistical inefficiency) is reported with
each PMF. Tested on synthetic
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

Checks run on every 0.5 ns chunk (`aamd/checks.py`, where the thresholds are
defined):

| check | measured per chunk | warn | fail |
|---|---|---|---|
| finite | NaN or inf in energies, coordinate, final positions; an integrator stop | - | any |
| temperature | mean kinetic temperature minus 298.15 K | > 2 K | > 5 K |
| temperature_peak | largest instantaneous kinetic temperature | - | > 400 K |
| energy_drift | change of mean potential energy since the previous chunk | > 0.3 % | > 2 % |
| volume_drift | change of mean box volume since the previous chunk | > 1 % | > 3 % |
| bond_stretch | longest bond between solute heavy atoms in any frame | > 0.20 nm | > 0.25 nm |
| helix_rmsd_local | backbone RMSD to the built helix of each two-triplet block (all three chains, superposed on its own), the largest, chunk mean | > 0.20 nm | > 0.30 nm |
| helix_rmsd_global | the same over the whole segment | > 0.50 nm | - |
| interchain_hbonds | backbone N-H...O=C bonds between chains of one segment (N...O < 0.35 nm, angle > 120 deg), fraction of the built helix's count, smaller segment | < 60 % | < 40 % |
| umbrella_centre | mean separation minus the window centre | > 0.15 nm | > 0.30 nm |
| lateral_offset | mean |Delta y| and mean |Delta z - z0| of the centroids | > 0.05 nm | > 0.15 nm |
| ion_count | ions of each kind in the topology vs. the build record | - | any difference |
| ion_coordination | mean number of O within 0.26 nm of Mg2+ / 0.30 nm of Ca2+, final frame | Mg outside 5.5-6.5, Ca outside 6-8.5 | Mg outside 4-8, Ca outside 4-10 |
| (status) overlap | sum of min(p, q) of neighbouring windows' histograms, 0.01 nm bins | < 0.03 | - |
| (status) convergence | largest difference between first-half and second-half PMF, 1.25-2.75 nm | > 1 kT | - |

Two thresholds were set from the first healthy run rather than in advance.
The temperature estimator reads about +1 K at 4 fs (the kinetic energy of the
Langevin middle integrator at the half step), so the warning starts at 2 K.
The first integrity measure was the RMSD of the whole segment; on a healthy run
it reached 0.30-0.33 nm with every inter-chain hydrogen bond present and no
two-triplet block above 0.13 nm: the 9 nm segment bends and its twist relaxes
away from the (Pro-Pro-Gly)10 template. Fraying is what the check is for, so
the graded measure is the local one and the global RMSD only warns at 0.5 nm.

**The checks fire** (`python -m aamd.selftest native-1D 6`, three short runs
of the 2.0 nm window; output in `aamd/results/selftest.json`):

| run | what was broken | outcome | checks not ok |
|---|---|---|---|
| healthy | nothing (0.02 ns equilibration + 2 x 0.05 ns) | completed | none |
| bad_timestep | 12 fs instead of 4 fs | stopped in equilibration | finite: fail ("Particle coordinate is NaN") |
| no_restraint | umbrella, lateral and alignment restraints k = 0, window centre moved 1 nm from the start state | stopped after the first chunk | umbrella_centre: fail (-0.37 nm) |

A failing check writes one line to WARNINGS, for example

```
2026-11-17T03:22:31 selftest/no_restraint chunk 0 umbrella_centre FAIL value=-0.3723 warn[|d|>0.15 nm] fail[|d|>0.30 nm] r0=3.000
```

and a hard failure marks the job failed and frees its worker for the next job.
If failures stop the campaign as a whole (every remaining job of a PMF failed,
or the workers die), that is reported as blocked rather than worked around.

**How the campaign ended.** 112 jobs, 889.5 ns, no FAIL. Four WARN lines, all
`umbrella_centre` in windows on the steepest part of a PMF (succ-1D-Ca 1.7 nm,
succ-1D-SO4 1.6 nm): there the mean force is about 50 kT/nm, which a
1000 kJ/mol/nm^2 spring balances 0.12-0.18 nm inside its centre; neighbouring
windows still overlap (smallest overlap 0.07-0.24 per PMF). The rolling
convergence check stayed at WARN for every PMF: the largest first-half against
second-half difference anywhere in 1.25-2.75 nm is 1.5-2.7 kT (6.4 kT for
succ-1D-Ca), against 0.0-1.3 kT at the minima (4.6 kT for succ-1D-Ca). The
half-profiles of 4 ns windows are noisy at the 1-2 kT level on their own, so
the flag is read together with the block errors in the results above. One
episode of host memory pressure stalled the three running jobs for about three
hours; they resumed from where they stood with no check failing.

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
python -m aamd.ions_compare SYSTEM RUN          # RUN: w00, c1.300, ...
# viewing: trajectories, README, PyMOL sessions and movies
micromamba create -n viz --file aamd/environment-viz.lock.txt
<viz env>/bin/python -m aamd.export /mnt/d/collSIm-aamd-trajectories
```

Large outputs (topologies, trajectories, restart files) stay in the
git-ignored `aamd/work/`; the analysed results are in `aamd/results/`.

## Viewing the trajectories

`aamd.export` copies every system's `system.pdb` and `segments.pdb` and every
run's solute trajectory (protein and ions, no water; 0.5 ns per DCD file, a
frame every 10 ps) to `/mnt/d/collSIm-aamd-trajectories/<system>/<run>/`
(on Windows `D:\collSIm-aamd-trajectories`), with the layout of `aamd/work/`.
Each run gets `solute_chains.pdb`, the DCD topology with chain ids (A, B =
alpha1(I), C = alpha2(I) of segment 1; D, E, F the same for the staggered
partner; ions chain I). `README.txt` there lists every system and run (centre,
mean separation, length) and how to load a trajectory in PyMOL, ChimeraX, VMD
and mdtraj.

For each system the run nearest contact (mean separation closest to the PMF
minimum; the contact run for the ion-only systems) also has a PyMOL session
`session.pse` (open in PyMOL; up to 400 frames, fitted on segment 1) and a
150-frame movie (`movie.mp4`, `movie.gif`): helix axes horizontal, the two
segments one above the other; chains in distinct colours (pale and dark
green, orange, pink); Asp/Glu/succinyl-lysine side chains red and
Arg/Lys/His blue sticks; Ca2+ yellow and Mg2+ cyan spheres, sulfate
purple spheres; Na+ and Cl- hidden.
