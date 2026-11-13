# Modified collagen: per-molecule sites, random modification, two-molecule stagger scan

`modcoll/` is a Python package, separate from the C++ engine, that asks one
question of the sequence: **if the lysines of type I collagen are
succinylated and/or acetylated at random, does a pair of molecules still
prefer an axial register, and which one?** It has no divalent ions (those are
explicit particles in `modcoll/ions/`), no dynamics and no many-body effects.
It is the level-zero model that later levels have to agree with.

Three rules it does not break:

- **No mean field over modification.** Every molecule carries its own
  concrete pattern. Energies are computed for concrete pairs of patterns;
  anything reported for an ensemble is a statistic over sampled molecules or
  sampled pairs, with the spread next to the mean.
- **Every random number comes from a seeded `numpy.random.Generator`.**
  Re-running a command with the same seed reproduces its tables byte for byte.
- **The native molecule is the control.** Nothing is produced for modified
  collagen unless unmodified collagen at pH 7.4 shows the D-register first.

## Run it

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-modcoll.txt
.venv/bin/python -m pytest tests/modcoll --ignore-glob='*test_ions*'   # 191 tests, about 50 s
.venv/bin/python -m modcoll.cli_ensemble              # S1, about 2 min
.venv/bin/python -m modcoll.cli_stagger --chemistry succ    # S2, about 3 min on 12 threads
.venv/bin/python -m modcoll.cli_stagger --chemistry acet
.venv/bin/python -m modcoll.cli_stagger --chemistry mixed
.venv/bin/python -m modcoll.cli_stagger --chemistry succ --f 0,0.8,0.9,0.97,1 \
    --ph 3.8,4.0,4.2,4.4,4.6 --outdir results/modcoll/stagger_succ_phscan
.venv/bin/python -m modcoll.cli_composition           # composition scan, about 16 min on 12 threads
```

The tests of the ion model (`tests/modcoll/test_ions_*`) are described in
`docs/modified_collagen_ions.md`; they need numba to finish in reasonable time.

Results land in `results/modcoll/`: CSV/JSON summaries and PNG figures are
committed, the per-pair table and the profile arrays go to `raw/` and are
git-ignored (regenerate them with the commands above). The geometry is read
from `assets/tropocollagen.pdb`; if that git-ignored file is missing,
`tools/build_tropocollagen.py` is run once to make it.

## The site model (`modcoll/sites.py`)

A molecule is an array of 3157 sites: all 3154 residues of the three chains
(human COL1A1 P02452 twice, COL1A2 P08123 once; 1014-residue helices plus
telopeptides) and the three chain C-termini. Each site has a chain, a residue
index, a type, an axial position `s` (nm, 0 = N-terminal end of the helix), an
azimuth `phi` (rad), a radial offset `r` (nm) and a Kyte-Doolittle weight.

| type | count | charge when ionised | pKa, recommended set | pKa, repository set |
|---|---|---|---|---|
| Asp | 92 | -1 | 3.9 | 3.65 |
| Glu | 143 | -1 | 4.3 | 4.25 |
| His | 18 | +1 | 6.5 | 6.0 |
| Lys (incl. hydroxylysine) | 107 | +1 | 10.4 | 10.53 |
| Arg | 161 | +1 | 13.8 | 12.48 |
| Tyr | 13 | -1 | 9.8 | does not titrate |
| chain C-terminus | 3 | -1 | 3.7 | 3.67 |
| chain N-terminus (if free) | 0 | +1 | 8.0 | 8.0 |
| Asn / Gln | 46 / 79 | 0 | deamidate to Asp / Glu | |
| succinyl-Lys, succinyl-N-terminus | - | -1 | 4.5 | 4.5 |
| acetyl-Lys, acetyl-N-terminus | - | 0 | | |

**Geometry is reused, not re-derived.** Positions come from the full-atom
model that `tools/build_tropocollagen.py` threads through the 1K6F screw
operator, reduced to one point per residue exactly as
`tools/azimuthal_profile.py::load_residue_surface` does: the centroid of the
side-chain heavy atoms (CA for glycine). A test compares the two residue by
residue. The radial offsets that fall out are 0.19 nm for glycine and
0.50-0.77 nm for anything that can carry a charge. A second position per site,
the titratable group itself (Lys NZ, Arg CZ, the carboxylate oxygens, ...,
0.55-0.98 nm from the axis), is stored and selectable with
`--position group`; the default stays with the existing tool's centroid
because the side-chain rotamers of the model are ideal conformers, not a
sampled ensemble, so the group position is not obviously better.

Two things are deliberately **not** copied from the existing tool:

1. **Axial origin.** `azimuthal_profile.py` reads the un-shifted z of the PDB
   file, which puts the helix at s = -13.9 ... 276.3 nm, but places its
   telopeptides relative to 0 ... L (N-telopeptides at -4.5 ... -0.3 nm,
   C-telopeptides at 290.6 ... 297.4 nm). In that tool the N-telopeptides
   therefore sit inside the helix and the C-telopeptides 14 nm beyond its end,
   and the first 6 nm of helix are clipped onto one grid row. Here both use the
   origin `build_tropocollagen.py` itself uses for `atoms.bin` and
   `residues.npz` (helix from 0 to L = 290.4 nm). The existing tool is left
   untouched by this package; its "telopeptides included" numbers in the
   README were computed with that offset.
2. **Helix anchor and chain ends.** The existing builder takes the helix to
   be the longest run with Gly at every third position and the existing
   profile tool takes 16 residues before and 25 after that run as
   telopeptides. Both rules are wrong in ways that happen not to hurt the
   human helix: for human alpha2 the longest run is 1017 residues, because
   the C-telopeptide starts GGG, and is only right after truncation to the
   alpha1 length (for bovine alpha2 the same rule starts one triplet early);
   and the 16/25 window on alpha2 holds 5 N-propeptide and 10 C-propeptide
   residues, among them a lysine that would become a modifiable site, while
   skipping the first three telopeptide glycines. Here the helix is anchored
   on the annotated region (UniProt: alpha1 179-1192, alpha2 91-1104) and
   checked to be Gly-X-Y throughout, and the chain ends are explicit, per
   extraction method (`--telopeptides`):

   | model | chain ends | Lys / Arg / His / Asp / Glu | free N-termini |
   |---|---|---|---|
   | `mature` (default) | intact mature chains, alpha1 162-1218, alpha2 80-1119 | 107 / 161 / 18 / 92 / 143 | 0 |
   | `pepsin` | alpha1 loses pQLSYGY and the distal 19 residues of its C-telopeptide, alpha2 loses two N-terminal residues (bovine cut sites; Qian et al. 2017, Galea et al. 2000) | 105 / 157 / 16 / 90 / 141 | 3 |
   | `none` | helix only | 102 / 156 / 16 / 82 / 139 | 3 |
   | `tool` | the old 16/25 window, for comparison with the existing tool only | 108 / 163 / 18 / 94 / 143 | 3 |

   The fully succinylated isoelectric point differs by less than 0.03 pH
   between the first three (4.29, 4.27, 4.30) and the net charge at pH 4.2 by
   a few e (+16, +12, +18). The native gate passes with all four models. Another species needs its own anchor entries (`sites.CHAINS`), not
   just another FASTA file, and the full-atom model has to be rebuilt for it;
   the site array is checked against the model residue by residue and a
   mismatch raises. All results below use `mature`.

Telopeptides are placed as the existing tool places them: 0.28 nm per residue
off the helix ends, azimuth uniform (a flexible end is carried as a ring of
charge). The ring radius, 0.55 nm, is an assumption.

Things that are simply not modelled: hydroxylysine is treated as lysine and
its glycosylation is ignored; there are no crosslinks; tyrosine and cysteine do
does not titrate, and hydroxyl groups are never acylated (anhydrides can
also acylate tyrosine); both intact mature N-termini are pyroglutamate and
therefore carry no amine. Pepsin-cut chains do have free alpha-amines, which
titrate (pKa 8.0) and are acylated like lysines; `n_terminus="free"` forces
them onto intact chains.

## Random modification (`modcoll/modify.py`)

`modify(molecule, rng, f_succ, f_acet, f_deam, weights=None)` returns a copy
with one random pattern. Each amine is, independently, succinylated (+1 ->
-1), acetylated (+1 -> 0) or left alone; each Asn/Gln is independently
deamidated. `f_succ`, `f_acet`, `f_deam` are mean fractions of the eligible
sites.

With non-uniform reactivity `weights`, site i is modified with probability
`p_i = 1 - exp(-w_i tau)`, the result for independent pseudo-first-order
reactions after a common reaction extent `tau`, and `tau` is solved so that
the mean of `p_i` equals the requested fraction (the quantity a free-amine
assay measures). Uniform weights reduce to `p_i = f`.

Uniform reactivity is the default. Two alternatives are provided
(`accessibility_weights`):

- `"crosslink_sites"`: the six helical cross-linking lysines (alpha1 Lys87
  and Lys930 on both chains, alpha2 Lys87 and Lys933) react ten times faster.
  These are the residues that cyanate and TNBS label first in native
  collagen, with a pKa near 8.5 (Gonzalez et al. 1984), the only sites
  glucose modifies appreciably (Hudson et al. 2018) and the preferred
  carbamylation site (Taga et al. 2017). That they are also first toward
  anhydrides, and the factor of ten (plausibly 3-30), are inferences: no
  site-resolved acylation map of collagen exists. In S1 this weighting moves
  the isoelectric point by at most 0.02 pH at any f; what it changes is
  *which* lysines are modified at low f.
- `"ion_pair"`: the 36 helix lysines in a K-G-D / K-G-E motif react half as
  fast. An assumption with no supporting data, kept as a differently shaped
  weighting for sensitivity tests.

At the 90 % or so that succinylation of native collagen is reported to reach
(above 90 %, Zhang et al. 2007; 90.2 %, Xu et al. 2017), site preference
only decides which ten or so lysines stay unmodified, and nothing is known
about which those are.

`modify` draws exactly one uniform deviate per amine and one per amide
whatever the fractions, so two calls with the same generator state give
**nested** patterns: the sites modified at f = 0.25 are a subset of those at
f = 0.5. Both command-line tools use this, which makes trends across f less
noisy; the price is that the values at different f are not independent
samples. The pair-to-pair spread at each f is unaffected.

What modification does not change here: positions (a succinyl carboxylate is
about 0.5 nm further out than the amine it replaces) and hydrophobic weights
(the Kyte-Doolittle scale has no entry for acyl-lysines). The second point
matters and is taken up under "Known limits".

## Protonation (`modcoll/electrostatics.py`)

Each site titrates independently (Henderson-Hasselbalch). Two named pKa sets
exist (`electrostatics.PKA_SETS`, `--pka-set`):

- **`recommended`**, the default for every tabulated result: model-compound
  values at 25 C and ionic strength 0.05-0.1. Asp 3.9 (alanine pentapeptides,
  potentiometry and NMR, Grimsley et al. 2009, which revised the 3.67 of
  Thurlkill et al. 2006; Gly tripeptides 3.86-3.88, Platzer et al. 2014),
  Glu 4.3, His 6.5, Lys 10.4, Tyr 9.8 (Thurlkill et al. 2006; Platzer et al.
  2014), Arg 13.8 (Fitch et al. 2015), termini 3.7 and 8.0.
- **`repository`**: the README's set, used for the native gate so that the
  gate stays comparable with the existing tools.

The difference that matters is Asp: 3.65 against 3.9 is 10.5 fewer negative
charges on the 92 Asp at pH 4.2, 14.8 e on the whole molecule with the other
shifts. Results at pH 4.2 computed with the repository set would be off by
that much.

The succinyl carboxylate is the free acid end of a succinic monoamide. No
measured pKa exists for succinyl-lysine or for any N-alkyl succinamic acid.
Succinic acid has pKa1 = 4.21, and because its first proton can leave from
either of two equivalent carboxyls, one carboxyl alone has a microscopic pKa
of 4.21 + log10(2) = 4.51; related monoamides and keto acids fall at 4.4-4.7.
The value used is 4.5 with an uncertainty of about +-0.15, which is +-9 e on
the net charge at pH 4.2. At pH 4.2 a succinyl carboxylate is one third
ionised.

Protonation is a function `(molecule, environment) -> charges`, passed as
`charge_model` to everything that needs charges. A charge-regulation model
replaces it without touching the site array or the energy code; a test does
exactly that with a stand-in.

### S1 sanity gates

| gate | expected | recommended set | repository set | verdict |
|---|---|---|---|---|
| fully succinylated pI | 4 - 4.5 | **4.29** | 4.21 | passes |
| native pI | about 9 | **9.84** | 10.11 | **fails** |

The native value is what the sequence gives: 107 Lys + 161 Arg against 92 Asp
+ 143 Glu + 3 C-termini leaves +32 at pH 7.4, and the first groups able to
remove that excess are the 13 tyrosines and then the lysines. Extracted
collagens are partly deamidated; in this model deamidating **18 %** of
Asn/Gln brings the pI to 9.0.
That is a statement about the model, not a measurement of any preparation,
and no deamidation is applied by default (`--f-deam` on both tools adds it).

One more number that does not reproduce: the README quotes a net charge of
about +81 e at pH 3.5. Its own pKa set and residue counts give +229 e (+225
with the mature chain ends; +239 with the recommended set).

## Pair energy (`modcoll/electrostatics.py`, `modcoll/stagger.py`)

Two straight molecules, axes parallel, at a fixed axis separation. Molecule B
is displaced by the stagger along the common axis; each molecule turns a
"facing" azimuth toward the other. Antiparallel pairs have B turned end over
end first (s -> L - s, phi -> -phi, the existing tool's convention). The
energy is the literal double sum over the sites of the two patterns:

```
E(stagger, facing A, facing B) =
      sum_ij q_i q_j  l_B exp(-r_eff / lambda_D) / r_eff      r_eff = sqrt(r_ij^2 + a^2)
    - eps_h sum_ij h_i h_j exp(-r_ij^2 / 2 sigma_h^2)
```

| parameter | value | status |
|---|---|---|
| Bjerrum length `l_B` | 0.71 nm | water, 298 K |
| Debye length `lambda_D` | 0.305 nm / sqrt(I [M]) | from the monovalent ionic strength; 3.05, 1.36, 0.79, 0.56 nm at 10, 50, 150, 300 mM |
| soft core `a` | 0.35 nm | regularisation: caps a unit-charge pair near 2 kT |
| axis separation | 1.65 nm | 2 x 0.75 nm nominal radius + 0.15 nm gap; the README's refitted contact distance |
| hydrophobic weights `h` | centred Kyte-Doolittle | as the existing tools |
| hydrophobic range `sigma_h` | 0.5 nm | the existing tool's 0.35 nm smoothing applied to both partners |
| hydrophobic strength `eps_h` | 0.05 kT per KD unit squared | **assumed**: two leucine-like side chains in contact gain about 1 kT |

The sum is evaluated for all staggers (0.2 nm steps) and all 24 x 24 facing
angles at once by depositing the sites on a cylindrical grid and doing one
three-dimensional FFT cross-correlation per pair of radial shells. Nothing is
averaged over azimuth and the kernel is the pair potential at the true
three-dimensional distance. The only approximation is the grid (linear
assignment in s, phi and r). Against the brute-force double sum, which is kept
in the package as the oracle, it is exact to 1e-9 for sites on grid nodes and
within 0.05-0.1 kT rms for real molecules, where the features of interest are
5-25 kT.

Each profile is reduced over the two facing angles three ways: the minimum
(`tot_min`), the free energy of a pair free to rotate, `-ln <exp(-E)>`
(`tot_boltz`), and the plain average (`tot_mean`, the azimuth-blind limit that
corresponds to the existing 1D tool). The electrostatic term alone is reduced
too (`el_min`, `el_boltz`).

Numbers reported from a profile:

- **leading minima**: local minima beyond 5 nm stagger, deepest first, at
  least 15 nm apart (the existing tool's rule);
- **D-comb z-score**: the mean depth of the profile within +-3 nm of 1 D, 2 D,
  3 D, 4 D, compared with the same comb at every other phase, in units of the
  spread over phases. Scale-free. `rank = 0` means no other phase is deeper;
- **fraction of pairs at k D**: the share of pattern pairs whose own deepest
  minimum lies within 3 nm of a D multiple. Chance level is 8.1 %;
- **profile autocorrelation** at a shift of D and the dominant period, which
  assume nothing about where in a period the minima sit. The molecule is 4.3 D
  long, so periods cannot be resolved finely.

## The native control

Parallel pair of unmodified molecules, pH 7.4, 150 mM, repository pKa set
(`results/modcoll/stagger_succ/fig_native_gate.png`):

| profile | five deepest minima (stagger in D, energy in kT) |
|---|---|
| minimum over facing angles | **1.01** (-25.0), 1.31 (-18.1), 0.43 (-16.4), **1.99** (-15.9), **3.00** (-15.3) |
| averaged over facing angles | **1.01** (-17.3), **1.99** (-10.4), **3.00** (-9.3), 0.62 (-7.1), 1.31 (-6.7) |
| electrostatics alone, minimum over angles | 0.23 (-8.7), 1.42 (-8.2), 1.01 (-8.1), 1.90 (-7.1), 0.61 (-7.0) |

The gate (`stagger.native_gate`) requires the deepest minimum over facing
angles, and the three deepest azimuth-averaged minima, to lie within 0.05 D of
integer multiples of D. It passes: the azimuth-averaged profile gives the
README's 1.00 D / 1.99 D / 3.00 D, and the azimuth-resolved profile has its
deepest minimum at 1.01 D, 7 kT below the next. With the recommended pKa set
the same control gives the same minima within 0.1 kT (recorded in `run.json`
as `native_control_with_result_pka_set`).

What the gate does not hide:

- With facing angles resolved, two non-D registers (1.31 D, 0.43 D) rank
  second and third. The README says the same of the existing azimuthal table.
- **The D-register comes from the hydrophobic term.** Electrostatics alone
  does not put a leading minimum at a D multiple, here or in the existing 1D
  tool: its raw electrostatic channel, before the funnel reconstruction, has
  minima at 0.18 D, 1.89 D, 1.42 D, 0.91 D and its raw hydrophobic channel at
  1.00 D, 0.36 D, 1.99 D, 3.00 D. S1 shows why (below): the positions of
  charged residues repeat with D, their signs do not.
- So the gate depends on the one assumed strength:

| variant | deepest (D) | gate |
|---|---|---|
| default (`eps_h` = 0.05) | 1.01 | passes |
| `eps_h` = 0.1 | 1.00 | passes |
| `eps_h` = 0.025 | 1.01 | fails: azimuth-averaged top three are 1.01, 3.00, 0.44 |
| `eps_h` = 0.0125 | 1.01 | fails likewise |
| `eps_h` = 0 | 0.23 | fails |
| charge at the titratable group | 1.01 | passes |
| telopeptides as the existing tool / none | 1.01 / 1.00 | passes / passes |
| gap 0.05, 0.3 nm | 1.01 | passes |
| gap 0.5 nm | 1.00 | fails: 0.44 D enters the azimuth-averaged top three |
| soft core 0.25, 0.5 nm | 1.01 | passes |

The 1 D minimum itself survives a fourfold reduction of the hydrophobic
strength; the full 1 D / 2 D / 3 D ladder needs about the default.

Two more properties of the control worth knowing. At pH 3.5 the native pair
has no bound register at 10-150 mM (net charge +239 e each, energies positive
over most of the overlap), which is the acid-stock behaviour. And the model
**does not prefer parallel over antiparallel pairs**: the deepest antiparallel
minimum is 1.5 kT *below* the deepest parallel one at pH 7.4 and 150 mM, and
further below at low salt. The antiparallel scan covers twice the stagger
range, which accounts for some of this, but polarity selection is not
something this model can be used to argue.

## S1: the modified ensemble (`results/modcoll/ensemble/`)

256 molecules per point, seed 20261112, recommended pKa set.

**Isoelectric point** (`pi_vs_f.csv`, `fig_pi_vs_f.png`), each chemistry
separately because the isoelectric point alone cannot tell them apart:

| f | 0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.97 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| succinylation | 9.84 | 9.22 | 5.71 | 4.83 | 4.51 | **4.37** | 4.31 | 4.29 |
| acetylation | 9.84 | 9.63 | 8.80 | 5.66 | 4.97 | 4.76 | 4.67 | 4.64 |
| succinyl + acetyl 1:1 | 9.84 | 9.48 | 6.54 | 5.07 | 4.67 | 4.51 | 4.45 | 4.42 |
| deamidation only | 9.84 | 9.54 | 7.49 | 5.41 | 4.96 | 4.81 | 4.76 | 4.74 |

- Lysine succinylation alone reaches 4.29 at f = 1 and 4.37 at the more
  realistic f = 0.9. A measured pI of 4.2 therefore points to near-complete
  succinylation plus something else that this table does not fix: some
  deamidation, or pKa values 0.1 lower than assumed. Measured: 4.01 at 90 %
  succinylation (Xu et al. 2017). The composition scan below maps every
  mixture of the three chemistries that gives 4.2.
- The pI drops steeply between f = 0.1 and 0.3 and that is where molecules
  differ most: the molecule-to-molecule standard deviation of the pI is 0.7 pH
  units at f = 0.1-0.2 for succinylation (single molecules from 5.4 to 9.4 at
  f = 0.2) and below 0.1 for f >= 0.5.

**Net charge between molecules** (`net_charge.csv`: mean, sd, minimum, 5th
percentile, median, 95th percentile, maximum and the share of net-positive
molecules; `net_charge_histogram.csv`, `fig_net_charge_distribution.png` at
pH 4.2):

| succinylation, pH 4.2 | f = 0 | 0.5 | 0.8 | 0.9 | 0.97 | 1.0 |
|---|---|---|---|---|---|---|
| mean (e) | +159 | +88 | +45 | +31 | +21 | +16 |
| sd between molecules (e) | 0 | 6.8 | 5.4 | 4.1 | 2.3 | 0 |
| range over 256 molecules (e) | | +70 ... +104 | +32 ... +66 | +20 ... +42 | +16 ... +28 | |

- At pH 4.2 every sampled molecule is net positive at every f, for all three
  chemistries: with these pKa values pH 4.2 is 0.1-0.2 below the isoelectric
  point of 90-100 % succinylated collagen, not at it. The molecule there is a
  small difference of large numbers: 179 positive charges on Arg and His plus
  the unmodified lysines, against about 160 negative.
- At pH 7.4 the same molecules carry -160 +- 6 (f = 0.9) and -182 (f = 1).
- Fully acetylated collagen keeps +52 at pH 4.2; the 1:1 mixture +34 +- 2.

**Charge pattern at the D period** (`d_lag_correlation.csv`,
`fig_charge_autocorrelation_pH*.png`). The axial charge pattern (all three
chains, one bin per residue rise) is correlated with itself, or with the
pattern of a second, independently modified molecule, at a lag of 234
residues:

| pattern, pH 7.4 | native | succinyl f = 0.5 (two molecules) | succinyl f = 1 | acetyl f = 1 |
|---|---|---|---|---|
| signed charge | -0.002 | -0.004 +- 0.024 | +0.013 | -0.002 |
| charge magnitude | +0.161 | +0.161 | +0.161 | +0.145 |

- The **signed** charge pattern has no D-periodicity to lose: its correlation
  at lag D is indistinguishable from other lags (z = -0.05) in native collagen.
- **Where the charged residues are** is D-periodic: lag 234 is the global
  maximum over lags 30-300, 4.0 standard deviations above the rest (0.129,
  z = 3.3 at pH 4.2). The hydrophobic weights behave the same way (0.117,
  z = 2.9).
- Succinylation swaps signs and keeps charged sites charged, so at pH 7.4 the
  magnitude pattern is untouched at every f. Acetylation removes charged sites
  and lowers it by 10 %; at f >= 0.9 lag 234 is no longer the largest
  correlation (lag 103 is). At pH 4.2, where succinyl groups are only a third
  ionised, the same happens for succinylation at f >= 0.9.
- Random modification adds noise to the signed pattern between molecules
  (+-0.02) around a mean that was already zero.

## S2: two-molecule stagger scan (`results/modcoll/stagger_*/`)

f in {0, 0.25, 0.5, 0.75, 0.9, 1}, pH in {3.5, 4.2, 5, 7.4}, 10-300 mM, 64
independent pattern pairs per point, parallel and antiparallel, for
succinylation, acetylation and the 1:1 mixture, recommended pKa set.
`stagger_succ_phscan/` repeats succinylation at f in {0, 0.8, 0.9, 0.97, 1}
over pH 3.8-4.6, because the net charge at exactly pH 4.2 is a small
difference that moves by 15 e with one pKa. `summary.csv` has one row per
condition, `minima.csv` the six leading minima of three profiles with the
pair-to-pair sd at each. f = 0 and (for a single chemistry) f = 1 have no
pattern randomness, so their spread is zero by construction.

Succinylation, parallel pairs. In each cell: stagger of the deepest minimum
of the pair-averaged profile (in D); each pair's own deepest minimum, mean
and pair-to-pair sd (kT); share of the 64 pairs whose own deepest minimum is
within 3 nm of a D multiple. A deepest minimum at 4.0 or 4.3 D is an end
contact of a pair that otherwise repels, not a register.

| f | pH 4.2, 150 mM | pH 4.2, 50 mM | pH 4.2, 10 mM | pH 7.4, 150 mM |
|---|---|---|---|---|
| 0 | 1.00, -10.8, 100 % | 4.29, -4.8 (end contact) | 4.29, -5.3 (end contact) | 1.01, -25.0, 100 % |
| 0.25 | 1.00, -16.7 (1.3), 100 % | 1.01, -8.9 (1.6), 95 % | 4.02, -7.1 (1.8) (end contact) | 1.01, -26.1 (2.0), 98 % |
| 0.5 | 1.00, -20.9 (1.5), 100 % | 1.01, -18.5 (2.6), 100 % | 4.02, -13.7 (3.2) (end contact) | 1.01, -23.4 (3.1), 97 % |
| 0.75 | 1.01, -23.0 (1.2), 100 % | 1.01, -24.2 (2.2), 95 % | 2.21, -28.9 (3.7), 22 % | 1.01, -17.8 (2.0), 55 % |
| **0.9** | **1.01, -23.7 (0.9), 100 %** | 1.01, -26.4 (1.6), 97 % | 0.58, -37.9 (2.7), 0 % | 1.75, -15.5 (1.3), 9 % |
| 1.0 | 1.01, -23.8, 100 % | 1.01, -27.0, 100 % | 0.58, -44.3, 0 % | 1.75, -13.9, 0 % |

The realistic case, f = 0.9, across pH (net charge of a molecule; deepest
minimum as above):

| f = 0.9 | pH 3.8 (+97 e) | pH 4.0 (+66 e) | pH 4.2 (+31 e) | pH 4.4 (-5 e) | pH 4.6 (-39 e) |
|---|---|---|---|---|---|
| 150 mM | 1.00, -19.3 (0.7), 100 % | 1.01, -21.9 (0.7), 100 % | 1.01, -23.7 (0.9), 100 % | 1.01, -24.1 (1.1), 100 % | 1.01, -23.5 (1.3), 100 % |
| 50 mM | 1.01, -15.1 (1.3), 100 % | 1.01, -21.9 (1.6), 98 % | 1.01, -26.4 (1.6), 97 % | 1.01, -28.4 (1.4), 53 % | 0.58, -30.0 (1.6), 12 % |
| 10 mM | 2.21, -9.4 (2.3), 33 % | 2.21, -23.9 (2.7), 5 % | 0.58, -37.9 (2.7), 0 % | 0.58, -47.8 (3.1), 0 % | 0.58, -50.4 (3.3), 0 % |

What the scan says, in the model's own terms:

1. **At 150 mM or more, the 1 D register survives modification at every
   fraction, for every pattern pair, at every pH from 3.8 to 4.6**, for all
   three chemistries, and it gets deeper as the net charge is removed: -11 kT
   for native at pH 4.2 (+159 e) against -24 kT at f = 0.9 (+31 e), the depth
   native collagen has at pH 7.4. This conclusion does not depend on the sign
   of the net charge at pH 4.2. The profile keeps its 67 nm period (dominant
   period 67.0-67.2 nm in the autocorrelation) up to f = 0.9; at f = 1 the
   correlation at 67 nm (0.31) is overtaken by one at 91 nm (0.38), although
   the three leading minima are still 1.01 D, 0.36 D, 1.99 D.
2. **It survives on the hydrophobic term alone.** The electrostatic
   contribution to D-selectivity (D-comb z of the electrostatic profile) falls
   from +1.4 for native to +0.2 at f = 0.9 and +0.1 at f = 1 at pH 4.2, and is
   negative at pH 5 and above from f = 0.5 on. Modification does not turn
   electrostatics into a new source of D-register; it removes the weak one
   that was there.
3. **At low salt and high modification the preferred register is not a D
   multiple.** At 10 mM the deepest minimum is at 2.21 D (148 nm) or 0.58 D
   (39 nm) for f >= 0.75 at pH 4.2, and for f >= 0.9 at every pH from 3.8 to
   4.6; at pH 4.2 it lies 7 kT
   (f = 0.9) to 13 kT (f = 1) below the 1 D minimum of the same averaged
   profile, and no pair prefers a D multiple at f >= 0.9. These two registers
   are not one period apart and the profile has no D-periodicity there. 50 mM
   is the crossover, and it is pH-sensitive: at f = 0.9, 1 D holds for 97 % of
   pairs at pH 4.2, 53 % at pH 4.4, 12 % at pH 4.6. Acetate buffer adds little
   ionic strength of its own, so which side of this crossover an experiment
   sits on is set by the added salt.
4. **At pH 7.4, where the modified material is soluble, the register is
   lost** as the molecules charge up: 97 % of pairs at 1 D for f = 0.5, 55 %
   at f = 0.75, 9 % at f = 0.9, none at f = 1 (-182 e, weak minimum at 1.75 D).
   Acetylation, which stops at -75 e, keeps 1 D at 150 mM for every f; the
   1:1 mixture keeps it for 70 % of pairs at f = 1.
5. **Partial modification does not blur registry more than complete
   modification.** Mean D-selectivity is lowest at f = 0 or at f = 1 in
   essentially every condition; the largest exception in 96 conditions per
   chemistry is pH 5 and 150 mM, where 88 % of pairs keep 1 D at f = 0.9
   against 100 % at f = 1. What partial modification adds is disagreement
   between pairs: the pair-to-pair sd of the deepest minimum peaks at
   f = 0.5-0.75 (1.5 kT at pH 4.2 and 150 mM, 3.1 kT at pH 7.4, 3.7-5.7 kT at
   10 mM) and is zero at f = 1, where all molecules are identical. Near a
   crossover the sample splits: at pH 4.2, 10 mM and f = 0.75, 22 % of pairs
   prefer a D multiple and the rest do not.
6. Acetylation and the mixture follow the same pattern at pH 4.2: 1 D for
   every pair at 150 mM, off-D registers (2.21 D, 0.58 D) at 10 mM for
   f >= 0.75.

Not run here, cheap to add with the existing options: S2 with the
`crosslink_sites` weighting (`--weights`), with pepsin-cut ends
(`--telopeptides pepsin`), and with the succinyl and Asp pKa varied.

## Composition scan: which mixtures fit, and what holds for all of them (`results/modcoll/composition/`)

What the material is made of is not known. Its lysines may be succinylated,
acetylated or both, and its Asn/Gln may be deamidated. What is known is that
it precipitates near pH 4.2 and is soluble at pH 7. `modcoll/cli_composition.py`
therefore turns the question round: it scans the three fractions together and
asks which mixtures are compatible with those two observations, what would
tell them apart at the bench, and which results of S2 hold for all of them.

A composition is (f_succ, f_acet, f_deam): the mean fractions of amines
carrying a succinyl group, of amines carrying an acetyl group (f_succ +
f_acet <= 1, an amine carries one group) and of Asn/Gln converted to Asp/Glu.
The grid has a spacing of 0.025 in all three, 861 x 41 = 35 301 compositions,
plus the slice f_deam = 0.18 (**liming reference**: liming lowers amide
nitrogen by about 0.6 % of total nitrogen, Beek & Sookne 1939, which is 18 % of
the Asn + Gln of the sequence, and the same fraction brings the native
isoelectric point of this model from 9.8 to 9.0). At every composition 256
molecules are modified site by site by `modify()`, each with its own pattern;
seed 20261112; recommended pKa set, intact chain ends and human sequence unless
a table says otherwise.

### How it is computed (`modcoll/composition.py`)

- **Counts, not a mean field.** With independent-site protonation the net
  charge of a molecule depends on its pattern only through how many sites of
  each type it has: `Q(pH) = sum_t n_t q_t(pH)`, with `q_t` the
  Henderson-Hasselbalch charge of one site of type t. This is an identity (a
  test compares it with `electrostatics.net_charge` molecule by molecule), so
  each sampled molecule is reduced to its vector of type counts, and charges
  and isoelectric points of the nine million sampled molecules are then
  solved by bisection in bulk. Where the modified sites are, as opposed to how many
  there are, matters for pair energies only, and is kept in the stagger stage.
- **Two isoelectric points.** `pI_ensemble` is the pH at which the summed
  charge of the 256 molecules is zero: what a precipitation maximum or a
  mobility zero of the bulk material corresponds to, and the number the
  inverse problem uses. Its standard error is from a bootstrap over molecules
  (200 resamples). `pI_molecule_*` are the mean, the molecule-to-molecule sd,
  the standard error of the mean, and the range of the isoelectric points of
  the single molecules.
- **Standard errors.** Every tabulated mean has one: sd between molecules /
  sqrt(256) for charges, the bootstrap for the sample isoelectric point.
- **Charge per unit length** is the net charge divided by the axial extent of
  the site array, 302.4 nm (helix 290.4 nm plus telopeptides).
- **Nested patterns.** Molecule k uses the same random deviates at every
  composition, so its pattern grows monotonically with each fraction. Maps
  are therefore smooth and region boundaries do not flicker, at the price
  that values at neighbouring compositions are not independent samples.

### Forward map

Files: `forward_map.csv`, `pure_chemistries.csv`, `fig_forward_map.png`.

`forward_map.csv` holds the grid at a spacing of 0.1 plus the liming slice
(792 rows, flagged `liming_reference`); the full grid is regenerated under
`raw/`. A few rows, as succinylated / acetylated / deamidated fractions:

| composition | isoelectric point of the sample | sd between molecules | net charge at pH 4.2, mean (sd) | net charge at pH 7.0 | charge per nm at pH 7.0 |
|---|---|---|---|---|---|
| 0 / 0 / 0, unmodified | 9.84 | 0 | +159.0 | +34.6 | +0.11 |
| 0 / 0 / 0.18, liming only | 9.12 | 0.40 | +147.2 (2.2) | +12.2 | +0.04 |
| 1 / 0 / 0 | 4.288 | 0 | +16.3 | -179.0 | -0.59 |
| 0.9 / 0 / 0 | 4.371 | 0.024 | +30.6 (3.9) | -157.6 | -0.52 |
| 0.9 / 0 / 0.18 | 4.298 | 0.024 | +18.8 (4.4) | -180.0 | -0.60 |
| 0 / 1 / 0 | 4.641 | 0 | +52.0 | -72.3 | -0.24 |
| 0 / 1 / 0.18 | 4.498 | 0.022 | +40.2 (2.2) | -94.7 | -0.31 |
| 0 / 0 / 1 | 4.736 | 0 | +93.4 | -90.2 | -0.30 |
| 0.5 / 0.5 / 0 | 4.420 | 0.016 | +34.1 (1.8) | -125.9 | -0.42 |
| 0.5 / 0.5 / 0.5 | 4.206 | 0.017 | +1.2 (3.3) | -188.5 | -0.62 |
| 0 / 1 / 0.5 | 4.318 | 0.020 | +19.1 (2.8) | -135.0 | -0.45 |
| 0 / 1 / 1 | 4.132 | 0 | -13.6 | -197.1 | -0.65 |
| 1 / 0 / 0.5 | 4.124 | 0.012 | -16.6 (2.8) | -241.6 | -0.80 |
| 1 / 0 / 1, everything modified | 4.001 | 0 | -49.3 | -303.8 | -1.00 |

- The three pure-chemistry axes are the S1 ensembles and reproduce the S1
  table: 9.84, 4.29 at complete succinylation, 4.37 at 90 %, 4.64 at complete
  acetylation, 4.74 at complete deamidation.
- Standard errors are small against everything discussed below: at most
  0.0034 on the isoelectric point of the sample for any composition within
  4.2 +- 0.3, and at most 0.5 e (pH 4.2) and 0.8 e (pH 7.0) on a mean net
  charge anywhere on the grid.
- **The isoelectric point has a floor at 4.00**, reached with every lysine
  succinylated and every amide deamidated. Nothing in this composition space
  goes below it, so a target of 4.2 sits 0.2 above the lowest value the model
  can produce and 0.09 below what complete succinylation alone gives.
- A sample whose isoelectric point is 4.2 is not a sample of neutral
  molecules at pH 4.2. At 0.5 / 0.5 / 0.5 the mean net charge is +1.2 e, the
  sd between molecules 3.3 e, and 61 % of the molecules are net positive.
  Over the 974 grid compositions whose isoelectric point is within 0.01 of
  4.2, the sd between molecules is 2.4 to 6.8 e and 22 to 77 % of the
  molecules are net positive; only compositions without pattern randomness
  (every fraction 0 or 1) have identical molecules.
- At pH 7 every heavily modified composition is strongly negative: -0.24 e/nm
  for complete acetylation, -0.59 e/nm for complete succinylation, -1.0 e/nm
  with full deamidation on top. With the Bjerrum length of 0.71 nm these are
  Manning parameters of 0.17 to 0.71, below the condensation threshold of 1.

### Inverse problem

Files: `region_summary.csv`, `consistent_region.csv`, `min_succinylation.csv`, `fig_consistent_region.png`, `fig_min_succinylation.png`.

Two observations are matched.

1. **Precipitation near pH 4.2.** The isoelectric point of the sample is
   required to be 4.2 within a tolerance band. How a precipitation pH maps
   onto an isoelectric point is itself uncertain, so everything is reported
   for three bands: +-0.1, +-0.2 and +-0.3.
2. **Solubility at pH 7.** The model computes no solubility. As a stand-in
   the molecule is required to carry at pH 7.0 at least twice the net charge
   of unmodified collagen at that pH (2 x 34.6 = 69 e), unmodified collagen
   being what does assemble there (`--soluble-factor`). This turns out not to
   constrain anything: the least charged composition in the widest band still
   carries -95 e (`solubility_threshold.csv`; the threshold would have to
   exceed 95 e to remove a single composition, and at 150 e two thirds of the
   +-0.3 set are still there). **Solubility at pH 7 follows from the
   isoelectric point being near 4.2 and adds no information about the
   composition.**

| | 4.2 +- 0.1 | 4.2 +- 0.2 | 4.2 +- 0.3 |
|---|---|---|---|
| consistent compositions, of 36 162 on the grid | 9 180 (25 %) | 16 108 (45 %) | 21 485 (59 %) |
| succinylation alone | from 98.5 % | from 86.7 % | from 75.7 % |
| acetylation alone | never (stops at 4.64) | never | never |
| deamidation alone | never (stops at 4.74) | never | never |
| least total acylation of any consistent composition | 50 % | 37.5 % | 25 % |
| least succinylation, no deamidation | 95 % | 57.5 % | 30 % |
| least succinylation, deamidation at most the liming level | 65 % | 27.5 % | 0 (with complete acetylation) |
| least succinylation, any deamidation | 0 | 0 | 0 |
| a composition without succinylation needs at least | 70 % acetylation and 55 % deamidation | 52.5 % and 35 % | 35 % and 18 % |
| smallest net charge at pH 7.0 in the set | -141 e | -116 e | -95 e |

Said plainly:

- **The isoelectric point does not identify the composition.** Between a
  quarter and six tenths of all compositions on the grid are consistent with
  it, depending on the band. The consistent set is a thick slab across
  composition space (`fig_consistent_region.png`), not a point or a line.
- **Pure succinylation can reach it**, but for the narrow band only when it is
  essentially complete (98.5 %); the reported 90 % gives 4.37, inside +-0.2.
- **Pure acetylation cannot**, in any band: complete acetylation stops at 4.64.
- **Pure deamidation cannot**, in any band: complete deamidation stops at 4.74.
- **No succinylation is needed at all.** Acetylation of 70 % or more of the
  lysines together with deamidation of 55 % or more of the amides gives 4.2
  +- 0.1 without a single succinyl group. What a solution does need is
  acylation of some kind: at least 50 %, 37.5 % or 25 % of the amines, for the
  three bands.
- How much succinylation is needed depends on how much deamidation is
  allowed (`fig_min_succinylation.png`). If the material is no more
  deamidated than limed hide (0.18), a solution needs at least 65 %
  succinylation for +-0.1 and 27.5 % for +-0.2, the remaining lysines being
  acetylated; for +-0.3 complete acetylation with liming-level deamidation is
  already inside (4.498). With no deamidation at all the figures are 95 %,
  57.5 % and 30 %.
- The lower edge of the band matters only for +-0.1: compositions that are
  both heavily succinylated and heavily deamidated fall below 4.1 (complete
  succinylation with more than about 58 % deamidation).

`consistent_region.csv` lists the consistent compositions on the 0.1 grid and
the liming slice with the narrowest band each falls in; the 0.025 grid is under
`raw/`. Compositions within two standard errors of a band edge are flagged
(6 % of the consistent compositions of the fine grid).

### Sensitivity of the consistent set

Files: `sensitivity.csv`, `sensitivity_statements.csv`, `sensitivity_min_succinylation.csv`, `fig_sensitivity.png`.

The inverse problem is repeated for all 108 combinations of

- the succinyl carboxyl pKa: 4.35, **4.5**, 4.65 (no measurement exists);
- the Asp pKa: 3.65, **3.9**, 4.0 (Thurlkill et al. 2006; Grimsley et al. 2009;
  Nozaki & Tanford 1967);
- the Glu pKa: 4.25, **4.3**, 4.35 (Thurlkill et al. 2006; Platzer et al. 2014);
- the chain ends: **intact**, pepsin-cut (resampled: 105 lysines and three
  free alpha-amines, which are acylated like lysines);
- the sequence: **human**, or human without the four alpha2 histidines
  (helix positions 486, 876, 980, 995) that bovine and porcine alpha2 lack.
  Modification never touches histidine, so this is the same sample with four
  fewer protonated groups per molecule; a test checks that against sampling
  the corrected molecule.

One parameter at a time, band +-0.2 (base case in the first row):

| changed | isoelectric point at complete succinylation | succinylation alone enters +-0.2 at | least succinylation, deamidation at most liming | acetylation alone stops at | deamidation alone stops at | consistent compositions |
|---|---|---|---|---|---|---|
| nothing | 4.288 | 86.7 % | 27.5 % | 4.641 | 4.736 | 16 108 |
| succinyl pKa 4.35 | 4.241 | 81.6 % | 22.5 % | 4.641 | 4.736 | 17 110 |
| succinyl pKa 4.65 | 4.334 | 92.0 % | 32.5 % | 4.641 | 4.736 | 15 010 |
| Asp pKa 3.65 | 4.234 | 81.2 % | 10 % | 4.580 | 4.681 | 18 998 |
| Asp pKa 4.0 | 4.312 | 89.4 % | 37.5 % | 4.670 | 4.764 | 14 388 |
| Glu pKa 4.25 | 4.266 | 83.9 % | 17.5 % | 4.606 | 4.700 | 17 785 |
| Glu pKa 4.35 | 4.310 | 89.3 % | 37.5 % | 4.677 | 4.773 | 14 354 |
| pepsin-cut ends | 4.267 | 84.2 % | 22.5 % | 4.612 | 4.745 | 16 857 |
| bovine / porcine histidines | 4.267 | 84.0 % | 20 % | 4.602 | 4.708 | 17 436 |

Each of these moves the isoelectric point of a heavily modified molecule by
0.02 to 0.05; together they span 4.12 to 4.38 for complete succinylation.
The least succinylation needed at a given deamidation is the sensitive
quantity: it swings by tens of per cent (shaded range in
`fig_min_succinylation.png`).

**Statements that hold in all 108 parameter sets:**

1. Deamidation alone never reaches the isoelectric point, in any band (it
   stops between 4.61 and 4.81).
2. Acetylation alone never reaches 4.2 +- 0.1 or +- 0.2 (it stops between 4.47
   and 4.70).
3. Succinylation alone reaches 4.2 +- 0.2 and +- 0.3 (from 70-98 % and from
   61-86 % succinylation).
4. A consistent composition without any succinylation exists, in every
   band. The minimum succinylation any solution needs is zero.
5. Every consistent composition has a substantial share of its amines
   acylated: at least 32.5 % for +-0.1, 20 % for +-0.2 and 10 % for +-0.3.
6. The solubility criterion at pH 7 never excludes a composition; the least
   charged consistent composition carries 71 e or more.
7. The consistent set is large: 15-32 %, 32-58 % and 48-75 % of the grid for
   the three bands.

**Statements that do not survive:**

- "Succinylation alone reaches 4.2 +- 0.1" holds in 83 of the 108 sets. In the
  other 25 complete succinylation stays above 4.3.
- "Acetylation alone does not reach 4.2 +- 0.3" fails in 3 of the 108 sets:
  Asp 3.65 and Glu 4.25 together with pepsin-cut ends and the bovine/porcine
  histidines, where complete acetylation gives 4.47 (the three differ only in
  the succinyl pKa, which does not enter).
- How much succinylation is needed when deamidation is limited: without
  deamidation, 37.5 % to 97.5 % for +-0.1 (and no solution at all in 25 sets),
  15 % to 92.5 % for +-0.2, 0 to 52.5 % for +-0.3; with deamidation at most the
  liming level, 10 % to 97.5 %, 0 to 57.5 % and 0 to 20 %.

### Which measurement would identify the composition

Files: `assays.csv`, `assays_representatives.csv`, `assay_ranking.csv`, `assay_sets.csv`, `titration_curves.csv`, `fig_assays.png`, `fig_titration.png`.

For every consistent composition (spacing 0.05; 2 206 candidates for +-0.2)
`composition.assay_readouts` computes what a bulk measurement would read.
Each readout is linear in the type counts, or the zero of a linear function
of them, so the bulk value is exact for the sampled molecules:

| readout | definition |
|---|---|
| free amines remaining | unmodified amines / amines of the unmodified molecule (a TNBS-type assay against the unmodified control) |
| amide content | Asn + Gln left, per molecule and in mmol/g |
| carboxyl content | Asp + Glu + chain C-termini + succinyl carboxyls, per molecule and in mmol/g |
| mass shift | +100.016 Da per succinyl, +42.011 Da per acetyl, +0.984 Da per deamidated amide (monoisotopic; the average-mass values are 100.073, 42.037, 0.985), on an unmodified molecule of 288 kDa computed from the sequence |
| net charge against pH | the titration curve; tabulated at pH 3, 5.5, 7, 9, with the zero crossing and the slope dQ/dpH there |
| salt shift of the zero crossing | zero-charge pH at 150 mM minus at 10 mM, with pKa' = pKa -+ A F(I), F(I) = sqrt(I)/(1 + sqrt(I)) - 0.3 I, A = 0.509 (Davies 1962): acids get stronger with salt, bases weaker |

The assays are ranked by the share of pairs of candidates they tell apart: a
measurement made on candidate i excludes candidate j if their predicted
readouts differ by more than two standard deviations of the assay. The
precisions are **assumptions**, listed in `composition.ASSAYS`, and the ranking
is repeated with each halved and doubled. "Left open" is the span of each
fraction among the candidates one measurement would not exclude, averaged
over the candidates; before any measurement it is about 1 for all three.

| rank | assay (assumed 1 sd) | range over the candidates | pairs told apart (sd halved / doubled) | left open: succinyl / acetyl / deamidation |
|---|---|---|---|---|
| 1 | amide content (6 per molecule) | 0 to 125 | 83 % (94 / 63) | 0.98 / 0.98 / 0.10 |
| 2 | mass shift of intact chains (300 Da per molecule) | 2.6 to 10.8 kDa | 83 % (91 / 66) | 0.36 / 0.69 / 0.86 |
| 3 | free amines remaining (0.03) | 0 to 0.60 | 68 % (88 / 51) | 0.88 / 0.88 / 0.90 |
| 4 | carboxyl content (5 %) | 282 to 470 | 46 % (71 / 13) | 0.96 / 0.96 / 0.91 |
| 5 | net charge at pH 7 (10 %) | -304 to -116 e | 44 % (70 / 12) | 0.98 / 0.99 / 0.97 |
| 6 | titration slope at the zero crossing (10 %) | -240 to -144 e/pH | 15 % (48 / 1) | 1.00 / 1.00 / 1.00 |
| 7 | zero-charge pH (0.1) | 4.0 to 4.4 | 14 % (46 / 0) | 1.00 / 1.00 / 1.00 |
| 8 | net charge at pH 3 (10 %) | +149 to +216 e | 8 % (38 / 0) | 1.00 / 1.00 / 1.00 |
| 9 | salt shift of the zero-charge pH (0.05) | -0.074 for every candidate | 0 % | 1.00 / 1.00 / 1.00 |

Four compositions with the same isoelectric point, 4.20, and 90 %
succinylation for comparison (`assays_representatives.csv`):

| succinyl / acetyl / deamidated | free amines | amides per molecule | carboxyls per molecule | mass shift | net charge at pH 7 |
|---|---|---|---|---|---|
| 1.00 / 0 / 0.25 | 0 % | 94 | 376 | +10.7 kDa | -210 e |
| 0 / 0.90 / 0.95 | 10 % | 6 | 357 | +4.2 kDa | -180 e |
| 0 / 1.00 / 0.80 | 0 % | 25 | 338 | +4.6 kDa | -172 e |
| 0.35 / 0.40 / 1.00 | 25 % | 0 | 401 | +5.7 kDa | -208 e |
| 0.90 / 0 / 0 (isoelectric point 4.37) | 10 % | 125 | 334 | +9.6 kDa | -158 e |

- **Three assays do the work, and each fixes a different direction.** The
  amide content gives the deamidated fraction and nothing else. The free
  amines give the total acylation but cannot tell succinyl from acetyl. The
  mass shift tells them apart (100 against 42 Da per group) but confounds a
  little succinyl with a lot of acetyl. The order of these three is the same
  for the three bands and for halved or doubled precisions, except that mass
  and amide content swap first and second place for +-0.1 (81 against 80 %).
- **Together they identify the composition**: free amines + amide content +
  mass shift tell 98.8 % of candidate pairs apart and leave spans of 0.18 in
  succinylation, 0.25 in acetylation and 0.10 in deamidation, that is about
  +-0.09, +-0.13 and +-0.05. The best pairs are amide content + mass shift
  (97 %) and free amines + mass shift (93 %, but blind to deamidation).
  Carboxyl content can stand in for mass (free amines + amide + carboxyl:
  96 %, spans 0.58 / 0.62 / 0.10) at a clear loss.
- **Titration and electrokinetics discriminate poorly.** All candidates cross
  zero within the band by construction and their curves run together from pH
  3.5 to 5 (`fig_titration.png`); they separate only on the alkaline plateau,
  where the net charge at pH 7 is carboxyls minus arginines minus free
  lysines: a single number that mixes all three fractions.
- **Null result: the salt shift of the zero-charge pH is the same for every
  candidate**, -0.074 pH between 10 and 150 mM (to within 0.0003). Near pH 4.2
  only carboxyls titrate, every carboxyl pKa moves by the same -A dF, and the
  bases stay fully protonated, so the crossing moves by that amount whatever
  the composition. In this model the assay cannot discriminate at all.

### Do the S2 conclusions depend on the composition?

Files: `stagger_representatives.csv`, `stagger_summary.csv`, `stagger_conclusions.csv`, `stagger_crossover.csv`, `fig_stagger_robustness.png`, `fig_stagger_crossover.png`.

Eight compositions were picked to span the +-0.3 set (its six extremes, ties
broken by the isoelectric point nearest 4.2, then the two compositions
farthest from all those already chosen), and 90 % succinylation was added as
the case S2 treats as realistic. The S2 two-molecule scan was run on each,
unchanged (`cli_stagger`'s pattern generators, pair descriptors and summary;
a test checks that a pure-chemistry composition gives exactly what
`cli_stagger` writes): pH 3.8, 4.0, 4.2, 4.4, 4.6; 10, 25, 50, 100, 150 mM; 64
pattern pairs; parallel and antiparallel. 25 and 100 mM were added to the
three ionic strengths of S2 to bracket the crossover.

| | succinyl / acetyl / deamidated | chosen as | isoelectric point | net charge at pH 4.2, mean (sd) |
|---|---|---|---|---|
| c0 | 1.00 / 0 / 0.25 | most succinylation | 4.20 | 0 (2.5) |
| c1 | 0 / 0.90 / 0.95 | least succinylation | 4.20 | 0 (3.2) |
| c2 | 0 / 1.00 / 0.80 | most acetylation | 4.20 | 0 (2.4) |
| c3 | 0.35 / 0.40 / 1.00 | most deamidation | 4.20 | 0 (5.4) |
| c4 | 1.00 / 0 / 0 | least deamidation | 4.29 | +16 |
| c5 | 0.25 / 0 / 1.00 | least acylation | 4.48 | +58 (6.0) |
| c6 | 0.35 / 0.65 / 0 | farthest from the others | 4.47 | +39 (1.7) |
| c7 | 1.00 / 0 / 1.00 | farthest from the others | 4.00 | -49 |
| c8 | 0.90 / 0 / 0 | S2 reference | 4.37 | +31 (3.9) |

Row c8 reproduces the committed S2 numbers (pH 4.2: 1.01 D, -23.7 (0.9) kT,
100 % at 150 mM; -26.4 (1.6) kT, 97 % at 50 mM; 0.58 D, -37.9 (2.7) kT, 0 % at
10 mM).

| S2 conclusion | read from | over the nine compositions | verdict |
|---|---|---|---|
| at 150 mM every pattern pair prefers a D multiple, at every pH from 3.8 to 4.6 | share of pairs whose deepest minimum is within 3 nm of a D multiple, worst pH | 100 % at all nine | **holds** |
| that register is 1 D | deepest minimum of the pair-averaged profile, pH 4.2 | 1.01 D at all nine | **holds** |
| its depth is that of native collagen at pH 7.4 | each pair's deepest minimum, pH 4.2, 150 mM | -20.8 to -23.8 kT | **holds** (within 3 kT) |
| pattern pairs agree | pair-to-pair sd of the deepest minimum, pH 4.2 | 0 to 1.3 kT at 150 mM, 0 to 3.1 kT at 10 mM | **holds** |
| parallel is not preferred over antiparallel | antiparallel minus parallel deepest minimum, pH 4.2, 150 mM | -1.8 to -4.3 kT | **holds** (still no polarity selection) |
| at 10 mM the D register is lost | share of pairs at a D multiple, 10 mM | at each composition's own isoelectric point: 0 to 6 % at all nine. At pH 4.2: 0 to 6 % for eight, 25 % for c5 | **holds at the isoelectric point**; at a fixed pH it depends on how far that pH is from it |
| ... at every pH from 3.8 to 4.6 | the same, worst pH | 0 % for c0, c4, c6, c7; 20 to 92 % for c1, c2, c3, c5; 33 % for c8 | **depends on composition** |
| which register replaces it | deepest minimum of the pair-averaged profile, 10 mM, pH 4.2 | 0.58 D (c0, c2, c4, c7, c8), 0.61-0.62 D (c1, c3), 1.99 D (c5), 2.21 D (c6) | **depends on composition** |
| 50 mM is the crossover | ionic strength at which half the pairs prefer a D multiple | at each composition's own isoelectric point: 33 to 48 mM. At pH 4.2: 19 to 71 mM; at pH 4.6: 38 to 122 mM | **holds at the isoelectric point** (factor 1.5); at a fixed pH it depends on composition (factor 3.6) |
| the profile repeats with the D period | dominant period of the profile, pH 4.2, 150 mM | 67.0-67.2 nm for six; 91 nm for the three with complete succinylation (c0, c4, c7) | **depends**: lost only at complete succinylation, as in S2 |
| electrostatics does not select the D register | D-comb z-score of the electrostatic term, pH 4.2, 150 mM | 0.10 to 0.72 for c0, c4, c6, c7, c8; 1.15 to 1.21 for c1, c2, c3, c5 | **depends on composition** |

What this says:

1. **The headline of S2 does not depend on the composition.** At 150 mM every
   one of the nine compositions has its deepest minimum at 1 D for every
   pattern pair at every pH from 3.8 to 4.6, 21 to 24 kT deep, with pairs
   agreeing within about 1 kT. It carries the caveat it had in S2: this
   register comes from the hydrophobic term, which modification does not
   change in this model.
2. **At its own isoelectric point every composition behaves alike in salt.**
   The D register is lost at 10 mM (at most 6 % of pairs keep a D multiple)
   and comes back between 33 and 48 mM, whether the charge was removed by
   succinylation, by acetylation with deamidation, or by a mixture. If the
   precipitation pH is the isoelectric point, the statement "below about 30 mM
   the pair register is not a D multiple, above about 50 mM it is 1 D" holds
   without knowing the composition. For the two compositions without pattern
   randomness (c4, c7) the crossover is resolved only to the interval between
   two scanned ionic strengths.
3. **Away from the isoelectric point the chemistry shows.** On the acid side
   (pH 3.8-4.0, molecules at +40 to +80 e) compositions built on acetylation
   and deamidation keep a D multiple, 2 D, down to 10 mM in 73-75 % of pairs
   (c1, c3), where succinyl-rich ones have lost it entirely. On the alkaline
   side, at a like net charge, succinyl-rich compositions need about twice
   the salt: at pH 4.6, c0 (-78 e) needs 75 mM where c1 and c2 (-72, -69 e)
   need 42 and 38 mM (`fig_stagger_crossover.png`).
4. **Which off-D register appears at low salt depends on the composition**:
   0.58 D for succinyl-rich and fully acetylated compositions at their
   isoelectric point, 0.61-0.62 D with heavy deamidation, 1.99 D or 2.21 D
   where the molecules are still net positive at pH 4.2. S2's "0.58 D or
   2.21 D" is the succinylation case.
5. **"Modification removes the weak electrostatic D-selectivity" is a
   statement about succinylation.** In the four compositions with heavy
   deamidation and incomplete succinylation the electrostatic term alone
   keeps a D-comb z-score of about 1.2, near the 1.4 of unmodified collagen;
   it is 0.1 to 0.7 in the others. It is too weak in either case to set the
   register against the hydrophobic term.
6. The loss of the 67 nm period of the whole profile is specific to
   complete succinylation and returns at 90 %.

### What the scan cannot determine

- **The composition.** The isoelectric point is one number and the
  composition has three; the consistent set is a slab that fills a quarter to
  more than half of composition space. Nothing in the two observations (where
  it precipitates, that it dissolves at pH 7) can narrow it further in this
  model. The assays above can.
- **Whether precipitation pH equals the isoelectric point.** The bands stand
  in for that. Measured isoelectric points of succinylated collagens span 3.5
  to 4.8 depending on method and electrolyte (Zhang et al. 2007; Miyata et al.
  1979, US 4,164,559; Miyata & Dunn 1988, US 4,748,152); a band of +-0.3 does
  not cover that whole span.
- **Solubility.** "Soluble at pH 7" is a threshold on the net charge. The model
  computes no solubility, and the threshold happens not to matter: no
  composition in any band comes near it.
- **Ion binding.** The zero-charge pH here is the zero of the proton charge.
  An electrokinetic isoelectric point also moves with bound or accumulated
  ions, by two pH units for native collagen between 0.1 and 10 mM KCl
  (Freudenberg et al. 2007). The salt shift computed here (-0.07 pH from 10 to
  150 mM, the same for every composition) is the activity-coefficient effect
  only; that it does not discriminate says the model has no
  composition-dependent salt effect, not that an experiment would show none.
- **Charge regulation and site coupling.** Every site titrates on its own
  with a model-compound pKa. The pKa ranges of the sensitivity scan are the
  nearest thing to an error bar on that.
- **Chemistry outside the three fractions.** O-acylation of tyrosine and
  other hydroxyls (a few per cent of the amide content in gelatin, Zhu et al.
  2019) would add carboxyls that no fraction here represents. Telopeptide
  lysines that are aldehydes or cross-linked cannot be acylated; here all 107
  lysines are amines. Reactivity is uniform over sites (`--weights` changes
  that; S1 found the isoelectric point moves by at most 0.02 with the
  cross-link-site weighting).
- **Assay precision.** The ranking rests on assumed precisions. It is stable
  when each is halved or doubled, but it is a ranking of predicted readouts,
  not of instruments. Trinitrobenzenesulfonate on a native helix counts only
  about half the amines the sequence has (0.184 mmol/g found, Zhang et al.
  2007, against 0.37 mmol/g), so the free-amine readout is meaningful relative
  to the unmodified control, not as an absolute content. Intact-chain mass
  cannot see deamidation: 125 amides are 123 Da on 288 kDa.
- **More than nine compositions.** The stagger scan was run on eight
  compositions chosen to span the +-0.3 set and on 90 % succinylation.
  "Holds across the region" means at those nine, at pH 3.8-4.6 and 10-150 mM.
  Everything S2 says about its own limits (the hydrophobic strength, fixed
  separation, Debye-Hueckel) applies unchanged.

## Known limits

Read the results above through these.

- **The headline rests on `eps_h`.** The register that survives is the
  hydrophobic one, its strength is a one-number assumption, and modification
  leaves hydrophobic weights unchanged by construction. Succinylation adds 107
  four-carbon acyl groups with a charged end, acetylation 107 methyl amides;
  neither is in the Kyte-Doolittle scale, and the centred-product form
  (inherited from the existing tools) also makes two matched polar patches
  attract as much as two matched apolar ones. "The 1 D register survives" is
  therefore closer to "nothing in this model can remove it at 150 mM" than to a
  prediction. The electrostatic statements (points 2-4) do not depend on
  `eps_h`; where the crossover in point 3 falls does.
- **Debye-Hueckel with bulk water everywhere.** Linearised, uniform
  dielectric, point charges with a soft core. Contact ion pairs, dielectric
  exclusion by the protein and ion correlations are outside it. It is at its
  worst exactly where the interesting numbers are: 10 mM and tens of
  elementary charges on each molecule.
- **Independent-site protonation.** No charge regulation: a carboxylate next
  to a neighbour's carboxylate at pH 4.2 would protonate, which this model
  cannot do. Near pH 4.2 the molecule's charge changes by about 180 e per pH
  unit, so a local potential of 1 kT/e shifts protonation appreciably and
  fixed mean charges overestimate both attraction and repulsion at contact.
  The interface for replacing the charge model exists.
- **The net charge at pH 4.2 is not robust.** It moves by 15 e with the Asp
  pKa, 9 e with the succinyl pKa and 7 e with 5 % in the degree of
  succinylation. Statements here that hold across pH 3.8-4.6 are safer than
  ones tied to pH 4.2 exactly.
- **Rigid, straight, one side-chain conformation.** Side chains are ideal
  conformers placed once; the persistence length is 60 nm, not infinity; two
  molecules, not a lattice with five neighbours and a 0.54 D gap.
- **Fixed separation.** One axis distance (1.65 nm) for every chemistry. A
  succinylated surface is bulkier and more hydrated; the equilibrium distance
  is not computed.
- **Energies, not free energies of binding.** Minimum and Boltzmann average
  over two facing angles are the only entropy in it. A -24 kT pair minimum is
  not a statement that pairs form at a given concentration.
- **No polarity selection** (see the native control).
- **Zero stagger is excluded.** Like-on-like register is strongly favoured by
  the centred hydrophobic product and strongly disfavoured by charge; the
  5 nm exclusion follows the existing tools and hides whichever wins.
- **The native isoelectric point is 9.8, not 9.**
- **Nested patterns across f** correlate neighbouring f values.
- **Human sequences, intact chain ends, every lysine an amine.** Collagen of
  another species differs in some charged residues and needs its own site
  array. Telopeptide lysines that are oxidised to aldehydes or cross-linked in
  tissue collagen (Yamauchi & Sricholpech 2012) cannot be acylated; here all
  five are amines. The pepsin cut sites are bovine and the fate of the alpha2
  C-telopeptide is unknown.

## Files

```
modcoll/sites.py           site array, geometry, telopeptides
modcoll/modify.py          quenched random modification, reactivity weights
modcoll/electrostatics.py  pKa, Henderson-Hasselbalch, Debye-Hueckel, direct pair sum
modcoll/stagger.py         FFT stagger scan, angle reductions, profile analysis, native gate
modcoll/composition.py     type counts, vectorised charge and isoelectric point, pKa
                           variants, salt dependence of pKa, masses, assay readouts,
                           assay ranking, choice of representative compositions
modcoll/cli_ensemble.py    S1
modcoll/cli_stagger.py     S2
modcoll/cli_composition.py composition scan: forward map, inverse problem,
                           sensitivity, assays, stagger scan across compositions
modcoll/composition_figures.py  its figures
modcoll/cli_common.py      CSV/JSON writers, figure style
tests/modcoll/             pytest; oracles are closed forms, published numbers,
                           hand-built molecules and the brute-force pair sum
results/modcoll/           committed summaries and figures
```

## References

- Hulmes, Miller, Parry, Piez, Woodhead-Galloway (1973) J. Mol. Biol. 79:137
- Kyte, Doolittle (1982) J. Mol. Biol. 157:105
- Thurlkill, Grimsley, Scholtz, Pace (2006) Protein Sci. 15:1214
- Grimsley, Scholtz, Pace (2009) Protein Sci. 18:247
- Platzer, Okon, McIntosh (2014) J. Biomol. NMR 60:109
- Fitch, Platzer, Okon, Garcia-Moreno, McIntosh (2015) Protein Sci. 24:752
- Persikov, Ramshaw, Kirkpatrick, Brodsky (2005) Biochemistry 44:1414
- Gonzalez, Hamabata, Rojkind (1984) Coll. Relat. Res. 4:339
- Hudson, Archer, King, Eyre (2018) J. Biol. Chem. 293:15620
- Taga, Tanaka, Hamada, Kusubata, Ogawa-Goto, Hattori (2017) Cell Chem. Biol. 24:1276
- Yamauchi, Sricholpech (2012) Essays Biochem. 52:113
- Qian et al. (2017) Biosci. Biotechnol. Biochem. 81:514
- Galea, Dalrymple, Kuypers, Blakeley (2000) Protein Sci. 9:1947
- Zhang, Liu, Li, Li (2007) Biosci. Biotechnol. Biochem. 71:2057
- Xu, Guan, Liu, Fan, Chen (2017) J. Appl. Polym. Sci. 134:45424
- Israelachvili, Intermolecular and Surface Forces, 3rd ed. (2011), ch. 14
- Berisio, Vitagliano, Mazzarella, Zagari (2002) Protein Sci. 11:262 (PDB 1K6F)
- Beek, Sookne (1939) J. Res. Natl. Bur. Stand. 23:271
- Nozaki, Tanford (1967) Methods Enzymol. 11:715
- Davies (1962) Ion Association, Butterworths, London
- Robinson, Stokes (1959) Electrolyte Solutions, 2nd ed., Butterworths, London
- Gasteiger et al. (2005) in The Proteomics Protocols Handbook, Humana Press, pp. 571-607
- Freudenberg et al. (2007) Biophys. J. 92:2108
- Miyata, Rubin, Stenzel, Dunn (1979) US Patent 4,164,559
- Miyata, Dunn (1988) US Patent 4,748,152
- Zhu, Wang, Ferracci, Zheng, Cho, Lee (2019) Sci. Rep. 9:6863
