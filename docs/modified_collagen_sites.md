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
.venv/bin/python -m pytest tests/modcoll              # 129 tests, about 30 s
.venv/bin/python -m modcoll.cli_ensemble              # S1, about 2 min
.venv/bin/python -m modcoll.cli_stagger --chemistry succ    # S2, about 3 min on 12 threads
.venv/bin/python -m modcoll.cli_stagger --chemistry acet
.venv/bin/python -m modcoll.cli_stagger --chemistry mixed
.venv/bin/python -m modcoll.cli_stagger --chemistry succ --f 0,0.8,0.9,0.97,1 \
    --ph 3.8,4.0,4.2,4.4,4.6 --outdir results/modcoll/stagger_succ_phscan
```

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
  succinylation (Xu et al. 2017).
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
modcoll/cli_ensemble.py    S1
modcoll/cli_stagger.py     S2
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
