"""Composition space of modified collagen: which mixtures of succinylation,
acetylation and deamidation are compatible with an observed isoelectric
point, and what a bench assay would read for each of them.

A composition is the triple (f_succ, f_acet, f_deam): the mean fractions of
amines carrying a succinyl group, of amines carrying an acetyl group
(f_succ + f_acet <= 1, an amine carries one group) and of Asn/Gln converted to
Asp/Glu. Nothing here is a mean-field calculation: every number is a
statistic over concrete molecules, each modified site by site by
``modify.modify`` with its own pattern.

What makes a scan over thousands of compositions affordable is that, with
independent-site protonation, the net charge of a molecule depends on its
pattern only through *how many sites of each type it has*:

    Q(pH) = sum over site types t of  n_t * q_t(pH)

with q_t the Henderson-Hasselbalch charge of one site of type t. That is an
identity, not an approximation (the tests compare it with
``electrostatics.net_charge`` molecule by molecule), so a sampled molecule
is reduced to its vector of type counts and charges and isoelectric points are
then computed for many molecules at once. Which sites are modified, as opposed
to how many, matters for the pair energies of the stagger scan and is kept
there.

Units: charges in elementary charges, lengths nm, masses Da (g/mol),
contents mmol per gram of protein, ionic strength mol/L.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import electrostatics as es
from .modify import modify
from .sites import Molecule, SiteType

N_TYPES = max(int(t) for t in SiteType) + 1

_T = {t.name: int(t) for t in SiteType}
AMINE_TYPES = (_T["LYS"], _T["NTERM"])
SUCC_TYPES = (_T["SUCC_LYS"], _T["SUCC_NTERM"])
ACET_TYPES = (_T["ACET_LYS"], _T["ACET_NTERM"])
AMIDE_TYPES = (_T["ASN"], _T["GLN"])
CARBOXYL_TYPES = (_T["ASP"], _T["GLU"], _T["CTERM"]) + SUCC_TYPES

# Deamidation extent of limed hide, the reference marked in every table and
# figure: liming lowers amide nitrogen by about 0.6 % of total nitrogen (Beek
# & Sookne 1939, J. Res. Natl. Bur. Stand. 23:271), which is about 18 % of the
# Asn + Gln of the sequence. In this model the same fraction is also what
# brings the native isoelectric point from 9.8 to 9.0.
LIMING_F_DEAM = 0.18


# ------------------------------------------------------------ type counts ---
def type_counts(molecule: Molecule) -> np.ndarray:
    """Number of sites of each SiteType in one molecule (index = int(type))."""
    return np.bincount(molecule.site_type, minlength=N_TYPES).astype(np.int64)


def _valence_and_pka(pka):
    """(z, pKa) arrays over site types; z = 0 for a type that does not titrate
    in the table. ``pka`` is a table {SiteType: pKa} or the name of one of
    ``electrostatics.PKA_SETS``."""
    table = es.PKA_SETS[pka] if isinstance(pka, str) else pka
    z = np.zeros(N_TYPES)
    pk = np.zeros(N_TYPES)
    for site_type, value in table.items():
        z[int(site_type)] = es.VALENCE[site_type]
        pk[int(site_type)] = value
    return z, pk


def type_charges(pH, pka=es.DEFAULT_PKA_SET) -> np.ndarray:
    """Henderson-Hasselbalch charge of one site of each type at ``pH``
    (scalar or array; the type axis is appended last):

        acid:  q = -1 / (1 + 10^(pKa - pH))      base:  q = +1 / (1 + 10^(pH - pKa))

    the per-type form of ``electrostatics.henderson_hasselbalch``."""
    z, pk = _valence_and_pka(pka)
    x = z * (np.asarray(pH, dtype=float)[..., None] - pk)
    return z / (1.0 + 10.0 ** np.clip(x, -300.0, 300.0))


def net_charge_from_counts(counts, pH, pka=es.DEFAULT_PKA_SET) -> np.ndarray:
    """Net charge (e) of molecules given as type-count vectors [..., N_TYPES]
    at ``pH`` (scalar, or an array broadcasting against the leading axes)."""
    return np.sum(np.asarray(counts, dtype=float) * type_charges(pH, pka), axis=-1)


def charge_slope_from_counts(counts, pH, pka=es.DEFAULT_PKA_SET) -> np.ndarray:
    """dQ/dpH (e per pH unit) of count vectors: the negative of the buffer
    capacity a titration measures. For one Henderson-Hasselbalch site with
    ionised fraction theta, d(theta)/d(pH) = +-ln(10) theta (1 - theta), and
    ionisation lowers the charge of an acid and of a base alike, so

        dQ/dpH = -ln(10) * sum_t n_t theta_t (1 - theta_t).
    """
    theta = np.abs(type_charges(pH, pka))
    return -np.log(10.0) * np.sum(np.asarray(counts, dtype=float) * theta * (1.0 - theta),
                                  axis=-1)


def isoelectric_point_from_counts(counts, pka=es.DEFAULT_PKA_SET, lo: float = 0.0,
                                  hi: float = 14.0, iterations: int = 50) -> np.ndarray:
    """pH of zero net charge for every count vector of ``counts``
    [..., N_TYPES], by bisection on all rows at once. The net charge of any
    set of acids and bases falls monotonically with pH, so the root is
    unique; NaN where there is none in [lo, hi]. A row may be one molecule or
    the summed counts of a sample (the isoelectric point of the sample)."""
    counts = np.asarray(counts, dtype=float)
    shape = counts.shape[:-1]
    c = counts.reshape(-1, N_TYPES)
    z, pk = _valence_and_pka(pka)
    use = np.flatnonzero((z != 0.0) & c.any(axis=0))
    c, z, pk = c[:, use], z[use], pk[use]
    out = np.empty(len(c))
    for start in range(0, len(c), 65536):                 # bound the temporaries
        block = c[start:start + 65536]

        def charge(pH):
            return np.einsum("ij,ij->i", block, z / (1.0 + 10.0 ** (z * (pH[:, None] - pk))))

        a = np.full(len(block), lo)
        b = np.full(len(block), hi)
        ok = (charge(a) > 0.0) & (charge(b) < 0.0)
        for _ in range(iterations):
            mid = 0.5 * (a + b)
            positive = charge(mid) > 0.0
            a = np.where(positive, mid, a)
            b = np.where(positive, b, mid)
        out[start:start + 65536] = np.where(ok, 0.5 * (a + b), np.nan)
    return out.reshape(shape)


# ---------------------------------------------------------------- pKa sets ---
# Ranges scanned by the sensitivity analysis.
#   succinyl carboxylate 4.35 - 4.65: the working value 4.5 +- 0.15 (no
#       measurement exists; see electrostatics.PKA_SETS).
#   Asp 3.65 - 4.0: from the 3.67 of Thurlkill et al. 2006 (Protein Sci.
#       15:1214), the value the repository's own tools use, to the 4.0 of
#       Nozaki & Tanford 1967 (Methods Enzymol. 11:715); 3.9 is the revised
#       model-peptide value (Grimsley, Scholtz & Pace 2009, Protein Sci. 18:247).
#   Glu 4.25 - 4.35: 4.25 +- 0.05 (Thurlkill et al. 2006) and 4.34 +- 0.04
#       (Platzer, Okon & McIntosh 2014, J. Biomol. NMR 60:109).
PKA_RANGES = {"succinyl": (4.35, 4.5, 4.65), "asp": (3.65, 3.9, 4.0),
              "glu": (4.25, 4.3, 4.35)}


def pka_variant(base=es.DEFAULT_PKA_SET, succinyl: float | None = None,
                asp: float | None = None, glu: float | None = None) -> dict:
    """Copy of a pKa table with the succinyl carboxylate (both succinylated
    amine types), Asp and/or Glu replaced."""
    table = dict(es.PKA_SETS[base] if isinstance(base, str) else base)
    if succinyl is not None:
        table[SiteType.SUCC_LYS] = table[SiteType.SUCC_NTERM] = float(succinyl)
    if asp is not None:
        table[SiteType.ASP] = float(asp)
    if glu is not None:
        table[SiteType.GLU] = float(glu)
    return table


# Debye-Hueckel limiting slope for log10 of an activity coefficient,
# (mol/L)^-1/2: A = 1.8246e6 / (eps_r T)^(3/2) = 0.509 in water at 298.15 K
# (eps_r = 78.54; Robinson & Stokes 1959, Electrolyte Solutions, 2nd ed.,
# Butterworths).
DEBYE_HUECKEL_A = 0.509

# Ionic strength the "recommended" pKa set refers to, mol/L: the model
# peptides were titrated in 0.1 M KCl (Thurlkill et al. 2006).
PKA_REFERENCE_IONIC_M = 0.1


def davies_term(ionic_strength_M) -> np.ndarray:
    """A * (sqrt(I) / (1 + sqrt(I)) - 0.3 I): minus log10 of the activity
    coefficient of a monovalent ion (Davies 1962, Ion Association,
    Butterworths), usable to about I = 0.5 mol/L."""
    i = np.asarray(ionic_strength_M, dtype=float)
    root = np.sqrt(i)
    return DEBYE_HUECKEL_A * (root / (1.0 + root) - 0.3 * i)


def ionic_strength_pka(pka, ionic_strength_M: float,
                       reference_M: float = PKA_REFERENCE_IONIC_M) -> dict:
    """pKa table moved from the ionic strength it was measured at to another.

    With pH on the activity scale (what a pH meter reads) and concentrations
    for the protein groups, the apparent constant of an acid HA = H+ + A-
    carries the activity coefficient of A-, that of a base BH+ = B + H+ the
    coefficient of BH+:

        acid:  pKa' = pKa0 - A F(I)        base:  pKa' = pKa0 + A F(I)

    with A F(I) = :func:`davies_term`. Salt therefore makes acids stronger and
    bases weaker. This is the salt dependence of an isolated monovalent group;
    the coupling between the sites of one molecule, and any specific binding
    of ions, are not in it."""
    table = dict(es.PKA_SETS[pka] if isinstance(pka, str) else pka)
    shift = float(davies_term(ionic_strength_M) - davies_term(reference_M))
    return {t: v + es.VALENCE[t] * shift for t, v in table.items()}


# ------------------------------------------------------------------ masses ---
# Average isotopic masses of amino-acid residues, Da (residue = amino acid
# minus water), the values of the ExPASy Compute pI/Mw and ProtParam tools
# (Gasteiger et al. 2005, in The Proteomics Protocols Handbook, Humana Press,
# pp. 571-607). 'O' is 4-hydroxyproline = Pro + O (15.9994).
RESIDUE_MASS_DA = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594, "N": 114.1038,
    "D": 115.0886, "Q": 128.1307, "K": 128.1741, "E": 129.1155, "M": 131.1926,
    "H": 137.1411, "F": 147.1766, "R": 156.1875, "Y": 163.1760, "W": 186.2132,
    "O": 113.1161,
}
WATER_MASS_DA = 18.0153
AMMONIA_MASS_DA = 17.0305     # lost when an N-terminal Gln cyclises to pyroglutamate

# Mass added per modified site, Da, monoisotopic (Unimod: Succinyl C4H4O3
# 100.016044, Acetyl C2H2O 42.010565, Deamidated H-1N-1O 0.984016). The
# average-mass values, which is what an intact-chain measurement sees, are
# 100.073, 42.037 and 0.985: larger by at most 0.06 %.
MASS_SHIFT_DA = {"succinyl": 100.016, "acetyl": 42.011, "deamidation": 0.984}


def molecular_mass_da(molecule: Molecule) -> float:
    """Average mass of the unmodified molecule from its sequence, Da: residue
    masses, one water per chain, minus ammonia for each pyroglutamate
    N-terminus. Hydroxylysine and its sugars are not in the site model, so
    the result (288 kDa) is a few kDa below a real molecule."""
    res = molecule.is_residue
    mass = sum(RESIDUE_MASS_DA[a] for a in molecule.aa[res])
    n_chains = len(np.unique(molecule.chain))
    # a blocked N-terminal Gln is the only Gln whose native type is OTHER
    pyroglu = np.count_nonzero(res & (molecule.aa == "Q")
                               & (molecule.native_type == int(SiteType.OTHER)))
    return float(mass + n_chains * WATER_MASS_DA - pyroglu * AMMONIA_MASS_DA)


def modification_counts(counts, native_counts) -> dict:
    """Numbers of succinyl groups, acetyl groups, deamidated amides and
    remaining free amines per molecule, from type counts [..., N_TYPES]."""
    c = np.asarray(counts, dtype=float)
    n = np.asarray(native_counts, dtype=float)
    return {"succinyl": c[..., SUCC_TYPES].sum(-1), "acetyl": c[..., ACET_TYPES].sum(-1),
            "deamidated": n[..., AMIDE_TYPES].sum(-1) - c[..., AMIDE_TYPES].sum(-1),
            "free_amines": c[..., AMINE_TYPES].sum(-1)}


# ---------------------------------------------------------------- sequence ---
# Histidines of the human alpha2(I) helix (1-based helix positions) that are
# Pro, Gln, Gln and Ser in bovine (P02465) and porcine alpha2(I). They are the
# only residues of the helix at which human differs from bovine or pig in a
# group that carries charge: Lys, Arg, Asp and Glu sit at the same helix
# positions in all three (alignment of UniProt P02452/P08123 with
# P02453/P02465).
BOVINE_ABSENT_HIS_ALPHA2 = (486, 876, 980, 995)


def without_alpha2_histidines(molecule: Molecule,
                              positions=BOVINE_ABSENT_HIS_ALPHA2) -> Molecule:
    """Copy of a human molecule with the four alpha2 histidines that bovine
    and porcine collagen lack turned into non-titrating sites. Positions and
    hydrophobic weights are left as they are: this corrects the charge
    inventory, it does not build the other species."""
    hit = ((molecule.chain == 2) & molecule.is_residue & ~molecule.smeared
           & np.isin(molecule.res_index, [p - 1 for p in positions]))
    if np.count_nonzero(hit) != len(positions) or np.any(
            molecule.native_type[hit] != int(SiteType.HIS)):
        raise ValueError("the alpha2 positions are not histidines in this molecule")
    site_type = molecule.site_type.copy()
    native_type = molecule.native_type.copy()
    site_type[hit] = native_type[hit] = int(SiteType.OTHER)
    return dataclasses.replace(molecule, site_type=site_type, native_type=native_type)


# -------------------------------------------------------------------- grid ---
def composition_grid(step: float, extra_f_deam=()) -> np.ndarray:
    """All (f_succ, f_acet, f_deam) on a regular grid of spacing ``step`` with
    f_succ + f_acet <= 1 and f_deam in [0, 1], plus the same amine grid at
    each ``extra_f_deam``. Values are multiples of the step rounded to six
    decimals, so they compare exactly. Sorted by (f_deam, f_succ, f_acet)."""
    n = int(round(1.0 / step))
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError("step must divide 1")
    levels = np.round(np.arange(n + 1) * step, 6)
    deam = sorted(set(levels.tolist()) | {round(float(x), 6) for x in extra_f_deam})
    rows = [(levels[i], levels[j], d) for d in deam
            for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(rows)


def on_grid(values, step: float) -> np.ndarray:
    """True where every value of a row is a multiple of ``step``."""
    v = np.asarray(values, dtype=float) / step
    return np.all(np.abs(v - np.round(v)) < 1e-6, axis=-1)


# ---------------------------------------------------------------- sampling ---
def molecule_rng(seed: int, k: int) -> np.random.Generator:
    """Generator of sampled molecule ``k``. It does not depend on the
    composition, so molecule k has nested patterns across the whole grid (see
    modify.modify): differences between neighbouring compositions carry less
    sampling noise than each value does, and the values at different
    compositions are not independent samples."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(k,)))


def sample_counts(native: Molecule, f_succ: float, f_acet: float, f_deam: float,
                  n_molecules: int, seed: int, weights=None) -> np.ndarray:
    """Type counts [n_molecules, N_TYPES] of independently modified molecules
    at one composition, each with its own site-by-site pattern."""
    out = np.empty((n_molecules, N_TYPES), dtype=np.int64)
    for k in range(n_molecules):
        out[k] = type_counts(modify(native, molecule_rng(seed, k), f_succ=f_succ,
                                    f_acet=f_acet, f_deam=f_deam, weights=weights))
    return out


def _sd(x) -> float:
    """Sample standard deviation, exactly 0 for identical values."""
    x = np.asarray(x, dtype=float)
    if x.size < 2 or np.ptp(x) == 0.0:
        return 0.0
    return float(np.std(x, ddof=1))


def bootstrap_rng(seed: int, f_succ: float, f_acet: float, f_deam: float):
    """Generator of the bootstrap at one composition, keyed by the composition
    itself (in units of 1e-6) so that it does not depend on the grid."""
    key = tuple(int(round(1e6 * f)) for f in (f_succ, f_acet, f_deam))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xB007,) + key))


def point_statistics(counts, pka=es.DEFAULT_PKA_SET, pH_values=(4.2, 7.0),
                     length_nm: float = 1.0, rng: np.random.Generator | None = None,
                     n_boot: int = 200) -> dict:
    """Statistics of one sample of molecules (type counts [n, N_TYPES]).

    For the isoelectric point two things are reported. ``pI_ensemble`` is the
    pH at which the summed charge of the sample is zero, what a precipitation
    maximum or a mobility zero of the bulk material corresponds to; its
    standard error comes from a bootstrap over molecules (``rng``; omitted if
    no generator is given). ``pI_molecule_*`` describe the isoelectric points
    of the individual molecules: mean, molecule-to-molecule sd, the standard
    error of that mean (sd / sqrt(n)), minimum and maximum.

    For each pH: mean, sd between molecules and standard error of the net
    charge, the same divided by ``length_nm`` (linear charge density, e/nm),
    and the share of molecules that are net positive."""
    counts = np.asarray(counts)
    n = len(counts)
    pi_mol = isoelectric_point_from_counts(counts, pka)
    out = {"n_molecules": n,
           "pI_ensemble": float(isoelectric_point_from_counts(counts.sum(axis=0), pka)),
           "pI_molecule_mean": float(pi_mol.mean()), "pI_molecule_sd": _sd(pi_mol),
           "pI_molecule_se": _sd(pi_mol) / np.sqrt(n),
           "pI_molecule_min": float(pi_mol.min()), "pI_molecule_max": float(pi_mol.max())}
    if rng is not None and n_boot > 0:
        if np.ptp(counts, axis=0).max() == 0:
            out["pI_ensemble_se"] = 0.0                    # identical molecules
        else:
            pick = rng.integers(0, n, size=(n_boot, n))
            out["pI_ensemble_se"] = _sd(isoelectric_point_from_counts(
                counts[pick].sum(axis=1), pka))
    for pH in pH_values:
        q = net_charge_from_counts(counts, pH, pka)
        tag = f"pH{pH:g}"
        sd = _sd(q)
        out.update({f"net_charge_{tag}_mean": float(q.mean()), f"net_charge_{tag}_sd": sd,
                    f"net_charge_{tag}_se": sd / np.sqrt(n),
                    f"charge_per_nm_{tag}_mean": float(q.mean()) / length_nm,
                    f"charge_per_nm_{tag}_se": sd / np.sqrt(n) / length_nm,
                    f"fraction_net_positive_{tag}": float(np.mean(q > 0.0))})
    return out


def contour_length_nm(molecule: Molecule) -> float:
    """Axial extent of the site array, nm: the helix plus the telopeptides at
    0.28 nm per residue. The length net charges are divided by."""
    return float(molecule.s.max() - molecule.s.min())


# ----------------------------------------------------------------- inverse ---
def in_band(pI, target: float, half_width: float) -> np.ndarray:
    """True where the isoelectric point lies within target +- half_width
    (closed interval, with a 1e-9 allowance for round-off)."""
    return np.abs(np.asarray(pI, dtype=float) - target) <= half_width + 1e-9


def soluble_at(net_charge, threshold_e: float) -> np.ndarray:
    """Charge proxy for "soluble": the molecule carries a net charge of at
    least ``threshold_e`` in magnitude. The model has no solubility
    calculation; see the model document for what this stands in for."""
    return np.abs(np.asarray(net_charge, dtype=float)) >= threshold_e


def pure_chemistry(points) -> np.ndarray:
    """Label of each composition row: "succinylation", "acetylation" or
    "deamidation" if only that fraction is non-zero, "native" if none is,
    otherwise ""."""
    p = np.asarray(points, dtype=float)
    nz = p > 0.0
    labels = np.full(len(p), "", dtype=object)
    labels[~nz.any(axis=1)] = "native"
    for col, name in enumerate(("succinylation", "acetylation", "deamidation")):
        labels[nz[:, col] & (nz.sum(axis=1) == 1)] = name
    return labels


def region_summary(points, pI, ok) -> dict:
    """What a consistent set (boolean ``ok`` over composition rows) looks like.

    Returns the number of consistent grid points, the range of each fraction
    and of the total acylation over them, and for each pure chemistry
    whether any point on its axis is consistent, with the range of the
    fraction where it is and the isoelectric point closest to the band that
    the axis reaches at all."""
    p = np.asarray(points, dtype=float)
    pI = np.asarray(pI, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    out = {"n_points": int(ok.sum())}
    cols = {"f_succ": p[:, 0], "f_acet": p[:, 1], "f_deam": p[:, 2],
            "f_acyl": p[:, 0] + p[:, 1]}
    for name, v in cols.items():
        out[f"{name}_min"] = float(v[ok].min()) if ok.any() else float("nan")
        out[f"{name}_max"] = float(v[ok].max()) if ok.any() else float("nan")
    labels = pure_chemistry(p)
    for col, name in enumerate(("succinylation", "acetylation", "deamidation")):
        axis = (labels == name) | (labels == "native")
        hit = axis & ok
        out[f"pure_{name}_reaches"] = bool(hit.any())
        out[f"pure_{name}_f_min"] = float(p[hit, col].min()) if hit.any() else float("nan")
        out[f"pure_{name}_f_max"] = float(p[hit, col].max()) if hit.any() else float("nan")
        out[f"pure_{name}_lowest_pI"] = float(np.nanmin(pI[axis])) if axis.any() else float("nan")
    return out


def minimum_succinylation(points, ok, f_deam_caps) -> list[dict]:
    """Smallest f_succ of any consistent composition whose deamidation does
    not exceed each cap, with the composition that attains it (ties: least
    acetylation, then least deamidation). ``reached`` is False if no
    consistent composition exists under the cap."""
    p = np.asarray(points, dtype=float)
    rows = []
    for cap in f_deam_caps:
        sel = np.flatnonzero(np.asarray(ok, bool) & (p[:, 2] <= cap + 1e-9))
        row = {"f_deam_cap": float(cap), "reached": bool(len(sel))}
        if len(sel):
            best = sel[np.lexsort((p[sel, 2], p[sel, 1], p[sel, 0]))[0]]
            row.update({"f_succ_min": p[best, 0], "at_f_acet": p[best, 1],
                        "at_f_deam": p[best, 2]})
        rows.append(row)
    return rows


# ------------------------------------------------------------------ assays ---
@dataclasses.dataclass(frozen=True)
class Assay:
    """One bench measurement and the precision assumed for it.

    ``sigma`` is one standard deviation of a single determination, absolute in
    the unit of the readout or, if ``relative``, a fraction of the readout.
    These precisions are assumptions (typical figures, stated in ``basis``),
    not properties of any particular instrument; the ranking is repeated with
    each halved and doubled."""

    key: str
    label: str
    unit: str
    sigma: float
    relative: bool
    basis: str


ASSAYS = (
    Assay("free_amine_fraction", "free amines remaining (TNBS)", "fraction of native",
          0.03, False,
          "trinitrobenzenesulfonate against the unmodified control; 3 % of the "
          "native amines"),
    Assay("amide_per_molecule", "amide content (Asn + Gln)", "per molecule", 6.0, False,
          "ammonia released on mild acid hydrolysis; 5 % of the 125 amides"),
    Assay("carboxyl_per_molecule", "carboxyl content", "per molecule", 0.05, True,
          "titration of the acid branch or carbodiimide coupling; 5 % of the value"),
    Assay("mass_shift_Da", "mass shift (intact chains)", "Da per molecule", 300.0, False,
          "mass spectrometry of intact alpha chains; 100 Da per 95 kDa chain"),
    Assay("net_charge_pH7", "net charge at pH 7", "e per molecule", 0.10, True,
          "titration from the isoelectric point to pH 7, or mobility; 10 % of the value"),
    Assay("net_charge_pH3", "net charge at pH 3", "e per molecule", 0.10, True,
          "titration from the isoelectric point to pH 3; 10 % of the value"),
    Assay("charge_slope_at_pI", "titration slope at the zero crossing", "e per pH", 0.10,
          True, "buffer capacity at the isoelectric point; 10 % of the value"),
    Assay("pI", "zero-charge pH", "pH", 0.10, False,
          "zeta-potential zero crossing or precipitation maximum; 0.1 pH"),
    Assay("pI_salt_shift", "shift of the zero-charge pH, 10 to 150 mM", "pH", 0.05, False,
          "difference of two zero crossings; 0.05 pH"),
)


def assay_readouts(mean_counts, native_counts, native_mass_da: float,
                   pka=es.DEFAULT_PKA_SET, salt_mM=(10.0, 150.0)) -> dict:
    """What each bench assay would read for a bulk sample whose molecules
    have, on average, the type counts ``mean_counts`` [..., N_TYPES].

    Every readout is either linear in the counts (contents, mass, the net
    charge at a fixed pH), so the bulk value is the function of the mean
    counts, or the zero of a linear function of them (the zero-charge pH of
    the sample). ``native_counts`` are the counts of the unmodified molecule.

    free_amine_fraction    unmodified amines / amines of the native molecule
    free_amine_mmol_g      the same as a content of the modified protein
    carboxyl_per_molecule  Asp + Glu + chain C-termini + succinyl carboxyls
    amide_per_molecule     Asn + Gln left
    mass_shift_Da          MASS_SHIFT_DA summed over the modified sites
    net_charge_pH*         at pH 3, 5.5, 7 and 9
    pI, charge_slope_at_pI zero crossing of the charge and dQ/dpH there
    pI_salt_shift          zero-charge pH at salt_mM[1] minus at salt_mM[0],
                           from :func:`ionic_strength_pka`
    """
    c = np.asarray(mean_counts, dtype=float)
    n = np.asarray(native_counts, dtype=float)
    mod = modification_counts(c, n)
    shift = (MASS_SHIFT_DA["succinyl"] * mod["succinyl"] + MASS_SHIFT_DA["acetyl"] * mod["acetyl"]
             + MASS_SHIFT_DA["deamidation"] * mod["deamidated"])
    mass = native_mass_da + shift                       # g/mol of the modified molecule
    carboxyl = c[..., CARBOXYL_TYPES].sum(-1)
    amide = c[..., AMIDE_TYPES].sum(-1)
    pI = isoelectric_point_from_counts(c, pka)
    lo, hi = (isoelectric_point_from_counts(c, ionic_strength_pka(pka, 1e-3 * s))
              for s in salt_mM)
    out = {"free_amine_fraction": mod["free_amines"] / n[..., AMINE_TYPES].sum(-1),
           "free_amine_mmol_g": 1e3 * mod["free_amines"] / mass,
           "succinyl_per_molecule": mod["succinyl"], "acetyl_per_molecule": mod["acetyl"],
           "deamidated_per_molecule": mod["deamidated"],
           "carboxyl_per_molecule": carboxyl, "carboxyl_mmol_g": 1e3 * carboxyl / mass,
           "amide_per_molecule": amide, "amide_mmol_g": 1e3 * amide / mass,
           "mass_shift_Da": shift, "pI": pI,
           "charge_slope_at_pI": charge_slope_from_counts(c, pI, pka),
           "pI_salt_shift": hi - lo}
    for pH in (3.0, 5.5, 7.0, 9.0):
        out[f"net_charge_pH{pH:g}"] = net_charge_from_counts(c, pH, pka)
    return out


def separated(readout, assay: Assay, n_sigma: float = 2.0, scale: float = 1.0) -> np.ndarray:
    """Matrix [i, j]: would a measurement made on candidate i exclude
    candidate j? True where their predicted readouts differ by more than
    ``n_sigma`` standard deviations of the assay (evaluated at candidate i
    for a relative precision; ``scale`` multiplies the assumed sigma)."""
    r = np.asarray(readout, dtype=float)
    sigma = scale * assay.sigma * (np.abs(r) if assay.relative else np.ones_like(r))
    return np.abs(r[:, None] - r[None, :]) > n_sigma * sigma[:, None]


def residual_ranges(points, excluded) -> np.ndarray:
    """Mean, over candidates taken in turn as the truth, of the range of each
    composition fraction among the candidates a measurement would *not*
    exclude: how much of (f_succ, f_acet, f_deam) the measurement leaves
    open. ``excluded`` is a matrix from :func:`separated` (or the logical OR
    of several, for assays used together)."""
    p = np.asarray(points, dtype=float)
    keep = ~np.asarray(excluded, dtype=bool)
    out = np.empty(p.shape[1])
    for col in range(p.shape[1]):
        v = np.where(keep, p[None, :, col], np.nan)
        out[col] = np.mean(np.nanmax(v, axis=1) - np.nanmin(v, axis=1))
    return out


def rank_assays(points, readouts: dict, assays=ASSAYS, n_sigma: float = 2.0,
                scales=(0.5, 1.0, 2.0)) -> list[dict]:
    """One row per assay, best first: the share of candidate pairs it tells
    apart at ``n_sigma`` (with the assumed precision, and with it scaled by
    each of ``scales``), the spread of its readout over the candidates in
    units of its precision, and the composition ranges it leaves open."""
    p = np.asarray(points, dtype=float)
    off = ~np.eye(len(p), dtype=bool)
    rows = []
    for a in assays:
        r = np.asarray(readouts[a.key], dtype=float)
        sigma = a.sigma * (np.abs(r).mean() if a.relative else 1.0)
        ex = separated(r, a, n_sigma)
        left = residual_ranges(p, ex)
        row = {"assay": a.key, "label": a.label, "unit": a.unit,
               "sigma_assumed": a.sigma, "sigma_is_relative": a.relative,
               "readout_min": float(r.min()), "readout_max": float(r.max()),
               "range_over_sigma": float(np.ptp(r) / sigma) if sigma > 0 else float("nan"),
               "pairs_separated": float(ex[off].mean()) if off.any() else float("nan")}
        for s in scales:
            if s != 1.0:
                row[f"pairs_separated_sigma_x{s:g}"] = float(
                    separated(r, a, n_sigma, s)[off].mean()) if off.any() else float("nan")
        row.update({"f_succ_range_left": left[0], "f_acet_range_left": left[1],
                    "f_deam_range_left": left[2], "basis": a.basis})
        rows.append(row)
    rows.sort(key=lambda row: -row["pairs_separated"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def rank_assay_sets(points, readouts: dict, keys, size: int, n_sigma: float = 2.0,
                    assays=ASSAYS) -> list[dict]:
    """All combinations of ``size`` assays out of ``keys`` used together (a
    pair of candidates is told apart if any assay of the set tells it apart),
    best first."""
    import itertools

    p = np.asarray(points, dtype=float)
    off = ~np.eye(len(p), dtype=bool)
    by_key = {a.key: a for a in assays}
    single = {k: separated(readouts[k], by_key[k], n_sigma) for k in keys}
    rows = []
    for combo in itertools.combinations(keys, size):
        ex = np.logical_or.reduce([single[k] for k in combo])
        left = residual_ranges(p, ex)
        rows.append({"assays": " + ".join(combo), "n_assays": size,
                     "pairs_separated": float(ex[off].mean()) if off.any() else float("nan"),
                     "f_succ_range_left": left[0], "f_acet_range_left": left[1],
                     "f_deam_range_left": left[2]})
    rows.sort(key=lambda row: -row["pairs_separated"])
    return rows


# --------------------------------------------------------- representatives ---
def pick_representatives(points, pI, target: float, n: int = 8):
    """``n`` compositions spanning a consistent set: (row indices, labels).

    First its extremes: the largest and the smallest succinylation, the
    largest acetylation, the largest and the smallest deamidation, and the
    smallest total acylation (ties broken by the isoelectric point closest to
    ``target``, then by row order; a composition that is extreme in two ways
    is taken once and carries both labels). The rest, labelled "fill", are
    added one at a time, each the composition farthest (Euclidean distance in
    the three fractions) from all those already chosen. Deterministic."""
    p = np.asarray(points, dtype=float)
    pI = np.asarray(pI, dtype=float)
    if len(p) == 0:
        return np.zeros(0, dtype=int), []
    off = np.round(np.abs(pI - target), 9)
    order = np.arange(len(p))
    chosen, labels = [], []
    for label, key in (("most succinylation", -p[:, 0]), ("least succinylation", p[:, 0]),
                       ("most acetylation", -p[:, 1]), ("most deamidation", -p[:, 2]),
                       ("least deamidation", p[:, 2]),
                       ("least acylation", p[:, 0] + p[:, 1])):
        i = int(np.lexsort((order, off, np.round(key, 9)))[0])
        if i in chosen:
            labels[chosen.index(i)] += ", " + label
        elif len(chosen) < n:
            chosen.append(i)
            labels.append(label)
    while len(chosen) < min(n, len(p)):
        d = np.min(np.linalg.norm(p[:, None, :] - p[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.lexsort((order, off, -np.round(d, 9)))[0]))
        labels.append("fill")
    return np.array(chosen, dtype=int), labels
