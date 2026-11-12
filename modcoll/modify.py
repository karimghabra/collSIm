"""Quenched random chemical modification of a tropocollagen site array.

Every molecule is modified site by site and keeps its own pattern. There is
no mean-field path: nothing here returns fractional site occupancies, and
ensemble statistics are obtained only by sampling many molecules.

Chemistry
---------
* succinylation: an amine (Lys N-epsilon, free chain N-terminus) reacts with
  succinic anhydride and becomes a carboxylate, charge +1 -> -1 when both are
  ionised (SiteType.SUCC_LYS / SUCC_NTERM);
* acetylation: the amine becomes a neutral amide (ACET_LYS / ACET_NTERM);
* deamidation: Asn -> Asp, Gln -> Glu (isoaspartate is not distinguished).

Hydrophobic weights are left unchanged by modification (the Kyte-Doolittle
scale has no entry for the acylated residues); see the model document.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .sites import Molecule, SiteType

_AMINE_TYPES = (int(SiteType.LYS), int(SiteType.NTERM))
_AMIDE_TYPES = (int(SiteType.ASN), int(SiteType.GLN))
_SUCC = {int(SiteType.LYS): int(SiteType.SUCC_LYS),
         int(SiteType.NTERM): int(SiteType.SUCC_NTERM)}
_ACET = {int(SiteType.LYS): int(SiteType.ACET_LYS),
         int(SiteType.NTERM): int(SiteType.ACET_NTERM)}
_DEAM = {int(SiteType.ASN): int(SiteType.ASP),
         int(SiteType.GLN): int(SiteType.GLU)}


def modification_probabilities(weights, fraction: float) -> np.ndarray:
    """Per-site probability of being modified, for relative reactivities
    ``weights`` and a target mean modified ``fraction``.

    Derivation: independent sites reacting with pseudo-first-order rate
    constants proportional to w_i have reacted with probability
    p_i = 1 - exp(-w_i * tau) after a common reaction extent tau. tau is
    fixed by requiring mean_i(p_i) = fraction, which is what an assay of the
    overall degree of modification measures. For uniform weights this is
    p_i = fraction exactly. Sites with w_i = 0 never react, so the reachable
    maximum is the fraction of sites with w_i > 0.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return np.zeros(0)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-negative")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must lie in [0, 1]")
    reachable = np.count_nonzero(w > 0) / w.size
    if fraction > reachable + 1e-12:
        raise ValueError(f"fraction {fraction} exceeds the reachable maximum "
                         f"{reachable:.4f} (sites with zero weight never react)")
    if fraction == 0.0:
        return np.zeros_like(w)
    if fraction >= reachable:
        return (w > 0).astype(float)
    if np.all(w == w[0]):
        return np.full_like(w, fraction)

    def mean_p(tau):
        return float(np.mean(-np.expm1(-w * tau)))

    lo, hi = 0.0, 1.0
    while mean_p(hi) < fraction:
        hi *= 2.0
    for _ in range(200):                      # bisection on a monotone function
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < fraction:
            lo = mid
        else:
            hi = mid
    return -np.expm1(-w * 0.5 * (lo + hi))


def modify(molecule: Molecule, rng: np.random.Generator, f_succ: float = 0.0,
           f_acet: float = 0.0, f_deam: float = 0.0, weights=None) -> Molecule:
    """Return a copy of ``molecule`` with one random modification pattern.

    Each amine (Lys, free N-terminus) is independently succinylated,
    acetylated or left alone; each Asn/Gln is independently deamidated.
    ``f_succ`` and ``f_acet`` are the mean fractions of amines carrying each
    group (f_succ + f_acet <= 1), ``f_deam`` the mean fraction of amides
    converted. ``weights`` is an optional array over all sites of relative
    reactivity (default uniform; see :func:`accessibility_weights`); an amine
    is modified with the probability of :func:`modification_probabilities`
    for the total fraction f_succ + f_acet and the two chemistries share the
    modified sites in the ratio f_succ : f_acet.

    Randomness: exactly one uniform deviate per amine and one per amide is
    drawn from ``rng``, in site order, whatever the fractions are. Two calls
    with generators in the same state therefore use the same deviates, so the
    set of modified sites grows monotonically with the fraction (nested
    patterns), which removes sampling noise from comparisons across fractions.

    Sites already modified are not eligible, so the function is applied to a
    native molecule.
    """
    for name, f in (("f_succ", f_succ), ("f_acet", f_acet), ("f_deam", f_deam)):
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    f_tot = f_succ + f_acet
    if f_tot > 1.0 + 1e-12:
        raise ValueError("f_succ + f_acet must not exceed 1")
    st = molecule.site_type
    w = np.ones(len(molecule)) if weights is None else np.asarray(weights, float)
    if w.shape != st.shape:
        raise ValueError("weights must have one entry per site")

    amine = np.flatnonzero(np.isin(st, _AMINE_TYPES))
    amide = np.flatnonzero(np.isin(st, _AMIDE_TYPES))
    u = rng.random(amine.size)
    v = rng.random(amide.size)

    new = st.copy()
    if f_tot > 0.0 and amine.size:
        p = modification_probabilities(w[amine], min(f_tot, 1.0))
        succ = u < p * (f_succ / f_tot)
        acet = (u < p) & ~succ
        for src, dst in _SUCC.items():
            new[amine[succ & (st[amine] == src)]] = dst
        for src, dst in _ACET.items():
            new[amine[acet & (st[amine] == src)]] = dst
    if f_deam > 0.0 and amide.size:
        hit = v < modification_probabilities(w[amide], f_deam)
        for src, dst in _DEAM.items():
            new[amide[hit & (st[amide] == src)]] = dst
    return dataclasses.replace(molecule, site_type=new)


# Helical cross-linking lysines, as (chain type, 1-based helix position).
CROSSLINK_LYSINES = {"alpha1": (87, 930), "alpha2": (87, 933)}


def accessibility_weights(molecule: Molecule, model: str = "uniform",
                          paired_weight: float = 0.5,
                          fast_weight: float = 10.0) -> np.ndarray:
    """Relative reactivity of every site (1 = reference).

    "uniform"   all sites equally reactive (the default of :func:`modify`).

    "crosslink_sites"  the four helical cross-linking lysines (alpha1 Lys87
                and Lys930, alpha2 Lys87 and Lys933; six residues per
                molecule) react ``fast_weight`` (default 10) times faster than
                the rest. Evidence, all from other electrophiles: a small set
                of abnormally reactive hydroxylysines with pKa about 8.5 in
                native collagen (Gonzalez, Hamabata & Rojkind 1984, Coll.
                Relat. Res. 4:339); glycation confined to exactly these sites
                (Hudson et al. 2018, J. Biol. Chem. 293:15620); carbamylation
                preferring alpha2 Lys933 (Taga et al. 2017, Cell Chem. Biol.
                24:1276). A pKa of 8.5 against 10.4 is roughly a ten-fold rate
                advantage at pH 8-9 if the neutral amine reacts. That these
                are the reactive residues toward anhydrides, and the factor
                itself (plausibly 3-30), are inferences: no site-resolved
                acylation map of collagen exists.

    "ion_pair"  ASSUMPTION with no supporting data: a helix lysine in the Y
                position followed by Gly-Asp or Gly-Glu (the K-G-D / K-G-E
                motif) is given ``paired_weight`` (default 0.5). Motivation:
                in host-guest triple-helical peptides these sequences form
                interchain Lys-carboxylate ion pairs (Persikov et al. 2005,
                Biochemistry 44:1414), and an ion-paired amine is a poorer
                nucleophile. Kept as a second, differently shaped weighting
                for sensitivity tests only.
    """
    w = np.ones(len(molecule))
    if model == "uniform":
        return w
    if model == "crosslink_sites":
        if fast_weight < 0.0:
            raise ValueError("fast_weight must be non-negative")
        helix = molecule.is_residue & ~molecule.smeared
        for chains, positions in (((0, 1), CROSSLINK_LYSINES["alpha1"]),
                                  ((2,), CROSSLINK_LYSINES["alpha2"])):
            hit = (helix & np.isin(molecule.chain, chains)
                   & np.isin(molecule.res_index, [pos - 1 for pos in positions]))
            if np.any(molecule.native_type[hit] != int(SiteType.LYS)):
                raise ValueError("a cross-link position is not a lysine in this molecule")
            w[hit] = fast_weight
        return w
    if model != "ion_pair":
        raise ValueError(f"unknown accessibility model {model!r}")
    if not 0.0 <= paired_weight:
        raise ValueError("paired_weight must be non-negative")
    n_helix = int(molecule.res_index[~molecule.smeared].max()) + 1
    for chain in np.unique(molecule.chain):
        rows = np.flatnonzero((molecule.chain == chain) & molecule.is_residue
                              & ~molecule.smeared)
        by_index = {int(molecule.res_index[k]): k for k in rows}
        for ri, k in by_index.items():
            if molecule.native_type[k] != int(SiteType.LYS) or ri % 3 != 2:
                continue                      # Y position: G=0, X=1, Y=2
            if ri + 2 >= n_helix:
                continue
            g, x = by_index[ri + 1], by_index[ri + 2]
            if molecule.aa[g] == "G" and molecule.aa[x] in ("D", "E"):
                w[k] = paired_weight
    return w
