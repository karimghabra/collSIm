"""Composition space: type counts, isoelectric points, masses, assays.

Oracles: the existing per-site functions (``electrostatics.net_charge`` and
``isoelectric_point``), closed forms for hand-built sets of sites, limits
(f = 0, f = 1), hand arithmetic for masses and activity terms, and the
committed S1 tables of ``results/modcoll/ensemble``.
"""
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from modcoll import composition as co
from modcoll import electrostatics as es
from modcoll.modify import modify
from modcoll.sites import SiteType, make_molecule, native_molecule

ROOT = Path(__file__).resolve().parents[2]
S1 = ROOT / "results" / "modcoll" / "ensemble"
T = {t.name: int(t) for t in SiteType}


def counts_of(**n):
    c = np.zeros(co.N_TYPES)
    for name, value in n.items():
        c[T[name]] = value
    return c


@pytest.fixture(scope="module")
def native():
    return native_molecule()


# ------------------------------------------------------- counts <-> sites ---
@pytest.mark.parametrize("ends", ["mature", "pepsin"])
@pytest.mark.parametrize("pka", ["recommended", "repository"])
def test_counts_give_the_per_site_charge_and_isoelectric_point(ends, pka):
    """Q = sum_t n_t q_t is an identity for independent-site protonation: it
    must reproduce the per-site sum of the existing functions exactly."""
    mol0 = native_molecule(ends)
    model = es.charge_model_for(pka)
    for k, (fs, fa, fd) in enumerate([(0, 0, 0), (0.5, 0.3, 0.2), (0.9, 0, 0.18), (0, 1, 1)]):
        mol = modify(mol0, np.random.default_rng(k), fs, fa, fd)
        counts = co.type_counts(mol)
        assert counts.sum() == len(mol)
        for pH in (2.0, 4.2, 7.0, 9.5, 12.0):
            assert co.net_charge_from_counts(counts, pH, pka) == pytest.approx(
                es.net_charge(mol, es.Environment(pH), model), abs=1e-9)
        assert co.isoelectric_point_from_counts(counts, pka) == pytest.approx(
            es.isoelectric_point(mol, charge_model=model), abs=1e-9)


def test_isoelectric_point_of_a_sample_is_that_of_its_summed_counts(native):
    mols = [modify(native, np.random.default_rng(k), 0.6, 0.2, 0.3) for k in range(5)]
    counts = np.array([co.type_counts(m) for m in mols])
    assert co.isoelectric_point_from_counts(counts.sum(axis=0)) == pytest.approx(
        es.isoelectric_point(mols), abs=1e-9)


def test_isoelectric_point_closed_forms():
    # one acid and one base in equal numbers: pI = (pKa_acid + pKa_base) / 2
    assert co.isoelectric_point_from_counts(counts_of(GLU=7, HIS=7)) == pytest.approx(
        0.5 * (4.3 + 6.5), abs=1e-9)
    # n_a acids against n_b < n_a bases that stay protonated (Arg, pKa 13.8):
    # n_b = n_a / (1 + 10^(pKa - pH))  =>  pH = pKa + log10(n_b / (n_a - n_b))
    for n_a, n_b in ((100, 40), (100, 50), (300, 290)):
        assert co.isoelectric_point_from_counts(counts_of(ASP=n_a, ARG=n_b)) == pytest.approx(
            3.9 + np.log10(n_b / (n_a - n_b)), abs=1e-6)
    # rows are solved independently; no root (acids only) gives NaN
    rows = np.array([counts_of(GLU=7, HIS=7), counts_of(ASP=10), counts_of(ASP=100, ARG=50)])
    out = co.isoelectric_point_from_counts(rows)
    assert out[0] == pytest.approx(5.4, abs=1e-9) and np.isnan(out[1])
    assert out[2] == pytest.approx(3.9, abs=1e-6)


def test_isoelectric_point_keeps_the_leading_shape():
    rows = np.array([[counts_of(GLU=7, HIS=7)] * 3] * 2)
    assert co.isoelectric_point_from_counts(rows).shape == (2, 3)


def test_charge_slope_closed_form_and_finite_difference(native):
    # a single acid at its pKa: theta = 1/2, dQ/dpH = -ln(10) / 4 per site
    assert co.charge_slope_from_counts(counts_of(ASP=8), 3.9) == pytest.approx(
        -8 * np.log(10.0) / 4.0)
    counts = co.type_counts(modify(native, np.random.default_rng(3), 0.7, 0.2, 0.4))
    h = 1e-5
    numeric = (co.net_charge_from_counts(counts, 4.2 + h)
               - co.net_charge_from_counts(counts, 4.2 - h)) / (2 * h)
    assert co.charge_slope_from_counts(counts, 4.2) == pytest.approx(numeric, rel=1e-7)
    assert numeric < 0


def test_net_charge_broadcasts_over_pH():
    counts = counts_of(GLU=3, HIS=2)
    q = co.net_charge_from_counts(counts, np.array([2.0, 5.4, 12.0]))
    assert q.shape == (3,)
    assert q[0] == pytest.approx(2.0, abs=0.02) and q[2] == pytest.approx(-3.0, abs=1e-4)


# ------------------------------------------------------------------ limits ---
def test_unmodified_and_fully_succinylated_limits_reproduce_s1(native):
    """f = 0 and f_succ = 1 have no randomness: they must give the committed
    S1 gate numbers exactly, with zero spread."""
    gates = json.loads((S1 / "gates.json").read_text())["by_pka_set"]["recommended"]
    length = co.contour_length_nm(native)
    for comp, pi_key, q_key in (((0.0, 0.0, 0.0), "native_pI", "native_net_charge_pH4.2"),
                                ((1.0, 0.0, 0.0), "fully_succinylated_pI",
                                 "fully_succinylated_net_charge_pH4.2")):
        counts = co.sample_counts(native, *comp, n_molecules=8, seed=1)
        s = co.point_statistics(counts, length_nm=length, rng=np.random.default_rng(0))
        assert s["pI_ensemble"] == pytest.approx(gates[pi_key], abs=1e-9)
        assert s["pI_molecule_mean"] == pytest.approx(gates[pi_key], abs=1e-9)
        assert s["net_charge_pH4.2_mean"] == pytest.approx(gates[q_key], abs=1e-9)
        assert s["pI_molecule_sd"] == 0.0 and s["pI_ensemble_se"] == 0.0
        assert s["net_charge_pH4.2_sd"] == 0.0 and s["net_charge_pH4.2_se"] == 0.0
        assert s["charge_per_nm_pH4.2_mean"] == pytest.approx(gates[q_key] / length)
    assert np.array_equal(co.sample_counts(native, 0, 0, 0, 3, seed=5),
                          np.tile(co.type_counts(native), (3, 1)))


def _s1_rows(name):
    with open(S1 / name, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("chemistry, f, comp", [
    ("succ", 0.25, (0.25, 0, 0)), ("succ", 0.5, (0.5, 0, 0)), ("succ", 0.9, (0.9, 0, 0)),
    ("acet", 0.5, (0, 0.5, 0)), ("acet", 1.0, (0, 1, 0)),
    ("deam", 0.5, (0, 0, 0.5)), ("deam", 1.0, (0, 0, 1)),
    ("mixed", 0.5, (0.25, 0.25, 0)), ("mixed", 1.0, (0.5, 0.5, 0)),
])
def test_single_chemistry_axes_reproduce_the_committed_s1_tables(native, chemistry, f, comp):
    """The pure-chemistry axes of the composition grid are the S1 ensembles.
    S1 drew other molecules, so agreement is within the sampling error of the
    two samples (5 combined standard errors), and exact where a point has no
    randomness."""
    pi = next(r for r in _s1_rows("pi_vs_f.csv")
              if r["chemistry"] == chemistry and float(r["f"]) == f)
    q = next(r for r in _s1_rows("net_charge.csv") if r["chemistry"] == chemistry
             and float(r["f"]) == f and float(r["pH"]) == 4.2)
    n1 = int(pi["n_molecules"])
    counts = co.sample_counts(native, *comp, n_molecules=256, seed=20261112)
    s = co.point_statistics(counts, pH_values=(4.2,), rng=co.bootstrap_rng(1, *comp))
    se_pi = np.hypot(s["pI_ensemble_se"], float(pi["pI_molecule_sd"] or 0) / np.sqrt(n1))
    assert s["pI_ensemble"] == pytest.approx(float(pi["pI_ensemble"]), abs=5 * se_pi + 1e-5)
    se_q = np.hypot(s["net_charge_pH4.2_se"], float(q["net_charge_sd"] or 0) / np.sqrt(n1))
    assert s["net_charge_pH4.2_mean"] == pytest.approx(float(q["net_charge_mean"]),
                                                       abs=5 * se_q + 1e-3)
    assert s["net_charge_pH4.2_sd"] == pytest.approx(float(q["net_charge_sd"] or 0),
                                                     rel=0.2, abs=1e-9)


@pytest.mark.parametrize("chemistry, f", [("succ", 0.25), ("succ", 0.9), ("acet", 0.75),
                                          ("mixed", 0.5), ("deam", 0.3)])
def test_count_statistics_reproduce_the_committed_s1_rows_exactly(native, chemistry, f):
    """Fed the very molecules S1 sampled, the count-based statistics must give
    the committed S1 numbers to the six digits the tables carry."""
    from modcoll import cli_ensemble
    mols = cli_ensemble.sample(native, chemistry, f, 256, 20261112)
    s = co.point_statistics(np.array([co.type_counts(m) for m in mols]), pH_values=(4.2, 7.4))
    pi = next(r for r in _s1_rows("pi_vs_f.csv")
              if r["chemistry"] == chemistry and float(r["f"]) == f)
    for key in ("pI_ensemble", "pI_molecule_mean", "pI_molecule_sd", "pI_molecule_min",
                "pI_molecule_max"):
        assert s[key] == pytest.approx(float(pi[key]), rel=6e-6), key
    for pH in (4.2, 7.4):
        q = next(r for r in _s1_rows("net_charge.csv") if r["chemistry"] == chemistry
                 and float(r["f"]) == f and float(r["pH"]) == pH)
        assert s[f"net_charge_pH{pH:g}_mean"] == pytest.approx(float(q["net_charge_mean"]),
                                                               rel=6e-6)
        assert s[f"net_charge_pH{pH:g}_sd"] == pytest.approx(float(q["net_charge_sd"]),
                                                             rel=6e-6)
        assert s[f"fraction_net_positive_pH{pH:g}"] == pytest.approx(
            float(q["fraction_net_positive"]), abs=1e-6)


def test_net_charge_mean_and_spread_match_the_multinomial_closed_form(native):
    """Each amine is succinylated, acetylated or left alone with probabilities
    (f_s, f_a, 1 - f_s - f_a), each amide deamidated with probability f_d, all
    independently. With q the charge of each outcome at pH 4.2,
        mean = sum_sites sum_k p_k q_k,
        var  = sum_sites [ sum_k p_k q_k^2 - (sum_k p_k q_k)^2 ].
    """
    fs, fa, fd = 0.5, 0.3, 0.2
    n = 3000
    q = co.type_charges(4.2)
    nat = co.type_counts(native)
    p = np.array([fs, fa, 1 - fs - fa])
    q_amine = np.array([q[T["SUCC_LYS"]], q[T["ACET_LYS"]], q[T["LYS"]]])
    mean = (nat * q).sum() - nat[T["LYS"]] * q[T["LYS"]] + nat[T["LYS"]] * (p * q_amine).sum()
    var = nat[T["LYS"]] * ((p * q_amine ** 2).sum() - (p * q_amine).sum() ** 2)
    for amide, acid in (("ASN", "ASP"), ("GLN", "GLU")):
        mean += nat[T[amide]] * fd * q[T[acid]]
        var += nat[T[amide]] * fd * (1 - fd) * q[T[acid]] ** 2
    s = co.point_statistics(co.sample_counts(native, fs, fa, fd, n, seed=11), pH_values=(4.2,))
    assert s["net_charge_pH4.2_mean"] == pytest.approx(mean, abs=4 * np.sqrt(var / n))
    # relative error of a sample sd is about 1 / sqrt(2 n) = 1.3 %
    assert s["net_charge_pH4.2_sd"] == pytest.approx(np.sqrt(var), rel=0.06)
    assert s["net_charge_pH4.2_se"] == pytest.approx(s["net_charge_pH4.2_sd"] / np.sqrt(n))


def test_sampling_is_seeded_nested_and_independent_between_molecules(native):
    a = co.sample_counts(native, 0.6, 0.2, 0.3, 12, seed=3)
    assert np.array_equal(a, co.sample_counts(native, 0.6, 0.2, 0.3, 12, seed=3))
    assert not np.array_equal(a, co.sample_counts(native, 0.6, 0.2, 0.3, 12, seed=4))
    assert len({tuple(row) for row in a}) > 1                 # molecules differ
    lo = co.sample_counts(native, 0.3, 0.2, 0.1, 12, seed=3)
    hi = co.sample_counts(native, 0.6, 0.2, 0.3, 12, seed=3)
    # molecule k keeps its deviates: more succinylation and more deamidation
    # at the larger fractions, molecule by molecule
    assert np.all(hi[:, T["SUCC_LYS"]] >= lo[:, T["SUCC_LYS"]])
    assert np.all(hi[:, T["ASN"]] <= lo[:, T["ASN"]])
    # every sampled molecule conserves amines and amides
    assert np.all(a[:, [T["LYS"], T["SUCC_LYS"], T["ACET_LYS"]]].sum(axis=1) == 107)
    assert np.all(a[:, T["ASN"]] + a[:, T["GLN"]] + a[:, T["ASP"]] + a[:, T["GLU"]]
                  == 46 + 79 + 92 + 143)


def test_bootstrap_standard_error_of_the_sample_isoelectric_point(native):
    counts = co.sample_counts(native, 0.9, 0.0, 0.0, 256, seed=2)
    a = co.point_statistics(counts, rng=co.bootstrap_rng(7, 0.9, 0, 0))
    b = co.point_statistics(counts, rng=co.bootstrap_rng(7, 0.9, 0, 0))
    assert a["pI_ensemble_se"] == b["pI_ensemble_se"] > 0
    # near pH 4.2 the pI is nearly linear in the counts, so the pI of the
    # sample is the mean of the molecules' and its error is sd / sqrt(n)
    assert a["pI_ensemble"] == pytest.approx(a["pI_molecule_mean"], abs=2e-3)
    assert a["pI_ensemble_se"] == pytest.approx(a["pI_molecule_se"], rel=0.25)
    assert "pI_ensemble_se" not in co.point_statistics(counts)
    # the generator depends on the composition, not on any grid
    assert (co.bootstrap_rng(7, 0.9, 0, 0).random() != co.bootstrap_rng(7, 0.9, 0, 0.1).random())


# ------------------------------------------------------------ pKa and salt ---
def test_pka_variant_replaces_only_what_is_asked():
    base = es.PKA_SETS["recommended"]
    v = co.pka_variant(succinyl=4.35, asp=3.65)
    assert v[SiteType.SUCC_LYS] == v[SiteType.SUCC_NTERM] == 4.35 and v[SiteType.ASP] == 3.65
    assert v[SiteType.GLU] == base[SiteType.GLU] and base[SiteType.ASP] == 3.9   # no aliasing
    assert co.pka_variant() == base
    for name, (lo, mid, hi) in co.PKA_RANGES.items():
        assert lo < mid < hi
    assert co.PKA_RANGES["succinyl"][1] == base[SiteType.SUCC_LYS]
    assert co.PKA_RANGES["asp"][1] == base[SiteType.ASP]
    assert co.PKA_RANGES["glu"][1] == base[SiteType.GLU]


def test_davies_term_against_hand_arithmetic():
    # 0.509 * (sqrt(I) / (1 + sqrt(I)) - 0.3 I)
    assert co.davies_term(0.01) == pytest.approx(0.509 * (0.1 / 1.1 - 0.003))
    assert co.davies_term(0.1) == pytest.approx(0.509 * (0.3162278 / 1.3162278 - 0.03), rel=1e-6)
    np.testing.assert_allclose(co.davies_term([0.01, 0.02, 0.05, 0.10, 0.15, 0.30]),
                               [0.045, 0.060, 0.085, 0.107, 0.119, 0.134], atol=6e-4)
    assert co.davies_term(0.0) == 0.0


def test_salt_strengthens_acids_and_weakens_bases():
    base = es.PKA_SETS["recommended"]
    assert co.ionic_strength_pka(base, co.PKA_REFERENCE_IONIC_M) == pytest.approx(base)
    lo, hi = co.ionic_strength_pka(base, 0.01), co.ionic_strength_pka(base, 0.15)
    d = float(co.davies_term(0.15) - co.davies_term(0.01))
    assert d == pytest.approx(0.0744, abs=2e-4)
    for t in base:
        assert hi[t] - lo[t] == pytest.approx(es.VALENCE[t] * d)
    assert hi[SiteType.ASP] < lo[SiteType.ASP] and hi[SiteType.LYS] > lo[SiteType.LYS]


def test_zero_charge_pH_shift_with_salt_closed_form():
    """If only acids titrate near the zero crossing (the bases stay fully
    protonated), every pKa that matters moves by the same -A dF, and so does
    the zero crossing, whatever the composition."""
    d = float(co.davies_term(0.15) - co.davies_term(0.01))
    nat = counts_of(ASP=100, GLU=80, ARG=120, ASN=10, LYS=20)
    for counts in (counts_of(ASP=100, GLU=80, ARG=120), counts_of(ASP=60, SUCC_LYS=90, ARG=70)):
        r = co.assay_readouts(counts, nat, 1e5)
        assert r["pI_salt_shift"] == pytest.approx(-d, abs=1e-6)


# ------------------------------------------------------------------ masses ---
def test_molecular_mass_of_a_hand_built_tripeptide():
    mol = make_molecule([0, 1, 2], [0, 0, 0], [0.5, 0.5, 0.5],
                        [SiteType.OTHER, SiteType.OTHER, SiteType.LYS])
    mol = dataclasses.replace(mol, aa=np.array(["G", "A", "K"]))
    # Gly-Ala-Lys, C11H22N4O4: 274.32 g/mol
    assert co.molecular_mass_da(mol) == pytest.approx(274.32, abs=0.01)


def test_molecular_mass_of_the_native_molecule(native):
    """Three alpha chains of about 95 kDa. Collagen has a mean residue mass
    near 91 Da, and 107 lysines in 3154 residues are 34 per 1000, so the
    amine content must come out near 34 / 91 = 0.37 mmol/g."""
    mass = co.molecular_mass_da(native)
    assert 280e3 < mass < 295e3
    assert mass / 3154 == pytest.approx(91.4, abs=0.5)
    assert 1e3 * 107 / mass == pytest.approx(0.37, abs=0.01)
    # hydroxyproline is proline plus one oxygen (15.999)
    assert co.RESIDUE_MASS_DA["O"] - co.RESIDUE_MASS_DA["P"] == pytest.approx(15.9994, abs=1e-3)
    # the intact chains begin with pyroglutamate (-17.03 each); pepsin-cut
    # chains do not, and are shorter
    pep = native_molecule("pepsin")
    assert co.molecular_mass_da(pep) < mass - 3000


def test_assay_readouts_of_a_hand_built_molecule():
    """4 Lys, 2 Asn, 2 Gln, 1 Asp, 3 Arg; then 2 Lys succinylated, 1
    acetylated, 1 Asn and 1 Gln deamidated."""
    native = counts_of(LYS=4, ASN=2, GLN=2, ASP=1, ARG=3)
    modified = counts_of(LYS=1, SUCC_LYS=2, ACET_LYS=1, ASN=1, GLN=1, ASP=2, GLU=1, ARG=3)
    mass0 = 2000.0
    r = co.assay_readouts(modified, native, mass0)
    shift = 2 * 100.016 + 42.011 + 2 * 0.984
    assert r["mass_shift_Da"] == pytest.approx(shift) == pytest.approx(244.011)
    assert r["free_amine_fraction"] == pytest.approx(0.25)
    assert r["succinyl_per_molecule"] == 2 and r["acetyl_per_molecule"] == 1
    assert r["deamidated_per_molecule"] == 2
    assert r["carboxyl_per_molecule"] == 5          # Asp 2 + Glu 1 + succinyl 2
    assert r["amide_per_molecule"] == 2
    assert r["carboxyl_mmol_g"] == pytest.approx(1e3 * 5 / (mass0 + shift))
    assert r["free_amine_mmol_g"] == pytest.approx(1e3 * 1 / (mass0 + shift))
    # pH 9: carboxyls ionised (-5), Arg +3, the one Lys +0.96
    assert r["net_charge_pH9"] == pytest.approx(-5 + 3 + 1 / (1 + 10 ** (9 - 10.4)), abs=1e-3)
    assert co.net_charge_from_counts(modified, r["pI"]) == pytest.approx(0.0, abs=1e-9)
    assert r["charge_slope_at_pI"] < 0
    # unmodified molecule: nothing shifted, every amine free
    r0 = co.assay_readouts(native, native, mass0)
    assert r0["mass_shift_Da"] == 0 and r0["free_amine_fraction"] == 1.0
    # vectorised over candidates
    both = co.assay_readouts(np.array([native, modified]), native, mass0)
    assert both["mass_shift_Da"] == pytest.approx([0.0, shift])
    assert both["pI"][1] == pytest.approx(float(r["pI"]))


def test_mass_shifts_are_the_monoisotopic_group_masses():
    # C4H4O3, C2H2O and -NH2 +OH from monoisotopic atomic masses
    c, h, o, n = 12.0, 1.0078250319, 15.9949146221, 14.0030740052
    assert co.MASS_SHIFT_DA["succinyl"] == pytest.approx(4 * c + 4 * h + 3 * o, abs=1e-3)
    assert co.MASS_SHIFT_DA["acetyl"] == pytest.approx(2 * c + 2 * h + o, abs=1e-3)
    assert co.MASS_SHIFT_DA["deamidation"] == pytest.approx(o - n - h, abs=1e-3)


# ---------------------------------------------------------------- sequence ---
def test_bovine_porcine_histidine_correction(native):
    corrected = co.without_alpha2_histidines(native)
    assert native.count(SiteType.HIS) - corrected.count(SiteType.HIS) == 4
    changed = np.flatnonzero(native.site_type != corrected.site_type)
    assert set(native.chain[changed]) == {2}
    assert sorted(native.res_index[changed] + 1) == list(co.BOVINE_ABSENT_HIS_ALPHA2)
    assert set(native.aa[changed]) == {"H"}
    assert native.count(SiteType.HIS) == 18                    # the input is not altered
    # at pH 4.2 a histidine (pKa 6.5) is 99.5 % protonated
    env = es.Environment(4.2)
    assert es.net_charge(native, env) - es.net_charge(corrected, env) == pytest.approx(
        4 / (1 + 10 ** (4.2 - 6.5)))
    assert es.isoelectric_point(corrected) < es.isoelectric_point(native)
    with pytest.raises(ValueError):
        co.without_alpha2_histidines(native, positions=(1, 2, 3, 4))   # Gly, not His
    with pytest.raises(ValueError):
        co.without_alpha2_histidines(corrected)                        # already removed


def test_histidine_correction_leaves_the_modification_pattern_alone(native):
    """Modification never touches histidine, so the corrected molecule has
    the same patterns: counts differ by the four sites only. This is what
    lets the sensitivity scan reuse one sample for both sequences."""
    corrected = co.without_alpha2_histidines(native)
    a = co.sample_counts(native, 0.6, 0.3, 0.4, 10, seed=9)
    b = co.sample_counts(corrected, 0.6, 0.3, 0.4, 10, seed=9)
    diff = a - b
    assert np.all(diff[:, T["HIS"]] == 4) and np.all(diff[:, T["OTHER"]] == -4)
    assert not np.delete(diff, [T["HIS"], T["OTHER"]], axis=1).any()


# -------------------------------------------------------------------- grid ---
def test_composition_grid():
    g = co.composition_grid(0.25)
    # 15 ways to split the amines in quarters, 5 deamidation levels
    assert len(g) == 15 * 5
    assert np.all(g[:, 0] + g[:, 1] <= 1 + 1e-12) and g.min() == 0.0 and g.max() == 1.0
    assert len({tuple(row) for row in g}) == len(g)
    g2 = co.composition_grid(0.25, extra_f_deam=[co.LIMING_F_DEAM, 0.5])
    assert len(g2) == 15 * 6 and np.count_nonzero(g2[:, 2] == 0.18) == 15
    fine = co.composition_grid(0.025)
    assert len(fine) == (41 * 42 // 2) * 41
    assert (0.9, 0.0, 0.0) in {tuple(row) for row in fine}     # exact, not 0.9000000001
    np.testing.assert_array_equal(co.on_grid(g2, 0.25), g2[:, 2] != 0.18)
    with pytest.raises(ValueError):
        co.composition_grid(0.3)


# ----------------------------------------------------------------- inverse ---
def test_bands_are_closed_and_nested():
    pI = np.array([3.89, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.51])
    n = [int(co.in_band(pI, 4.2, b).sum()) for b in (0.1, 0.2, 0.3)]
    assert n == [3, 5, 7]
    for narrow, wide in ((0.1, 0.2), (0.2, 0.3)):
        assert np.all(co.in_band(pI, 4.2, wide)[co.in_band(pI, 4.2, narrow)])
    assert co.soluble_at([-120.0, 30.0, 70.0], 66.0).tolist() == [True, False, True]


def test_region_summary_and_minimum_succinylation_on_a_synthetic_map():
    """A made-up linear map pI = 5 - f_succ - 0.5 f_acet - 0.4 f_deam on a
    quarter grid, target 4.2 +- 0.2 (so 4.0 ... 4.4)."""
    p = co.composition_grid(0.25)
    pI = 5.0 - p[:, 0] - 0.5 * p[:, 1] - 0.4 * p[:, 2]
    ok = co.in_band(pI, 4.2, 0.2)
    s = co.region_summary(p, pI, ok)
    assert s["n_points"] == int(ok.sum()) > 0
    # succinylation alone: f = 0.75 gives 4.25, f = 1 gives 4.0; both in
    assert s["pure_succinylation_reaches"] and s["pure_succinylation_f_min"] == 0.75
    assert s["pure_succinylation_lowest_pI"] == pytest.approx(4.0)
    # acetylation alone stops at 4.5, deamidation alone at 4.6
    assert not s["pure_acetylation_reaches"]
    assert s["pure_acetylation_lowest_pI"] == pytest.approx(4.5)
    assert not s["pure_deamidation_reaches"]
    assert np.isnan(s["pure_deamidation_f_min"])
    # without succinylation: 5 - 0.5 a - 0.4 d <= 4.4 needs a = 1, d >= 0.25
    rows = co.minimum_succinylation(p, ok, [0.0, 0.25, 1.0])
    assert [r["f_deam_cap"] for r in rows] == [0.0, 0.25, 1.0]
    # no deamidation: s + a/2 >= 0.6 with s + a <= 1 -> s = 0.25, a = 0.75
    assert (rows[0]["f_succ_min"], rows[0]["at_f_acet"]) == (0.25, 0.75)
    assert (rows[1]["f_succ_min"], rows[1]["at_f_acet"], rows[1]["at_f_deam"]) == (0.0, 1.0, 0.25)
    assert rows[2]["f_succ_min"] == 0.0
    none = co.minimum_succinylation(p, np.zeros(len(p), bool), [1.0])[0]
    assert none["reached"] is False and "f_succ_min" not in none
    assert co.pure_chemistry([(0, 0, 0), (0.5, 0, 0), (0, 1, 0), (0, 0, 0.2), (0.1, 0, 0.2)]
                             ).tolist() == ["native", "succinylation", "acetylation",
                                            "deamidation", ""]


# ------------------------------------------------------------------ assays ---
def test_separation_of_two_candidates_closed_form():
    absolute = co.Assay("x", "x", "", sigma=1.0, relative=False, basis="")
    r = np.array([10.0, 13.0])                      # 3 sd apart
    assert co.separated(r, absolute).tolist() == [[False, True], [True, False]]
    assert not co.separated(r, absolute, scale=2.0).any()          # 1.5 sd apart
    assert not co.separated(r, absolute, n_sigma=3.0).any()        # strictly more than
    relative = co.Assay("x", "x", "", sigma=0.1, relative=True, basis="")
    # 2 sd is 2.0 seen from 10 and 2.6 seen from 13: asymmetric at a gap of 2.5
    assert co.separated(np.array([10.0, 12.5]), relative).tolist() == [[False, True],
                                                                       [False, False]]


def test_ranking_prefers_the_assay_that_spreads_the_candidates():
    points = np.array([(0.0, 1.0, 0.5), (0.5, 0.5, 0.5), (1.0, 0.0, 0.5)])
    assays = (co.Assay("sharp", "sharp", "", 1.0, False, ""),
              co.Assay("blunt", "blunt", "", 1.0, False, ""))
    readouts = {"sharp": np.array([0.0, 10.0, 20.0]), "blunt": np.array([0.0, 1.0, 5.0])}
    rows = co.rank_assays(points, readouts, assays)
    assert [r["assay"] for r in rows] == ["sharp", "blunt"] and rows[0]["rank"] == 1
    assert rows[0]["pairs_separated"] == 1.0
    assert rows[0]["range_over_sigma"] == pytest.approx(20.0)
    # the sharp assay leaves each candidate alone; nothing is left open
    assert rows[0]["f_succ_range_left"] == 0.0 and rows[0]["f_deam_range_left"] == 0.0
    # the blunt one separates only the third candidate from the other two:
    # 4 of 6 ordered pairs; the first two remain confused (range 0.5 for two
    # of the three truths)
    assert rows[1]["pairs_separated"] == pytest.approx(4 / 6)
    assert rows[1]["f_succ_range_left"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)
    sets = co.rank_assay_sets(points, readouts, ["sharp", "blunt"], 2, assays=assays)
    assert sets[0]["assays"] == "sharp + blunt" and sets[0]["pairs_separated"] == 1.0


def test_residual_ranges_without_any_exclusion_are_the_full_ranges():
    points = np.array([(0.0, 0.2, 0.0), (0.4, 0.6, 1.0)])
    np.testing.assert_allclose(co.residual_ranges(points, np.zeros((2, 2), bool)),
                               [0.4, 0.4, 1.0])


def test_assay_table_is_complete():
    keys = [a.key for a in co.ASSAYS]
    assert len(set(keys)) == len(keys)
    native = counts_of(LYS=4, ASN=2, ASP=3, ARG=3)
    r = co.assay_readouts(native, native, 1000.0)
    assert set(keys) <= set(r)
    assert all(a.sigma > 0 and a.basis for a in co.ASSAYS)


# --------------------------------------------------------- representatives ---
def test_representatives_span_the_set_and_are_deterministic():
    p = co.composition_grid(0.25)
    pI = 5.0 - p[:, 0] - 0.5 * p[:, 1] - 0.4 * p[:, 2]
    sel = np.flatnonzero(co.in_band(pI, 4.2, 0.2))
    idx, labels = co.pick_representatives(p[sel], pI[sel], 4.2, n=7)
    idx2, labels2 = co.pick_representatives(p[sel], pI[sel], 4.2, n=7)
    assert idx.tolist() == idx2.tolist() and labels == labels2
    assert len(idx) == len(set(idx.tolist())) == len(labels) == 7
    chosen = p[sel][idx]
    for col in range(3):                              # each fraction's extremes are in
        assert chosen[:, col].max() == p[sel][:, col].max()
    assert chosen[:, 0].min() == p[sel][:, 0].min()
    assert (chosen[:, 0] + chosen[:, 1]).min() == (p[sel][:, 0] + p[sel][:, 1]).min()
    assert labels[0].startswith("most succinylation") and "fill" in labels
    # a tie between extremes is broken by the isoelectric point nearest the target
    first = chosen[0]
    ties = p[sel][p[sel][:, 0] == first[0]]
    assert abs(5.0 - first[0] - 0.5 * first[1] - 0.4 * first[2] - 4.2) == pytest.approx(
        np.abs(5.0 - ties[:, 0] - 0.5 * ties[:, 1] - 0.4 * ties[:, 2] - 4.2).min())
    # fewer candidates than asked for: all of them, once
    few, _ = co.pick_representatives(p[sel][:3], pI[sel][:3], 4.2, n=8)
    assert sorted(few.tolist()) == [0, 1, 2]
    assert len(co.pick_representatives(p[:0], pI[:0], 4.2)[0]) == 0
