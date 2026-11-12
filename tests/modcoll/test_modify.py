"""Quenched random modification: statistics, reproducibility, chemistry.

Oracles: binomial sampling theory, the closed-form solution of the
two-class reactivity model, and hand-built sequences.
"""
import dataclasses

import numpy as np
import pytest

from modcoll import electrostatics as es
from modcoll.modify import accessibility_weights, modification_probabilities, modify
from modcoll.sites import SiteType, make_molecule, native_molecule


def counts(m):
    return {t: m.count(t) for t in SiteType}


def test_zero_fractions_leave_the_molecule_unchanged():
    native = native_molecule()
    out = modify(native, np.random.default_rng(0))
    np.testing.assert_array_equal(out.site_type, native.site_type)
    assert out is not native


def test_complete_succinylation_and_acetylation():
    native = native_molecule("mature", "free")
    n_lys, n_nt = native.count(SiteType.LYS), native.count(SiteType.NTERM)
    succ = modify(native, np.random.default_rng(1), f_succ=1.0)
    assert succ.count(SiteType.SUCC_LYS) == n_lys and succ.count(SiteType.LYS) == 0
    assert succ.count(SiteType.SUCC_NTERM) == n_nt and succ.count(SiteType.NTERM) == 0
    acet = modify(native, np.random.default_rng(1), f_acet=1.0)
    assert acet.count(SiteType.ACET_LYS) == n_lys and acet.count(SiteType.ACET_NTERM) == n_nt


def test_only_eligible_sites_change():
    native = native_molecule()
    out = modify(native, np.random.default_rng(2), f_succ=0.4, f_acet=0.3, f_deam=0.5)
    changed = out.site_type != native.site_type
    assert set(native.site_type[changed]) <= {int(SiteType.LYS), int(SiteType.NTERM),
                                              int(SiteType.ASN), int(SiteType.GLN)}
    for t in (SiteType.ARG, SiteType.HIS, SiteType.CTERM, SiteType.OTHER):
        assert out.count(t) == native.count(t)
    # geometry and hydrophobic weights are untouched; native_type remembers
    np.testing.assert_array_equal(out.s, native.s)
    np.testing.assert_array_equal(out.hydro, native.hydro)
    np.testing.assert_array_equal(out.native_type, native.site_type)


def test_deamidation_maps_asn_to_asp_and_gln_to_glu():
    native = native_molecule()
    out = modify(native, np.random.default_rng(3), f_deam=1.0)
    assert out.count(SiteType.ASN) == 0 and out.count(SiteType.GLN) == 0
    assert out.count(SiteType.ASP) == native.count(SiteType.ASP) + native.count(SiteType.ASN)
    assert out.count(SiteType.GLU) == native.count(SiteType.GLU) + native.count(SiteType.GLN)


def test_modified_fraction_follows_the_binomial_distribution():
    native = native_molecule()
    n_lys, f, n_mol = native.count(SiteType.LYS), 0.3, 400
    rng = np.random.default_rng(4)
    k = np.array([modify(native, rng, f_succ=f).count(SiteType.SUCC_LYS)
                  for _ in range(n_mol)])
    # mean of n_mol binomial(n_lys, f) draws: within 4 standard errors
    se = np.sqrt(n_lys * f * (1 - f) / n_mol)
    assert abs(k.mean() - n_lys * f) < 4 * se
    # variance of a binomial count, n_lys f (1 - f), to 4 sigma of its estimator
    var = n_lys * f * (1 - f)
    assert abs(k.var(ddof=1) - var) < 4 * var * np.sqrt(2.0 / (n_mol - 1))
    assert len(set(k)) > 10                 # molecules differ: no mean-field path


def test_every_molecule_gets_its_own_pattern():
    native = native_molecule()
    rng = np.random.default_rng(5)
    a, b = modify(native, rng, f_succ=0.5), modify(native, rng, f_succ=0.5)
    assert np.any(a.site_type != b.site_type)
    # sites are hit independently: the two patterns overlap at chance level
    lys = native.site_type == int(SiteType.LYS)
    both = np.mean((a.site_type[lys] == int(SiteType.SUCC_LYS))
                   & (b.site_type[lys] == int(SiteType.SUCC_LYS)))
    assert abs(both - 0.25) < 4 * np.sqrt(0.25 * 0.75 / lys.sum())


def test_same_seed_same_pattern():
    native = native_molecule()
    a = modify(native, np.random.default_rng(6), f_succ=0.4, f_acet=0.2, f_deam=0.1)
    b = modify(native, np.random.default_rng(6), f_succ=0.4, f_acet=0.2, f_deam=0.1)
    np.testing.assert_array_equal(a.site_type, b.site_type)
    c = modify(native, np.random.default_rng(7), f_succ=0.4, f_acet=0.2, f_deam=0.1)
    assert np.any(a.site_type != c.site_type)


def test_patterns_are_nested_across_fractions_for_one_generator_state():
    native = native_molecule()
    prev = np.zeros(len(native), bool)
    for f in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        m = modify(native, np.random.default_rng(8), f_succ=f)
        hit = m.site_type == int(SiteType.SUCC_LYS)
        assert np.all(hit[prev])            # nothing un-modifies as f grows
        assert hit.sum() >= prev.sum()
        prev = hit


def test_mixed_chemistry_splits_modified_sites_in_the_requested_ratio():
    native = native_molecule()
    rng = np.random.default_rng(9)
    n_s = n_a = n_tot = 0
    for _ in range(200):
        m = modify(native, rng, f_succ=0.45, f_acet=0.15)
        n_s += m.count(SiteType.SUCC_LYS); n_a += m.count(SiteType.ACET_LYS)
        n_tot += native.count(SiteType.LYS)
    assert abs(n_s / n_tot - 0.45) < 4 * np.sqrt(0.45 * 0.55 / n_tot)
    assert abs(n_a / n_tot - 0.15) < 4 * np.sqrt(0.15 * 0.85 / n_tot)


def test_uniform_weights_give_probability_equal_to_the_fraction():
    np.testing.assert_allclose(modification_probabilities(np.full(7, 2.5), 0.37), 0.37)


def test_two_class_reactivity_matches_closed_form():
    # weights 1 and 1/2 in equal numbers: with x = exp(-tau/2) the mean
    # modified fraction is 1 - (x^2 + x)/2 = f  =>  x = (-1 + sqrt(9 - 8f))/2
    w = np.array([1.0, 0.5] * 50)
    for f in (0.1, 0.5, 0.9):
        x = (-1.0 + np.sqrt(9.0 - 8.0 * f)) / 2.0
        p = modification_probabilities(w, f)
        np.testing.assert_allclose(p[0], 1 - x * x, rtol=1e-10)
        np.testing.assert_allclose(p[1], 1 - x, rtol=1e-10)
        assert p.mean() == pytest.approx(f, abs=1e-12)


def test_zero_weight_sites_never_react_and_cap_the_fraction():
    w = np.array([1.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(modification_probabilities(w, 0.5), [1, 1, 0, 0])
    with pytest.raises(ValueError):
        modification_probabilities(w, 0.6)
    np.testing.assert_array_equal(modification_probabilities(w, 0.0), 0.0)


def test_weights_bias_which_sites_are_modified():
    native = native_molecule()
    lys = np.flatnonzero(native.site_type == int(SiteType.LYS))
    w = np.ones(len(native))
    slow = lys[::2]
    w[slow] = 0.25
    p = modification_probabilities(w[lys], 0.5)
    rng = np.random.default_rng(10)
    hits = np.zeros(len(native))
    n_mol = 300
    for _ in range(n_mol):
        hits += modify(native, rng, f_succ=0.5, weights=w).site_type == int(SiteType.SUCC_LYS)
    freq = hits[lys] / n_mol
    assert abs(freq.mean() - 0.5) < 0.01
    # per-site frequencies follow the model probabilities
    assert np.all(np.abs(freq - p) < 5 * np.sqrt(p * (1 - p) / n_mol))
    assert freq[::2].mean() < freq[1::2].mean() - 0.2


def test_invalid_arguments():
    native = native_molecule()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        modify(native, rng, f_succ=0.7, f_acet=0.4)
    with pytest.raises(ValueError):
        modify(native, rng, f_succ=-0.1)
    with pytest.raises(ValueError):
        modify(native, rng, f_deam=1.5)
    with pytest.raises(ValueError):
        modify(native, rng, f_succ=0.5, weights=np.ones(3))
    with pytest.raises(ValueError):
        modification_probabilities([1.0, -1.0], 0.5)


def test_succinylating_one_lysine_changes_the_net_charge_by_minus_two():
    # at pH 7.4 Lys is +1 (pKa 10.53) and a succinyl carboxylate -1 (pKa 4.5)
    mol = make_molecule([0.0, 1.0], [0.0, 0.0], [0.5, 0.5],
                        [SiteType.LYS, SiteType.ARG])
    env = es.Environment(pH=7.4)
    succ = modify(mol, np.random.default_rng(0), f_succ=1.0)
    acet = modify(mol, np.random.default_rng(0), f_acet=1.0)
    assert es.net_charge(succ, env) - es.net_charge(mol, env) == pytest.approx(-2.0, abs=3e-3)
    assert es.net_charge(acet, env) - es.net_charge(mol, env) == pytest.approx(-1.0, abs=3e-3)


def toy_helix(sequence):
    n = len(sequence)
    types = [{"K": SiteType.LYS, "D": SiteType.ASP, "E": SiteType.GLU}.get(a, SiteType.OTHER)
             for a in sequence]
    mol = make_molecule(0.2858 * np.arange(n), np.zeros(n), np.full(n, 0.5), types)
    return dataclasses.replace(mol, aa=np.array(list(sequence), dtype="<U1"))


def test_ion_pair_weights_pick_out_y_position_lys_followed_by_gly_acid():
    #          0123456789012345678
    seq = "GPKGEPGKPGDKGAKGDAG"
    # K2: Y position, then G, E  -> paired
    # K7: X position             -> not paired
    # K11: Y position, then G, A -> not paired
    # K14: Y position, then G, D -> paired
    mol = toy_helix(seq)
    w = accessibility_weights(mol, "ion_pair", paired_weight=0.3)
    expected = np.ones(len(seq))
    expected[[2, 14]] = 0.3
    np.testing.assert_array_equal(w, expected)
    np.testing.assert_array_equal(accessibility_weights(mol, "uniform"), 1.0)
    with pytest.raises(ValueError):
        accessibility_weights(mol, "buried")


def test_crosslink_site_weights_mark_the_six_cross_linking_lysines():
    native = native_molecule()
    w = accessibility_weights(native, "crosslink_sites")
    fast = np.flatnonzero(w != 1.0)
    assert len(fast) == 6 and np.all(w[fast] == 10.0)
    assert np.all(native.site_type[fast] == int(SiteType.LYS))
    # helix positions alpha1 87 and 930 (two chains each), alpha2 87 and 933
    got = sorted((int(native.chain[k]), int(native.res_index[k]) + 1) for k in fast)
    assert got == [(0, 87), (0, 930), (1, 87), (1, 930), (2, 87), (2, 933)]
    # their published sequence context: G-M-K-G-H-R, G-I-K-G-H-R, G-F-K-G-I-R, G-L-K-G-H-N
    def context(chain, pos):
        rows = {int(native.res_index[k]): native.aa[k] for k in np.flatnonzero(
            (native.chain == chain) & native.is_residue & ~native.smeared)}
        return "".join(rows[i] for i in range(pos - 3, pos + 3))
    assert [context(c, p) for c, p in ((0, 87), (0, 930), (2, 87), (2, 933))] == [
        "GMKGHR", "GIKGHR", "GFKGIR", "GLKGHN"]
    assert np.all(accessibility_weights(native, "crosslink_sites", fast_weight=3.0)[fast] == 3.0)


def test_crosslink_sites_fill_first_at_low_modification():
    native = native_molecule()
    w = accessibility_weights(native, "crosslink_sites")
    fast = np.flatnonzero(w != 1.0)
    lys = np.flatnonzero(native.site_type == int(SiteType.LYS))
    p = modification_probabilities(w[lys], 0.2)
    # closed form: p_fast = 1 - (1 - p_slow)^10 for a ten-fold rate constant
    p_slow, p_fast = p[w[lys] == 1.0][0], p[w[lys] == 10.0][0]
    assert p_fast == pytest.approx(1.0 - (1.0 - p_slow) ** 10, rel=1e-9)
    assert p_fast > 0.8 and p_slow < 0.2
    rng = np.random.default_rng(21)
    hits = sum(int(np.sum(modify(native, rng, f_succ=0.2, weights=w).site_type[fast]
                          == int(SiteType.SUCC_LYS))) for _ in range(100))
    assert abs(hits / 600 - p_fast) < 4 * np.sqrt(p_fast * (1 - p_fast) / 600)


def test_ion_pair_weights_on_collagen_touch_only_lysines():
    native = native_molecule()
    w = accessibility_weights(native, "ion_pair")
    lowered = w < 1.0
    assert 0 < lowered.sum() < native.count(SiteType.LYS)
    assert np.all(native.site_type[lowered] == int(SiteType.LYS))
    assert np.all(~native.smeared[lowered])       # telopeptide Lys keep weight 1
