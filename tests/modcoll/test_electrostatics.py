"""Protonation and pair interactions.

Oracles: CODATA constants, textbook Debye lengths, the closed-form
Henderson-Hasselbalch and Yukawa expressions, and hand-worked geometry.
"""
import functools

import numpy as np
import pytest

from modcoll import electrostatics as es
from modcoll.modify import modify
from modcoll.sites import SiteType, make_molecule, native_molecule


def test_bjerrum_length_from_fundamental_constants():
    # l_B = e^2 / (4 pi eps0 eps_r k T), CODATA 2018, water eps_r = 78.4 at 298.15 K
    e, eps0, k_b = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23
    l_b_nm = e ** 2 / (4 * np.pi * eps0 * 78.4 * k_b * 298.15) * 1e9
    assert es.BJERRUM_NM == pytest.approx(l_b_nm, rel=0.01)


@pytest.mark.parametrize("ionic_M, expected_nm", [
    # Israelachvili: 0.304 / sqrt(I) nm for a 1:1 electrolyte at 25 C
    (0.001, 9.61), (0.01, 3.04), (0.1, 0.961), (0.15, 0.785), (1.0, 0.304),
])
def test_debye_length_matches_textbook_values(ionic_M, expected_nm):
    assert es.debye_length_nm(ionic_M) == pytest.approx(expected_nm, rel=0.01)


def test_debye_length_scales_as_inverse_square_root_of_ionic_strength():
    assert es.debye_length_nm(0.04) / es.debye_length_nm(0.16) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        es.debye_length_nm(0.0)
    assert es.Environment(7.0, 0.05).debye_length_nm == es.debye_length_nm(0.05)


def one_site(site_type):
    return make_molecule([0.0], [0.0], [0.5], [site_type])


@pytest.mark.parametrize("site_type, sign", [
    (SiteType.ASP, -1), (SiteType.GLU, -1), (SiteType.CTERM, -1), (SiteType.TYR, -1),
    (SiteType.SUCC_LYS, -1), (SiteType.SUCC_NTERM, -1),
    (SiteType.HIS, +1), (SiteType.LYS, +1), (SiteType.ARG, +1), (SiteType.NTERM, +1),
])
def test_henderson_hasselbalch_half_charge_at_pka_and_limits(site_type, sign):
    mol = one_site(site_type)
    pka = es.PKA[site_type]
    q = lambda pH: es.henderson_hasselbalch(mol, es.Environment(pH))[0]
    assert q(pka) == pytest.approx(0.5 * sign)
    # one pH unit on the ionised side: 10/11 of a charge; on the other: 1/11
    assert q(pka + sign * -1.0) == pytest.approx(sign * 10 / 11)
    assert q(pka + sign * 1.0) == pytest.approx(sign * 1 / 11)


@pytest.mark.parametrize("site_type", [SiteType.OTHER, SiteType.ASN, SiteType.GLN,
                                       SiteType.ACET_LYS, SiteType.ACET_NTERM])
def test_neutral_types_carry_no_charge(site_type):
    for pH in (2.0, 7.0, 12.0):
        assert es.henderson_hasselbalch(one_site(site_type), es.Environment(pH))[0] == 0.0


def test_named_pka_sets():
    side = (SiteType.ASP, SiteType.GLU, SiteType.HIS, SiteType.LYS, SiteType.ARG)
    # README: "Asp 3.65, Glu 4.25, His 6.0, Lys 10.53, Arg 12.48"
    assert [es.PKA_SETS["repository"][t] for t in side] == [3.65, 4.25, 6.0, 10.53, 12.48]
    # model-compound values: Grimsley et al. 2009 (Asp), Thurlkill et al. 2006
    # (Glu, His, Lys, Tyr, termini), Fitch et al. 2015 (Arg)
    rec = es.PKA_SETS["recommended"]
    assert [rec[t] for t in side] == [3.9, 4.3, 6.5, 10.4, 13.8]
    assert rec[SiteType.TYR] == 9.8 and rec[SiteType.CTERM] == 3.7
    assert rec[SiteType.SUCC_LYS] == es.PKA_SETS["repository"][SiteType.SUCC_LYS] == 4.5
    assert es.PKA is rec and es.DEFAULT_PKA_SET == "recommended"
    with pytest.raises(ValueError):
        es.charge_model_for("textbook")


def test_tyrosine_titrates_only_in_the_recommended_set():
    tyr = one_site(SiteType.TYR)
    env = es.Environment(9.8)
    assert es.charge_model_for("recommended")(tyr, env)[0] == pytest.approx(-0.5)
    assert es.charge_model_for("repository")(tyr, env)[0] == 0.0
    assert abs(es.henderson_hasselbalch(tyr, es.Environment(4.2))[0]) < 1e-5


def test_asp_pka_shift_moves_the_net_charge_at_ph_4_2_by_about_15():
    """Hand calculation for the 92 Asp, 143 Glu, 3 C-termini and 18 His of a
    molecule, repository -> recommended set at pH 4.2:
    Asp  92 [1/(1+10^(3.65-4.2)) - 1/(1+10^(3.9-4.2))] = 10.5
    Glu 143 [1/(1+10^(4.25-4.2)) - 1/(1+10^(4.3-4.2))] =  4.1
    C-termini 0.04; His 18 [1/(1+10^(4.2-6.5)) - 1/(1+10^(4.2-6.0))] = 0.19.
    Lys and Arg are fully protonated in both sets."""
    native = native_molecule()
    env = es.Environment(4.2)
    shift = (es.net_charge(native, env, es.charge_model_for("recommended"))
             - es.net_charge(native, env, es.charge_model_for("repository")))
    acid = lambda pka: 1.0 / (1.0 + 10.0 ** (pka - 4.2))
    base = lambda pka: 1.0 / (1.0 + 10.0 ** (4.2 - pka))
    expected = (92 * (acid(3.65) - acid(3.9)) + 143 * (acid(4.25) - acid(4.3))
                + 3 * (acid(3.67) - acid(3.7)) + 18 * (base(6.5) - base(6.0)))
    assert shift == pytest.approx(expected, abs=1e-4)
    assert 92 * (acid(3.65) - acid(3.9)) == pytest.approx(10.5, abs=0.1)
    assert shift == pytest.approx(14.8, abs=0.1)


def test_isoelectric_point_of_one_acid_one_base_is_the_mean_pka():
    # zwitterion with well separated pKa: pI = (pKa_acid + pKa_base) / 2 exactly
    mol = make_molecule([0, 1], [0, 0], [0.5, 0.5], [SiteType.ASP, SiteType.LYS])
    assert es.isoelectric_point(mol) == pytest.approx((3.9 + 10.4) / 2, abs=1e-9)
    assert es.net_charge(mol, es.Environment(7.15)) == pytest.approx(0.0, abs=1e-9)
    repo = es.charge_model_for("repository")
    assert es.isoelectric_point(mol, charge_model=repo) == pytest.approx(
        (3.65 + 10.53) / 2, abs=1e-9)


def test_isoelectric_point_two_acids_one_base():
    # 2 Asp + 1 Lys: zero net charge where 2 / (1 + 10^(3.9 - pH)) = 1 (Lys
    # fully protonated there), i.e. pH = 3.9
    mol = make_molecule([0, 1, 2], [0] * 3, [0.5] * 3,
                        [SiteType.ASP, SiteType.ASP, SiteType.LYS])
    assert es.isoelectric_point(mol) == pytest.approx(3.9, abs=1e-5)


def test_isoelectric_point_is_nan_without_a_sign_change():
    acids = make_molecule([0, 1], [0, 0], [0.5, 0.5], [SiteType.ASP, SiteType.GLU])
    assert np.isnan(es.isoelectric_point(acids))


def test_isoelectric_point_of_an_ensemble_is_where_the_mean_charge_vanishes():
    a = make_molecule([0, 1], [0, 0], [0.5, 0.5], [SiteType.ASP, SiteType.LYS])
    b = make_molecule([0, 1, 2], [0] * 3, [0.5] * 3,
                      [SiteType.ASP, SiteType.ASP, SiteType.LYS])
    pi = es.isoelectric_point([a, b])
    total = es.net_charge(a, es.Environment(pi)) + es.net_charge(b, es.Environment(pi))
    assert total == pytest.approx(0.0, abs=1e-9)
    assert es.isoelectric_point(b) < pi < es.isoelectric_point(a)


def test_charge_model_is_replaceable():
    mol = make_molecule([0, 1], [0, 0], [0.5, 0.5], [SiteType.ASP, SiteType.LYS])

    def shifted(molecule, env):          # a stand-in for charge regulation
        table = dict(es.PKA)
        table[SiteType.ASP] = 5.65
        return es.henderson_hasselbalch(molecule, env, pka=table)

    assert es.isoelectric_point(mol, charge_model=shifted) == pytest.approx(
        (5.65 + 10.4) / 2, abs=1e-9)
    assert es.net_charge(mol, es.Environment(5.65), shifted) == pytest.approx(0.5, abs=1e-4)


def test_native_and_fully_succinylated_isoelectric_points():
    """Sanity gates of the charge model. Fully succinylated collagen must
    come out near pH 4-4.5 (measured: 4.0 at 90 % conversion, Xu et al. 2017,
    J. Appl. Polym. Sci. 134:45424). The unmodified sequence gives 9.8-10.1,
    not the ~9 usually quoted for collagen; the test pins the model's actual
    values so the gap stays visible."""
    native = native_molecule()
    succ = modify(native, np.random.default_rng(0), f_succ=1.0)
    for name, pi_native, pi_succ, q_succ in (("repository", 10.11, 4.21, 1.5),
                                             ("recommended", 9.84, 4.29, 16.3)):
        model = es.charge_model_for(name)
        assert es.isoelectric_point(native, charge_model=model) == pytest.approx(pi_native, abs=0.01)
        assert es.isoelectric_point(succ, charge_model=model) == pytest.approx(pi_succ, abs=0.01)
        assert 4.0 <= es.isoelectric_point(succ, charge_model=model) <= 4.5
        assert es.net_charge(succ, es.Environment(4.2), model) == pytest.approx(q_succ, abs=0.1)
    # about one fifth of Asn/Gln deamidated brings the native pI down to 9
    rng = np.random.default_rng(1)
    deam = [modify(native, rng, f_deam=0.2) for _ in range(40)]
    assert es.isoelectric_point(deam) == pytest.approx(9.0, abs=0.3)


def test_extraction_changes_the_fully_succinylated_charge_only_slightly():
    # pepsin-cut ends: pI 4.27, +12 e at pH 4.2; helix alone: 4.30, +18 e
    for telopeptides, pi, q in (("mature", 4.29, 16.3), ("pepsin", 4.27, 12.2),
                                ("none", 4.30, 18.4)):
        mol = modify(native_molecule(telopeptides), np.random.default_rng(0), f_succ=1.0)
        assert es.isoelectric_point(mol) == pytest.approx(pi, abs=0.01)
        assert es.net_charge(mol, es.Environment(4.2)) == pytest.approx(q, abs=0.2)


def test_native_net_charge_follows_from_the_counts():
    # at pH 7.4: Lys, Arg fully +, Asp, Glu, C-termini fully -, His (pKa 6.5)
    # 1/(1+10^0.9); the 13 Tyr (pKa 9.8) contribute -0.05 in total
    m = native_molecule()
    n = {t: m.count(t) for t in SiteType}
    expected = (n[SiteType.LYS] + n[SiteType.ARG] - n[SiteType.ASP] - n[SiteType.GLU]
                - n[SiteType.CTERM] + n[SiteType.HIS] / (1 + 10 ** 0.9))
    assert es.net_charge(m, es.Environment(7.4)) == pytest.approx(expected, abs=0.2)
    assert es.net_charge(m, es.Environment(7.4)) == pytest.approx(32.0, abs=0.3)


def test_screened_coulomb_closed_forms():
    lam = 0.8
    r = np.array([0.3, 1.0, 2.5])
    np.testing.assert_allclose(es.screened_coulomb(r, lam, core_nm=0.0),
                               0.71 * np.exp(-r / lam) / r)
    # soft core: finite at contact, l_B exp(-a/lambda) / a
    a = es.SOFT_CORE_NM
    assert es.screened_coulomb(0.0, lam) == pytest.approx(0.71 * np.exp(-a / lam) / a)
    # far beyond the core: r_eff = r + a^2 / 2r to first order, so the bare
    # Yukawa form is scaled by exp(-a^2 / (2 r lambda)) (1 - a^2 / (2 r^2))
    r_far = 5.0
    bare = 0.71 * np.exp(-r_far / lam) / r_far
    correction = np.exp(-a ** 2 / (2 * r_far * lam)) * (1 - a ** 2 / (2 * r_far ** 2))
    assert es.screened_coulomb(r_far, lam) == pytest.approx(bare * correction, rel=1e-4)
    assert 0.97 < correction < 1.0
    # unscreened limit at long Debye length: Coulomb's law in units of l_B
    assert es.screened_coulomb(2.0, 1e9, core_nm=0.0) == pytest.approx(0.71 / 2.0)


def test_hydrophobic_contact_kernel():
    assert es.hydrophobic_contact(0.0) == 1.0
    assert es.hydrophobic_contact(0.5, sigma_nm=0.5) == pytest.approx(np.exp(-0.5))
    assert es.hydrophobic_contact(3.0) < 1e-7


def test_transverse_distance_hand_worked():
    d = 1.65
    f = es.transverse_distance_sq
    assert np.sqrt(f(0.6, 0.0, 0.5, 0.0, d)) == pytest.approx(d - 1.1)        # face to face
    assert np.sqrt(f(0.6, np.pi, 0.5, np.pi, d)) == pytest.approx(d + 1.1)    # back to back
    assert np.sqrt(f(0.6, np.pi, 0.5, 0.0, d)) == pytest.approx(d + 0.1)
    # both a quarter turn in their own sense: B is A rotated by pi, so the two
    # sites are on opposite sides of the plane through the axes
    assert f(0.6, np.pi / 2, 0.5, np.pi / 2, d) == pytest.approx(d ** 2 + 1.1 ** 2)
    assert f(0.6, np.pi / 2, 0.6, -np.pi / 2, d) == pytest.approx(d ** 2)


def point(s, phi, r, site_type=SiteType.LYS, length=10.0):
    return make_molecule([s], [phi], [r], [site_type], helix_length_nm=length)


def test_direct_pair_energy_of_two_point_charges():
    lam, d = 0.9, 1.65
    kern = functools.partial(es.screened_coulomb, debye_nm=lam)
    a, b = point(2.0, 0.3, 0.6), point(5.0, -1.1, 0.5)
    # facing each other (fa = phi_a, fb = phi_b) and axially aligned by the stagger
    e = es.pair_energy_direct(a, [1.0], b, [-1.0], kern, stagger_nm=-3.0,
                              facing_a=0.3, facing_b=-1.1, axis_separation_nm=d)
    assert e == pytest.approx(-es.screened_coulomb(d - 1.1, lam))
    # 0.7 nm further along the axis
    e = es.pair_energy_direct(a, [1.0], b, [-1.0], kern, stagger_nm=-2.3,
                              facing_a=0.3, facing_b=-1.1, axis_separation_nm=d)
    assert e == pytest.approx(-es.screened_coulomb(np.hypot(d - 1.1, 0.7), lam))
    # A turned half a turn away
    e = es.pair_energy_direct(a, [2.0], b, [3.0], kern, stagger_nm=-3.0,
                              facing_a=0.3 + np.pi, facing_b=-1.1, axis_separation_nm=d)
    assert e == pytest.approx(6.0 * es.screened_coulomb(d + 0.6 - 0.5, lam))


def test_antiparallel_flip_reverses_axis_and_handedness():
    s, phi = es.flip_antiparallel(np.array([1.0, 9.0]), np.array([0.4, -2.0]), 10.0)
    np.testing.assert_allclose(s, [9.0, 1.0])
    np.testing.assert_allclose(phi, [-0.4, 2.0])
    lam, d = 0.9, 1.65
    kern = functools.partial(es.screened_coulomb, debye_nm=lam)
    a, b = point(2.0, 0.0, 0.6), point(1.0, 0.7, 0.5, length=10.0)
    # flipped, B's site sits at s = 9 and azimuth -0.7: stagger -7 aligns it with A
    e = es.pair_energy_direct(a, [1.0], b, [1.0], kern, stagger_nm=-7.0, facing_a=0.0,
                              facing_b=-0.7, axis_separation_nm=d, antiparallel=True)
    assert e == pytest.approx(es.screened_coulomb(d - 1.1, lam))


def test_smeared_site_is_the_ring_average():
    lam, d, r_ring = 0.9, 1.65, 0.55
    kern = functools.partial(es.screened_coulomb, debye_nm=lam)
    a = point(0.0, 0.0, 0.6)
    ring = make_molecule([0.0], [0.0], [r_ring], [SiteType.ASP], smeared=[True])
    got = es.pair_energy_direct(a, [1.0], ring, [1.0], kern, 0.0, 0.0, 0.0, d, n_ring=720)
    beta = np.linspace(0.0, 2 * np.pi, 20001)
    dist = np.sqrt(es.transverse_distance_sq(0.6, 0.0, r_ring, beta, d))
    y = es.screened_coulomb(dist, lam)
    exact = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(beta)) / (2 * np.pi)   # trapezoid
    assert got == pytest.approx(exact, rel=1e-6)
    # and it does not depend on the facing angle of the smeared molecule
    turned = es.pair_energy_direct(a, [1.0], ring, [1.0], kern, 0.0, 0.0, 1.234, d,
                                   n_ring=720)
    assert turned == pytest.approx(got, rel=1e-6)


def test_like_charges_repel_less_at_higher_salt():
    d = 1.65
    a, b = point(0.0, 0.0, 0.6), point(0.0, 0.0, 0.6)
    e = [es.pair_energy_direct(a, [1.0], b, [1.0],
                               functools.partial(es.screened_coulomb,
                                                 debye_nm=es.debye_length_nm(i)),
                               0.0, 0.0, 0.0, d) for i in (0.01, 0.05, 0.15, 0.3)]
    assert all(x > y > 0 for x, y in zip(e, e[1:]))
