"""Pair second virial coefficient (modcoll.solubility).

Oracles: closed forms (water permittivity, Bjerrum length, the engine's
hydrophobic temperature law, pH of a strong and of a weak acid), limits where
an interaction vanishes (no charges, no hydrophobic term, very high salt), a
hand calculation of the virial integral for two point charges done by direct
quadrature in (rho, Delta) without the FFT machinery, and the symmetry of B2
under exchange of the two molecules.
"""
from __future__ import annotations

import numpy as np
import pytest

from modcoll import composition as co
from modcoll import electrostatics as es
from modcoll import solubility as so
from modcoll.sites import SiteType, make_molecule, native_molecule

FAST = dict(near_offsets_nm=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0),
            far_offsets_nm=(5.0, 8.0), n_phi=12, n_shell_charge=2, n_shell_hydro=2)


def toy_molecule(seed: int, n: int = 60, length_nm: float = 20.0):
    """Sites on grid nodes (s a multiple of 0.2 nm, phi of 30 degrees, r one
    of two shells) so that turning the molecule end over end maps nodes to
    nodes: the grid deposit is then exact for both orientations."""
    rng = np.random.default_rng(seed)
    s = 0.2 * rng.integers(0, int(length_nm / 0.2) + 1, n)
    phi = np.deg2rad(30.0 * rng.integers(0, 12, n))
    r = rng.choice([0.5, 0.7], n)
    r[:2] = (0.5, 0.7)
    types = rng.choice([int(SiteType.ASP), int(SiteType.GLU), int(SiteType.LYS),
                        int(SiteType.ARG), int(SiteType.OTHER)], n)
    hydro = rng.normal(0.0, 2.0, n)
    return make_molecule(s, phi, r, types, hydro=hydro, helix_length_nm=length_nm)


# ------------------------------------------------------------- temperature ---
def test_permittivity_and_bjerrum_length():
    # Malmberg & Maryott (1956): 87.74 at 0 C, 78.30 at 25 C, 74.15 at 37 C
    assert so.water_permittivity(0.0) == pytest.approx(87.740)
    assert so.water_permittivity(25.0) == pytest.approx(78.303, abs=1e-3)
    assert so.water_permittivity(37.0) == pytest.approx(74.152, abs=2e-3)
    assert so.bjerrum_length_nm(25.0) == pytest.approx(es.BJERRUM_NM)
    for t in (4.0, 37.0):
        expect = es.BJERRUM_NM * (78.303 * 298.15) / (so.water_permittivity(t) * (t + 273.15))
        assert so.bjerrum_length_nm(t) == pytest.approx(expect, rel=1e-4)
    # eps_r T is nearly constant: l_B moves by less than 4 % over 4-37 C
    assert abs(so.bjerrum_length_nm(4.0) / so.bjerrum_length_nm(37.0) - 1.0) < 0.04
    assert so.debye_length_nm(0.15) == pytest.approx(es.debye_length_nm(0.15))


def test_hydrophobic_scale_follows_the_engine_law():
    assert so.hydrophobic_scale(25.0) == pytest.approx(1.0)
    ref = (1 + 0.013 * (298.15 - 310.15)) / 298.15
    for t in (4.0, 37.0):
        tk = t + 273.15
        assert so.hydrophobic_scale(t) == pytest.approx((1 + 0.013 * (tk - 310.15)) / tk / ref)
    assert so.hydrophobic_scale(4.0) == pytest.approx(0.728, abs=1e-3)
    assert so.hydrophobic_scale(37.0) == pytest.approx(1.139, abs=1e-3)


def test_kadler_law_follows_the_measured_free_energy():
    """s(T) = g(T) / g(298.15 K), g = (dH - T dS) / RT with dH = 56 kcal/mol,
    dS = 0.220 kcal/(K mol) (Kadler et al. 1987); zero where dH = T dS."""
    def g(t_k):
        return (56.0 - 0.220 * t_k) / (1.987204e-3 * t_k)
    for t in (4.0, 37.0):
        assert so.hydrophobic_scale(t, "kadler1987") == pytest.approx(
            g(t + 273.15) / g(298.15), rel=1e-12)
    assert so.hydrophobic_scale(25.0, "kadler1987") == pytest.approx(1.0)
    assert so.hydrophobic_scale(56.0 / 0.220 - 273.15, "kadler1987") == pytest.approx(0.0,
                                                                                       abs=1e-12)
    # steeper than the engine's law in the cold
    assert so.hydrophobic_scale(4.0, "kadler1987") < so.hydrophobic_scale(4.0)
    with pytest.raises(ValueError):
        so.hydrophobic_scale(25.0, "nonsense")


# --------------------------------------------------------------- the integral ---
def test_radial_rule_is_exact_for_exponentials_and_linear_functions():
    """int x g(x) dx, closed forms: for g = A e^{-k x},
    A [1/k^2 - e^{-k X} (X/k + 1/k^2)] from 0 to X; for g linear, a cubic."""
    x = np.array([0.0, 0.3, 1.0, 2.5, 6.0])
    for a, k in ((3.0, 1.7), (-2.0, 0.4)):
        exact = a * (1 / k ** 2 - np.exp(-k * 6.0) * (6.0 / k + 1 / k ** 2))
        assert so.radial_integral(x, a * np.exp(-k * x)) == pytest.approx(exact, rel=1e-12)
    assert so.radial_integral(x, 2.0 + 0 * x) == pytest.approx(36.0)
    # one interval with a sign change: int_0^2 x (x - 1) dx = 8/3 - 2
    assert so.radial_integral([0.0, 2.0], [-1.0, 1.0]) == pytest.approx(2.0 / 3.0)


def test_no_interaction_gives_the_hard_core_value_exactly():
    mol = toy_molecule(1)
    pv = so.PairVirial(mol, so.VirialSettings(eps_hydrophobic=0.0, **FAST))
    zero = lambda m, env: np.zeros(len(m))                       # noqa: E731
    rows = pv.virial(mol, mol, [so.Conditions(7.0, 0.01)], terms=("full", "el", "net"),
                     charge_model=zero)
    for r in rows:
        assert r["B2_nm3"] == pytest.approx(np.pi * pv.settings.contact_nm ** 2 * 20.0,
                                            rel=1e-12)
        assert r["B2_star"] == pytest.approx(1.0, abs=1e-12)


def test_electrostatics_vanishes_at_very_high_salt():
    mol = toy_molecule(2)
    pv = so.PairVirial(mol, so.VirialSettings(**FAST))
    rows = {r["term"]: r for r in pv.virial(mol, mol, [so.Conditions(7.0, 50.0)],
                                            terms=("full", "el", "hy"))}
    assert rows["el"]["B2_star"] == pytest.approx(1.0, abs=1e-3)
    assert rows["full"]["B2_nm3"] == pytest.approx(rows["hy"]["B2_nm3"], rel=1e-3)
    # the hydrophobic term alone does not see pH or salt
    hy2 = [r for r in pv.virial(mol, mol, [so.Conditions(3.0, 0.001)], terms=("hy",))][0]
    assert hy2["B2_nm3"] == pytest.approx(rows["hy"]["B2_nm3"], rel=1e-9)


def test_two_point_charges_against_direct_quadrature():
    """One charge on the axis of each molecule: E depends only on
    r^2 = rho^2 + Delta^2, so the virial integral is a plain double integral
    that is done here on a fine grid, independently of the FFT scan."""
    length = 10.0
    mol = make_molecule([0.0], [0.0], [0.0], [int(SiteType.LYS)], helix_length_nm=length)
    settings = so.VirialSettings(eps_hydrophobic=0.0, n_shell_charge=1, n_phi=4,
                                 in_register_exclusion_nm=0.0)
    pv = so.PairVirial(mol, settings)
    for q, ionic in ((5.0, 0.01), (-1.0, 0.01), (3.0, 0.15)):
        def model(m, env, q=q):
            return np.array([q]) if m is mol else np.array([abs(q)])
        mol_b = make_molecule([0.0], [0.0], [0.0], [int(SiteType.LYS)],
                              helix_length_nm=length)
        cond = so.Conditions(7.0, ionic)
        row = [r for r in pv.virial(mol, mol_b, [cond], terms=("el",), charge_model=model)][0]
        lam = cond.debye_length_nm
        rho = settings.contact_nm + np.linspace(0.0, settings.reach_debye * lam, 6001)
        dl = np.linspace(-80.0, 80.0, 16001)
        r_eff = np.sqrt(rho[:, None] ** 2 + dl[None, :] ** 2 + settings.core_nm ** 2)
        e = q * abs(q) * es.BJERRUM_NM * np.exp(-r_eff / lam) / r_eff
        g = np.trapezoid(np.expm1(-e), dl, axis=1)
        tail = -np.pi * np.trapezoid(rho * g, rho)
        b2_hc = np.pi * settings.contact_nm ** 2 * length
        assert row["B2_hc_nm3"] == pytest.approx(b2_hc)
        # the Delta sum is a Riemann sum at 0.2 nm, the rho rule on ~45 nodes
        assert row["B2_nm3"] - b2_hc == pytest.approx(tail, rel=0.01)
        assert row["B2_parallel_nm3"] == pytest.approx(row["B2_antiparallel_nm3"], rel=1e-4)


def test_exchange_of_the_two_molecules_leaves_b2_unchanged():
    a, b = toy_molecule(3), toy_molecule(4)
    pv = so.PairVirial(a, so.VirialSettings(in_register_exclusion_nm=0.0, **FAST))
    conds = [so.Conditions(4.2, 0.01), so.Conditions(7.0, 0.15, 37.0)]
    ab = pv.virial(a, b, conds, terms=("full", "net_hy"))
    ba = pv.virial(b, a, conds, terms=("full", "net_hy"))
    for x, y in zip(ab, ba):
        assert x["term"] == y["term"]
        for o in ("parallel", "antiparallel"):
            assert x[f"B2_{o}_nm3"] == pytest.approx(y[f"B2_{o}_nm3"], rel=1e-9)
        assert x["F_best_kT"] == pytest.approx(y["F_best_kT"], abs=1e-9)
        assert x["E_min_kT"] == pytest.approx(y["E_min_kT"], abs=1e-9)


def test_temperature_scales_the_hydrophobic_minimum_linearly():
    mol = toy_molecule(5)
    pv = so.PairVirial(mol, so.VirialSettings(**FAST))
    rows = pv.virial(mol, mol, [so.Conditions(7.0, 0.15, t) for t in (4.0, 25.0, 37.0)],
                     terms=("hy",))
    e = {r["temperature_C"]: r["E_min_kT"] for r in rows}
    for t in (4.0, 37.0):
        assert e[t] == pytest.approx(so.hydrophobic_scale(t) * e[25.0], rel=1e-9)
    b2 = [r["B2_nm3"] for r in rows]
    assert b2[0] > b2[1] > b2[2]           # weaker attraction in the cold


def test_in_register_window_only_removes_parallel_staggers_near_zero():
    mol = toy_molecule(6)
    cond = [so.Conditions(7.0, 0.15)]
    open_ = so.PairVirial(mol, so.VirialSettings(in_register_exclusion_nm=0.0, **FAST))
    shut = so.PairVirial(mol, so.VirialSettings(in_register_exclusion_nm=5.0, **FAST))
    a = open_.virial(mol, mol, cond, terms=("hy",))[0]
    b = shut.virial(mol, mol, cond, terms=("hy",))[0]
    assert a["B2_antiparallel_nm3"] == pytest.approx(b["B2_antiparallel_nm3"], rel=1e-12)
    # the centred hydrophobic product makes a molecule bind its unshifted copy
    assert a["F_best_parallel_stagger_nm"] == pytest.approx(0.0, abs=0.2)
    assert abs(b["F_best_parallel_stagger_nm"]) >= 5.0 - 1e-9
    assert b["B2_parallel_nm3"] > a["B2_parallel_nm3"]


def test_uniform_net_charge_keeps_the_net_charge():
    mol = toy_molecule(7)
    q = es.henderson_hasselbalch(mol, es.Environment(7.0, 0.15))
    w, smeared = so.uniform_net_charge(mol, q)
    assert w.sum() == pytest.approx(q.sum())
    assert np.all(w[mol.native_type == int(SiteType.OTHER)] == 0.0)
    assert np.all(smeared == (mol.native_type != int(SiteType.OTHER)))
    assert np.ptp(w[smeared]) == 0.0


# ---------------------------------------------------------- interpretation ---
def test_dimer_fraction():
    assert so.dimer_fraction(100.0, 1e-3) == 0.0          # repulsive: no dimers
    # K c = 1: 2 m^2 + m - 1 = 0 in units of c, m = c / 2
    assert so.dimer_fraction(-1000.0, 1e-3) == pytest.approx(0.5)
    assert so.per_nm3(1.0, 6.02214076e23 * 1e-24 * 1e3) == pytest.approx(1e-3)


def test_solution_pH_of_a_strong_and_of_a_weak_acid():
    none = np.zeros(co.N_TYPES)
    c, n = 1e-4, 10.0               # 1 mM of strong acid
    h = 0.5 * (c * n + np.sqrt((c * n) ** 2 + 4e-14))
    assert so.solution_pH(none, c, acid_per_molecule=n) == pytest.approx(-np.log10(h), abs=1e-9)
    # 1 mM acetic acid, pKa 4.76: h^2 + Ka h - Ka C = 0 (water negligible)
    ka = 10 ** -4.76
    h = 0.5 * (-ka + np.sqrt(ka * ka + 4 * ka * 1e-3))
    assert so.solution_pH(none, 1e-4, weak_acid_per_molecule=10.0) == pytest.approx(
        -np.log10(h), abs=1e-4)
    assert so.solution_pH(none, 1e-4) == pytest.approx(7.0, abs=1e-9)


def test_counter_anions_invert_the_solution_pH():
    native = native_molecule()
    counts = co.type_counts(native)
    for n_s in (0.0, 40.0, 150.0):
        pH = so.solution_pH(counts, 1.7e-5, acid_per_molecule=n_s)
        assert so.counter_anions_per_molecule(counts, 1.7e-5, pH) == pytest.approx(n_s, abs=1e-6)
    # isoionic native collagen in water sits near, not at, its isoelectric point
    pI = float(co.isoelectric_point_from_counts(counts))
    assert abs(so.solution_pH(counts, 1.7e-5) - pI) < 1.0
