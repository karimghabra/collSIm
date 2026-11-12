"""Association constants: the integrator against an independent quadrature,
the calibration gate against the targets, and limiting laws."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import calibrate as cal  # noqa: E402
from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.params import PER_NM3_PER_MOLAR, debye_kappa, load_params  # noqa: E402

O_CARB = geo.TYPE_INDEX["O_carboxylate"]


@pytest.fixture(scope="module")
def params():
    return load_params()


def midpoints(a, b, n):
    h = (b - a) / n
    return a + h * (np.arange(n) + 0.5), h


def test_integrator_matches_two_dimensional_quadrature_for_a_one_site_ion(params):
    """A cut-down calcium with a single coordination site and its whole charge
    on the centre, next to one carboxylate oxygen. By symmetry the weight
    depends only on the distance r and the angle theta between the site axis
    and the direction to the oxygen, so

        K_c = N_A / z_free * int 4 pi r^2 dr int (sin theta / 2) d theta  w_c(r, theta)

    which is evaluated here on a midpoint grid from the formulas of the model
    written out independently of the kernels.
    """
    T = params.tables()
    ca = params.species_index("Ca")
    nsite = T.nsite.copy()
    nsite[ca] = 1
    dirs = T.dirs.copy()
    dirs[ca] = 0.0
    dirs[ca, 0] = [0.0, 0.0, 1.0]
    q_centre, q_sat = T.q_centre.copy(), T.q_sat.copy()
    q_centre[ca], q_sat[ca] = 2.0, 0.0
    T1 = T._replace(nsite=nsite, dirs=dirs, q_centre=q_centre, q_sat=q_sat)

    I = 0.05
    lb = params.bjerrum_length
    kappa = debye_kappa(I, lb)
    d = float(T.d_site[ca])
    w_r = float(T.w_r[ca])
    cos_flat, cos_max = float(T.cos_flat[ca]), float(T.cos_max[ca])
    d_out, w_out = float(T.d_out[ca]), float(T.w_out[ca])
    eps_in, eps_out = float(T.eps_in[ca, O_CARB]), float(T.eps_out[ca, O_CARB])
    dg = float(T.dg_des[ca])
    r_o = float(T.R_type[O_CARB])
    r_core = float(T.R_centre[ca]) + r_o
    r_water = float(T.R_water[ca]) + r_o
    r_contact = float(T.r_contact[ca])
    q_atom = -0.5

    def wca(r, rmin):
        s = rmin / 2 ** (1 / 6)
        rr = np.maximum(r, 0.3 * rmin)
        return np.where(r < rmin, 4 * ((s / rr) ** 12 - (s / rr) ** 6) + 1.0, 0.0)

    def bump(x):
        return np.where(np.abs(x) < 1, (1 - x * x) ** 2, 0.0)

    def weights(r, th):
        cos_t = np.cos(th)
        u0 = lb * 2.0 * q_atom * np.exp(-kappa * r) / r + wca(r, r_core)
        dist_water = np.sqrt(r * r + d * d - 2 * r * d * cos_t)
        u_w = wca(dist_water, r_water) - eps_out * bump((r - d_out) / w_out)
        y = np.clip((cos_flat - cos_t) / (cos_flat - cos_max), 0.0, 1.0)
        ang = np.where(cos_t <= cos_max, 0.0, (1 - y * y) ** 2)
        well = bump((r - d) / w_r) * ang
        u_l = dg - eps_in * well
        return np.exp(-(u0 + u_w)), np.exp(-(u0 + u_l)), well > 0

    z_free = 1.0 + math.exp(-dg)
    # inner: the L state with the oxygen inside the well
    r, hr = midpoints(d - w_r, d + w_r, 1500)
    th, ht = midpoints(0.0, math.acos(cos_max), 1500)
    R, TH = np.meshgrid(r, th, indexing="ij")
    _, w_l, held = weights(R, TH)
    k_inner = np.sum(4 * math.pi * R ** 2 * 0.5 * np.sin(TH) * w_l * held) * hr * ht
    # outer: everything else with the oxygen within r_contact
    r, hr = midpoints(0.03, r_contact, 2500)
    th, ht = midpoints(0.0, math.pi, 2500)
    R, TH = np.meshgrid(r, th, indexing="ij")
    w_w, w_l, held = weights(R, TH)
    k_outer = np.sum(4 * math.pi * R ** 2 * 0.5 * np.sin(TH) * (w_w + w_l * ~held)) * hr * ht
    k_inner *= PER_NM3_PER_MOLAR / z_free
    k_outer *= PER_NM3_PER_MOLAR / z_free

    lig = geo.LigandSet(pos=[[0.0, 0.0, 0.0]], type=[O_CARB], charge=[q_atom], mol=[0])
    got = cal.association_constants(params, "Ca", lig, I, np.random.default_rng(314),
                                    n_samples=3_000_000, tables=T1)
    assert got.K_multi == 0.0                       # one site cannot be multidentate
    assert got.err["mono"] < 0.03 * k_inner and got.err["outer"] < 0.01 * k_outer
    assert abs(got.K_mono - k_inner) < 5 * got.err["mono"] + 0.005 * k_inner
    assert abs(got.K_outer - k_outer) < 5 * got.err["outer"] + 0.005 * k_outer


def test_integrator_error_bar_covers_the_scatter_between_seeds(params):
    lig = cal.acetate()
    runs = [cal.association_constants(params, "Ca", lig, 0.1, np.random.default_rng(s), n_samples=300_000)
            for s in range(6)]
    totals = np.array([r.K_total for r in runs])
    quoted = np.mean([math.sqrt(sum(v ** 2 for v in r.err.values())) for r in runs])
    assert totals.std(ddof=1) < 3 * quoted
    assert totals.std(ddof=1) > quoted / 3


# ------------------------------------------------------------ calibration gate
@pytest.fixture(scope="module")
def gate(params):
    """Every calibration pair evaluated with a seed the fit never saw."""
    out = {}
    for spec in cal.FITS:
        out[spec.name] = cal.evaluate(params, spec, seed=987654, n_samples=1_500_000)
    return out


@pytest.mark.parametrize("name", [s.name for s in cal.FITS])
def test_total_association_constant_reproduces_its_target(params, gate, name):
    K_total, _, _ = cal.target_values(params, name)
    a = gate[name]
    # 0.03 in log10 K, well inside the spread between literature evaluations
    assert abs(math.log10(a.K_total) - math.log10(K_total)) < 0.03


@pytest.mark.parametrize("name", ["Ca_acetate", "Mg_acetate", "Ca_sulfate", "Mg_sulfate"])
def test_inner_sphere_constant_reproduces_its_target(params, gate, name):
    _, K_inner, _ = cal.target_values(params, name)
    a = gate[name]
    err = math.hypot(a.err["mono"], a.err["multi"])
    assert abs(a.K_inner - K_inner) < 5 * err + 0.08 * K_inner


def test_acetate_targets_are_the_cited_numbers(params):
    """Guards the transcription of the targets: NIST SRD 46 log K (I = 0, 25 C)
    1.18 for CaOAc+ and 1.26 for MgOAc+; Raman-MCR contact-pair constants
    3.6 and 3.9 /M (Mendes de Oliveira et al. 2020)."""
    assert cal.target_values(params, "Ca_acetate") == (pytest.approx(10 ** 1.18), 3.6, 0.0)
    assert cal.target_values(params, "Mg_acetate") == (pytest.approx(10 ** 1.26), 3.9, 0.0)
    assert params.value("targets.Ca_acetate.bidentate_fraction") == 0.27


def test_inner_sphere_share_with_acetate_is_what_the_data_imply(params, gate):
    """Contact pairs are 3.6 / 15.1 = 24% (Ca) and 3.9 / 18.2 = 21% (Mg) of all
    pairs: both ions are mostly outer-sphere with a single acetate, and they
    do not differ by more than the measurement errors. Affinity does not
    distinguish the two ions; geometry does (next test)."""
    assert gate["Ca_acetate"].inner_fraction == pytest.approx(0.24, abs=0.04)
    assert gate["Mg_acetate"].inner_fraction == pytest.approx(0.21, abs=0.04)


def test_magnesium_pays_more_to_shed_water_and_gains_more_on_contact(params):
    """The net inner-sphere gain eps_in - dG_des is similar for the two ions
    (the data force that), so the larger desolvation cost of Mg2+ is offset by
    a larger contact energy rather than showing up as a population."""
    net = {}
    for name in ("Mg", "Ca"):
        net[name] = (float(params.value(f"contact.{name}.O_carboxylate.eps_in"))
                     - float(params.value(f"species.{name}.dg_des")))
    assert float(params.value("contact.Mg.O_carboxylate.eps_in")) > \
        float(params.value("contact.Ca.O_carboxylate.eps_in")) + 3.0
    assert abs(net["Mg"] - net["Ca"]) < 4.0


def test_bidentate_carboxylate_is_a_calcium_feature(params, gate):
    """Mg2+ binds carboxylate 'almost exclusively in a monodentate way' (0% of
    Asp and Glu contacts bidentate in protein sites; > 25 kJ/mol above
    monodentate in ab initio MD), while 22-32% of Ca2+-carboxylate contacts
    are bidentate (sources in params.toml). Mg comes out monodentate without
    fitting; for Ca the angular slack is fitted to 27% and must stay inside
    the measured 20-35%. Denticity is the number of oxygens held in open
    sites, whether on one calcium site or on two."""
    mg, ca = gate["Mg_acetate"], gate["Ca_acetate"]
    assert mg.K_multi / mg.K_inner < 0.01
    share = ca.K_multi / ca.K_inner
    err = share * math.hypot(ca.err["multi"] / ca.K_multi, ca.err["mono"] / ca.K_mono)
    assert 0.20 - 3 * err < share < 0.35 + 3 * err


def test_sulfate_binds_arginine_more_than_lysine(params, gate):
    assert gate["SO4_guanidinium"].K_total > gate["SO4_imidazolium"].K_total > gate["SO4_ammonium"].K_total
    # weak: no stronger than the 5-30 /M estimated for one contact at I = 0
    assert gate["SO4_guanidinium"].K_total < 30.0


def test_guanidinium_can_grip_sulfate_with_two_hydrogen_bonds(params, gate):
    """The direct (hydrogen-bonded) pairs must be a real share of the total,
    otherwise the fitted hydrogen-bond energy does nothing and sulfate would
    be a featureless point charge to arginine."""
    a = gate["SO4_guanidinium"]
    assert a.K_direct / a.K_total > 0.2
    assert float(params.value("contact.SO4.N_guanidinium.eps_hb")) > 0.0


# ------------------------------------------------------------- limiting laws
def test_ionic_strength_dependence_follows_the_debye_hueckel_limiting_law(params):
    """For M(2+) + L(-) = ML(+) the limiting law gives
    log10 K(I) - log10 K(0) = -A (z_M^2 + z_L^2 - z_ML^2) sqrt(I) = -4 A sqrt(I)
    with A = 0.509 (water, 25 C). At I = 0.1 mM that is -0.0204. The screened
    Coulomb term must reproduce it without any fitting. (The law is a limit:
    the model departs from it at first order in kappa times the size of the
    pair, 0.6% here and 2-3% already at 1 mM; lB = 0.71 instead of 0.715 nm
    accounts for another 0.7%.)"""
    spec = cal.FITS[0]
    k0 = cal.evaluate(params, spec, seed=5, n_samples=400_000, ionic_strength=0.0).K_total
    k1 = cal.evaluate(params, spec, seed=5, n_samples=400_000, ionic_strength=1e-4).K_total
    assert math.log10(k1 / k0) == pytest.approx(-4 * 0.509 * math.sqrt(1e-4), rel=0.02)


def test_binding_weakens_monotonically_with_ionic_strength(params):
    spec = cal.FITS[1]
    ks = [cal.evaluate(params, spec, seed=6, n_samples=200_000, ionic_strength=I).K_total
          for I in (0.0, 0.01, 0.1, 0.5)]
    assert all(a > b for a, b in zip(ks, ks[1:]))
    # at I = 0.1 M measured log K for Ca and Mg carboxylates is near 0.5-0.75
    assert 0.3 < math.log10(ks[2]) < 0.95


def test_a_protonated_carboxyl_group_offers_no_partner(params):
    lig = geo.carboxylate([0, 0, 0], [1, 0, 0], protonated=True)
    with pytest.raises(ValueError):
        cal.association_constants(params, "Ca", lig, 0.0, np.random.default_rng(1), n_samples=1000)


def test_calibration_ligands_are_neutral_or_singly_charged_as_intended(params):
    assert cal.acetate().charge.sum() == pytest.approx(-1.0)
    assert cal.methylguanidinium().charge.sum() == pytest.approx(1.0)
    assert cal.methylammonium().charge.sum() == pytest.approx(1.0)
    assert cal.imidazolium_ring().charge.sum() == pytest.approx(1.0)
    assert cal.sulfate_as_ligand(params).charge.sum() == pytest.approx(-2.0)
    ring = cal.imidazolium_ring()
    n = ring.pos[ring.type == geo.TYPE_INDEX["N_imidazolium"]]
    # the 1,3 nitrogens of a regular pentagon of circumradius 0.115 nm
    assert np.linalg.norm(n[0] - n[1]) == pytest.approx(2 * 0.115 * math.sin(math.radians(72)), rel=1e-6)


def test_succinate_conformers_have_the_textbook_chain_geometry(params):
    """All-trans four-carbon chain with bond b and bond angle theta has its end
    carbons b * sqrt(1 + 8 sin^2(theta/2)) apart: 0.390 nm for b = 0.153 nm,
    theta = 112 deg. The gauche conformer brings them closer."""
    inert = geo.TYPE_INDEX["inert"]
    ext, gau = cal.succinate(180.0), cal.succinate(65.0)
    for lig in (ext, gau):
        assert lig.charge.sum() == pytest.approx(-2.0)
        assert sorted(set(lig.group.tolist())) == [0, 1]
        c = lig.pos[lig.type == inert][[0, 2, 3, 1]]          # stored as C1, C4, C2, C3
        assert np.allclose(np.linalg.norm(np.diff(c, axis=0), axis=1), 0.153)
    c = ext.pos[ext.type == inert][[0, 2, 3, 1]]
    expected = 0.153 * math.sqrt(1 + 8 * math.sin(math.radians(56.0)) ** 2)
    assert np.linalg.norm(c[0] - c[3]) == pytest.approx(expected, rel=1e-9)
    cg = gau.pos[gau.type == inert][[0, 2, 3, 1]]
    assert np.linalg.norm(cg[0] - cg[3]) < np.linalg.norm(c[0] - c[3]) - 0.05
    # each carboxylate keeps its own geometry
    o = ext.pos[ext.type == O_CARB]
    assert np.linalg.norm(o[0] - o[1]) == pytest.approx(2 * 0.125 * math.sin(math.radians(62.5)))


def test_constants_the_model_was_not_fitted_to_stay_within_the_documented_residuals(params):
    """Measured numbers held back from the fit (params.toml [checks]). The
    model is not expected to hit them; the documentation states by how much it
    misses, and this pins those statements.

    At I = 0.1 the screened Coulomb term alone carries the change from I = 0:
    the 2:1 pairs come out 0.1-0.25 log units too strong and the 2:2 pairs
    0.15-0.25 too weak. Succinate (two carboxylates on one cation) is over-bound
    by 0.2-0.7 log units in either rigid conformer."""
    rows = {c["check"]: c for c in cal.checks(params, seed=4242, n_samples=400_000)}
    for name in ("Ca_acetate", "Mg_acetate"):
        c = rows[f"{name}.log_K_total_I0p1"]
        assert 0.0 < c["log_K_total"] - c["measured"] < 0.3
    for name in ("Ca_sulfate", "Mg_sulfate"):
        c = rows[f"{name}.log_K_total_I0p1"]
        assert -0.3 < c["log_K_total"] - c["measured"] < 0.0
    for name in ("Ca_succinate", "Mg_succinate"):
        c = rows[f"{name}.log_K_total"]
        for conformer in ("gauche", "extended"):
            assert 0.0 < c[f"log_K_total_{conformer}"] - c["measured"] < 0.8
        assert c["log_K_total_gauche"] > c["log_K_total_extended"]
