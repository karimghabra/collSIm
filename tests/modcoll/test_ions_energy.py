"""Energy terms against hand-worked values and textbook numbers."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.energy import (MAX_SITES, bump, e_cation_atom, e_internal, e_ion_ion,  # noqa: E402
                                 e_oxyanion_atom, inner_weight, make_elec, rotate_dirs, wca, yukawa)
from modcoll.ions.params import debye_kappa, load_params  # noqa: E402

O_CARB = geo.TYPE_INDEX["O_carboxylate"]
N_GUA = geo.TYPE_INDEX["N_guanidinium"]


@pytest.fixture(scope="module")
def params():
    return load_params()


def frame(T, sp, q=(1.0, 0.0, 0.0, 0.0)):
    U = np.zeros((MAX_SITES, 3))
    rotate_dirs(T, sp, np.asarray(q, float), U)
    return U


# ---------------------------------------------------------------- elementary
def test_bump_values():
    assert bump(0.0) == 1.0
    assert bump(0.25) == pytest.approx(0.5625)   # x = 0.5: (1 - 0.25)^2
    assert bump(1.0) == 0.0
    assert bump(4.0) == 0.0


def test_wca_landmarks():
    rmin, eps = 0.3, 1.0
    sigma = rmin / 2 ** (1 / 6)
    assert wca(rmin ** 2, rmin, eps) == 0.0
    assert wca((1.01 * rmin) ** 2, rmin, eps) == 0.0
    assert wca(sigma ** 2, rmin, eps) == pytest.approx(eps)      # u(sigma) = eps
    # independent evaluation of 4 eps [(s/r)^12 - (s/r)^6] + eps at r = 0.8 rmin
    r = 0.8 * rmin
    assert wca(r * r, rmin, eps) == pytest.approx(4 * ((sigma / r) ** 12 - (sigma / r) ** 6) + 1)
    # continuous at rmin, finite at overlap
    assert wca((rmin * (1 - 1e-6)) ** 2, rmin, eps) < 1e-9
    assert np.isfinite(wca(0.0, rmin, eps))


def test_two_unit_charges_one_bjerrum_length_apart_have_energy_kT():
    # the definition of the Bjerrum length
    lb = 0.71
    assert yukawa(lb * lb, 1.0, lb, 0.0, math.inf, 0.0) == pytest.approx(1.0)
    assert yukawa(lb * lb, -2.0, lb, 0.0, math.inf, 0.0) == pytest.approx(-2.0)


def test_yukawa_screening_and_shift():
    lb, kappa = 0.71, 1.25
    r = 1.0 / kappa
    assert yukawa(r * r, 1.0, lb, kappa, math.inf, 0.0) == pytest.approx(lb * kappa * math.exp(-1))
    E = make_elec(lb, kappa, 2.0)
    assert yukawa(2.0 ** 2, 1.0, E.lb, E.kappa, E.rcut, E.shift) == 0.0
    assert yukawa(1.9999 ** 2, 1.0, E.lb, E.kappa, E.rcut, E.shift) == pytest.approx(0.0, abs=1e-4)
    assert yukawa(3.0 ** 2, 1.0, E.lb, E.kappa, E.rcut, E.shift) == 0.0


def test_debye_length_matches_the_textbook_number():
    # Israelachvili, Intermolecular and Surface Forces: 1/kappa = 0.304 / sqrt(I) nm
    # for a 1:1 electrolyte in water at 25 C (I in mol/L).
    for I in (0.01, 0.1, 0.15):
        assert 1.0 / debye_kappa(I, 0.71) == pytest.approx(0.304 / math.sqrt(I), rel=0.01)


def test_zero_ionic_strength_means_no_screening(params):
    assert params.kappa(0.0) == 0.0


# ------------------------------------------------------------- inner well
def test_inner_weight_radial_and_angular_profile(params):
    T = params.tables()
    sp = params.species_index("Mg")
    U = frame(T, sp)
    d = float(params.value("species.Mg.d_MO"))
    w_r = float(params.value("species.Mg.well_radial_halfwidth"))
    k = 0  # site along +x
    assert inner_weight(T, sp, U, k, d, 0.0, 0.0, d) == pytest.approx(1.0)
    # half-way out radially: B(0.5) = 0.5625
    r = d + 0.5 * w_r
    assert inner_weight(T, sp, U, k, r, 0.0, 0.0, r) == pytest.approx(0.5625)
    assert inner_weight(T, sp, U, k, d + w_r, 0.0, 0.0, d + w_r) == 0.0
    # half-way out in angle, with theta_flat = 0:
    # y = (1 - cos(theta)) / (1 - cos(theta_max)), weight = (1 - y^2)^2
    th_max = float(params.value("species.Mg.well_angular_halfwidth_deg"))
    assert float(params.value("species.Mg.well_angular_flat_deg")) == 0.0
    th = math.radians(0.5 * th_max)
    y = (1 - math.cos(th)) / (1 - math.cos(math.radians(th_max)))
    got = inner_weight(T, sp, U, k, d * math.cos(th), d * math.sin(th), 0.0, d)
    assert got == pytest.approx((1 - y * y) ** 2)
    assert 0.8 < got < 0.95                      # y is about 1/4: a flat-topped well
    th = math.radians(th_max + 0.5)
    assert inner_weight(T, sp, U, k, d * math.cos(th), d * math.sin(th), 0.0, d) == 0.0


def test_calcium_well_is_flat_inside_its_angular_slack(params):
    T = params.tables()
    sp = params.species_index("Ca")
    U = frame(T, sp)
    d = float(params.value("species.Ca.d_MO"))
    flat = float(params.value("species.Ca.well_angular_flat_deg"))
    k = 0  # site along +z
    for deg in (0.0, 0.5 * flat, 0.99 * flat):
        th = math.radians(deg)
        assert inner_weight(T, sp, U, k, d * math.sin(th), 0.0, d * math.cos(th), d) == pytest.approx(1.0)
    th = math.radians(flat + 5.0)
    w = inner_weight(T, sp, U, k, d * math.sin(th), 0.0, d * math.cos(th), d)
    assert 0.0 < w < 1.0


def test_calcium_tolerates_more_angular_error_than_magnesium(params):
    T = params.tables()
    out = {}
    for name in ("Mg", "Ca"):
        sp = params.species_index(name)
        U = frame(T, sp)
        d = float(T.d_site[sp])
        u = U[0]
        perp = np.cross(u, [0.3, 0.5, 0.8])
        perp /= np.linalg.norm(perp)
        th = math.radians(12.0)
        v = d * (math.cos(th) * u + math.sin(th) * perp)
        out[name] = inner_weight(T, sp, U, 0, v[0], v[1], v[2], d)
    assert out["Ca"] > out["Mg"]


# ------------------------------------------------- cation-atom, hand-worked
def test_calcium_carboxylate_oxygen_in_an_open_site_by_hand(params):
    """One carboxylate oxygen (-1/2 e) exactly on the +z site of Ca2+, that
    site in state L, I = 0. Expected energy, term by term:
      Coulomb   lB * (+2) * (-1/2) / d, the whole charge being on the centre
      core      zero: the core starts exactly at the bond length
      waters    zero: the nearest water is d * sqrt(2) = 0.337 nm away, beyond
                R_w + R_O = 0.275 nm
      inner     -eps_in (weight 1)
      outer     zero: r = d is outside the outer shell
      internal  +dG_des for the one emptied site
    """
    T = params.tables()
    E = make_elec(0.71, 0.0)
    sp = params.species_index("Ca")
    U = frame(T, sp)
    d = float(params.value("species.Ca.d_MO"))
    eps_in = float(params.value("contact.Ca.O_carboxylate.eps_in"))
    dg = float(params.value("species.Ca.dg_des"))
    state = np.zeros(MAX_SITES, dtype=np.uint8)
    state[0] = 1
    expected = 0.71 * 2.0 * (-0.5) / d - eps_in + dg
    got = e_cation_atom(T, E, sp, U, state, 6, 0.0, 0.0, d, O_CARB, -0.5) + e_internal(T, sp, state)
    assert got == pytest.approx(expected, rel=1e-12)


def test_dummy_charge_layout_by_hand_and_how_little_it_matters(params):
    """With centre_charge = 0 the charge sits on seven dummy points 0.09 nm
    out (the Saxena-Sept layout). The Coulomb energy with an oxygen on the +z
    site is then the sum over the seven points, written out here; it differs
    from the point-charge value by a few per cent, and between an on-site and
    a between-sites direction by a fraction of kT."""
    p = params.with_overrides({"species.Ca.centre_charge": 0.0})
    T = p.tables()
    E = make_elec(0.71, 0.0)
    sp = p.species_index("Ca")
    assert T.q_centre[sp] == 0.0 and T.q_sat[sp] == pytest.approx(2.0 / 7.0)
    U = frame(T, sp)
    d = float(T.d_site[sp])
    state = np.zeros(MAX_SITES, dtype=np.uint8)
    state[:7] = 1                                   # all sites open: no waters, no outer term
    dummies = 0.09 * geo.pentagonal_bipyramid()

    def by_hand(atom):
        return sum(0.71 * (2.0 / 7.0) * (-0.5) / np.linalg.norm(atom - q) for q in dummies)

    # isolate the Coulomb part by differencing against an uncharged atom
    def coulomb(atom):
        return (e_cation_atom(T, E, sp, U, state, 0, *atom, O_CARB, -0.5)
                - e_cation_atom(T, E, sp, U, state, 0, *atom, O_CARB, 0.0))

    on_site = np.array([0.0, 0.0, d])
    th, ph = math.radians(50.0), math.radians(36.0)
    between = d * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    for atom in (on_site, between):
        assert coulomb(atom) == pytest.approx(by_hand(atom), rel=1e-12)
    point = 0.71 * 2.0 * -0.5 / d
    assert by_hand(on_site) == pytest.approx(point, rel=0.04)
    assert abs(by_hand(on_site) - by_hand(between)) < 0.3


def test_a_hydrated_site_keeps_the_ligand_out(params):
    T = params.tables()
    E = make_elec(0.71, 0.0)
    sp = params.species_index("Ca")
    U = frame(T, sp)
    d = float(T.d_site[sp])
    state = np.zeros(MAX_SITES, dtype=np.uint8)   # all W
    e = e_cation_atom(T, E, sp, U, state, 7, 0.0, 0.0, d, O_CARB, -0.5)
    assert e > 1e4   # the atom sits on top of the water


def test_outer_sphere_term_scales_with_the_hydrated_fraction(params):
    """An uncharged partner atom far off every site axis, at the centre of the
    outer shell: only the outer term acts, and it is -eps_out * n_W / n."""
    p = params.with_overrides({"contact.Ca.O_carboxylate.eps_out": 1.7})
    T = p.tables()
    E = make_elec(0.71, 0.0)
    sp = p.species_index("Ca")
    U = frame(T, sp)
    r = float(T.d_out[sp])
    # a direction between the +z site and two equatorial sites: every water is
    # then farther from the atom than R_w + R_O = 0.275 nm and does not push
    th, ph = math.radians(50.0), math.radians(36.0)
    u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
    waters = float(T.d_site[sp]) * U[:7]
    assert np.linalg.norm(waters - r * u, axis=1).min() > 0.275
    state = np.zeros(MAX_SITES, dtype=np.uint8)
    e7 = e_cation_atom(T, E, sp, U, state, 7, *(r * u), O_CARB, 0.0)
    assert e7 == pytest.approx(-1.7)
    state[3] = 1   # an equatorial site empties; nothing sits in it
    e6 = e_cation_atom(T, E, sp, U, state, 6, *(r * u), O_CARB, 0.0)
    assert e6 == pytest.approx(-1.7 * 6 / 7)


def test_energy_is_linear_in_the_site_states(params):
    """U(s) = U(all W) + sum over L sites of [U(only k is L) - U(all W)].
    The exact state sum in the calibration rests on this."""
    T = params.tables()
    E = make_elec(0.71, 0.3)
    rng = np.random.default_rng(42)
    lig = geo.carboxylate([0, 0, 0], [1, 0, 0])
    for name in ("Mg", "Ca"):
        sp = params.species_index(name)
        n = int(T.nsite[sp])
        for _ in range(20):
            U = frame(T, sp, geo.random_quaternion(rng))
            r = rng.normal(size=3)
            r *= rng.uniform(0.25, 0.6) / np.linalg.norm(r)

            def energy(state):
                nw = n - int(state[:n].sum())
                e = e_internal(T, sp, state)
                for x, t, q in zip(lig.pos, lig.type, lig.charge):
                    e += e_cation_atom(T, E, sp, U, state, nw, *(x - r), t, q)
                return e

            all_w = np.zeros(MAX_SITES, dtype=np.uint8)
            e0 = energy(all_w)
            single = []
            for k in range(n):
                s = all_w.copy()
                s[k] = 1
                single.append(energy(s) - e0)
            state = np.zeros(MAX_SITES, dtype=np.uint8)
            state[:n] = rng.integers(0, 2, size=n)
            expected = e0 + sum(single[k] for k in range(n) if state[k])
            assert energy(state) == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ------------------------------------------------- sulfate and guanidinium
def fork_geometry(T, sp):
    """Sulfate gripped end-on by NH1/NH2 of a guanidinium in the xy plane:
    two sulfate oxygens each exactly d_hb from one nitrogen, all in plane.
    Returns (ligands, sulfur position, quaternion)."""
    lig = geo.guanidinium([0, 0, 0], [1, 0, 0], normal=[0, 0, 1])
    nh1, nh2 = lig.pos[1], lig.pos[2]
    d_so, d_hb = float(T.d_site[sp]), float(T.d_hb[sp])
    half_oo = d_so * math.sqrt(8 / 3) / 2            # half the O...O edge
    half_nn = np.linalg.norm(nh1 - nh2) / 2
    h = math.sqrt(d_hb ** 2 - (half_oo - half_nn) ** 2)
    mid = 0.5 * (nh1 + nh2) + np.array([h, 0.0, 0.0])
    oa, ob = mid + [0, half_oo, 0], mid - [0, half_oo, 0]
    s = mid + np.array([math.sqrt(d_so ** 2 - half_oo ** 2), 0.0, 0.0])
    # rotation taking tetrahedron vertices 0 and 1 onto the two oxygens
    body = geo.tetrahedron()
    b1, b2 = body[0], body[1]
    l1, l2 = (oa - s) / d_so, (ob - s) / d_so

    def triad(a, b):
        e1 = a / np.linalg.norm(a)
        e3 = np.cross(a, b)
        e3 /= np.linalg.norm(e3)
        return np.stack([e1, np.cross(e3, e1), e3], axis=1)

    R = triad(l1, l2) @ triad(b1, b2).T
    return lig, s, geo.quat_from_matrix(R), (oa, ob)


def sulfate_energy(T, E, sp, lig, s, q):
    U = frame(T, sp, q)
    e = 0.0
    for x, t, ch, nrm in zip(lig.pos, lig.type, lig.charge, lig.normal):
        e += e_oxyanion_atom(T, E, sp, U, *(x - s), t, ch, *nrm)
    return e


def test_guanidinium_grips_two_sulfate_oxygens(params):
    """In the fork geometry exactly two O...N pairs are at the hydrogen-bond
    distance and in the donor plane, so the hydrogen-bond energy is -2 eps_hb.
    It is isolated by differencing against the same model with eps_hb = 0."""
    key = "contact.SO4.N_guanidinium.eps_hb"
    p1 = params.with_overrides({key: 2.5})
    p0 = params.with_overrides({key: 0.0})
    sp = p1.species_index("SO4")
    T1, T0 = p1.tables(), p0.tables()
    E = make_elec(0.71, 1.0)
    lig, s, q, (oa, ob) = fork_geometry(T1, sp)
    # the construction really did put the oxygens where intended
    U = frame(T1, sp, q)
    pos_o = s + float(T1.d_site[sp]) * U[:4]
    assert np.allclose(pos_o[0], oa) and np.allclose(pos_o[1], ob)
    assert np.linalg.norm(oa - lig.pos[1]) == pytest.approx(float(T1.d_hb[sp]))
    hb = sulfate_energy(T1, E, sp, lig, s, q) - sulfate_energy(T0, E, sp, lig, s, q)
    assert hb == pytest.approx(-2 * 2.5)


def test_sulfate_stacked_on_the_guanidinium_face_is_not_hydrogen_bonded(params):
    key = "contact.SO4.N_guanidinium.eps_hb"
    p1 = params.with_overrides({key: 2.5})
    p0 = params.with_overrides({key: 0.0})
    sp = p1.species_index("SO4")
    T1, T0 = p1.tables(), p0.tables()
    E = make_elec(0.71, 1.0)
    lig = geo.guanidinium([0, 0, 0], [1, 0, 0], normal=[0, 0, 1])
    d_hb = float(T1.d_hb[sp])
    # one sulfate oxygen directly above NH1, at the hydrogen-bond distance
    o = lig.pos[1] + np.array([0.0, 0.0, d_hb])
    s = o + np.array([0.0, 0.0, float(T1.d_site[sp])])
    q = geo.quat_from_two_vectors(geo.tetrahedron()[0], [0, 0, -1])
    hb = sulfate_energy(T1, E, sp, lig, s, q) - sulfate_energy(T0, E, sp, lig, s, q)
    assert hb == pytest.approx(0.0, abs=1e-12)
    # the same oxygen-nitrogen distance in the plane does count
    o = lig.pos[1] + d_hb * np.array([0.0, 1.0, 0.0])
    s = o + np.array([0.0, float(T1.d_site[sp]), 0.0])
    q = geo.quat_from_two_vectors(geo.tetrahedron()[0], [0, -1, 0])
    hb = sulfate_energy(T1, E, sp, lig, s, q) - sulfate_energy(T0, E, sp, lig, s, q)
    assert hb <= -2.5 + 1e-9


def test_ammonium_has_no_plane_restriction(params):
    key = "contact.SO4.N_ammonium.eps_hb"
    p1 = params.with_overrides({key: 1.5})
    p0 = params.with_overrides({key: 0.0})
    sp = p1.species_index("SO4")
    T1, T0 = p1.tables(), p0.tables()
    E = make_elec(0.71, 1.0)
    lig = geo.ammonium([0, 0, 0])
    d_hb = float(T1.d_hb[sp])
    for direction in ([0, 0, 1], [1, 0, 0], [0.6, 0.0, 0.8]):
        u = np.asarray(direction, float)
        s = (d_hb + float(T1.d_site[sp])) * u
        q = geo.quat_from_two_vectors(geo.tetrahedron()[0], -u)
        hb = sulfate_energy(T1, E, sp, lig, s, q) - sulfate_energy(T0, E, sp, lig, s, q)
        assert hb == pytest.approx(-1.5)


# ----------------------------------------------------------------- ion-ion
@pytest.mark.parametrize("a,b,qq", [("Mg", "Ca", 4.0), ("Ca", "Ca", 4.0), ("Ca", "SO4", -4.0),
                                    ("SO4", "SO4", 4.0), ("Mg", "SO4", -4.0)])
def test_far_apart_ions_interact_as_point_charges(params, a, b, qq):
    """At 3 nm only the monopoles matter: u = lB q_a q_b exp(-kappa r) / r.
    Checks that each ion's charge points add up to its formal charge."""
    T = params.tables()
    kappa = 0.5
    E = make_elec(0.71, kappa)
    rng = np.random.default_rng(9)
    sa, sb = params.species_index(a), params.species_index(b)
    Ua, Ub = frame(T, sa, geo.random_quaternion(rng)), frame(T, sb, geo.random_quaternion(rng))
    st = np.zeros(MAX_SITES, dtype=np.uint8)
    r = 3.0
    e = e_ion_ion(T, E, sa, Ua, st, int(T.nsite[sa]), sb, Ub, st, int(T.nsite[sb]), r, 0.0, 0.0)
    assert e == pytest.approx(0.71 * qq * math.exp(-kappa * r) / r, rel=5e-3)


def test_ion_pair_energy_does_not_depend_on_which_ion_is_named_first(params):
    T = params.tables()
    E = make_elec(0.71, 1.0, 2.0)
    rng = np.random.default_rng(10)
    st = np.zeros(MAX_SITES, dtype=np.uint8)
    for a, b in (("Ca", "SO4"), ("Mg", "Ca"), ("SO4", "SO4")):
        sa, sb = params.species_index(a), params.species_index(b)
        for _ in range(10):
            Ua, Ub = frame(T, sa, geo.random_quaternion(rng)), frame(T, sb, geo.random_quaternion(rng))
            v = rng.normal(size=3)
            v *= rng.uniform(0.3, 0.9) / np.linalg.norm(v)
            e_ab = e_ion_ion(T, E, sa, Ua, st, int(T.nsite[sa]), sb, Ub, st, int(T.nsite[sb]), *v)
            e_ba = e_ion_ion(T, E, sb, Ub, st, int(T.nsite[sb]), sa, Ua, st, int(T.nsite[sa]), *(-v))
            assert e_ab == pytest.approx(e_ba, rel=1e-10, abs=1e-10)


def test_sulfate_oxygen_can_enter_a_cation_site(params):
    """A sulfate oxygen is a ligand atom for a cation: with the facing site
    open and the oxygen on it, the pair gains eps_in relative to the same
    geometry with the inner term switched off."""
    key = "contact.Ca.O_sulfate.eps_in"
    eps = float(params.value(key))
    p0 = params.with_overrides({key: 0.0})
    T1, T0 = params.tables(), p0.tables()
    E = make_elec(0.71, 0.0)
    ca, so4 = params.species_index("Ca"), params.species_index("SO4")
    Uc = frame(T1, ca)
    d, d_so = float(T1.d_site[ca]), float(T1.d_site[so4])
    # sulfate beyond the +z site, one oxygen pointing back at the calcium
    q = geo.quat_from_two_vectors(geo.tetrahedron()[0], [0, 0, -1])
    Us = frame(T1, so4, q)
    st_c = np.zeros(MAX_SITES, dtype=np.uint8)
    st_c[0] = 1
    st_s = np.zeros(MAX_SITES, dtype=np.uint8)
    args = (ca, Uc, st_c, 6, so4, Us, st_s, 4, 0.0, 0.0, d + d_so)
    assert e_ion_ion(T1, E, *args) - e_ion_ion(T0, E, *args) == pytest.approx(-eps)
