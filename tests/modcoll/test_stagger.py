"""Stagger scan (grid/FFT evaluation of the pair sum) and profile analysis.

Oracles: the literal double sum of electrostatics.pair_energy_direct, closed
forms for point charges, synthetic profiles and molecules with a constructed
period, and the D-register of native collagen stated in the README.
"""
import functools

import numpy as np
import pytest

from modcoll import electrostatics as es
from modcoll import stagger as st
from modcoll.modify import modify
from modcoll.sites import SiteType, make_molecule, native_molecule

D_AXIS = 1.65
SMALL = st.ScanGrid(ds_nm=0.2, n_phi=24, n_z=512, s_origin_nm=-10.0)


def yukawa(lam=0.8):
    return functools.partial(es.screened_coulomb, debye_nm=lam)


def on_grid_molecule(rng, n, radii, length=30.0):
    """Random charges on grid nodes: the grid evaluation is then exact."""
    s = 0.2 * rng.integers(0, int(length / 0.2), n)
    phi = 2 * np.pi * rng.integers(0, 24, n) / 24
    r = rng.choice(radii, n)
    mol = make_molecule(s, phi, r, np.full(n, int(SiteType.LYS)), helix_length_nm=length)
    return mol, rng.normal(size=n)


@pytest.mark.parametrize("antiparallel", [False, True])
def test_scan_equals_direct_sum_for_sites_on_grid_nodes(antiparallel):
    rng = np.random.default_rng(11)
    radii = np.array([0.45, 0.6, 0.8])
    a, wa = on_grid_molecule(rng, 40, radii)
    b, wb = on_grid_molecule(rng, 35, radii)
    sc = st.StaggerScanner(yukawa(), radii, D_AXIS, SMALL)
    energy = sc.scan(a, wa, b, wb, antiparallel=antiparallel)
    ang = SMALL.facing_angles
    for _ in range(25):
        j = int(rng.integers(-150, 151))
        ia, ib = rng.integers(0, 24, 2)
        direct = es.pair_energy_direct(a, wa, b, wb, yukawa(), j * 0.2, ang[ia], ang[ib],
                                       D_AXIS, antiparallel=antiparallel)
        assert energy[ia, ib, j % SMALL.n_z] == pytest.approx(direct, rel=1e-9, abs=1e-11)


def test_scan_of_two_point_charges_finds_the_contact_and_its_energy():
    lam = 0.8
    a = make_molecule([5.0], [np.pi / 2], [0.6], [SiteType.LYS])
    b = make_molecule([8.0], [np.pi], [0.6], [SiteType.ASP])
    sc = st.StaggerScanner(yukawa(lam), [0.6], D_AXIS, SMALL)
    energy = sc.scan(a, [1.0], b, [-1.0])
    ia, ib, j = np.unravel_index(np.argmin(energy), energy.shape)
    # opposite charges: deepest where they face each other at equal height
    assert SMALL.facing_angles[ia] == pytest.approx(np.pi / 2)
    assert SMALL.facing_angles[ib] == pytest.approx(np.pi)
    assert (j - SMALL.n_z) * SMALL.ds_nm == pytest.approx(-3.0)
    assert energy.min() == pytest.approx(-es.screened_coulomb(D_AXIS - 1.2, lam), rel=1e-9)
    # and nowhere positive: a single attractive pair
    assert energy.max() < 1e-9


def test_scan_is_symmetric_under_exchange_of_identical_molecules():
    rng = np.random.default_rng(12)
    a, wa = on_grid_molecule(rng, 30, np.array([0.5, 0.7]))
    sc = st.StaggerScanner(yukawa(), [0.5, 0.7], D_AXIS, SMALL)
    e = sc.scan(a, wa, a, wa)
    # E(D, fa, fb) = E(-D, fb, fa)
    mirrored = np.roll(e[:, :, ::-1], 1, axis=2).transpose(1, 0, 2)
    np.testing.assert_allclose(e, mirrored, atol=1e-10)


@pytest.mark.parametrize("antiparallel", [False, True])
def test_scan_of_real_molecules_agrees_with_direct_sum(antiparallel):
    """Off-grid sites: the cloud-in-cell error must stay far below the
    several-kT features of the profiles."""
    native = native_molecule()
    env = es.Environment(7.4, 0.15)
    mod = modify(native, np.random.default_rng(13), f_succ=0.5)
    qa, qb = es.henderson_hasselbalch(native, env), es.henderson_hasselbalch(mod, env)
    kern = yukawa(env.debye_length_nm)
    sc = st.StaggerScanner(kern, st.charged_shell_radii(native, 2), D_AXIS)
    energy = sc.scan(native, qa, mod, qb, antiparallel=antiparallel)
    rng = np.random.default_rng(14)
    err, ref = [], []
    for _ in range(8):
        j = int(rng.integers(-1400, 1401))
        ia, ib = rng.integers(0, 24, 2)
        direct = es.pair_energy_direct(native, qa, mod, qb, kern, j * 0.2,
                                       sc.grid.facing_angles[ia], sc.grid.facing_angles[ib],
                                       D_AXIS, antiparallel=antiparallel)
        err.append(energy[ia, ib, j % sc.grid.n_z] - direct)
        ref.append(direct)
    assert np.max(np.abs(err)) < 0.4
    assert np.sqrt(np.mean(np.square(err))) < 0.2
    assert np.std(ref) > 1.0                 # the sample spans real structure


def test_hydrophobic_scan_agrees_with_direct_sum():
    native = native_molecule()
    h = native.centered_hydrophobicity()
    sc = st.StaggerScanner(es.hydrophobic_contact, st.residue_shell_radii(native, 4), D_AXIS)
    energy = sc.scan(native, h, native, h)
    rng = np.random.default_rng(15)
    for _ in range(4):
        j = int(rng.integers(30, 1400))
        ia, ib = rng.integers(0, 24, 2)
        direct = es.pair_energy_direct(native, h, native, h, es.hydrophobic_contact, j * 0.2,
                                       sc.grid.facing_angles[ia], sc.grid.facing_angles[ib],
                                       D_AXIS)
        assert energy[ia, ib, j] == pytest.approx(direct, abs=0.03 * np.abs(energy).max())


def test_deposit_conserves_weight_and_spreads_smeared_sites_uniformly():
    sc = st.StaggerScanner(yukawa(), [0.4, 0.6, 0.8], D_AXIS, SMALL)
    rng = np.random.default_rng(16)
    n = 50
    s, phi = rng.uniform(0, 30, n), rng.uniform(-np.pi, np.pi, n)
    r, w = rng.uniform(0.4, 0.8, n), rng.normal(size=n)
    rho = sc.deposit(s, phi, r, w)
    assert rho.sum() == pytest.approx(w.sum())
    # first moments are preserved by linear assignment
    z_nodes = SMALL.s_origin_nm + SMALL.ds_nm * np.arange(SMALL.n_z)
    assert (rho.sum(axis=(0, 1)) * z_nodes).sum() == pytest.approx((w * s).sum())
    assert (rho.sum(axis=(1, 2)) * sc.radii).sum() == pytest.approx((w * r).sum())
    ring = sc.deposit([3.1], [0.0], [0.6], [2.0], smeared=[True])
    per_angle = ring.sum(axis=(0, 2))
    np.testing.assert_allclose(per_angle, 2.0 / SMALL.n_phi)
    with pytest.raises(ValueError):
        sc.deposit([3.0], [0.0], [0.95], [1.0])       # outside the shells
    with pytest.raises(ValueError):
        sc.deposit([80.0], [0.0], [0.6], [1.0])       # outside the axial grid


def test_lag_slice_maps_signed_staggers():
    sc = st.StaggerScanner(yukawa(), [0.6], D_AXIS, SMALL)
    idx, lags = sc.lag_slice(-1.0, 1.0)
    np.testing.assert_allclose(lags, np.arange(-5, 6) * 0.2)
    np.testing.assert_array_equal(idx[:5], SMALL.n_z + np.arange(-5, 0))
    np.testing.assert_array_equal(idx[5:], np.arange(0, 6))
    with pytest.raises(ValueError):
        sc.lag_slice(0.0, 60.0)


def test_reduce_over_angles_closed_form_and_bounds():
    # two facing states per stagger with energies 0 and e: F = -ln((1 + exp(-e)) / 2)
    e = np.array([0.0, 1.0, 5.0, -3.0])
    energy = np.stack([np.zeros(4), e]).reshape(2, 1, 4)
    e_min, f = st.reduce_over_angles(energy)
    np.testing.assert_allclose(e_min, np.minimum(0.0, e))
    np.testing.assert_allclose(f, -np.log(0.5 * (1.0 + np.exp(-e))))
    rng = np.random.default_rng(17)
    big = rng.normal(scale=30.0, size=(6, 6, 50))        # no overflow at large |E|
    e_min, f = st.reduce_over_angles(big)
    assert np.all(np.isfinite(f))
    assert np.all(f >= e_min - 1e-12) and np.all(f <= big.mean(axis=(0, 1)) + 1e-12)
    assert np.all(f <= e_min + np.log(36) + 1e-12)       # one state out of 36 dominates


def test_leading_minima_of_an_analytic_profile():
    x = np.arange(0.0, 300.01, 0.1)
    # wells of depth 3, 2, 1 at 60, 150, 240 nm and a deep one at 2 nm (excluded)
    u = (-3 * np.exp(-((x - 60) / 4) ** 2) - 2 * np.exp(-((x - 150) / 4) ** 2)
         - 1 * np.exp(-((x - 240) / 4) ** 2) - 9 * np.exp(-((x - 2) / 1.0) ** 2))
    mins = st.leading_minima(x, u, n=3)
    assert [round(p) for p, _ in mins] == [60, 150, 240]
    assert [v for _, v in mins] == pytest.approx([-3, -2, -1], abs=1e-6)
    # two minima closer than min_sep: only the deeper one is kept
    u2 = -3 * np.exp(-((x - 60) / 2) ** 2) - 2.5 * np.exp(-((x - 68) / 2) ** 2)
    assert [round(p) for p, _ in st.leading_minima(x, u2, n=5)] == [60]
    assert len(st.leading_minima(x, u2, n=5, min_sep_nm=5.0)) == 2


def comb_profile(period, phase=0.0, depth=1.0, width=1.5, noise=0.0, seed=0):
    x = np.arange(0.0, 300.01, 0.2)
    u = np.zeros_like(x)
    for k in range(-1, 12):
        u -= depth * np.exp(-((x - (k * period + phase)) / width) ** 2)
    if noise:
        u += noise * np.random.default_rng(seed).normal(size=len(x))
    return x, u


def test_comb_selectivity_detects_wells_at_integer_multiples():
    x, u = comb_profile(67.0)
    cs = st.comb_selectivity(x, u, 67.0)
    assert cs["rank"] == 0.0 and cs["z"] > 2.0
    assert cs["best_phase_nm"] == pytest.approx(0.0, abs=3.0)   # anywhere within the window
    assert cs["best_z"] == pytest.approx(cs["z"])


def test_comb_selectivity_reports_the_phase_of_shifted_wells():
    x, u = comb_profile(67.0, phase=20.0)
    cs = st.comb_selectivity(x, u, 67.0)
    assert cs["z"] < 0.5                                   # not at integer multiples
    assert cs["best_phase_nm"] == pytest.approx(20.0, abs=3.0)
    assert cs["best_z"] > 2.0
    x, u = comb_profile(67.0, phase=-25.0)
    assert st.comb_selectivity(x, u, 67.0)["best_phase_nm"] == pytest.approx(-25.0, abs=3.0)


def test_comb_selectivity_is_scale_free_and_flat_profiles_score_zero():
    x, u = comb_profile(67.0, noise=0.05, seed=1)
    a, b = st.comb_selectivity(x, u, 67.0), st.comb_selectivity(x, 40.0 * u + 7.0, 67.0)
    assert a["z"] == pytest.approx(b["z"]) and a["rank"] == b["rank"]
    flat = st.comb_selectivity(x, np.zeros_like(x), 67.0)
    assert flat["z"] == 0.0


def test_pure_noise_has_no_comb_selectivity_on_average():
    z = [st.comb_selectivity(*comb_profile(67.0, depth=0.0, noise=1.0, seed=s), 67.0)["z"]
         for s in range(60)]
    assert abs(np.mean(z)) < 4 * np.std(z) / np.sqrt(len(z))


def test_autocorrelation_recovers_a_constructed_period_at_any_phase():
    shifts = np.arange(10.0, 141.0, 0.2)
    for phase in (0.0, 11.0, 30.0):
        x, u = comb_profile(48.0, phase=phase, noise=0.05, seed=2)
        r = st.profile_autocorrelation(x, u, shifts)
        period, value = st.dominant_period(shifts, r)
        assert period == pytest.approx(48.0, abs=0.4)
        assert value > 0.8
        assert r[np.argmin(np.abs(shifts - 24.0))] < 0.0      # half a period: anticorrelated
        assert r[np.argmin(np.abs(shifts - 96.0))] > 0.8      # the second repeat


def test_dominant_period_prefers_the_fundamental_and_handles_no_peak():
    shifts = np.arange(1.0, 100.0)
    r = 0.8 * np.cos(2 * np.pi * shifts / 20.0)
    r[39] = 0.85                                              # a slightly higher 2nd repeat
    assert st.dominant_period(shifts, r)[0] == 20.0
    assert np.isnan(st.dominant_period(shifts, -np.ones_like(shifts))[0])
    assert np.isnan(st.dominant_period(shifts, np.linspace(0.5, -0.5, len(shifts)))[0])


def test_noise_profile_has_no_autocorrelation_beyond_its_grain():
    x, u = comb_profile(67.0, depth=0.0, noise=1.0, seed=5)
    r = st.profile_autocorrelation(x, u, np.arange(10.0, 141.0, 1.0))
    assert np.nanmax(np.abs(r)) < 5 / np.sqrt(len(x) - 700)


def test_chance_fraction_near_multiples():
    # 4 windows of 6 nm in the 295 nm between 5 and 300 nm (D = 66.9 nm)
    assert st.near_multiple_fraction(300.0, 66.9) == pytest.approx(24.0 / 295.0)
    # a window cut by the end of the range counts only its inside part
    assert st.near_multiple_fraction(101.0, 50.0, window_nm=3.0) == pytest.approx(
        (6.0 + 4.0) / 96.0)


def test_boltzmann_fraction_limits():
    x = np.arange(0.0, 300.01, 0.2)
    flat = st.boltzmann_multiple_fraction(x, np.zeros_like(x), 66.9)
    assert flat == pytest.approx(st.near_multiple_fraction(300.0, 66.9), abs=0.005)
    _, well_at_d = comb_profile(66.9, depth=30.0)
    assert st.boltzmann_multiple_fraction(x, well_at_d, 66.9) > 0.999
    _, well_off = comb_profile(66.9, phase=30.0, depth=30.0)
    assert st.boltzmann_multiple_fraction(x, well_off, 66.9) < 1e-3


def test_nearest_multiple():
    assert st.nearest_multiple(135.0, 66.9) == (2, pytest.approx(1.2))
    assert st.nearest_multiple(20.0, 66.9) == (0, pytest.approx(20.0))


def test_axial_profile_bins_by_residue_rise():
    mol = make_molecule([0.05, 0.10, 0.35, 0.95], [0, 1, 2, 3], [0.5] * 4,
                        [SiteType.LYS] * 4, d_period_nm=234 * 0.3)   # rise 0.3 nm
    centres, prof = st.axial_profile(mol, [1.0, 2.0, -1.0, 5.0])
    np.testing.assert_allclose(centres, [0.15, 0.45, 0.75, 1.05])
    np.testing.assert_allclose(prof, [3.0, -1.0, 0.0, 5.0])


def test_lagged_correlation_of_a_periodic_pattern():
    x = np.tile([1.0, 1.0, -1.0, -1.0, 0.0, 0.0], 40)      # period 6
    assert st.lagged_correlation(x, x, 6) == pytest.approx(1.0)
    assert st.lagged_correlation(x, x, 12) == pytest.approx(1.0)
    assert st.lagged_correlation(x, x, 3) < -0.4
    assert st.lagged_correlation(x, -x, 6) == pytest.approx(-1.0)
    rng = np.random.default_rng(18)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    assert abs(st.lagged_correlation(a, b, 234)) < 4 / np.sqrt(4000 - 234)
    assert np.isnan(st.lagged_correlation([1.0, 2.0], [1.0, 2.0], 1))


def test_scan_of_a_periodic_charge_pattern_shows_its_period():
    """Molecule with alternating +/- charges every 5 nm on one face: against a
    copy of itself the energy must be lowest at staggers of 5, 15, 25 nm
    (opposite charges aligned) and highest at 10, 20 nm."""
    s = np.arange(0.0, 60.0, 5.0)
    types = [SiteType.LYS if k % 2 == 0 else SiteType.ASP for k in range(len(s))]
    mol = make_molecule(s, np.zeros(len(s)), np.full(len(s), 0.6), types)
    q = es.henderson_hasselbalch(mol, es.Environment(7.0))
    np.testing.assert_allclose(q, [1, -1] * 6, atol=1e-3)
    grid = st.ScanGrid(n_z=1024)
    sc = st.StaggerScanner(yukawa(0.8), [0.6], D_AXIS, grid)
    idx, lags = sc.lag_slice(0.0, 40.0)
    e_min, _ = st.reduce_over_angles(sc.scan(mol, q, mol, q)[:, :, idx])
    mins = sorted(p for p, _ in st.leading_minima(lags, e_min, n=4, min_sep_nm=4.0,
                                                  lag_min_nm=1.0))
    np.testing.assert_allclose(mins, [5.0, 15.0, 25.0, 35.0], atol=0.2)
    assert e_min[np.searchsorted(lags, 5.0)] < 0 < e_min[np.searchsorted(lags, 10.0)]
    shifts = np.arange(2.0, 21.0, 0.2)
    r = st.profile_autocorrelation(lags, e_min, shifts, lag_min_nm=1.0)
    assert st.dominant_period(shifts, r)[0] == pytest.approx(10.0, abs=0.21)


@pytest.fixture(scope="module")
def native_profiles():
    # the gate is defined on the repository pKa set, like the existing tools
    native = native_molecule()
    model = st.PairModel(native, [0.15], orientations=("parallel", "antiparallel"))
    return native, model, model.profiles(native, native, [7.4, 3.5],
                                         es.charge_model_for("repository"))


def test_native_control_binds_in_d_register(native_profiles):
    """README: quarter-stagger emerges from the sequence, minima at 1.00 D,
    1.99 D and 3.00 D."""
    native, model, prof = native_profiles
    p = prof[(7.4, 0.15, "parallel")]
    lags, d = model.lags["parallel"], native.d_period_nm
    gate = st.native_gate(lags, p[0], p[2], d)
    assert gate["passed"]
    assert gate["min_over_angles_minima"][0]["stagger_D"] == pytest.approx(1.0, abs=0.02)
    top3 = sorted(m["stagger_D"] for m in gate["azimuth_mean_minima"][:3])
    np.testing.assert_allclose(top3, [1.0, 2.0, 3.0], atol=0.02)
    assert gate["min_over_angles_minima"][0]["energy_kT"] < -10.0
    assert st.comb_selectivity(lags, p[0], d)["rank"] == 0.0


def test_native_control_does_not_bind_in_acid(native_profiles):
    # README: "at pH 3.5 (net strongly positive) they repel and disperse"
    native, model, prof = native_profiles
    lags = model.lags["parallel"]
    overlap = (lags > 5.0) & (lags < 200.0)
    acid, neutral = prof[(3.5, 0.15, "parallel")][0], prof[(7.4, 0.15, "parallel")][0]
    assert es.net_charge(native, es.Environment(3.5)) > 150
    assert acid[overlap].min() > neutral[overlap].min() + 10.0
    assert np.median(acid[overlap]) > 0.0 > np.median(neutral[overlap])


def test_profile_reductions_are_ordered(native_profiles):
    _, _, prof = native_profiles
    for p in prof.values():
        tot_min, tot_boltz, tot_mean, el_min, el_boltz = p
        assert np.all(tot_min <= tot_boltz + 1e-9) and np.all(tot_boltz <= tot_mean + 1e-9)
        assert np.all(el_min <= el_boltz + 1e-9)


def test_parallel_profile_vanishes_without_overlap_and_antiparallel_covers_both_signs(
        native_profiles):
    native, model, prof = native_profiles
    lags = model.lags["parallel"]
    assert lags[0] == 0.0 and lags[-1] == pytest.approx(300.0)
    assert np.all(np.diff(lags) == pytest.approx(0.2))
    assert abs(prof[(7.4, 0.15, "parallel")][0][-1]) < 2.0     # ends barely touch
    anti = model.lags["antiparallel"]
    assert anti[0] == pytest.approx(-300.0) and anti[-1] == pytest.approx(300.0)


def test_pair_model_uses_the_supplied_charge_model():
    native = native_molecule()
    model = st.PairModel(native, [0.15], orientations=("parallel",))
    neutral = lambda mol, env: np.zeros(len(mol))
    p = model.profiles(native, native, [7.4], charge_model=neutral)[(7.4, 0.15, "parallel")]
    np.testing.assert_allclose(p[3], 0.0, atol=1e-9)            # no electrostatics left
    # what remains is the hydrophobic channel, which carries the D-register
    d = native.d_period_nm
    deepest = st.leading_minima(model.lags["parallel"], p[0], n=1)[0][0]
    assert deepest / d == pytest.approx(1.0, abs=0.02)


def test_pair_model_distinguishes_patterns():
    native = native_molecule()
    model = st.PairModel(native, [0.15], orientations=("parallel",))
    a = modify(native, np.random.default_rng(1), f_succ=0.5)
    b = modify(native, np.random.default_rng(2), f_succ=0.5)
    c = modify(native, np.random.default_rng(3), f_succ=0.5)
    key = (4.2, 0.15, "parallel")
    p_ab = model.profiles(a, b, [4.2])[key]
    p_ac = model.profiles(a, c, [4.2])[key]
    p_ab2 = model.profiles(a, b, [4.2])[key]
    np.testing.assert_array_equal(p_ab, p_ab2)
    assert np.std(p_ab[0] - p_ac[0]) > 0.5      # different partners, different energies


def test_native_gate_fails_on_a_profile_without_d_register():
    x, off = comb_profile(66.9, phase=25.0, depth=5.0)
    gate = st.native_gate(x, off, off, 66.9)
    assert not gate["passed"]
    x, on = comb_profile(66.9, depth=5.0, noise=0.01, seed=3)
    assert st.native_gate(x, on, on, 66.9)["passed"]


def test_pair_model_rejects_a_grid_whose_periodic_images_would_interact():
    native = native_molecule()
    short = st.PairSettings(grid=st.ScanGrid(n_z=3200))    # 640 nm box, 300 nm molecule
    with pytest.raises(ValueError):
        st.PairModel(native, [0.01], short, ("parallel",))


def test_row_wise_lagged_correlation_matches_the_scalar_one():
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(5, 400)), rng.normal(size=(5, 400))
    y[2] = 0.7                                   # a constant row has no correlation
    lags = [0, 7, 234, 399]
    rows = st.lagged_correlation_rows(x, y, lags)
    assert rows.shape == (5, 4)
    for i, lag in enumerate(lags):
        for k in range(5):
            expected = st.lagged_correlation(x[k], y[k], lag)
            assert (np.isnan(rows[k, i]) and np.isnan(expected)) or rows[k, i] == pytest.approx(
                expected, abs=1e-10)
    assert np.all(np.isnan(rows[:, 3])) and np.all(np.isnan(rows[2]))
    with pytest.raises(ValueError):
        st.lagged_correlation_rows(x, y[:, :100], lags)


def test_native_control_keeps_its_register_with_the_recommended_pka_set():
    native = native_molecule()
    model = st.PairModel(native, [0.15], orientations=("parallel",))
    p = model.profiles(native, native, [7.4])[(7.4, 0.15, "parallel")]
    gate = st.native_gate(model.lags["parallel"], p[0], p[2], native.d_period_nm)
    assert gate["passed"]
    assert gate["min_over_angles_minima"][0]["stagger_D"] == pytest.approx(1.0, abs=0.02)
