"""The Markov chain against closed-form populations.

Every test is seeded. Statistical comparisons use block averages for the
error bar and accept a deviation of up to five standard errors.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.bridges import INNER, BridgeAnalyzer  # noqa: E402
from modcoll.ions.calibrate import acetate, association_constants  # noqa: E402
from modcoll.ions.params import PER_NM3_PER_MOLAR, load_params  # noqa: E402
from modcoll.ions.sampler import IonSampler  # noqa: E402


@pytest.fixture(scope="module")
def params():
    return load_params()


def block_mean_and_error(x, n_blocks=25):
    x = np.asarray(x, dtype=float)
    m = len(x) // n_blocks
    blocks = x[: m * n_blocks].reshape(n_blocks, m).mean(axis=1)
    return blocks.mean(), blocks.std(ddof=1) / math.sqrt(n_blocks)


def empty():
    return geo.LigandSet()


# ----------------------------------------------------------- ideal-gas limit
def test_ideal_gas_density_and_free_ion_site_statistics(params):
    """Uncharged 'calcium' with a low desolvation cost, nothing else in the
    box. Closed forms:
      <N>          = c N_A V                    (bath concentration means the
                                                 density of free ions)
      P(N = 0)     = exp(-<N>)                  (Poisson)
      var N        = <N>
      P(site is L) = 1 / (1 + exp(dG_des))      (independent two-state sites)
    The last two test the activity convention zeta = rho / z_free together
    with the rule that only fully hydrated ions are deleted.
    """
    dg = 1.0
    p = params.with_overrides({"species.Ca.charge": 0.0, "species.Ca.dg_des": dg})
    box = [4.0, 4.0, 4.0]
    conc = 0.05
    # mostly flip moves: with dG_des this low an ion is rarely fully hydrated,
    # and only fully hydrated ions can leave, so site switching sets the pace
    s = IonSampler(p, empty(), box, [True] * 3, 0.1, np.random.default_rng(101), bath={"Ca": conc},
                   move_probabilities=(0.15, 0.05, 0.8, 0.0))
    s.sweep(2000)
    counts, l_frac = [], []
    for _ in range(30000):
        s.sweep(8)
        snap = s.snapshot()
        counts.append(len(snap.species))
        if len(snap.species):
            l_frac.append(snap.state[:, :7].mean())
    expected_n = conc * PER_NM3_PER_MOLAR * 64.0
    mean_n, err_n = block_mean_and_error(counts)
    assert abs(mean_n - expected_n) < 5 * err_n
    assert err_n < 0.03 * expected_n                      # the test has teeth
    counts = np.asarray(counts)
    assert counts.var() == pytest.approx(expected_n, rel=0.06)
    assert (counts == 0).mean() == pytest.approx(math.exp(-expected_n), rel=0.08)
    mean_l, err_l = block_mean_and_error(l_frac)
    assert abs(mean_l - 1.0 / (1.0 + math.exp(dg))) < 5 * err_l
    assert err_l < 0.01


def test_sulfate_ideal_gas_density(params):
    p = params.with_overrides({"species.SO4.charge": 0.0, "species.SO4.oxygen_charge": 0.0})
    s = IonSampler(p, empty(), [4.0, 4.0, 4.0], [True] * 3, 0.1, np.random.default_rng(5),
                   bath={"SO4": 0.03})
    s.sweep(1000)
    counts = []
    for _ in range(20000):
        s.sweep(1)
        counts.append(s.n_ions)
    expected = 0.03 * PER_NM3_PER_MOLAR * 64.0
    mean_n, err_n = block_mean_and_error(counts)
    # sulfates have a real excluded volume: second virial estimate with a
    # 0.6 nm exclusion sphere lowers the density by 2 B2 rho ~ 1.6%
    assert abs(mean_n - expected) < 5 * err_n + 0.02 * expected


def test_two_species_are_exchanged_independently(params):
    p = params.with_overrides({"species.Ca.charge": 0.0, "species.Mg.charge": 0.0})
    s = IonSampler(p, empty(), [4.0, 4.0, 4.0], [True] * 3, 0.1, np.random.default_rng(8),
                   bath={"Ca": 0.02, "Mg": 0.06})
    s.sweep(1000)
    ca, mg = [], []
    for _ in range(20000):
        s.sweep(1)
        ca.append(s.count("Ca"))
        mg.append(s.count("Mg"))
    for series, conc in ((ca, 0.02), (mg, 0.06)):
        mean_n, err_n = block_mean_and_error(series)
        assert abs(mean_n - conc * PER_NM3_PER_MOLAR * 64.0) < 5 * err_n + 0.01


# ------------------------------------------------------- two-state closed form
def test_site_switching_reproduces_two_state_populations(params):
    """A calcium held in place (flip moves only) with one uncharged ligand
    oxygen on the axis of the +z site, half-way out of the inner well, and the
    site water's repulsion switched off so that both states are populated.
      site 0: P(L) / P(W) = exp(-(dG_des - eps_in * B(1/2))),  B(1/2) = 0.5625
      others: P(L) / P(W) = exp(-dG_des)
    """
    dg, eps = 2.0, 3.0
    p = params.with_overrides({
        "species.Ca.dg_des": dg, "contact.Ca.O_carboxylate.eps_in": eps,
        "contact.Ca.O_carboxylate.eps_out": 0.0, "species.Ca.water_radius": -0.14,
    })
    d = float(p.value("species.Ca.d_MO"))
    w_r = float(p.value("species.Ca.well_radial_halfwidth"))
    centre = np.array([2.0, 2.0, 2.0])
    lig = geo.LigandSet(pos=[centre + [0.0, 0.0, d + 0.5 * w_r]], type=[geo.TYPE_INDEX["O_carboxylate"]],
                        charge=[0.0], mol=[0])
    s = IonSampler(p, lig, [4.0, 4.0, 4.0], [True] * 3, 0.1, np.random.default_rng(77),
                   move_probabilities=(0.0, 0.0, 1.0, 0.0), flip_move_probability=0.0)
    s.add_ion("Ca", centre)
    s.sweep(500)
    states = []
    for _ in range(40000):
        s.sweep(5)
        states.append(s.snapshot().state[0, :7].copy())
    states = np.asarray(states, dtype=float)
    p0 = 1.0 / (1.0 + math.exp(dg - eps * 0.5625))
    p_other = 1.0 / (1.0 + math.exp(dg))
    m0, e0 = block_mean_and_error(states[:, 0])
    assert abs(m0 - p0) < 5 * e0 and e0 < 0.01
    m1, e1 = block_mean_and_error(states[:, 1:].mean(axis=1))
    assert abs(m1 - p_other) < 5 * e1 and e1 < 0.005
    # the ion never moved
    assert np.allclose(s.snapshot().pos[0], centre)


# --------------------------------------------- inner/outer balance, full move set
def classify(analyzer, snap):
    """0 unbound, 1 outer, 2 inner through one site, 3 inner through several."""
    c = analyzer.contacts(snap)
    if len(c) == 0:
        return 0
    n_inner = int((c.cls == INNER).sum())
    return 1 if n_inner == 0 else (2 if n_inner == 1 else 3)


@pytest.mark.parametrize("species,seed", [("Ca", 2024), ("Mg", 2025)])
def test_inner_and_outer_sphere_populations_match_the_association_integrals(params, species, seed):
    """One cation and one acetate in a periodic box, all moves active. In the
    canonical ensemble the time spent in each contact class is proportional to
    that class's association integral, so

        P(inner) / P(outer) = K_inner / K_outer,

    with the K from importance-sampled quadrature of the same energy function
    (no Markov chain involved). Inner and outer states are connected only
    through the dock/undock moves, whose acceptance rule carries a non-trivial
    proposal ratio, so this is a direct test of their detailed balance.
    """
    box = np.array([2.4, 2.4, 2.4])       # small, so the ion is bound half the time
    lig = acetate().translated(box / 2)
    I = 0.15
    s = IonSampler(params, lig, box, [True] * 3, I, np.random.default_rng(seed))
    s.add_ion(species, [0.5, 0.5, 0.5])
    an = BridgeAnalyzer(params, lig, box, [True] * 3)
    s.sweep(20000)
    cls = []
    for _ in range(250000):
        s.sweep(20)
        cls.append(classify(an, s.snapshot()))
    cls = np.asarray(cls)
    inner = (cls >= 2).astype(float)
    outer = (cls == 1).astype(float)
    # the chain really did go back and forth
    crossings = int(np.sum((cls[1:] >= 2) != (cls[:-1] >= 2)))
    assert crossings > 200
    acc = s.acceptance()
    assert acc["dock"] > 0 and acc["undock"] > 0

    ref = association_constants(params, species, lig, I, np.random.default_rng(seed + 1),
                                n_samples=4_000_000, coulomb_cutoff=s.coulomb_cutoff)
    # ratio estimate with a block error bar
    n_blocks = 20
    m = len(cls) // n_blocks
    bi = inner[: m * n_blocks].reshape(n_blocks, m).sum(axis=1)
    bo = outer[: m * n_blocks].reshape(n_blocks, m).sum(axis=1)
    ratio = bi.sum() / bo.sum()
    # jackknife error of the ratio
    jk = np.array([(bi.sum() - bi[k]) / (bo.sum() - bo[k]) for k in range(n_blocks)])
    err = math.sqrt((n_blocks - 1) / n_blocks * np.sum((jk - jk.mean()) ** 2))
    expected = ref.K_inner / ref.K_outer
    ref_err = expected * math.hypot(math.hypot(ref.err["mono"], ref.err["multi"]) / ref.K_inner,
                                    ref.err["outer"] / ref.K_outer)
    assert err < 0.12 * expected                          # resolved to about 10%
    assert abs(ratio - expected) < 4 * math.hypot(err, ref_err)


# ------------------------------------------------------------ binding isotherm
def langmuir_guess(K, conc):
    return K * conc / (1.0 + K * conc)


def bound_counts(sampler, analyzer, n_samples, stride):
    out = []
    for _ in range(n_samples):
        sampler.sweep(stride)
        c = analyzer.contacts(sampler.snapshot())
        out.append(len(np.unique(c.ion)))
    return np.asarray(out)


def test_ideal_ion_isotherm_is_poisson_with_the_integrated_K(params):
    """Grand-canonical calcium around one acetate with ion-ion terms off. The
    number of ions in contact with the acetate is then exactly Poisson with
    mean K c, K being the association integral at the sampler's ionic strength
    and cutoff. So <n_bound> = K c and P(bound) = 1 - exp(-K c), at any c.
    """
    box = np.array([3.2, 3.2, 3.2])
    lig = acetate().translated(box / 2)
    I = 0.15
    an = BridgeAnalyzer(params, lig, box, [True] * 3)
    K = None
    # ideal ions, so any concentration is legitimate; high ones give statistics
    for conc, seed in ((0.05, 31), (0.20, 32)):
        s = IonSampler(params, lig, box, [True] * 3, I, np.random.default_rng(seed),
                       bath={"Ca": conc}, interacting=False, local_moves_per_ion=8)
        if K is None:
            ref = association_constants(params, "Ca", lig, I, np.random.default_rng(99),
                                        n_samples=3_000_000, coulomb_cutoff=s.coulomb_cutoff)
            K = ref.K_total
            K_err = math.sqrt(sum(v ** 2 for v in ref.err.values()))
        s.sweep(3000)
        nb = bound_counts(s, an, 40000, 10)
        mean_nb, err_nb = block_mean_and_error(nb)
        assert err_nb < 0.08 * K * conc
        assert abs(mean_nb - K * conc) < 5 * math.hypot(err_nb, K_err * conc)
        p_bound, err_p = block_mean_and_error(nb > 0)
        assert abs(p_bound - (1.0 - math.exp(-K * conc))) < 5 * math.hypot(err_p, K_err * conc)


def test_interacting_isotherm_follows_the_langmuir_form_at_low_coverage(params):
    """With ion-ion repulsion on, one acetate holds at most one calcium to a
    good approximation, and the occupancy follows theta = K c / (1 + K c). At
    the concentrations used the Langmuir and Poisson forms differ by less than
    2%, and ion-ion correlations (not in either form) shift the result by a
    few per cent, so the comparison is made to 8% plus the sampling error.
    """
    box = np.array([3.2, 3.2, 3.2])
    lig = acetate().translated(box / 2)
    I = 0.15
    an = BridgeAnalyzer(params, lig, box, [True] * 3)
    s0 = IonSampler(params, lig, box, [True] * 3, I, np.random.default_rng(1))
    K = association_constants(params, "Ca", lig, I, np.random.default_rng(98), n_samples=3_000_000,
                              coulomb_cutoff=s0.coulomb_cutoff).K_total
    thetas = []
    for conc, seed in ((0.005, 41), (0.02, 42)):
        # a tenth of an ion in the box on average: give the few ions many local
        # moves per sweep, or a bound one outlasts the run
        s = IonSampler(params, lig, box, [True] * 3, I, np.random.default_rng(seed), bath={"Ca": conc},
                       local_moves_per_ion=20)
        s.sweep(3000)
        nb = bound_counts(s, an, 40000, 20)
        theta, err = block_mean_and_error(nb > 0)
        assert err < 0.10 * langmuir_guess(K, conc)
        langmuir = K * conc / (1.0 + K * conc)
        assert abs(theta - langmuir) < 5 * err + 0.08 * langmuir
        thetas.append(theta)
    assert thetas[1] > 2.5 * thetas[0]      # occupancy rises with concentration


# ------------------------------------------------------------ bookkeeping
def mixed_system(params, seed, **kw):
    box = np.array([4.0, 4.0, 6.0])
    parts = []
    rng = np.random.default_rng(seed)
    for m in range(6):
        c = rng.uniform([0.5, 0.5, 0.5], box - 0.5)
        u = rng.normal(size=3)
        kind = ["carboxylate", "guanidinium", "ammonium"][m % 3]
        parts.append(geo.GROUP_BUILDERS[kind](c, u / np.linalg.norm(u), m % 2))
    lig = geo.LigandSet.concat(parts)
    s = IonSampler(params, lig, box, [True, True, True], 0.1, np.random.default_rng(seed + 1),
                   bath={"Ca": 0.05, "Mg": 0.05, "SO4": 0.05}, **kw)
    return s


def test_incremental_energies_are_consistent_with_the_total(params):
    """sum_i e_one(i) counts every pair twice and every one-body term once:
    total = one-body + pairs, so sum_i e_one(i) = 2 total - one-body. With
    interactions off, the total is the one-body sum."""
    s = mixed_system(params, 3)
    s.sweep(300)
    n = s.n_ions
    assert n >= 3 and s.count("SO4") >= 1
    per_ion = sum(s.ion_energy(i) for i in range(n))
    total = s.energy()
    s_ideal = mixed_system(params, 3, interacting=False)
    for arr_new, arr_old in zip(s_ideal.I, s.I):
        arr_new[: len(arr_old)] = arr_old[: len(arr_new)]
    s_ideal._counters[:] = s._counters
    one_body = s_ideal.energy()
    assert per_ion == pytest.approx(2 * total - one_body, rel=1e-9, abs=1e-8)


def test_cell_list_agrees_with_a_single_cell(params):
    """Same configuration evaluated with the default cells and with one cell
    spanning the box (a cutoff too long to subdivide the box)."""
    s = mixed_system(params, 4)
    s.sweep(200)
    assert s.S.ncell.prod() > 1
    big = np.array([4.0, 4.0, 6.0])
    from modcoll.ions.sampler import build_system, total_energy
    S1, _ = build_system(s.ligands, big, [True] * 3, None, 1.9)   # floor(4 / 1.9) = 2 < 3 cells
    assert S1.ncell.max() <= 3
    S1 = S1._replace(reach=s.S.reach)
    e_cells = s.energy()
    e_one_cell = float(total_energy(s.T, s.E, S1, s.I, s.n_ions, True))
    assert e_cells == pytest.approx(e_one_cell, rel=1e-10, abs=1e-9)


def test_energy_is_invariant_under_a_lattice_translation_of_one_ion(params):
    s = mixed_system(params, 6)
    s.sweep(200)
    e0 = s.energy()
    for i in range(s.n_ions):
        s.I.pos[i] += np.array([4.0, -4.0, 6.0]) * (1 if i % 2 else -1)
    assert s.energy() == pytest.approx(e0, rel=1e-10, abs=1e-9)


def test_same_seed_same_trajectory_and_different_seed_differs(params):
    a = mixed_system(params, 12)
    b = mixed_system(params, 12)
    a.sweep(150)
    b.sweep(100)
    b.sweep(50)   # chunking must not matter
    sa, sb = a.snapshot(), b.snapshot()
    assert np.array_equal(sa.pos, sb.pos) and np.array_equal(sa.state, sb.state)
    assert np.array_equal(sa.uid, sb.uid)
    c = mixed_system(params, 13)
    c.sweep(150)
    sc = c.snapshot()
    assert sc.pos.shape != sa.pos.shape or not np.allclose(sc.pos, sa.pos)


def test_rng_must_be_an_explicit_generator(params):
    with pytest.raises(TypeError):
        IonSampler(params, empty(), [3, 3, 3], [True] * 3, 0.1, rng=None)
    with pytest.raises(TypeError):
        IonSampler(params, empty(), [3, 3, 3], [True] * 3, 0.1, rng=np.random)


def test_storage_grows_and_uids_stay_unique(params):
    p = params.with_overrides({"species.Ca.charge": 0.0})
    s = IonSampler(p, empty(), [5.0, 5.0, 5.0], [True] * 3, 0.1, np.random.default_rng(2),
                   bath={"Ca": 0.3}, capacity=2)
    s.sweep(400)
    snap = s.snapshot()
    assert len(snap.uid) > 8
    assert len(set(snap.uid.tolist())) == len(snap.uid)


def test_walls_confine_ions_on_non_periodic_axes(params):
    p = params.with_overrides({"species.Ca.charge": 0.0})
    box = np.array([3.0, 3.0, 4.0])
    s = IonSampler(p, empty(), box, [False, False, True], 0.1, np.random.default_rng(3),
                   bath={"Ca": 0.2})
    for _ in range(50):
        s.sweep(20)
        pos = s.snapshot().pos
        assert np.all(pos >= 0.0) and np.all(pos <= box)


def test_excluded_cylinder_density_profile_integrates_to_the_boltzmann_value(params):
    """A z-parallel cylinder of radius R repels ion centres through a WCA wall
    in the gap g = rho - R, with onset at g = wall_radius. For an uncharged
    ideal gas the density is c N_A exp(-u(g)), so

        <N> = c N_A Lz [ A_box - pi (R + Rw)^2 + int_R^(R+Rw) 2 pi rho exp(-u) d rho ],

    the last integral done here by an independent trapezoid rule on the WCA
    formula written out by hand.
    """
    p = params.with_overrides({"species.Ca.charge": 0.0})
    box = np.array([4.0, 4.0, 4.0])
    R, conc = 0.8, 0.1
    s = IonSampler(p, empty(), box, [True] * 3, 0.1, np.random.default_rng(4), bath={"Ca": conc},
                   cylinders=[[2.0, 2.0, R]])
    s.sweep(1000)
    counts, rho_min = [], np.inf
    for _ in range(20000):
        s.sweep(1)
        pos = s.snapshot().pos
        counts.append(len(pos))
        if len(pos):
            rho_min = min(rho_min, np.hypot(pos[:, 0] - 2.0, pos[:, 1] - 2.0).min())
    rw = float(p.value("species.Ca.wall_radius"))
    assert rho_min > R + 0.5 * rw                      # nothing deep inside the wall
    g = np.linspace(1e-4, rw, 4001)
    sigma = rw / 2 ** (1 / 6)
    u = 4 * ((sigma / g) ** 12 - (sigma / g) ** 6) + 1.0
    f = 2 * math.pi * (R + g) * np.exp(-u)
    layer = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(g)))
    expected = conc * PER_NM3_PER_MOLAR * 4.0 * (16.0 - math.pi * (R + rw) ** 2 + layer)
    mean_n, err_n = block_mean_and_error(counts)
    assert err_n < 0.02 * expected
    assert abs(mean_n - expected) < 5 * err_n


def test_cutoff_longer_than_half_a_periodic_box_is_refused(params):
    with pytest.raises(ValueError):
        IonSampler(params, empty(), [3.0, 3.0, 3.0], [True] * 3, 0.1, np.random.default_rng(1),
                   coulomb_cutoff=2.0)


def test_ligands_outside_the_box_are_refused(params):
    lig = geo.carboxylate([5.0, 1.0, 1.0], [1, 0, 0])
    with pytest.raises(ValueError):
        IonSampler(params, lig, [3.0, 3.0, 3.0], [True] * 3, 0.1, np.random.default_rng(1))


def test_overlapping_groups_trigger_a_warning(params):
    a = geo.carboxylate([1.5, 1.5, 1.5], [1, 0, 0], mol=0)
    b = geo.carboxylate([1.5, 1.5, 1.6], [1, 0, 0], mol=1)       # 0.1 nm away: unphysical
    with pytest.warns(UserWarning, match="share one coordination"):
        IonSampler(params, geo.LigandSet.concat([a, b]), [3.5, 3.5, 3.5], [True] * 3, 0.1,
                   np.random.default_rng(1))
    import warnings as w
    c = geo.carboxylate([1.5, 1.5, 2.2], [1, 0, 0], mol=1)       # well separated: silent
    with w.catch_warnings():
        w.simplefilter("error")
        IonSampler(params, geo.LigandSet.concat([a, c]), [3.5, 3.5, 3.5], [True] * 3, 0.1,
                   np.random.default_rng(1))


@pytest.mark.parametrize("n_sites", [6, 8])
def test_calcium_variants_run_and_keep_the_bath_density(n_sites):
    """The six- and eight-site calcium are the same model with another frame;
    an uncharged gas of either must still sit at the bath density."""
    p = load_params(ca_coordination=n_sites, overrides={"species.Ca.charge": 0.0})
    s = IonSampler(p, empty(), [4.0, 4.0, 4.0], [True] * 3, 0.1, np.random.default_rng(60 + n_sites),
                   bath={"Ca": 0.05})
    s.sweep(1000)
    counts = []
    for _ in range(15000):
        s.sweep(2)
        counts.append(s.n_ions)
    mean_n, err_n = block_mean_and_error(counts)
    assert abs(mean_n - 0.05 * PER_NM3_PER_MOLAR * 64.0) < 5 * err_n
    assert s.snapshot().state.shape[1] >= n_sites


def test_flip_with_displacement_keeps_free_ion_statistics(params):
    """With flips that also move the ion (the default) a lone, uncharged
    'calcium' must still have independent two-state sites and sit at the bath
    density; with the displacement switched off the ion's position must not
    change under flip moves alone (covered in the two-state test)."""
    dg = 1.5
    p = params.with_overrides({"species.Ca.charge": 0.0, "species.Ca.dg_des": dg})
    s = IonSampler(p, empty(), [3.0, 3.0, 3.0], [True] * 3, 0.1, np.random.default_rng(909),
                   move_probabilities=(0.0, 0.0, 1.0, 0.0), flip_move_probability=1.0)
    start = np.array([1.5, 1.5, 1.5])
    s.add_ion("Ca", start)
    frac = []
    for _ in range(20000):
        s.sweep(5)
        frac.append(s.snapshot().state[0, :7].mean())
    mean_l, err_l = block_mean_and_error(frac)
    assert abs(mean_l - 1.0 / (1.0 + math.exp(dg))) < 5 * err_l and err_l < 0.01
    assert np.linalg.norm(s.snapshot().pos[0] - start) > 0.1      # it did move
    with pytest.raises(ValueError):
        IonSampler(p, empty(), [3.0, 3.0, 3.0], [True] * 3, 0.1, np.random.default_rng(1),
                   local_moves_per_ion=0)
