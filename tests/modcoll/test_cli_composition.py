"""Composition scan command: inverse problem, sensitivity, stagger stage.

The grids here are tiny. Oracles: closed forms for the small helpers, the
library functions for the tables, ``cli_stagger`` for the stagger stage (a
pure-chemistry composition must give exactly what the S2 tool gives), and
the reproducibility of the written files.
"""
import csv
import json

import numpy as np
import pytest

from modcoll import cli_composition as cli
from modcoll import cli_stagger
from modcoll import composition as co
from modcoll import electrostatics as es
from modcoll.sites import SiteType, native_molecule


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- helpers ---
def test_axis_crossing_interpolates_linearly():
    f = [0.0, 0.5, 1.0]
    assert cli.axis_crossing(f, [5.0, 4.6, 4.2], 4.4) == pytest.approx(0.75)
    assert cli.axis_crossing(f, [5.0, 4.6, 4.2], 4.6) == pytest.approx(0.5)
    assert cli.axis_crossing(f, [4.0, 3.9, 3.8], 4.4) == 0.0          # already below
    assert np.isnan(cli.axis_crossing(f, [5.0, 4.9, 4.8], 4.4))        # never reached


def test_smallest_band():
    out = cli.smallest_band(np.array([4.2, 4.31, 4.05, 3.95, 4.6]), 4.2, [0.1, 0.2, 0.3])
    assert out[:4].tolist() == [0.1, 0.2, 0.2, 0.3] and np.isnan(out[4])


def test_crossover_salt_closed_forms():
    salts = [10.0, 50.0, 150.0]
    # halfway between 10 and 50 mM on a log axis: sqrt(10 * 50)
    value, status = cli.crossover_salt_mM(salts, [0.0, 1.0, 1.0])
    assert status == "bracketed" and value == pytest.approx(np.sqrt(500.0))
    # a quarter of the way from 50 to 150
    value, _ = cli.crossover_salt_mM(salts, [0.0, 0.25, 1.25])
    assert value == pytest.approx(50.0 * 3.0 ** 0.25)
    # order of the input does not matter
    assert cli.crossover_salt_mM(salts[::-1], [1.0, 1.0, 0.0])[0] == pytest.approx(np.sqrt(500.0))
    assert cli.crossover_salt_mM(salts, [0.9, 1.0, 1.0]) == (10.0, "below_range")
    value, status = cli.crossover_salt_mM(salts, [0.0, 0.1, 0.4])
    assert np.isnan(value) and status == "above_range"
    assert cli.crossover_salt_mM(salts, [0.0, 0.9, 0.2])[1] == "not_monotonic"
    # a dip at low salt: the last upward crossing counts
    value, status = cli.crossover_salt_mM([10.0, 25.0, 50.0, 150.0], [0.6, 0.0, 1.0, 1.0])
    assert status == "bracketed" and value == pytest.approx(np.sqrt(25.0 * 50.0))


def test_minimum_succinylation_curve_is_cumulative_in_the_cap():
    p = co.composition_grid(0.5)
    ok = np.zeros(len(p), bool)
    ok[np.all(p == (1.0, 0.0, 0.0), axis=1)] = True      # f_deam 0: needs f_succ 1
    ok[np.all(p == (0.5, 0.5, 0.5), axis=1)] = True      # f_deam 0.5: 0.5 is enough
    caps, curve = cli.min_succinylation_curve(p, ok)
    assert caps.tolist() == [0.0, 0.5, 1.0] and curve.tolist() == [1.0, 0.5, 0.5]
    assert np.isnan(cli.min_succinylation_curve(p, np.zeros(len(p), bool))[1]).all()


def test_variants_cover_the_full_factorial_once():
    v = cli.variants()
    assert len(v) == 3 * 3 * 3 * 2 * 2 == len({tuple(sorted(x.items())) for x in v})
    assert v[0] == cli.BASE_VARIANT
    assert {x["succinyl_pKa"] for x in v} == {4.35, 4.5, 4.65}
    assert {x["asp_pKa"] for x in v} == {3.65, 3.9, 4.0}
    assert {x["chain_ends"] for x in v} == {"mature", "pepsin"}


def test_variant_counts_remove_four_histidines_per_molecule():
    native = native_molecule()
    nat = co.type_counts(native)
    sums = co.sample_counts(native, 0.5, 0.2, 0.3, 6, seed=1).sum(axis=0)[None, :]
    s, n = cli.variant_counts(sums, nat, 6, "bovine_porcine")
    explicit = co.sample_counts(co.without_alpha2_histidines(native), 0.5, 0.2, 0.3, 6,
                                seed=1).sum(axis=0)
    assert np.array_equal(s[0], explicit)
    assert np.array_equal(n, co.type_counts(co.without_alpha2_histidines(native)))
    assert sums[0, int(SiteType.HIS)] == 6 * 18          # the input is not altered
    same, _ = cli.variant_counts(sums, nat, 6, "human")
    assert np.array_equal(same, sums)
    with pytest.raises(ValueError):
        cli.variant_counts(sums, nat, 6, "chicken")


# ---------------------------------------------------------------- sampling ---
def test_sample_map_matches_the_library_and_ignores_the_worker_count():
    native = native_molecule()
    points = co.composition_grid(0.5, extra_f_deam=[co.LIMING_F_DEAM])
    sums1, stats1 = cli.sample_map(native, points, 6, seed=4, workers=1, n_boot=20)
    sums3, stats3 = cli.sample_map(native, points, 6, seed=4, workers=3, n_boot=20)
    assert np.array_equal(sums1, sums3) and stats1 == stats3
    i = int(np.flatnonzero(np.all(points == (0.5, 0.5, 0.18), axis=1))[0])
    counts = co.sample_counts(native, 0.5, 0.5, 0.18, 6, seed=4)
    assert np.array_equal(sums1[i], counts.sum(axis=0))
    expect = co.point_statistics(counts, pH_values=(4.2, 7.0),
                                 length_nm=co.contour_length_nm(native),
                                 rng=co.bootstrap_rng(4, 0.5, 0.5, 0.18), n_boot=20)
    assert stats1[i] == expect
    only_sums, none = cli.sample_map(native, points[:3], 6, seed=4, statistics=False)
    assert none is None and np.array_equal(only_sums, sums1[:3])


# ------------------------------------------------------------- the command ---
ARGS = ["--step", "0.25", "--coarse-step", "0.5", "--assay-step", "0.25", "--molecules", "24",
        "--bootstrap", "20", "--stages", "map,inverse,sensitivity,assays", "--no-figures"]


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    out = tmp_path_factory.mktemp("composition")
    code = cli.run(ARGS + ["--seed", "5", "--workers", "1", "--outdir", str(out)])
    return code, out


def test_command_writes_its_tables(scan):
    code, out = scan
    assert code == 0
    for name in ("forward_map.csv", "pure_chemistries.csv", "region_summary.csv",
                 "consistent_region.csv", "min_succinylation.csv", "solubility_threshold.csv",
                 "sensitivity.csv", "sensitivity_statements.csv",
                 "sensitivity_min_succinylation.csv", "assays.csv", "assay_ranking.csv",
                 "assay_sets.csv", "assays_representatives.csv", "titration_curves.csv",
                 "run.json",
                 "raw/forward_map_fine.csv", "raw/sampled_counts.npz"):
        assert (out / name).exists(), name
    fine = read_csv(out / "raw" / "forward_map_fine.csv")
    assert len(fine) == 15 * 6                        # quarters, plus the liming slice
    coarse = read_csv(out / "forward_map.csv")
    assert len(coarse) == 6 * 4                       # halves: 6 amine splits, 3 + liming
    liming = [r for r in coarse if r["liming_reference"] == "True"]
    assert len(liming) == 6 and {float(r["f_deam"]) for r in liming} == {0.18}
    for r in fine:
        assert int(r["n_molecules"]) == 24
        for key in ("pI_ensemble_se", "pI_molecule_sd", "pI_molecule_se",
                    "net_charge_pH4.2_se", "net_charge_pH7_se", "charge_per_nm_pH4.2_mean",
                    "charge_per_nm_pH7_se"):
            assert key in r
    run = json.loads((out / "run.json").read_text())
    assert run["seed"] == 5 and run["molecules_per_composition"] == 24
    assert run["pKa"]["SUCC_LYS"] == 4.5 and run["pKa"]["ASP"] == 3.9
    assert run["native"]["pI"] == pytest.approx(9.84, abs=0.005)
    assert run["liming_f_deam"] == 0.18


def test_forward_map_limits_and_charge_per_length(scan):
    _, out = scan
    fine = read_csv(out / "raw" / "forward_map_fine.csv")
    run = json.loads((out / "run.json").read_text())

    def at(fs, fa, fd):
        return next(r for r in fine if (float(r["f_succ"]), float(r["f_acet"]),
                                        float(r["f_deam"])) == (fs, fa, fd))

    native = at(0, 0, 0)
    assert float(native["pI_ensemble"]) == pytest.approx(9.84174, abs=1e-4)
    assert float(native["net_charge_pH4.2_mean"]) == pytest.approx(159.042, abs=1e-3)
    assert float(native["pI_molecule_sd"]) == 0 and float(native["net_charge_pH4.2_se"]) == 0
    assert float(at(1, 0, 0)["pI_ensemble"]) == pytest.approx(4.28809, abs=1e-4)
    assert float(at(0, 1, 0)["pI_ensemble"]) == pytest.approx(4.64, abs=0.005)   # S1: 4.64
    assert float(at(0, 0, 1)["pI_ensemble"]) == pytest.approx(4.74, abs=0.005)   # S1: 4.74
    # 107 lysines, each +1 -> -1 at pH 7
    assert (float(native["net_charge_pH7_mean"]) - float(at(1, 0, 0)["net_charge_pH7_mean"])
            == pytest.approx(2 * 107, abs=1.0))
    mixed = at(0.5, 0.25, 0.5)
    assert float(mixed["pI_molecule_sd"]) > 0 and float(mixed["pI_ensemble_se"]) > 0
    assert float(mixed["net_charge_pH4.2_se"]) == pytest.approx(
        float(mixed["net_charge_pH4.2_sd"]) / np.sqrt(24), rel=1e-4)
    assert float(mixed["charge_per_nm_pH7_mean"]) == pytest.approx(
        float(mixed["net_charge_pH7_mean"]) / run["native"]["contour_length_nm"], rel=1e-4)
    assert float(mixed["mean_succinylated"]) == pytest.approx(
        0.5 * 107, abs=4 * np.sqrt(107 / 4 / 24))


def test_inverse_problem_tables(scan):
    _, out = scan
    summary = {float(r["band"]): r for r in read_csv(out / "region_summary.csv")}
    assert sorted(summary) == [0.1, 0.2, 0.3]
    n = [int(summary[b]["n_points"]) for b in (0.1, 0.2, 0.3)]
    assert 0 < n[0] <= n[1] <= n[2] < 90
    for b in (0.1, 0.2, 0.3):
        r = summary[b]
        assert r["pure_acetylation_reaches"] == "False"       # stops at 4.64
        assert r["pure_deamidation_reaches"] == "False"       # stops at 4.74
        assert float(r["pure_acetylation_lowest_pI"]) == pytest.approx(4.64, abs=0.005)
        assert int(r["n_excluded_as_insoluble"]) == 0
        # every consistent composition is far more charged at pH 7 than native
        assert float(r["abs_net_charge_pH7_min"]) > float(r["soluble_threshold_e"]) > 60
    # complete succinylation gives 4.288: inside +-0.1 (4.1 ... 4.3)
    assert summary[0.1]["pure_succinylation_reaches"] == "True"
    assert float(summary[0.3]["pure_succinylation_f_reaching_band"]) < float(
        summary[0.1]["pure_succinylation_f_reaching_band"]) <= 1.0
    region = read_csv(out / "raw" / "consistent_region_fine.csv")
    assert len(region) == n[2]
    assert all(abs(float(r["pI_ensemble"]) - 4.2) <= 0.3 + 1e-9 for r in region)
    assert sum(float(r["band"]) == 0.1 for r in region) == n[0]
    # the least succinylation falls as more deamidation is allowed
    rows = [r for r in read_csv(out / "min_succinylation.csv") if float(r["band"]) == 0.2
            and r["reached"] == "True"]
    values = [float(r["f_succ_min"]) for r in sorted(rows, key=lambda r: float(r["f_deam_cap"]))]
    assert values == sorted(values, reverse=True) and values[-1] == 0.0
    # raising the solubility threshold can only remove compositions
    sol = [r for r in read_csv(out / "solubility_threshold.csv") if float(r["band"]) == 0.3]
    sol.sort(key=lambda r: float(r["soluble_threshold_e"]))
    kept = [int(r["n_consistent"]) for r in sol]
    assert kept == sorted(kept, reverse=True) and kept[0] == n[2] and kept[-1] < n[2]
    assert sum(r["is_default"] == "True" for r in sol) == 1


def test_sensitivity_tables(scan):
    _, out = scan
    rows = read_csv(out / "sensitivity.csv")
    assert len(rows) == 108 * 3
    base = [r for r in rows if r["is_base"] == "True"]
    assert len(base) == 3 and base[0]["chain_ends"] == "mature" and base[0]["species"] == "human"
    summary = {float(r["band"]): r for r in read_csv(out / "region_summary.csv")}
    for r in base:                                   # the base variant is the inverse table
        assert int(r["n_points"]) == int(summary[float(r["band"])]["n_points"])
    # a more acidic Asp lowers the isoelectric point everywhere
    def pick(**kw):
        want = dict(cli.BASE_VARIANT, **kw)
        return next(r for r in rows if float(r["band"]) == 0.2 and all(
            (float(r[k]) == v if k.endswith("pKa") else r[k] == v) for k, v in want.items()))
    key = "pure_succinylation_lowest_pI"
    assert float(pick(asp_pKa=3.65)[key]) < float(pick()[key]) < float(pick(asp_pKa=4.0)[key])
    assert float(pick(succinyl_pKa=4.35)["pure_succinylation_lowest_pI"]) < float(
        pick(succinyl_pKa=4.65)["pure_succinylation_lowest_pI"])
    # four fewer histidines: fewer positive charges, lower isoelectric point
    assert float(pick(species="bovine_porcine")["native_pI"]) < float(pick()["native_pI"])
    # the documented pepsin value: fully succinylated 4.27 against 4.29 intact
    assert float(pick(chain_ends="pepsin")["pure_succinylation_lowest_pI"]) == pytest.approx(
        4.27, abs=0.006)
    statements = read_csv(out / "sensitivity_statements.csv")
    assert len(statements) == 3 * len(cli.STATEMENTS)
    deam = [s for s in statements if s["quantity"] == "pure_deamidation_reaches"]
    assert all(s["base"] == "False" and s["true_in_variants"] == "0"
               and s["same_in_all_variants"] == "True" and s["n_variants"] == "108" for s in deam)
    curves = read_csv(out / "sensitivity_min_succinylation.csv")
    for r in curves:
        if r["f_succ_min_base"]:
            assert float(r["f_succ_min_lowest_variant"]) <= float(r["f_succ_min_base"]) <= float(
                r["f_succ_min_highest_variant"])
    one = cli.one_at_a_time([dict(r, band=float(r["band"]), **{
        k: (float(r[k]) if k.endswith("pKa") else r[k]) for k in cli.BASE_VARIANT})
        for r in rows], 0.2)
    # base + 2 other values for each pKa + 1 for chain ends + 1 for species
    assert len(one) == 1 + 2 * 3 + 1 + 1


def test_assay_tables(scan):
    _, out = scan
    assays = read_csv(out / "raw" / "assays_fine.csv")
    region = read_csv(out / "raw" / "consistent_region_fine.csv")
    assert len(assays) == len(region)
    for r in assays:
        free = float(r["free_amine_fraction"])
        assert free == pytest.approx(1 - float(r["f_succ"]) - float(r["f_acet"]), abs=0.12)
        assert float(r["amide_per_molecule"]) == pytest.approx(125 * (1 - float(r["f_deam"])),
                                                               abs=12)
        assert float(r["mass_shift_Da"]) == pytest.approx(
            100.016 * float(r["succinyl_per_molecule"]) + 42.011 * float(r["acetyl_per_molecule"])
            + 0.984 * float(r["deamidated_per_molecule"]), rel=1e-5)
        # only carboxyls titrate at the zero crossing: the salt shift is the
        # same for every candidate (-0.074 pH from 10 to 150 mM)
        assert float(r["pI_salt_shift"]) == pytest.approx(-0.0744, abs=5e-4)
    ranking = [r for r in read_csv(out / "assay_ranking.csv") if float(r["band"]) == 0.3]
    assert [int(r["rank"]) for r in ranking] == list(range(1, len(co.ASSAYS) + 1))
    shares = [float(r["pairs_separated"]) for r in ranking]
    assert shares == sorted(shares, reverse=True)
    by = {r["assay"]: r for r in ranking}
    assert float(by["pI_salt_shift"]["pairs_separated"]) == 0.0
    for r in ranking:                                # a sharper assay separates more
        assert float(r["pairs_separated_sigma_x0.5"]) >= float(r["pairs_separated"]) >= float(
            r["pairs_separated_sigma_x2"])
    sets = [r for r in read_csv(out / "assay_sets.csv") if float(r["band"]) == 0.3]
    best_pair = max(float(r["pairs_separated"]) for r in sets if r["n_assays"] == "2")
    best_triple = max(float(r["pairs_separated"]) for r in sets if r["n_assays"] == "3")
    assert best_triple >= best_pair >= shares[0]
    reps = read_csv(out / "assays_representatives.csv")
    for r in reps:                                   # the same numbers as the big table
        match = next(a for a in assays if all(a[k] == r[k] for k in ("f_succ", "f_acet",
                                                                     "f_deam")))
        assert r["mass_shift_Da"] == match["mass_shift_Da"] and r["pI"] == match["pI"]
    curves = read_csv(out / "titration_curves.csv")
    assert {r["composition"] for r in curves} == {r["composition"] for r in reps} | {"-1"}
    native = [r for r in curves if r["composition"] == "-1"]
    assert len(native) == 101
    at7 = next(float(r["net_charge_mean"]) for r in native if float(r["pH"]) == 7.0)
    assert at7 == pytest.approx(es.net_charge(native_molecule(), es.Environment(7.0)), abs=1e-3)


def test_command_is_reproducible(scan, tmp_path):
    _, out = scan
    code = cli.run(ARGS + ["--seed", "5", "--workers", "2", "--outdir", str(tmp_path)])
    assert code == 0
    for name in ("forward_map.csv", "region_summary.csv", "consistent_region.csv",
                 "min_succinylation.csv", "sensitivity.csv", "sensitivity_statements.csv",
                 "assays.csv", "assay_ranking.csv", "titration_curves.csv"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes(), name
    other = tmp_path / "other"
    cli.run(["--step", "0.5", "--coarse-step", "0.5", "--assay-step", "0.5", "--molecules", "8",
             "--bootstrap", "5", "--stages", "map", "--no-figures", "--seed", "6",
             "--workers", "1", "--outdir", str(other)])
    assert (other / "forward_map.csv").exists() and not (other / "region_summary.csv").exists()
    with pytest.raises(SystemExit):
        cli.run(["--stages", "map,nonsense", "--outdir", str(other)])


def test_figures_are_written(tmp_path):
    code = cli.run(["--step", "0.25", "--coarse-step", "0.5", "--assay-step", "0.25",
                    "--molecules", "8", "--bootstrap", "5",
                    "--stages", "map,inverse,sensitivity,assays", "--seed", "5",
                    "--workers", "1", "--outdir", str(tmp_path)])
    assert code == 0
    for name in ("fig_forward_map.png", "fig_consistent_region.png", "fig_min_succinylation.png",
                 "fig_sensitivity.png", "fig_assays.png", "fig_titration.png"):
        assert (tmp_path / name).stat().st_size > 10_000, name


# ---------------------------------------------------------- stagger stage ---
STAG = dict(ph=[4.2], ionic_M=[0.05, 0.15], pairs=2, seed=7, orientations=("parallel",))


@pytest.fixture(scope="module")
def stagger(tmp_path_factory):
    native = native_molecule()
    result = cli.stagger_scan(native, [(0.9, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.5)],
                              **STAG)
    out = tmp_path_factory.mktemp("s2")
    code = cli_stagger.run(["--chemistry", "succ", "--f", "0.9,1", "--ph", "4.2",
                            "--ionic-mM", "50,150", "--pairs", "2", "--seed", "7",
                            "--orientations", "parallel", "--workers", "1", "--no-figures",
                            "--outdir", str(out)])
    assert code == 0
    return result, read_csv(out / "summary.csv")


def test_stagger_stage_reproduces_cli_stagger_for_a_pure_chemistry(stagger):
    """Compositions (0.9, 0, 0) and (1, 0, 0) are succinylation at f = 0.9
    and 1: the stage must give what the S2 tool writes for them."""
    (summary, minima, pairs, gate), reference = stagger
    assert gate["passed"] and gate["pka_set"] == "repository"
    for comp, f in ((0, 0.9), (1, 1.0)):
        for salt in (50.0, 150.0):
            mine = next(r for r in summary if r["composition"] == comp
                        and r["ionic_mM"] == pytest.approx(salt))
            ref = next(r for r in reference if float(r["f"]) == f
                       and float(r["ionic_mM"]) == salt)
            # one pattern at f = 1: the stage evaluates it once, the S2 tool
            # evaluates the same pair twice
            assert mine["n_pairs"] == (int(ref["n_pairs"]) if f < 1.0 else 1)
            for key in ("net_charge_mean", "deepest_D", "deepest_kT", "deepest_pair_sd_kT",
                        "comb_z_mean_profile", "el_comb_z_mean_profile", "dominant_period_nm",
                        "deepest_kT_pairs_mean", "deepest_kT_pairs_sd",
                        "pairs_deepest_at_multiple_mean"):
                assert mine[key] == pytest.approx(float(ref[key] or "nan"), rel=2e-5, abs=1e-6,
                                                  nan_ok=True), key
            assert (mine["f_succ"], mine["f_acet"], mine["f_deam"]) == (f, 0.0, 0.0)
            assert "f" not in mine and "chemistry" not in mine


def test_stagger_stage_bookkeeping(stagger):
    (summary, minima, pairs, _), _ = stagger
    assert len(summary) == 3 * 1 * 2
    # full succinylation has one pattern: one pair, no spread; the others two
    n = {r["composition"]: r["n_pairs"] for r in summary}
    assert n == {0: 2, 1: 1, 2: 2}
    full = [r for r in summary if r["composition"] == 1]
    assert all(r["deepest_kT_pairs_sd"] == 0.0 for r in full)
    assert {(r["composition"], r["pair"]) for r in pairs} == {(0, 0), (0, 1), (1, 0),
                                                             (2, 0), (2, 1)}
    third = next(r for r in pairs if r["composition"] == 2)
    assert (third["f_succ"], third["f_acet"], third["f_deam"]) == (0.0, 1.0, 0.5)
    assert {r["profile"] for r in minima} == {"tot_min", "tot_boltz", "el_min"}
    again = cli.stagger_scan(native_molecule(), [(0.0, 1.0, 0.5)], workers=2, **STAG)[0]
    for a, b in zip(again, [r for r in summary if r["composition"] == 2]):
        assert a["deepest_kT_pairs_mean"] == pytest.approx(b["deepest_kT_pairs_mean"], rel=1e-9)


def _row(comp, pH, salt, share, deepest_d=1.0, period=66.9, depth=-20.0, sd=1.0, el_z=0.2):
    return {"composition": comp, "orientation": "parallel", "pH": pH, "ionic_mM": salt,
            "f_succ": 0.5, "f_acet": 0.0, "f_deam": 0.0, "net_charge_mean": 0.0,
            "pairs_deepest_at_multiple_mean": share, "deepest_D": deepest_d,
            "dominant_period_nm": period, "deepest_kT_pairs_mean": depth,
            "deepest_kT_pairs_sd": sd, "el_comb_z_mean_profile": el_z}


def test_stagger_conclusions_on_a_synthetic_summary():
    """Two made-up compositions: 0 behaves as S2 describes; 1 keeps the D
    register at low salt, has another period and a shallower minimum."""
    rows = []
    for pH in (4.2, 4.6):
        rows += [_row(0, pH, 10.0, 0.0, deepest_d=0.58), _row(0, pH, 50.0, 1.0),
                 _row(0, pH, 150.0, 1.0),
                 _row(1, pH, 10.0, 0.8, deepest_d=2.0), _row(1, pH, 50.0, 1.0),
                 _row(1, pH, 150.0, 1.0, period=91.0, depth=-12.0, el_z=1.4)]
    out = {r["conclusion"]: r for r in cli.stagger_conclusions(
        rows, d_nm=66.9, target_pH=4.2, own_pI={0: 4.25, 1: 4.55})}
    holds = {k: v["holds_at_all_compositions"] for k, v in out.items()}
    assert holds["register_at_high_salt"] and holds["register_is_1D"]
    assert not holds["period_at_high_salt"]
    assert out["period_at_high_salt"]["compositions_where_it_fails"] == "1"
    assert out["period_at_high_salt"]["c1"] == 91.0
    assert not holds["depth_at_high_salt"]                       # 8 kT apart
    assert (out["depth_at_high_salt"]["min_over_compositions"],
            out["depth_at_high_salt"]["max_over_compositions"]) == (-20.0, -12.0)
    assert not holds["electrostatics_does_not_select_D"]
    assert not holds["register_lost_at_low_salt"]
    assert out["register_lost_at_low_salt"]["compositions_where_it_fails"] == "1"
    assert out["register_lost_at_low_salt"]["c0"] == 0.0
    assert not holds["low_salt_register"]
    # composition 0 crosses between 10 and 50 mM; composition 1 never drops below one half
    cross = out["crossover_salt_pH4.2"]
    assert cross["c0"] == pytest.approx(np.sqrt(500.0)) and cross["c1"] == 10.0
    assert not cross["holds_at_all_compositions"] and "c1:below_range" in cross["note"]
    assert out["crossover_salt_own_pI"]["c0"] == pytest.approx(np.sqrt(500.0))
    assert holds["pair_spread_150mM"] and holds["pair_spread_10mM"]
    assert "no_polarity_selection" not in out                    # no antiparallel rows
    # with composition 1 behaving like composition 0, everything holds
    same = [dict(r, composition=1) for r in rows if r["composition"] == 0] + [
        r for r in rows if r["composition"] == 0]
    assert all(r["holds_at_all_compositions"] for r in cli.stagger_conclusions(same, 66.9))
    table = cli.crossover_table(rows)
    assert len(table) == 4 and {r["status"] for r in table} == {"bracketed", "below_range"}
