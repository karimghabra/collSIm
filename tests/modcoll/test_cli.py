"""Command-line tools: outputs, reproducibility, and the native gate.

The grids here are tiny; the science is tested in the other modules. What is
checked is that the tools write what they say, that the same seed gives the
same files, and that a failed native gate blocks modified-collagen output.
"""
import csv
import json

import numpy as np
import pytest

from modcoll import cli_common, cli_ensemble, cli_stagger


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


STAGGER_ARGS = ["--f", "0,1", "--ph", "4.2,7.4", "--ionic-mM", "150", "--pairs", "3",
                "--no-figures"]


@pytest.fixture(scope="module")
def stagger_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("stagger")
    code = cli_stagger.run(STAGGER_ARGS + ["--seed", "7", "--workers", "1",
                                           "--outdir", str(out)])
    return code, out


def test_stagger_cli_writes_summaries(stagger_run):
    code, out = stagger_run
    assert code == 0
    for name in ("summary.csv", "minima.csv", "run.json", "raw/pairs.csv",
                 "raw/profiles.npz"):
        assert (out / name).exists()
    rows = read_csv(out / "summary.csv")
    # 2 fractions x 2 pH x 1 salt x 2 orientations
    assert len(rows) == 8
    native = [r for r in rows if float(r["f"]) == 0.0]
    assert all(int(r["n_pairs"]) == 1 for r in native)          # deterministic control
    assert all(float(r["comb_z_pairs_sd"] or 0) == 0.0 for r in native
               if r["orientation"] == "parallel")
    modified = [r for r in rows if float(r["f"]) == 1.0]
    assert all(int(r["n_pairs"]) == 3 for r in modified)
    run = json.loads((out / "run.json").read_text())
    assert run["native_gate"]["passed"] is True
    assert run["native_gate"]["pka_set"] == "repository"
    assert run["pka_set"] == "recommended" and run["pKa"]["ASP"] == 3.9
    assert run["native_control_with_result_pka_set"]["passed"] is True
    assert {r["pka_set"] for r in rows} == {"recommended"}
    assert run["seed"] == 7 and run["pairs"] == 3
    assert run["settings"]["axis_separation_nm"] == pytest.approx(1.65)


def test_stagger_cli_native_row_reproduces_the_gate(stagger_run):
    _, out = stagger_run
    row = next(r for r in read_csv(out / "summary.csv") if float(r["f"]) == 0.0
               and float(r["pH"]) == 7.4 and r["orientation"] == "parallel")
    assert float(row["deepest_D"]) == pytest.approx(1.0, abs=0.02)
    assert row["deepest_at_multiple"] == "True"
    # net charge of the native molecule at pH 7.4 is about +32 e
    assert float(row["net_charge_mean"]) == pytest.approx(32, abs=2)
    full = next(r for r in read_csv(out / "summary.csv") if float(r["f"]) == 1.0
                and float(r["pH"]) == 7.4 and r["orientation"] == "parallel")
    # every lysine succinylated: two charges lost per lysine (107 of them)
    assert float(full["net_charge_mean"]) == pytest.approx(32 - 2 * 107, abs=3)


def test_stagger_cli_profiles_file_has_mean_and_spread(stagger_run):
    _, out = stagger_run
    data = np.load(out / "raw" / "profiles.npz")
    mean, sd = data["mean_parallel"], data["sd_parallel"]
    assert mean.shape == sd.shape == (2, 2, 1, 5, 1501)
    np.testing.assert_allclose(data["lags_parallel"][[0, -1]], [0.0, 300.0])
    assert np.all(sd[0] == 0.0)                 # one native pair: no spread
    # fully succinylated molecules are all identical too (every site modified)
    assert np.all(sd[1] < 1e-3)
    assert data["mean_antiparallel"].shape[-1] == 3001


def test_stagger_cli_is_reproducible_and_independent_of_worker_count(stagger_run, tmp_path):
    _, out = stagger_run
    args = ["--f", "0,0.5", "--ph", "4.2", "--ionic-mM", "150", "--pairs", "3",
            "--no-figures", "--orientations", "parallel"]
    runs = {}
    for label, extra in (("a", ["--seed", "7", "--workers", "1"]),
                         ("b", ["--seed", "7", "--workers", "2"]),
                         ("c", ["--seed", "8", "--workers", "1"])):
        d = tmp_path / label
        assert cli_stagger.run(args + extra + ["--outdir", str(d)]) == 0
        runs[label] = {n: (d / n).read_text() for n in ("summary.csv", "minima.csv",
                                                        "raw/pairs.csv")}
    assert runs["a"] == runs["b"]               # same seed: byte-identical
    assert runs["a"]["raw/pairs.csv"] != runs["c"]["raw/pairs.csv"]
    # partial modification: pattern pairs differ, and the spread is reported
    row = next(r for r in read_csv(tmp_path / "a" / "summary.csv") if float(r["f"]) == 0.5)
    assert float(row["deepest_kT_pairs_sd"]) > 0.0
    assert float(row["deepest_pair_sd_kT"]) > 0.0


def test_stagger_cli_refuses_modified_results_when_the_native_gate_fails(tmp_path):
    # without the hydrophobic channel the native control has no D-register
    args = STAGGER_ARGS + ["--eps-hydrophobic", "0", "--workers", "1",
                           "--outdir", str(tmp_path)]
    assert cli_stagger.run(args) == 2
    assert not (tmp_path / "summary.csv").exists()
    run = json.loads((tmp_path / "run.json").read_text())
    assert run["native_gate"]["passed"] is False
    assert cli_stagger.run(args + ["--gate-only"]) == 2


def test_stagger_cli_figures(tmp_path):
    args = ["--f", "0,1", "--ph", "7.4", "--ionic-mM", "150", "--pairs", "2",
            "--workers", "1", "--orientations", "parallel", "--outdir", str(tmp_path)]
    assert cli_stagger.run(args) == 0
    for name in ("fig_native_gate.png", "fig_profiles_vs_f.png",
                 "fig_selectivity_vs_f.png", "fig_deepest_minimum_positions.png"):
        assert (tmp_path / name).stat().st_size > 10_000


def test_pattern_generators_are_independent_per_pair_and_molecule():
    draws = {(p, k): cli_stagger.pattern_rng(5, p, k).random(4).tolist()
             for p in range(3) for k in range(2)}
    assert len({tuple(v) for v in draws.values()}) == 6
    assert cli_stagger.pattern_rng(5, 1, 0).random(4).tolist() == draws[(1, 0)]
    assert cli_stagger.pattern_rng(6, 1, 0).random(4).tolist() != draws[(1, 0)]


def test_chemistry_fractions():
    assert cli_stagger.chemistry_fractions("succ", 0.6) == (0.6, 0.0)
    assert cli_stagger.chemistry_fractions("acet", 0.6) == (0.0, 0.6)
    assert cli_stagger.chemistry_fractions("mixed", 0.6) == (0.3, 0.3)
    with pytest.raises(ValueError):
        cli_stagger.chemistry_fractions("phospho", 0.5)


@pytest.fixture(scope="module")
def ensemble_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("ensemble")
    code = cli_ensemble.run(["--f", "0,0.5,1", "--molecules", "12", "--seed", "3",
                             "--outdir", str(out)])
    return code, out


def test_ensemble_cli_outputs_and_gates(ensemble_run):
    code, out = ensemble_run
    assert code == 0
    for name in ("pi_vs_f.csv", "net_charge.csv", "net_charge_histogram.csv",
                 "titration.csv", "d_lag_correlation.csv",
                 "autocorrelation_profile.csv", "gates.json", "fig_pi_vs_f.png",
                 "fig_net_charge.png", "fig_charge_autocorrelation_pH4.2.png"):
        assert (out / name).exists()
    gates = json.loads((out / "gates.json").read_text())
    assert gates["fully_succinylated_pI_in_4.0_4.6"] is True
    assert gates["native_pI_near_9"] is False        # reported, not hidden
    assert gates["pka_set"] == "recommended"
    assert gates["native_pI"] == pytest.approx(9.84, abs=0.02)
    assert gates["by_pka_set"]["repository"]["native_pI"] == pytest.approx(10.11, abs=0.02)
    assert gates["by_pka_set"]["repository"]["fully_succinylated_pI"] == pytest.approx(
        4.21, abs=0.01)
    assert gates["by_pka_set"]["recommended"]["fully_succinylated_pI"] == pytest.approx(
        4.29, abs=0.01)
    assert 0.0 < gates["deamidated_fraction_giving_pI_9"] < 0.5


def test_ensemble_cli_isoelectric_point_falls_with_modification(ensemble_run):
    _, out = ensemble_run
    rows = read_csv(out / "pi_vs_f.csv")
    for chem in ("succ", "acet", "mixed", "deam"):
        pi = [float(r["pI_ensemble"]) for r in sorted(
            (r for r in rows if r["chemistry"] == chem), key=lambda r: float(r["f"]))]
        assert pi[0] > pi[1] > pi[2]
    by = {(r["chemistry"], float(r["f"])): r for r in rows}
    # f = 0 and f = 1 leave no randomness: every molecule has the same pI
    assert float(by[("succ", 0.0)]["pI_molecule_sd"]) == 0.0
    assert float(by[("succ", 1.0)]["pI_molecule_sd"]) == 0.0
    assert float(by[("succ", 0.5)]["pI_molecule_sd"]) > 0.0
    # succinylation (charge reversal) lowers the pI further than acetylation
    assert float(by[("succ", 1.0)]["pI_ensemble"]) < float(by[("acet", 1.0)]["pI_ensemble"])
    assert float(by[("succ", 0.5)]["mean_succinylated"]) == pytest.approx(53.5, abs=6)


def test_ensemble_cli_net_charge_statistics(ensemble_run):
    _, out = ensemble_run
    rows = {(r["chemistry"], float(r["f"]), float(r["pH"])): r
            for r in read_csv(out / "net_charge.csv")}
    native = float(rows[("succ", 0.0, 7.4)]["net_charge_mean"])
    full = float(rows[("succ", 1.0, 7.4)]["net_charge_mean"])
    half = rows[("succ", 0.5, 7.4)]
    assert native - full == pytest.approx(2 * 107, abs=1)
    assert full < float(half["net_charge_mean"]) < native
    # binomial spread of the number of modified lysines, two charges each:
    # sd = 2 sqrt(107 * 0.25) = 10.3 e; 12 molecules estimate it only roughly
    assert 4.0 < float(half["net_charge_sd"]) < 20.0
    assert float(rows[("succ", 0.0, 7.4)]["net_charge_sd"]) == 0.0
    assert float(half["net_charge_min"]) <= float(half["net_charge_median"]) <= float(
        half["net_charge_max"])
    assert float(rows[("succ", 0.0, 7.4)]["fraction_net_positive"]) == 1.0
    assert float(rows[("succ", 1.0, 7.4)]["fraction_net_positive"]) == 0.0
    # the histogram at the target pH holds every molecule exactly once
    hist = [r for r in read_csv(out / "net_charge_histogram.csv")
            if r["chemistry"] == "succ" and float(r["f"]) == 0.5]
    assert sum(int(r["molecules"]) for r in hist) == 12
    assert all(float(r["pH"]) == 4.2 for r in hist)


def test_ensemble_cli_d_lag_correlation(ensemble_run):
    _, out = ensemble_run
    rows = {(r["chemistry"], float(r["f"]), float(r["pH"])): r
            for r in read_csv(out / "d_lag_correlation.csv")}
    native = rows[("succ", 0.0, 7.4)]
    # where the charged residues are repeats with D; their signs do not
    assert float(native["abs_self_z_vs_other_lags"]) > 3.0
    assert int(native["abs_cross_lag_of_maximum"]) == 234
    assert abs(float(native["signed_self_z_vs_other_lags"])) < 1.0
    # one pattern only at f = 0: same molecule and two molecules agree
    assert native["abs_self_r_mean"] == native["abs_cross_r_mean"]
    half = rows[("succ", 0.5, 7.4)]
    assert float(half["signed_cross_r_sd"]) > 0.0


def test_ensemble_cli_is_reproducible(tmp_path):
    args = ["--f", "0,0.5", "--molecules", "6", "--chemistries", "succ", "--no-figures"]
    texts = []
    for label, seed in (("a", "3"), ("b", "3"), ("c", "4")):
        d = tmp_path / label
        assert cli_ensemble.run(args + ["--seed", seed, "--outdir", str(d)]) == 0
        texts.append((d / "net_charge.csv").read_text() + (d / "pi_vs_f.csv").read_text())
    assert texts[0] == texts[1] and texts[0] != texts[2]


def test_ensemble_samples_are_nested_across_fractions():
    from modcoll.sites import SiteType, native_molecule
    native = native_molecule()
    low = cli_ensemble.sample(native, "succ", 0.25, 5, seed=1)
    high = cli_ensemble.sample(native, "succ", 0.75, 5, seed=1)
    for a, b in zip(low, high):
        hit_low = a.site_type == int(SiteType.SUCC_LYS)
        assert np.all(b.site_type[hit_low] == int(SiteType.SUCC_LYS))
    assert np.any(low[0].site_type != low[1].site_type)


def test_csv_and_json_writers(tmp_path):
    rows = [{"a": 1, "b": 0.123456789, "c": "x"}, {"a": 2, "b": float("nan"), "d": True}]
    cli_common.write_csv(tmp_path / "t.csv", rows)
    assert (tmp_path / "t.csv").read_text() == "a,b,c,d\n1,0.123457,x,\n2,,,True\n"
    cli_common.write_json(tmp_path / "t.json", {"x": np.float64(1.5), "y": np.arange(3),
                                                "z": float("nan"), "w": np.bool_(True)})
    assert json.loads((tmp_path / "t.json").read_text()) == {
        "w": True, "x": 1.5, "y": [0, 1, 2], "z": None}
    assert cli_common.float_list("0, 0.25,1") == [0.0, 0.25, 1.0]


def test_ensemble_cli_repository_pka_set(tmp_path):
    assert cli_ensemble.run(["--f", "0,1", "--molecules", "3", "--chemistries", "succ",
                             "--pka-set", "repository", "--no-figures",
                             "--outdir", str(tmp_path)]) == 0
    rows = {float(r["f"]): r for r in read_csv(tmp_path / "pi_vs_f.csv")}
    assert float(rows[0.0]["pI_ensemble"]) == pytest.approx(10.11, abs=0.02)
    assert float(rows[1.0]["pI_ensemble"]) == pytest.approx(4.21, abs=0.01)


def test_crosslink_weighting_shares_deviates_with_uniform_succinylation():
    from modcoll.sites import SiteType, native_molecule
    native = native_molecule()
    uniform = cli_ensemble.sample(native, "succ", 1.0, 3, seed=2)
    weighted = cli_ensemble.sample(native, "succ_crosslink", 1.0, 3, seed=2)
    for a, b in zip(uniform, weighted):
        np.testing.assert_array_equal(a.site_type, b.site_type)     # f = 1: all modified
    low = cli_ensemble.sample(native, "succ_crosslink", 0.1, 40, seed=2)
    fast = [(0, 86), (1, 86), (2, 86), (0, 929), (1, 929), (2, 932)]
    hit = np.mean([[m.site_type[(m.chain == c) & (m.res_index == i) & ~m.smeared][0]
                    == int(SiteType.SUCC_LYS) for c, i in fast] for m in low])
    assert hit > 0.4                      # against 0.1 for an average lysine
