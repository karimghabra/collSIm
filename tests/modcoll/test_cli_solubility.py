"""The solubility command end to end, on a coarse separation grid (--quick).

What is checked against independent facts: the counter-ion content implied
by the readings of the material in water equals the number of basic groups
(the charge balance of a fully protonated strong-acid salt), and the
inverse tables are consistent with the per-composition table.
"""
from __future__ import annotations

import csv

import pytest

from modcoll import cli_solubility as cs


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("solubility")
    code = cs.run(["--quick", "--stages", "inverse,water", "--inverse-step", "1.0",
                   "--inverse-pairs", "1", "--pairs", "1", "--molecules", "8",
                   "--inverse-band", "99",
                   "--inverse-variants", "base", "--water-compositions", "",
                   "--workers", "1", "--no-figures", "--outdir", str(out)])
    return code, out


def test_writes_its_tables(run):
    code, out = run
    assert code == 0
    for name in ("inverse_compositions.csv", "inverse_summary.csv", "water_dilution.csv",
                 "water_fit.csv", "run.json"):
        assert (out / name).exists(), name


def test_inverse_summary_counts_the_composition_table(run):
    _, out = run
    rows = read_csv(out / "inverse_compositions.csv")
    summary = {r["pI_band"]: r for r in read_csv(out / "inverse_summary.csv")}
    assert int(summary["any"]["compositions"]) == len(rows) == 9
    met = sum(r["matches_all_pair_criteria"] == "True" for r in rows)
    assert int(summary["any"]["match all pair criteria"]) == met
    assert int(summary["any"]["excluded by the pair criteria"]) == len(rows) - met
    for name, *_ in cs.BENCH:
        hits = sum(r[f"matches [{name}]"] == "True" for r in rows)
        assert int(summary["any"][f"match [{name}]"]) == hits
    # the bands are nested
    n = [int(summary[f"4.2 +- {b:g}"]["compositions"]) for b in (0.1, 0.2, 0.3)]
    assert n == sorted(n)


def test_water_readings_imply_one_anion_per_basic_group(run):
    _, out = run
    fits = {r["composition"]: r for r in read_csv(out / "water_fit.csv")}
    for name, f in fits.items():
        ratio = float(f["fitted_anions_per_molecule"]) / float(f["basic_groups_per_molecule"])
        assert 0.75 < ratio < 1.1, name
        assert float(f["rms_pH_first_five"]) < 0.15
        # no fixed counter-ion content reaches the last reading
        assert float(f["pH_predicted_last"]) < 5.0 < float(f["pH_measured_last"])
    rows = [r for r in read_csv(out / "water_dilution.csv")
            if r["composition"] == "succinylated 100 %"]
    # below the isoelectric point (net positive) for the four strongest solutions
    assert all(float(r["pH_minus_pI"]) < 0 for r in rows[:4])
