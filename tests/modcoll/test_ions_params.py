"""The parameter file: completeness, provenance and derived tables."""
import math
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.calibrate import write_fitted  # noqa: E402
from modcoll.ions.energy import KIND_CATION, KIND_OXYANION  # noqa: E402
from modcoll.ions.params import (PARAMS_PATH, STATUSES, load_params, validate_raw)  # noqa: E402


@pytest.fixture(scope="module")
def params():
    return load_params()


def test_every_parameter_states_value_units_source_and_status(params):
    entries = [(k, e) for k, e in params.entries() if not k.startswith("meta.")]
    assert len(entries) > 60
    for path, entry in entries:
        for key in ("value", "units", "source", "status"):
            assert key in entry, f"{path} lacks {key}"
        assert entry["status"] in STATUSES, path
        assert len(entry["source"].strip()) > 5, path


def test_fitted_entries_are_exactly_the_contact_energies(params):
    fitted = {k for k, e in params.entries() if e.get("status") == "fitted here"}
    contact = {k for k, _ in params.entries() if k.startswith("contact.")}
    assert fitted == contact | {"species.Ca.well_angular_flat_deg"}


def test_ranges_contain_their_values(params):
    with_range = [(k, e) for k, e in params.entries() if "lo" in e]
    assert len(with_range) > 20
    for path, e in with_range:
        assert e["lo"] <= e["value"] <= e["hi"], path


def test_sources_cite_publications_not_internal_documents(params):
    for path, e in params.entries():
        if path.startswith("meta."):
            continue
        assert "report.md" not in e["source"] and "/home/" not in e["source"], path


def test_loader_rejects_incomplete_or_mislabelled_entries():
    good = {"a": {"b": {"value": 1.0, "units": "nm", "source": "somewhere", "status": "literature"}}}
    validate_raw(good)
    for missing in ("units", "source", "status"):
        bad = {"a": {"b": {k: v for k, v in good["a"]["b"].items() if k != missing}}}
        with pytest.raises(ValueError):
            validate_raw(bad)
    bad = {"a": {"b": dict(good["a"]["b"], status="guessed")}}
    with pytest.raises(ValueError):
        validate_raw(bad)
    bad = {"a": {"b": dict(good["a"]["b"], source="  ")}}
    with pytest.raises(ValueError):
        validate_raw(bad)


def test_each_ion_carries_its_formal_charge(params):
    T = params.tables()
    for name, q in (("Mg", 2.0), ("Ca", 2.0), ("SO4", -2.0)):
        sp = params.species_index(name)
        assert T.q_centre[sp] + T.nsite[sp] * T.q_sat[sp] == pytest.approx(q)
    assert T.kind[params.species_index("Mg")] == KIND_CATION
    assert T.kind[params.species_index("SO4")] == KIND_OXYANION


def test_site_counts_and_bond_lengths(params):
    T = params.tables()
    mg, ca, so4 = (params.species_index(n) for n in ("Mg", "Ca", "SO4"))
    assert (T.nsite[mg], T.nsite[ca], T.nsite[so4]) == (6, 7, 4)
    # Harding 2006: Mg-O 2.07 A, Ca-O 2.36-2.39 A; sulfate S-O 1.49-1.495 A
    assert T.d_site[mg] == pytest.approx(0.207, abs=0.003)
    assert T.d_site[ca] == pytest.approx(0.238, abs=0.004)
    assert T.d_site[so4] == pytest.approx(0.149, abs=0.002)
    # calcium is the softer of the two cations
    assert T.cos_max[ca] < T.cos_max[mg]
    assert T.cos_flat[ca] < T.cos_flat[mg] == 1.0


@pytest.mark.parametrize("n", [6, 7, 8])
def test_calcium_coordination_number_is_selectable(n):
    p = load_params(ca_coordination=n, overrides={"species.Ca.centre_charge": 0.0})
    T = p.tables()
    ca = p.species_index("Ca")
    assert T.nsite[ca] == n
    assert T.q_sat[ca] == pytest.approx(2.0 / n)       # dummy layout shares the charge
    assert np.allclose(np.linalg.norm(T.dirs[ca, :n], axis=1), 1.0)
    assert not T.dirs[ca, n:].any()
    assert p.free_ion_partition("Ca") == pytest.approx((1 + math.exp(-float(p.value("species.Ca.dg_des")))) ** n)


def test_other_coordination_numbers_are_refused():
    for n in (5, 9):
        with pytest.raises(ValueError):
            load_params(ca_coordination=n)


def test_by_default_the_whole_charge_sits_on_the_centre(params):
    T = params.tables()
    for name in ("Mg", "Ca"):
        sp = params.species_index(name)
        assert T.q_centre[sp] == 2.0 and T.q_sat[sp] == 0.0


def test_core_meets_a_ligand_oxygen_at_the_bond_length(params):
    T = params.tables()
    r_o = T.R_type[geo.TYPE_INDEX["O_carboxylate"]]
    for name in ("Mg", "Ca"):
        sp = params.species_index(name)
        assert T.R_centre[sp] + r_o == pytest.approx(T.d_site[sp])
    # and the implied ionic radii are Shannon's (Mg 0.072, Ca 0.100-0.106 nm) to 0.01 nm
    assert T.R_centre[params.species_index("Mg")] == pytest.approx(0.072, abs=0.01)
    assert T.R_centre[params.species_index("Ca")] == pytest.approx(0.103, abs=0.01)


def test_free_ion_partition_function(params):
    dg = float(params.value("species.Mg.dg_des"))
    assert params.free_ion_partition("Mg") == pytest.approx((1 + math.exp(-dg)) ** 6)
    assert params.free_ion_partition("SO4") == 1.0
    p = params.with_overrides({"species.Ca.dg_des": 0.0})
    assert p.free_ion_partition("Ca") == pytest.approx(2.0 ** 7)


def test_desolvation_cost_follows_from_the_water_exchange_rate(params):
    """Eyring: dG = kT ln(kB T / (h k)). For k = 6.7e5 /s (Mg, Bleuzen et al.
    1997) this is 16.0 kT; the tabulated dH and dS give 16.1."""
    kB, h, T0 = 1.380649e-23, 6.62607015e-34, 298.15
    dg = math.log(kB * T0 / h / 6.7e5)
    assert dg == pytest.approx(16.04, abs=0.01)
    from_activation = (49.1e3 - T0 * 31.1) / (8.314462618 * T0)
    assert from_activation == pytest.approx(16.07, abs=0.01)
    assert float(params.value("species.Mg.dg_des")) == pytest.approx(from_activation, abs=0.1)
    # magnesium holds its water more tightly than calcium
    assert float(params.value("species.Mg.dg_des")) > float(params.value("species.Ca.dg_des")) + 4


def test_overrides_do_not_touch_the_original(params):
    before = float(params.value("species.Ca.d_MO"))
    p = params.with_overrides({"species.Ca.d_MO": 0.25})
    assert float(p.value("species.Ca.d_MO")) == 0.25
    assert float(params.value("species.Ca.d_MO")) == before
    with pytest.raises(KeyError):
        params.with_overrides({"species.Ca.no_such_thing": 1.0})


def test_what_one_site_may_hold_is_enforced(params):
    """Mg2+: one oxygen per site, so its well must be narrower than half the
    carboxylate O...O distance (0.111 nm). Ca2+: both oxygens of one
    carboxylate allowed, oxygens of two groups never, so narrower than one
    oxygen radius (0.140 nm)."""
    half_oo = 0.125 * math.sin(math.radians(62.5))
    d_mg = float(params.value("species.Mg.d_MO"))
    d_ca = float(params.value("species.Ca.d_MO"))
    th_mg = math.radians(float(params.value("species.Mg.well_angular_halfwidth_deg")))
    th_ca = math.radians(float(params.value("species.Ca.well_angular_halfwidth_deg")))
    assert d_mg * math.sin(th_mg) < half_oo
    assert half_oo < d_ca * math.sin(th_ca) < 0.140
    with pytest.raises(ValueError):
        load_params(overrides={"species.Mg.well_angular_halfwidth_deg": 33.0})   # 0.113 nm
    with pytest.raises(ValueError):
        load_params(overrides={"species.Ca.well_angular_halfwidth_deg": 37.0})   # 0.143 nm
    with pytest.raises(ValueError):
        load_params(overrides={"species.Ca.bidentate_carboxylate": "maybe"})
    # a calcium that is told to be monodentate must also be narrow
    with pytest.raises(ValueError):
        load_params(overrides={"species.Ca.bidentate_carboxylate": "no"})
    load_params(overrides={"species.Ca.bidentate_carboxylate": "no",
                           "species.Ca.well_angular_halfwidth_deg": 27.0,
                           "species.Ca.well_angular_flat_deg": 10.0})


def test_partner_types(params):
    T = params.tables()
    ca, so4 = params.species_index("Ca"), params.species_index("SO4")
    assert T.partner[ca, geo.TYPE_INDEX["O_carboxylate"]]
    assert not T.partner[ca, geo.TYPE_INDEX["O_carboxylic_acid"]]
    assert not T.partner[ca, geo.TYPE_INDEX["N_guanidinium"]]
    assert T.partner[so4, geo.TYPE_INDEX["N_guanidinium"]]
    assert T.partner[so4, geo.TYPE_INDEX["N_ammonium"]]
    assert T.partner[so4, geo.TYPE_INDEX["N_imidazolium"]]
    assert not T.partner[so4, geo.TYPE_INDEX["O_carboxylate"]]


def test_protonated_acid_has_no_contact_energy(params):
    T = params.tables()
    t = geo.TYPE_INDEX["O_carboxylic_acid"]
    assert not T.eps_in[:, t].any() and not T.eps_out[:, t].any() and not T.eps_hb[:, t].any()


def test_write_fitted_changes_only_the_named_value(tmp_path):
    copy = tmp_path / "params.toml"
    shutil.copy(PARAMS_PATH, copy)
    before = copy.read_text().splitlines()
    write_fitted({"contact.Ca.O_carboxylate.eps_in": 12.345}, copy)
    after = copy.read_text().splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 1
    assert changed[0][1] == "value = 12.345"
    raw = tomllib.loads(copy.read_text())
    assert raw["contact"]["Ca"]["O_carboxylate"]["eps_in"]["value"] == 12.345
    assert load_params(copy).value("contact.Ca.O_carboxylate.eps_in") == 12.345
    with pytest.raises(ValueError):
        write_fitted({"contact.Ca.no_such.eps_in": 1.0}, copy)
