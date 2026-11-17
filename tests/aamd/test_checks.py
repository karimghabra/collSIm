"""The live checks grade what they should, on synthetic data."""
import numpy as np
import pytest

from aamd import checks


def verdicts(cs):
    return {c.name: c.verdict for c in cs}


def test_temperature_and_peak():
    rng = np.random.default_rng(0)
    ok = 298.15 + 0.8 + rng.normal(scale=2.0, size=500)
    assert set(verdicts(checks.check_temperature(ok)).values()) == {"ok"}
    assert verdicts(checks.check_temperature(ok + 3.0))["temperature"] == "warn"
    assert verdicts(checks.check_temperature(ok + 6.0))["temperature"] == "fail"
    hot = ok.copy()
    hot[10] = 450.0
    assert verdicts(checks.check_temperature(hot))["temperature_peak"] == "fail"


def test_finite_and_drift():
    assert checks.check_finite(np.ones(4), np.zeros((2, 3))).verdict == "ok"
    assert checks.check_finite(np.array([1.0, np.nan])).verdict == "fail"
    assert checks.check_drift("energy_drift", -5.000e5, -5.001e5, 0.003, 0.02).verdict == "ok"
    assert checks.check_drift("energy_drift", -5.0e5, -5.1e5, 0.003, 0.02).verdict == "warn"
    assert checks.check_drift("energy_drift", -5.0e5, -6.0e5, 0.003, 0.02).verdict == "fail"
    assert checks.check_drift("energy_drift", -5.0e5, None, 0.003, 0.02).verdict == "ok"


def test_umbrella_and_lateral():
    rng = np.random.default_rng(1)
    cv = np.column_stack([2.0 + 0.05 * rng.normal(size=500), 0.01 * rng.normal(size=500),
                          0.3 + 0.01 * rng.normal(size=500)])
    assert set(verdicts(checks.check_umbrella(cv, 2.0, 0.3)).values()) == {"ok"}
    assert verdicts(checks.check_umbrella(cv, 2.2, 0.3))["umbrella_centre"] == "warn"
    assert verdicts(checks.check_umbrella(cv, 2.4, 0.3))["umbrella_centre"] == "fail"
    drift = cv.copy()
    drift[:, 1] += 0.3
    assert verdicts(checks.check_umbrella(drift, 2.0, 0.3))["lateral_offset"] == "fail"


def _helix(n_res=30, twist=0.0, noise=0.0, rng=None):
    """Three helical chains of backbone atoms (N, CA, C per residue)."""
    rng = np.random.default_rng(2) if rng is None else rng
    pts = []
    for c in range(3):
        for k in range(n_res):
            for a in range(3):
                t = 0.6 * k + 2.1 * c + 0.2 * a + twist * k
                pts.append([0.3 * np.cos(t), 0.3 * np.sin(t), 0.286 * k + 0.1 * a])
    x = np.array(pts)
    return x + noise * rng.normal(size=x.shape)


def test_structure_checks_detect_local_damage_not_global_bending():
    x0 = _helix()
    idx = np.arange(len(x0))
    ref = checks.IntegrityRef([idx, idx], [x0, x0], [[], []], [0, 0], np.zeros((0, 2), int))
    # a gentle uniform extra twist changes the global RMSD but no block much
    twisted = _helix(twist=0.01)
    cs = verdicts(checks.check_structure(twisted[None], ref))
    assert cs["helix_rmsd_local"] == "ok"
    # scramble two triplets of every chain: a local failure
    broken = x0.copy()
    rng = np.random.default_rng(3)
    for c in range(3):
        sl = slice(c * 90 + 30, c * 90 + 48)
        broken[sl] += rng.normal(scale=0.6, size=broken[sl].shape)
    assert verdicts(checks.check_structure(broken[None], ref))["helix_rmsd_local"] == "fail"


def test_hbond_count_and_bond_stretch():
    # one donor N-H pointing at an acceptor 0.29 nm away, one pointing away
    xyz = np.array([[0, 0, 0], [0.1, 0, 0], [0.29, 0, 0],      # N, H, O (bonded geometry)
                    [1, 0, 0], [0.9, 0, 0], [1.29, 0, 0.0]])  # N, H pointing away, O
    donors = np.array([[0, 1], [3, 4]])
    assert checks.count_hbonds(xyz, donors, np.array([2, 5])) == 1
    ref = checks.IntegrityRef([np.arange(3)], [xyz[:3]], [[(donors, np.array([2, 5]))]], [1],
                              np.array([[0, 1]]))
    stretched = xyz.copy()
    stretched[1] = [0.3, 0, 0]
    v = verdicts(checks.check_structure(stretched[None], ref))
    assert v["bond_stretch"] == "fail"
    assert v["interchain_hbonds"] == "fail"


def test_ion_checks():
    assert checks.check_ion_count({"Na+": 35, "CA": 12}, {"Na+": 35, "CA": 12}).verdict == "ok"
    assert checks.check_ion_count({"Na+": 35}, {"Na+": 35, "CA": 12}).verdict == "fail"
    v = verdicts(checks.check_ion_coordination(np.array([6, 6, 6]), np.array([7, 8, 7])))
    assert v == {"ion_coordination_Mg": "ok", "ion_coordination_Ca": "ok"}
    assert verdicts(checks.check_ion_coordination(np.array([9, 9]), []))["ion_coordination_Mg"] == "fail"


def test_overlap_coefficient():
    rng = np.random.default_rng(4)
    a = rng.normal(2.0, 0.05, 5000)
    assert checks.overlap(a, rng.normal(2.0, 0.05, 5000)) > 0.85
    assert checks.overlap(a, rng.normal(2.1, 0.05, 5000)) == pytest.approx(0.32, abs=0.06)
    assert checks.overlap(a, rng.normal(3.0, 0.05, 5000)) == 0.0


def test_record_writes_one_line_per_non_ok(tmp_path):
    w = tmp_path / "WARNINGS"
    cs = [checks.Check("a", 0, "-", "-", "ok"), checks.Check("b", 1, "x", "y", "warn"),
          checks.Check("c", 2, "x", "y", "fail")]
    assert checks.record(cs, "sys/w00 chunk 0", w) == "fail"
    lines = w.read_text().splitlines()
    assert len(lines) == 2 and " b WARN " in lines[0] and " c FAIL " in lines[1]
    assert checks.record(cs[:2], "x", w) == "warn"


def test_interface_waters_counts_bridging_waters_across_the_box_edge():
    from aamd.umbrella import interface_waters
    box = np.array([5.0, 5.0, 5.0])
    a = np.array([[4.9, 1.0, 1.0]])              # segment A atom near the +x edge
    b = np.array([[5.4, 1.0, 1.0]])              # segment B, unwrapped (OpenMM keeps the solute whole)
    waters = np.array([[0.15, 1.0, 1.0],         # between them, stored wrapped through the boundary
                       [4.6, 1.0, 1.0],          # near A only
                       [2.5, 2.5, 2.5]])         # bulk
    pos = np.vstack([a, b, waters])
    out = interface_waters(pos, box, [np.array([0]), np.array([1])], np.array([2, 3, 4]))
    assert out == [1, 2, 1]
