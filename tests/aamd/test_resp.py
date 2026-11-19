"""RESP fitting against cases with known answers."""
import numpy as np
import pytest

from aamd.params import resp


def _toy(rng, n_atom=5, charges=None):
    xyz = rng.normal(scale=1.2, size=(n_atom, 3))
    q = rng.normal(scale=0.4, size=n_atom) if charges is None else np.asarray(charges)
    q -= (q.sum() - round(q.sum())) / n_atom
    grid = resp.mk_grid(xyz, ["C"] * n_atom)
    esp = resp.design_matrix(xyz, grid) @ q
    return xyz, grid, esp, q


def test_grid_lies_on_scaled_shells_outside_all_atoms():
    xyz = np.array([[0.0, 0, 0], [1.1, 0, 0]])
    grid = resp.mk_grid(xyz, ["C", "H"])
    d = np.linalg.norm(grid[:, None] - xyz[None], axis=-1)
    radii = np.array([1.5, 1.2])
    ratio = (d / radii).min(axis=1)
    assert ratio.min() > 1.4 - 1e-6
    assert ratio.max() < 2.0 + 1e-6
    assert len(grid) > 200


def test_unrestrained_fit_recovers_exact_point_charges():
    rng = np.random.default_rng(1)
    xyz, grid, esp, q = _toy(rng)
    c = resp.Conformer(xyz, grid, esp)
    fit, rrms = resp.fit([c], round(q.sum()), [[i] for i in range(5)], a=0.0)
    assert np.allclose(fit, q, atol=1e-6)
    assert rrms < 1e-8


def test_constraints_total_fixed_and_equivalence_are_exact():
    rng = np.random.default_rng(2)
    xyz, grid, esp, q = _toy(rng, 6)
    c = resp.Conformer(xyz, grid, esp)
    fit, _ = resp.fit([c], -1.0, [[0, 1], [2], [3], [4], [5]], fixed={5: 0.3}, a=0.0005,
                      restrained=np.ones(6, bool))
    assert fit.sum() == pytest.approx(-1.0, abs=1e-10)
    assert fit[5] == pytest.approx(0.3)
    assert fit[0] == pytest.approx(fit[1])


def test_restraint_pulls_buried_charges_towards_zero():
    rng = np.random.default_rng(3)
    xyz, grid, esp, q = _toy(rng)
    c = resp.Conformer(xyz, grid, esp)
    free, _ = resp.fit([c], round(q.sum()), [[i] for i in range(5)], a=0.0)
    strong, _ = resp.fit([c], round(q.sum()), [[i] for i in range(5)], a=0.05,
                         restrained=np.ones(5, bool))
    assert np.abs(strong).sum() < np.abs(free).sum()


def test_two_stage_equivalences_methyl_hydrogens():
    # methane-like: C at the origin, four H; charges asymmetric before stage 2
    h = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * (1.09 / np.sqrt(3))
    xyz = np.vstack([[0, 0, 0], h])
    q_true = np.array([-0.4, 0.12, 0.08, 0.1, 0.1])
    grid = resp.mk_grid(xyz, ["C", "H", "H", "H", "H"])
    esp = resp.design_matrix(xyz, grid) @ q_true
    q, info = resp.two_stage([resp.Conformer(xyz, grid, esp)], 0.0, ["C", "H", "H", "H", "H"],
                             [(0, k) for k in range(1, 5)])
    assert np.allclose(q[1:], q[1])
    assert q.sum() == pytest.approx(0.0, abs=1e-10)
    assert abs(q[0]) < abs(q_true[0])     # the restraint shrinks the buried carbon
    assert info["rrms_stage2"] >= info["rrms_stage1"] - 1e-12


def test_two_stage_without_methyl_groups_returns_stage_one():
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
    xyz = np.vstack([[0, 0, 0], 1.49 * t])
    el = ["S", "O", "O", "O", "O"]
    q_true = np.array([1.2, -0.8, -0.8, -0.8, -0.8])
    grid = resp.mk_grid(xyz, el)
    esp = resp.design_matrix(xyz, grid) @ q_true
    q, info = resp.two_stage([resp.Conformer(xyz, grid, esp)], -2.0, el, [(0, k) for k in range(1, 5)],
                             symmetric=[[1, 2, 3, 4]])
    assert q.sum() == pytest.approx(-2.0)
    assert np.allclose(q[1:], q[1])
    assert info["rrms_stage2"] == info["rrms_stage1"]
