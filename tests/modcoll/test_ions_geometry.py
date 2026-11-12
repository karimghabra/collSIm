"""Coordination polyhedra, rotations and ligand groups against closed forms."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.energy import MAX_SITES, rotate_dirs  # noqa: E402
from modcoll.ions.params import load_params  # noqa: E402


@pytest.fixture(scope="module")
def params():
    return load_params()


def angles_deg(U):
    c = np.clip(U @ U.T, -1, 1)
    return np.degrees(np.arccos(c))


@pytest.mark.parametrize("name", list(geo.POLYHEDRA))
def test_polyhedron_vectors_are_unit(name):
    U = geo.POLYHEDRA[name]()
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)


def test_octahedron_has_four_cis_and_one_trans_neighbour():
    A = angles_deg(geo.octahedron())
    for row in A:
        others = np.sort(np.delete(row, np.argmin(row)))
        assert np.allclose(others, [90, 90, 90, 90, 180])


def test_pentagonal_bipyramid_angles():
    U = geo.pentagonal_bipyramid()
    assert len(U) == 7
    A = angles_deg(U)
    assert A[0, 1] == pytest.approx(180.0)                 # the two axial sites
    assert np.allclose(A[0, 2:], 90.0)                     # axial to equatorial
    eq = np.sort(A[2, 3:])                                 # one equatorial to the other four
    assert np.allclose(eq, [72, 72, 144, 144])


def test_square_antiprism_has_sixteen_equal_edges():
    U = geo.square_antiprism()
    assert len(U) == 8
    d = np.linalg.norm(U[:, None] - U[None], axis=2)
    iu = np.triu_indices(8, 1)
    dist = np.sort(d[iu])
    # uniform antiprism: 8 square edges + 8 lateral edges, all the same length
    assert np.allclose(dist[:16], dist[0], rtol=1e-12)
    assert dist[16] > 1.2 * dist[0]
    # each vertex touches exactly four edges
    assert np.all((np.isclose(d, dist[0])).sum(axis=1) == 4)


def test_tetrahedron_angle_and_edge():
    U = geo.tetrahedron()
    A = angles_deg(U)
    assert np.allclose(A[np.triu_indices(4, 1)], math.degrees(math.acos(-1 / 3)))
    # edge of a tetrahedron inscribed in a sphere of radius d is d * sqrt(8/3):
    # sulfate O...O for S-O = 0.149 nm is 0.2433 nm
    assert np.linalg.norm(0.149 * (U[0] - U[1])) == pytest.approx(0.149 * math.sqrt(8 / 3))


def test_quaternion_matrix_is_a_proper_rotation_and_round_trips():
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = geo.random_quaternion(rng)
        R = geo.quat_to_matrix(q)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        q2 = geo.quat_from_matrix(R)
        assert np.allclose(geo.quat_to_matrix(q2), R, atol=1e-12)


def test_quaternion_known_rotation():
    # 90 deg about z takes x to y
    q = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    assert np.allclose(geo.quat_to_matrix(q) @ [1, 0, 0], [0, 1, 0])


def test_quat_multiply_composes_rotations():
    rng = np.random.default_rng(3)
    a, b = geo.random_quaternion(rng), geo.random_quaternion(rng)
    v = rng.normal(size=3)
    lhs = geo.quat_to_matrix(geo.quat_multiply(a, b)) @ v
    rhs = geo.quat_to_matrix(a) @ (geo.quat_to_matrix(b) @ v)
    assert np.allclose(lhs, rhs)


def test_quat_from_two_vectors_including_antiparallel():
    rng = np.random.default_rng(11)
    for _ in range(10):
        a, b = rng.normal(size=3), rng.normal(size=3)
        q = geo.quat_from_two_vectors(a, b)
        assert np.allclose(geo.quat_to_matrix(q) @ (a / np.linalg.norm(a)), b / np.linalg.norm(b))
    q = geo.quat_from_two_vectors([0, 0, 1], [0, 0, -1])
    assert np.allclose(geo.quat_to_matrix(q) @ [0, 0, 1], [0, 0, -1])


def test_random_quaternion_is_isotropic():
    # For a Haar-uniform rotation the image of any fixed unit vector is uniform
    # on the sphere: mean 0 and <x_a x_b> = delta_ab / 3.
    rng = np.random.default_rng(123)
    n = 40000
    v = np.array([geo.quat_to_matrix(geo.random_quaternion(rng)) @ [0.0, 0.0, 1.0] for _ in range(n)])
    assert np.all(np.abs(v.mean(axis=0)) < 4 * math.sqrt(1 / 3 / n))
    cov = v.T @ v / n
    # var of x^2 on the sphere is 1/5 - 1/9 = 4/45
    assert np.allclose(np.diag(cov), 1 / 3, atol=4 * math.sqrt(4 / 45 / n))
    assert np.all(np.abs(cov[np.triu_indices(3, 1)]) < 4 * math.sqrt(1 / 15 / n))


def test_kernel_rotation_matches_matrix(params):
    T = params.tables()
    rng = np.random.default_rng(5)
    q = geo.random_quaternion(rng)
    for name in params.species:
        sp = params.species_index(name)
        out = np.zeros((MAX_SITES, 3))
        rotate_dirs(T, sp, q, out)
        U = params.polyhedron(name)
        assert np.allclose(out[: len(U)], U @ geo.quat_to_matrix(q).T)


def test_carboxylate_geometry_and_charge():
    lig = geo.carboxylate([1.0, 2.0, 3.0], [0, 0, 1])
    # O...O = 2 * 0.125 nm * sin(62.5 deg)
    assert np.linalg.norm(lig.pos[0] - lig.pos[1]) == pytest.approx(2 * 0.125 * math.sin(math.radians(62.5)))
    assert np.allclose(np.linalg.norm(lig.pos - [1.0, 2.0, 3.0], axis=1), 0.125)
    assert lig.charge.sum() == pytest.approx(-1.0)
    assert set(lig.type) == {geo.TYPE_INDEX["O_carboxylate"]}
    assert lig.group[0] == lig.group[1]
    # oxygens lean outward along the direction
    assert np.all((lig.pos - [1.0, 2.0, 3.0]) @ [0, 0, 1] > 0)


def test_protonated_carboxyl_is_neutral_and_of_the_weak_type():
    lig = geo.carboxylate([0, 0, 0], [1, 0, 0], protonated=True)
    assert np.all(lig.charge == 0.0)
    assert set(lig.type) == {geo.TYPE_INDEX["O_carboxylic_acid"]}


def test_guanidinium_is_planar_threefold_and_carries_plus_one():
    c = np.array([0.5, 0.5, 0.5])
    lig = geo.guanidinium(c, [1, 0, 0], normal=[0, 0, 1])
    d = np.linalg.norm(lig.pos[:, None] - lig.pos[None], axis=2)
    # N...N = C-N * sqrt(3)
    assert np.allclose(d[np.triu_indices(3, 1)], 0.133 * math.sqrt(3))
    assert lig.charge.sum() == pytest.approx(1.0)
    assert np.allclose((lig.pos - c) @ [0, 0, 1], 0.0)        # in the plane
    assert np.allclose(lig.normal, [0, 0, 1])
    # NH1 and NH2 are the outward pair, NE points back
    assert (lig.pos[0] - c) @ [1, 0, 0] < 0
    assert np.all((lig.pos[1:] - c) @ [1, 0, 0] > 0)


def test_ammonium_and_imidazolium_charges():
    assert geo.ammonium([0, 0, 0]).charge.sum() == pytest.approx(1.0)
    im = geo.imidazolium([0, 0, 0], [1, 0, 0])
    assert im.charge.sum() == pytest.approx(1.0)
    assert np.linalg.norm(im.pos[0] - im.pos[1]) == pytest.approx(0.22)


def test_place_on_surface_follows_the_s_phi_convention():
    # phi = atan2(y, x), s = z: a site at phi = 90 deg, s = 3 nm, rho = 0.6 nm
    # has its carboxyl carbon at (0, 0.6, 3) and points along +y.
    lig = geo.place_on_surface([3.0], [math.pi / 2], 0.6, "carboxylate", mol=4)
    centre = lig.pos.mean(axis=0)
    assert centre[0] == pytest.approx(0.0, abs=1e-12)
    assert centre[2] == pytest.approx(3.0)
    assert centre[1] == pytest.approx(0.6 + 0.125 * math.cos(math.radians(62.5)))
    assert np.all(lig.mol == 4)
    # and the azimuth read back from the placed atoms is the one put in
    assert math.atan2(centre[1], centre[0]) == pytest.approx(math.pi / 2)


def test_place_on_surface_gives_each_site_its_own_group_and_molecule():
    lig = geo.place_on_surface([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 0.6,
                               ["carboxylate", "guanidinium", "ammonium"], mol=[0, 0, 1],
                               axis_xy=(2.0, 2.0))
    assert len(lig) == 2 + 3 + 1
    assert list(lig.group) == [0, 0, 1, 1, 1, 2]
    assert list(lig.mol) == [0, 0, 0, 0, 0, 1]
    assert lig.charge.sum() == pytest.approx(-1 + 1 + 1)


def test_concat_keeps_groups_distinct():
    a = geo.carboxylate([0, 0, 0], [1, 0, 0])
    b = geo.carboxylate([1, 0, 0], [1, 0, 0], mol=1)
    both = geo.LigandSet.concat([a, b])
    assert list(both.group) == [0, 0, 1, 1]
    assert both.min_separation() == pytest.approx(
        np.min(np.linalg.norm(a.pos[:, None] - b.pos[None], axis=2)))


def test_ligandset_rejects_inconsistent_arrays():
    with pytest.raises(ValueError):
        geo.LigandSet(pos=[[0, 0, 0], [1, 0, 0]], type=[0], charge=[0, 0], mol=[0, 0])
    with pytest.raises(ValueError):
        geo.LigandSet(pos=[[0, 0, 0]], type=[99], charge=[0], mol=[0])


def test_henderson_hasselbalch_values():
    assert geo.fraction_deprotonated(4.25, 4.25) == pytest.approx(0.5)
    assert geo.fraction_deprotonated(5.25, 4.25) == pytest.approx(10 / 11)
    assert geo.fraction_deprotonated(3.25, 4.25) == pytest.approx(1 / 11)
    # Glu (pKa 4.25, as in tools/azimuthal_profile.py) at pH 4.2 is 47% ionised
    assert geo.fraction_deprotonated(4.2, 4.25) == pytest.approx(0.4712, abs=1e-4)


def test_sample_protonation_frequency_and_reproducibility():
    n = 20000
    prot = geo.sample_protonation(n, 4.2, 4.25, np.random.default_rng(1))
    p = 1 - geo.fraction_deprotonated(4.2, 4.25)
    assert abs(prot.mean() - p) < 4 * math.sqrt(p * (1 - p) / n)
    again = geo.sample_protonation(n, 4.2, 4.25, np.random.default_rng(1))
    assert np.array_equal(prot, again)
