"""aamd.coarse is the modcoll site model, evaluated at all-atom geometry."""
import numpy as np
import pytest

from modcoll import electrostatics as es
from modcoll.sites import SiteType, make_molecule

from aamd import coarse


@pytest.fixture(scope="module")
def pair():
    return coarse.segment(624), coarse.segment(624 - 234)


def test_segment_has_thirty_residues_per_chain(pair):
    a, _ = pair
    assert len(a) == 90
    assert sorted(set(a.chain.tolist())) == [0, 1, 2]
    assert a.res_index.min() == 624 and a.res_index.max() == 653


def test_cartesian_sum_equals_modcolls_direct_pair_sum(pair):
    a, b = pair
    qa, qb = coarse.mean_charges(a, 7.4), coarse.mean_charges(b, 7.4)
    stagger = 234 * 66.88 / 234
    d = 1.65
    lam = es.debye_length_nm(0.15)
    mol_a = make_molecule(a.s, a.phi, a.r, a.site_type, helix_length_nm=300.0)
    mol_b = make_molecule(b.s, b.phi, b.r, b.site_type, helix_length_nm=300.0)
    for fa, fb in ((0.0, 0.0), (0.7, -1.9), (2.5, 3.1)):
        ref = es.pair_energy_direct(mol_a, qa, mol_b, qb, lambda r: es.screened_coulomb(r, lam),
                                    stagger, fa, fb, d)
        xa, xb = coarse.pair_positions(a, b, stagger, d, rot_a=-fa, rot_b=np.pi - fb)
        el, hy = coarse.pair_terms(xa, qa, a.h, xb, qb, b.h)
        assert el == pytest.approx(ref, rel=1e-9, abs=1e-9)
        ref_h = -es.EPS_HYDROPHOBIC * es.pair_energy_direct(
            mol_a, a.h, mol_b, b.h, es.hydrophobic_contact, stagger, fa, fb, d)
        assert hy == pytest.approx(ref_h, rel=1e-9, abs=1e-9)


def test_discrete_charges_match_henderson_hasselbalch_counts(pair):
    a, _ = pair
    s = coarse.succinylate(a)
    q = coarse.discrete_charges(s, 4.2)
    mean = coarse.mean_charges(s, 4.2)
    for st in (SiteType.ASP, SiteType.GLU, SiteType.SUCC_LYS, SiteType.ARG):
        m = s.site_type == int(st)
        assert np.sum(np.abs(q[m])) == pytest.approx(round(abs(mean[m].sum())), abs=0.5 + 1e-9)
    assert set(np.unique(q)) <= {-1.0, 0.0, 1.0}
    assert not np.any(s.site_type == int(SiteType.LYS))


def test_boltzmann_reduction_bounds():
    e = np.random.default_rng(0).normal(size=(3, 6, 6))
    f = coarse.reduce_boltzmann(e)
    assert np.all(f >= coarse.reduce_min(e) - 1e-12)
    assert np.all(f <= e.reshape(3, -1).mean(axis=1) + 1e-12)
