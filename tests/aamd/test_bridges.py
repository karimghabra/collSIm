"""aamd.bridges against modcoll.ions.bridges on configurations where an
all-atom frame and a modcoll snapshot describe the same thing."""
import numpy as np
import pytest

pytest.importorskip("numba")
from modcoll.ions import BridgeAnalyzer, IonSampler, LigandSet, carboxylate, guanidinium, load_params  # noqa: E402
from modcoll.ions.geometry import TYPE_INDEX, quat_from_two_vectors  # noqa: E402

from aamd import bridges as B  # noqa: E402

BOX = np.array([6.0, 6.0, 6.0])


def _partners_from_ligands(lig: LigandSet):
    """aamd Partners for the same atoms (frame = ligand positions)."""
    kinds = {TYPE_INDEX["O_carboxylate"]: "O", TYPE_INDEX["N_guanidinium"]: "N_guanidinium",
             TYPE_INDEX["N_ammonium"]: "N_ammonium"}
    rows = [i for i, t in enumerate(lig.type) if int(t) in kinds]
    plane = []
    for i in rows:
        if kinds[int(lig.type[i])] == "N_guanidinium":
            g = [j for j in rows if lig.group[j] == lig.group[i]]
            plane.append(tuple(g[:3]))
        else:
            plane.append(None)
    return B.Partners(np.array(rows), np.array([kinds[int(lig.type[i])] for i in rows]),
                      lig.mol[rows].copy(), lig.group[rows].copy(), plane)


def _modcoll_kinds(params, lig, sampler):
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    return sorted(b.kind for b in an.bridges(sampler.snapshot()))


def test_cation_inner_outer_bridge_matches_modcoll():
    params = load_params()
    lig = LigandSet.concat([carboxylate([2.5, 3.0, 3.0], [1, 0, 0], mol=0),
                            carboxylate([3.3, 3.0, 3.0], [-1, 0, 0], mol=1)])
    o_a = lig.pos[0]
    T = params.tables()
    sp = params.species_index("Ca")
    u0 = T.dirs[sp, 0]
    toward_a = np.array([-1.0, 0.0, 0.0])
    q = quat_from_two_vectors(u0, toward_a)                  # site 0 along -x, the ion between
    centre = o_a - params.value("species.Ca.d_MO") * toward_a  # the rows, pointing at O of molecule 0
    s = IonSampler(params, lig, BOX, [True] * 3, 0.1, np.random.default_rng(0))
    states = np.zeros(T.nsite[sp], np.uint8)
    states[0] = 1
    s.add_ion("Ca", centre, quat=q, states=states)
    ref = _modcoll_kinds(params, lig, s)

    xyz = np.vstack([lig.pos, centre])
    part = _partners_from_ligands(lig)
    mine = sorted(B.kind(b) for b in B.bridges(xyz, BOX, [("Ca", len(lig.pos), ())], part,
                                               B.Thresholds.from_params(params)))
    assert ref == mine
    assert ref == ["inner-outer"]


@pytest.mark.parametrize("shift", [0.0, 0.3])
def test_outer_outer_and_no_bridge_match_modcoll(shift):
    params = load_params()
    lig = LigandSet.concat([carboxylate([2.6 - shift, 3.0, 3.0], [1, 0, 0], mol=0),
                            carboxylate([3.4 + shift, 3.0, 3.0], [-1, 0, 0], mol=1)])
    centre = np.array([3.0, 3.0, 3.0])
    s = IonSampler(params, lig, BOX, [True] * 3, 0.1, np.random.default_rng(0))
    s.add_ion("Mg", centre)
    ref = _modcoll_kinds(params, lig, s)
    xyz = np.vstack([lig.pos, centre])
    mine = sorted(B.kind(b) for b in B.bridges(xyz, BOX, [("Mg", len(lig.pos), ())],
                                               _partners_from_ligands(lig), B.Thresholds.from_params(params)))
    assert ref == mine
    assert (ref == ["outer-outer"]) if shift == 0.0 else (ref == [])


def test_sulfate_direct_bridge_matches_modcoll():
    params = load_params()
    lig = LigandSet.concat([guanidinium([2.55, 3.0, 3.0], [1, 0, 0], normal=[0, 0, 1], mol=0),
                            guanidinium([3.6, 3.0, 3.0], [-1, 0, 0], normal=[0, 0, 1], mol=1)])
    s = IonSampler(params, lig, BOX, [True] * 3, 0.1, np.random.default_rng(0))
    s.add_ion("SO4", [3.07, 3.0, 3.0])
    snap = s.snapshot()
    ref = _modcoll_kinds(params, lig, s)
    d = params.value("species.SO4.d_SO")
    oxy = snap.pos[0] + d * snap.U[0, :4]
    xyz = np.vstack([lig.pos, snap.pos[0], oxy])
    n = len(lig.pos)
    mine = sorted(B.kind(b) for b in B.bridges(xyz, BOX, [("SO4", n, (n + 1, n + 2, n + 3, n + 4))],
                                               _partners_from_ligands(lig), B.Thresholds.from_params(params)))
    assert ref == mine
    assert ref


def test_thresholds_are_modcolls():
    t = B.Thresholds.from_params()
    assert t.inner["Mg"] == pytest.approx(0.247)
    assert t.inner["Ca"] == pytest.approx(0.298)
    assert t.outer["Mg"] == pytest.approx(0.50)
    assert t.outer["Ca"] == pytest.approx(0.538)
    assert t.so4_outer == pytest.approx(0.60)
