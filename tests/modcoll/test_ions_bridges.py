"""Contact classes, bridge records and lifetimes on hand-made configurations."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root

from modcoll.ions import geometry as geo  # noqa: E402
from modcoll.ions.bridges import (INNER, OUTER, Bridge, BridgeAnalyzer, BridgeTracker,  # noqa: E402
                                  axial_histogram, count_bridges)
from modcoll.ions.energy import MAX_SITES, rotate_dirs  # noqa: E402
from modcoll.ions.params import load_params  # noqa: E402
from modcoll.ions.sampler import Snapshot  # noqa: E402

BOX = np.array([6.0, 6.0, 6.0])
C = BOX / 2
O_CARB = geo.TYPE_INDEX["O_carboxylate"]
O_ACID = geo.TYPE_INDEX["O_carboxylic_acid"]
N_AMM = geo.TYPE_INDEX["N_ammonium"]


@pytest.fixture(scope="module")
def params():
    return load_params()


def atoms(pos, types, mols, charges=None):
    n = len(pos)
    return geo.LigandSet(pos=pos, type=types, charge=charges if charges is not None else [0.0] * n,
                         mol=mols, group=list(range(n)))


def snapshot(params, ions):
    """ions: list of (species name, position, quaternion, states or None)."""
    T = params.tables()
    n = len(ions)
    sp = np.zeros(n, dtype=np.int64)
    pos = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    U = np.zeros((n, MAX_SITES, 3))
    state = np.zeros((n, MAX_SITES), dtype=np.uint8)
    for i, (name, r, q, st) in enumerate(ions):
        sp[i] = params.species_index(name)
        pos[i] = r
        quat[i] = q
        rotate_dirs(T, sp[i], quat[i], U[i])
        if st is not None:
            state[i, : len(st)] = st
    return Snapshot(sp, pos, quat, U, state, np.arange(100, 100 + n))


IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_calcium_between_two_molecules_through_two_open_sites(params):
    """Ca2+ at the centre, its +z and -z sites open, one carboxylate oxygen of
    molecule 0 on the +z site and one of molecule 1 on the -z site."""
    d = float(params.value("species.Ca.d_MO"))
    lig = atoms([C + [0, 0, d], C - [0, 0, d]], [O_CARB, O_CARB], [0, 1])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    snap = snapshot(params, [("Ca", C, IDENT, [1, 1, 0, 0, 0, 0, 0])])
    c = an.contacts(snap)
    assert sorted(c.atom.tolist()) == [0, 1]
    assert np.all(c.cls == INNER)
    assert np.allclose(c.dist, d)
    (b,) = an.bridges(snap)
    assert (b.mol_a, b.mol_b) == (0, 1)
    assert b.kind == "inner-inner"
    assert b.species == "Ca" and b.uid == 100
    assert b.angle_deg == pytest.approx(180.0)
    assert b.span == pytest.approx(2 * d)
    assert (b.dist_a, b.dist_b) == pytest.approx((d, d))
    assert (b.denticity_a, b.denticity_b) == (1, 1)
    assert (b.ligands_a, b.ligands_b, b.n_ligands) == (1, 1, 2)
    assert b.z == pytest.approx(C[2])
    assert (b.atom_a, b.atom_b) == (0, 1)


def test_the_same_atoms_with_closed_sites_are_not_inner_contacts(params):
    """Geometry alone does not make an inner-sphere contact: the site must
    have given up its water."""
    d = float(params.value("species.Ca.d_MO"))
    lig = atoms([C + [0, 0, d], C - [0, 0, d]], [O_CARB, O_CARB], [0, 1])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    snap = snapshot(params, [("Ca", C, IDENT, None)])
    c = an.contacts(snap)
    assert np.all(c.cls == OUTER)     # within r_contact, but through no open site


def test_magnesium_bridging_through_its_water_shell(params):
    """Mg2+ fully hydrated, a carboxylate oxygen of each molecule in the
    second shell on opposite sides: an outer-outer bridge, 2 * 0.42 nm long."""
    r = float(params.value("species.Mg.outer_shell_distance"))
    u = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
    lig = atoms([C + r * u, C - r * u], [O_CARB, O_CARB], [3, 7])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("Mg", C, IDENT, None)]))
    assert b.kind == "outer-outer"
    assert (b.mol_a, b.mol_b) == (3, 7)
    assert b.span == pytest.approx(2 * r)
    assert (b.denticity_a, b.denticity_b) == (0, 0)


def test_mixed_bridge_and_class_ordering(params):
    d = float(params.value("species.Ca.d_MO"))
    r = float(params.value("species.Ca.outer_shell_distance"))
    lig = atoms([C + [0, 0, d], C + [r, 0, 0]], [O_CARB, O_CARB], [0, 1])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("Ca", C, IDENT, [1, 0, 0, 0, 0, 0, 0])]))
    assert b.kind == "inner-outer"
    assert (b.class_a, b.class_b) == ("inner", "outer")
    assert b.angle_deg == pytest.approx(90.0)
    assert b.span == pytest.approx(math.hypot(d, r))


def test_one_carboxylate_alone_is_not_a_bridge(params):
    """An ion on a single carboxylate, inner- or outer-sphere, joins nothing."""
    d = float(params.value("species.Ca.d_MO"))
    lig = geo.carboxylate(C, [0, 0, 1], mol=0)
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    ion = lig.pos[0] + np.array([0.0, 0.0, d])
    snap = snapshot(params, [("Ca", ion, geo.quat_from_two_vectors([0, 0, 1], [0, 0, -1]),
                              [1, 0, 0, 0, 0, 0, 0])])
    c = an.contacts(snap)
    assert INNER in c.cls.tolist()
    assert an.bridges(snap) == []


def test_two_contacts_with_one_molecule_are_not_a_bridge(params):
    d = float(params.value("species.Ca.d_MO"))
    lig = atoms([C + [0, 0, d], C - [0, 0, d]], [O_CARB, O_CARB], [5, 5])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    snap = snapshot(params, [("Ca", C, IDENT, [1, 1, 0, 0, 0, 0, 0])])
    assert len(an.contacts(snap)) == 2
    assert an.bridges(snap) == []


def test_protonated_acid_and_far_atoms_are_not_contacts(params):
    r = float(params.value("species.Ca.outer_shell_distance"))
    rc = r + float(params.value("species.Ca.outer_shell_halfwidth"))
    lig = atoms([C + [r, 0, 0], C - [rc + 0.02, 0, 0], C + [0, rc - 0.02, 0]],
                [O_ACID, O_CARB, O_CARB], [0, 1, 2])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    c = an.contacts(snapshot(params, [("Ca", C, IDENT, None)]))
    assert c.atom.tolist() == [2]            # only the charged oxygen inside r_contact


def test_bidentate_side_counts_two_inner_contacts(params):
    """A carboxylate of molecule 0 chelating two adjacent equatorial sites of
    Ca2+ (72 deg apart; the 55 deg bite fits within the angular slack), and an
    outer-sphere oxygen of molecule 1."""
    T = params.tables()
    sp = params.species_index("Ca")
    d = float(T.d_site[sp])
    U = np.zeros((MAX_SITES, 3))
    rotate_dirs(T, sp, IDENT, U)
    e1, e2 = U[2], U[3]                           # two neighbouring equatorial sites
    bis = (e1 + e2) / np.linalg.norm(e1 + e2)
    perp = (e2 - e1) / np.linalg.norm(e2 - e1)
    half_oo = 0.125 * math.sin(math.radians(62.5))
    reach = math.sqrt(d ** 2 - half_oo ** 2)
    o1, o2 = C + reach * bis - half_oo * perp, C + reach * bis + half_oo * perp
    r_out = float(T.d_out[sp])
    lig = geo.LigandSet(pos=[o1, o2, C - r_out * bis], type=[O_CARB] * 3, charge=[-0.5, -0.5, -0.5],
                        mol=[0, 0, 1], group=[0, 0, 1])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("Ca", C, IDENT, [0, 0, 1, 1, 0, 0, 0])]))
    assert b.kind == "inner-outer"
    assert (b.denticity_a, b.denticity_b) == (2, 0)
    assert b.group_a == 0 and b.group_b == 1
    # two oxygens of one carboxylate are one protein ligand
    assert (b.ligands_a, b.ligands_b, b.n_ligands) == (1, 1, 2)


def test_both_oxygens_of_a_carboxylate_on_one_calcium_site(params):
    """The calcium site on the carboxylate's bisector holds both oxygens (each
    27.8 deg off the axis at the bond length): two inner contacts through one
    open site, one protein ligand."""
    T = params.tables()
    sp = params.species_index("Ca")
    d = float(T.d_site[sp])
    half_oo = 0.125 * math.sin(math.radians(62.5))
    reach = math.sqrt(d ** 2 - half_oo ** 2)
    o1, o2 = C + [half_oo, 0.0, reach], C + [-half_oo, 0.0, reach]      # around the +z site
    lig = geo.LigandSet(pos=[o1, o2], type=[O_CARB] * 2, charge=[-0.5, -0.5], mol=[0, 0], group=[0, 0])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    c = an.contacts(snapshot(params, [("Ca", C, IDENT, [1, 0, 0, 0, 0, 0, 0])]))
    assert sorted(c.atom.tolist()) == [0, 1] and np.all(c.cls == INNER)
    assert np.allclose(c.dist, d)
    # magnesium in the same place does not: its well is too narrow
    d_mg = float(params.value("species.Mg.d_MO"))
    reach = math.sqrt(d_mg ** 2 - half_oo ** 2)
    lig = geo.LigandSet(pos=[C + [half_oo, 0.0, reach], C + [-half_oo, 0.0, reach]], type=[O_CARB] * 2,
                        charge=[-0.5, -0.5], mol=[0, 0], group=[0, 0])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    q = geo.quat_from_two_vectors([1, 0, 0], [0, 0, 1])                 # an octahedral site on +z
    c = an.contacts(snapshot(params, [("Mg", C, q, [1, 0, 0, 0, 0, 0])]))
    assert INNER not in c.cls.tolist()


def test_sulfate_bridging_two_ammonium_groups(params):
    """Sulfate with one oxygen hydrogen-bonded to an ammonium nitrogen of each
    of two molecules: a direct-direct ('inner-inner') bridge."""
    T = params.tables()
    sp = params.species_index("SO4")
    d, d_hb = float(T.d_site[sp]), float(T.d_hb[sp])
    U = np.zeros((MAX_SITES, 3))
    rotate_dirs(T, sp, IDENT, U)
    n1 = C + (d + d_hb) * U[0]
    n2 = C + (d + d_hb) * U[1]
    lig = atoms([n1, n2], [N_AMM, N_AMM], [0, 1], charges=[1.0, 1.0])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("SO4", C, IDENT, None)]))
    assert b.species == "SO4" and b.kind == "inner-inner"
    assert b.angle_deg == pytest.approx(math.degrees(math.acos(-1 / 3)))
    # moved out by a hydrogen-bond half-width and more, the pair is only 'outer'
    n1_far = C + (d + d_hb + 0.08) * U[0]
    lig = atoms([n1_far, n2], [N_AMM, N_AMM], [0, 1], charges=[1.0, 1.0])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("SO4", C, IDENT, None)]))
    assert b.kind == "inner-outer"


def test_cations_do_not_count_ammonium_nor_sulfate_carboxylate(params):
    lig = atoms([C + [0.4, 0, 0], C - [0.4, 0, 0]], [N_AMM, O_CARB], [0, 1], charges=[1.0, -0.5])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    c = an.contacts(snapshot(params, [("Ca", C, IDENT, None)]))
    assert c.atom.tolist() == [1]
    c = an.contacts(snapshot(params, [("SO4", C, IDENT, None)]))
    assert c.atom.tolist() == [0]


def test_ion_touching_three_molecules_gives_three_bridges(params):
    r = float(params.value("species.Mg.outer_shell_distance"))
    dirs = [np.array(v, float) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1])]
    lig = atoms([C + r * v / np.linalg.norm(v) for v in dirs], [O_CARB] * 3, [0, 1, 2])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    bridges = an.bridges(snapshot(params, [("Mg", C, IDENT, None)]))
    assert sorted((b.mol_a, b.mol_b) for b in bridges) == [(0, 1), (0, 2), (1, 2)]
    assert count_bridges(bridges) == {("Mg", "outer-outer"): 3}
    assert all(b.n_ligands == 3 for b in bridges)
    assert count_bridges(bridges, by_ligands=True) == {("Mg", "outer-outer", 3): 3}


def test_contact_across_a_periodic_boundary(params):
    r = float(params.value("species.Mg.outer_shell_distance"))
    ion = np.array([0.1, 3.0, 3.0])
    lig = atoms([[BOX[0] - (r - 0.1), 3.0, 3.0], [0.1 + r, 3.0, 3.0]], [O_CARB, O_CARB], [0, 1])
    an = BridgeAnalyzer(params, lig, BOX, [True] * 3)
    (b,) = an.bridges(snapshot(params, [("Mg", ion, IDENT, None)]))
    assert b.span == pytest.approx(2 * r)
    assert b.angle_deg == pytest.approx(180.0)
    # with a wall instead of a periodic boundary the far atom is out of reach
    an = BridgeAnalyzer(params, lig, BOX, [False, True, True])
    assert an.bridges(snapshot(params, [("Mg", ion, IDENT, None)])) == []


def make_bridge(uid, kind=("outer", "outer"), z=1.0, mol=(0, 1), species="Ca", n_ligands=2):
    return Bridge(uid=uid, species=species, position=(0.0, 0.0, z), z=z, mol_a=mol[0], mol_b=mol[1],
                  class_a=kind[0], class_b=kind[1], atom_a=0, atom_b=1, group_a=0, group_b=1,
                  dist_a=0.4, dist_b=0.4, denticity_a=0, denticity_b=0, ligands_a=1, ligands_b=1,
                  n_ligands=n_ligands, angle_deg=180.0, span=0.8)


def test_lifetimes_from_a_hand_made_presence_pattern():
    """Ion 1 bridges in samples 0-2 and again in 4-5; ion 2 in sample 1 only;
    ion 3 from sample 4 to the end. Sampling interval 10 sweeps.
    Completed lifetimes: ion 1 -> 30, ion 2 -> 10, ion 1 again -> 20.
    Still alive at the end (censored): ion 3 -> 30."""
    pattern = [
        [1], [1, 2], [1], [], [1, 3], [1, 3], [3],
    ]
    tr = BridgeTracker(interval=10)
    for present in pattern:
        tr.update([make_bridge(u) for u in present])
    tr.finish()
    assert sorted(tr.lifetimes().tolist()) == [10.0, 20.0, 30.0]
    assert tr.lifetimes(include_censored=True).sum() == 10 + 20 + 30 + 30
    assert [t for _, _, t in tr.censored] == [30]


def test_tracker_distinguishes_molecule_pairs_and_optionally_kinds():
    tr = BridgeTracker(interval=1)
    tr.update([make_bridge(1, mol=(0, 1))])
    tr.update([make_bridge(1, mol=(0, 2))])     # same ion, different pair: a new bridge
    tr.finish()
    assert tr.lifetimes().tolist() == [1.0]
    assert len(tr.censored) == 1
    # a bridge that changes from outer-outer to inner-outer
    seq = [("outer", "outer"), ("outer", "outer"), ("inner", "outer")]
    plain, by_kind = BridgeTracker(1), BridgeTracker(1, by_kind=True)
    for kind in seq:
        plain.update([make_bridge(5, kind=kind)])
        by_kind.update([make_bridge(5, kind=kind)])
    plain.finish()
    by_kind.finish()
    assert plain.lifetimes(include_censored=True).tolist() == [3.0]
    assert sorted(by_kind.lifetimes(include_censored=True).tolist()) == [1.0, 2.0]


def test_axial_histogram_and_counts():
    bridges = [make_bridge(1, z=0.5), make_bridge(2, z=1.5), make_bridge(3, z=1.7, species="Mg"),
               make_bridge(4, kind=("inner", "outer"), z=9.0)]
    assert axial_histogram(bridges, [0, 1, 2, 10]).tolist() == [1, 2, 1]
    assert count_bridges(bridges) == {("Ca", "outer-outer"): 2, ("Mg", "outer-outer"): 1,
                                      ("Ca", "inner-outer"): 1}
    assert make_bridge(1, kind=("outer", "inner")).kind == "inner-outer"
