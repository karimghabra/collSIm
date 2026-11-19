"""Ion contacts and bridges in all-atom frames, with the definitions of
``modcoll.ions.bridges`` so that the two can be compared number for number.

modcoll's definitions, and how each maps onto an all-atom frame (every
threshold is read from ``modcoll/ions/params.toml`` through ``load_params``):

=====================  ==========================================  =========================================
contact                modcoll                                     all-atom
=====================  ==========================================  =========================================
cation, inner          a site in state L holds the O in its well    O within d_MO + w_r of the cation
                       (|r - d_MO| < w_r, inside the angular cone)  (Mg 0.247, Ca 0.298 nm): the first shell;
                                                                    an all-atom ion has no sites, and an O
                                                                    that close has displaced a water
cation, outer          not inner, O within d_out + w_out            the same (Mg 0.50, Ca 0.538 nm)
sulfate, direct        a sulfate O within d_hb +- w_hb of the       the same, with the four real oxygens
                       donor N (0.25-0.33 nm), and for guanidinium
                       and imidazolium within 35 deg of the plane
sulfate, outer         not direct, N within 0.60 nm of S            the same
=====================  ==========================================  =========================================

Partners: oxygens of deprotonated carboxylates (Asp OD1/OD2, Glu OE1/OE2,
succinyl-Lys OS2/OS3) for the cations; never protonated acid oxygens or
backbone carbonyls. For sulfate: Arg NE/NH1/NH2, charged Lys NZ, His(+)
ND1/NE2.

A bridge is an ion in contact with atoms of both segments; its kind is the
best contact class on each side (inner beats outer), as in modcoll.

Units: nm.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from modcoll.ions import geometry as mgeo
from modcoll.ions.params import load_params

CARBOXYLATE_O = {"ASP": ("OD1", "OD2", "CG"), "GLU": ("OE1", "OE2", "CD"), "SLL": ("OS2", "OS3", "CS4")}
CARBOXYLIC_ACID_O = {"ASH": ("OD1", "OD2", "CG"), "GLH": ("OE1", "OE2", "CD"), "SLH": ("OS2", "OS3", "CS4")}
DONORS = {"ARG": (("NE", "NH1", "NH2"), "N_guanidinium", ("CZ",)),
          "LYS": (("NZ",), "N_ammonium", ()),
          "HIP": (("ND1", "NE2"), "N_imidazolium", ())}
ION_RESNAMES = {"MG": "Mg", "CA": "Ca", "SO4": "SO4"}


@dataclasses.dataclass
class Partners:
    """Partner atoms of the solute, as indices into a frame's coordinates."""

    index: np.ndarray        # atom index
    kind: np.ndarray         # "O" (carboxylate) or the donor type name
    mol: np.ndarray          # 0 = segment A, 1 = segment B
    group: np.ndarray        # one id per chemical group
    plane: list              # per atom: tuple of three atom indices spanning the
                             # donor plane, or None


def find_partners(topology, segment_of_residue) -> Partners:
    """``segment_of_residue(residue) -> 0, 1 or None``."""
    idx, kind, mol, grp, plane = [], [], [], [], []
    g = 0
    for res in topology.residues():
        s = segment_of_residue(res)
        if s is None:
            continue
        names = {a.name: a.index for a in res.atoms()}
        if res.name in CARBOXYLATE_O:
            o1, o2, _ = CARBOXYLATE_O[res.name]
            for o in (o1, o2):
                idx.append(names[o]); kind.append("O"); mol.append(s); grp.append(g); plane.append(None)
            g += 1
        elif res.name in DONORS:
            atoms, t, _ = DONORS[res.name]
            if res.name == "ARG":
                pl = (names["NE"], names["NH1"], names["NH2"])
            elif res.name == "HIP":
                pl = (names["CG"], names["ND1"], names["NE2"])
            else:
                pl = None
            for a in atoms:
                idx.append(names[a]); kind.append(t); mol.append(s); grp.append(g); plane.append(pl)
            g += 1
    return Partners(np.array(idx, int), np.array(kind), np.array(mol, int), np.array(grp, int), plane)


@dataclasses.dataclass(frozen=True)
class Thresholds:
    inner: dict      # cation -> radius
    outer: dict      # cation -> radius
    d_hb: float
    w_hb: float
    sin_plane: float
    so4_outer: float

    @classmethod
    def from_params(cls, params=None):
        p = load_params() if params is None else params
        inner = {c: p.value(f"species.{c}.d_MO") + p.value(f"species.{c}.well_radial_halfwidth")
                 for c in ("Mg", "Ca")}
        outer = {c: p.value(f"species.{c}.outer_shell_distance") + p.value(f"species.{c}.outer_shell_halfwidth")
                 for c in ("Mg", "Ca")}
        return cls(inner, outer, p.value("species.SO4.hbond_distance"),
                   p.value("species.SO4.hbond_halfwidth"),
                   float(np.sin(np.radians(p.value("species.SO4.hbond_plane_halfwidth_deg")))),
                   p.value("species.SO4.outer_contact_distance"))


INNER, OUTER = 1, 2


def _mi(d, box):
    return d - box * np.round(d / box)


def contacts(xyz, box, ions, partners: Partners, thr: Thresholds):
    """Contact list of one frame.

    ``ions``: list of (species, centre atom index, oxygen atom indices) for
    every Mg, Ca or SO4 in the frame. Returns rows (ion number, partner row,
    class, distance)."""
    out = []
    P = xyz[partners.index]
    for n, (sp, centre, oxy) in enumerate(ions):
        c = xyz[centre]
        if sp in ("Mg", "Ca"):
            m = partners.kind == "O"
            if not m.any():
                continue
            rows = np.flatnonzero(m)
            d = np.linalg.norm(_mi(P[rows] - c, box), axis=1)
            for r, dist in zip(rows, d):
                if dist < thr.inner[sp]:
                    out.append((n, r, INNER, dist))
                elif dist < thr.outer[sp]:
                    out.append((n, r, OUTER, dist))
        else:
            rows = np.flatnonzero(partners.kind != "O")
            if rows.size == 0:
                continue
            dS = np.linalg.norm(_mi(P[rows] - c, box), axis=1)
            for r, dist in zip(rows, dS):
                if dist > max(thr.so4_outer, 0.15 + thr.d_hb + thr.w_hb):
                    continue
                cls = 0
                N = P[r]
                pl = partners.plane[r]
                normal = None
                if pl is not None:
                    a, b, e = (xyz[i] for i in pl)
                    normal = np.cross(_mi(b - a, box), _mi(e - a, box))
                    normal /= np.linalg.norm(normal)
                for o in oxy:
                    v = _mi(xyz[o] - N, box)
                    aa = np.linalg.norm(v)
                    if abs(aa - thr.d_hb) < thr.w_hb:
                        if normal is None or abs(v @ normal) / aa < thr.sin_plane:
                            cls = INNER
                            break
                if cls == 0 and dist < thr.so4_outer:
                    cls = OUTER
                if cls:
                    out.append((n, r, cls, dist))
    return out


def bridges(xyz, box, ions, partners: Partners, thr: Thresholds):
    """Bridges of one frame: list of (species, class on A, class on B,
    n protein groups held, z of the ion)."""
    rows = contacts(xyz, box, ions, partners, thr)
    by_ion = {}
    for n, r, cls, d in rows:
        by_ion.setdefault(n, []).append((r, cls))
    out = []
    for n, lst in by_ion.items():
        side = {}
        groups = set()
        for r, cls in lst:
            m = int(partners.mol[r])
            side[m] = min(side.get(m, 9), cls)
            groups.add(int(partners.group[r]))
        if 0 in side and 1 in side:
            name = {INNER: "inner", OUTER: "outer"}
            out.append((ions[n][0], name[side[0]], name[side[1]], len(groups), float(xyz[ions[n][1], 2])))
    return out


def kind(b):
    return "-".join(sorted((b[1], b[2])))


def find_ions(topology):
    """(species, centre atom, oxygen atoms) of every divalent ion / sulfate."""
    out = []
    for res in topology.residues():
        sp = ION_RESNAMES.get(res.name)
        if sp is None:
            continue
        atoms = {a.name: a.index for a in res.atoms()}
        if sp == "SO4":
            out.append((sp, atoms["S"], tuple(atoms[k] for k in ("O1", "O2", "O3", "O4"))))
        else:
            out.append((sp, next(iter(atoms.values())), ()))
    return out


# --------------------------------------------- the coarse model, same frame ---
def ligandset_from_frame(xyz, topology, segment_of_residue, shift=None) -> mgeo.LigandSet:
    """modcoll LigandSet from the real atom positions of one frame: every
    carboxylate (both O and the carbon as inert), carboxylic acid (O as the
    weak-ligand type), guanidinium (three N and CZ), charged Lys NZ and His(+)
    ring N, with modcoll's charges, molecule ids from the segment and plane
    normals from the real groups."""
    T = mgeo.TYPE_INDEX
    pos, typ, q, mol, grp, nrm = [], [], [], [], [], []
    g = 0
    sh = np.zeros(3) if shift is None else np.asarray(shift)

    def add(i, t, c, s, n=(0.0, 0.0, 0.0)):
        pos.append(xyz[i] + sh); typ.append(T[t]); q.append(c); mol.append(s); grp.append(g); nrm.append(n)

    for res in topology.residues():
        s = segment_of_residue(res)
        if s is None:
            continue
        a = {x.name: x.index for x in res.atoms()}
        if res.name in CARBOXYLATE_O or res.name in CARBOXYLIC_ACID_O:
            acid = res.name in CARBOXYLIC_ACID_O
            o1, o2, c = (CARBOXYLIC_ACID_O if acid else CARBOXYLATE_O)[res.name]
            n = np.cross(xyz[a[o1]] - xyz[a[c]], xyz[a[o2]] - xyz[a[c]])
            n /= np.linalg.norm(n)
            for o in (o1, o2):
                add(a[o], "O_carboxylic_acid" if acid else "O_carboxylate", 0.0 if acid else -0.5, s, n)
            add(a[c], "inert", 0.0, s)
            g += 1
        elif res.name == "ARG":
            n = np.cross(xyz[a["NH1"]] - xyz[a["NE"]], xyz[a["NH2"]] - xyz[a["NE"]])
            n /= np.linalg.norm(n)
            for o in ("NE", "NH1", "NH2"):
                add(a[o], "N_guanidinium", 1.0 / 3.0, s, n)
            add(a["CZ"], "inert", 0.0, s)
            g += 1
        elif res.name == "LYS":
            add(a["NZ"], "N_ammonium", 1.0, s)
            g += 1
        elif res.name == "HIP":
            n = np.cross(xyz[a["ND1"]] - xyz[a["CG"]], xyz[a["NE2"]] - xyz[a["CG"]])
            n /= np.linalg.norm(n)
            for o in ("ND1", "NE2"):
                add(a[o], "N_imidazolium", 0.5, s, n)
            g += 1
    return mgeo.LigandSet(pos=pos, type=typ, charge=q, mol=mol, group=grp, normal=nrm)
