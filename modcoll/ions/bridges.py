"""Contacts and bridges.

A *contact* is geometric and joins one ion to one partner atom:

  cation, inner   a site in state L has the atom inside its inner well
  cation, outer   not inner, and the atom is within r_contact of the centre
                  (r_contact = outer shell distance + half-width)
  sulfate, inner  ("direct") an oxygen is inside the hydrogen-bond well of the
                  donor atom
  sulfate, outer  not direct, and the donor atom is within r_contact of sulfur

Partner atoms are carboxylate oxygens for the cations and guanidinium,
ammonium and imidazolium nitrogens for sulfate. Protonated carboxylic acid
oxygens are not partners.

A *bridge* is an ion in contact with atoms of two different molecules; an ion
on a single carboxylate, or on several groups of one molecule, is not one. It
is recorded once per pair of molecules, with the contact class on each side
kept separately, so inner-inner, inner-outer and outer-outer bridges can be
told apart, and with the number of protein ligand groups the ion holds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from ._jit import njit
from .energy import KIND_CATION, L_STATE, inner_weight
from .params import IonParams
from .sampler import Snapshot, _cell_index, _cell_range, _mi, build_system

INNER = 1
OUTER = 2
CLASS_NAMES = {INNER: "inner", OUTER: "outer"}


@njit
def contacts_kernel(T, S, sp_arr, pos, U, state, out_ion, out_atom, out_class, out_dist, out_vec):
    """Enumerate ion-atom contacts. Writes up to len(out_ion) rows and returns
    the number found (which may exceed the capacity; the caller then retries).
    out_atom indexes the cell-sorted atom arrays of S; out_vec is the
    minimum-image vector from the ion centre to the atom."""
    cap = out_ion.shape[0]
    count = 0
    for i in range(sp_arr.shape[0]):
        sp = sp_arr[i]
        n = T.nsite[sp]
        cation = T.kind[sp] == KIND_CATION
        rc = T.r_contact[sp]
        # direct sulfate contacts can lie slightly beyond r_contact from sulfur
        reach = max(rc, T.d_site[sp] + T.d_hb[sp] + T.w_hb[sp])
        lo0, hi0, _ = _cell_range(S, 0, pos[i, 0])
        lo1, hi1, _ = _cell_range(S, 1, pos[i, 1])
        lo2, hi2, _ = _cell_range(S, 2, pos[i, 2])
        for j0 in range(lo0, hi0):
            c0 = _cell_index(S, 0, j0)
            if c0 < 0:
                continue
            for j1 in range(lo1, hi1):
                c1 = _cell_index(S, 1, j1)
                if c1 < 0:
                    continue
                for j2 in range(lo2, hi2):
                    c2 = _cell_index(S, 2, j2)
                    if c2 < 0:
                        continue
                    cid = (c0 * S.ncell[1] + c1) * S.ncell[2] + c2
                    a0 = S.cstart[cid]
                    for a in range(a0, a0 + S.ccount[cid]):
                        t = S.atype[a]
                        if not T.partner[sp, t]:
                            continue
                        dx = _mi(S, 0, S.apos[a, 0] - pos[i, 0])
                        dy = _mi(S, 1, S.apos[a, 1] - pos[i, 1])
                        dz = _mi(S, 2, S.apos[a, 2] - pos[i, 2])
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 >= reach * reach or r2 < 1e-18:
                            continue
                        r = math.sqrt(r2)
                        cls = 0
                        if cation:
                            for k in range(n):
                                if state[i, k] == L_STATE and inner_weight(T, sp, U[i], k, dx, dy, dz, r) > 0.0:
                                    cls = INNER
                                    break
                        else:
                            d = T.d_site[sp]
                            planar = (S.anorm[a, 0] ** 2 + S.anorm[a, 1] ** 2 + S.anorm[a, 2] ** 2) > 0.25
                            for k in range(n):
                                ax = d * U[i, k, 0] - dx
                                ay = d * U[i, k, 1] - dy
                                az = d * U[i, k, 2] - dz
                                aa = math.sqrt(ax * ax + ay * ay + az * az)
                                if abs(aa - T.d_hb[sp]) < T.w_hb[sp]:
                                    if planar:
                                        s = (ax * S.anorm[a, 0] + ay * S.anorm[a, 1] + az * S.anorm[a, 2]) \
                                            / (aa * T.sin_plane_max[sp])
                                        if s * s < 1.0:
                                            cls = INNER
                                            break
                                    else:
                                        cls = INNER
                                        break
                        if cls == 0 and r < rc:
                            cls = OUTER
                        if cls != 0:
                            if count < cap:
                                out_ion[count] = i
                                out_atom[count] = a
                                out_class[count] = cls
                                out_dist[count] = r
                                out_vec[count, 0] = dx
                                out_vec[count, 1] = dy
                                out_vec[count, 2] = dz
                            count += 1
    return count


@dataclass
class Contacts:
    """Ion-atom contacts of one snapshot (parallel arrays)."""

    ion: np.ndarray    # index into the snapshot
    atom: np.ndarray   # index into the LigandSet handed to the analyzer
    cls: np.ndarray    # INNER or OUTER
    dist: np.ndarray   # ion centre to atom, nm
    vec: np.ndarray    # minimum-image vector centre -> atom, nm

    def __len__(self) -> int:
        return len(self.ion)


@dataclass(frozen=True)
class Bridge:
    """One ion joining two molecules."""

    uid: int            # persistent ion identity
    species: str
    position: tuple     # ion centre (x, y, z), nm
    z: float            # ion position along the molecule axis, nm
    mol_a: int
    mol_b: int          # mol_a < mol_b
    class_a: str        # "inner" or "outer": best contact with molecule a
    class_b: str
    atom_a: int         # the closest atom of the best class on each side
    atom_b: int
    group_a: int
    group_b: int
    dist_a: float       # ion centre to atom_a, nm
    dist_b: float
    denticity_a: int    # number of inner contacts with molecule a
    denticity_b: int
    ligands_a: int      # distinct chemical groups of molecule a in contact
    ligands_b: int
    n_ligands: int      # distinct protein groups in contact with the ion, all
                        # molecules together (2 = the minimal bridge)
    angle_deg: float    # angle atom_a - ion - atom_b
    span: float         # distance atom_a to atom_b through the ion's image, nm

    @property
    def kind(self) -> str:
        """'inner-inner', 'inner-outer' or 'outer-outer'."""
        return "-".join(sorted((self.class_a, self.class_b)))


class BridgeAnalyzer:
    """Finds contacts and bridges for snapshots of one system."""

    def __init__(self, params: IonParams, ligands: geo.LigandSet, box, periodic):
        self.params = params
        self.ligands = ligands
        self.T = params.tables()
        reach = float(max(self.T.r_contact.max(), (self.T.d_site + self.T.d_hb + self.T.w_hb).max())) + 0.01
        self.S, self.order = build_system(ligands, box, periodic, None, reach)

    def contacts(self, snap: Snapshot) -> Contacts:
        cap = max(64, 16 * len(snap.species))
        while True:
            oi = np.zeros(cap, dtype=np.int64)
            oa = np.zeros(cap, dtype=np.int64)
            oc = np.zeros(cap, dtype=np.int64)
            od = np.zeros(cap)
            ov = np.zeros((cap, 3))
            n = contacts_kernel(self.T, self.S, snap.species, snap.pos, snap.U, snap.state, oi, oa, oc, od, ov)
            if n <= cap:
                break
            cap = 2 * n
        return Contacts(oi[:n], self.order[oa[:n]], oc[:n], od[:n], ov[:n])

    def bridges(self, snap: Snapshot, contacts: Contacts | None = None) -> list[Bridge]:
        c = self.contacts(snap) if contacts is None else contacts
        lig = self.ligands
        out = []
        for i in np.unique(c.ion):
            rows = np.flatnonzero(c.ion == i)
            mols = lig.mol[c.atom[rows]]
            sides = {}
            for m in np.unique(mols):
                r = rows[mols == m]
                best_cls = int(c.cls[r].min())              # INNER (1) beats OUTER (2)
                r_best = r[c.cls[r] == best_cls]
                pick = r_best[np.argmin(c.dist[r_best])]
                sides[int(m)] = (best_cls, int(pick), int((c.cls[r] == INNER).sum()),
                                 len(np.unique(lig.group[c.atom[r]])))
            keys = sorted(sides)
            n_ligands = sum(v[3] for v in sides.values())
            for ia in range(len(keys)):
                for ib in range(ia + 1, len(keys)):
                    ca, ra, da, la = sides[keys[ia]]
                    cb, rb, db, lb = sides[keys[ib]]
                    va, vb = c.vec[ra], c.vec[rb]
                    cosang = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
                    out.append(Bridge(
                        uid=int(snap.uid[i]), species=self.params.species[int(snap.species[i])],
                        position=tuple(float(x) for x in snap.pos[i]), z=float(snap.pos[i, 2]),
                        mol_a=keys[ia], mol_b=keys[ib],
                        class_a=CLASS_NAMES[ca], class_b=CLASS_NAMES[cb],
                        atom_a=int(c.atom[ra]), atom_b=int(c.atom[rb]),
                        group_a=int(lig.group[c.atom[ra]]), group_b=int(lig.group[c.atom[rb]]),
                        dist_a=float(c.dist[ra]), dist_b=float(c.dist[rb]),
                        denticity_a=da, denticity_b=db, ligands_a=la, ligands_b=lb, n_ligands=n_ligands,
                        angle_deg=math.degrees(math.acos(max(-1.0, min(1.0, cosang)))),
                        span=float(np.linalg.norm(va - vb)),
                    ))
        return out


def count_bridges(bridges, by_ligands: bool = False) -> dict:
    """Number of bridges by (species, kind), or by (species, kind, number of
    protein ligand groups on the ion) when ``by_ligands`` is set."""
    out: dict = {}
    for b in bridges:
        key = (b.species, b.kind, b.n_ligands) if by_ligands else (b.species, b.kind)
        out[key] = out.get(key, 0) + 1
    return out


def axial_histogram(bridges, edges) -> np.ndarray:
    """Bridge counts along the molecule axis in the given z bins."""
    return np.histogram([b.z for b in bridges], bins=edges)[0]


class BridgeTracker:
    """Lifetimes of bridges in Monte Carlo sweeps.

    A bridge is identified by (ion uid, mol_a, mol_b). Call ``update`` at a
    fixed sampling interval. A bridge seen in m consecutive samples has lived
    m * interval sweeps; that is a resolution-limited lower bound on a quantity
    that is itself a property of the Markov chain's move set, not a physical
    time. Bridges still alive at ``finish`` are reported separately as
    censored.
    """

    def __init__(self, interval: int = 1, by_kind: bool = False):
        self.interval = int(interval)
        self.by_kind = by_kind
        self._alive: dict = {}       # key -> (samples seen, species, kind at birth)
        self.completed: list = []    # (species, kind, lifetime in sweeps)
        self.censored: list = []

    def _key(self, b: Bridge):
        return (b.uid, b.mol_a, b.mol_b, b.kind) if self.by_kind else (b.uid, b.mol_a, b.mol_b)

    def update(self, bridges) -> None:
        now = {}
        for b in bridges:
            key = self._key(b)
            seen, species, kind = self._alive.get(key, (0, b.species, b.kind))
            now[key] = (seen + 1, species, kind)
        for key, (seen, species, kind) in self._alive.items():
            if key not in now:
                self.completed.append((species, kind, seen * self.interval))
        self._alive = now

    def finish(self) -> None:
        for seen, species, kind in self._alive.values():
            self.censored.append((species, kind, seen * self.interval))
        self._alive = {}

    def lifetimes(self, species: str | None = None, include_censored: bool = False) -> np.ndarray:
        rows = self.completed + (self.censored if include_censored else [])
        return np.array([t for s, _, t in rows if species is None or s == species], dtype=float)
