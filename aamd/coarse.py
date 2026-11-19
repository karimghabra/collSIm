"""The coarse model's prediction for a segment pair, at the geometry the
all-atom runs use.

A *segment* is ``n_res`` consecutive helix residues of each of the three
chains (helix numbering, 0 = first Gly), cut from the modcoll site array of
the native molecule. A *pair* places segment A on the z axis and segment B
parallel to it at axis separation ``d`` along +x. Segment B starts ``stagger``
residues before segment A, so that the residues facing each other are those
of two whole molecules staggered by ``stagger`` residues (B's window = A's
window shifted by -stagger, the convention of tools/pmf_harness.py and of
modcoll.electrostatics.pair_energy_direct, where a site of B at s sits at
s + stagger in A's frame).

The energy is the modcoll site model, unchanged:

    E = sum_ij q_i q_j l_B exp(-r_eff/lambda_D)/r_eff  -  eps_h sum_ij h_i h_j exp(-r_ij^2 / 2 sigma^2)

with modcoll's kernels, soft core, Kyte-Doolittle centring (over the whole
molecule) and eps_h. It is evaluated by a direct double sum in Cartesian
coordinates, so that it can be given the exact azimuths an all-atom frame
has, and it is tested against modcoll.electrostatics.pair_energy_direct.

Charges are either the Henderson-Hasselbalch mean charges of modcoll or the
discrete protonation states given to the all-atom system (``Protonation``).

Units: nm, kT at 298 K, rad.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from modcoll import electrostatics as es
from modcoll.sites import Molecule, SiteType, native_molecule


ACIDS = (SiteType.ASP, SiteType.GLU, SiteType.SUCC_LYS, SiteType.CTERM)
BASES = (SiteType.LYS, SiteType.ARG, SiteType.HIS, SiteType.NTERM)


@dataclasses.dataclass(frozen=True)
class Segment:
    """Sites of one segment: helix residues [start, start + n_res) of each chain."""

    start: int
    n_res: int
    chain: np.ndarray
    res_index: np.ndarray
    aa: np.ndarray
    site_type: np.ndarray
    s: np.ndarray        # nm, along the molecule (not re-centred)
    phi: np.ndarray      # rad, template azimuth
    r: np.ndarray        # nm
    h: np.ndarray        # centred Kyte-Doolittle weight (whole-molecule mean)

    def __len__(self):
        return len(self.s)

    def with_types(self, site_type) -> "Segment":
        return dataclasses.replace(self, site_type=np.asarray(site_type, np.int8))


def segment(start: int, n_res: int = 30, mol: Molecule | None = None,
            position: str = "centroid") -> Segment:
    """Helix residues [start, start + n_res) of all three chains."""
    mol = native_molecule() if mol is None else mol
    h_all = mol.centered_hydrophobicity()
    keep = (mol.is_residue & ~mol.smeared & (mol.res_index >= start)
            & (mol.res_index < start + n_res))
    if keep.sum() != 3 * n_res:
        raise ValueError(f"window {start}+{n_res} is not inside the helix")
    s, phi, r = mol.positions(position)
    idx = np.flatnonzero(keep)
    return Segment(start, n_res, mol.chain[idx].copy(), mol.res_index[idx].copy(),
                   mol.aa[idx].copy(), mol.native_type[idx].copy(), s[idx].copy(),
                   phi[idx].copy(), r[idx].copy(), h_all[idx].copy())


def succinylate(seg: Segment) -> Segment:
    """Every lysine (and hydroxylysine) succinylated."""
    st = seg.site_type.copy()
    st[st == int(SiteType.LYS)] = int(SiteType.SUCC_LYS)
    return seg.with_types(st)


def acetylate(seg: Segment) -> Segment:
    st = seg.site_type.copy()
    st[st == int(SiteType.LYS)] = int(SiteType.ACET_LYS)
    return seg.with_types(st)


# ------------------------------------------------------------- protonation ---
def mean_charges(seg: Segment, pH: float, pka_set: str = "recommended") -> np.ndarray:
    """Henderson-Hasselbalch mean charge per site (modcoll's charge model)."""
    table = es.PKA_SETS[pka_set]
    q = np.zeros(len(seg))
    for st, pk in table.items():
        m = seg.site_type == int(st)
        if m.any():
            z = es.VALENCE[st]
            q[m] = z / (1.0 + 10.0 ** (z * (pH - pk)))
    return q


def discrete_charges(seg: Segment, pH: float, pka_set: str = "recommended",
                     rng: np.random.Generator | None = None, offset: float = 0.0) -> np.ndarray:
    """Integer charges whose count per site type matches the
    Henderson-Hasselbalch expectation, rounded to the nearest integer.

    Which sites of a type carry the charge is a draw from ``rng`` (default
    seed 0), so that the pattern is reproducible. ``offset`` shifts the
    fractional part before rounding (used to split a pair's total between
    two segments without bias: the second segment gets offset 0.5 - frac).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    table = es.PKA_SETS[pka_set]
    q = np.zeros(len(seg))
    for st in sorted(table, key=int):
        idx = np.flatnonzero(seg.site_type == int(st))
        if idx.size == 0:
            continue
        z = es.VALENCE[st]
        frac = 1.0 / (1.0 + 10.0 ** (z * (pH - table[st])))
        n_on = int(np.floor(frac * idx.size + 0.5 + offset))
        n_on = min(max(n_on, 0), idx.size)
        on = rng.permutation(idx)[:n_on]
        q[on] = z
    return q


# ---------------------------------------------------------------- geometry ---
def cartesian(seg: Segment, rotation: float = 0.0, axis_xy=(0.0, 0.0),
              z_shift: float = 0.0) -> np.ndarray:
    """(n, 3) site positions: the segment rotated by ``rotation`` about its
    own axis, the axis moved to ``axis_xy``, translated by ``z_shift``."""
    a = seg.phi + rotation
    return np.column_stack([axis_xy[0] + seg.r * np.cos(a),
                            axis_xy[1] + seg.r * np.sin(a), seg.s + z_shift])


def pair_positions(seg_a: Segment, seg_b: Segment, stagger_nm: float, d_nm: float,
                   rot_a: float = 0.0, rot_b: float = 0.0):
    """Site positions of A on the z axis and of B at (d, 0), B's sites moved
    by +stagger along z (so the residues facing each other are those of two
    whole molecules at that stagger)."""
    return (cartesian(seg_a, rot_a),
            cartesian(seg_b, rot_b, axis_xy=(d_nm, 0.0), z_shift=stagger_nm))


@dataclasses.dataclass(frozen=True)
class CoarseParams:
    ionic_strength_M: float = 0.15
    eps_h: float = es.EPS_HYDROPHOBIC
    sigma_h: float = es.HYDROPHOBIC_SIGMA_NM
    core_nm: float = es.SOFT_CORE_NM

    @property
    def debye_nm(self) -> float:
        return es.debye_length_nm(self.ionic_strength_M)


def pair_terms(xa, qa, ha, xb, qb, hb, params: CoarseParams = CoarseParams()):
    """(electrostatic, hydrophobic) energy in kT between two site sets."""
    d = np.linalg.norm(xa[:, None, :] - xb[None, :, :], axis=-1)
    el = float(qa @ es.screened_coulomb(d, params.debye_nm, params.core_nm) @ qb)
    hy = float(-params.eps_h * (ha @ es.hydrophobic_contact(d, params.sigma_h) @ hb))
    return el, hy


def energy_grid(seg_a: Segment, qa, seg_b: Segment, qb, stagger_nm: float, d_values,
                n_angle: int = 24, params: CoarseParams = CoarseParams()):
    """E_el and E_hy [n_d, n_angle, n_angle] over rotations of A and B."""
    angles = 2.0 * np.pi * np.arange(n_angle) / n_angle
    d_values = np.atleast_1d(d_values)
    el = np.zeros((len(d_values), n_angle, n_angle))
    hy = np.zeros_like(el)
    xa_all = [cartesian(seg_a, a) for a in angles]
    for k, d in enumerate(d_values):
        xb_all = [cartesian(seg_b, b, axis_xy=(d, 0.0), z_shift=stagger_nm) for b in angles]
        for i, xa in enumerate(xa_all):
            for j, xb in enumerate(xb_all):
                el[k, i, j], hy[k, i, j] = pair_terms(xa, qa, seg_a.h, xb, qb, seg_b.h, params)
    return el, hy


def reduce_boltzmann(e):
    """-ln <exp(-E)> over the last two axes (uniform prior on rotations)."""
    flat = e.reshape(e.shape[0], -1)
    m = flat.min(axis=1)
    return m - np.log(np.mean(np.exp(-(flat - m[:, None])), axis=1))


def reduce_min(e):
    return e.reshape(e.shape[0], -1).min(axis=1)
