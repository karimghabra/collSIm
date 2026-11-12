"""Metropolis Monte Carlo for explicit ions around fixed ligand atoms.

Ensemble
--------
Ligand atoms are fixed. Ions are rigid bodies (centre r, orientation q) and
each cation carries one two-valued state per coordination site. Ion numbers
fluctuate against a bath (grand canonical). The stationary distribution is

    P(N, {r, q, s}) ~ prod_species zeta_s^N_s / N_s!
                      * exp(-U({r, q, s}))  d^N r  d^N q / (8 pi^2)^N

with U in kT from ``energy.py``.

Activity convention
-------------------
``bath[species]`` is the concentration c (mol/L) of the *free* ion in a
reservoir whose only other content is the implicit monovalent electrolyte, and
in which the free ion's energy is the zero of U. The number density of free
ions there is rho = c * 0.602214 /nm^3 = zeta * z_free, where
z_free = (1 + exp(-dG_des))^n is the free cation's internal partition function
(1 for sulfate). Hence zeta = rho / z_free. The bath concentration is an
activity on the ideal-dilute scale with the monovalent Debye background as the
reference state; it is not the total analytical concentration once ion pairs or
bound ions exist, and no activity coefficient for divalent-divalent
interactions in the bath is applied.

Moves (each satisfies detailed balance on its own)
--------------------------------------------------
translate  centre displaced uniformly in a cube (symmetric); the cube edge
           is the full step or a fifth of it, chosen at random.
rotate     orientation composed with a rotation about a uniformly random axis
           by an angle uniform in [-a, a] (symmetric), a being the full step
           or a fifth of it; with a small probability a fresh Haar-uniform
           orientation instead.
flip       one site, chosen uniformly, changes state W <-> L; half the time
           together with a displacement and rotation drawn as in translate
           and rotate (symmetric either way).
dock       for a site k in state W: choose a ligand atom j uniformly among the
           C dockable atoms within R_nb of the centre; draw a Haar-uniform
           orientation and a point xi uniform in a ball of radius rho; put site
           k on x_j + xi and set its state to L.
undock     for a site k in state L that has M >= 1 dockable atoms within rho
           of the site position: choose one of them, j; draw a Haar-uniform
           orientation and a centre uniform in the ball of radius R_nb about
           x_j; set the state to W.
           The pair is balanced by
             A(dock)   = min(1, exp(-dU) * C * v_rho / (M' * V_nb))
             A(undock) = min(1, exp(-dU) * M * V_nb / (C' * v_rho))
           where primes are counted in the proposed configuration, v_rho and
           V_nb are the two ball volumes, and the 1/(8 pi^2) orientation
           densities cancel. (Derivation in docs/modified_collagen_ions.md.)
insert     a new ion, uniform in the box, Haar orientation, all sites W:
             A = min(1, zeta V / (N_s + 1) * exp(-dU))
delete     an ion of the species chosen uniformly; refused unless all its
           sites are W (the reverse of insert):
             A = min(1, N_s / (zeta V) * exp(-dU))

All randomness comes from the numpy Generator handed to the sampler.
"""
from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np

from . import geometry as geo
from ._jit import njit
from .energy import (KIND_CATION, L_STATE, MAX_SITES, W_STATE, Elec, Tables, e_internal,
                     e_ion_atom, e_ion_ion, rotate_dirs, wca)
from .params import PER_NM3_PER_MOLAR, IonParams

MOVE_NAMES = ("translate", "rotate", "flip", "dock", "undock", "insert", "delete")
M_TRANSLATE, M_ROTATE, M_FLIP, M_DOCK, M_UNDOCK, M_INSERT, M_DELETE = range(7)

# Oxygens and nitrogens of different chemical groups do not come closer than
# about 0.28 nm (two contact radii). The site-occupancy rule of the inner wells
# assumes it; below this value the sampler warns.
MIN_GROUP_SEPARATION = 0.25

# margin added to the Coulomb cutoff so that satellite points (at most 0.15 nm
# from a centre, on each of two bodies) are always inside the neighbour search
_SAT_MARGIN = 0.35


class System(NamedTuple):
    """Static part of a simulation: box, excluded bodies, fixed atoms in cells."""

    box: np.ndarray       # (3,) nm
    periodic: np.ndarray  # (3,) bool
    cyl: np.ndarray       # (M, 3): x, y, radius of z-parallel excluded cylinders
    apos: np.ndarray      # (A, 3) fixed atoms, sorted by cell
    atype: np.ndarray     # (A,)
    aq: np.ndarray        # (A,)
    anorm: np.ndarray     # (A, 3)
    ncell: np.ndarray     # (3,) cells per axis
    csize: np.ndarray     # (3,) cell edge, nm
    cstart: np.ndarray    # (ncells,) first atom of each cell
    ccount: np.ndarray    # (ncells,) atoms in each cell
    reach: float          # neighbour-search radius for ion-atom and ion-ion, nm


class Ions(NamedTuple):
    """Mutable ion arrays; the first n rows are live."""

    sp: np.ndarray     # (cap,) species index
    pos: np.ndarray    # (cap, 3)
    quat: np.ndarray   # (cap, 4)
    U: np.ndarray      # (cap, MAX_SITES, 3) lab-frame site unit vectors
    state: np.ndarray  # (cap, MAX_SITES) W_STATE / L_STATE
    nw: np.ndarray     # (cap,) number of W sites
    uid: np.ndarray    # (cap,) persistent identity


def build_system(ligands: geo.LigandSet, box, periodic, cylinders, reach: float):
    """Sort the fixed atoms into cells of edge >= reach. Returns the System and
    the permutation ``order`` with sorted_atom[i] = original_atom[order[i]]."""
    box = np.ascontiguousarray(np.asarray(box, dtype=float))
    periodic = np.ascontiguousarray(np.asarray(periodic, dtype=np.bool_))
    if math.isinf(reach):
        ncell = np.ones(3, dtype=np.int64)
    else:
        ncell = np.maximum(1, np.floor(box / reach)).astype(np.int64)
    csize = box / ncell
    pos = ligands.pos
    idx = np.clip(np.floor(pos / csize).astype(np.int64), 0, ncell - 1) if len(pos) else np.zeros((0, 3), np.int64)
    cid = (idx[:, 0] * ncell[1] + idx[:, 1]) * ncell[2] + idx[:, 2]
    order = np.argsort(cid, kind="stable")
    ncells = int(ncell.prod())
    ccount = np.bincount(cid, minlength=ncells).astype(np.int64)
    cstart = np.concatenate([[0], np.cumsum(ccount)[:-1]]).astype(np.int64)
    cyl = np.zeros((0, 3)) if cylinders is None else np.asarray(cylinders, dtype=float).reshape(-1, 3)
    sysm = System(
        box=box, periodic=periodic, cyl=np.ascontiguousarray(cyl),
        apos=np.ascontiguousarray(pos[order]), atype=np.ascontiguousarray(ligands.type[order]),
        aq=np.ascontiguousarray(ligands.charge[order]), anorm=np.ascontiguousarray(ligands.normal[order]),
        ncell=ncell, csize=csize, cstart=cstart, ccount=ccount, reach=float(reach),
    )
    return sysm, order


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------
@njit
def _mi(S, a, d):
    """Minimum-image component along axis a."""
    if S.periodic[a]:
        L = S.box[a]
        d -= L * math.floor(d / L + 0.5)
    return d


@njit
def _wrap(S, r):
    """Wrap r into the box on periodic axes; False if it lies outside a wall."""
    for a in range(3):
        L = S.box[a]
        if S.periodic[a]:
            r[a] -= L * math.floor(r[a] / L)
        elif r[a] < 0.0 or r[a] > L:
            return False
    return True


@njit
def _cell_range(S, a, x):
    """Inclusive-exclusive range of cell offsets to visit along axis a and the
    home cell index. With fewer than three cells every cell is visited once."""
    n = S.ncell[a]
    c = int(math.floor(x / S.csize[a]))
    if c < 0:
        c = 0
    if c >= n:
        c = n - 1
    if n < 3:
        return 0, n, c
    return c - 1, c + 2, c


@njit
def _cell_index(S, a, j):
    """Map a raw cell coordinate to a valid one, or -1 if outside a wall."""
    n = S.ncell[a]
    if n < 3:
        return j
    if j < 0 or j >= n:
        if S.periodic[a]:
            return (j + n) % n
        return -1
    return j


@njit
def e_fixed(T, E, S, sp, r, U, state, nw):
    """Energy of one ion with the excluded cylinders and all fixed atoms."""
    e = 0.0
    for c in range(S.cyl.shape[0]):
        dx = _mi(S, 0, r[0] - S.cyl[c, 0])
        dy = _mi(S, 1, r[1] - S.cyl[c, 1])
        # WCA in the distance from the cylinder *surface*, so the wall is as
        # steep as an atomic contact whatever the cylinder radius
        gap = math.sqrt(dx * dx + dy * dy) - S.cyl[c, 2]
        if gap < 0.0:
            gap = 0.0
        e += wca(gap * gap, T.R_wall[sp], T.eps_wca)
    reach2 = S.reach * S.reach
    lo0, hi0, _ = _cell_range(S, 0, r[0])
    lo1, hi1, _ = _cell_range(S, 1, r[1])
    lo2, hi2, _ = _cell_range(S, 2, r[2])
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
                    dx = _mi(S, 0, S.apos[a, 0] - r[0])
                    dy = _mi(S, 1, S.apos[a, 1] - r[1])
                    dz = _mi(S, 2, S.apos[a, 2] - r[2])
                    if dx * dx + dy * dy + dz * dz < reach2:
                        e += e_ion_atom(T, E, sp, U, state, nw, dx, dy, dz, S.atype[a], S.aq[a],
                                        S.anorm[a, 0], S.anorm[a, 1], S.anorm[a, 2])
    return e


@njit
def e_others(T, E, S, I, n, skip, sp, r, U, state, nw):
    """Energy of one ion with every other live ion except index ``skip``."""
    e = 0.0
    reach2 = S.reach * S.reach
    for j in range(n):
        if j == skip:
            continue
        dx = _mi(S, 0, I.pos[j, 0] - r[0])
        dy = _mi(S, 1, I.pos[j, 1] - r[1])
        dz = _mi(S, 2, I.pos[j, 2] - r[2])
        if dx * dx + dy * dy + dz * dz < reach2:
            e += e_ion_ion(T, E, sp, U, state, nw, I.sp[j], I.U[j], I.state[j], I.nw[j], dx, dy, dz)
    return e


@njit
def e_one(T, E, S, I, n, skip, sp, r, U, state, nw):
    """Everything that depends on one ion: internal, fixed atoms, other ions."""
    return e_internal(T, sp, state) + e_fixed(T, E, S, sp, r, U, state, nw) \
        + e_others(T, E, S, I, n, skip, sp, r, U, state, nw)


@njit
def total_energy(T, E, S, I, n, interacting):
    """Total energy by direct summation over ions and ion pairs."""
    e = 0.0
    for i in range(n):
        e += e_internal(T, I.sp[i], I.state[i])
        e += e_fixed(T, E, S, I.sp[i], I.pos[i], I.U[i], I.state[i], I.nw[i])
        if not interacting:
            continue
        for j in range(i + 1, n):
            dx = _mi(S, 0, I.pos[j, 0] - I.pos[i, 0])
            dy = _mi(S, 1, I.pos[j, 1] - I.pos[i, 1])
            dz = _mi(S, 2, I.pos[j, 2] - I.pos[i, 2])
            if dx * dx + dy * dy + dz * dz < S.reach * S.reach:
                e += e_ion_ion(T, E, I.sp[i], I.U[i], I.state[i], I.nw[i],
                               I.sp[j], I.U[j], I.state[j], I.nw[j], dx, dy, dz)
    return e


@njit
def _scan_dockable(T, S, I, n, skip, sp, cx, cy, cz, radius, target, out):
    """Count the atoms the cation species ``sp`` could take into a site (fixed
    atoms and oxygens of other ions with a non-zero inner contact energy)
    within ``radius`` of the point (cx, cy, cz). If ``target`` >= 0, also write
    the minimum-image position of the target-th such atom into ``out``."""
    count = 0
    r2max = radius * radius
    lo0, hi0, _ = _cell_range(S, 0, cx)
    lo1, hi1, _ = _cell_range(S, 1, cy)
    lo2, hi2, _ = _cell_range(S, 2, cz)
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
                    if T.eps_in[sp, S.atype[a]] == 0.0:
                        continue
                    dx = _mi(S, 0, S.apos[a, 0] - cx)
                    dy = _mi(S, 1, S.apos[a, 1] - cy)
                    dz = _mi(S, 2, S.apos[a, 2] - cz)
                    if dx * dx + dy * dy + dz * dz < r2max:
                        if count == target:
                            out[0] = cx + dx
                            out[1] = cy + dy
                            out[2] = cz + dz
                        count += 1
    for j in range(n):
        if j == skip:
            continue
        spj = I.sp[j]
        if T.kind[spj] == KIND_CATION or T.eps_in[sp, T.t_sat[spj]] == 0.0:
            continue
        d = T.d_site[spj]
        bx = _mi(S, 0, I.pos[j, 0] - cx)
        by = _mi(S, 1, I.pos[j, 1] - cy)
        bz = _mi(S, 2, I.pos[j, 2] - cz)
        for k in range(T.nsite[spj]):
            dx = bx + d * I.U[j, k, 0]
            dy = by + d * I.U[j, k, 1]
            dz = bz + d * I.U[j, k, 2]
            if dx * dx + dy * dy + dz * dz < r2max:
                if count == target:
                    out[0] = cx + dx
                    out[1] = cy + dy
                    out[2] = cz + dz
                count += 1
    return count


@njit
def _haar(rng, q):
    """Haar-uniform unit quaternion: normalised vector of four normals."""
    s = 0.0
    for a in range(4):
        q[a] = rng.normal()
        s += q[a] * q[a]
    s = math.sqrt(s)
    for a in range(4):
        q[a] /= s


@njit
def _ball(rng, radius, out):
    """Point uniform in a ball: isotropic direction, radius R u^(1/3)."""
    s = 0.0
    for a in range(3):
        out[a] = rng.normal()
        s += out[a] * out[a]
    f = radius * rng.random() ** (1.0 / 3.0) / math.sqrt(s)
    for a in range(3):
        out[a] *= f


@njit
def _small_rotation(rng, q_old, amax, q_new):
    """q_new = dq * q_old with dq a rotation by an angle uniform in
    [-amax, amax] about an isotropic axis. The proposal density is symmetric:
    the inverse rotation is drawn with the same probability."""
    ax = rng.normal()
    ay = rng.normal()
    az = rng.normal()
    nrm = math.sqrt(ax * ax + ay * ay + az * az)
    ang = (2.0 * rng.random() - 1.0) * amax
    s = math.sin(0.5 * ang) / nrm
    w = math.cos(0.5 * ang)
    x = ax * s
    y = ay * s
    z = az * s
    q_new[0] = w * q_old[0] - x * q_old[1] - y * q_old[2] - z * q_old[3]
    q_new[1] = w * q_old[1] + x * q_old[0] + y * q_old[3] - z * q_old[2]
    q_new[2] = w * q_old[2] - x * q_old[3] + y * q_old[0] + z * q_old[1]
    q_new[3] = w * q_old[3] + x * q_old[2] - y * q_old[1] + z * q_old[0]
    nq = math.sqrt(q_new[0] ** 2 + q_new[1] ** 2 + q_new[2] ** 2 + q_new[3] ** 2)
    for a in range(4):
        q_new[a] /= nq


@njit
def _commit(I, i, r, q, U, st, nw, nsite):
    for a in range(3):
        I.pos[i, a] = r[a]
    for a in range(4):
        I.quat[i, a] = q[a]
    for k in range(nsite):
        I.state[i, k] = st[k]
        for a in range(3):
            I.U[i, k, a] = U[k, a]
    I.nw[i] = nw


@njit
def _accept(rng, log_ratio):
    """Metropolis test on the log of the acceptance ratio."""
    if log_ratio >= 0.0:
        return True
    if log_ratio < -700.0:
        return False
    return rng.random() < math.exp(log_ratio)


@njit
def sweep_kernel(T, E, S, I, counters, zeta, moves, steps, dock_rho, stats, rng, n_sweeps, n_gc):
    """Run n_sweeps sweeps. One sweep is max(N, 1) local moves on randomly
    chosen ions followed by n_gc insertion/deletion attempts.

    counters  [n_live, next_uid], updated in place
    zeta      (S,) activity per species in 1/nm^3 (0 = species not exchanged)
    moves     cumulative probabilities of translate, rotate, flip, dock
    steps     [d_translate nm, a_rotate rad, p_full_rotation, R_nb nm, unused,
               1 if ions interact with one another else 0]
    dock_rho  (S,) radius rho of the dock ball per species, nm
    stats     (7, 2) attempts and acceptances per move type, updated in place
    Returns the number of insertions refused for lack of array capacity.
    """
    n = counters[0]
    cap = I.sp.shape[0]
    vol = S.box[0] * S.box[1] * S.box[2]
    r_new = np.empty(3)
    q_new = np.empty(4)
    U_new = np.zeros((MAX_SITES, 3))
    st_new = np.zeros(MAX_SITES, dtype=np.uint8)
    xi = np.empty(3)
    xj = np.empty(3)
    d_tr = steps[0]
    a_rot = steps[1]
    p_full = steps[2]
    R_nb = steps[3]
    interacting = steps[5] > 0.5
    V_nb = 4.0 / 3.0 * math.pi * R_nb ** 3
    n_gc_species = 0
    for s in range(zeta.shape[0]):
        if zeta[s] > 0.0:
            n_gc_species += 1
    overflow = 0

    for _ in range(n_sweeps):
        n_local = n if n > 0 else 0
        nn = n if interacting else 0  # ions seen by one another
        for _m in range(n_local):
            i = int(rng.integers(0, n))
            sp = I.sp[i]
            ns = T.nsite[sp]
            u = rng.random()
            if u < moves[0]:
                mv = M_TRANSLATE
            elif u < moves[1]:
                mv = M_ROTATE
            elif u < moves[2]:
                mv = M_FLIP
            else:
                mv = M_DOCK
            if T.kind[sp] != KIND_CATION and mv >= M_FLIP:
                mv = M_ROTATE  # ions without site states only move
            for a in range(3):
                r_new[a] = I.pos[i, a]
            for a in range(4):
                q_new[a] = I.quat[i, a]
            for k in range(ns):
                st_new[k] = I.state[i, k]
            nw_new = I.nw[i]
            log_bias = 0.0
            ok = True

            if mv == M_TRANSLATE:
                # half the attempts use a fifth of the step: an ion in a narrow
                # contact well needs small moves, one in the bulk large ones.
                # The choice does not depend on the state, so it stays symmetric.
                scale = 1.0 if rng.random() < 0.5 else 0.2
                for a in range(3):
                    r_new[a] += (2.0 * rng.random() - 1.0) * d_tr * scale
                ok = _wrap(S, r_new)
            elif mv == M_ROTATE:
                u_rot = rng.random()
                if u_rot < p_full:
                    _haar(rng, q_new)
                elif u_rot < 0.5 * (1.0 + p_full):
                    _small_rotation(rng, I.quat[i], a_rot, q_new)
                else:
                    _small_rotation(rng, I.quat[i], 0.2 * a_rot, q_new)
            elif mv == M_FLIP:
                k = int(rng.integers(0, ns))
                if st_new[k] == W_STATE:
                    st_new[k] = L_STATE
                    nw_new -= 1
                else:
                    st_new[k] = W_STATE
                    nw_new += 1
                if rng.random() < 0.5:
                    # half the flips come with a rigid displacement and rotation
                    # drawn as in the translate and rotate moves. The combined
                    # proposal is still symmetric. It lets an ion that already
                    # holds one ligand swing a second site onto another while
                    # that site opens, which no other move can do in one step.
                    for a in range(3):
                        r_new[a] += (2.0 * rng.random() - 1.0) * d_tr
                    _small_rotation(rng, I.quat[i], a_rot, q_new)
                    ok = _wrap(S, r_new)
            else:
                k = int(rng.integers(0, ns))
                d = T.d_site[sp]
                rho = dock_rho[sp]
                v_rho = 4.0 / 3.0 * math.pi * rho ** 3
                if st_new[k] == W_STATE:
                    mv = M_DOCK
                    C = _scan_dockable(T, S, I, nn, i, sp, r_new[0], r_new[1], r_new[2], R_nb, -1, xj)
                    if C == 0:
                        ok = False
                    else:
                        pick = int(rng.integers(0, C))
                        _scan_dockable(T, S, I, nn, i, sp, r_new[0], r_new[1], r_new[2], R_nb, pick, xj)
                        _haar(rng, q_new)
                        rotate_dirs(T, sp, q_new, U_new)
                        _ball(rng, rho, xi)
                        for a in range(3):
                            xj[a] += xi[a]          # the new site position
                            r_new[a] = xj[a] - d * U_new[k, a]
                        M = _scan_dockable(T, S, I, nn, i, sp, xj[0], xj[1], xj[2], rho, -1, xi)
                        st_new[k] = L_STATE
                        nw_new -= 1
                        log_bias = math.log(C * v_rho / (M * V_nb))
                        ok = _wrap(S, r_new)
                else:
                    mv = M_UNDOCK
                    px = r_new[0] + d * I.U[i, k, 0]
                    py = r_new[1] + d * I.U[i, k, 1]
                    pz = r_new[2] + d * I.U[i, k, 2]
                    M = _scan_dockable(T, S, I, nn, i, sp, px, py, pz, rho, -1, xj)
                    if M == 0:
                        ok = False
                    else:
                        pick = int(rng.integers(0, M))
                        _scan_dockable(T, S, I, nn, i, sp, px, py, pz, rho, pick, xj)
                        _haar(rng, q_new)
                        _ball(rng, R_nb, xi)
                        for a in range(3):
                            r_new[a] = xj[a] + xi[a]
                        ok = _wrap(S, r_new)
                        if ok:
                            C = _scan_dockable(T, S, I, nn, i, sp, r_new[0], r_new[1], r_new[2], R_nb, -1, xi)
                            st_new[k] = W_STATE
                            nw_new += 1
                            # C >= 1: atom j is within R_nb of the new centre
                            log_bias = math.log(M * V_nb / (C * v_rho))

            stats[mv, 0] += 1
            if not ok:
                continue
            rotate_dirs(T, sp, q_new, U_new)
            e_old = e_one(T, E, S, I, nn, i, sp, I.pos[i], I.U[i], I.state[i], I.nw[i])
            e_new = e_one(T, E, S, I, nn, i, sp, r_new, U_new, st_new, nw_new)
            if _accept(rng, log_bias - (e_new - e_old)):
                _commit(I, i, r_new, q_new, U_new, st_new, nw_new, ns)
                stats[mv, 1] += 1

        for _g in range(n_gc):
            if n_gc_species == 0:
                break
            pick = int(rng.integers(0, n_gc_species))
            sp = -1
            for s in range(zeta.shape[0]):
                if zeta[s] > 0.0:
                    if pick == 0:
                        sp = s
                        break
                    pick -= 1
            ns = T.nsite[sp]
            nn = n if interacting else 0
            n_s = 0
            for j in range(n):
                if I.sp[j] == sp:
                    n_s += 1
            if rng.random() < 0.5:
                stats[M_INSERT, 0] += 1
                if n >= cap:
                    overflow += 1
                    continue
                for a in range(3):
                    r_new[a] = rng.random() * S.box[a]
                _haar(rng, q_new)
                rotate_dirs(T, sp, q_new, U_new)
                for k in range(ns):
                    st_new[k] = W_STATE
                de = e_one(T, E, S, I, nn, -1, sp, r_new, U_new, st_new, ns)
                if _accept(rng, math.log(zeta[sp] * vol / (n_s + 1)) - de):
                    I.sp[n] = sp
                    _commit(I, n, r_new, q_new, U_new, st_new, ns, ns)
                    I.uid[n] = counters[1]
                    counters[1] += 1
                    n += 1
                    stats[M_INSERT, 1] += 1
            else:
                stats[M_DELETE, 0] += 1
                if n_s == 0:
                    continue
                pick = int(rng.integers(0, n_s))
                i = -1
                for j in range(n):
                    if I.sp[j] == sp:
                        if pick == 0:
                            i = j
                            break
                        pick -= 1
                if I.nw[i] != ns:
                    continue  # only fully hydrated ions leave: reverse of insert
                e_i = e_one(T, E, S, I, nn, i, sp, I.pos[i], I.U[i], I.state[i], I.nw[i])
                if _accept(rng, math.log(n_s / (zeta[sp] * vol)) + e_i):
                    last = n - 1
                    if i != last:
                        I.sp[i] = I.sp[last]
                        I.uid[i] = I.uid[last]
                        _commit(I, i, I.pos[last], I.quat[last], I.U[last], I.state[last],
                                I.nw[last], T.nsite[I.sp[last]])
                    n -= 1
                    stats[M_DELETE, 1] += 1
    counters[0] = n
    return overflow


# --------------------------------------------------------------------------
# Python front end
# --------------------------------------------------------------------------
class Snapshot(NamedTuple):
    """Copy of the live ions at one instant."""

    species: np.ndarray  # (N,) index into IonParams.species
    pos: np.ndarray      # (N, 3) nm
    quat: np.ndarray     # (N, 4)
    U: np.ndarray        # (N, MAX_SITES, 3) lab-frame site unit vectors
    state: np.ndarray    # (N, MAX_SITES) 0 = water, 1 = ligand
    uid: np.ndarray      # (N,) persistent ion identity


class IonSampler:
    """Grand-canonical Monte Carlo of Mg2+, Ca2+ and SO4 2- around fixed atoms.

    params          IonParams (see ``load_params``)
    ligands         LigandSet of fixed atoms; positions must lie inside the box
    box             (Lx, Ly, Lz) in nm, origin at the corner
    periodic        three booleans; a non-periodic axis has hard walls for the
                    ion centres at 0 and L
    ionic_strength  mol/L of the implicit monovalent background (sets kappa)
    rng             numpy Generator; the only source of randomness
    bath            {species name: free-ion concentration in mol/L}; species
                    absent or at 0 are not exchanged with the bath
    cylinders       optional (M, 3) array of z-parallel excluded cylinders
                    (x, y, radius): the molecular cores
    interacting     False switches off every ion-ion term: the ions are then
                    an ideal gas in the field of the fixed atoms, the exact
                    infinite-dilution limit (bound-ion numbers are Poisson
                    with mean K c). For validation; production runs keep True.
    dock_radius, dock_ball
                    R_nb and rho of the dock/undock moves, nm. R_nb must cover
                    the outer-sphere shell (0.54 nm for Ca2+); rho defaults to
                    the lateral radius of each species' inner well. They
                    affect efficiency only.
    coulomb_cutoff  truncation of the screened Coulomb term, nm. Default
                    4 / kappa clipped to [1, 5] nm (the neglected tail of a
                    2:2 pair is then below 0.02 kT), reduced if necessary to
                    fit inside half the shortest periodic box edge.
    """

    def __init__(self, params: IonParams, ligands: geo.LigandSet, box, periodic,
                 ionic_strength: float, rng: np.random.Generator, bath: dict | None = None,
                 cylinders=None, coulomb_cutoff: float | None = None, capacity: int = 64,
                 move_probabilities=(0.35, 0.25, 0.10, 0.30), translate_step: float = 0.15,
                 rotate_step: float = 0.6, full_rotation_probability: float = 0.1,
                 dock_radius: float = 0.56, dock_ball: float | None = None,
                 gc_attempts_per_sweep: int | None = None, interacting: bool = True):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator (explicit seed required)")
        self.params = params
        self.ligands = ligands
        self.rng = rng
        self.T: Tables = params.tables()
        self.ionic_strength = float(ionic_strength)
        box = np.asarray(box, dtype=float)
        periodic = np.asarray(periodic, dtype=bool)
        kappa = params.kappa(ionic_strength)
        if coulomb_cutoff is None:
            coulomb_cutoff = float(np.clip(4.0 / kappa if kappa > 0 else np.inf, 1.0, 5.0))
            if periodic.any():  # minimum image needs the search sphere inside half a box
                coulomb_cutoff = min(coulomb_cutoff, 0.5 * float(box[periodic].min()) - _SAT_MARGIN)
        self.coulomb_cutoff = float(coulomb_cutoff)
        reach = self.coulomb_cutoff + _SAT_MARGIN
        if periodic.any() and reach > 0.5 * box[periodic].min():
            raise ValueError(
                f"Coulomb cutoff {self.coulomb_cutoff:.2f} nm (+{_SAT_MARGIN} nm margin) exceeds half "
                f"the shortest periodic box edge {box[periodic].min():.2f} nm")
        if reach < float(self.T.r_short.max()):
            raise ValueError("Coulomb cutoff is shorter than the range of the contact terms")
        if len(ligands) and ((ligands.pos < 0).any() or (ligands.pos > box).any()):
            raise ValueError("ligand atoms must lie inside the box")
        if len(ligands) > 1 and ligands.min_separation() < MIN_GROUP_SEPARATION:
            warnings.warn(
                f"ligand atoms of different groups come within {ligands.min_separation():.3f} nm of each "
                f"other (< {MIN_GROUP_SEPARATION} nm): two groups could then share one coordination "
                "site, which the pairwise contact terms would count twice", stacklevel=2)
        self.E: Elec = params.elec(ionic_strength, self.coulomb_cutoff)
        self.S, self.atom_order = build_system(ligands, box, periodic, cylinders, reach)
        self._alloc(capacity)
        self._counters = np.array([0, 0], dtype=np.int64)
        self.zeta = np.zeros(len(params.species))
        self.bath = dict(bath or {})
        for name, conc in self.bath.items():
            # zeta = rho_bath / z_free, see the module docstring
            self.zeta[params.species_index(name)] = conc * PER_NM3_PER_MOLAR / params.free_ion_partition(name)
        p = np.asarray(move_probabilities, dtype=float)
        if p.shape != (4,) or (p < 0).any() or p.sum() <= 0:
            raise ValueError("move_probabilities: four non-negative numbers (translate, rotate, flip, dock)")
        self._moves = np.cumsum(p / p.sum())
        self._steps = np.array([translate_step, rotate_step, full_rotation_probability,
                                dock_radius, 0.0, 1.0 if interacting else 0.0], dtype=float)
        if dock_ball is None:
            # the lateral radius of each species' inner well: large enough for the
            # ball around one oxygen to contain the site position of a calcium
            # holding both oxygens of a carboxylate (0.114 nm from each)
            sin_max = np.sqrt(np.maximum(0.0, 1.0 - self.T.cos_max ** 2))
            self._dock_rho = np.maximum(np.maximum(self.T.w_r, self.T.d_site * sin_max), 0.05)
        else:
            self._dock_rho = np.full(len(params.species), float(dock_ball))
        self.interacting = bool(interacting)
        self.stats = np.zeros((len(MOVE_NAMES), 2), dtype=np.int64)
        if gc_attempts_per_sweep is None:
            expected = sum(c * PER_NM3_PER_MOLAR for c in self.bath.values()) * float(box.prod())
            gc_attempts_per_sweep = int(max(4, round(expected))) if self.bath else 0
        self.gc_attempts_per_sweep = int(gc_attempts_per_sweep)
        self.sweeps_done = 0

    # -- storage -----------------------------------------------------------
    def _alloc(self, cap: int):
        old = getattr(self, "I", None)
        self.I = Ions(
            sp=np.zeros(cap, dtype=np.int64), pos=np.zeros((cap, 3)), quat=np.zeros((cap, 4)),
            U=np.zeros((cap, MAX_SITES, 3)), state=np.zeros((cap, MAX_SITES), dtype=np.uint8),
            nw=np.zeros(cap, dtype=np.int64), uid=np.zeros(cap, dtype=np.int64),
        )
        if old is not None:
            m = len(old.sp)
            for new_arr, old_arr in zip(self.I, old):
                new_arr[:m] = old_arr

    @property
    def n_ions(self) -> int:
        return int(self._counters[0])

    def count(self, species: str) -> int:
        return int((self.I.sp[: self.n_ions] == self.params.species_index(species)).sum())

    def add_ion(self, species: str, pos, quat=None, states=None) -> int:
        """Place one ion by hand. Returns its persistent uid."""
        n = self.n_ions
        if n >= len(self.I.sp):
            self._alloc(2 * len(self.I.sp))
        sp = self.params.species_index(species)
        ns = int(self.T.nsite[sp])
        q = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else np.asarray(quat, float)
        q = q / np.linalg.norm(q)
        st = np.zeros(MAX_SITES, dtype=np.uint8)
        if states is not None:
            if self.T.kind[sp] != KIND_CATION and np.any(states):
                raise ValueError("only cations have site states")
            st[:ns] = np.asarray(states, dtype=np.uint8)[:ns]
        self.I.sp[n] = sp
        self.I.pos[n] = np.asarray(pos, float)
        self.I.quat[n] = q
        rotate_dirs(self.T, sp, q, self.I.U[n])
        self.I.state[n] = st
        self.I.nw[n] = ns - int(st[:ns].sum())
        uid = int(self._counters[1])
        self.I.uid[n] = uid
        self._counters[0] = n + 1
        self._counters[1] = uid + 1
        return uid

    # -- running -----------------------------------------------------------
    def sweep(self, n_sweeps: int = 1) -> None:
        """Advance by n_sweeps sweeps (see ``sweep_kernel``)."""
        done = 0
        while done < n_sweeps:
            free = len(self.I.sp) - self.n_ions
            per = max(self.gc_attempts_per_sweep, 1)
            if free < per:
                self._alloc(2 * len(self.I.sp) + per)
                continue
            chunk = min(n_sweeps - done, free // per)
            overflow = sweep_kernel(self.T, self.E, self.S, self.I, self._counters, self.zeta,
                                    self._moves, self._steps, self._dock_rho, self.stats, self.rng, chunk,
                                    self.gc_attempts_per_sweep)
            assert overflow == 0, "ion array capacity was exceeded"
            done += chunk
        self.sweeps_done += n_sweeps

    def energy(self) -> float:
        """Total energy in kT by direct summation."""
        return float(total_energy(self.T, self.E, self.S, self.I, self.n_ions, self.interacting))

    def ion_energy(self, i: int) -> float:
        """Energy of live ion i with everything else, including its internal term."""
        return float(e_one(self.T, self.E, self.S, self.I, self.n_ions if self.interacting else 0, i,
                           self.I.sp[i], self.I.pos[i],
                           self.I.U[i], self.I.state[i], self.I.nw[i]))

    def snapshot(self) -> Snapshot:
        n = self.n_ions
        return Snapshot(self.I.sp[:n].copy(), self.I.pos[:n].copy(), self.I.quat[:n].copy(),
                        self.I.U[:n].copy(), self.I.state[:n].copy(), self.I.uid[:n].copy())

    def acceptance(self) -> dict:
        """Fraction of accepted attempts per move type (nan if never tried)."""
        out = {}
        for name, (att, acc) in zip(MOVE_NAMES, self.stats):
            out[name] = acc / att if att else float("nan")
        return out
