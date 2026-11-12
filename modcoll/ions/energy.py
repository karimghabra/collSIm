"""Energy function of the explicit-ion model.

Units: lengths nm, energies kT at 298.15 K, charges e.

Every term is a pair term between two rigid bodies (ion-atom or ion-ion) or a
one-body term, and each is smooth with compact support, so the same function
can later be differentiated for a dynamics engine.

The bodies
----------
cation (Mg2+, Ca2+): a centre and n coordination sites at distance d along
    body-fixed unit vectors. Each site k carries a state s_k: W (a water sits
    there) or L (the water has left; a ligand atom may sit there instead).
    Charge: q_centre on the centre and q_sat on n "dummy" points at radius
    r_sat along the same unit vectors (the Aqvist-Warshel layout).
oxyanion (SO4 2-): a centre (S) and n = 4 oxygens at distance d, charge q_sat
    on each oxygen. No internal state.
atom: a ligand atom of the protein (fixed) or an atom of another ion.

The terms, for a cation c and a foreign atom j at distance r from its centre
---------------------------------------------------------------------------
  u_C    screened Coulomb between every charge point of c and q_j
  u_core WCA(r; R_c + R_j)                       ion core against the atom
  per site k in state W:
         WCA(|x_j - p_k|; R_w + R_j)             the water occupies the site
  per site k in state L:
         - eps_in(c, t_j) * B((r - d) / w_r) * B_ang(theta_k)
                                                 inner-sphere contact well
  - (n_W / n) * eps_out(c, t_j) * B((r - d_out) / w_out)
                                                 outer-sphere (water-mediated)
  one-body: + dG_des(c) for every site in state L (the water is given up)

with p_k = r_c + d u_k the site position, theta_k the angle between u_k and
the centre->atom vector, B(x) = (1 - x^2)^2 for |x| < 1 and 0 outside, and

    B_ang(theta) = 1                 for theta <= theta_flat
                 = B(y)              for theta_flat < theta < theta_max
                 = 0                 beyond,
    y = (cos theta_flat - cos theta) / (cos theta_flat - cos theta_max).

theta_flat is the angular slack of the coordination shell: zero for the rigid
Mg2+ octahedron, several degrees for the loose Ca2+ shell. B_ang and its first
derivative are continuous at both ends.

For the sulfate, each oxygen a and a donor nitrogen j:
         - eps_hb(t_j) * B((|x_j - o_a| - d_hb) / w_hb) * P_j
where P_j = B(sin(out-of-plane angle) / sin_max) restricts the hydrogen bond
to the donor's plane when the donor has one (guanidinium, imidazolium).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ._jit import njit

W_STATE = 0  # site holds its water
L_STATE = 1  # site has given up its water

KIND_CATION = 0
KIND_OXYANION = 1

MAX_SITES = 8

# 2^(1/3): sigma^2 = r_min^2 / 2^(1/3) for the WCA potential
_CBRT2 = 2.0 ** (1.0 / 3.0)


class Tables(NamedTuple):
    """Numeric parameter tables, indexed by species (S) and ligand type (T)."""

    kind: np.ndarray          # (S,) KIND_CATION or KIND_OXYANION
    nsite: np.ndarray         # (S,) number of sites
    dirs: np.ndarray          # (S, MAX_SITES, 3) body-frame unit vectors
    d_site: np.ndarray        # (S,) centre-to-site distance, nm
    q_centre: np.ndarray      # (S,) charge on the centre, e
    q_sat: np.ndarray         # (S,) charge on each satellite point, e
    r_sat: np.ndarray         # (S,) radius of the satellite charge points, nm
    R_centre: np.ndarray      # (S,) contact radius of the centre, nm
    R_sat: np.ndarray         # (S,) contact radius of satellite atoms, nm (oxyanion O)
    R_water: np.ndarray       # (S,) contact radius of a site water, nm (cations)
    dg_des: np.ndarray        # (S,) free energy of emptying one site, kT
    w_r: np.ndarray           # (S,) radial half-width of the inner well, nm
    cos_flat: np.ndarray      # (S,) cos(theta_flat): inner well is flat inside this angle
    cos_max: np.ndarray       # (S,) cos(theta_max): inner well vanishes beyond this angle
    d_out: np.ndarray         # (S,) centre of the outer-sphere shell, nm
    w_out: np.ndarray         # (S,) half-width of the outer-sphere shell, nm
    eps_in: np.ndarray        # (S, T) inner-sphere contact energy, kT
    eps_out: np.ndarray       # (S, T) outer-sphere contact energy, kT
    eps_hb: np.ndarray        # (S, T) hydrogen-bond energy per O...N pair, kT
    d_hb: np.ndarray          # (S,) O...N hydrogen-bond distance, nm
    w_hb: np.ndarray          # (S,) half-width of the hydrogen-bond well, nm
    sin_plane_max: np.ndarray  # (S,) largest out-of-plane sine for a planar donor
    R_type: np.ndarray        # (T,) contact radius of each ligand atom type, nm
    t_centre: np.ndarray      # (S,) ligand type code of the centre atom
    t_sat: np.ndarray         # (S,) ligand type code of the satellite atoms
    R_wall: np.ndarray        # (S,) closest approach of the centre to a body surface, nm
    r_short: np.ndarray       # (S,) range of all non-Coulomb terms from the centre, nm
    partner: np.ndarray       # (S, T) bool: atom types that count as a contact partner
    r_contact: np.ndarray     # (S,) centre-to-atom distance below which a partner
                              #      atom is in (at least outer-sphere) contact, nm
    eps_wca: float            # WCA energy scale, kT


class Elec(NamedTuple):
    """Electrostatics of the implicit monovalent background."""

    lb: float     # Bjerrum length, nm
    kappa: float  # inverse Debye length, 1/nm
    rcut: float   # truncation distance, nm (inf for none)
    shift: float  # exp(-kappa rcut) / rcut, subtracted so u(rcut) = 0


def make_elec(lb: float, kappa: float, rcut: float = math.inf) -> Elec:
    shift = 0.0 if math.isinf(rcut) else math.exp(-kappa * rcut) / rcut
    return Elec(float(lb), float(kappa), float(rcut), float(shift))


# --------------------------------------------------------------------------
# Elementary functions
# --------------------------------------------------------------------------
@njit
def bump(x2):
    """B as a function of x^2: (1 - x^2)^2 inside |x| < 1, zero outside.
    Continuous with continuous first derivative at the edge."""
    if x2 >= 1.0:
        return 0.0
    y = 1.0 - x2
    return y * y


@njit
def wca(r2, rmin, eps):
    """Weeks-Chandler-Andersen repulsion as a function of r^2.

    u = 4 eps [(sigma/r)^12 - (sigma/r)^6] + eps for r < rmin = 2^(1/6) sigma,
    zero beyond (Weeks, Chandler & Andersen, J. Chem. Phys. 54 (1971) 5237).
    Parametrised by rmin, the distance at which the repulsion starts. r is
    floored at 0.3 rmin so a wild trial move returns a large finite number.
    """
    rm2 = rmin * rmin
    if r2 >= rm2:
        return 0.0
    if r2 < 0.09 * rm2:
        r2 = 0.09 * rm2
    x = rm2 / (_CBRT2 * r2)  # (sigma / r)^2
    x3 = x * x * x
    return 4.0 * eps * (x3 * x3 - x3) + eps


@njit
def yukawa(r2, qq, lb, kappa, rcut, shift):
    """Screened Coulomb energy of two point charges (Debye-Hueckel):

        u(r) = lb q1 q2 [exp(-kappa r) / r - exp(-kappa rcut) / rcut], r < rcut

    kT units; lb is the Bjerrum length e^2 / (4 pi eps0 eps_r kT). The shift
    makes u continuous at the cutoff. r is floored at 0.05 nm.
    """
    if qq == 0.0:
        return 0.0
    r = math.sqrt(r2)
    if r >= rcut:
        return 0.0
    if r < 0.05:
        r = 0.05
    return lb * qq * (math.exp(-kappa * r) / r - shift)


@njit
def rotate_dirs(T, sp, quat, out):
    """Write the lab-frame site unit vectors of species sp into out[:n]."""
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    n = T.nsite[sp]
    for k in range(n):
        a = T.dirs[sp, k, 0]
        b = T.dirs[sp, k, 1]
        c = T.dirs[sp, k, 2]
        out[k, 0] = r00 * a + r01 * b + r02 * c
        out[k, 1] = r10 * a + r11 * b + r12 * c
        out[k, 2] = r20 * a + r21 * b + r22 * c


@njit
def inner_weight(T, sp, U, k, dx, dy, dz, r):
    """Geometric weight in [0, 1] of an atom at (dx, dy, dz) from the centre
    for inner-sphere contact through site k: B_r * B_ang (module docstring)."""
    x = (r - T.d_site[sp]) / T.w_r[sp]
    x2 = x * x
    if x2 >= 1.0:
        return 0.0
    cos_t = (dx * U[k, 0] + dy * U[k, 1] + dz * U[k, 2]) / r
    if cos_t <= T.cos_max[sp]:
        return 0.0
    if cos_t >= T.cos_flat[sp]:
        return bump(x2)
    y = (T.cos_flat[sp] - cos_t) / (T.cos_flat[sp] - T.cos_max[sp])
    return bump(x2) * bump(y * y)


# --------------------------------------------------------------------------
# Pair energies
# --------------------------------------------------------------------------
@njit
def e_cation_atom(T, E, sp, U, state, n_w, dx, dy, dz, t, q):
    """Energy of cation ``sp`` (lab-frame site vectors U, site states ``state``
    of which n_w are W) with one foreign atom of type t and charge q located
    at (dx, dy, dz) from the cation centre."""
    n = T.nsite[sp]
    r2 = dx * dx + dy * dy + dz * dz
    e = 0.0
    if q != 0.0:
        e += yukawa(r2, T.q_centre[sp] * q, E.lb, E.kappa, E.rcut, E.shift)
        qs = T.q_sat[sp] * q
        if qs != 0.0:
            rs = T.r_sat[sp]
            for k in range(n):
                ax = dx - rs * U[k, 0]
                ay = dy - rs * U[k, 1]
                az = dz - rs * U[k, 2]
                e += yukawa(ax * ax + ay * ay + az * az, qs, E.lb, E.kappa, E.rcut, E.shift)
    rsh = T.r_short[sp]
    if r2 >= rsh * rsh:
        return e
    r = math.sqrt(r2)
    Rj = T.R_type[t]
    e += wca(r2, T.R_centre[sp] + Rj, T.eps_wca)
    d = T.d_site[sp]
    ein = T.eps_in[sp, t]
    rw = T.R_water[sp] + Rj
    for k in range(n):
        if state[k] == W_STATE:
            ax = dx - d * U[k, 0]
            ay = dy - d * U[k, 1]
            az = dz - d * U[k, 2]
            e += wca(ax * ax + ay * ay + az * az, rw, T.eps_wca)
        elif ein != 0.0 and r > 1e-9:
            e -= ein * inner_weight(T, sp, U, k, dx, dy, dz, r)
    eout = T.eps_out[sp, t]
    if eout != 0.0 and n_w > 0:
        x = (r - T.d_out[sp]) / T.w_out[sp]
        e -= eout * (n_w / n) * bump(x * x)
    return e


@njit
def e_oxyanion_atom(T, E, sp, U, dx, dy, dz, t, q, nx, ny, nz):
    """Energy of oxyanion ``sp`` (lab-frame oxygen unit vectors U) with one
    foreign atom of type t, charge q and plane normal (nx, ny, nz) located at
    (dx, dy, dz) from the central atom."""
    n = T.nsite[sp]
    r2 = dx * dx + dy * dy + dz * dz
    d = T.d_site[sp]
    e = 0.0
    if q != 0.0:
        e += yukawa(r2, T.q_centre[sp] * q, E.lb, E.kappa, E.rcut, E.shift)
        qs = T.q_sat[sp] * q
        for k in range(n):
            ax = dx - d * U[k, 0]
            ay = dy - d * U[k, 1]
            az = dz - d * U[k, 2]
            e += yukawa(ax * ax + ay * ay + az * az, qs, E.lb, E.kappa, E.rcut, E.shift)
    rsh = T.r_short[sp]
    if r2 >= rsh * rsh:
        return e
    Rj = T.R_type[t]
    e += wca(r2, T.R_centre[sp] + Rj, T.eps_wca)
    ehb = T.eps_hb[sp, t]
    planar = (nx * nx + ny * ny + nz * nz) > 0.25
    rmin_o = T.R_sat[sp] + Rj
    for k in range(n):
        # vector from the donor atom to this oxygen
        ax = d * U[k, 0] - dx
        ay = d * U[k, 1] - dy
        az = d * U[k, 2] - dz
        a2 = ax * ax + ay * ay + az * az
        e += wca(a2, rmin_o, T.eps_wca)
        if ehb != 0.0:
            a = math.sqrt(a2)
            x = (a - T.d_hb[sp]) / T.w_hb[sp]
            x2 = x * x
            if x2 < 1.0 and a > 1e-9:
                w = bump(x2)
                if planar:
                    s = (ax * nx + ay * ny + az * nz) / (a * T.sin_plane_max[sp])
                    w *= bump(s * s)
                e -= ehb * w
    return e


@njit
def _e_cation_oxyanion(T, E, sp_c, U_c, st_c, nw_c, sp_o, U_o, dx, dy, dz):
    """Cation against every atom of an oxyanion; (dx, dy, dz) points from the
    cation centre to the oxyanion centre. Each oxyanion atom is a foreign atom
    to the cation, so its oxygens can enter the cation's coordination sites."""
    e = e_cation_atom(T, E, sp_c, U_c, st_c, nw_c, dx, dy, dz, T.t_centre[sp_o], T.q_centre[sp_o])
    d = T.d_site[sp_o]
    for k in range(T.nsite[sp_o]):
        e += e_cation_atom(T, E, sp_c, U_c, st_c, nw_c, dx + d * U_o[k, 0], dy + d * U_o[k, 1],
                           dz + d * U_o[k, 2], T.t_sat[sp_o], T.q_sat[sp_o])
    return e


@njit
def e_ion_ion(T, E, sp_a, U_a, st_a, nw_a, sp_b, U_b, st_b, nw_b, dx, dy, dz):
    """Energy between two ions; (dx, dy, dz) points from centre a to centre b."""
    ka = T.kind[sp_a]
    kb = T.kind[sp_b]
    na = T.nsite[sp_a]
    nb = T.nsite[sp_b]
    e = 0.0
    if ka == KIND_CATION and kb == KIND_OXYANION:
        return _e_cation_oxyanion(T, E, sp_a, U_a, st_a, nw_a, sp_b, U_b, dx, dy, dz)
    if ka == KIND_OXYANION and kb == KIND_CATION:
        return _e_cation_oxyanion(T, E, sp_b, U_b, st_b, nw_b, sp_a, U_a, -dx, -dy, -dz)
    if ka == KIND_CATION:
        # cation-cation: charge points against charge points, core against core
        r2 = dx * dx + dy * dy + dz * dz
        e += wca(r2, T.R_centre[sp_a] + T.R_centre[sp_b], T.eps_wca)
        ra = T.r_sat[sp_a]
        rb = T.r_sat[sp_b]
        qa0 = T.q_centre[sp_a]
        qb0 = T.q_centre[sp_b]
        qa = T.q_sat[sp_a]
        qb = T.q_sat[sp_b]
        e += yukawa(r2, qa0 * qb0, E.lb, E.kappa, E.rcut, E.shift)
        for k in range(nb):
            bx = dx + rb * U_b[k, 0]
            by = dy + rb * U_b[k, 1]
            bz = dz + rb * U_b[k, 2]
            e += yukawa(bx * bx + by * by + bz * bz, qa0 * qb, E.lb, E.kappa, E.rcut, E.shift)
        for i in range(na):
            ax = dx - ra * U_a[i, 0]
            ay = dy - ra * U_a[i, 1]
            az = dz - ra * U_a[i, 2]
            e += yukawa(ax * ax + ay * ay + az * az, qa * qb0, E.lb, E.kappa, E.rcut, E.shift)
            for k in range(nb):
                bx = ax + rb * U_b[k, 0]
                by = ay + rb * U_b[k, 1]
                bz = az + rb * U_b[k, 2]
                e += yukawa(bx * bx + by * by + bz * bz, qa * qb, E.lb, E.kappa, E.rcut, E.shift)
        return e
    # oxyanion-oxyanion: all atom pairs, Coulomb and cores
    da = T.d_site[sp_a]
    db = T.d_site[sp_b]
    for i in range(-1, na):
        if i < 0:
            ax = 0.0
            ay = 0.0
            az = 0.0
            qa = T.q_centre[sp_a]
            Ra = T.R_centre[sp_a]
        else:
            ax = da * U_a[i, 0]
            ay = da * U_a[i, 1]
            az = da * U_a[i, 2]
            qa = T.q_sat[sp_a]
            Ra = T.R_sat[sp_a]
        for k in range(-1, nb):
            if k < 0:
                bx = dx
                by = dy
                bz = dz
                qb = T.q_centre[sp_b]
                Rb = T.R_centre[sp_b]
            else:
                bx = dx + db * U_b[k, 0]
                by = dy + db * U_b[k, 1]
                bz = dz + db * U_b[k, 2]
                qb = T.q_sat[sp_b]
                Rb = T.R_sat[sp_b]
            r2 = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
            e += yukawa(r2, qa * qb, E.lb, E.kappa, E.rcut, E.shift)
            e += wca(r2, Ra + Rb, T.eps_wca)
    return e


@njit
def e_ion_atom(T, E, sp, U, state, n_w, dx, dy, dz, t, q, nx, ny, nz):
    """Dispatch on the ion kind."""
    if T.kind[sp] == KIND_CATION:
        return e_cation_atom(T, E, sp, U, state, n_w, dx, dy, dz, t, q)
    return e_oxyanion_atom(T, E, sp, U, dx, dy, dz, t, q, nx, ny, nz)


@njit
def count_w(T, sp, state):
    n = T.nsite[sp]
    c = 0
    for k in range(n):
        if state[k] == W_STATE:
            c += 1
    return c


@njit
def e_internal(T, sp, state):
    """One-body energy: dG_des for every site that has given up its water."""
    return T.dg_des[sp] * (T.nsite[sp] - count_w(T, sp, state))
