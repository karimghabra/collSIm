"""Protonation and site-site interactions between two straight molecules.

Units throughout: lengths nm, energies kT at 298 K, charges in elementary
charges, concentrations mol/L.

The energy between two molecules is always a sum over the concrete sites of
two concrete modification patterns:

    E = sum_i sum_j  q_i q_j  u_el(r_ij)  -  eps_h sum_i sum_j h_i h_j g(r_ij)

with the screened-Coulomb pair term u_el and the hydrophobic contact kernel g
defined below. ``pair_energy_direct`` evaluates that double sum literally for
one relative placement; ``modcoll.stagger`` evaluates the same sum for all
staggers and facing angles at once on a grid and is tested against it.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .sites import Molecule, SiteType

# Bjerrum length of water at 298 K, nm: l_B = e^2 / (4 pi eps0 eps_r kT) with
# eps_r = 78.4 gives 0.715 nm; the conventional rounded value 0.71 nm is used
# (Israelachvili, Intermolecular and Surface Forces, 3rd ed., ch. 14).
BJERRUM_NM = 0.71

# Avogadro constant, 1/mol (exact, SI 2019).
AVOGADRO = 6.02214076e23

# Soft-core length of the screened Coulomb term, nm. The pair distance enters
# as sqrt(r^2 + a^2), so the energy of two unit charges saturates at
# l_B exp(-kappa a) / a instead of diverging. a = 0.35 nm is a typical
# heavy-atom N...O distance of a salt bridge; the cap is then about 2 kT at
# low salt. It is a regularisation, not a fitted contact energy.
SOFT_CORE_NM = 0.35

# Hydrophobic contact kernel g(r) = exp(-r^2 / (2 sigma^2)), nm. The existing
# surface-correlation tool smooths each density with a 0.35 nm Gaussian
# (tools/azimuthal_profile.py SIG_S); the product of two such densities is a
# Gaussian in separation of width sqrt(2) * 0.35 = 0.49 nm.
HYDROPHOBIC_SIGMA_NM = 0.5

# Strength of the hydrophobic channel, kT per (Kyte-Doolittle unit)^2.
# ASSUMPTION / calibration: two leucine-like side chains (centred index about
# +4.6 each) in full contact gain about 1 kT, the order of the contact minimum
# of small-hydrophobe potentials of mean force in water (e.g. methane-methane,
# about -0.7 kcal/mol = -1.2 kT). 1 / 4.6^2 = 0.047, rounded to 0.05.
EPS_HYDROPHOBIC = 0.05

# Nominal molecular radius, nm (README: "A = pi (0.75 nm)^2"). The surface
# separation of two parallel molecules is axis distance minus 2 * this.
MOLECULE_RADIUS_NM = 0.75

# pKa sets, by name.
#
# "repository": the side-chain values the existing tools and the README use
# ("Asp 3.65, Glu 4.25, His 6.0, Lys 10.53, Arg 12.48"); tyrosine does not
# titrate in it. It is what the native gate is evaluated with, so that gate
# stays comparable with the existing tools.
#
# "recommended" (the default): model-compound values at 25 C and ionic
# strength 0.05-0.1.
#   Asp 3.9   alanine pentapeptides, potentiometric 3.90 and NMR 3.98
#             (Grimsley, Scholtz & Pace 2009, Protein Sci. 18:247, revising
#             the 3.67 of Thurlkill et al. 2006); Gly tripeptides 3.86-3.88
#             (Platzer, Okon & McIntosh 2014, J. Biomol. NMR 60:109)
#   Glu 4.3   4.25 (Thurlkill et al. 2006, Protein Sci. 15:1214); 4.34 (Platzer)
#   His 6.5   6.54 (Thurlkill); 6.44 (Platzer)
#   Lys 10.4  10.40 (Thurlkill); 10.34 (Platzer)
#   Arg 13.8  Fitch et al. 2015, Protein Sci. 24:752
#   Tyr 9.8   9.84 (Thurlkill)
# Of these only Asp matters at pH 4.2: 3.65 -> 3.9 leaves about 15 fewer
# negative charges on a molecule there.
#
# Both sets share the groups the repository never modelled.
#   alpha-carboxyl 3.67 (3.7), alpha-amino 8.00: Thurlkill et al. 2006.
#   succinyl carboxylate 4.5: the free acid end of a succinic monoamide. No
#   measured value exists for N-epsilon-succinyl-lysine. Succinic acid has
#   pKa1 = 4.21 at 25 C and zero ionic strength (NIST critically selected
#   constants); its first proton can leave from either of two equivalent
#   carboxyls, so one carboxyl alone has the microscopic constant
#   pKa1 + log10(2) = 4.51. Related monoamides and keto acids fall at 4.4-4.7.
#   Uncertainty about +-0.15, which is +-9 e on the net charge at pH 4.2.
_SHARED_PKA = {SiteType.NTERM: 8.00, SiteType.SUCC_LYS: 4.5, SiteType.SUCC_NTERM: 4.5}
PKA_SETS = {
    "repository": {SiteType.ASP: 3.65, SiteType.GLU: 4.25, SiteType.HIS: 6.0,
                   SiteType.LYS: 10.53, SiteType.ARG: 12.48, SiteType.CTERM: 3.67,
                   **_SHARED_PKA},
    "recommended": {SiteType.ASP: 3.9, SiteType.GLU: 4.3, SiteType.HIS: 6.5,
                    SiteType.LYS: 10.4, SiteType.ARG: 13.8, SiteType.TYR: 9.8,
                    SiteType.CTERM: 3.7, **_SHARED_PKA},
}
DEFAULT_PKA_SET = "recommended"
PKA = PKA_SETS[DEFAULT_PKA_SET]

# Charge of the ionised form: acids -1 (deprotonated), bases +1 (protonated).
VALENCE = {
    SiteType.ASP: -1, SiteType.GLU: -1, SiteType.CTERM: -1, SiteType.TYR: -1,
    SiteType.SUCC_LYS: -1, SiteType.SUCC_NTERM: -1,
    SiteType.HIS: +1, SiteType.LYS: +1, SiteType.ARG: +1, SiteType.NTERM: +1,
}


@dataclasses.dataclass(frozen=True)
class Environment:
    """Solution conditions seen by a molecule.

    ``ionic_strength_M`` is the ionic strength of the monovalent background
    electrolyte in mol/L; it enters only through the Debye length. Divalent
    ions are not part of this module (they are explicit particles elsewhere).
    Temperature is fixed at 298 K by the Bjerrum length.
    """

    pH: float
    ionic_strength_M: float = 0.15

    @property
    def debye_length_nm(self) -> float:
        return debye_length_nm(self.ionic_strength_M)


def debye_length_nm(ionic_strength_M: float, bjerrum_nm: float = BJERRUM_NM) -> float:
    """Debye screening length, nm.

    kappa^2 = 8 pi l_B N_A I, with I in mol/L converted to a number density
    per nm^3 (1 L = 1e24 nm^3): n = I * N_A * 1e-24. For l_B = 0.71 nm this is
    0.305 nm / sqrt(I), the textbook 0.304 / sqrt(I) for a 1:1 salt at 25 C
    (Israelachvili, eq. 14.36).
    """
    if ionic_strength_M <= 0:
        raise ValueError("ionic strength must be positive")
    kappa2 = 8.0 * np.pi * bjerrum_nm * AVOGADRO * 1e-24 * ionic_strength_M
    return float(1.0 / np.sqrt(kappa2))


# -------------------------------------------------------------- protonation ---
def henderson_hasselbalch(molecule: Molecule, env: Environment, pka=None) -> np.ndarray:
    """Mean charge of every site at ``env.pH``, each group titrating
    independently with its model pKa:

        acid:  q = -1 / (1 + 10^(pKa - pH))
        base:  q = +1 / (1 + 10^(pH - pKa))

    ``pka`` is a table {SiteType: pKa} or the name of one of PKA_SETS; the
    default is the "recommended" set. A type missing from the table carries
    no charge.

    This is the default *charge model*: a function (molecule, environment) ->
    per-site charges. Everything downstream takes such a function, so a
    charge-regulation model that shifts pKa with the local potential can
    replace it without touching the site array or the energy code.
    """
    table = PKA if pka is None else (PKA_SETS[pka] if isinstance(pka, str) else pka)
    q = np.zeros(len(molecule))
    for st, pk in table.items():
        mask = molecule.site_type == int(st)
        if not mask.any():
            continue
        z = VALENCE[st]
        q[mask] = z / (1.0 + 10.0 ** (z * (env.pH - pk)))
    return q


def charge_model_for(pka_set: str):
    """Henderson-Hasselbalch charge model using the named pKa set."""
    if pka_set not in PKA_SETS:
        raise ValueError(f"unknown pKa set {pka_set!r}; choose from {sorted(PKA_SETS)}")

    def model(molecule: Molecule, env: Environment) -> np.ndarray:
        return henderson_hasselbalch(molecule, env, PKA_SETS[pka_set])

    model.__name__ = f"henderson_hasselbalch_{pka_set}"
    return model


def net_charge(molecule: Molecule, env: Environment,
               charge_model=henderson_hasselbalch) -> float:
    """Net charge of one molecule, elementary charges."""
    return float(charge_model(molecule, env).sum())


def isoelectric_point(molecules, ionic_strength_M: float = 0.15,
                      charge_model=henderson_hasselbalch,
                      lo: float = 0.0, hi: float = 14.0) -> float:
    """pH at which the summed net charge of ``molecules`` (one Molecule or a
    sequence of them) is zero, by bisection. The net charge of any set of
    acids and bases decreases monotonically with pH, so the root is unique;
    NaN is returned if there is none in [lo, hi]."""
    mols = [molecules] if isinstance(molecules, Molecule) else list(molecules)

    def total(pH):
        env = Environment(pH=pH, ionic_strength_M=ionic_strength_M)
        return sum(net_charge(m, env, charge_model) for m in mols)

    q_lo, q_hi = total(lo), total(hi)
    if not (q_lo > 0.0 > q_hi):
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if total(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------- kernels ---
def screened_coulomb(r_nm, debye_nm: float, core_nm: float = SOFT_CORE_NM,
                     bjerrum_nm: float = BJERRUM_NM):
    """Debye-Hueckel energy of two unit charges at distance r, in kT:

        u(r) = l_B exp(-r_eff / lambda_D) / r_eff,   r_eff = sqrt(r^2 + a^2)

    (linearised Poisson-Boltzmann in a uniform dielectric; a = soft core,
    a = 0 recovers the bare Yukawa form)."""
    r_eff = np.sqrt(np.asarray(r_nm, dtype=float) ** 2 + core_nm ** 2)
    return bjerrum_nm * np.exp(-r_eff / debye_nm) / r_eff


def hydrophobic_contact(r_nm, sigma_nm: float = HYDROPHOBIC_SIGMA_NM):
    """Dimensionless contact kernel g(r) = exp(-r^2 / (2 sigma^2))."""
    return np.exp(-np.asarray(r_nm, dtype=float) ** 2 / (2.0 * sigma_nm ** 2))


# ---------------------------------------------------------------- geometry ---
def flip_antiparallel(s, phi, helix_length_nm: float):
    """Coordinates of a molecule turned end over end about an axis
    perpendicular to its own (a 180 degree rotation about the lab x axis):
    s -> L - s, phi -> -phi. Same convention as the existing tool."""
    return helix_length_nm - np.asarray(s), -np.asarray(phi)


def transverse_distance_sq(r_a, alpha, r_b, beta, axis_separation_nm: float):
    """Squared distance, in the plane normal to the axes, between a site of
    molecule A at radius r_a and angle ``alpha`` from the direction A->B, and
    a site of molecule B at radius r_b and angle ``beta`` from the direction
    B->A (both angles counted in each molecule's own azimuth sense).

    With A's axis at the origin and B's axis at (d, 0), molecule B is A's
    frame rotated by pi about z, so the two sites are at
    (r_a cos a, r_a sin a) and (d - r_b cos b, -r_b sin b).
    """
    dx = axis_separation_nm - r_a * np.cos(alpha) - r_b * np.cos(beta)
    dy = r_a * np.sin(alpha) + r_b * np.sin(beta)
    return dx * dx + dy * dy


def _expand_smeared(s, phi, r, w, smeared, n_ring):
    """Replace each azimuthally uniform site by ``n_ring`` equally weighted
    points on its ring (the quadrature the grid method uses)."""
    if not np.any(smeared):
        return s, phi, r, w
    ring = 2.0 * np.pi * np.arange(n_ring) / n_ring
    keep = ~smeared
    return (np.concatenate([s[keep], np.repeat(s[smeared], n_ring)]),
            np.concatenate([phi[keep], np.tile(ring, int(smeared.sum()))]),
            np.concatenate([r[keep], np.repeat(r[smeared], n_ring)]),
            np.concatenate([w[keep], np.repeat(w[smeared] / n_ring, n_ring)]))


def pair_energy_direct(mol_a: Molecule, w_a, mol_b: Molecule, w_b, kernel,
                       stagger_nm: float, facing_a: float, facing_b: float,
                       axis_separation_nm: float, antiparallel: bool = False,
                       position: str = "centroid", n_ring: int = 24) -> float:
    """sum_i sum_j w_a[i] w_b[j] kernel(r_ij) for one placement, by brute force.

    Molecule A lies on the z axis; molecule B is parallel to it at
    ``axis_separation_nm``, displaced by ``stagger_nm`` along +z (a site of B
    at axial coordinate s sits at s + stagger in A's frame). ``facing_a`` is
    the azimuth of A that points at B, ``facing_b`` the azimuth of B that
    points at A. ``antiparallel`` first turns B end over end
    (:func:`flip_antiparallel`); ``facing_b`` then refers to the flipped
    molecule's azimuth. ``kernel`` maps distance (nm) to energy per unit
    weight product; with w = charges and kernel = :func:`screened_coulomb`
    the result is the electrostatic energy in kT.
    """
    sa, pa, ra = mol_a.positions(position)
    sb, pb, rb = mol_b.positions(position)
    if antiparallel:
        sb, pb = flip_antiparallel(sb, pb, mol_b.helix_length_nm)
    sa, pa, ra, wa = _expand_smeared(sa, pa, ra, np.asarray(w_a, float),
                                     mol_a.smeared, n_ring)
    sb, pb, rb, wb = _expand_smeared(sb, pb, rb, np.asarray(w_b, float),
                                     mol_b.smeared, n_ring)
    ka, kb = np.flatnonzero(wa), np.flatnonzero(wb)
    sa, pa, ra, wa = sa[ka], pa[ka], ra[ka], wa[ka]
    sb, pb, rb, wb = sb[kb], pb[kb], rb[kb], wb[kb]
    total = 0.0
    for lo in range(0, len(sa), 256):          # chunked to bound memory
        sl = slice(lo, lo + 256)
        dz = sb[None, :] + stagger_nm - sa[sl, None]
        rho2 = transverse_distance_sq(ra[sl, None], pa[sl, None] - facing_a,
                                      rb[None, :], pb[None, :] - facing_b,
                                      axis_separation_nm)
        total += float(np.sum(wa[sl, None] * wb[None, :]
                              * kernel(np.sqrt(dz * dz + rho2))))
    return total
