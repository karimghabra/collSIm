"""Association constants of one ion with one ligand group, and the fit of the
contact energies to the targets in ``params.toml``.

Definition
----------
For one ion and a fixed ligand group,

    K_c = (N_A * 1e-24 L/nm^3) / z_free
          * integral d^3r  <  sum_{s in class c} exp(-U(r, Omega, s))  >_Omega

in L/mol, where the average is over orientations (Haar measure), the sum over
the 2^n site states that belong to class c, and z_free = (1 + exp(-dG_des))^n
is the same state sum for a free ion. The classes are geometric:

  inner, m atoms   sites in state L hold m >= 1 partner atoms inside their
                   inner wells (m = 1 monodentate, m >= 2 multidentate)
  outer            no such site, but a partner atom within r_contact of the
                   centre
  (for sulfate)    direct: an oxygen inside a hydrogen-bond well of a donor;
                   outer: none, but a donor atom within r_contact of sulfur

K_total = K_inner + K_outer is what a potentiometric stability constant
measures; K_inner is what a contact-sensitive probe (Raman) measures.

The state sum is done exactly: U is linear in the site indicators, so

    sum_s exp(-U) = exp(-U_allW) * prod_k (1 + exp(-D_k)),
    D_k = U(only site k in state L) - U_allW,

and the class weights follow by keeping or dropping the second term of each
factor. The remaining six-dimensional integral over position and orientation
is done by importance sampling from a mixture of (uniform in a ball, Haar) and
"docked" configurations (a site placed on a partner atom), with the exact
mixture density in the weights. This shares only the energy function with the
Markov chain in ``sampler.py``; agreement between the two is a test of the
chain's detailed balance.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from ._jit import njit
from .energy import (KIND_CATION, L_STATE, MAX_SITES, W_STATE, e_cation_atom, e_oxyanion_atom,
                     inner_weight, rotate_dirs)
from .params import PARAMS_PATH, PER_NM3_PER_MOLAR, IonParams, load_params

N_CLASS = 4  # columns: mono, multi, outer, direct(unused for cations)


# --------------------------------------------------------------------------
# Small-molecule ligands used for calibration
# --------------------------------------------------------------------------
def acetate(mol: int = 0) -> geo.LigandSet:
    """Acetate at the origin, carboxylate pointing along +x: two oxygens plus
    the carboxyl and methyl carbons as inert atoms, so an ion cannot approach
    from the methyl side. C-C 0.15 nm."""
    ox = geo.carboxylate([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], perp=[0.0, 1.0, 0.0], mol=mol)
    inert = geo.TYPE_INDEX["inert"]
    methyl = geo.LigandSet(pos=[[-0.15, 0.0, 0.0]], type=[inert], charge=[0.0], mol=[mol], group=[0])
    out = geo.LigandSet.concat([ox, methyl])
    out.group[:] = 0
    return out


def succinate(dihedral_deg: float = 65.0, mol: int = 0) -> geo.LigandSet:
    """Succinate dianion, (-)OOC-CH2-CH2-COO(-), as a rigid conformer.

    Carbon chain C1-C2-C3-C4 with C-C 0.153 nm and C-C-C 112 deg; the default
    dihedral 65 deg is the gauche conformer, in which the two carboxylates can
    close on one cation, 180 deg the extended one. Each carboxylate points
    along its C(H2)->C bond with its O-C-O plane containing the chain's
    neighbouring bond direction; the four carbons are inert atoms (order:
    O, O, C of the first carboxylate, O, O, C of the second, then C2, C3). The two
    carboxylates are separate groups of one molecule.
    """
    b, ang, phi = 0.153, math.radians(112.0), math.radians(dihedral_deg)
    c2 = np.zeros(3)
    c3 = np.array([b, 0.0, 0.0])
    c1 = c2 + b * np.array([math.cos(ang), math.sin(ang), 0.0])
    # C4 from C3 at the same bond angle, rotated by the dihedral about C2-C3
    c4 = c3 + b * np.array([-math.cos(ang), math.sin(ang) * math.cos(phi), math.sin(ang) * math.sin(phi)])
    parts = []
    for c_carboxyl, c_prev, c_other in ((c1, c2, c3), (c4, c3, c2)):
        parts.append(geo.carboxylate(c_carboxyl, c_carboxyl - c_prev, perp=c_other - c_prev, mol=mol))
    inert = geo.TYPE_INDEX["inert"]
    methylenes = geo.LigandSet(pos=[c2, c3], type=[inert] * 2, charge=[0.0] * 2, mol=[mol] * 2, group=[0, 1])
    out = geo.LigandSet.concat(parts + [methylenes])      # each carboxylate: O, O, C
    out.group[:] = [0, 0, 0, 1, 1, 1, 0, 1]
    return out


def methylguanidinium(mol: int = 0) -> geo.LigandSet:
    """Methylguanidinium at the origin, NH2 pair pointing along +x, in the xy
    plane: three nitrogens plus the central and methyl carbons as inert atoms
    (N-CH3 0.146 nm beyond NE)."""
    n = geo.guanidinium([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], mol=mol)
    inert = geo.TYPE_INDEX["inert"]
    methyl = geo.LigandSet(pos=[[-geo.GUANIDINIUM_CN_NM - 0.146, 0.0, 0.0]], type=[inert],
                           charge=[0.0], mol=[mol], group=[0])
    out = geo.LigandSet.concat([n, methyl])
    out.group[:] = 0
    return out


def methylammonium(mol: int = 0) -> geo.LigandSet:
    """Methylammonium: nitrogen at the origin, methyl carbon 0.149 nm behind."""
    n = geo.ammonium([0.0, 0.0, 0.0], mol=mol)
    c = geo.LigandSet(pos=[[-0.149, 0.0, 0.0]], type=[geo.TYPE_INDEX["inert"]], charge=[0.0],
                      mol=[mol], group=[0])
    out = geo.LigandSet.concat([n, c])
    out.group[:] = 0
    return out


def imidazolium_ring(mol: int = 0) -> geo.LigandSet:
    """Imidazolium: two nitrogens straddling the origin in the xy plane, with
    the ring's three carbons as inert atoms on a regular pentagon of
    circumradius 0.115 nm (ring bond about 0.135 nm)."""
    R = 0.115
    ang = np.radians([90.0 + 72.0 * k for k in range(5)])
    ring = np.stack([R * np.cos(ang), R * np.sin(ang), np.zeros(5)], axis=1)
    # vertices 1 and 4 are the 1,3 pair (two bonds apart): the nitrogens
    tn, ti = geo.TYPE_INDEX["N_imidazolium"], geo.TYPE_INDEX["inert"]
    types = [ti, tn, ti, ti, tn]
    q = [0.0, 0.5, 0.0, 0.0, 0.5]
    normal = np.zeros((5, 3))
    normal[[1, 4]] = [0.0, 0.0, 1.0]
    return geo.LigandSet(pos=ring, type=types, charge=q, mol=[mol] * 5, group=[0] * 5, normal=normal)


def sulfate_as_ligand(params: IonParams, mol: int = 0) -> geo.LigandSet:
    """A sulfate frozen at the origin, presented to a cation as fixed atoms."""
    T = params.tables()
    sp = params.species_index("SO4")
    n = int(T.nsite[sp])
    pos = np.concatenate([np.zeros((1, 3)), T.d_site[sp] * T.dirs[sp, :n]])
    types = [int(T.t_centre[sp])] + [int(T.t_sat[sp])] * n
    q = [float(T.q_centre[sp])] + [float(T.q_sat[sp])] * n
    return geo.LigandSet(pos=pos, type=types, charge=q, mol=[mol] * (n + 1), group=[0] * (n + 1))


# --------------------------------------------------------------------------
# Integrator
# --------------------------------------------------------------------------
@njit
def _haar(rng, q):
    s = 0.0
    for a in range(4):
        q[a] = rng.normal()
        s += q[a] * q[a]
    s = math.sqrt(s)
    for a in range(4):
        q[a] /= s


@njit
def _ball(rng, radius, out):
    s = 0.0
    for a in range(3):
        out[a] = rng.normal()
        s += out[a] * out[a]
    f = radius * rng.random() ** (1.0 / 3.0) / math.sqrt(s)
    for a in range(3):
        out[a] *= f


@njit
def _logaddexp(a, b):
    """log(exp(a) + exp(b)) without overflow."""
    if a < b:
        a, b = b, a
    if b == -np.inf:
        return a
    return a + math.log1p(math.exp(b - a))


@njit
def _safe_exp(x):
    if x < -745.0:
        return 0.0
    if x > 700.0:
        x = 700.0
    return math.exp(x)


@njit
def class_weights(T, E, sp, r, U, apos, atype, aq, anorm, w):
    """Boltzmann weights of the contact classes for one ion at centre r with
    lab-frame site vectors U, summed over site states. Fills w[0:4] with
    (mono, multi, outer, direct); returns the total weight of all states,
    bound or not."""
    na = apos.shape[0]
    n = T.nsite[sp]
    for c in range(N_CLASS):
        w[c] = 0.0
    if T.kind[sp] != KIND_CATION:
        e = 0.0
        direct = False
        outer = False
        d = T.d_site[sp]
        for j in range(na):
            dx = apos[j, 0] - r[0]
            dy = apos[j, 1] - r[1]
            dz = apos[j, 2] - r[2]
            t = atype[j]
            e += e_oxyanion_atom(T, E, sp, U, dx, dy, dz, t, aq[j], anorm[j, 0], anorm[j, 1], anorm[j, 2])
            if not T.partner[sp, t]:
                continue
            if dx * dx + dy * dy + dz * dz < T.r_contact[sp] ** 2:
                outer = True
            planar = (anorm[j, 0] ** 2 + anorm[j, 1] ** 2 + anorm[j, 2] ** 2) > 0.25
            for k in range(n):
                ax = d * U[k, 0] - dx
                ay = d * U[k, 1] - dy
                az = d * U[k, 2] - dz
                a = math.sqrt(ax * ax + ay * ay + az * az)
                if abs(a - T.d_hb[sp]) < T.w_hb[sp]:
                    if planar:
                        s = (ax * anorm[j, 0] + ay * anorm[j, 1] + az * anorm[j, 2]) / (a * T.sin_plane_max[sp])
                        if s * s < 1.0:
                            direct = True
                    else:
                        direct = True
        b = math.exp(-e) if e < 700.0 else 0.0
        if direct:
            w[3] = b
        elif outer:
            w[2] = b
        return b

    state = np.zeros(MAX_SITES, dtype=np.uint8)
    # energy with every site hydrated, and flags
    e_w = 0.0
    outer = False
    held = np.zeros(MAX_SITES, dtype=np.int64)   # partner atoms inside each site's well
    for j in range(na):
        dx = apos[j, 0] - r[0]
        dy = apos[j, 1] - r[1]
        dz = apos[j, 2] - r[2]
        t = atype[j]
        e_w += e_cation_atom(T, E, sp, U, state, n, dx, dy, dz, t, aq[j])
        if T.partner[sp, t]:
            rr = math.sqrt(dx * dx + dy * dy + dz * dz)
            if rr < T.r_contact[sp]:
                outer = True
            if rr > 1e-9:
                for k in range(n):
                    if inner_weight(T, sp, U, k, dx, dy, dz, rr) > 0.0:
                        held[k] += 1
    # D_k = U(site k alone in state L) - U(all W). An atom sitting in a site
    # makes U_allW astronomically large (it overlaps the water) and D_k
    # correspondingly negative, so everything is combined in logarithms.
    # lc[m] = log of the coefficient of t^m in prod_{k holding h_k atoms}
    # (1 + x_k t^h_k), x_k = exp(-D_k): the weight of the open sites holding
    # exactly m atoms between them, i.e. of denticity m. (A calcium site may
    # hold both oxygens of one carboxylate, so h_k can be 2.)
    max_m = 2 * MAX_SITES
    lc = np.full(max_m + 1, -np.inf)
    lc[0] = 0.0
    n_held = 0
    log_free = 0.0  # log prod over sites that hold no atom of (1 + x_k)
    for k in range(n):
        state[k] = L_STATE
        e_k = T.dg_des[sp]
        for j in range(na):
            e_k += e_cation_atom(T, E, sp, U, state, n - 1, apos[j, 0] - r[0], apos[j, 1] - r[1],
                                 apos[j, 2] - r[2], atype[j], aq[j])
        state[k] = W_STATE
        lx = -(e_k - e_w)
        if held[k] > 0:
            h = min(held[k], 2)
            n_held = min(n_held + h, max_m)
            for m in range(n_held, h - 1, -1):
                lc[m] = _logaddexp(lc[m], lc[m - h] + lx)
        else:
            log_free += _logaddexp(0.0, lx)
    log_base = -e_w + log_free
    l_multi = -np.inf
    for m in range(2, n_held + 1):
        l_multi = _logaddexp(l_multi, lc[m])
    l_all = _logaddexp(_logaddexp(lc[0], lc[1]), l_multi)
    w[0] = _safe_exp(log_base + lc[1])
    w[1] = _safe_exp(log_base + l_multi)
    if outer:
        w[2] = _safe_exp(log_base)
    return _safe_exp(log_base + l_all)


@njit
def _integrate(T, E, sp, apos, atype, aq, anorm, centre, R_dom, alpha, rho, n_samples, z_free, rng,
               sums, sums2):
    """Importance-sampled integral of the class weights over the ball of radius
    R_dom about ``centre`` and over orientations. sums[c] / n_samples estimates
    integral d^3r <w_c>_Omega / z_free in nm^3."""
    n = T.nsite[sp]
    d = T.d_site[sp]
    na = apos.shape[0]
    cation = T.kind[sp] == KIND_CATION
    V_dom = 4.0 / 3.0 * math.pi * R_dom ** 3
    v_rho = 4.0 / 3.0 * math.pi * rho ** 3
    n_p = 0
    for j in range(na):
        if T.partner[sp, atype[j]]:
            n_p += 1
    if not cation or n_p == 0:
        alpha = 1.0
    q = np.empty(4)
    r = np.empty(3)
    xi = np.empty(3)
    U = np.zeros((MAX_SITES, 3))
    w = np.zeros(N_CLASS)
    for _ in range(n_samples):
        _haar(rng, q)
        rotate_dirs(T, sp, q, U)
        if rng.random() < alpha:
            _ball(rng, R_dom, xi)
            for a in range(3):
                r[a] = centre[a] + xi[a]
        else:
            k = int(rng.integers(0, n))
            pick = int(rng.integers(0, n_p))
            for j in range(na):
                if T.partner[sp, atype[j]]:
                    if pick == 0:
                        _ball(rng, rho, xi)
                        for a in range(3):
                            r[a] = apos[j, a] + xi[a] - d * U[k, a]
                        break
                    pick -= 1
        dx = r[0] - centre[0]
        dy = r[1] - centre[1]
        dz = r[2] - centre[2]
        if dx * dx + dy * dy + dz * dz >= R_dom * R_dom:
            continue  # outside the integration domain: integrand defined as 0
        # exact density of the mixture at (r, q), orientation measure normalised
        dens = alpha / V_dom
        if alpha < 1.0:
            cnt = 0
            for k in range(n):
                px = r[0] + d * U[k, 0]
                py = r[1] + d * U[k, 1]
                pz = r[2] + d * U[k, 2]
                for j in range(na):
                    if T.partner[sp, atype[j]]:
                        if (px - apos[j, 0]) ** 2 + (py - apos[j, 1]) ** 2 + (pz - apos[j, 2]) ** 2 < rho * rho:
                            cnt += 1
            dens += (1.0 - alpha) * cnt / (n * n_p * v_rho)
        class_weights(T, E, sp, r, U, apos, atype, aq, anorm, w)
        for c in range(N_CLASS):
            f = w[c] / (z_free * dens)
            sums[c] += f
            sums2[c] += f * f


@dataclass
class Association:
    """Association constants in L/mol with one-sigma sampling errors."""

    K_mono: float
    K_multi: float
    K_outer: float
    K_direct: float
    err: dict

    @property
    def K_inner(self) -> float:
        return self.K_mono + self.K_multi

    @property
    def K_total(self) -> float:
        return self.K_mono + self.K_multi + self.K_outer + self.K_direct

    @property
    def inner_fraction(self) -> float:
        """Share of inner-sphere (cation) or direct (sulfate) pairs."""
        return (self.K_inner + self.K_direct) / self.K_total

    @property
    def dG_total(self) -> float:
        """Standard binding free energy -ln(K * 1 mol/L) in kT."""
        return -math.log(self.K_total)


def association_constants(params: IonParams, species: str, ligands: geo.LigandSet,
                          ionic_strength: float, rng: np.random.Generator,
                          n_samples: int = 400_000, coulomb_cutoff: float = math.inf,
                          dock_fraction: float = 0.5, dock_ball: float | None = None,
                          tables=None) -> Association:
    """Association constants of one ion of ``species`` with the fixed group
    ``ligands`` (all atoms are treated as one group).

    coulomb_cutoff  inf for an isolated pair; pass the sampler's cutoff to get
                    the constant the sampler should reproduce.
    dock_ball       radius of the ball in which "site on a ligand atom"
                    samples are drawn; default the lateral radius of the
                    species' inner well. Affects the variance only.
    tables          optional replacement for ``params.tables()``
    """
    T = params.tables() if tables is None else tables
    E = params.elec(ionic_strength, coulomb_cutoff)
    sp = params.species_index(species)
    part = T.partner[sp, ligands.type]
    if not part.any():
        raise ValueError(f"{species} has no contact partner among these atoms")
    centre = ligands.pos[part].mean(axis=0)
    extent = float(np.linalg.norm(ligands.pos[part] - centre, axis=1).max())
    # every bound configuration has the ion centre inside this ball
    R_dom = extent + float(T.r_contact[sp]) + 0.02
    if dock_ball is None:
        dock_ball = max(float(T.w_r[sp]), float(T.d_site[sp] * math.sqrt(max(0.0, 1.0 - T.cos_max[sp] ** 2))), 0.05)
    # free-ion state sum (1 + exp(-dG_des))^n; exactly 1 for sulfate (dG_des = 0
    # is never used there, so guard on the kind)
    z_free = (1.0 + math.exp(-float(T.dg_des[sp]))) ** int(T.nsite[sp]) if T.kind[sp] == KIND_CATION else 1.0
    sums = np.zeros(N_CLASS)
    sums2 = np.zeros(N_CLASS)
    _integrate(T, E, sp, ligands.pos, ligands.type, ligands.charge, ligands.normal, centre, R_dom,
               1.0 - dock_fraction, dock_ball, int(n_samples), z_free, rng, sums, sums2)
    mean = sums / n_samples
    var = np.maximum(sums2 / n_samples - mean ** 2, 0.0) / n_samples
    K = mean * PER_NM3_PER_MOLAR
    err = np.sqrt(var) * PER_NM3_PER_MOLAR
    return Association(K[0], K[1], K[2], K[3],
                       {"mono": err[0], "multi": err[1], "outer": err[2], "direct": err[3]})


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
@dataclass
class FitSpec:
    """One calibration: which ion, which ligand, which targets, which knobs."""

    name: str            # key under [targets]
    species: str
    ligand_type: str     # key under [contact.<species>]
    build: object        # callable(params) -> LigandSet


FITS = (
    FitSpec("Ca_acetate", "Ca", "O_carboxylate", lambda p: acetate()),
    FitSpec("Mg_acetate", "Mg", "O_carboxylate", lambda p: acetate()),
    FitSpec("Ca_sulfate", "Ca", "O_sulfate", sulfate_as_ligand),
    FitSpec("Mg_sulfate", "Mg", "O_sulfate", sulfate_as_ligand),
    FitSpec("SO4_guanidinium", "SO4", "N_guanidinium", lambda p: methylguanidinium()),
    FitSpec("SO4_ammonium", "SO4", "N_ammonium", lambda p: methylammonium()),
    FitSpec("SO4_imidazolium", "SO4", "N_imidazolium", lambda p: imidazolium_ring()),
)


def target_values(params: IonParams, name: str):
    """(K_total, K_inner or None, ionic strength) of a target block."""
    block = params.raw["targets"][name]
    if "log_K_total" in block:
        K_total = 10.0 ** float(block["log_K_total"]["value"])
        I = float(block["log_K_total"]["ionic_strength"])
    else:
        K_total = float(block["K_total"]["value"])
        I = float(block["K_total"]["ionic_strength"])
    K_inner = None
    if "K_inner" in block:
        K_inner = float(block["K_inner"]["value"])
    elif "inner_fraction" in block:
        K_inner = float(block["inner_fraction"]["value"]) * K_total
    return K_total, K_inner, I


def _solve_log(f, x0, target, slope0=1.0, tol=2e-3, max_iter=25, x_min=-20.0, x_max=60.0):
    """Solve f(x) = target for a positive, monotonically increasing f by a
    secant iteration on log f. slope0 is the first guess of d ln f / dx (1 for
    a Boltzmann factor exp(x))."""
    x = x0
    y = math.log(f(x) / target)
    slope = slope0
    for _ in range(max_iter):
        if abs(y) < tol:
            return x
        step = -y / slope
        step = max(-3.0, min(3.0, step))
        x_new = min(x_max, max(x_min, x + step))
        y_new = math.log(f(x_new) / target)
        if x_new != x and (y_new - y) / (x_new - x) > 1e-3:
            slope = (y_new - y) / (x_new - x)
        x, y = x_new, y_new
    raise RuntimeError(f"no convergence: log ratio {y:+.3f} at x = {x:.3f}")


def evaluate(params: IonParams, spec: FitSpec, seed: int = 20261112, n_samples: int = 400_000,
             ionic_strength: float | None = None) -> Association:
    """Association constants of a calibration pair at its target ionic strength
    (or another one). The seed is fixed so the result is a deterministic,
    smooth function of the parameters (common random numbers)."""
    _, _, I = target_values(params, spec.name)
    if ionic_strength is not None:
        I = ionic_strength
    return association_constants(params, spec.species, spec.build(params), I,
                                 np.random.default_rng(seed), n_samples=n_samples)


def fit_one(params: IonParams, spec: FitSpec, seed: int = 20261112, n_samples: int = 400_000,
            rounds: int = 4) -> dict:
    """Solve for the adjustable parameters of one pair; returns {path: value}.

    Sulfate with a donor: eps_hb from K_total.
    Cation with a ligand: eps_in from K_inner and eps_out from K_total,
    alternated because each shifts the other slightly. If the target block has
    a ``bidentate_fraction`` the cation's angular slack (well_angular_flat_deg)
    is fitted to it in the same loop; that parameter belongs to the cation, so
    it is fitted on one ligand only and then kept for the others.
    """
    K_total, K_inner, _ = target_values(params, spec.name)
    base = f"contact.{spec.species}.{spec.ligand_type}"
    if spec.species == "SO4":
        key = f"{base}.eps_hb"
        x = _solve_log(lambda v: evaluate(params.with_overrides({key: v}), spec, seed, n_samples).K_total,
                       float(params.value(key)), K_total, slope0=0.7)
        return {key: round(x, 3)}
    k_in, k_out = f"{base}.eps_in", f"{base}.eps_out"
    cur = {k_in: float(params.value(k_in)), k_out: float(params.value(k_out))}
    block = params.raw["targets"][spec.name]
    k_flat = None
    if "bidentate_fraction" in block:
        k_flat = f"species.{spec.species}.well_angular_flat_deg"
        cur[k_flat] = float(params.value(k_flat))
        f_bi = float(block["bidentate_fraction"]["value"])
        flat_max = float(params.value(f"species.{spec.species}.well_angular_halfwidth_deg")) - 1.0

    def ev(over):
        return evaluate(params.with_overrides({**cur, **over}), spec, seed, n_samples)

    for _ in range(rounds):
        if k_flat is not None:
            # odds of bidentate against monodentate, monotonic in the slack
            odds = f_bi / (1.0 - f_bi)
            cur[k_flat] = _solve_log(lambda v: (lambda a: a.K_multi / a.K_mono)(ev({k_flat: v})),
                                     cur[k_flat], odds, slope0=0.4, tol=1e-2, x_min=0.0, x_max=flat_max)
        cur[k_in] = _solve_log(lambda v: ev({k_in: v}).K_inner, cur[k_in], K_inner)
        cur[k_out] = _solve_log(lambda v: ev({k_out: v}).K_total, cur[k_out], K_total, slope0=0.5)
    return {k: round(v, 3) for k, v in cur.items()}


def fit_all(params: IonParams, seed: int = 20261112, n_samples: int = 400_000, log=None) -> dict:
    """Fit every pair in the order of FITS. A cation's carboxylate fit comes
    before its sulfate fit, so a fitted angular slack is in place for the
    latter."""
    out = {}
    for spec in FITS:
        fitted = fit_one(params.with_overrides(out), spec, seed, n_samples)
        out.update(fitted)
        if log is not None:
            log(spec.name, fitted)
    return out


def write_fitted(values: dict, path=PARAMS_PATH) -> None:
    """Rewrite the ``value`` line of each given table in place, leaving every
    other character of the file untouched."""
    text = open(path, encoding="utf-8").read()
    for dotted, val in values.items():
        pattern = re.compile(r"(^\[" + re.escape(dotted) + r"\]\s*\nvalue\s*=\s*)([^\n]*)", re.M)
        text, count = pattern.subn(lambda m: m.group(1) + repr(float(val)), text)
        if count != 1:
            raise ValueError(f"could not find a unique value line for [{dotted}]")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def checks(params: IonParams, seed: int = 20261112, n_samples: int = 400_000) -> list[dict]:
    """Measured constants the model was not fitted to, against the model."""
    by_name = {spec.name: spec for spec in FITS}
    rows = []
    for name, block in params.raw.get("checks", {}).items():
        for key, entry in block.items():
            I = float(entry["ionic_strength"])
            if name in by_name:
                a = evaluate(params, by_name[name], seed, n_samples, ionic_strength=I)
                model = {"log_K_total": math.log10(a.K_total)}
            else:   # succinate: both rigid conformers
                species = name.split("_")[0]
                model = {}
                for label, dihedral in (("gauche", 65.0), ("extended", 180.0)):
                    a = association_constants(params, species, succinate(dihedral), I,
                                              np.random.default_rng(seed), n_samples=n_samples)
                    model[f"log_K_total_{label}"] = math.log10(a.K_total)
            rows.append({"check": f"{name}.{key}", "ionic_strength": I, "measured": float(entry["value"]),
                         "lo": float(entry["lo"]), "hi": float(entry["hi"]), **model})
    return rows


def report(params: IonParams, seed: int = 20261112, n_samples: int = 400_000) -> list[dict]:
    """Model versus target for every calibration pair."""
    rows = []
    for spec in FITS:
        K_total, K_inner, I = target_values(params, spec.name)
        a = evaluate(params, spec, seed, n_samples)
        rows.append({
            "pair": spec.name, "ionic_strength": I,
            "K_total_target": K_total, "K_total_model": a.K_total,
            "K_inner_target": K_inner, "K_inner_model": a.K_inner + a.K_direct,
            "K_mono": a.K_mono, "K_multi": a.K_multi, "K_outer": a.K_outer, "K_direct": a.K_direct,
            "inner_fraction": a.inner_fraction, "dG_total_kT": a.dG_total,
            "bidentate_fraction": (a.K_multi / a.K_inner) if a.K_inner > 0 else None,
            "errors": {k: float(v) for k, v in a.err.items()},
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="refit and rewrite the [contact.*] values")
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args(argv)
    params = load_params()
    if args.write:
        fitted = fit_all(params, args.seed, args.samples,
                         log=lambda name, vals: print(f"fitted {name}: {vals}", flush=True))
        write_fitted(fitted)
        params = load_params()
    print(f"{'pair':18s} {'I':>5s} {'K_tot target':>12s} {'model':>8s} {'K_in target':>11s} "
          f"{'model':>8s} {'mono':>7s} {'multi':>7s} {'outer':>7s} {'inner frac':>10s}")
    for r in report(params, args.seed, args.samples):
        kin = "-" if r["K_inner_target"] is None else f"{r['K_inner_target']:.2f}"
        print(f"{r['pair']:18s} {r['ionic_strength']:5.2f} {r['K_total_target']:12.2f} "
              f"{r['K_total_model']:8.2f} {kin:>11s} {r['K_inner_model']:8.2f} {r['K_mono']:7.2f} "
              f"{r['K_multi']:7.2f} {r['K_outer']:7.2f} {r['inner_fraction']:10.2f}")
    print("\nnot fitted:")
    for c in checks(params, args.seed, args.samples):
        model = ", ".join(f"{k[12:] or 'model'} {v:.2f}" for k, v in c.items() if k.startswith("log_K_total"))
        print(f"  {c['check']:36s} I = {c['ionic_strength']:.1f}  measured {c['measured']:.2f} "
              f"({c['lo']:.2f}-{c['hi']:.2f})  model: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
