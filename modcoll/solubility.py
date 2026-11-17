"""Pair interaction of two collagen molecules in solution: the second virial
coefficient over stagger, facing angles and axis separation, as a function of
pH, ionic strength and temperature.

Why a pair measure
------------------
Whether molecules stay dispersed or come together is decided by the balance
of attraction and repulsion between two of them, not by the net charge of
one. A molecule with a net charge of +35 e (native collagen at pH 7 in this
model) can still bind its neighbour if charge patterns and apolar patches
match at some stagger; a molecule with -180 e may still bind at high salt.
The quantity that weighs both, and that light scattering and osmometry
measure, is the osmotic second virial coefficient B2 (McMillan & Mayer 1945;
for proteins George & Wilson 1994, Neal, Asthagiri & Lenhoff 1998):

    Pi / (c kT) = 1 + B2 c + ...        B2 = -1/2 int dX ( exp(-U(X)/kT) - 1 )

over all relative placements X of molecule B with respect to molecule A.
B2 > 0: on balance the pair repels (good solvent, the solution is stable);
B2 < 0: on balance it attracts (poor solvent: aggregation or precipitation).

The aligned-rod B2
------------------
A placement here is: B's axis parallel to A's at transverse offset rho
(a vector in the plane normal to the axes), displaced by the stagger Delta
along it, either parallel or antiparallel (each with probability 1/2), and
both molecules turned about their own axes by uniformly distributed angles.
Turning both molecules and moving B round A at fixed |rho| only changes the
two facing azimuths (fa, fb) of ``modcoll.stagger``, so the integral over the
direction of rho and the two rotations is the uniform average over (fa, fb):

    B2 = 1/2 sum_o [ pi d_c^2 L                                   (hard core)
                     - pi int_{d_c}^{inf} rho drho int dDelta
                           < exp(-E_o(rho, Delta, fa, fb)) - 1 >_{fa, fb} ]

E_o is the site-by-site energy of ``electrostatics`` and ``stagger``
(screened Coulomb plus the hydrophobic contact term) evaluated at axis
separation rho. Two parallel cylinders of diameter d_c (the contact
distance of the stagger scan, 1.65 nm) and length L (the helix, 290 nm)
cannot overlap: for rho < d_c and |Delta| < L the Mayer function is -1,
which gives the first term, pi d_c^2 L = B2_hc, the hard-core value (half
the excluded volume of two parallel cylinders). Placements with rho < d_c
but no axial overlap (end to end) are left out (f = 0 there). Integrating
over rho >= d_c, every Delta and the facing angles is done numerically:
the inner sum is one FFT stagger scan per separation (``StaggerScanner``),
the Delta sum a Riemann sum at the scan resolution, the rho integral the
trapezoidal rule on a grid that is fine near contact, where the attraction
lives, and runs out to ``reach_debye`` Debye lengths, where the screened
interaction has died out.

Reported per condition:
  B2 (nm^3) and its two orientations; b2 = B2 / L (nm^2, per unit length);
  B2* = B2 / B2_hc (1 = hard rods; < 0 net attraction);
  the deepest bound state: F_best = min over (orientation, rho, Delta) of
  -ln < exp(-E) >_{fa, fb}, the free energy of a pair at fixed axial and
  transverse placement free to rotate; and the single lowest energy E_min
  with its electrostatic and hydrophobic parts.

Limits of the measure (spelled out in docs/modified_collagen_sites.md):
  * Aligned rods only. In an isotropic solution most encounters are crossed;
    crossed rods add Onsager's (pi/4) L^2 d of excluded volume per pair and
    cannot form the side-by-side contacts. Binding in solution therefore
    also costs the orientational entropy of aligning two 290 nm rods, which
    this measure does not charge.
  * Rigid straight molecules at fixed side-chain positions, pair additive,
    Debye-Hueckel electrostatics and independent-site protonation, as in the
    stagger scan.
  * B2 describes pairs. Fibril growth is cooperative and nucleated; a pair
    measure says in which direction the balance tips, not how fast or into
    what structure.

Temperature
-----------
Energies are in kT at the temperature of the solution. Two terms change:
  * the Bjerrum length l_B = e^2 / (4 pi eps0 eps_r(T) kT), with eps_r(T)
    of water from Malmberg & Maryott (1956); scaled so that it equals the
    package value 0.71 nm at 25 C (it varies by 4 % over 4-37 C, because
    eps_r T is nearly constant). The Debye length follows.
  * the hydrophobic strength, which grows with temperature (the entropic
    hydrophobic effect). The engine (src/sim.cpp; README "Temperature")
    multiplies its hydrophobic channel by 1 + 0.013 (T - 310.15 K) while kT
    scales with T. In kT units at T, relative to 25 C:
        s(T) = [(1 + 0.013 (T - 310.15)) / T] / [(1 + 0.013 (298.15 - 310.15)) / 298.15]
    which is 0.73 at 4 C and 1.14 at 37 C. pKa values are not moved with
    temperature.

Units: nm, kT at the solution temperature, mol/L, degrees Celsius for T.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import electrostatics as es
from . import stagger as st
from .sites import Molecule, SiteType

T_REF_C = 25.0            # temperature the package's energies refer to
KELVIN = 273.15
ENGINE_HYDROPHOBIC_SLOPE_PER_K = 0.013   # src/sim.cpp: 1 + 0.013 (T - 310.15)
ENGINE_HYDROPHOBIC_ANCHOR_K = 310.15


def water_permittivity(t_c) -> np.ndarray:
    """Static relative permittivity of water at t_c degrees Celsius
    (0-100 C), Malmberg & Maryott (1956), J. Res. Natl. Bur. Stand. 56:1:
        eps = 87.740 - 0.40008 t + 9.398e-4 t^2 - 1.410e-6 t^3."""
    t = np.asarray(t_c, dtype=float)
    return 87.740 - 0.40008 * t + 9.398e-4 * t ** 2 - 1.410e-6 * t ** 3


def bjerrum_length_nm(t_c: float) -> float:
    """l_B(T) = e^2 / (4 pi eps0 eps_r(T) k T), nm, scaled to the package's
    0.71 nm at 25 C: l_B(T) = 0.71 * eps_r(25) * 298.15 / (eps_r(T) * T)."""
    ref = float(water_permittivity(T_REF_C)) * (T_REF_C + KELVIN)
    return float(es.BJERRUM_NM * ref / (float(water_permittivity(t_c)) * (t_c + KELVIN)))


def debye_length_nm(ionic_strength_M: float, t_c: float = T_REF_C) -> float:
    """Debye length at ionic strength I and temperature T, nm
    (``electrostatics.debye_length_nm`` with the Bjerrum length at T)."""
    return es.debye_length_nm(ionic_strength_M, bjerrum_length_nm(t_c))


# Thermodynamics of type I collagen fibril formation (Kadler, Hojima & Prockop
# 1987, J. Biol. Chem. 262:15696): dH = +56 kcal/mol, dS = +220 cal/(K mol),
# per mole of molecules incorporated, constant over 29-41 C.
KADLER_DH_KCAL = 56.0
KADLER_DS_KCAL_PER_K = 0.220
GAS_CONSTANT_KCAL = 1.987204e-3          # kcal/(K mol)
HYDROPHOBIC_LAWS = ("engine", "kadler1987")


def hydrophobic_scale(t_c, law: str = "engine") -> np.ndarray:
    """Strength of the contact (hydrophobic) term in kT(T) relative to its
    value at 25 C.

    "engine"      the engine's law (module docstring); clipped at 0.05 of the
                  engine value, as the engine does (matters below about -30 C).
    "kadler1987"  the temperature dependence of the measured free energy of
                  fibril formation, g(T) = (dH - T dS) / RT with Kadler's dH
                  and dS: s(T) = g(T) / g(298.15 K). 0.56 at 4 C, 1.23 at 37 C.
                  Only the shape is taken from the measurement; the 25 C
                  strength stays the assumed eps_h."""
    if law == "kadler1987":
        def g(t_k):
            return (KADLER_DH_KCAL - t_k * KADLER_DS_KCAL_PER_K) / (GAS_CONSTANT_KCAL * t_k)
        t_k = np.asarray(t_c, dtype=float) + KELVIN
        return g(t_k) / g(T_REF_C + KELVIN)
    if law != "engine":
        raise ValueError(f"unknown hydrophobic law {law!r}; choose from {HYDROPHOBIC_LAWS}")

    def engine_over_kt(t_k):
        f = np.maximum(0.05, 1.0 + ENGINE_HYDROPHOBIC_SLOPE_PER_K
                       * (t_k - ENGINE_HYDROPHOBIC_ANCHOR_K))
        return f / t_k
    t_k = np.asarray(t_c, dtype=float) + KELVIN
    return engine_over_kt(t_k) / engine_over_kt(T_REF_C + KELVIN)


@dataclasses.dataclass(frozen=True)
class Conditions:
    """Solution conditions of one virial evaluation."""

    pH: float
    ionic_strength_M: float
    temperature_C: float = T_REF_C

    @property
    def debye_length_nm(self) -> float:
        return debye_length_nm(self.ionic_strength_M, self.temperature_C)


# Separations at which the pair is evaluated, as offsets from contact (nm).
# Fine near contact, where the hydrophobic term and the Boltzmann weight of a
# bound state change on the 0.05 nm scale; then 0.1 and 0.25 nm steps to
# 3 nm, and geometric (ratio 1.2) beyond, where only the screened net charge
# is left and the Mayer function decays exponentially (the radial rule,
# :func:`radial_integral`, is exact for exponentials).
NEAR_OFFSETS_NM = tuple(np.round(np.concatenate([
    np.arange(0.0, 0.3001, 0.025), np.arange(0.4, 1.0001, 0.1),
    np.arange(1.25, 3.0001, 0.25)]), 4))
FAR_OFFSETS_NM = tuple(np.round(3.0 * 1.2 ** np.arange(1, 28), 3))   # 3.6 ... 412 nm


def radial_integral(x, g) -> float:
    """int x g(x) dx over the nodes x (the rho integral of B2, rho drho).

    On each interval g is interpolated by the exponential through its two end
    values when they have the same sign (exact for g = A e^{bx}: the form of a
    decaying Mayer function far out and of a bound state's Boltzmann weight
    near contact), and linearly otherwise (then x g is quadratic and
    Simpson's rule is exact). For the exponential,
        int_{x1}^{x2} x g dx = (x2 g2 - x1 g1) / b - (g2 - g1) / b^2,
        b = ln(g2 / g1) / (x2 - x1)."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    x1, x2, g1, g2 = x[:-1], x[1:], g[:-1], g[1:]
    h = x2 - x1
    linear = h / 6.0 * (x1 * g1 + 2.0 * (x1 + x2) * (g1 + g2) / 2.0 + x2 * g2)
    same = g1 * g2 > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        bh = np.where(same, np.log(np.where(same, g2 / np.where(same, g1, 1.0), 1.0)), 0.0)
        b = bh / h
        expo = (x2 * g2 - x1 * g1) / b - (g2 - g1) / b ** 2
    use = same & (np.abs(bh) > 1e-4)
    return float(np.sum(np.where(use, expo, linear)))


# Which energies a virial coefficient is computed for:
#   full    screened Coulomb of the real charge pattern + hydrophobic term
#   el      the real charge pattern alone
#   hy      the hydrophobic term alone (charges off)
#   net_hy  the net charge spread uniformly over the chargeable sites, as an
#           azimuthally uniform line charge, + hydrophobic term: what is left
#           when the charge *pattern* is removed but the net charge kept
#   net     that uniform net charge alone
TERMS = ("full", "el", "hy", "net_hy", "net")


@dataclasses.dataclass(frozen=True)
class VirialSettings:
    """Geometry, quadrature and interaction parameters of the virial integral."""

    contact_nm: float = 2.0 * es.MOLECULE_RADIUS_NM + 0.15   # = stagger scan's 1.65 nm
    near_offsets_nm: tuple = NEAR_OFFSETS_NM
    far_offsets_nm: tuple = FAR_OFFSETS_NM
    reach_debye: float = 12.0        # rho_max = contact + max(min_reach, reach * lambda_D)
    min_reach_nm: float = 3.0        # covers the hydrophobic range at any salt
    lag_reach_debye: float = 12.0    # Delta beyond the overlap kept: this many lambda_D
    core_nm: float = es.SOFT_CORE_NM
    eps_hydrophobic: float = es.EPS_HYDROPHOBIC   # kT per KD unit^2 at 25 C
    hydrophobic_sigma_nm: float = es.HYDROPHOBIC_SIGMA_NM
    position: str = "centroid"
    n_shell_charge: int = 2
    n_shell_hydro: int = 4
    ds_nm: float = 0.2
    n_phi: int = 24
    orientations: tuple = ("parallel", "antiparallel")
    # Parallel staggers closer to zero than this are left out (f = 0 there),
    # as the stagger scan leaves out staggers below its lag_min of 5 nm. The
    # centred hydrophobic product makes a molecule attract an unshifted copy
    # of itself by about 90 kT whatever its residues are (sum of h_i^2), so
    # with this window closed the in-register parallel pair dominates every
    # attractive result. 0 keeps every stagger.
    in_register_exclusion_nm: float = 5.0
    hydrophobic_law: str = "engine"          # temperature law, see hydrophobic_scale

    def separations_nm(self) -> np.ndarray:
        off = np.unique(np.concatenate([self.near_offsets_nm, self.far_offsets_nm]))
        return self.contact_nm + off

    def rho_max_nm(self, cond: Conditions) -> float:
        return self.contact_nm + max(self.min_reach_nm,
                                     self.reach_debye * cond.debye_length_nm)


def _grid_for(extent_nm: float, lag_max_nm: float, reach_nm: float,
              settings: VirialSettings) -> st.ScanGrid:
    """Smallest power-of-two axial grid that holds a molecule of the given
    extent on its first half and keeps the periodic image of the partner,
    at any stagger up to lag_max, further away than ``reach_nm``."""
    need = (extent_nm + 10.0 + lag_max_nm + reach_nm) / settings.ds_nm
    half = (extent_nm + 10.0) / settings.ds_nm + 2
    n = 1024
    while n < need or n // 2 <= max(half, lag_max_nm / settings.ds_nm + 1):
        n *= 2
    return st.ScanGrid(ds_nm=settings.ds_nm, n_phi=settings.n_phi, n_z=n, s_origin_nm=-10.0)


def lag_slice(grid: st.ScanGrid, lag_min_nm: float, lag_max_nm: float):
    """(indices into the lag axis of a scan on ``grid``, staggers in nm) for
    the closed range, increasing (``StaggerScanner.lag_slice`` without a
    scanner)."""
    j = np.arange(int(np.ceil(lag_min_nm / grid.ds_nm - 1e-9)),
                  int(np.floor(lag_max_nm / grid.ds_nm + 1e-9)) + 1)
    if np.abs(j).max() >= grid.n_z // 2:
        raise ValueError("stagger range exceeds the axial grid")
    return j % grid.n_z, j * grid.ds_nm


def uniform_net_charge(molecule: Molecule, q) -> tuple[np.ndarray, np.ndarray]:
    """(weights, smeared mask) that spread the net charge sum(q) evenly over
    every site of ``molecule`` that can carry a charge (any type but OTHER in
    the native molecule), each as an azimuthally uniform ring: the charge
    *pattern* removed, the net charge and its axial extent kept."""
    can = molecule.native_type != int(SiteType.OTHER)
    w = np.where(can, float(np.sum(q)) / max(int(can.sum()), 1), 0.0)
    return w, can.copy()


@dataclasses.dataclass
class SlabResult:
    """What one separation rho contributes, for one condition and term."""

    g: dict            # orientation -> int dDelta <exp(-E) - 1>_{fa,fb}  (nm)
    f_best: dict       # orientation -> (F_min kT, Delta nm)
    e_best: dict       # orientation -> (E_min kT, Delta nm, E_el kT, E_hy kT)


def _reduce(energy, lags, ds_nm, keep=None, e_el=None, e_hy=None):
    """Integral over Delta of the angle-averaged Mayer function, the lowest
    rotationally averaged free energy and the lowest energy (with its parts)
    of one energy table E[fa, fb, lag]. Lags where ``keep`` is False count as
    f = 0 and are not candidates for the minima."""
    flat = energy.reshape(-1, energy.shape[-1])
    if keep is not None and not keep.all():
        flat = flat[:, keep]
        lags = lags[keep]
        e_el = None if e_el is None else e_el.reshape(-1, energy.shape[-1])[:, keep]
        e_hy = None if e_hy is None else e_hy.reshape(-1, energy.shape[-1])[:, keep]
    if flat.min() < -650.0:
        raise FloatingPointError("pair energy below -650 kT: exp(-E) would overflow")
    e0 = flat.min(axis=0)
    # F = -ln < exp(-E) >_{fa, fb} per stagger, without underflow where the
    # pair repels by hundreds of kT
    f_lag = e0 - np.log(np.exp(-(flat - e0)).mean(axis=0))
    g = float(np.expm1(-f_lag).sum() * ds_nm)
    i = int(np.argmin(f_lag))
    k = int(np.argmin(flat))
    a, j = divmod(k, flat.shape[1])
    parts = (float(e_el.reshape(flat.shape)[a, j]) if e_el is not None else float("nan"),
             float(e_hy.reshape(flat.shape)[a, j]) if e_hy is not None else float("nan"))
    return g, (float(f_lag[i]), float(lags[i])), (float(flat[a, j]), float(lags[j])) + parts


class PairVirial:
    """Second virial coefficient of two concrete modification patterns.

    The electrostatic scanners depend on (separation, ionic strength,
    temperature) and are built on demand; the deposits of the charges on the
    grid depend only on the charges and are cached per pattern and pH; the
    hydrophobic table depends only on the separation and is scaled with
    temperature. Everything is built from the native molecule's geometry,
    which modification does not change."""

    def __init__(self, native: Molecule, settings: VirialSettings = VirialSettings()):
        self.native = native
        self.settings = settings
        s = settings
        self.length_nm = float(native.helix_length_nm)
        self.b2_hc = float(np.pi * s.contact_nm ** 2 * self.length_nm)
        self.charge_radii = st.charged_shell_radii(native, s.n_shell_charge, s.position)
        self.hydro_radii = st.residue_shell_radii(native, s.n_shell_hydro)
        pos = native.positions(s.position)[0]
        flipped = es.flip_antiparallel(native.s, native.phi, native.helix_length_nm)[0]
        self.extent_nm = float(max(pos.max(), flipped.max()) - min(pos.min(), flipped.min()))
        self._hy_grid = _grid_for(self.extent_nm, self.extent_nm + 2.0, 6.0, s)

    # -- grids -------------------------------------------------------------
    def lag_max_nm(self, cond: Conditions) -> float:
        # never shorter than the hydrophobic table (overlap + 2 nm)
        return self.extent_nm + max(2.0 + 2.0 * self.settings.ds_nm,
                                    self.settings.lag_reach_debye * cond.debye_length_nm)

    def grid(self, cond: Conditions) -> st.ScanGrid:
        lam = cond.debye_length_nm
        return _grid_for(self.extent_nm, self.lag_max_nm(cond),
                         self.settings.lag_reach_debye * lam, self.settings)

    def separations(self, conditions) -> np.ndarray:
        """Common separation grid, up to the largest reach of any condition."""
        rho = self.settings.separations_nm()
        top = max(self.settings.rho_max_nm(c) for c in conditions)
        return rho[rho <= top + 1e-9]

    # -- one separation -------------------------------------------------------
    def slab(self, mol_a: Molecule, mol_b: Molecule, rho_nm: float, conditions,
             terms=TERMS, charge_model=es.henderson_hasselbalch) -> dict:
        """{(condition index, term): SlabResult} at axis separation rho, for
        every condition whose reach includes rho."""
        s = self.settings
        terms = tuple(terms)
        need_el = any(t in terms for t in ("full", "el"))
        need_net = any(t in terms for t in ("net_hy", "net"))
        need_hy = any(t in terms for t in ("full", "hy", "net_hy"))
        out = {}
        live = [k for k, c in enumerate(conditions) if rho_nm <= s.rho_max_nm(c) + 1e-9]
        if not live:
            return out
        e_hy_ref, hy_lags = {}, None
        if need_hy and s.eps_hydrophobic != 0.0:
            hy = st.StaggerScanner(
                lambda r: es.hydrophobic_contact(r, s.hydrophobic_sigma_nm),
                self.hydro_radii, rho_nm, self._hy_grid)
            ha, hb = mol_a.centered_hydrophobicity(), mol_b.centered_hydrophobicity()
            hy_idx, hy_lags = lag_slice(self._hy_grid, -(self.extent_nm + 2.0),
                                        self.extent_nm + 2.0)
            for o in s.orientations:
                e = hy.scan(mol_a, ha, mol_b, hb, antiparallel=(o == "antiparallel"),
                            position=s.position)
                e_hy_ref[o] = -s.eps_hydrophobic * e[:, :, hy_idx]
        # group the live conditions by electrostatic kernel (I, T)
        hy_done: dict = {}
        by_kernel: dict = {}
        for k in live:
            c = conditions[k]
            by_kernel.setdefault((c.ionic_strength_M, c.temperature_C), []).append(k)
        for (ionic, t_c), ks in by_kernel.items():
            cond0 = conditions[ks[0]]
            grid = self.grid(cond0)
            lam = cond0.debye_length_nm
            lb = bjerrum_length_nm(t_c)
            idx, lags = lag_slice(grid, -self.lag_max_nm(cond0), self.lag_max_nm(cond0))
            sc = None
            if need_el or need_net:
                sc = st.StaggerScanner(
                    lambda r, lam=lam, lb=lb: es.screened_coulomb(r, lam, s.core_nm, lb),
                    self.charge_radii, rho_nm, grid)
            ex = s.in_register_exclusion_nm
            keeps = {o: (np.abs(lags) >= ex - 1e-9 if (o == "parallel" and ex > 0) else None)
                     for o in s.orientations}
            scale = float(hydrophobic_scale(t_c, s.hydrophobic_law))
            e_hy = {}
            for o in s.orientations:
                full = np.zeros((s.n_phi, s.n_phi, len(lags)))
                if o in e_hy_ref:
                    j0 = int(round((hy_lags[0] - lags[0]) / s.ds_nm))
                    full[:, :, j0:j0 + len(hy_lags)] = scale * e_hy_ref[o]
                e_hy[o] = full
            for k in ks:
                c = conditions[k]
                env = es.Environment(pH=c.pH, ionic_strength_M=c.ionic_strength_M)
                q_a, q_b = charge_model(mol_a, env), charge_model(mol_b, env)
                tabs = {}
                if need_el:
                    a_hat = sc.transform_a(sc.deposit_molecule(mol_a, q_a, s.position))
                    for o in s.orientations:
                        b_til = sc.transform_b(sc.deposit_molecule(
                            mol_b, q_b, s.position, antiparallel=(o == "antiparallel")))
                        tabs[("el", o)] = sc.correlate(a_hat, b_til)[:, :, idx].copy()
                if need_net:
                    wa, ma = uniform_net_charge(mol_a, q_a)
                    wb, mb = uniform_net_charge(mol_b, q_b)
                    sa, pa, ra = mol_a.positions(s.position)
                    a_hat = sc.transform_a(sc.deposit(sa, pa, ra, wa, ma))
                    for o in s.orientations:
                        sb, pb, rb = mol_b.positions(s.position)
                        if o == "antiparallel":
                            sb, pb = es.flip_antiparallel(sb, pb, mol_b.helix_length_nm)
                        b_til = sc.transform_b(sc.deposit(sb, pb, rb, wb, mb))
                        tabs[("net", o)] = sc.correlate(a_hat, b_til)[:, :, idx].copy()
                for term in terms:
                    if term == "hy" and t_c in hy_done:
                        out[(k, term)] = hy_done[t_c]
                        continue
                    res = SlabResult({}, {}, {})
                    for o in s.orientations:
                        keep = keeps[o]
                        if term == "full":
                            e_el = tabs[("el", o)]
                            red = _reduce(e_el + e_hy[o], lags, s.ds_nm, keep, e_el, e_hy[o])
                        elif term == "el":
                            red = _reduce(tabs[("el", o)], lags, s.ds_nm, keep, tabs[("el", o)])
                        elif term == "hy":
                            red = _reduce(e_hy[o], lags, s.ds_nm, keep, None, e_hy[o])
                        elif term == "net_hy":
                            e_el = tabs[("net", o)]
                            red = _reduce(e_el + e_hy[o], lags, s.ds_nm, keep, e_el, e_hy[o])
                        elif term == "net":
                            red = _reduce(tabs[("net", o)], lags, s.ds_nm, keep, tabs[("net", o)])
                        else:
                            raise ValueError(f"unknown term {term!r}")
                        res.g[o], res.f_best[o], res.e_best[o] = red
                    out[(k, term)] = res
                    if term == "hy":
                        # the hydrophobic term depends on temperature only
                        hy_done[t_c] = res
        return out

    # -- the integral -----------------------------------------------------------
    def integrate(self, rho, slabs: dict, conditions, terms=TERMS) -> list[dict]:
        """B2 and the bound-state descriptors from per-separation results.

        ``slabs`` maps each separation in ``rho`` (same order) to the output
        of :meth:`slab`. Returns one dict per (condition, term)."""
        s = self.settings
        rho = np.asarray(rho, dtype=float)
        rows = []
        for k, c in enumerate(conditions):
            use = rho <= s.rho_max_nm(c) + 1e-9
            r = rho[use]
            for term in terms:
                res = [slabs[i][(k, term)] for i in np.flatnonzero(use)]
                row = {"pH": c.pH, "ionic_strength_M": c.ionic_strength_M,
                       "temperature_C": c.temperature_C, "term": term,
                       "debye_length_nm": c.debye_length_nm, "B2_hc_nm3": self.b2_hc}
                b2_o = []
                for o in s.orientations:
                    g = np.array([x.g[o] for x in res])
                    tail = -np.pi * radial_integral(r, g)
                    b2 = self.b2_hc + tail
                    b2_o.append(b2)
                    row[f"B2_{o}_nm3"] = b2
                    # last node's share of the tail: a truncation check
                    row[f"tail_end_{o}"] = float(abs(np.pi * r[-1] * g[-1] * (r[-1] - r[-2]))
                                                 / max(abs(tail), 1e-300))
                    fs = [(x.f_best[o][0], x.f_best[o][1], rr) for x, rr in zip(res, r)]
                    f, dl, rr = min(fs)
                    row[f"F_best_{o}_kT"], row[f"F_best_{o}_stagger_nm"] = f, dl
                    row[f"F_best_{o}_rho_nm"] = rr
                    es_ = [(x.e_best[o], rr) for x, rr in zip(res, r)]
                    (e, de, eel, ehy), re_ = min(es_, key=lambda t: t[0][0])
                    row[f"E_min_{o}_kT"], row[f"E_min_{o}_stagger_nm"] = e, de
                    row[f"E_min_{o}_rho_nm"] = re_
                    row[f"E_min_{o}_el_kT"], row[f"E_min_{o}_hy_kT"] = eel, ehy
                b2 = float(np.mean(b2_o))
                row["B2_nm3"] = b2
                row["b2_per_length_nm2"] = b2 / self.length_nm
                row["B2_star"] = b2 / self.b2_hc
                best = min(s.orientations, key=lambda o: row[f"F_best_{o}_kT"])
                row["F_best_kT"] = row[f"F_best_{best}_kT"]
                row["F_best_orientation"] = best
                row["F_best_stagger_nm"] = row[f"F_best_{best}_stagger_nm"]
                row["F_best_rho_nm"] = row[f"F_best_{best}_rho_nm"]
                low = min(s.orientations, key=lambda o: row[f"E_min_{o}_kT"])
                for key in ("kT", "stagger_nm", "rho_nm", "el_kT", "hy_kT"):
                    row[f"E_min_{key}"] = row[f"E_min_{low}_{key}"]
                row["E_min_orientation"] = low
                row["rho_max_nm"] = float(r[-1])
                rows.append(row)
        return rows

    def virial(self, mol_a: Molecule, mol_b: Molecule, conditions, terms=TERMS,
               charge_model=es.henderson_hasselbalch, map_fn=map) -> list[dict]:
        """B2 of one pair of patterns for every condition and term. ``map_fn``
        may be a process pool's map over separations."""
        conditions = list(conditions)
        rho = self.separations(conditions)
        slabs = list(map_fn(lambda x: self.slab(mol_a, mol_b, x, conditions, terms,
                                                charge_model), rho))
        return self.integrate(rho, slabs, conditions, terms)


# ------------------------------------------------------------ interpretation ---
def dimer_fraction(b2_attractive_nm3: float, conc_per_nm3: float) -> float:
    """Fraction of molecules paired at number concentration c for an
    association constant K = -B2 (nm^3, monomer-dimer equilibrium
    c = m + 2 K m^2): 1 - m / c. Zero for K <= 0."""
    k = -float(b2_attractive_nm3)
    if k <= 0.0 or conc_per_nm3 <= 0.0:
        return 0.0
    m = (-1.0 + np.sqrt(1.0 + 8.0 * k * conc_per_nm3)) / (4.0 * k)
    return float(1.0 - m / conc_per_nm3)


def per_nm3(mass_conc_g_per_L: float, molar_mass_da: float) -> float:
    """Molecules per nm^3 at a mass concentration (g/L = mg/mL)."""
    return mass_conc_g_per_L / molar_mass_da * es.AVOGADRO * 1e-24


# ------------------------------------------------- the material in pure water ---
WATER_KW = 1.0e-14     # ion product of water at 25 C, (mol/L)^2


def solution_pH(counts, conc_M: float, acid_per_molecule: float = 0.0,
                weak_acid_per_molecule: float = 0.0, weak_acid_pKa: float = 4.76,
                pka=es.DEFAULT_PKA_SET, base_M: float = 0.0) -> float:
    """pH of the material dissolved in pure water, from the charge balance

        [H+] - Kw/[H+] + c Z(pH) - c n_s - c n_w / (1 + 10^(pKa_w - pH)) + b = 0

    for molecules of type counts ``counts`` at molar concentration c with
    n_s strong-acid counter-anions (e.g. chloride) and n_w molecules of a weak
    acid (e.g. acetic acid, pKa 4.76) per molecule, plus a strong base b (mol/L)
    in the diluent. Z(pH) is the Henderson-Hasselbalch net charge of the
    molecule (``composition.net_charge_from_counts``). Concentrations stand
    for activities: at the ionic strengths of such solutions (about 1 mM)
    the error is a few hundredths of a pH unit. The left side falls
    monotonically with pH, so bisection finds the unique root. The numeric
    arguments broadcast against each other (one bisection for all)."""
    from . import composition as co
    n_s, n_w, b, c = np.broadcast_arrays(np.asarray(acid_per_molecule, float),
                                         np.asarray(weak_acid_per_molecule, float),
                                         np.asarray(base_M, float), np.asarray(conc_M, float))

    def balance(pH):
        h = 10.0 ** (-pH)
        z = co.net_charge_from_counts(counts, pH, pka)
        weak = n_w / (1.0 + 10.0 ** (weak_acid_pKa - pH))
        return h - WATER_KW / h + c * (z - n_s - weak) + b

    lo, hi = np.zeros(n_s.shape), np.full(n_s.shape, 14.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        up = balance(mid) > 0.0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    out = 0.5 * (lo + hi)
    return float(out) if out.ndim == 0 else out


def counter_anions_per_molecule(counts, conc_M: float, pH: float,
                                pka=es.DEFAULT_PKA_SET) -> float:
    """Strong-acid counter-anions per molecule that make a solution of the
    material in pure water read ``pH`` (the charge balance of
    :func:`solution_pH` solved for n_s): n_s = Z(pH) + ([H+] - [OH-]) / c.
    Negative values mean counter-cations (a salt of the acid groups)."""
    from . import composition as co
    h = 10.0 ** (-pH)
    return float(co.net_charge_from_counts(counts, pH, pka)) + (h - WATER_KW / h) / conc_M
