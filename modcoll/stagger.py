"""Two-molecule stagger scan: pair energy for every axial stagger and every
pair of facing azimuths at once, and the analysis of the resulting profiles.

Method
------
For two straight parallel molecules the pair sum

    E(Delta, fa, fb) = sum_i sum_j w_i w_j K(s_j + Delta - s_i, rho_ij)

depends on the stagger Delta only through the axial separation and on the
facing azimuths (fa, fb) only through the angles phi_i - fa and phi_j - fb
that fix the transverse distance rho_ij. Depositing the site weights of each
molecule on a cylindrical grid (axial s, azimuth phi, radial shell r) turns
the sum into a three-dimensional cross-correlation in (Delta, fa, fb), which
is evaluated with one FFT per radial shell pair. No azimuthal or axial
averaging of the interaction is involved: the kernel is the full pair
potential at the true three-dimensional distance, sampled on the grid.

The only approximation relative to ``electrostatics.pair_energy_direct`` is
the grid: weights are assigned to the neighbouring grid nodes by linear
(cloud-in-cell) interpolation in s, phi and r. The tests measure that error
against the direct double sum.

Units: nm, kT, rad.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import electrostatics as es
from .sites import Molecule, SiteType


@dataclasses.dataclass(frozen=True)
class ScanGrid:
    """Cylindrical grid shared by both molecules of a scan."""

    ds_nm: float = 0.2          # axial node spacing = stagger resolution
    n_phi: int = 24             # azimuthal nodes = facing-angle resolution
    n_z: int = 4096             # axial FFT length (periodic box n_z * ds)
    s_origin_nm: float = -10.0  # axial coordinate of node 0

    @property
    def facing_angles(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.n_phi) / self.n_phi


class StaggerScanner:
    """Evaluates sum_ij w_i w_j kernel(r_ij) on the (stagger, fa, fb) grid for
    one pair kernel, one axis separation and a fixed set of radial shells."""

    def __init__(self, kernel, radii_nm, axis_separation_nm: float,
                 grid: ScanGrid = ScanGrid()):
        self.grid = grid
        self.radii = np.atleast_1d(np.asarray(radii_nm, dtype=float))
        if np.any(np.diff(self.radii) <= 0):
            raise ValueError("shell radii must be strictly increasing")
        self.axis_separation_nm = float(axis_separation_nm)
        g = grid
        idx = np.arange(g.n_z)
        u = np.minimum(idx, g.n_z - idx) * g.ds_nm        # circular |lag|
        ang = g.facing_angles
        n_r = len(self.radii)
        # K~[k, shell_a, shell_b, m, n], frequency first so that a block of
        # frequencies is one contiguous, cache-sized slab (see correlate)
        self._kt = np.empty((g.n_z // 2 + 1, n_r, n_r, g.n_phi, g.n_phi))
        for ia, ra in enumerate(self.radii):
            for ib in range(ia, n_r):
                rho2 = es.transverse_distance_sq(ra, ang[:, None], self.radii[ib],
                                                 ang[None, :], self.axis_separation_nm)
                k = kernel(np.sqrt(u[None, None, :] ** 2 + rho2[:, :, None]))
                # K~[m, n, k] = sum K[a, b, u] e^{-i w u} e^{+i m a} e^{+i n b}.
                # K is even in u and invariant under (a, b) -> (-a, -b), so the
                # transform is real; only the real part is kept.
                kt = (np.fft.ifft2(np.fft.rfft(k, axis=2), axes=(0, 1)) * g.n_phi ** 2).real
                self._kt[:, ia, ib] = kt.transpose(2, 0, 1)
                # shells swapped <=> the two angles swapped
                self._kt[:, ib, ia] = kt.transpose(2, 1, 0)
        self._work = None

    # -- weights on the grid ------------------------------------------------
    def deposit(self, s, phi, r, w, smeared=None) -> np.ndarray:
        """Cloud-in-cell deposit of site weights: rho[shell, phi, z].

        ``smeared`` sites are spread uniformly over all azimuthal nodes."""
        g = self.grid
        s = np.asarray(s, float); phi = np.asarray(phi, float)
        r = np.asarray(r, float); w = np.asarray(w, float)
        smeared = np.zeros(len(s), bool) if smeared is None else np.asarray(smeared, bool)
        live = w != 0.0
        s, phi, r, w, smeared = s[live], phi[live], r[live], w[live], smeared[live]
        rho = np.zeros((len(self.radii), g.n_phi, g.n_z))
        if not len(s):
            return rho
        fz = (s - g.s_origin_nm) / g.ds_nm
        iz = np.floor(fz).astype(int)
        tz = fz - iz
        if iz.min() < 0 or iz.max() + 1 >= g.n_z // 2:
            raise ValueError("sites fall outside the half of the axial grid that "
                             "keeps periodic images apart")
        if len(self.radii) == 1:
            ir = np.zeros(len(s), int); tr = np.zeros(len(s))
        else:
            if r.min() < self.radii[0] - 1e-9 or r.max() > self.radii[-1] + 1e-9:
                raise ValueError("site radius outside the radial shells")
            ir = np.clip(np.searchsorted(self.radii, r, side="right") - 1,
                         0, len(self.radii) - 2)
            tr = np.clip((r - self.radii[ir]) / (self.radii[ir + 1] - self.radii[ir]),
                         0.0, 1.0)
        fp = np.mod(phi, 2.0 * np.pi) / (2.0 * np.pi) * g.n_phi
        ip = np.floor(fp).astype(int)
        tp = fp - ip
        ip %= g.n_phi
        pt = ~smeared
        for dr, wr in ((0, 1.0 - tr), (1, tr)):
            if len(self.radii) == 1 and dr == 1:
                continue
            for dz, wz in ((0, 1.0 - tz), (1, tz)):
                for dp, wp in ((0, 1.0 - tp), (1, tp)):
                    np.add.at(rho, (ir[pt] + dr, (ip[pt] + dp) % g.n_phi, iz[pt] + dz),
                              (w * wr * wz * wp)[pt])
                if smeared.any():
                    ring = np.zeros((len(self.radii), g.n_z))
                    np.add.at(ring, (ir[smeared] + dr, iz[smeared] + dz),
                              (w * wr * wz)[smeared] / g.n_phi)
                    rho += ring[:, None, :]
        return rho

    def deposit_molecule(self, mol: Molecule, w, position: str = "centroid",
                         antiparallel: bool = False) -> np.ndarray:
        s, phi, r = mol.positions(position)
        if antiparallel:
            s, phi = es.flip_antiparallel(s, phi, mol.helix_length_nm)
        return self.deposit(s, phi, r, w, mol.smeared)

    # -- transforms -----------------------------------------------------------
    def transform_a(self, rho) -> np.ndarray:
        """A^[k, shell, m] = sum rho[a, s] e^{-i w s} e^{-i m a}."""
        a = np.fft.fft(np.fft.rfft(rho, axis=2), axis=1)
        return np.ascontiguousarray(a.transpose(2, 0, 1))

    def transform_b(self, rho) -> np.ndarray:
        """B~[k, shell, n] = sum rho[b, s] e^{+i w s} e^{-i n b}."""
        b = np.conj(np.fft.ifft(np.fft.rfft(rho, axis=2), axis=1)) * self.grid.n_phi
        return np.ascontiguousarray(b.transpose(2, 0, 1))

    def correlate(self, a_hat, b_tilde) -> np.ndarray:
        """E[fa, fb, lag] for transformed weights of A and B. The result
        lives in a reused buffer: it is valid until the next call.

        Derivation (all sums over grid nodes, K the kernel table):
            E[D, fa, fb] = sum A[a, s] B[b, s'] K[s' + D - s, a - fa, b - fb]
        Taking the DFT over (fa, fb, D) and substituting the kernel's own
        arguments as summation variables factorises the sum into
            E^[m, n, k] = A^[m, k] * B~[n, k] * K~[m, n, k],
        summed over radial shell pairs. Lag index j means stagger j * ds for
        j < n_z / 2 and (j - n_z) * ds otherwise.

        The product and the two azimuthal inverse transforms are done in
        blocks of frequencies small enough to stay in cache: on whole arrays
        the scan is limited by memory bandwidth and several processes running
        scans side by side gain nothing.
        """
        g = self.grid
        acc, out = self._workspace()
        nk = self._kt.shape[0]
        for k0 in range(0, nk, self._block):
            sl = slice(k0, k0 + self._block)
            blk = np.einsum("kam,kbn,kabmn->kmn", a_hat[sl], b_tilde[sl], self._kt[sl])
            acc[:, :, sl] = np.fft.ifft2(blk, axes=(1, 2)).transpose(1, 2, 0)
        np.fft.irfft(acc, n=g.n_z, axis=2, out=out)
        return out

    _block = 32   # frequencies per block: 24 x 24 x 32 complex = 0.3 MB

    def _workspace(self):
        """Scratch arrays reused by :meth:`correlate`. The returned energy
        array is overwritten by the next call: copy what must be kept."""
        if self._work is None:
            g = self.grid
            self._work = (np.empty((g.n_phi, g.n_phi, g.n_z // 2 + 1), complex),
                          np.empty((g.n_phi, g.n_phi, g.n_z)))
        return self._work

    def scan(self, mol_a: Molecule, w_a, mol_b: Molecule, w_b,
             antiparallel: bool = False, position: str = "centroid") -> np.ndarray:
        """E[fa, fb, lag] for two molecules with per-site weights (a copy)."""
        a_hat = self.transform_a(self.deposit_molecule(mol_a, w_a, position))
        b_til = self.transform_b(self.deposit_molecule(mol_b, w_b, position,
                                                       antiparallel))
        return self.correlate(a_hat, b_til).copy()

    def lag_slice(self, lag_min_nm: float, lag_max_nm: float):
        """(index array into the lag axis, stagger values in nm) for the
        closed stagger range, in increasing order (negative lags wrap)."""
        g = self.grid
        j = np.arange(int(np.ceil(lag_min_nm / g.ds_nm - 1e-9)),
                      int(np.floor(lag_max_nm / g.ds_nm + 1e-9)) + 1)
        if np.abs(j).max() >= g.n_z // 2:
            raise ValueError("stagger range exceeds the axial grid")
        return j % g.n_z, j * g.ds_nm


def charged_shell_radii(mol: Molecule, n_shells: int, position: str = "centroid"):
    """Radial shells spanning every site that can ever carry a charge
    (anything but SiteType.OTHER), so that one scanner serves all
    modification patterns of the same molecule."""
    r = mol.positions(position)[2][mol.site_type != int(SiteType.OTHER)]
    return np.linspace(r.min(), r.max(), n_shells) if n_shells > 1 else np.array([r.mean()])


def residue_shell_radii(mol: Molecule, n_shells: int):
    """Radial shells spanning the side-chain centroids of all residues."""
    r = mol.r[mol.is_residue]
    return np.linspace(r.min(), r.max(), n_shells) if n_shells > 1 else np.array([r.mean()])


# ------------------------------------------------------- angle reductions ---
def reduce_over_angles(energy):
    """Collapse E[fa, fb, lag] (kT) over the two facing angles.

    Returns (e_min, f_boltz): the minimum over angles, and the free energy of
    a pair free to rotate at fixed stagger with a uniform prior on angles,
        F(lag) = -ln < exp(-E) >_{fa, fb}.
    F <= <E> and F >= e_min, with F -> e_min when one facing dominates.
    """
    flat = energy.reshape(-1, energy.shape[-1])
    e_min = flat.min(axis=0)
    f_boltz = e_min - np.log(np.mean(np.exp(-(flat - e_min)), axis=0))
    return e_min, f_boltz


# ---------------------------------------------------------------- analysis ---
def leading_minima(lags_nm, profile, n: int = 6, min_sep_nm: float = 15.0,
                   lag_min_nm: float = 5.0):
    """The ``n`` deepest local minima of a profile, as (position, value)
    pairs sorted by depth. Candidates are strict interior local minima beyond
    ``lag_min_nm`` (the in-register neighbourhood of zero stagger is
    excluded, as in the existing tools); they are taken deepest first and a
    candidate closer than ``min_sep_nm`` to one already taken is dropped."""
    lags = np.asarray(lags_nm); u = np.asarray(profile)
    cand = [i for i in range(1, len(u) - 1)
            if u[i] < u[i - 1] and u[i] <= u[i + 1] and lags[i] > lag_min_nm]
    cand.sort(key=lambda i: u[i])
    kept = []
    for i in cand:
        if all(abs(lags[i] - lags[j]) >= min_sep_nm for j in kept):
            kept.append(i)
        if len(kept) == n:
            break
    return [(float(lags[i]), float(u[i])) for i in kept]


def nearest_multiple(x_nm: float, period_nm: float):
    """(k, offset in nm) of the integer multiple k * period nearest to x."""
    k = int(round(x_nm / period_nm))
    return k, x_nm - k * period_nm


def _window_min(lags, u, centre, window):
    lo = np.searchsorted(lags, centre - window, side="left")
    hi = np.searchsorted(lags, centre + window, side="right")
    return u[lo:hi].min() if hi > lo else np.nan


def comb_scores(lags_nm, profile, period_nm: float, window_nm: float = 3.0,
                n_teeth: int = 4, phase_step_nm: float = 0.5,
                lag_min_nm: float = 5.0):
    """Depth of a comb of ``n_teeth`` staggers spaced by ``period_nm``, for
    every phase of the comb.

    A comb with offset p has its teeth at p + k * period, k = 0 .. n_teeth-1;
    its score is the mean over the teeth of the profile minimum within
    +-window_nm of each tooth. Offsets run over one full period starting at
    lag_min + window, so no tooth window reaches into the in-register
    neighbourhood of zero stagger that :func:`leading_minima` also excludes.
    The offset equal to the period is the comb at integer multiples
    1 .. n_teeth of the period. Returns (offsets, scores, index of the
    integer-multiple comb); a comb reaching beyond the profile scores NaN.
    """
    lags = np.asarray(lags_nm); u = np.asarray(profile)
    n_ph = max(int(round(period_nm / phase_step_nm)), 1)
    step = period_nm / n_ph
    first = lag_min_nm + window_nm
    i0 = int(np.ceil((period_nm - first) / step - 1e-9))   # steps below the period
    offsets = period_nm + step * (np.arange(n_ph) - i0)
    scores = np.empty(n_ph)
    for i, p in enumerate(offsets):
        centres = p + period_nm * np.arange(n_teeth)
        if centres[-1] + window_nm > lags[-1] or centres[0] - window_nm < lags[0]:
            scores[i] = np.nan
            continue
        scores[i] = np.mean([_window_min(lags, u, c, window_nm) for c in centres])
    return offsets, scores, i0


def comb_selectivity(lags_nm, profile, period_nm: float, window_nm: float = 3.0,
                     n_teeth: int = 4, phase_step_nm: float = 0.5,
                     lag_min_nm: float = 5.0) -> dict:
    """How much deeper the comb at integer multiples of ``period_nm`` is than
    a comb of the same period at any other phase.

    z      (mean over phases - score at integer multiples) / sd over phases;
           positive = the integer-multiple comb is deeper than a typical one.
           Dimensionless, so insensitive to the overall energy scale.
    rank   fraction of phases strictly deeper than the integer-multiple comb
           (0 = it is the deepest comb of this period).
    best_phase_nm  phase of the deepest comb relative to the integer
           multiples, folded to (-period/2, period/2] (0 = at the multiples).
    best_z z of that deepest comb.
    """
    nan = {"z": np.nan, "rank": np.nan, "best_phase_nm": np.nan, "best_z": np.nan}
    if period_nm <= lag_min_nm + window_nm:
        return nan
    offsets, scores, i_int = comb_scores(lags_nm, profile, period_nm, window_nm,
                                         n_teeth, phase_step_nm, lag_min_nm)
    ok = np.isfinite(scores)
    if ok.sum() < 3 or not ok[i_int]:
        return nan
    mean, sd = scores[ok].mean(), scores[ok].std()
    if sd == 0.0:
        return {"z": 0.0, "rank": 0.0, "best_phase_nm": 0.0, "best_z": 0.0}
    i_best = int(np.nanargmin(scores))
    ph = (offsets[i_best] - period_nm + period_nm / 2) % period_nm - period_nm / 2
    if np.isclose(ph, -period_nm / 2):
        ph = period_nm / 2
    return {"z": float((mean - scores[i_int]) / sd),
            "rank": float(np.mean(scores[ok] < scores[i_int])),
            "best_phase_nm": float(ph),
            "best_z": float((mean - scores[i_best]) / sd)}


def profile_autocorrelation(lags_nm, profile, shifts_nm, lag_min_nm: float = 5.0):
    """Pearson correlation of the profile with itself shifted by each of
    ``shifts_nm`` (over the overlapping part, staggers beyond lag_min only).
    A profile that repeats with period P has maxima at P, 2P, ...; nothing is
    assumed about where within a period its minima sit. The molecule is only
    4.3 D long, so shifts beyond about 2 D rest on little overlap."""
    lags = np.asarray(lags_nm); u = np.asarray(profile, dtype=float)[lags > lag_min_nm]
    step = float(lags[1] - lags[0])
    out = np.full(len(shifts_nm), np.nan)
    for i, shift in enumerate(shifts_nm):
        n = int(round(shift / step))
        if n < 1 or len(u) - n < 10:
            continue
        a, b = u[:-n] - u[:-n].mean(), u[n:] - u[n:].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out[i] = (a * b).sum() / den if den > 0 else np.nan
    return out


def dominant_period(shifts_nm, autocorrelation, tolerance: float = 0.9):
    """(period, correlation) read off an autocorrelation curve: the smallest
    shift that is a local maximum reaching ``tolerance`` times the highest
    local maximum. Taking the smallest such shift prefers a fundamental over
    its multiples, which correlate about as well. (nan, nan) if the curve has
    no positive local maximum."""
    r = np.asarray(autocorrelation, dtype=float)
    peaks = [i for i in range(1, len(r) - 1)
             if np.isfinite(r[i - 1:i + 2]).all() and r[i] >= r[i - 1] and r[i] > r[i + 1]]
    if not peaks or max(r[i] for i in peaks) <= 0.0:
        return float("nan"), float("nan")
    top = max(r[i] for i in peaks)
    i = next(i for i in peaks if r[i] >= tolerance * top)
    return float(shifts_nm[i]), float(r[i])


def near_multiple_fraction(lag_max_nm: float, period_nm: float, window_nm: float = 3.0,
                           lag_min_nm: float = 5.0) -> float:
    """Fraction of the stagger range (lag_min, lag_max] lying within
    +-window of a non-zero integer multiple of the period: the chance level
    for a stagger drawn uniformly at random."""
    k = np.arange(1, int(np.floor((lag_max_nm + window_nm) / period_nm)) + 1)
    lo = np.maximum(k * period_nm - window_nm, lag_min_nm)
    hi = np.minimum(k * period_nm + window_nm, lag_max_nm)
    return float(np.clip(hi - lo, 0.0, None).sum() / (lag_max_nm - lag_min_nm))


def boltzmann_multiple_fraction(lags_nm, free_energy, period_nm: float,
                                window_nm: float = 3.0, lag_min_nm: float = 5.0) -> float:
    """Equilibrium probability that a pair confined to the scanned stagger
    range (beyond lag_min) sits within +-window of a non-zero integer
    multiple of the period: sum exp(-F) over those staggers / sum over all.
    Compare with :func:`near_multiple_fraction`."""
    lags = np.asarray(lags_nm); f = np.asarray(free_energy, dtype=float)
    use = lags > lag_min_nm
    w = np.exp(-(f[use] - f[use].min()))
    k = np.round(lags[use] / period_nm)
    near = (k >= 1) & (np.abs(lags[use] - k * period_nm) <= window_nm)
    return float(w[near].sum() / w.sum())


# ------------------------------------------------ charge-pattern statistics ---
def axial_profile(mol: Molecule, weights, bin_nm: float | None = None,
                  position: str = "centroid"):
    """Sum of per-site weights in axial bins of one residue rise (default):
    the molecule's pattern projected on its axis, all three chains together.
    Returns (bin centres nm, summed weight)."""
    step = mol.rise_nm if bin_nm is None else bin_nm
    s = mol.positions(position)[0]
    lo = np.floor(s.min() / step)
    idx = (np.floor(s / step) - lo).astype(int)
    prof = np.bincount(idx, weights=np.asarray(weights, float), minlength=idx.max() + 1)
    return (lo + np.arange(len(prof)) + 0.5) * step, prof


def lagged_correlation(x, y, lag: int) -> float:
    """Pearson correlation of x[n] with y[n + lag] over their overlap
    (lag >= 0, x and y on the same axial bins). With y = x this is the
    autocorrelation of one pattern; with two molecules it is the correlation
    of the charges that face each other at that stagger."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = min(len(x), len(y)) - lag
    if n < 3:
        return float("nan")
    a, b = x[:n], y[lag:lag + n]
    raw = (a * a).sum(), (b * b).sum()
    a = a - a.mean(); b = b - b.mean()
    va, vb = (a * a).sum(), (b * b).sum()
    # a constant window is uncorrelated with anything; its variance is zero
    # only up to the round-off of its mean
    if va <= 1e-20 * raw[0] or vb <= 1e-20 * raw[1]:
        return float("nan")
    return float((a * b).sum() / np.sqrt(va * vb))


# ------------------------------------------------------------ pair model ---
@dataclasses.dataclass(frozen=True)
class PairSettings:
    """Geometry and interaction parameters of a two-molecule scan."""

    # Gap between the nominal molecular surfaces, nm. Axis separation is
    # 2 * MOLECULE_RADIUS_NM + gap = 1.65 nm by default, the contact distance
    # the README's measured radial envelope was refitted to ("attR0 1.65 nm").
    surface_separation_nm: float = 0.15
    core_nm: float = es.SOFT_CORE_NM
    eps_hydrophobic: float = es.EPS_HYDROPHOBIC
    hydrophobic_sigma_nm: float = es.HYDROPHOBIC_SIGMA_NM
    position: str = "centroid"      # where a site's charge sits
    n_shell_charge: int = 2
    n_shell_hydro: int = 4
    lag_max_nm: float = 300.0
    grid: ScanGrid = ScanGrid()

    @property
    def axis_separation_nm(self) -> float:
        return 2.0 * es.MOLECULE_RADIUS_NM + self.surface_separation_nm


# Reductions of E(stagger, fa, fb) returned by PairModel.profiles, in order.
REDUCTIONS = ("tot_min", "tot_boltz", "tot_mean", "el_min", "el_boltz")


class PairModel:
    """Stagger profiles of two modification patterns of one parent molecule.

    The electrostatic scanners (one per ionic strength) and the hydrophobic
    energy table are built once from the native molecule. Modification leaves
    site positions and hydrophobic weights unchanged, so both serve every
    pattern; only the charges are recomputed, pattern by pattern.

    Parallel scans cover staggers 0 .. lag_max; antiparallel scans cover
    -lag_max .. +lag_max (no symmetry relates the two signs there).
    """

    def __init__(self, native: Molecule, ionic_strengths_M,
                 settings: PairSettings = PairSettings(),
                 orientations=("parallel", "antiparallel")):
        self.native = native
        self.settings = settings
        self.orientations = tuple(orientations)
        st_ = settings
        d = st_.axis_separation_nm
        radii = charged_shell_radii(native, st_.n_shell_charge, st_.position)
        self.el = {}
        for ionic in ionic_strengths_M:
            lam = es.debye_length_nm(ionic)
            self.el[float(ionic)] = StaggerScanner(
                lambda r, lam=lam: es.screened_coulomb(r, lam, st_.core_nm),
                radii, d, st_.grid)
        any_scanner = next(iter(self.el.values()))
        # The grid is periodic along the axis. With both molecules deposited
        # on nodes [0, extent) and staggers up to lag_max, the nearest
        # periodic image of the partner is n_z - extent - lag_max nodes away;
        # that must exceed the reach of the pair kernels (30 Debye lengths
        # is exp(-30); the hydrophobic kernel is far shorter).
        g = st_.grid
        s_all = np.concatenate([native.s, es.flip_antiparallel(
            native.s, native.phi, native.helix_length_nm)[0]])
        extent = (s_all.max() - g.s_origin_nm) / g.ds_nm + 2
        reach = 30.0 * max(es.debye_length_nm(i) for i in ionic_strengths_M) / g.ds_nm
        if s_all.min() < g.s_origin_nm or extent + st_.lag_max_nm / g.ds_nm + reach > g.n_z:
            raise ValueError("axial grid too short for this molecule, stagger range "
                             "and Debye length: increase ScanGrid.n_z")
        self.lag_idx, self.lags = {}, {}
        for o in self.orientations:
            lo = 0.0 if o == "parallel" else -st_.lag_max_nm
            self.lag_idx[o], self.lags[o] = any_scanner.lag_slice(lo, st_.lag_max_nm)
        self.e_hy = {}
        if st_.eps_hydrophobic != 0.0:
            hy = StaggerScanner(
                lambda r: es.hydrophobic_contact(r, st_.hydrophobic_sigma_nm),
                residue_shell_radii(native, st_.n_shell_hydro), d, st_.grid)
            h = native.centered_hydrophobicity()
            for o in self.orientations:
                e = hy.scan(native, h, native, h, antiparallel=(o == "antiparallel"))
                self.e_hy[o] = -st_.eps_hydrophobic * e[:, :, self.lag_idx[o]]

    def profiles(self, mol_a: Molecule, mol_b: Molecule, pH_values,
                 charge_model=es.henderson_hasselbalch) -> dict:
        """{(pH, ionic strength, orientation): array [len(REDUCTIONS), n_lag]}
        in kT, for one concrete pair of patterns."""
        pos = self.settings.position
        out = {}
        for pH in pH_values:
            a_hat, b_til = {}, {}
            for ionic, sc in self.el.items():
                env = es.Environment(pH=pH, ionic_strength_M=ionic)
                q_a, q_b = charge_model(mol_a, env), charge_model(mol_b, env)
                # the deposits depend on the environment only through the
                # charges; reuse them when the charge model ignores salt
                key = (q_a.tobytes(), q_b.tobytes())
                if key not in a_hat:
                    a_hat[key] = sc.transform_a(sc.deposit_molecule(mol_a, q_a, pos))
                    b_til[key] = {
                        o: sc.transform_b(sc.deposit_molecule(
                            mol_b, q_b, pos, antiparallel=(o == "antiparallel")))
                        for o in self.orientations}
                for o in self.orientations:
                    e_el = sc.correlate(a_hat[key], b_til[key][o])[:, :, self.lag_idx[o]]
                    e_tot = e_el + self.e_hy[o] if o in self.e_hy else e_el
                    t_min, t_boltz = reduce_over_angles(e_tot)
                    e_min, e_boltz = reduce_over_angles(e_el)
                    out[(float(pH), ionic, o)] = np.stack(
                        [t_min, t_boltz, e_tot.mean(axis=(0, 1)), e_min, e_boltz])
        return out


def native_gate(lags_nm, tot_min, tot_mean, period_nm: float,
                tolerance_periods: float = 0.05) -> dict:
    """Sanity gate on the unmodified control (parallel pair, pH 7.4).

    Two conditions, both required:
      1. the deepest minimum of the min-over-facing-angles profile lies
         within ``tolerance_periods`` of a non-zero integer multiple of D;
      2. the three deepest minima of the azimuth-averaged profile all do.
    Condition 2 is the azimuth-blind statement the README makes for the 1D
    sequence profile ("1.00 D, 1.99 D, 3.00 D"); condition 1 is what the
    README reports for the azimuth-resolved table (deepest minimum at
    1.01 D, competitors close). The five deepest minima of both profiles are
    returned so that non-D competitors are visible, not hidden by the gate.
    """
    def describe(profile, n):
        out = []
        for x, v in leading_minima(lags_nm, profile, n=n):
            k, off = nearest_multiple(x, period_nm)
            out.append({"stagger_nm": x, "stagger_D": x / period_nm, "energy_kT": v,
                        "nearest_multiple": k, "offset_D": off / period_nm,
                        "near_multiple": bool(k >= 1 and abs(off) <= tolerance_periods * period_nm)})
        return out

    mins = describe(tot_min, 5)
    means = describe(tot_mean, 5)
    c1 = bool(mins and mins[0]["near_multiple"])
    c2 = bool(len(means) >= 3 and all(m["near_multiple"] for m in means[:3]))
    return {"passed": c1 and c2, "deepest_min_over_angles_near_multiple": c1,
            "azimuth_mean_top3_near_multiples": c2,
            "tolerance_D": tolerance_periods,
            "min_over_angles_minima": mins, "azimuth_mean_minima": means,
            "min_over_angles_top5_near_multiples": int(sum(m["near_multiple"] for m in mins))}


def lagged_correlation_rows(x, y, lags) -> np.ndarray:
    """:func:`lagged_correlation` of row k of x with row k of y, for every
    row and every lag at once: array [n_rows, n_lags]. Uses running sums, so
    no array of the size of the input is allocated per lag."""
    x = np.atleast_2d(np.asarray(x, float)); y = np.atleast_2d(np.asarray(y, float))
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    size = x.shape[1]
    zero = np.zeros((x.shape[0], 1))
    cx, cxx = (np.concatenate([zero, np.cumsum(a, axis=1)], axis=1) for a in (x, x * x))
    cy, cyy = (np.concatenate([zero, np.cumsum(a, axis=1)], axis=1) for a in (y, y * y))
    out = np.full((x.shape[0], len(lags)), np.nan)
    for i, lag in enumerate(lags):
        lag = int(lag)
        n = size - lag
        if n < 3:
            continue
        sa, saa = cx[:, n], cxx[:, n]                       # x[:n]
        sb, sbb = cy[:, size] - cy[:, lag], cyy[:, size] - cyy[:, lag]   # y[lag:]
        sab = np.einsum("ij,ij->i", x[:, :n], y[:, lag:])
        va, vb = saa - sa * sa / n, sbb - sb * sb / n
        # a constant window has zero variance only up to the round-off of the
        # running sums (about n * 1e-16 of the sum of squares)
        ok = (va > 1e-10 * saa) & (vb > 1e-10 * sbb)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, i] = np.where(ok, (sab - sa * sb / n) / np.sqrt(np.abs(va * vb)), np.nan)
    return out
