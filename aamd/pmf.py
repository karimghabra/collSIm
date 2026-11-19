"""Potential of mean force from umbrella windows.

Estimators
----------
* ``wham``: the weighted histogram analysis method (Kumar et al. 1992),
  self-consistent iteration on a fixed grid of bins. Pure numpy.
* ``mbar_pmf``: MBAR (Shirts & Chodera 2008) through pymbar, with the PMF
  read off as the weighted histogram of the unbiased state (pymbar's FES with
  a histogram). Used as the reported estimator; WHAM is the cross-check.

Both take, per window k, a time series x_k(t) of the coordinate and a bias
u_k(x) = 0.5 k_k (x - r0_k)^2 in kT.

Errors
------
* block error: production is cut into B consecutive blocks; the PMF is
  computed from block b of every window, for each b; the error is the
  standard deviation of the B profiles divided by sqrt(B);
* convergence: the PMF of the first and of the second half of production,
  separately.

Before all of this each window's leading non-stationary part is dropped
(``detect_equilibration``: the start that maximises the number of
independent samples, at most half the window); the blocks and halves are of
what remains.

A profile is given relative to its mean over a reference interval at large
separation (``ref``), the zero of interaction.

Units: nm, kT.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

KB_KJ = 0.0083144626           # kJ/mol/K
TEMPERATURE = 298.15
KT_KJ = KB_KJ * TEMPERATURE    # 2.479 kJ/mol


@dataclasses.dataclass
class Windows:
    centres: np.ndarray          # nm
    k_kt: np.ndarray             # spring constants, kT/nm^2
    series: list                 # per window, 1-D array of x (nm)
    times: list | None = None    # per window, times (ps)

    def bias(self, x):
        """u[k, n] = bias of window k at sample x[n], kT."""
        x = np.asarray(x)
        return 0.5 * self.k_kt[:, None] * (x[None, :] - self.centres[:, None]) ** 2

    def subset(self, frac_lo, frac_hi):
        """The [frac_lo, frac_hi) fraction of every window's series."""
        out = []
        for s in self.series:
            n = len(s)
            out.append(s[int(frac_lo * n):int(frac_hi * n)])
        return Windows(self.centres, self.k_kt, out)


def statistical_inefficiency(x) -> float:
    """g = 1 + 2 sum_t (1 - t/N) C(t), summed until C first drops below 0
    (the estimator of pymbar.timeseries.statistical_inefficiency)."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4:
        return 1.0
    dx = x - x.mean()
    var = dx @ dx / n
    if var == 0:
        return 1.0
    g = 1.0
    for t in range(1, n - 1):
        c = (dx[:-t] @ dx[t:]) / ((n - t) * var)
        if c <= 0:
            break
        g += 2.0 * c * (1.0 - t / n)
    return max(1.0, g)


def detect_equilibration(x, step=None, max_frac=0.5):
    """Start t0 of the stationary part of ``x``: the t0 (on a grid of
    ``step`` samples, at most ``max_frac`` of the series) that maximises the
    number of independent samples (N - t0) / g(x[t0:]) (Chodera, JCTC 12,
    1799 (2016))."""
    x = np.asarray(x, float)
    n = len(x)
    step = step or max(1, n // 40)
    best, best_t0 = -1.0, 0
    for t0 in range(0, int(max_frac * n) + 1, step):
        neff = (n - t0) / statistical_inefficiency(x[t0:])
        if neff > best:
            best, best_t0 = neff, t0
    return best_t0


def subsample(x, g=None):
    g = statistical_inefficiency(x) if g is None else g
    step = max(1, int(np.ceil(g)))
    return np.asarray(x)[::step], g


def wham(win: Windows, edges, tol=1e-10, max_iter=100000):
    """WHAM on bins ``edges``. Returns (bin centres, F in kT, minimum 0, NaN
    in empty bins), window free energies f_k."""
    edges = np.asarray(edges)
    xc = 0.5 * (edges[1:] + edges[:-1])
    counts = np.array([np.histogram(s, edges)[0] for s in win.series], float)
    N = counts.sum(axis=1)
    H = counts.sum(axis=0)
    # bias of each bin averaged over the bin (Boltzmann average on a fine
    # sub-grid), not taken at the bin centre: with stiff springs the bias
    # changes by several kT across a bin of a far window
    sub = np.linspace(0.0, 1.0, 9)
    xs = edges[:-1, None] + (edges[1:] - edges[:-1])[:, None] * sub[None, :]     # [M, 9]
    us = win.bias(xs.ravel()).reshape(len(win.series), len(xc), len(sub))
    umin = us.min(axis=2)
    ub = umin - np.log(np.mean(np.exp(-(us - umin[:, :, None])), axis=2))       # [K, M]
    f = np.zeros(len(win.series))
    for _ in range(max_iter):
        # p(x) = H / sum_k N_k exp(f_k - u_k)
        logden = np.log(np.sum(N[:, None] * np.exp(f[:, None] - ub), axis=0) + 1e-300)
        with np.errstate(divide="ignore"):
            logp = np.where(H > 0, np.log(H) - logden, -np.inf)
        f_new = -np.log(np.sum(np.exp(logp[None, :] - ub), axis=1))
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    F = -logp
    F = np.where(np.isfinite(F), F, np.nan)
    return xc, F - np.nanmin(F), f


def mbar_pmf(win: Windows, edges):
    """MBAR + histogram FES (pymbar). Returns (bin centres, F kT, dF kT)."""
    import pymbar

    x = np.concatenate(win.series)
    n_k = np.array([len(s) for s in win.series])
    u_kn = win.bias(x)
    fes = pymbar.FES(u_kn, n_k, verbose=False)
    edges = np.asarray(edges)
    fes.generate_fes(np.zeros_like(x), x, fes_type="histogram",
                     histogram_parameters={"bin_edges": edges})
    xc = 0.5 * (edges[1:] + edges[:-1])
    # pymbar only labels bins that hold samples; ask for those, NaN elsewhere
    occupied = np.histogram(x, edges)[0] > 0
    F, dF = np.full(len(xc), np.nan), np.full(len(xc), np.nan)
    res = fes.get_fes(xc[occupied], reference_point="from-lowest", uncertainty_method="analytical")
    F[occupied], dF[occupied] = res["f_i"], res["df_i"]
    return xc, F, dF


def reference_shift(xc, F, ref):
    m = (xc >= ref[0]) & (xc <= ref[1]) & np.isfinite(F)
    if not m.any():
        raise ValueError("no data in the reference interval")
    return F - np.mean(F[m])


@dataclasses.dataclass
class PMFResult:
    x: np.ndarray
    F: np.ndarray            # kT, zero = mean over the reference interval
    dF_block: np.ndarray     # kT, block standard error
    F_first: np.ndarray      # kT, first half of production
    F_second: np.ndarray
    F_wham: np.ndarray
    g: list                  # statistical inefficiency per window
    n_eff: list
    ref: tuple
    n_discarded: list = dataclasses.field(default_factory=list)   # leading samples dropped per window

    def minimum(self):
        k = int(np.nanargmin(self.F))
        return float(self.x[k]), float(self.F[k]), float(self.dF_block[k])

    def to_json(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(self).items()}
        return d


def analyse(win: Windows, edges, ref=(2.55, 2.80), n_blocks=4, estimator=mbar_pmf,
            equilibrate=True, stride=5) -> PMFResult:
    """Full analysis: drop each window's non-stationary start, then the MBAR
    profile, WHAM check, block error and halves.

    The estimators see every ``stride``-th sample, not only the uncorrelated
    ones: correlated samples leave the estimate unbiased and fill the
    histogram bins; the uncertainty comes from the blocks and halves, whose
    spread already reflects the correlation. ``n_eff`` (samples / g) is
    reported for each window."""
    t0s = [detect_equilibration(s) if equilibrate else 0 for s in win.series]
    win = Windows(win.centres, win.k_kt, [np.asarray(s)[t:] for s, t in zip(win.series, t0s)])
    gs = [statistical_inefficiency(s) for s in win.series]

    def thin(w):
        return Windows(w.centres, w.k_kt, [np.asarray(s)[::stride] for s in w.series])

    ws = thin(win)
    xc, F, _ = estimator(ws, edges)
    F = reference_shift(xc, F, ref)
    _, Fw, _ = wham(ws, edges)
    Fw = reference_shift(xc, Fw, ref)
    blocks = []
    for b in range(n_blocks):
        _, Fb, _ = estimator(thin(win.subset(b / n_blocks, (b + 1) / n_blocks)), edges)
        blocks.append(reference_shift(xc, Fb, ref))
    blocks = np.array(blocks)
    with np.errstate(invalid="ignore"):
        dF = np.nanstd(blocks, axis=0, ddof=1) / np.sqrt(n_blocks)
    halves = []
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        _, Fh, _ = estimator(thin(win.subset(lo, hi)), edges)
        halves.append(reference_shift(xc, Fh, ref))
    return PMFResult(xc, F, dF, halves[0], halves[1], Fw, [float(g) for g in gs],
                     [int(len(s) / g) for s, g in zip(win.series, gs)], tuple(ref), [int(t) for t in t0s])


# ------------------------------------------------------------- reading runs ---
def window_dirs(sysdir: Path, centres):
    """(centre, directory) of every umbrella run of a system, sorted by centre:
    the regular windows wNN (centre from ``centres``) and any extra run cR.RRR
    (centre in its name; the same spring and protocol, e.g. the inner window
    at 1.3 nm added when the minimum turned out to sit at 1.4 nm)."""
    sysdir = Path(sysdir)
    out = [(float(c), sysdir / f"w{i:02d}") for i, c in enumerate(centres)]
    for d in sorted(sysdir.glob("c[0-9]*")):
        try:
            out.append((float(d.name[1:]), d))
        except ValueError:
            pass
    return sorted(out)


def load_windows(sysdir: Path, centres, k_kj_nm2, column=1) -> Windows:
    """Read every umbrella run of a system that has data (cv_*.npy, column 1 = Delta x)."""
    series, times, cs, ks = [], [], [], []
    for c, d in window_dirs(sysdir, centres):
        files = sorted(d.glob("cv_*.npy"))
        if not files:
            continue
        a = np.concatenate([np.load(f) for f in files])
        series.append(a[:, column])
        times.append(a[:, 0])
        cs.append(c)
        ks.append(k_kj_nm2 / KT_KJ)
    return Windows(np.array(cs), np.array(ks), series, times)


def save(result: PMFResult, path: Path, meta: dict | None = None):
    d = result.to_json()
    if meta:
        d["meta"] = meta
    Path(path).write_text(json.dumps(d, indent=1))
