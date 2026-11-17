"""PMF estimators on synthetic umbrella data drawn from a known profile."""
import numpy as np
import pytest

from aamd import pmf

GRID = np.linspace(1.0, 3.0, 4001)


def true_F(x):
    return -4.0 * np.exp(-(x - 1.55) ** 2 / (2 * 0.08 ** 2)) + 2.0 * np.exp(-(x - 1.85) ** 2 / (2 * 0.06 ** 2))


def synthetic(rng, n=4000, k_kj=1000.0, centres=np.round(np.arange(1.2, 2.81, 0.1), 2)):
    k = k_kj / pmf.KT_KJ
    series = []
    for c in centres:
        w = np.exp(-true_F(GRID) - 0.5 * k * (GRID - c) ** 2)
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        series.append(np.interp(rng.random(n), cdf, GRID))
    return pmf.Windows(np.array(centres), np.full(len(centres), k), series)


@pytest.mark.parametrize("estimator", ["wham", "mbar"])
def test_estimators_recover_the_profile(estimator):
    rng = np.random.default_rng(7)
    win = synthetic(rng)
    edges = np.arange(1.15, 2.86, 0.025)
    if estimator == "wham":
        x, F, _ = pmf.wham(win, edges)
    else:
        x, F, _ = pmf.mbar_pmf(win, edges)
    F = pmf.reference_shift(x, F, (2.55, 2.8))
    ref = true_F(x) - np.mean(true_F(x[(x >= 2.55) & (x <= 2.8)]))
    m = (x > 1.2) & (x < 2.8)
    assert np.nanmax(np.abs(F[m] - ref[m])) < 0.3       # sampling noise + bin discretisation
    assert x[np.nanargmin(F)] == pytest.approx(1.55, abs=0.03)


def test_analyse_errors_and_halves_are_consistent():
    rng = np.random.default_rng(8)
    win = synthetic(rng, n=2000)
    res = pmf.analyse(win, np.arange(1.15, 2.86, 0.02))
    m = np.isfinite(res.F) & (res.x > 1.25) & (res.x < 2.75)
    assert np.all(res.dF_block[m] < 0.5)
    assert np.nanmax(np.abs(res.F_first[m] - res.F_second[m])) < 1.0
    assert np.nanmax(np.abs(res.F[m] - res.F_wham[m])) < 0.2
    xmin, fmin, _ = res.minimum()
    assert xmin == pytest.approx(1.55, abs=0.05)
    assert fmin == pytest.approx(-4.0, abs=0.4)


def test_statistical_inefficiency_of_ar1():
    rng = np.random.default_rng(9)
    phi = 0.8
    x = np.zeros(200000)
    e = rng.normal(size=x.size)
    for t in range(1, x.size):
        x[t] = phi * x[t - 1] + e[t]
    g = pmf.statistical_inefficiency(x[:20000])
    assert g == pytest.approx((1 + phi) / (1 - phi), rel=0.2)
    assert pmf.statistical_inefficiency(rng.normal(size=5000)) < 1.2
