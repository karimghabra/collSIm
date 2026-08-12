#include "kramers.h"

#include <algorithm>
#include <cmath>

namespace scale {

namespace {
// Trapezoid weight for sample i of a range [lo,hi] inclusive.
inline double trapW(int i, int lo, int hi) {
    return (i == lo || i == hi) ? 0.5 : 1.0;
}
}  // namespace

double mfpt(const Profile& p, int ia, int ib) {
    const int n = int(p.F.size());
    if (n < 3 || ia < 0 || ib >= n || ia >= ib) return 0.0;

    // Reference energy: the minimum over the reactant side. Factoring it out
    // cancels between the two integrals and keeps the inner one bounded by 1,
    // so a 30 kT barrier does not overflow before it is ever exponentiated.
    double Fref = p.F[ia];
    for (int i = 0; i <= ia; ++i) Fref = std::min(Fref, p.F[i]);

    // Running inner integral: P(y) = integral_0^y exp(-(F - Fref)) dz, with the
    // reflecting boundary at index 0.
    double inner = 0.0;
    double tau = 0.0;
    double prevZ = 0.0;
    for (int y = 0; y <= ib; ++y) {
        const double z = std::exp(-(p.F[y] - Fref));
        if (y > 0) inner += 0.5 * (z + prevZ) * p.ds;
        prevZ = z;
        if (y >= ia) {
            const double w = trapW(y, ia, ib) * p.ds;
            tau += w * std::exp(p.F[y] - Fref) / p.diff(y) * inner;
        }
    }
    return tau;
}

double rateMFPT(const Profile& p, int ia, int ib) {
    const double t = mfpt(p, ia, ib);
    return t > 0 ? 1.0 / t : 0.0;
}

double kramersParabolic(const Profile& p, int iMin, int iBarrier) {
    const int n = int(p.F.size());
    if (iMin <= 0 || iMin >= n - 1 || iBarrier <= 0 || iBarrier >= n - 1) return 0.0;
    const double ds2 = p.ds * p.ds;
    const double cMin = (p.F[iMin + 1] - 2.0 * p.F[iMin] + p.F[iMin - 1]) / ds2;
    const double cBar = (p.F[iBarrier + 1] - 2.0 * p.F[iBarrier] + p.F[iBarrier - 1]) / ds2;
    if (cMin <= 0 || cBar >= 0) return 0.0;   // not a well / not a barrier
    const double dG = p.F[iBarrier] - p.F[iMin];
    return p.diff(iBarrier) * std::sqrt(cMin * (-cBar)) / (2.0 * 3.14159265358979323846) *
           std::exp(-dG);
}

double basinRatio(const Profile& p, int aLo, int aHi, int bLo, int bHi) {
    const int n = int(p.F.size());
    auto clampi = [&](int i) { return std::max(0, std::min(n - 1, i)); };
    aLo = clampi(aLo); aHi = clampi(aHi); bLo = clampi(bLo); bHi = clampi(bHi);
    double Fref = p.F[aLo];
    for (int i = aLo; i <= aHi; ++i) Fref = std::min(Fref, p.F[i]);
    for (int i = bLo; i <= bHi; ++i) Fref = std::min(Fref, p.F[i]);
    double za = 0.0, zb = 0.0;
    for (int i = aLo; i <= aHi; ++i) za += trapW(i, aLo, aHi) * std::exp(-(p.F[i] - Fref));
    for (int i = bLo; i <= bHi; ++i) zb += trapW(i, bLo, bHi) * std::exp(-(p.F[i] - Fref));
    return za > 0 ? zb / za : 0.0;
}

}  // namespace scale
