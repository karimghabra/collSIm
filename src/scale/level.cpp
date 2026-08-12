#include "level.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace scale {

const char* provenanceName(Provenance p) {
    switch (p) {
        case Provenance::Measured:   return "measured";
        case Provenance::Derived:    return "derived";
        case Provenance::Literature: return "literature";
        default:                     return "ASSUMED";
    }
}

double checkGradient(const Level& lv, const double* q, double h) {
    const int n = lv.dim();
    std::vector<double> qq(q, q + n), g(n);
    lv.gradient(q, g.data());
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x0 = qq[i];
        // Scale the step to the coordinate so a coordinate of order 100 nm and
        // one of order 0.01 rad are both probed sensibly.
        const double hi = h * (1.0 + std::fabs(x0));
        qq[i] = x0 + hi; const double fp = lv.energy(qq.data());
        qq[i] = x0 - hi; const double fm = lv.energy(qq.data());
        qq[i] = x0;
        const double num = (fp - fm) / (2.0 * hi);
        const double scale = std::fabs(num) + std::fabs(g[i]) + 1e-9;
        const double rel = std::fabs(num - g[i]) / scale;
        if (rel > worst) worst = rel;
    }
    return worst;
}

bool hasAssumed(const Level& lv) {
    for (const Param& p : lv.provenance())
        if (p.src == Provenance::Assumed) return true;
    return false;
}

std::string provenanceReport(const Level& lv) {
    std::string out;
    char line[512];
    std::snprintf(line, sizeof line, "# rung: %s  (dim %d)\n", lv.name(), lv.dim());
    out += line;
    for (const Param& p : lv.provenance()) {
        std::snprintf(line, sizeof line, "#   %-22s %14.6g  [%-10s] %s\n",
                      p.name, p.value, provenanceName(p.src), p.note);
        out += line;
    }
    if (hasAssumed(lv))
        out += "# EXPLORATORY: this rung contains ASSUMED parameters. Results are\n"
               "# not quantitative and must not be reported as measured kinetics.\n";
    return out;
}

}  // namespace scale
