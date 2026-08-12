#include "microfibril.h"

#include <cmath>

namespace scale {

namespace {
struct V3 {
    double x, y, z;
    V3 operator-(const V3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    double dot(const V3& o) const { return x * o.x + y * o.y + z * o.z; }
    double len() const { return std::sqrt(x * x + y * y + z * z); }
};
inline V3 get(const double* q, int i) { return {q[i], q[i + 1], q[i + 2]}; }
inline void add(double* g, int i, const V3& v) { g[i] += v.x; g[i + 1] += v.y; g[i + 2] += v.z; }

// Displacement from the nearest registry minimum. Only the nearest period is
// kept: with sigma = 1.5 nm and D = 66.9 nm the next well contributes
// exp(-248), so the cusp this leaves at the half-period is below double
// precision and the potential is smooth for every purpose.
inline double wrapD(double d, double D) { return d - D * std::floor(d / D + 0.5); }
}  // namespace

Microfibril::Microfibril(const MicrofibrilParams& p) : p_(p) {
    nSlip_ = p_.nStrands - 1;
    if (nSlip_ < 0) nSlip_ = 0;
    dim_ = p_.nNodes * stride();
    diff_.assign(dim_, 0.0);
    for (int j = 0; j < p_.nNodes; ++j) {
        for (int c = 0; c < 3; ++c) diff_[posOf(j) + c] = p_.dTrans;
        diff_[twistOf(j)] = p_.dTwist;
        for (int m = 0; m < nSlip_; ++m) diff_[slipOf(j, m)] = p_.dSlip;
    }
}

void Microfibril::ideal(std::vector<double>& q) const {
    q.assign(dim_, 0.0);
    for (int j = 0; j < p_.nNodes; ++j) {
        q[posOf(j) + 0] = 0.0;
        q[posOf(j) + 1] = 0.0;
        q[posOf(j) + 2] = p_.D * j;
        q[twistOf(j)] = p_.psi0 * p_.D * j;
        for (int m = 0; m < nSlip_; ++m) q[slipOf(j, m)] = 0.0;
    }
}

double Microfibril::slipCurvature() const {
    // V = A[1 - exp(-w^2/2s^2)]  ->  V''(0) = A / s^2
    return p_.slipDepth / (p_.slipWidth * p_.slipWidth);
}

double Microfibril::energy(const double* q) const {
    const int N = p_.nNodes;
    const double s2 = p_.slipWidth * p_.slipWidth;
    double F = 0.0;

    // stretch + bend along the centerline
    for (int j = 0; j + 1 < N; ++j) {
        const V3 u = get(q, posOf(j + 1)) - get(q, posOf(j));
        const double L = u.len();
        const double e = L - p_.D;
        F += 0.5 * p_.kStretch * e * e;
    }
    for (int j = 0; j + 2 < N; ++j) {
        const V3 a = get(q, posOf(j + 1)) - get(q, posOf(j));
        const V3 b = get(q, posOf(j + 2)) - get(q, posOf(j + 1));
        const double la = a.len(), lb = b.len();
        if (la <= 0 || lb <= 0) continue;
        F += kappa() * (1.0 - a.dot(b) / (la * lb));
    }
    // twist
    for (int j = 0; j + 1 < N; ++j) {
        const double d = q[twistOf(j + 1)] - q[twistOf(j)] - p_.psi0 * p_.D;
        F += 0.5 * p_.kTwist * d * d;
    }
    // slip: shear-lag coupling between nodes, and the periodic registry well
    for (int m = 0; m < nSlip_; ++m) {
        const double w = weight(m);
        for (int j = 0; j + 1 < N; ++j) {
            const double d = q[slipOf(j + 1, m)] - q[slipOf(j, m)];
            F += 0.5 * shearK(j, m) * d * d;
        }
        for (int j = 0; j < N; ++j) {
            const double wd = wrapD(q[slipOf(j, m)], p_.D);
            F += w * p_.slipDepth * (1.0 - std::exp(-wd * wd / (2.0 * s2)));
        }
    }
    return F;
}

void Microfibril::gradient(const double* q, double* g) const {
    const int N = p_.nNodes;
    const double s2 = p_.slipWidth * p_.slipWidth;
    for (int i = 0; i < dim_; ++i) g[i] = 0.0;

    for (int j = 0; j + 1 < N; ++j) {
        V3 u = get(q, posOf(j + 1)) - get(q, posOf(j));
        const double L = u.len();
        if (L <= 0) continue;
        const double f = p_.kStretch * (L - p_.D) / L;   // dF/du = k(L-D) * u/L
        const V3 d{f * u.x, f * u.y, f * u.z};
        add(g, posOf(j + 1), d);
        add(g, posOf(j), {-d.x, -d.y, -d.z});
    }

    // U_bend = kappa (1 - c),  c = (a.b)/(|a||b|)
    //   dc/da = (1/|a|)(bhat - c ahat),  dc/db = (1/|b|)(ahat - c bhat)
    for (int j = 0; j + 2 < N; ++j) {
        const V3 a = get(q, posOf(j + 1)) - get(q, posOf(j));
        const V3 b = get(q, posOf(j + 2)) - get(q, posOf(j + 1));
        const double la = a.len(), lb = b.len();
        if (la <= 0 || lb <= 0) continue;
        const V3 ah{a.x / la, a.y / la, a.z / la};
        const V3 bh{b.x / lb, b.y / lb, b.z / lb};
        const double c = ah.dot(bh);
        const double k = kappa();
        const V3 dUda{-k * (bh.x - c * ah.x) / la,
                      -k * (bh.y - c * ah.y) / la,
                      -k * (bh.z - c * ah.z) / la};
        const V3 dUdb{-k * (ah.x - c * bh.x) / lb,
                      -k * (ah.y - c * bh.y) / lb,
                      -k * (ah.z - c * bh.z) / lb};
        add(g, posOf(j + 1), dUda);
        add(g, posOf(j), {-dUda.x, -dUda.y, -dUda.z});
        add(g, posOf(j + 2), dUdb);
        add(g, posOf(j + 1), {-dUdb.x, -dUdb.y, -dUdb.z});
    }

    for (int j = 0; j + 1 < N; ++j) {
        const double d = q[twistOf(j + 1)] - q[twistOf(j)] - p_.psi0 * p_.D;
        g[twistOf(j + 1)] += p_.kTwist * d;
        g[twistOf(j)] -= p_.kTwist * d;
    }

    for (int m = 0; m < nSlip_; ++m) {
        const double w = weight(m);
        for (int j = 0; j + 1 < N; ++j) {
            const double d = shearK(j, m) * (q[slipOf(j + 1, m)] - q[slipOf(j, m)]);
            g[slipOf(j + 1, m)] += d;
            g[slipOf(j, m)] -= d;
        }
        for (int j = 0; j < N; ++j) {
            const double wd = wrapD(q[slipOf(j, m)], p_.D);
            g[slipOf(j, m)] += w * p_.slipDepth * (wd / s2) * std::exp(-wd * wd / (2.0 * s2));
        }
    }
}

bool Microfibril::proposeCollective(int k, Rng& rng, const double* q, double* out,
                                    double& logQRatio) const {
    if (k != 0 || nSlip_ == 0 || p_.nNodes < 1) return false;
    for (int i = 0; i < dim_; ++i) out[i] = q[i];

    // Insert a slip kink into one strand: a smooth tanh step of total height D,
    // centred on a GAP BOND (a molecular terminus) with core half-width xi.
    //
    //   d_j += s * D * 0.5 * (1 + tanh((j - jGap - 0.5)/xi))
    //
    // Two things this move does NOT do, both deliberate:
    //   * it never proposes a sharp step inside a molecule. That puts a full D
    //     of differential slip across an EA/D spring, ~1.3e4 kT, and rejects
    //     with probability 1.
    //   * it never centres on an interior bond. Even at the optimal kink width
    //     an interior slip costs ~470 kT with the measured stiffness, so a
    //     proposal drawn uniformly over bonds would report zero acceptance and
    //     read as "registry is frozen" when the truth is that slip happens
    //     somewhere else entirely.
    // Both were measured, not guessed -- see the acceptance figures in the
    // self-test, which is why the move set looks like this.
    const int m = rng.index(nSlip_);
    // Gap bonds of strand m among bonds 0..nNodes-2. The set depends only on
    // (m, gapPeriod), never on the configuration, so the proposal stays
    // symmetric and logQRatio stays honestly zero.
    int nGap = 0;
    for (int j = 0; j + 1 < p_.nNodes; ++j) nGap += isGapBond(j, m) ? 1 : 0;
    if (nGap == 0) return false;
    int pick = rng.index(nGap), jGap = 0;
    for (int j = 0; j + 1 < p_.nNodes; ++j) {
        if (isGapBond(j, m) && pick-- == 0) { jGap = j; break; }
    }
    const double xi = p_.kinkWidthMin *
                      std::pow(p_.kinkWidthMax / p_.kinkWidthMin, rng.uniform());
    const double s = (rng.uniform() < 0.5 ? 1.0 : -1.0) * p_.D;
    for (int j = 0; j < p_.nNodes; ++j)
        out[slipOf(j, m)] += s * 0.5 * (1.0 + std::tanh((double(j) - double(jGap) - 0.5) / xi));

    // Symmetric. Strand, centre, width and sign are drawn from state-independent
    // uniforms; the exact reverse is the same draw with the sign flipped, which
    // has identical density. So logQRatio = 0 -- and unlike the retired docking
    // engine's dock hops, that is a statement we can actually defend.
    logQRatio = 0.0;
    return true;
}

Profile Microfibril::slipProfile(int nSamples) const {
    // UPPER BOUND on the kink-glide barrier: the substrate seen by one strand
    // at one node, with its neighbours held fixed. The true barrier is the
    // Peierls-Nabarro barrier of the kink, which is smaller -- often by orders
    // of magnitude when the core is wide -- and requires a string or umbrella
    // calculation in this rung's coordinates (closure task L2-5). Quoting this
    // number as the creep barrier would be exactly the kind of shortcut this
    // framework exists to refuse.
    Profile pr;
    pr.s0 = -0.5 * p_.D;
    pr.ds = p_.D / double(nSamples - 1);
    pr.F.resize(nSamples);
    const double A = p_.slipDepth * weight(0);
    const double s2 = p_.slipWidth * p_.slipWidth;
    for (int i = 0; i < nSamples; ++i) {
        const double wd = wrapD(pr.at(i), p_.D);
        pr.F[i] = A * (1.0 - std::exp(-wd * wd / (2.0 * s2)));
    }
    pr.D = {p_.dSlip};
    return pr;
}

std::vector<Param> Microfibril::provenance() const {
    return {
        {p_.D, Provenance::Measured, "D_period_nm",
         "sequence-derived registry minima, validated all-atom at 1D/2D/3D"},
        {kappa(), Provenance::Assumed, "kappa_bend_kT",
         "needs closure_l2: constrained end-to-end sampling of a 5-mer at rung 1"},
        {p_.kStretch, Provenance::Assumed, "k_stretch_kT_nm2",
         "needs closure_l2: axial pull on a 5-mer, entropic->enthalpic crossover"},
        {p_.kTwist, Provenance::Assumed, "k_twist_kT_rad2",
         "needs closure_l2: twist-restrained sampling; psi0 from the emergent supertwist"},
        {p_.kShear, Provenance::Assumed, "k_shear_kT_nm2",
         "EA/D at E ~ 1 GPa; needs closure_l2: axial pull on one strand at rung 1"},
        {p_.kGap, Provenance::Assumed, "k_gap_kT_nm2",
         "0 = uncrosslinked. Needs closure_l2: pull across a telopeptide crosslink "
         "at rung 1 (the xlinkBuf tethers). Sets elastic vs viscoplastic response"},
        {p_.slipDepth, Provenance::Measured, "slip_depth_kT",
         "PMF campaign well depth, capped at 6 kT/seg by the bootstrap shrinkage"},
        {p_.slipWidth, Provenance::Measured, "slip_width_nm",
         "PMF campaign well width (1.5-3 nm); the same width that broke the v0.3.0 quadrature"},
        {slipCurvature(), Provenance::Derived, "slip_curvature_kT_nm2",
         "depth/width^2 = 2.7; CROSS-CHECK: independently measured D-well stiffness ~2. "
         "If these ever disagree by more than 2x the well shape is wrong, not the depth"},
        {p_.dTrans, Provenance::Assumed, "D_trans_nm2_ns",
         "needs closure_l2: MSD of a free 5-mer at rung 1, or slender-body estimate"},
        {p_.dSlip, Provenance::Assumed, "D_slip_nm2_ns",
         "needs closure_l2: MSD of one strand's axial offset inside a held bundle"},
    };
}

}  // namespace scale
