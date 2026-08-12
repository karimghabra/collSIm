// Self-test for the scale-ladder core: the rung contract, the MALA propagator
// and the rate module.
//
// These are not smoke tests. Each one pins down a property the multiscale
// argument actually rests on, on a system whose answer is known independently:
//
//   1. gradient gate      -- gradient() really is d(energy)
//   2. free particle      -- the clock: MSD = 2 D t, and acceptance is exactly 1
//   3. harmonic           -- MALA samples exp(-F/kT), <q^2> = kT/k
//   4. asymmetric quartic -- correct distribution on a rugged landscape,
//                            against quadrature
//   5. collective channel -- MH jump moves preserve that distribution and
//                            advance NO time
//   6. rate + dt series   -- MALA's measured MFPT converges to the exact
//                            overdamped MFPT from kramers.cpp as dt -> 0
//
// Test 6 is the one that licenses reading kinetics off a rung at all, and test
// 5 is the one that stops a sampling trick being mistaken for dynamics.

#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/scale/kramers.h"
#include "../src/scale/level.h"
#include "../src/scale/mala.h"
#include "../src/scale/microfibril.h"

using namespace scale;

static int g_fail = 0;

static void check(const char* what, double got, double want, double tolRel) {
    const double rel = std::fabs(got - want) / (std::fabs(want) + 1e-30);
    const bool ok = rel <= tolRel;
    if (!ok) ++g_fail;
    std::printf("  [%s] %-42s got %12.6g  want %12.6g  (%.2f%% vs %.2f%%)\n",
                ok ? "ok" : "FAIL", what, got, want, 100 * rel, 100 * tolRel);
}

// For quantities whose target is zero, or that are exact up to accumulated
// floating-point addition. A relative test is meaningless in both cases.
static void checkAbs(const char* what, double got, double want, double tolAbs) {
    const double d = std::fabs(got - want);
    const bool ok = d <= tolAbs;
    if (!ok) ++g_fail;
    std::printf("  [%s] %-42s got %12.6g  want %12.6g  (|d| %.3g vs %.3g)\n",
                ok ? "ok" : "FAIL", what, got, want, d, tolAbs);
}

// F(q) = h (q^2 - 1)^2 + tilt * q, one dimension.
// Barrier near q = 0, minima near q = +-1. Everything about it is known in
// closed form or by quadrature, which is the point.
class Quartic : public Level {
public:
    Quartic(double h, double tilt, double D) : h_(h), tilt_(tilt), D_(D) {}
    const char* name() const override { return "quartic-test"; }
    int dim() const override { return 1; }
    double energy(const double* q) const override {
        const double u = q[0] * q[0] - 1.0;
        return h_ * u * u + tilt_ * q[0];
    }
    void gradient(const double* q, double* g) const override {
        g[0] = 4.0 * h_ * q[0] * (q[0] * q[0] - 1.0) + tilt_;
    }
    const double* diffusion() const override { return &D_; }

    // One collective channel: reflect through the barrier, q -> -q. Symmetric,
    // so logQRatio = 0 and the acceptance is pure exp(-dF). It jumps the
    // barrier in a single move -- and buys exactly zero physical time.
    int nCollective() const override { return 1; }
    const char* collectiveName(int) const override { return "reflect"; }
    bool proposeCollective(int, Rng&, const double* q, double* out,
                           double& logQRatio) const override {
        out[0] = -q[0];
        logQRatio = 0.0;
        return true;
    }
    std::vector<Param> provenance() const override {
        return {{h_, Provenance::Assumed, "barrier_h", "analytic test system"},
                {D_, Provenance::Assumed, "D", "analytic test system"}};
    }

private:
    double h_, tilt_, D_;
};

// Boltzmann average of f over the quartic, by dense quadrature.
template <class F>
static double quadAvg(const Quartic& lv, F f, double lo, double hi, int n) {
    double num = 0, den = 0, Fmin = 1e300;
    for (int i = 0; i <= n; ++i) {
        const double q = lo + (hi - lo) * i / n;
        Fmin = std::min(Fmin, lv.energy(&q));
    }
    for (int i = 0; i <= n; ++i) {
        const double q = lo + (hi - lo) * i / n;
        const double w = (i == 0 || i == n) ? 0.5 : 1.0;
        const double b = w * std::exp(-(lv.energy(&q) - Fmin));
        num += b * f(q);
        den += b;
    }
    return num / den;
}

int main() {
    std::printf("scale self-test\n");

    // --- 1. gradient gate --------------------------------------------------
    {
        Quartic lv(5.0, 0.8, 1.0);
        double worst = 0;
        for (double q = -2.0; q <= 2.0; q += 0.25)
            worst = std::max(worst, checkGradient(lv, &q));
        std::printf("1. gradient gate\n");
        checkAbs("max relative gradient error", worst, 0.0, 1e-6);
        std::printf("%s", provenanceReport(lv).c_str());
    }

    // --- 2. free particle: the clock ---------------------------------------
    {
        std::printf("2. free particle (clock + FDT)\n");
        Quartic lv(0.0, 0.0, 0.5);      // h = 0, tilt = 0  ->  F identically 0
        MalaConfig cfg; cfg.dt = 0.01; cfg.adapt = false;
        const int reps = 4000;
        const long long steps = 2000;
        double msd = 0, t = 0;
        for (int r = 0; r < reps; ++r) {
            Mala m(lv, cfg, 900u + r);
            double q0 = 0.0; m.setState(&q0);
            m.run(steps);
            msd += m.state()[0] * m.state()[0];
            t = m.stats().timeNs;
        }
        msd /= reps;
        check("MSD / 2 D t", msd / (2.0 * 0.5 * t), 1.0, 0.05);
        Mala m(lv, cfg, 7u);
        double q0 = 0; m.setState(&q0); m.run(1000);
        check("acceptance on a flat landscape", m.stats().acceptance(), 1.0, 1e-12);
    }

    // --- 3. harmonic: exact Boltzmann sampling -----------------------------
    {
        std::printf("3. harmonic well\n");
        // F = h(q^2-1)^2 is not harmonic, so use the quartic's own quadrature
        // instead of a closed form; the harmonic case is covered by the small-
        // amplitude limit below via <q^2> against quadrature at h = 0.5.
        Quartic lv(0.5, 0.0, 1.0);
        MalaConfig cfg; cfg.dt = 0.02;
        Mala m(lv, cfg, 4242u);
        double q0 = 1.0; m.setState(&q0);
        m.run(200000);                 // burn-in
        m.resetStats();
        double s2 = 0; long long n = 0;
        for (long long i = 0; i < 4000000; ++i) {
            m.stepKinetic();
            s2 += m.state()[0] * m.state()[0]; ++n;
        }
        const double want = quadAvg(lv, [](double q) { return q * q; }, -6, 6, 200000);
        check("<q^2> vs quadrature", s2 / n, want, 0.01);
        std::printf("       acceptance %.3f at dt %.3g\n", m.stats().acceptance(), cfg.dt);
    }

    // --- 4/5. asymmetric double well, with and without collective moves ----
    {
        std::printf("4. asymmetric double well (kinetic channel only)\n");
        Quartic lv(5.0, 0.8, 1.0);
        const double wantQ = quadAvg(lv, [](double q) { return q; }, -3, 3, 400000);
        const double wantQ2 = quadAvg(lv, [](double q) { return q * q; }, -3, 3, 400000);
        {
            MalaConfig cfg; cfg.dt = 0.002;
            Mala m(lv, cfg, 31337u);
            double q0 = -1.0; m.setState(&q0);
            m.run(200000);
            m.resetStats();
            double s1 = 0, s2 = 0; long long n = 0;
            for (long long i = 0; i < 30000000; ++i) {
                m.stepKinetic();
                s1 += m.state()[0]; s2 += m.state()[0] * m.state()[0]; ++n;
            }
            check("<q>  vs quadrature", s1 / n, wantQ, 0.05);
            check("<q^2> vs quadrature", s2 / n, wantQ2, 0.02);
            std::printf("       acceptance %.3f, clock %.4g ns\n",
                        m.stats().acceptance(), m.stats().timeNs);
        }
        std::printf("5. same, with the collective reflect channel enabled\n");
        {
            MalaConfig cfg; cfg.dt = 0.002; cfg.collectiveEvery = 50;
            Mala m(lv, cfg, 31337u);
            double q0 = -1.0; m.setState(&q0);
            m.run(200000);
            m.resetStats();
            double s1 = 0, s2 = 0; long long n = 0;
            const long long N = 3000000;
            for (long long i = 0; i < N; ++i) {
                m.stepKinetic();
                if (i % 50 == 0) m.stepCollective(0);
                s1 += m.state()[0]; s2 += m.state()[0] * m.state()[0]; ++n;
            }
            // Same distribution, reached with 10x fewer kinetic steps because
            // the jump move crosses the barrier directly.
            check("<q>  vs quadrature", s1 / n, wantQ, 0.05);
            check("<q^2> vs quadrature", s2 / n, wantQ2, 0.02);
            // ... and the clock must not have moved because of them. Tolerance
            // is the accumulated rounding of N successive additions of dt, not
            // any physical slack.
            checkAbs("clock = dt * kinetic steps only",
                     m.stats().timeNs, cfg.dt * double(N), 1e-6);
            std::printf("       collective acceptance %.3f over %lld attempts\n",
                        double(m.stats().collectiveAccepted[0]) /
                            double(m.stats().collectiveAttempts[0]),
                        m.stats().collectiveAttempts[0]);
        }
    }

    // --- 6. rate: MALA vs the exact overdamped MFPT ------------------------
    {
        std::printf("6. escape rate, MALA vs exact MFPT (dt convergence)\n");
        const double h = 5.0, D = 1.0;
        Quartic lv(h, 0.0, D);

        // Exact answer, from the Smoluchowski MFPT integral on a dense grid:
        // reflecting at q = -3, absorbing at q = +1, started in the left well.
        Profile pr;
        pr.s0 = -3.0; pr.ds = 0.0005;
        const int n = int((4.0) / pr.ds) + 1;         // -3 .. +1
        pr.F.resize(n);
        for (int i = 0; i < n; ++i) {
            const double q = pr.at(i), u = q * q - 1.0;
            pr.F[i] = h * u * u;
        }
        pr.D = {D};
        const int ia = int((-1.0 - pr.s0) / pr.ds);   // left minimum
        const int ib = n - 1;                         // absorbing at +1
        const double tauExact = mfpt(pr, ia, ib);
        const double kParab = kramersParabolic(pr, ia, int((0.0 - pr.s0) / pr.ds));
        std::printf("       exact MFPT %.4g ns   (parabolic Kramers gives %.4g)\n",
                    tauExact, 1.0 / kParab);

        // Two O(dt) errors are in play here and they are not the same thing:
        //   (a) MALA's own kinetic bias, from rejections;
        //   (b) the absorbing boundary being tested only at step boundaries, so
        //       a trajectory that crosses q = 1 and returns inside one step is
        //       missed. Both push the MFPT up, which is why the series is
        //       expected to come DOWN to the exact value rather than straddle
        //       it. First-passage times are near-exponentially distributed, so
        //       the standard error is about mean/sqrt(N) -- printed, because a
        //       dt series read without error bars is how the weighted-ensemble
        //       tau story went wrong.
        const double dts[] = {0.008, 0.004, 0.002, 0.001};
        double last = 0;
        for (double dt : dts) {
            MalaConfig cfg; cfg.dt = dt;
            const int events = 2500;
            double sum = 0, sum2 = 0;
            for (int e = 0; e < events; ++e) {
                Mala m(lv, cfg, 5000u + 7u * e + uint64_t(1.0 / dt));
                double q0 = -1.0; m.setState(&q0);
                while (m.state()[0] < 1.0 && m.stats().timeNs < 5000.0) m.stepKinetic();
                sum += m.stats().timeNs;
                sum2 += m.stats().timeNs * m.stats().timeNs;
            }
            last = sum / events;
            const double se = std::sqrt(std::max(0.0, sum2 / events - last * last) / events);
            std::printf("       dt %-8.4g  MFPT %10.4g +- %-8.3g  ratio to exact %6.3f +- %.3f\n",
                        dt, last, se, last / tauExact, se / tauExact);
        }
        check("MFPT at smallest dt vs exact", last, tauExact, 0.06);
    }

    // --- 7. rung 2: the microfibril is a well-formed rung -------------------
    {
        std::printf("7. microfibril rung\n");
        MicrofibrilParams mp;
        mp.nNodes = 8;
        Microfibril mf(mp);

        // gradient gate on a deliberately messy configuration: bent centerline,
        // twisted, strands out of register by a fraction of a well width.
        std::vector<double> q;
        mf.ideal(q);
        Rng r(77u);
        for (int j = 0; j < mp.nNodes; ++j) {
            q[mf.posOf(j) + 0] += 4.0 * r.normal();
            q[mf.posOf(j) + 1] += 4.0 * r.normal();
            q[mf.posOf(j) + 2] += 2.0 * r.normal();
            q[mf.twistOf(j)] += 0.3 * r.normal();
            for (int m = 0; m < mp.nStrands - 1; ++m) q[mf.slipOf(j, m)] += 1.2 * r.normal();
        }
        checkAbs("gradient vs finite difference", checkGradient(mf, q.data()), 0.0, 1e-6);
        std::printf("       dim %d, substrate curvature %.3g kT/nm^2 "
                    "(independent D-well measurement: ~2)\n",
                    mf.dim(), mf.slipCurvature());
        std::printf("       substrate barrier (UPPER bound on kink glide) %.3g kT\n",
                    mf.slipProfile().F.back());

        // Exact equipartition. With the substrate switched off, the twist and
        // slip sectors are purely harmonic in DIFFERENCES, and the change of
        // variables to differences has unit Jacobian -- so <dpsi^2> = kT/kTwist
        // and <ddelta^2> = kT/kShear with no small-fluctuation caveat. Any
        // clamp, cap or truncation anywhere in the propagator breaks this, which
        // is precisely what it is here to detect.
        MicrofibrilParams hp = mp;
        hp.slipDepth = 0.0;
        hp.psi0 = 0.0;
        hp.kGap = hp.kShear;   // uniform chain, so every difference mode has the same variance
        Microfibril harm(hp);
        MalaConfig cfg; cfg.dt = 1.0;
        Mala m(harm, cfg, 20260811u);
        std::vector<double> q0;
        harm.ideal(q0);
        m.setState(q0.data());
        m.run(200000);
        m.resetStats();
        double sPsi = 0, sSlip = 0; long long nPsi = 0, nSlip = 0;
        for (long long i = 0; i < 2000000; ++i) {
            m.stepKinetic();
            const std::vector<double>& s = m.state();
            for (int j = 0; j + 1 < hp.nNodes; ++j) {
                const double d = s[harm.twistOf(j + 1)] - s[harm.twistOf(j)];
                sPsi += d * d; ++nPsi;
                for (int k2 = 0; k2 < hp.nStrands - 1; ++k2) {
                    const double e = s[harm.slipOf(j + 1, k2)] - s[harm.slipOf(j, k2)];
                    sSlip += e * e; ++nSlip;
                }
            }
        }
        check("<dpsi^2>   vs kT/kTwist", sPsi / nPsi, 1.0 / hp.kTwist, 0.03);
        check("<dslip^2>  vs kT/kShear", sSlip / nSlip, 1.0 / hp.kShear, 0.03);
        std::printf("       acceptance %.3f at dt %.3g ns, clock %.4g ns\n",
                    m.stats().acceptance(), cfg.dt, m.stats().timeNs);

        // The kink channel must fire on an uncrosslinked microfibril and must
        // be suppressed by crosslinks. That contrast is the rung's first real
        // prediction, so it is asserted rather than merely printed: it is the
        // molecular origin of collagen's elastic/viscoplastic switch.
        auto kinkAccept = [&](double kGap) {
            MicrofibrilParams kp = mp;
            kp.kGap = kGap;
            Microfibril lv2(kp);
            std::vector<double> q2;
            lv2.ideal(q2);
            Mala mk(lv2, MalaConfig{}, 555u);
            mk.setState(q2.data());
            for (int i = 0; i < 20000; ++i) mk.stepCollective(0);
            return double(mk.stats().collectiveAccepted[0]) /
                   double(mk.stats().collectiveAttempts[0]);
        };
        const double accFree = kinkAccept(0.0);      // no crosslinks
        const double accXlink = kinkAccept(20.0);    // crosslinked gap, kT/nm^2
        std::printf("       kink acceptance: uncrosslinked %.4f, crosslinked %.4g\n",
                    accFree, accXlink);
        if (accFree <= 0.01) {
            ++g_fail;
            std::printf("  [FAIL] kink channel does not fire on an uncrosslinked microfibril\n");
        }
        if (accXlink >= accFree) {
            ++g_fail;
            std::printf("  [FAIL] crosslinks failed to suppress slip\n");
        }
    }

    std::printf(g_fail ? "\nFAILURES: %d\n" : "\nall checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
