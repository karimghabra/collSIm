#include "mala.h"

#include <cmath>

namespace scale {

Mala::Mala(const Level& lv, const MalaConfig& cfg, uint64_t seed)
    : lv_(lv), cfg_(cfg), rng_(seed), n_(lv.dim()), D_(lv.diffusion()) {
    q_.assign(n_, 0.0);
    g_.assign(n_, 0.0);
    qTrial_.assign(n_, 0.0);
    gTrial_.assign(n_, 0.0);
    st_.collectiveAttempts.assign(lv.nCollective(), 0);
    st_.collectiveAccepted.assign(lv.nCollective(), 0);
    st_.dt = cfg_.dt;
    refresh();
}

void Mala::setState(const double* q) {
    for (int i = 0; i < n_; ++i) q_[i] = q[i];
    refresh();
}

void Mala::refresh() {
    F_ = lv_.energy(q_.data());
    lv_.gradient(q_.data(), g_.data());
}

void Mala::resetStats() {
    double dt = st_.dt;
    st_ = MalaStats{};
    st_.collectiveAttempts.assign(lv_.nCollective(), 0);
    st_.collectiveAccepted.assign(lv_.nCollective(), 0);
    st_.dt = dt;
}

bool Mala::stepKinetic() {
    const double dt = cfg_.dt;

    // Forward proposal. Drift is -D grad F dt; the noise amplitude is tied to
    // it by the fluctuation-dissipation theorem, which is why D is the only
    // transport number a rung supplies -- there is no independent "temperature
    // of the coarse variable" to tune.
    double logTfwd = 0.0;
    for (int i = 0; i < n_; ++i) {
        const double var = 2.0 * D_[i] * dt;
        const double mu = q_[i] - D_[i] * g_[i] * dt;
        const double dx = std::sqrt(var) * rng_.normal();
        qTrial_[i] = mu + dx;
        logTfwd -= dx * dx / (2.0 * var);
    }

    const double Ftrial = lv_.energy(qTrial_.data());
    lv_.gradient(qTrial_.data(), gTrial_.data());

    // Reverse proposal density, evaluated at the trial point's own drift. The
    // Gaussian normalisations cancel: same D, same dt, both directions.
    double logTrev = 0.0;
    for (int i = 0; i < n_; ++i) {
        const double var = 2.0 * D_[i] * dt;
        const double muBack = qTrial_[i] - D_[i] * gTrial_[i] * dt;
        const double dx = q_[i] - muBack;
        logTrev -= dx * dx / (2.0 * var);
    }

    const double logAlpha = -(Ftrial - F_) + logTrev - logTfwd;
    bool accept = false;
    if (logAlpha >= 0.0) {
        accept = true;
    } else if (std::log(rng_.uniform()) < logAlpha) {
        accept = true;
    }
    // NaN guard: an energy that returned NaN must reject, never propagate.
    if (!(Ftrial == Ftrial)) accept = false;

    if (accept) {
        q_.swap(qTrial_);
        g_.swap(gTrial_);
        F_ = Ftrial;
    }

    ++st_.kineticSteps;
    st_.kineticAccepted += accept ? 1 : 0;
    // Time advances either way. A rejection is the system sitting still for dt.
    st_.timeNs += dt;

    if (cfg_.adapt) {
        // Robbins-Monro on log dt. Gain decays as 1/sqrt(steps) so the step
        // size settles rather than rattling; the caller must still freeze()
        // before collecting anything.
        const double g = cfg_.adaptGain / std::sqrt(1.0 + double(st_.kineticSteps));
        cfg_.dt *= std::exp(g * ((accept ? 1.0 : 0.0) - cfg_.targetAccept));
        st_.dt = cfg_.dt;
        st_.adaptedThisRun = true;
    }
    return accept;
}

bool Mala::stepCollective(int k) {
    if (k < 0 || k >= lv_.nCollective()) return false;
    double logQRatio = 0.0;
    if (!lv_.proposeCollective(k, rng_, q_.data(), qTrial_.data(), logQRatio)) {
        ++st_.collectiveAttempts[k];
        return false;
    }
    const double Ftrial = lv_.energy(qTrial_.data());
    const double logAlpha = -(Ftrial - F_) + logQRatio;

    bool accept = (logAlpha >= 0.0) || (std::log(rng_.uniform()) < logAlpha);
    if (!(Ftrial == Ftrial)) accept = false;

    if (accept) {
        q_.swap(qTrial_);
        F_ = Ftrial;
        lv_.gradient(q_.data(), g_.data());
    }
    ++st_.collectiveAttempts[k];
    st_.collectiveAccepted[k] += accept ? 1 : 0;
    // No clock. See the Channel comment in level.h.
    return accept;
}

void Mala::run(long long n) {
    for (long long s = 0; s < n; ++s) {
        stepKinetic();
        if (cfg_.collectiveEvery > 0 && lv_.nCollective() > 0 &&
            (st_.kineticSteps % cfg_.collectiveEvery) == 0) {
            stepCollective(rng_.index(lv_.nCollective()));
        }
    }
}

}  // namespace scale
