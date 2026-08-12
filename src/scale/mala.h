#pragma once
#include <vector>

#include "level.h"
#include "rng.h"

// ---------------------------------------------------------------------------
// The propagator shared by every rung of the ladder.
//
// Metropolis-adjusted Langevin (MALA). One proposal:
//
//   mu_i  = q_i - D_i (dF/dq_i) dt                 [kT = 1 in our units]
//   q'_i  = mu_i + sqrt(2 D_i dt) xi_i             xi ~ N(0,1), independent
//
// accepted with
//
//   alpha = min{1, exp[-(F(q') - F(q))] * T(q'->q)/T(q->q')}
//
// where T is the Gaussian above. What this buys, stated precisely because the
// two halves are routinely conflated:
//
//   THERMODYNAMICS is exact at any dt. The chain's invariant measure is
//   exp(-F/kT) exactly -- not to O(dt), exactly. This is what the fast BD
//   engine cannot claim: its displacement clamp fires on most steps and
//   truncates the Gaussian, so its invariant measure is a function of the
//   clamp. MALA needs no clamp, no force cap, no adaptive-dt guard. Those were
//   three hacks and the Metropolis test retires all three at once.
//
//   DYNAMICS is exact only as dt -> 0. Every rejection is a deviation from the
//   true Langevin trajectory, so a rate read off a MALA run is biased at O(dt).
//   The bias is one-signed and shrinks with dt, so any kinetic result must come
//   with a dt-convergence series (see dtConvergence() in the self-test). Aiming
//   at ~90% acceptance keeps that bias small; it is not a substitute for
//   checking it.
//
// The clock advances by dt on every kinetic step INCLUDING rejected ones -- a
// rejection means the system stayed where it was for dt, which is a physical
// outcome, not a wasted iteration. Collective moves advance no time at all.
// ---------------------------------------------------------------------------

namespace scale {

struct MalaConfig {
    double dt = 1.0;             // ns
    double targetAccept = 0.90;  // high on purpose: see the dynamics note above
    // Robbins-Monro adaptation of dt. LEGAL ONLY DURING EQUILIBRATION. While
    // dt is moving the transition kernel is not fixed, so the chain has no
    // single invariant measure and detailed balance means nothing. Production
    // must run with adapt = false; freeze() sets it.
    bool adapt = false;
    double adaptGain = 0.05;
    // Attempt one collective move (chosen uniformly among the rung's channels)
    // every N kinetic steps. 0 disables them, which is what a production
    // kinetics run wants -- collective moves are for equilibration and for
    // free-energy work, and the rate of the transition they represent belongs
    // in the Gillespie layer via kramers.h.
    int collectiveEvery = 0;
};

struct MalaStats {
    long long kineticSteps = 0;
    long long kineticAccepted = 0;
    std::vector<long long> collectiveAttempts;
    std::vector<long long> collectiveAccepted;
    double timeNs = 0;              // kinetic channel only
    double dt = 0;
    bool adaptedThisRun = false;    // if true, the run is equilibration, not production
    double acceptance() const {
        return kineticSteps ? double(kineticAccepted) / double(kineticSteps) : 0.0;
    }
};

class Mala {
public:
    Mala(const Level& lv, const MalaConfig& cfg, uint64_t seed);

    void setState(const double* q);
    const std::vector<double>& state() const { return q_; }
    double energy() const { return F_; }        // cached F(q), kT

    // One MALA step on the kinetic channel. Returns true if accepted.
    bool stepKinetic();
    // One attempt on collective channel k. Returns true if accepted.
    bool stepCollective(int k);
    // n kinetic steps, interleaving collective attempts per cfg.collectiveEvery.
    void run(long long n);

    // End adaptation. Call before any production run; the recorded dt is then
    // fixed for the rest of the chain's life.
    void freeze() { cfg_.adapt = false; }

    const MalaStats& stats() const { return st_; }
    void resetStats();
    const MalaConfig& config() const { return cfg_; }

private:
    void refresh();   // recompute F_ and g_ at q_

    const Level& lv_;
    MalaConfig cfg_;
    Rng rng_;
    int n_ = 0;
    const double* D_ = nullptr;
    std::vector<double> q_, g_, qTrial_, gTrial_;
    double F_ = 0;
    MalaStats st_;
};

}  // namespace scale
