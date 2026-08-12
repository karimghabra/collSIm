#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "rng.h"

// ---------------------------------------------------------------------------
// The rung contract.
//
// Units everywhere on the ladder: length nm, energy kT, time ns. Same as the
// molecular engine, so numbers can be compared across rungs without a
// conversion table sitting between them.
//
// A rung of the multiscale ladder is NOT "a coarser model of collagen". It is
// a specific mathematical object: a set of coarse coordinates q = M(x) of the
// rung below, together with
//
//   F(q)   = -kT ln integral over x of exp(-U(x)/kT) delta(M(x) - q)
//   D(q)   = the mobility of q, from the fluctuation-dissipation theorem
//
// F and D are not modelling choices. They are properties of the rung below,
// and the only legitimate way to obtain them is to measure them there. A rung
// whose F was written down by hand is a phenomenological model wearing a
// multiscale costume -- which is exactly what the deleted kmc.cpp was, and why
// it is deleted.
//
// Everything a rung must supply:
//   1. dim(), energy(), gradient()   -- F and its EXACT gradient
//   2. diffusion()                   -- D, per coarse DOF
//   3. collective moves (optional)   -- discrete transitions, MH-accepted
//   4. provenance()                  -- where every number came from
//
// Item 4 is not bookkeeping. checkGradient() and the provenance audit are the
// two gates that stop an assumption entering a run unannounced.
// ---------------------------------------------------------------------------

namespace scale {

// Where a number came from. A run that touches an Assumed parameter says so in
// its log and in every artefact it writes; that is the whole point.
enum class Provenance {
    Measured,    // closure run against the rung below, in this repo
    Derived,     // exact consequence of Measured values (a curvature, an FDT mobility)
    Literature,  // published experiment, cited in note
    Assumed      // placeholder. Legal only in a run flagged EXPLORATORY.
};

const char* provenanceName(Provenance p);

struct Param {
    double value = 0;
    Provenance src = Provenance::Assumed;
    const char* name = "";
    const char* note = "";   // citation, closure file, or what would falsify it
};

// A move channel. The distinction is load-bearing and is the single most
// error-prone thing on the ladder:
//
//   Kinetic  -- an MH-adjusted discretisation of the overdamped Langevin
//               generator. Advances the physical clock by dt on EVERY step,
//               accepted or rejected, because a rejection means the walker
//               genuinely stayed put for dt. Correct dynamics only as dt -> 0,
//               so kinetics from this channel require a dt-convergence study.
//
//   Sampling -- a collective move (registry jump, insertion, twist flip). It
//               satisfies detailed balance with respect to exp(-F/kT), so it
//               is exact for THERMODYNAMICS and worthless for kinetics. It
//               advances no time. To get the rate of such a transition,
//               measure its barrier and use kramers.h; do not divide a sweep
//               count by anything.
//
// The MC clock in the retired docking engine was wrong for precisely this
// reason -- crediting axial D-jumps and dock hops with diffusion time inflated
// it 18x. Here the propagator refuses to add time from a Sampling channel.
enum class Channel { Kinetic, Sampling };

class Level {
public:
    virtual ~Level() = default;

    virtual const char* name() const = 0;
    virtual int dim() const = 0;

    // F(q), kT.
    virtual double energy(const double* q) const = 0;

    // dF/dq_i, kT per unit of q_i. MUST be the analytic gradient of energy().
    // checkGradient() below is run as a unit test on every rung; a rung whose
    // gradient disagrees with its energy samples a distribution nobody wrote
    // down, which is the failure mode the fast BD engine's clamps have.
    virtual void gradient(const double* q, double* g) const = 0;

    // D_i for coordinate i, (unit of q_i)^2 / ns. Diagonal by construction:
    // the coarse coordinates are chosen so that the mobility matrix measured at
    // the rung below is diagonal to within its own error bars. If a rung cannot
    // meet that, it needs a full mobility matrix and a Cholesky factor in the
    // propagator -- say so rather than diagonalising by fiat.
    virtual const double* diffusion() const = 0;

    // --- collective (Sampling-channel) moves -------------------------------
    virtual int nCollective() const { return 0; }
    virtual const char* collectiveName(int) const { return "?"; }

    // Propose move k from q into qOut. Returns false if the move is not
    // applicable in this state (counts as a rejection, no time, no bias).
    //
    // logQRatio = ln[ T(q' -> q) / T(q -> q') ]. A symmetric move returns 0.
    // Any move that cannot state this number honestly does not belong here:
    // that was the flaw in the docking engine's dock hops, whose deterministic
    // placement had no library reverse, and which we therefore had to offer a
    // switch to disable.
    virtual bool proposeCollective(int k, Rng& rng, const double* q, double* qOut,
                                   double& logQRatio) const {
        (void)k; (void)rng; (void)q; (void)qOut; (void)logQRatio;
        return false;
    }

    // Every number this rung uses, with its source.
    virtual std::vector<Param> provenance() const { return {}; }
};

// --- gates -----------------------------------------------------------------

// Central-difference check of gradient() against energy(). Returns the largest
// relative discrepancy over the coordinates. Anything above ~1e-5 with a
// sensible h means the two disagree.
double checkGradient(const Level& lv, const double* q, double h = 1e-5);

// True if any parameter is Assumed. Runs that report kinetics must be clean.
bool hasAssumed(const Level& lv);

// One line per parameter, for the run log and for the header of any data file
// the rung emits. Numbers without provenance are how hacks survive.
std::string provenanceReport(const Level& lv);

}  // namespace scale
