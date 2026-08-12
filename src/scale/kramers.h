#pragma once
#include <vector>

// ---------------------------------------------------------------------------
// Rates for the collective transitions, from measured free-energy profiles.
//
// This is the file that replaces the retired kmc.cpp. That engine reached
// hours, but its rates (kn, k+, koff, kfrag) were knobs -- a fitting surface,
// not a measurement. Everything here takes a free-energy profile F(s) along a
// transition coordinate and the diffusion coefficient D(s) along that same
// coordinate, both measured at the rung below, and returns a rate with no
// adjustable prefactor.
//
// The primary routine is the exact overdamped mean first passage time
//
//   tau(a->b) = integral_a^b dy [1/D(y)] exp(F(y)) integral_{-inf}^{y} dz exp(-F(z))
//
// (Zwanzig, eq. for 1D Smoluchowski with a reflecting boundary below a and an
// absorbing boundary at b; F in kT). It needs no parabolic fit, no barrier
// location, and no transmission coefficient -- which is why it is preferred
// here over the textbook Kramers formula. kramersParabolic() is provided only
// as a cross-check: if the two disagree badly the barrier is not parabolic and
// the closed form is the one that is wrong.
//
// The rate that comes out is what the rung ABOVE consumes, either as a
// Gillespie propensity or as a bias-free acceptance rate. That is the whole
// hand-off: fine rung measures a barrier, coarse rung fires an event.
// ---------------------------------------------------------------------------

namespace scale {

struct Profile {
    double s0 = 0;            // coordinate of sample 0
    double ds = 1;            // uniform spacing, unit of s
    std::vector<double> F;    // free energy, kT
    std::vector<double> D;    // diffusion along s, (unit of s)^2 / ns.
                              // size 1 = constant, else same size as F
    double at(int i) const { return s0 + ds * i; }
    double diff(int i) const { return D.size() == 1 ? D[0] : D[i]; }
};

// Mean first passage time from the basin containing index ia to the absorbing
// point ib (ia < ib), in ns. Reflecting at index 0.
double mfpt(const Profile& p, int ia, int ib);

// 1/mfpt, in 1/ns.
double rateMFPT(const Profile& p, int ia, int ib);

// Closed-form overdamped Kramers, for cross-checking mfpt() only:
//   k = D sqrt(|F''_min| |F''_bar|) / (2 pi) * exp(-(F_bar - F_min))
// Curvatures are taken by finite difference at the located extrema.
double kramersParabolic(const Profile& p, int iMin, int iBarrier);

// Equilibrium constant between two basins, from the profile alone:
// K = integral over basin B of exp(-F) / integral over basin A of exp(-F).
// Detailed balance then fixes the reverse rate as kForward / K -- never
// measure both directions and let them disagree.
double basinRatio(const Profile& p, int aLo, int aHi, int bLo, int bHi);

}  // namespace scale
