#pragma once
#include <vector>

#include "kramers.h"
#include "level.h"

// ---------------------------------------------------------------------------
// Rung 2: the microfibril.
//
// The object is the five-stranded, D-periodic quasi-hexagonal unit that the
// molecular engine already produces spontaneously (the docking MC's mean
// cluster size sits at ~5.3). Coarse-graining stops here rather than at the
// molecule because the microfibril is the smallest unit whose internal
// registry is already resolved: below it, D-stagger is the thing being
// determined; above it, D-stagger is a property you carry.
//
// COORDINATES, per node. A node is one D-period (66.9 nm) of arc:
//
//   r_j      centerline position of node j                        3
//   psi_j    twist of the cross-section about the centerline      1
//   d_j,m    axial slip of strand m (m = 1..4) against strand 0    4
//                                                              -----
//                                                                8 per node
//
// Strand 0 has no slip coordinate on purpose. Its axial position IS the arc
// position of the node, already carried by r_j; giving it one too would leave
// F flat along the sum of the slips, and a coordinate with no restoring force
// and no physical meaning is a gauge freedom, not a degree of freedom. This is
// the sort of thing that silently poisons a mobility measurement.
//
// FREE ENERGY. Five terms, and each one names the measurement that sets it:
//
//   F_bend    = kappa sum_j (1 - cos theta_j)             theta between links
//   F_stretch = 1/2 kStretch sum_j (|u_j| - D)^2
//   F_twist   = 1/2 kTwist sum_j (dpsi_j - psi0 D)^2      psi0 = intrinsic supertwist
//   F_shear   = 1/2 kShear sum_j,m (d_{j+1,m} - d_{j,m})^2
//   F_slip    = sum_j,m w_m V(d_j,m)                      V periodic with period D
//
// F_shear is the staggered-composite shear-lag term, and F_slip is the
// registry landscape the PMF campaign measured. Their ratio is what decides
// whether a fibril responds to load elastically or by molecular slip, which is
// the whole reason this rung exists: it is where collagen's viscoplasticity
// lives, and it cannot be seen at rung 1 (too slow) or asserted at rung 3
// (that is where the hacks would go).
//
// WHAT THIS RUNG ACTUALLY IS. A harmonic chain (F_shear) sitting in a periodic
// substrate potential (F_slip) is the Frenkel-Kontorova model -- the canonical
// model of interfacial slip and dislocations, seventy years old and thoroughly
// understood. That identification is not decoration; it dictates the physics
// and it dictated a rewrite of the collective move:
//
//   * A strand cannot slip by D at one node and not the next. That would be
//     100% axial strain over one D-period, and with the strand's own measured
//     stiffness it costs ~1.3e4 kT. Slip spreads over a KINK of finite core
//     width, set by sqrt(kShear / V''), and the kink is the object that moves.
//   * So the collective move inserts a kink profile, not a step. A step move
//     is rejected with probability 1 and would have looked like "registry is
//     frozen" while actually meaning "the move set was wrong" -- the same
//     failure shape as the v0.3.0 contact-integral aliasing bug.
//   * The barrier that sets the creep rate is therefore the Peierls-Nabarro
//     barrier for kink glide, NOT the substrate amplitude. slipProfile()
//     returns the substrate profile, which is a strict UPPER bound on it and
//     is labelled as such. The real number needs a string/umbrella calculation
//     in this rung's own coordinates; that is closure task L2-5.
//
// V is periodic, so slip has degenerate minima one D apart. Kink crossings are
// far too rare for the kinetic channel, so they are a COLLECTIVE move here
// (thermodynamics, exact) and their RATE comes from kramers.h fed with a
// measured barrier (kinetics, exact) -- never from counting accepted moves.
//
// PROVENANCE. Every parameter below is currently Assumed or Derived from a
// single measured curvature. Nothing here should be reported as a kinetic
// result until tools/closure_l2.py has run. hasAssumed() enforces the label.
// ---------------------------------------------------------------------------

namespace scale {

struct MicrofibrilParams {
    int nNodes = 16;
    int nStrands = 5;
    double D = 66.9;          // nm, the D-period; node arc length

    // --- elastic constants ------------------------------------------------
    double kappaBend = 0;     // kT per node; kappa = Lp/D. 0 -> from lpNm
    double lpNm = 3000.0;     // microfibril persistence length, nm
    double kStretch = 30.0;   // kT/nm^2 per node (~5 strands x EA/D at E ~ 1 GPa)
    double kTwist = 300.0;    // kT/rad^2 per node
    double psi0 = 0.0;        // intrinsic twist, rad/nm (the emergent supertwist)

    // --- registry ---------------------------------------------------------
    // Axial stiffness of ONE strand over one node's length, EA/D. This is the
    // shear-lag spring, and it is large: a full-D differential slip across one
    // node costs 1/2 kShear D^2 ~ 1.3e4 kT.
    double kShear = 6.2;      // kT/nm^2, from E ~ 1 GPa, A ~ 1.8 nm^2, L = D

    // ...but a strand is NOT elastically continuous along the whole
    // microfibril, and assuming it was made a full-D slip cost ~470 kT even
    // spread over an optimal kink -- i.e. registry would be permanently frozen,
    // which is wrong. A tropocollagen molecule is ~4.46 D long, so every
    // gapPeriod nodes the strand ENDS and the next molecule begins across the
    // gap zone. Nothing covalent bridges that gap in an uncrosslinked fibril,
    // so the coupling there is not EA/D:
    //
    //   kGap = 0      uncrosslinked. Slip localises at molecular termini and
    //                 costs only the registry substrate -- a few kT. This is
    //                 why uncrosslinked collagen creeps.
    //   kGap > 0      enzymatic crosslinks bridge the gap; load transfers
    //                 elastically and slip is suppressed. This single number
    //                 is the elastic/viscoplastic switch, and it is the reason
    //                 the crosslink work at rung 1 matters upward.
    //
    // gapPeriod = 5 rounds the real 4.46 D molecular length to the D-stagger
    // lattice. That rounding is an approximation of the geometry, not of the
    // energetics; a non-integer version needs the gap bond to move along the
    // strand, which is closure task L2-6.
    double kGap = 0.0;        // kT/nm^2 at molecular termini
    int gapPeriod = 5;        // nodes between termini on one strand

    // The periodic substrate. A well of measured DEPTH and measured WIDTH,
    // repeated with period D:
    //     V(d) = slipDepth * [1 - exp(-w(d)^2 / (2 sigma^2))],  w = d mod D
    // Depth and width both come from the PMF campaign; the curvature at the
    // minimum, depth/sigma^2 = 2.7 kT/nm^2, then REPRODUCES the independently
    // measured D-well stiffness (~2 kT/nm^2) instead of being fitted to it.
    // That agreement is the only reason this functional form is admissible.
    double slipDepth = 6.0;   // kT per strand per node
    double slipWidth = 1.5;   // nm, Gaussian sigma
    std::vector<double> strandWeight;  // contacts per strand in the 5-mer; empty -> all 1

    // Kink core widths the collective move draws from, in nodes, LOG-uniformly.
    // Drawn rather than fixed because the equilibrium width follows from
    // kGap/kShear and V'' and has not been computed yet; letting Metropolis
    // choose costs only acceptance, whereas fixing it wrong costs the physics.
    // Log-uniform because the two regimes are orders apart -- an uncrosslinked
    // gap wants the sharpest kink available, a crosslinked one wants a wide
    // core -- and a linear draw spends 90% of its proposals in neither.
    double kinkWidthMin = 0.05, kinkWidthMax = 2.0;

    // --- transport (FDT) ---------------------------------------------------
    double dTrans = 2.0e-3;   // nm^2/ns, per centerline node
    double dTwist = 1.0e-5;   // rad^2/ns, per node
    double dSlip = 5.0e-4;    // nm^2/ns, a strand sliding inside the bundle
};

class Microfibril : public Level {
public:
    explicit Microfibril(const MicrofibrilParams& p);

    const char* name() const override { return "microfibril"; }
    int dim() const override { return dim_; }
    double energy(const double* q) const override;
    void gradient(const double* q, double* g) const override;
    const double* diffusion() const override { return diff_.data(); }

    int nCollective() const override { return 1; }
    const char* collectiveName(int) const override { return "slip-kink"; }
    bool proposeCollective(int k, Rng& rng, const double* q, double* out,
                           double& logQRatio) const override;

    std::vector<Param> provenance() const override;

    // Lay out a straight, perfectly registered microfibril along +z.
    void ideal(std::vector<double>& q) const;

    // --- coordinate accessors ---------------------------------------------
    int stride() const { return 4 + nSlip_; }     // 3 pos + 1 twist + slips
    int posOf(int j) const { return j * stride(); }
    int twistOf(int j) const { return j * stride() + 3; }
    int slipOf(int j, int m) const { return j * stride() + 4 + m; }  // m = 0..nSlip_-1

    const MicrofibrilParams& params() const { return p_; }

    // The SUBSTRATE profile for one strand at one node, for kramers.h. Read the
    // Frenkel-Kontorova note above before using it: this is the barrier for a
    // rigid single-node hop, which is an upper bound on the kink-glide barrier
    // that actually sets the creep rate. It is here so the bound can be quoted
    // and so the eventual measured barrier has something to be compared to.
    Profile slipProfile(int nSamples = 601) const;

    // Curvature of the substrate at its minimum, kT/nm^2. Derived, and the
    // cross-check against the independently measured D-well stiffness.
    double slipCurvature() const;

    // Axial coupling of strand m across the bond joining nodes j and j+1:
    // kGap at a molecular terminus, kShear inside a molecule.
    double shearK(int j, int m) const { return isGapBond(j, m) ? p_.kGap : p_.kShear; }
    bool isGapBond(int j, int m) const {
        if (p_.gapPeriod <= 0) return false;
        const int r = ((j - m) % p_.gapPeriod + p_.gapPeriod) % p_.gapPeriod;
        return r == 0;
    }

private:
    double kappa() const { return p_.kappaBend > 0 ? p_.kappaBend : p_.lpNm / p_.D; }
    double weight(int m) const {
        return m < int(p_.strandWeight.size()) ? p_.strandWeight[m] : 1.0;
    }

    MicrofibrilParams p_;
    int nSlip_ = 4;     // nStrands - 1
    int dim_ = 0;
    std::vector<double> diff_;
};

}  // namespace scale
