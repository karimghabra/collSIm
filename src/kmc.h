#pragma once
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "io.h"

// Molecule-level Gillespie kinetic Monte Carlo for fibrillation kinetics
// (nucleation - elongation - detachment - fragmentation). Rates are scaled by
// the same sequence-derived registry energetics the BD engine uses, so the
// two modes share one physical model at different resolutions.
struct KmcParams {
    double volUm3 = 8.0;      // virtual reaction volume
    double tMaxMin = 180.0;   // simulated span
    double kn = 2e3;          // nucleation prefactor (M^-(nc-1) s^-1)
    int nc = 3;               // critical nucleus size
    double kplus = 5e5;       // elongation, M^-1 s^-1 per end
    double koff0 = 2e-3;      // end detachment at reference conditions, s^-1
    double kfrag = 1e-8;      // fragmentation per internal bond, s^-1
    double dGscale = 5.0;     // kT of koff modulation per unit well-depth change
    // registry resolution: attachments Boltzmann-sample their stagger within a
    // scan window (how far a docked molecule explores before locking; measured
    // qualitatively in the BD pair tests), detachment satisfies detailed balance
    double scanWinNm = 15.0;
    double contactPairs = 25.0;  // converts kT/pair well depth to molecule dG
};

// 1D registry energy landscape at the current environment (kT/pair vs stagger)
struct KmcProfile {
    std::vector<float> E;     // best-over-facing-angles energy per stagger bin
    float dstep = 0.5f, D = 66.9f, L = 290.f;
};

struct KmcResult {
    std::vector<float> t;             // minutes
    std::vector<float> fibrilMass;    // fraction of total mass in fibrils
    std::vector<float> freeMono;      // fraction free
    std::vector<float> nFib;          // fibril count
    std::vector<float> bandOrder;     // D-registry phase coherence of fibril mass
    std::vector<float> stagHist;      // final stagger histogram (mod D, 24 bins)
    double lagMin = -1, t50Min = -1, plateau = 0;
    double finalOrder = 0;
    double fOff = 1, fNuc = 1;        // environment multipliers (reported)
    int events = 0;
};

// deepest combined registry well (kT/contact-pair, positive) near the
// D-stagger, at the given pH and epsilon weights, from the basis tables
double kmcWellDepth(const Basis2D& b, const std::vector<float>& parRG,
                    float pH, float epsEl, float epsHy,
                    const Corr2D* corr = nullptr, float epsCorr = 1.f);

// angular-free-energy 1D registry landscape at the given environment
KmcProfile kmcProfile1D(const Basis2D& b, const std::vector<float>& parRG,
                        float pH, float epsEl, float epsHy,
                        const Corr2D* corr = nullptr, float epsCorr = 1.f);

// idealized funnel landscape (telopeptide dock-and-lock stand-in, registry
// model 1) from the 1D profile tables
KmcProfile kmcProfileFunnel(const Profiles& p1, float epsEl, float epsHy,
                            float D, float L);

// A recorded still of the KMC state: which fibrils exist and, for every
// molecule in them, the stagger bin it attached at. That is enough to place
// every molecule in space, because a fibril of known staggers on the
// Hodge-Petruska lattice is a fully determined 3D structure -- so hours of
// assembly can be WATCHED at full-atom detail without integrating them.
// Rigorous here: when each molecule attaches and what registry it takes.
// Cosmetic: the path it travels between events.
struct KmcFrame {
    float tMin = 0;
    float freeFrac = 1;
    float massFrac = 0;
    std::vector<std::vector<int>> fibStag;   // [fibril][molecule] = stagger bin
};

KmcResult kmcRun(const KmcParams& p, const KmcProfile& prof, const KmcProfile& profRef,
                 double concMgMl, double mwKda, double tempC, uint32_t seed,
                 std::vector<KmcFrame>* frames = nullptr, int maxFrames = 240);

// Lay a recorded frame out in space: molecules are packed onto the
// quasi-hexagonal lattice (5D file period, neighbour files offset by
// (i+2j mod 5)*D) at their own KMC-sampled stagger, free monomers scattered
// through the box. Emits a straight-rod bead polyline per molecule, in the
// same layout the renderer already consumes.
void kmcFrameToBeads(const KmcFrame& f, const KmcProfile& prof, int nMolMax,
                     int nBeads, float segLen, const float boxHalf[3],
                     uint32_t seed, std::vector<glm::vec4>& out, int& nUsed);
