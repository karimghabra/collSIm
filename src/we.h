#pragma once
#include <cstdint>
#include <vector>
#include "sim.h"

// Weighted ensemble on the BD engine.
//
// The point: BD is the only engine here with trustworthy kinetics, but it
// reaches microseconds and nucleation is a rare event on the scale of minutes.
// WE does not alter the dynamics at all -- every walker is plain unadjusted
// Brownian dynamics -- it only redistributes COMPUTATIONAL effort, splitting
// walkers that make progress along a coordinate and merging ones that fall
// back, while carrying weights that keep the ensemble statistically exact.
// Rates therefore come out unbiased: this is a variance-reduction scheme, not
// an acceleration hack, and that distinction is the whole reason to prefer it
// over raising the MC step size.
//
// Progress coordinate = size of the largest connected cluster of molecules.
// Walkers reaching the target size have their weight harvested as flux and are
// recycled to a fresh dispersed state, so the steady-state flux is the mean
// first-passage rate of nucleation (Hill relation).
struct WeParams {
    int walkersPerBin = 4;
    int maxWalkers = 96;
    int tauSteps = 1500;       // BD steps between resampling
    int targetSize = 5;        // cluster size defining the nucleated state
    float contactNm = 2.5f;    // bead-bead distance counted as contact
    int nInitStates = 6;       // pool of dispersed starts for recycling
    // Integer cluster size alone is far too coarse to bin on: a walker is
    // either "n molecules" or "n+1", so nothing distinguishes one that has a
    // partner closing in from one with empty space around it, and WE has
    // nothing to select for. The coordinate is therefore continuous --
    // size + how close the nearest outsider has come to the cluster.
    float approachNm = 25.f;   // distance over which approach counts as progress
    int subBins = 4;           // bins per unit of cluster size
};

struct Walker {
    SimState st;
    double w = 0;
    int size = 1;
    float lambda = 1.f;
};

// Largest connected cluster (bead-contact percolation) plus how near the
// closest non-member molecule has come to it, as a fraction in [0,1).
struct Progress {
    int size = 1;
    float approach = 0.f;
    float value() const { return size + approach; }
};
Progress progressCoord(const std::vector<glm::vec4>& pos, int nMol, int nBeads,
                       float contact, float approachRange, bool pbc,
                       glm::vec3 boxHalf);
int largestCluster(const std::vector<glm::vec4>& pos, int nMol, int nBeads,
                   float contact, bool pbc, glm::vec3 boxHalf,
                   std::vector<int>* sizesOut = nullptr);

class WeRun {
public:
    void init(Sim& sim, const WeParams& pr, uint32_t seed);
    void iterate(Sim& sim);            // advance every walker by tau, resample
    // same work split per walker, so an interactive run can advance one walker
    // per frame and stay responsive (and show the walker being propagated)
    void advanceWalker(Sim& sim, int i);
    void finishIteration(Sim& sim);
    double ratePerNs() const;          // flux / total weighted time
    double mfptNs() const;             // 1 / rate
    int binOf(float lambda) const;
    int nBins() const;

    WeParams p;
    std::vector<Walker> walkers;
    std::vector<SimState> initPool;
    double fluxWeight = 0;             // total weight recycled
    double timeNs = 0;                 // iterations * tau * dt
    // Steady-state accumulators. The flux out of a WE run is meaningless until
    // weight has had time to spread up the coordinate, so the reported rate
    // uses only iterations after the burn-in.
    double fluxWeightSS = 0, timeNsSS = 0;
    int burnInIters = 10;
    double rateSSPerNs() const;
    int iters = 0, recycles = 0, recyclesSS = 0;
    int poolRejects = 0;               // starts discarded as already nucleated
    std::vector<int> binPop;           // walkers per bin, last iteration
    std::vector<double> binWeight;     // weight per bin, last iteration
    double tauNs = 0;

private:
    uint64_t rng = 0x243F6A8885A308D3ull;
    uint32_t seedCounter = 1;
    double rnd();
    void resample();
};
