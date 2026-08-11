#include "we.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

using glm::vec3;

namespace {
// Beads reach ~145 nm from their molecule's centre, so in a 400 nm periodic
// box they spill well outside it and a pair that touches across the seam lands
// in far-apart cells. Wrapping into the box and indexing the grid periodically
// is what makes those contacts visible at all.
struct ContactGrid {
    std::vector<glm::vec4> wrapped;
    std::vector<int> head, next;
    int dim[3];
    vec3 cell, lo;
    bool pbc;

    void build(const std::vector<glm::vec4>& pos, int nb, float contact, bool periodic,
               vec3 boxHalf) {
        pbc = periodic;
        wrapped.assign(pos.begin(), pos.begin() + nb);
        vec3 loB(1e30f), hiB(-1e30f);
        if (pbc) {
            vec3 bs = 2.f * boxHalf;
            for (int i = 0; i < nb; ++i) {
                vec3 q(wrapped[i]);
                q -= bs * glm::floor((q + boxHalf) / bs);
                wrapped[i] = glm::vec4(q, wrapped[i].w);
            }
            lo = -boxHalf;
            hiB = boxHalf;
        } else {
            for (int i = 0; i < nb; ++i) {
                vec3 q(wrapped[i]);
                loB = glm::min(loB, q);
                hiB = glm::max(hiB, q);
            }
            lo = loB;
        }
        vec3 ext = hiB - lo + vec3(1e-3f);
        float t = std::max(contact, cbrtf(ext.x * ext.y * ext.z / std::max(1, nb)));
        for (int k = 0; k < 3; ++k) {
            dim[k] = std::max(1, std::min(96, (int)(ext[k] / t)));
            cell[k] = ext[k] / dim[k];
        }
        head.assign(size_t(dim[0]) * dim[1] * dim[2], -1);
        next.assign(nb, -1);
        for (int i = 0; i < nb; ++i) {
            int ix[3];
            idx(vec3(wrapped[i]), ix);
            int c = (ix[2] * dim[1] + ix[1]) * dim[0] + ix[0];
            next[i] = head[c];
            head[c] = i;
        }
    }
    void idx(const vec3& q, int* ix) const {
        for (int k = 0; k < 3; ++k)
            ix[k] = std::clamp((int)((q[k] - lo[k]) / cell[k]), 0, dim[k] - 1);
    }
    // visit every bead in the cells within `span` of q, wrapping when periodic
    template <class F>
    void near(const vec3& q, int span, F&& fn) const {
        int base[3];
        idx(q, base);
        for (int dz = -span; dz <= span; ++dz)
            for (int dy = -span; dy <= span; ++dy)
                for (int dx = -span; dx <= span; ++dx) {
                    int c[3] = {base[0] + dx, base[1] + dy, base[2] + dz};
                    bool skip = false;
                    for (int k = 0; k < 3; ++k) {
                        if (pbc) c[k] = ((c[k] % dim[k]) + dim[k]) % dim[k];
                        else if (c[k] < 0 || c[k] >= dim[k]) { skip = true; break; }
                    }
                    if (skip) continue;
                    for (int j = head[(c[2] * dim[1] + c[1]) * dim[0] + c[0]]; j >= 0;
                         j = next[j])
                        fn(j);
                }
    }
};
}   // namespace

double WeRun::rnd() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return double(rng >> 11) / 9007199254740992.0;
}

Progress progressCoord(const std::vector<glm::vec4>& pos, int nMol, int nBeads,
                       float contact, float approachRange, bool pbc,
                       vec3 boxHalf) {
    Progress pr;
    int nb = nMol * nBeads;
    if (nb == 0) return pr;

    ContactGrid g;
    g.build(pos, nb, contact, pbc, boxHalf);

    std::vector<int> root(nMol);
    for (int i = 0; i < nMol; ++i) root[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; }
        return x;
    };
    vec3 bs = 2.f * boxHalf;
    float c2 = contact * contact;
    for (int i = 0; i < nb; ++i) {
        int mi = i / nBeads;
        vec3 qi(g.wrapped[i]);
        g.near(qi, 1, [&](int j) {
            if (j <= i) return;
            int mj = j / nBeads;
            if (mi == mj || find(mi) == find(mj)) return;
            vec3 d = vec3(g.wrapped[j]) - qi;
            if (pbc) d -= bs * glm::round(d / bs);
            if (glm::dot(d, d) < c2) root[find(mi)] = find(mj);
        });
    }
    std::vector<int> sz(nMol, 0);
    for (int i = 0; i < nMol; ++i) sz[find(i)]++;
    int best = 0;
    for (int i = 0; i < nMol; ++i)
        if (sz[find(i)] > sz[find(best)]) best = i;
    int bestRoot = find(best);
    pr.size = sz[bestRoot];

    // how near the closest non-member molecule has come to that cluster
    int span = std::max(1, (int)ceilf(approachRange /
                                      std::min({g.cell.x, g.cell.y, g.cell.z})));
    float bestD = approachRange;
    for (int i = 0; i < nb; ++i) {
        if (find(i / nBeads) == bestRoot) continue;
        vec3 qi(g.wrapped[i]);
        g.near(qi, span, [&](int j) {
            if (find(j / nBeads) != bestRoot) return;
            vec3 d = vec3(g.wrapped[j]) - qi;
            if (pbc) d -= bs * glm::round(d / bs);
            float dd = glm::length(d);
            if (dd < bestD) bestD = dd;
        });
    }
    float t = (bestD - contact) / std::max(1e-3f, approachRange - contact);
    pr.approach = std::clamp(1.f - t, 0.f, 0.999f);
    return pr;
}

int largestCluster(const std::vector<glm::vec4>& pos, int nMol, int nBeads,
                   float contact, bool pbc, vec3 boxHalf, std::vector<int>*) {
    return progressCoord(pos, nMol, nBeads, contact, contact * 2.f, pbc, boxHalf).size;
}

int WeRun::nBins() const { return std::max(1, (p.targetSize - 1) * p.subBins); }

int WeRun::binOf(float lambda) const {
    int b = (int)((lambda - 1.f) * p.subBins);
    return std::clamp(b, 0, nBins() - 1);
}


void WeRun::init(Sim& sim, const WeParams& pr, uint32_t seed) {
    p = pr;
    rng = seed * 6364136223846793005ull + 1442695040888963407ull;
    seedCounter = seed * 977u + 1u;
    tauNs = double(p.tauSteps) * sim.p.dt * sim.p.speed;

    // pool of independent dispersed starts (the reactant macrostate)
    initPool.clear();
    poolRejects = 0;
    for (int i = 0; i < p.nInitStates; ++i) {
        for (int attempt = 0; attempt < 40; ++attempt) {
            sim.p.seed = seedCounter++;
            sim.restart();
            // Run the soft-start ramp to completion, THEN clear it. restore()
            // clears burnin, so a walker resumed mid-ramp would silently jump
            // to full interaction strength -- any reference run must be
            // equalised the same way or it is handicapped exactly where the
            // events happen.
            sim.step(1500);
            sim.burnin = 0;
            sim.step(300);            // settle at full strength
            SimState s;
            sim.snapshot(s);
            s.simTimeNs = 0;
            s.totalSteps = 0;
            // A start that is ALREADY nucleated would re-cross the target the
            // instant any walker recycled onto it, manufacturing flux out of
            // nothing. The reactant basin has to be clean.
            if (largestCluster(s.pos, sim.p.nMol, sim.p.nBeads, p.contactNm,
                               sim.p.pbc != 0, sim.p.boxHalf) >= p.targetSize) {
                poolRejects++;
                continue;
            }
            initPool.push_back(std::move(s));
            break;
        }
    }
    if (initPool.empty()) return;   // caller checks; concentration is too high

    walkers.clear();
    int n0 = p.walkersPerBin;
    for (int i = 0; i < n0; ++i) {
        Walker w;
        w.st = initPool[i % initPool.size()];
        w.st.seed = seedCounter++;
        w.w = 1.0 / n0;
        w.size = 1;
        w.lambda = 1.f;
        walkers.push_back(std::move(w));
    }
    fluxWeight = 0;
    timeNs = 0;
    iters = recycles = 0;
    binPop.assign(nBins(), 0);
    binWeight.assign(nBins(), 0.0);
}

void WeRun::advanceWalker(Sim& sim, int i) {
    if (i < 0 || i >= (int)walkers.size()) return;
    Walker& w = walkers[i];
    sim.restore(w.st, false, 0);
    sim.step(p.tauSteps);
    sim.snapshot(w.st);
    Progress pr = progressCoord(w.st.pos, sim.p.nMol, sim.p.nBeads, p.contactNm,
                                p.approachNm, sim.p.pbc != 0, sim.p.boxHalf);
    w.size = pr.size;
    w.lambda = pr.value();
}

void WeRun::iterate(Sim& sim) {
    for (int i = 0; i < (int)walkers.size(); ++i) advanceWalker(sim, i);
    finishIteration(sim);
}

void WeRun::finishIteration(Sim& sim) {
    (void)sim;
    timeNs += tauNs;
    iters++;

    // harvest walkers that reached the product state, recycle their weight
    bool ss = iters > burnInIters;
    if (ss) timeNsSS += tauNs;
    for (auto& w : walkers) {
        if (w.size >= p.targetSize) {
            fluxWeight += w.w;
            recycles++;
            if (ss) { fluxWeightSS += w.w; recyclesSS++; }
            w.st = initPool[(size_t)(rnd() * initPool.size()) % initPool.size()];
            w.st.seed = seedCounter++;
            w.size = 1;
            w.lambda = 1.f;
        }
    }
    resample();

    binPop.assign(nBins(), 0);
    binWeight.assign(nBins(), 0.0);
    for (auto& w : walkers) {
        binPop[binOf(w.lambda)]++;
        binWeight[binOf(w.lambda)] += w.w;
    }
}

void WeRun::resample() {
    // split/merge to walkersPerBin in every occupied bin; weight is conserved
    // exactly, so the ensemble stays unbiased
    int nbins = nBins();
    std::vector<std::vector<int>> byBin(nbins);
    for (int i = 0; i < (int)walkers.size(); ++i) byBin[binOf(walkers[i].lambda)].push_back(i);

    std::vector<Walker> out;
    out.reserve(p.maxWalkers);
    for (int b = 0; b < nbins; ++b) {
        auto& idx = byBin[b];
        if (idx.empty()) continue;
        std::vector<Walker> grp;
        grp.reserve(idx.size());
        for (int i : idx) grp.push_back(walkers[i]);

        int target = p.walkersPerBin;
        // budget guard: never let one bin starve the others
        while ((int)grp.size() > target) {
            // merge the two lightest: survivor drawn proportional to weight
            std::sort(grp.begin(), grp.end(),
                      [](const Walker& a, const Walker& c) { return a.w < c.w; });
            double w1 = grp[0].w, w2 = grp[1].w, tot = w1 + w2;
            int keep = (rnd() * tot < w1) ? 0 : 1;
            Walker m = grp[keep];
            m.w = tot;
            grp.erase(grp.begin(), grp.begin() + 2);
            grp.push_back(m);
        }
        while ((int)grp.size() < target) {
            // split the heaviest; the clone gets its own noise stream
            auto it = std::max_element(
                grp.begin(), grp.end(),
                [](const Walker& a, const Walker& c) { return a.w < c.w; });
            Walker a = *it;
            a.w *= 0.5;
            Walker b2 = a;
            b2.st.seed = seedCounter++;
            *it = a;
            grp.push_back(b2);
        }
        for (auto& g : grp) out.push_back(std::move(g));
    }
    walkers.swap(out);

    // renormalise only against drift from floating-point merge/split rounding
    double tot = 0;
    for (auto& w : walkers) tot += w.w;
    if (tot > 0 && fabs(tot - 1.0) > 1e-12)
        for (auto& w : walkers) w.w /= tot;
}

double WeRun::ratePerNs() const {
    return timeNs > 0 ? fluxWeight / timeNs : 0.0;
}

double WeRun::rateSSPerNs() const {
    return timeNsSS > 0 ? fluxWeightSS / timeNsSS : ratePerNs();
}

double WeRun::mfptNs() const {
    double r = ratePerNs();
    return r > 0 ? 1.0 / r : 0.0;
}
