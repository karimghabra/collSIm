#define GLM_ENABLE_EXPERIMENTAL
#include "mcdock.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <glm/gtx/quaternion.hpp>

using glm::vec3;
using glm::quat;

static double chargeOfT(int type, double pH) {
    switch (type) {
        case 0: return -1.0 / (1.0 + pow(10.0, 3.65 - pH));
        case 1: return -1.0 / (1.0 + pow(10.0, 4.25 - pH));
        case 2: return +1.0 / (1.0 + pow(10.0, pH - 6.00));
        case 3: return +1.0 / (1.0 + pow(10.0, pH - 10.53));
        default: return +1.0 / (1.0 + pow(10.0, pH - 12.48));
    }
}

double McDock::rnd() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return double(rngState >> 11) / 9007199254740992.0;
}

void McDock::buildPoseLibrary(const Basis2D& b, const std::vector<float>& parRG,
                              float pH, float epsEl, float epsHy,
                              const Corr2D* corrTab, float epsCorr) {
    // combined pairwise landscape E(dz, pa, pb), then keep its local minima:
    // the collagen analog of the precomputed docking-match library
    int nt = (int)b.nTypes;
    double qs[8], w[16];
    for (int a = 0; a < nt; ++a) qs[a] = chargeOfT(a, pH);
    int kk = 0;
    for (int a = 0; a < nt; ++a)
        for (int bb = a; bb < nt; ++bb) w[kk++] = qs[a] * qs[bb];
    int nPairs = kk;
    size_t npts = size_t(b.nD) * b.nPhi * b.nPhi;
    bool useCorr = corrTab && corrTab->nD == b.nD && corrTab->nPhi == b.nPhi;
    std::vector<float> E(npts);
    for (size_t idx = 0; idx < npts; ++idx) {
        double el = 0;
        for (int k = 0; k < nPairs; ++k) el += w[k] * b.par[size_t(k) * npts + idx];
        el = std::clamp(el, -3.0, 3.0);
        E[idx] = float(epsEl * el + epsHy * parRG[2 * idx + 1] +
                       (useCorr ? epsCorr * corrTab->c[idx] : 0.f));
    }
    auto at = [&](int iz, int ia, int ib) {
        ia = (ia % (int)b.nPhi + b.nPhi) % b.nPhi;
        ib = (ib % (int)b.nPhi + b.nPhi) % b.nPhi;
        iz = std::clamp(iz, 0, (int)b.nD - 1);
        return E[(size_t(iz) * b.nPhi + ia) * b.nPhi + ib];
    };
    poses.clear();
    int izc = ((int)b.nD - 1) / 2;
    for (int iz = 1; iz < (int)b.nD - 1; ++iz)
        for (int ia = 0; ia < (int)b.nPhi; ++ia)
            for (int ib = 0; ib < (int)b.nPhi; ++ib) {
                float e = at(iz, ia, ib);
                if (e > p.poseCutoff) continue;
                if (e > at(iz - 1, ia, ib) || e > at(iz + 1, ia, ib) ||
                    e > at(iz, ia - 1, ib) || e > at(iz, ia + 1, ib) ||
                    e > at(iz, ia, ib - 1) || e > at(iz, ia, ib + 1))
                    continue;
                DockPose ps;
                ps.dz = (iz - izc) * b.dstep;
                ps.pa = (ia + 0.5f) / b.nPhi * 6.2831853f - 3.14159265f;
                ps.pb = (ib + 0.5f) / b.nPhi * 6.2831853f - 3.14159265f;
                ps.ePair = e;
                poses.push_back(ps);
            }
    std::sort(poses.begin(), poses.end(),
              [](const DockPose& x, const DockPose& y) { return x.ePair < y.ePair; });
    if ((int)poses.size() > p.maxPoses) poses.resize(p.maxPoses);
    dPeriod = b.D;
    molLen = b.L;
    printf("docking pose library: %zu poses (deepest %.2f kT/pair)\n",
           poses.size(), poses.empty() ? 0.f : poses[0].ePair);
}

void McDock::initFromScatter(int nMol, vec3 bh, float ml, bool pbcOn, uint32_t seed) {
    boxHalf = bh;
    molLen = ml;
    pbc = pbcOn;
    rngState = seed * 2654435761ull + 1442695040888963407ull;
    mols.assign(nMol, {});
    for (auto& m : mols) {
        vec3 d;
        do {
            d = vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1);
        } while (glm::dot(d, d) > 1.f || glm::dot(d, d) < 1e-4f);
        d = glm::normalize(d);
        for (int k = 0; k < 3; ++k) {
            float room = pbc ? boxHalf[k]
                             : std::max(5.f, boxHalf[k] - fabsf(d[k]) * molLen * 0.5f - 5.f);
            m.c[k] = float(rnd() * 2 - 1) * room;
        }
        m.q = glm::rotation(vec3(0, 0, 1), d);
        m.q = glm::angleAxis(float(rnd() * 6.2831853), d) * m.q;
    }
    matches.clear();
    molMatch.assign(nMol, {});
    simTimeNs = 0;
    sweepsDone = proposed = accepted = 0;
    computeStepTime();
}

void McDock::computeStepTime() {
    // translational diffusion of a 300 nm x 1.5 nm rod (Broersma/slender body):
    // D_t ~ kT ln(L/d) / (3 pi eta L)  ->  ~0.07 A^2/ns at 310 K
    double Dt = 0.007;   // nm^2/ns
    stepTimeNs = double(p.sigmaFree) * p.sigmaFree / (6.0 * Dt);
}

float McDock::capsuleDist(int iSelf, const vec3& c, const quat& q, int j) const {
    const Mol& B = mols[j];
    vec3 a1 = c - q * vec3(0, 0, molLen * 0.5f), a2 = c + q * vec3(0, 0, molLen * 0.5f);
    vec3 b1 = B.c - B.q * vec3(0, 0, molLen * 0.5f), b2 = B.c + B.q * vec3(0, 0, molLen * 0.5f);
    if (pbc) {
        vec3 bs = 2.f * boxHalf;
        vec3 sh = -bs * glm::round((0.5f * (b1 + b2) - 0.5f * (a1 + a2)) / bs);
        b1 += sh;
        b2 += sh;
    }
    vec3 d1 = a2 - a1, d2 = b2 - b1, r = a1 - b1;
    float A = glm::dot(d1, d1), E2 = glm::dot(d2, d2);
    float B2 = glm::dot(d1, d2), C = glm::dot(d1, r), F = glm::dot(d2, r);
    float den = A * E2 - B2 * B2;
    float s = den > 1e-7f ? glm::clamp((B2 * F - C * E2) / den, 0.f, 1.f) : 0.f;
    float t = E2 > 1e-7f ? (B2 * s + F) / E2 : 0.f;
    if (t < 0) { t = 0; s = glm::clamp(-C / A, 0.f, 1.f); }
    else if (t > 1) { t = 1; s = glm::clamp((B2 - C) / A, 0.f, 1.f); }
    return glm::length((a1 + s * d1) - (b1 + t * d2));
}

int McDock::neighborCount(int iSelf, const vec3& c, const quat& q) const {
    int n = 0;
    for (int j = 0; j < (int)mols.size(); ++j) {
        if (j == iSelf) continue;
        if (capsuleDist(iSelf, c, q, j) < p.contactR + p.captureR) n++;
    }
    return n;
}

void McDock::sweep(int nSweeps) {
    if (poses.empty() || mols.empty()) return;
    computeStepTime();
    float kT = std::max(0.05f, p.tempFactor);
    int N = (int)mols.size();
    for (int sw = 0; sw < nSweeps; ++sw) {
        for (int i = 0; i < N; ++i) {
            proposed++;
            // current bound energy of i
            float Ei = 0;
            for (int mi : molMatch[i]) Ei += matches[mi].E;
            int Ni = std::max(1, neighborCount(i, mols[i].c, mols[i].q));

            // propose: dock-hop to a neighbor pose, or free diffusion move
            bool dock = rnd() < 0.5;
            vec3 cNew;
            quat qNew;
            int partner = -1, poseIdx = -1;
            float Enew = 0;
            if (dock) {
                // pick a neighbor at random
                std::vector<int> nb;
                for (int j = 0; j < N; ++j)
                    if (j != i && capsuleDist(i, mols[i].c, mols[i].q, j) <
                                      p.contactR + p.captureR)
                        nb.push_back(j);
                if (nb.empty()) { continue; }
                partner = nb[std::min((size_t)(rnd() * nb.size()), nb.size() - 1)];
                poseIdx = std::min((int)(rnd() * poses.size()), (int)poses.size() - 1);
                const DockPose& ps = poses[poseIdx];
                const Mol& R = mols[partner];
                vec3 axR = R.q * vec3(0, 0, 1);
                vec3 nR = R.q * vec3(1, 0, 0);
                vec3 bR = R.q * vec3(0, 1, 0);
                // contact direction: receptor faces the ligand at angle pb
                vec3 ch = cosf(ps.pb) * nR + sinf(ps.pb) * bR;
                cNew = R.c + ps.dz * axR + p.contactR * ch;
                // ligand parallel, twisted so its facing angle toward -ch is pa
                qNew = R.q;
                // rotate about axis so that ligand normal has angle pa to -ch
                vec3 nL0 = qNew * vec3(1, 0, 0);
                float cur = atan2f(glm::dot(-ch, glm::cross(axR, nL0) * 1.f),
                                   glm::dot(-ch, nL0));
                qNew = glm::angleAxis(cur - ps.pa, axR) * qNew;
                float overlap = std::max(0.f, 1.f - fabsf(ps.dz) / molLen);
                Enew = ps.ePair * p.contactPairs * 0.1f * overlap * (molLen / 3.f);
            } else {
                float g1 = float(rnd() * 2 - 1), g2 = float(rnd() * 2 - 1), g3 = float(rnd() * 2 - 1);
                cNew = mols[i].c + vec3(g1, g2, g3) * p.sigmaFree;
                double Dr = 2.4e-5;   // rad^2/ns for the rod
                float srot = sqrtf(float(4.0 * Dr * stepTimeNs));
                vec3 ax = glm::normalize(vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1 + 1e-4));
                qNew = glm::angleAxis(float((rnd() * 2 - 1) * srot), ax) * mols[i].q;
                Enew = 0;
            }

            // collision check (skip the docking partner)
            bool clash = false;
            for (int j = 0; j < N && !clash; ++j) {
                if (j == i || j == partner) continue;
                if (capsuleDist(i, cNew, qNew, j) < p.collTol) clash = true;
            }
            if (clash) continue;

            int Nj = std::max(1, neighborCount(i, cNew, qNew));
            float dE = Enew - Ei;
            float acc = expf(std::clamp(-dE / kT, -40.f, 40.f)) * float(Ni) / float(Nj);
            if (rnd() < acc) {
                accepted++;
                // detach all current matches of i
                for (int mi : molMatch[i]) {
                    Match& mm = matches[mi];
                    int other = mm.a == i ? mm.b : mm.a;
                    auto& om = molMatch[other];
                    om.erase(std::remove(om.begin(), om.end(), mi), om.end());
                    mm.a = mm.b = -1;   // tombstone
                }
                molMatch[i].clear();
                mols[i].c = cNew;
                mols[i].q = qNew;
                if (pbc) {
                    vec3 bs = 2.f * boxHalf;
                    mols[i].c -= bs * glm::round(mols[i].c / bs);
                }
                if (partner >= 0) {
                    Match mm{i, partner, poseIdx, Enew};
                    int slot = -1;
                    for (int t = 0; t < (int)matches.size(); ++t)
                        if (matches[t].a < 0) { slot = t; break; }
                    if (slot < 0) { matches.push_back(mm); slot = (int)matches.size() - 1; }
                    else matches[slot] = mm;
                    molMatch[i].push_back(slot);
                    molMatch[partner].push_back(slot);
                }
            }
        }
        sweepsDone++;
        simTimeNs += stepTimeNs;
    }
}

void McDock::writeBeads(std::vector<glm::vec4>& out, int nBeads, float segLen) const {
    out.resize(size_t(mols.size()) * nBeads);
    for (size_t m = 0; m < mols.size(); ++m) {
        vec3 ax = mols[m].q * vec3(0, 0, 1);
        for (int i = 0; i < nBeads; ++i) {
            float t = (i - (nBeads - 1) * 0.5f) * segLen;
            out[m * nBeads + i] = glm::vec4(mols[m].c + ax * t, 0.f);
        }
    }
}

float McDock::boundFraction() const {
    int n = 0;
    for (const auto& mm : molMatch)
        if (!mm.empty()) n++;
    return mols.empty() ? 0.f : float(n) / mols.size();
}

float McDock::bandCoherence() const {
    double cs = 0, sn = 0;
    int n = 0;
    for (const auto& mm : matches) {
        if (mm.a < 0) continue;
        double ph = 6.2831853 * poses[mm.pose].dz / dPeriod;
        cs += cos(ph);
        sn += sin(ph);
        n++;
    }
    return n ? float(sqrt(cs * cs + sn * sn) / n) : 0.f;
}

int McDock::clusterCount(float& meanSize) const {
    int N = (int)mols.size();
    std::vector<int> root(N);
    for (int i = 0; i < N; ++i) root[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; }
        return x;
    };
    for (const auto& mm : matches)
        if (mm.a >= 0) root[find(mm.a)] = find(mm.b);
    std::vector<int> size(N, 0);
    for (int i = 0; i < N; ++i) size[find(i)]++;
    int nc = 0, tot = 0;
    for (int i = 0; i < N; ++i)
        if (root[i] == i && size[i] > 1) { nc++; tot += size[i]; }
    meanSize = nc ? float(tot) / nc : 0.f;
    return nc;
}
