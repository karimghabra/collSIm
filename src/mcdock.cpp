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
    // combined pairwise landscape E(dz, pa, pb): the CPU kernel every contact
    // integral samples, and whose local minima form the dock-move library
    int nt = (int)b.nTypes;
    double qs[8], w[16];
    for (int a = 0; a < nt; ++a) qs[a] = chargeOfT(a, pH);
    int kk = 0;
    for (int a = 0; a < nt; ++a)
        for (int bb = a; bb < nt; ++bb) w[kk++] = qs[a] * qs[bb];
    int nPairs = kk;
    size_t npts = size_t(b.nD) * b.nPhi * b.nPhi;
    bool useCorr = corrTab && corrTab->nD == b.nD && corrTab->nPhi == b.nPhi;
    kernel.resize(npts);
    for (size_t idx = 0; idx < npts; ++idx) {
        double el = 0;
        for (int k = 0; k < nPairs; ++k) el += w[k] * b.par[size_t(k) * npts + idx];
        el = std::clamp(el, -3.0, 3.0);
        kernel[idx] = float(epsEl * el + epsHy * parRG[2 * idx + 1] +
                            (useCorr ? epsCorr * corrTab->c[idx] : 0.f));
    }
    knD = (int)b.nD;
    knPhi = (int)b.nPhi;
    kdstep = b.dstep;
    auto at = [&](int iz, int ia, int ib) {
        ia = (ia % knPhi + knPhi) % knPhi;
        ib = (ib % knPhi + knPhi) % knPhi;
        iz = std::clamp(iz, 0, knD - 1);
        return kernel[(size_t(iz) * knPhi + ia) * knPhi + ib];
    };
    poses.clear();
    int izc = (knD - 1) / 2;
    for (int iz = 1; iz < knD - 1; ++iz)
        for (int ia = 0; ia < knPhi; ++ia)
            for (int ib = 0; ib < knPhi; ++ib) {
                float e = at(iz, ia, ib);
                if (e > p.poseCutoff) continue;
                if (e > at(iz - 1, ia, ib) || e > at(iz + 1, ia, ib) ||
                    e > at(iz, ia - 1, ib) || e > at(iz, ia + 1, ib) ||
                    e > at(iz, ia, ib - 1) || e > at(iz, ia, ib + 1))
                    continue;
                DockPose ps;
                ps.dz = (iz - izc) * b.dstep;
                ps.pa = (ia + 0.5f) / knPhi * 6.2831853f - 3.14159265f;
                ps.pb = (ib + 0.5f) / knPhi * 6.2831853f - 3.14159265f;
                ps.ePair = e;
                poses.push_back(ps);
            }
    std::sort(poses.begin(), poses.end(),
              [](const DockPose& x, const DockPose& y) { return x.ePair < y.ePair; });
    if ((int)poses.size() > p.maxPoses) poses.resize(p.maxPoses);
    dPeriod = b.D;
    molLen = b.L;
    printf("docking kernel %dx%dx%d, pose library: %zu poses (deepest %.2f kT/seg)\n",
           knD, knPhi, knPhi, poses.size(), poses.empty() ? 0.f : poses[0].ePair);
}

float McDock::kernelAt(float dz, float pa, float pb) const {
    float fz = dz / kdstep + (knD - 1) * 0.5f;
    if (fz < 0.f || fz > float(knD - 1)) return 0.f;
    int iz = std::min((int)fz, knD - 2);
    float t = fz - iz;
    auto ang = [&](float ph) {
        int i = (int)floorf((ph + 3.14159265f) / 6.2831853f * knPhi);
        return (i % knPhi + knPhi) % knPhi;
    };
    int ia = ang(pa), ib = ang(pb);
    size_t i0 = (size_t(iz) * knPhi + ia) * knPhi + ib;
    size_t i1 = (size_t(iz + 1) * knPhi + ia) * knPhi + ib;
    return kernel[i0] * (1.f - t) + kernel[i1] * t;
}

vec3 McDock::chainCenter(const Mol& st, int i) const {
    vec3 pos = st.p0;
    for (int k = 0; k < i; ++k) pos += st.q[k] * vec3(0, 0, linkLen);
    return pos + st.q[i] * vec3(0, 0, linkLen * 0.5f);
}

int McDock::cellOf(const vec3& c) const {
    int ix[3];
    for (int k = 0; k < 3; ++k) {
        int i = (int)floorf((c[k] + boxHalf[k]) / gCell[k]);
        if (pbc) i = ((i % gDim[k]) + gDim[k]) % gDim[k];
        else i = std::clamp(i, 0, gDim[k] - 1);
        ix[k] = i;
    }
    return (ix[2] * gDim[1] + ix[1]) * gDim[0] + ix[0];
}

void McDock::rebuildGrid() {
    int nL = p.nLinks, N = (int)mols.size();
    gReach = linkLen + p.contactR + std::max(p.captureR, p.cutoff) + 0.01f;
    for (int k = 0; k < 3; ++k) {
        gDim[k] = std::max(1, (int)floorf(2.f * boxHalf[k] / gReach));
        gCell[k] = 2.f * boxHalf[k] / gDim[k];
    }
    cellHead.assign(size_t(gDim[0]) * gDim[1] * gDim[2], -1);
    cellNext.assign(size_t(N) * nL, -1);
    for (int id = 0; id < N * nL; ++id) {
        int ci = cellOf(linkC[id]);
        cellNext[id] = cellHead[ci];
        cellHead[ci] = id;
    }
}

// visit every link whose cell is within one cell of c (cell >= reach, so this
// is a superset of the true neighbors); callers still distance-test
template <class F>
void McDock::forEachNear(const vec3& c, F&& fn) const {
    int idx[3][3], cnt[3];
    for (int k = 0; k < 3; ++k) {
        if (pbc && gDim[k] < 3) {   // wrap would revisit the same cells
            cnt[k] = gDim[k];
            for (int i = 0; i < gDim[k]; ++i) idx[k][i] = i;
        } else {
            int base = (int)floorf((c[k] + boxHalf[k]) / gCell[k]);
            cnt[k] = 0;
            for (int d = -1; d <= 1; ++d) {
                int i = base + d;
                if (pbc) i = ((i % gDim[k]) + gDim[k]) % gDim[k];
                else if (i < 0 || i >= gDim[k]) continue;
                idx[k][cnt[k]++] = i;
            }
        }
    }
    for (int a = 0; a < cnt[2]; ++a)
        for (int b = 0; b < cnt[1]; ++b)
            for (int d = 0; d < cnt[0]; ++d) {
                int ci = (idx[2][a] * gDim[1] + idx[1][b]) * gDim[0] + idx[0][d];
                for (int id = cellHead[ci]; id >= 0; id = cellNext[id])
                    fn(id / p.nLinks, id % p.nLinks);
            }
}

void McDock::cacheMol(int m) {
    const Mol& st = mols[m];
    vec3 pos = st.p0;
    for (int i = 0; i < p.nLinks; ++i) {
        linkC[size_t(m) * p.nLinks + i] = pos + st.q[i] * vec3(0, 0, linkLen * 0.5f);
        pos += st.q[i] * vec3(0, 0, linkLen);
    }
}

void McDock::initFromScatter(int nMol, vec3 bh, float ml, bool pbcOn, uint32_t seed) {
    boxHalf = bh;
    molLen = ml;
    pbc = pbcOn;
    p.nLinks = std::clamp(p.nLinks, 1, 16);
    linkLen = molLen / p.nLinks;
    segNorm = molLen / 95.f;   // engine bead segment: kernel is kT per this length
    rngState = seed * 2654435761ull + 1442695040888963407ull;
    mols.assign(nMol, {});
    for (auto& m : mols) {
        vec3 d;
        do {
            d = vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1);
        } while (glm::dot(d, d) > 1.f || glm::dot(d, d) < 1e-4f);
        d = glm::normalize(d);
        quat q = glm::rotation(vec3(0, 0, 1), d);
        q = glm::angleAxis(float(rnd() * 6.2831853), d) * q;
        vec3 c;
        for (int k = 0; k < 3; ++k) {
            float room = pbc ? boxHalf[k]
                             : std::max(5.f, boxHalf[k] - fabsf(d[k]) * molLen * 0.5f - 5.f);
            c[k] = float(rnd() * 2 - 1) * room;
        }
        m.p0 = c - d * (molLen * 0.5f);
        m.q.assign(p.nLinks, q);   // straight chain
    }
    linkC.assign(size_t(nMol) * p.nLinks, vec3(0));
    for (int m = 0; m < nMol; ++m) cacheMol(m);
    rebuildGrid();
    simTimeNs = 0;
    sweepsDone = proposed = accepted = 0;
    accWhole = accDock = accPivot = accSlide = 0;
    computeStepTime();
}

void McDock::computeStepTime() {
    // translational diffusion of a 300 nm x 1.5 nm rod (Broersma/slender body):
    // D_t ~ kT ln(L/d) / (3 pi eta L)  ->  ~0.007 nm^2/ns at 310 K.
    // Clock is calibrated to the BASE hop; adaptive free-molecule hops cover
    // more ground per sweep than this credits, so dilute-phase sim time is a
    // lower bound (we under-report, never over-report).
    double Dt = 0.007;   // nm^2/ns
    stepTimeNominalNs = double(p.sigmaFree) * p.sigmaFree / (6.0 * Dt);
    if (stepTimeNs <= 0) stepTimeNs = stepTimeNominalNs;   // until calibrated
}

// closest-approach between two link segments; B shifted per-pair to its
// minimum image (valid because the interaction cutoff << box size)
static float segSegDist(const vec3& cA, const vec3& aA, const vec3& cB0,
                        const vec3& aB, float hl, bool pbc, const vec3& boxHalf,
                        float& sOut, float& tOut, vec3& cBShift) {
    vec3 cB = cB0;
    if (pbc) {
        vec3 bs = 2.f * boxHalf;
        cB -= bs * glm::round((cB - cA) / bs);
    }
    cBShift = cB;
    vec3 a1 = cA - aA * hl, b1 = cB - aB * hl;
    vec3 d1 = aA * (2.f * hl), d2 = aB * (2.f * hl), r = a1 - b1;
    float A = glm::dot(d1, d1), E2 = glm::dot(d2, d2);
    float B2 = glm::dot(d1, d2), C = glm::dot(d1, r), F = glm::dot(d2, r);
    float den = A * E2 - B2 * B2;
    float s = den > 1e-7f ? glm::clamp((B2 * F - C * E2) / den, 0.f, 1.f) : 0.f;
    float t = E2 > 1e-7f ? glm::clamp((B2 * s + F) / E2, 0.f, 1.f) : 0.f;
    if (t <= 0.f || t >= 1.f)
        s = glm::clamp((t * B2 - C) / std::max(A, 1e-7f), 0.f, 1.f);
    sOut = s;
    tOut = t;
    return glm::length((a1 + s * d1) - (b1 + t * d2));
}

float McDock::linkPairE(const vec3& cA, const quat& qA, float contA,
                        int mB, int iB) const {
    const Mol& B = mols[mB];
    vec3 aA = qA * vec3(0, 0, 1);
    quat qB = B.q[iB];
    vec3 aB = qB * vec3(0, 0, 1);
    vec3 cB0 = linkC[size_t(mB) * p.nLinks + iB];
    float hl = linkLen * 0.5f;

    // quick reject on centers
    vec3 dc = cB0 - cA;
    if (pbc) {
        vec3 bs = 2.f * boxHalf;
        dc -= bs * glm::round(dc / bs);
    }
    float reach = linkLen + p.contactR + p.cutoff;
    if (glm::dot(dc, dc) > reach * reach) return 0.f;

    vec3 cB = cA + dc;
    float ca = glm::dot(aA, aB);
    float wp = ca > 0.f ? ca : 0.f;   // polarity gate (tilt verdict: ramp
    if (wp <= 0.f) return 0.f;        // prices tilt; only polarity gates)

    vec3 b1 = cB - aB * hl;
    // Integrate only over the arc of A that is actually within reach of B.
    // Separation from B's line is quadratic in arc length, so solve for the
    // interval: d^2(s) = (1-ca^2)s^2 + b2 s + c2 = R^2.
    float R = p.contactR + p.cutoff;
    vec3 w = cA - b1;
    float dw = glm::dot(w, aB);
    float a2 = 1.f - ca * ca;
    float b2 = 2.f * (glm::dot(w, aA) - dw * ca);
    float c2 = glm::dot(w, w) - dw * dw;
    float s0 = -hl, s1 = hl;
    if (a2 > 1e-6f) {
        float disc = b2 * b2 - 4.f * a2 * (c2 - R * R);
        if (disc <= 0.f) return 0.f;
        float sq = sqrtf(disc);
        s0 = std::max(s0, (-b2 - sq) / (2.f * a2));
        s1 = std::min(s1, (-b2 + sq) / (2.f * a2));
        if (s1 <= s0) return 0.f;
    } else if (c2 > R * R) {
        return 0.f;
    }
    // Sample density is set by how far the LOCAL STAGGER sweeps across the
    // contact (linkLen*(1-cos theta)), not by arc length: the measured wells
    // are 1.5-3 nm wide, so a tilted contact would alias them completely at a
    // fixed sample count. Parallel contacts sweep 0 and stay at 9.
    float arc = s1 - s0;
    const int nS = glm::clamp((int)ceilf(arc * (1.f - ca) / (0.5f * kdstep)), 9, 96);
    float ds = arc / nS;
    vec3 nA = qA * vec3(1, 0, 0), bA = qA * vec3(0, 1, 0);
    vec3 nB = qB * vec3(1, 0, 0), bB = qB * vec3(0, 1, 0);
    float contB0 = iB * linkLen;
    float E = 0.f;
    for (int k = 0; k < nS; ++k) {
        float t = s0 + (k + 0.5f) * ds;
        vec3 x = cA + aA * t;
        float proj = glm::clamp(glm::dot(x - b1, aB), 0.f, linkLen);
        vec3 y = b1 + aB * proj;
        vec3 dv = y - x;
        float d = glm::length(dv);
        if (d > p.contactR + p.cutoff || d < 1e-4f) continue;
        float u = (d - p.contactR) / p.envW;
        float env = expf(-u * u);
        float dz = (contA + t + hl) - (contB0 + proj);   // local sequence stagger
        vec3 dh = dv / d;
        float pa = atan2f(glm::dot(dh, bA), glm::dot(dh, nA));
        float pb = atan2f(glm::dot(-dh, bB), glm::dot(-dh, nB));
        E += wp * env * kernelAt(dz, pa, pb) * (ds / segNorm);
    }
    return E;
}

float McDock::molEnergy(int m, const Mol& state, bool& clash) const {
    clash = false;
    int N = (int)mols.size();
    float hl = linkLen * 0.5f;
    float reachC = linkLen + p.collTol;
    float reachE = linkLen + p.contactR + p.cutoff;
    float E = 0.f;

    // proposed link centers + axes
    vec3 cs[16];
    for (int i = 0; i < p.nLinks; ++i) cs[i] = chainCenter(state, i);

    // self sterics: non-adjacent own links must not overlap (chain folding)
    for (int i = 0; i < p.nLinks && !clash; ++i)
        for (int j = i + 2; j < p.nLinks; ++j) {
            float s, t;
            vec3 sh;
            vec3 aI = state.q[i] * vec3(0, 0, 1), aJ = state.q[j] * vec3(0, 0, 1);
            if (segSegDist(cs[i], aI, cs[j], aJ, hl, false, boxHalf, s, t, sh) <
                p.collTol) {
                clash = true;
                break;
            }
        }
    if (clash) return 0.f;

    for (int i = 0; i < p.nLinks; ++i) {
        vec3 aI = state.q[i] * vec3(0, 0, 1);
        float contI = i * linkLen;
        forEachNear(cs[i], [&](int j, int jb) {
            if (j == m || clash) return;
            vec3 dc = linkC[size_t(j) * p.nLinks + jb] - cs[i];
            if (pbc) {
                vec3 bs = 2.f * boxHalf;
                dc -= bs * glm::round(dc / bs);
            }
            if (glm::dot(dc, dc) > reachE * reachE) return;
            float s, t;
            vec3 sh;
            vec3 aJ = mols[j].q[jb] * vec3(0, 0, 1);
            float dmin = segSegDist(cs[i], aI, linkC[size_t(j) * p.nLinks + jb], aJ,
                                    hl, pbc, boxHalf, s, t, sh);
            if (dmin < p.collTol) {
                clash = true;
                return;
            }
            if (dmin < p.contactR + p.cutoff)
                E += linkPairE(cs[i], state.q[i], contI, j, jb);
        });
        if (clash) return 0.f;
    }
    return E;
}

float McDock::bendEnergy(const Mol& state) const {
    // discrete WLC joint: (Lp/l) kT (1 - cos theta)
    float kap = p.bendOn * persistLen / linkLen;
    float E = 0.f;
    for (int j = 1; j < p.nLinks; ++j) {
        vec3 a = state.q[j - 1] * vec3(0, 0, 1);
        vec3 b = state.q[j] * vec3(0, 0, 1);
        E += kap * (1.f - glm::dot(a, b));
    }
    return E;
}

void McDock::sweep(int nSweeps) {
    if (poses.empty() || mols.empty() || kernel.empty()) return;
    computeStepTime();
    float kT = std::max(0.05f, p.tempFactor);
    int N = (int)mols.size();
    int nL = p.nLinks;
    float capD = p.contactR + p.captureR;

    auto linkNbrs = [&](int self, const vec3& c, std::vector<std::pair<int, int>>* out) {
        int n = 0;
        float reach = linkLen + capD;
        forEachNear(c, [&](int j, int jb) {
            if (j == self) return;
            vec3 dc = linkC[size_t(j) * nL + jb] - c;
            if (pbc) {
                vec3 bs = 2.f * boxHalf;
                dc -= bs * glm::round(dc / bs);
            }
            if (glm::dot(dc, dc) < reach * reach) {
                n++;
                if (out) out->emplace_back(j, jb);
            }
        });
        return n;
    };

    rebuildGrid();
    std::vector<std::pair<int, int>> cand;
    std::vector<vec3> com0(N);
    std::vector<char> wasFree(N);
    auto comOf = [&](int m) {
        vec3 c(0);
        for (int i = 0; i < nL; ++i) c += linkC[size_t(m) * nL + i];
        return c / float(nL);
    };
    for (int sw = 0; sw < nSweeps; ++sw) {
        // Clock reference: molecules not in CONTACT (having a neighbour merely
        // within capture range does not arrest diffusion, so bound-ness, not
        // proximity, is the right test).
        for (int m = 0; m < N; ++m) wasFree[m] = 1;
        forEachContact(p.contactR + 1.f,
                       [&](int m, int, int j, int, float) { wasFree[m] = wasFree[j] = 0; });
        int nFree = 0;
        for (int m = 0; m < N; ++m) nFree += wasFree[m];
        double msdSweep = 0;
        for (int im = 0; im < N; ++im) {
            proposed++;
            Mol& M = mols[im];
            bool clash0 = false;
            float E0 = molEnergy(im, M, clash0) + bendEnergy(M);

            double r = rnd();
            if (nL == 1 && r >= 0.65) r -= 0.35;    // no joints to pivot
            if (!p.dockMoves && r >= 0.25 && r < 0.45) r += 0.20;
            Mol T = M;
            vec3 transDv(0);
            int kind;   // 0 transport, 1 dock, 2 pivot, 3 axial slide
            float hastings = 1.f;

            if (r < 0.25) {
                kind = 0;
                // adaptive transport hop: free molecules stride, crowded creep.
                // Uniform-cube proposals keep the Hastings ratio a volume ratio.
                bool free0 = true;
                for (int i = 0; i < nL && free0; ++i)
                    if (linkNbrs(im, linkC[size_t(im) * nL + i], nullptr)) free0 = false;
                float s1 = free0 ? p.sigmaFree * 3.f : p.sigmaFree;
                vec3 dv(float(rnd() * 2 - 1) * s1, float(rnd() * 2 - 1) * s1,
                        float(rnd() * 2 - 1) * s1);
                double Dr = 2.4e-5;   // rad^2/ns rod rotation
                float srot = sqrtf(float(4.0 * Dr * stepTimeNs));
                vec3 ax = glm::normalize(
                    vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1 + 1e-4));
                quat dq = glm::angleAxis(float((rnd() * 2 - 1) * srot), ax);
                vec3 com(0);
                for (int i = 0; i < nL; ++i) com += linkC[size_t(im) * nL + i];
                com /= float(nL);
                T.p0 = com + dq * (M.p0 - com) + dv;
                for (int i = 0; i < nL; ++i) T.q[i] = dq * M.q[i];
                transDv = dv;
                // reverse-hop width from the proposed state
                bool free1 = true;
                for (int i = 0; i < nL && free1; ++i) {
                    vec3 c = chainCenter(T, i);
                    if (linkNbrs(im, c, nullptr)) free1 = false;
                }
                float s2 = free1 ? p.sigmaFree * 3.f : p.sigmaFree;
                if (fabsf(dv.x) > s2 || fabsf(dv.y) > s2 || fabsf(dv.z) > s2)
                    continue;   // reverse proposal impossible
                hastings = (s1 * s1 * s1) / (s2 * s2 * s2);
            } else if (r < 0.45) {
                kind = 1;
                // dock hop: place one link at a library pose on a neighbor
                // link, drag the rest of the chain rigidly (bend preserved)
                int a = std::min(int(rnd() * nL), nL - 1);
                vec3 cA = linkC[size_t(im) * nL + a];
                cand.clear();
                int na = linkNbrs(im, cA, &cand);
                if (!na) continue;
                auto [pm, pl] = cand[std::min(size_t(rnd() * na), cand.size() - 1)];
                const DockPose& ps = poses[std::min(int(rnd() * poses.size()),
                                                    (int)poses.size() - 1)];
                quat qR = mols[pm].q[pl];
                vec3 cR = linkC[size_t(pm) * nL + pl];
                if (pbc) {
                    vec3 bs = 2.f * boxHalf;
                    cR -= bs * glm::round((cR - cA) / bs);
                }
                vec3 axR = qR * vec3(0, 0, 1);
                vec3 ch = cosf(ps.pb) * (qR * vec3(1, 0, 0)) +
                          sinf(ps.pb) * (qR * vec3(0, 1, 0));
                // molecule-frame stagger ps.dz realized locally at the contact
                float u = (a + 0.5f) * linkLen - (pl + 0.5f) * linkLen - ps.dz;
                vec3 cT = cR + u * axR + p.contactR * ch;
                quat qT = qR;
                vec3 nL0 = qT * vec3(1, 0, 0);
                float cur = atan2f(glm::dot(-ch, glm::cross(axR, nL0)),
                                   glm::dot(-ch, nL0));
                qT = glm::angleAxis(cur - ps.pa, axR) * qT;
                quat dq = qT * glm::inverse(M.q[a]);
                T.p0 = cT + dq * (M.p0 - cA);
                for (int i = 0; i < nL; ++i) T.q[i] = dq * M.q[i];
                int nb2 = linkNbrs(im, chainCenter(T, a), nullptr);
                hastings = float(na) / float(std::max(1, nb2));
            } else if (r < 0.65) {
                kind = 3;
                // axial slide: translate the chain along a link's own tangent.
                // Sweeps a thin tube, so it survives the dense gel where dock
                // hops cannot -- this is what anneals D-registry once bound.
                // Symmetric proposal (both signs, symmetric jump set).
                int i = std::min(int(rnd() * nL), nL - 1);
                vec3 ax = M.q[i] * vec3(0, 0, 1);
                float d;
                if (rnd() < 0.5) d = float((rnd() * 2 - 1) * 4.0);
                else d = dPeriod * (rnd() < 0.5 ? 1.f : 2.f) * (rnd() < 0.5 ? 1.f : -1.f);
                T.p0 = M.p0 + ax * d;
            } else {
                kind = 2;
                // pivot: rotate everything on one side of a joint about it
                int j = 1 + std::min(int(rnd() * (nL - 1)), nL - 2);
                bool down = rnd() < 0.5;
                vec3 jp = M.p0;
                for (int k = 0; k < j; ++k) jp += M.q[k] * vec3(0, 0, linkLen);
                vec3 ax = glm::normalize(
                    vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1 + 1e-4));
                quat dq = glm::angleAxis(float((rnd() * 2 - 1) * p.pivotAmp), ax);
                if (down) {
                    for (int i = j; i < nL; ++i) T.q[i] = dq * M.q[i];
                } else {
                    for (int i = 0; i < j; ++i) T.q[i] = dq * M.q[i];
                    T.p0 = jp + dq * (M.p0 - jp);
                }
            }

            bool clash1 = false;
            float E1 = molEnergy(im, T, clash1);
            if (clash1) continue;
            E1 += bendEnergy(T);
            float dE = E1 - E0;
            float acc = expf(std::clamp(-dE / kT, -40.f, 40.f)) * hastings;
            if (rnd() < acc) {
                accepted++;
                if (kind == 0) accWhole++;
                else if (kind == 1) accDock++;
                else if (kind == 3) accSlide++;
                else accPivot++;
                if (kind == 0 && wasFree[im]) msdSweep += glm::dot(transDv, transDv);
                M = T;
                if (pbc) {
                    vec3 com(0);
                    for (int i = 0; i < nL; ++i) com += chainCenter(M, i);
                    com /= float(nL);
                    vec3 bs = 2.f * boxHalf;
                    M.p0 -= bs * glm::round(com / bs);
                }
                cacheMol(im);
                rebuildGrid();
            }
        }
        // Calibrate on the DIFFUSIVE channel only: accepted transport hops of
        // free molecules. Axial D-jumps and dock hops also displace molecules,
        // often much further, but they are smart moves -- crediting them as
        // diffusion would inflate the clock badly (measured: 18x). They buy
        // sampling, not time, so structures here run ahead of the clock.
        lastFreeCount = nFree;
        clockExtrapolated = (nFree == 0);
        if (nFree > 0) {
            lastMsdFree = msdSweep / nFree;
            if (lastMsdFree > 0) stepTimeNs = lastMsdFree / (6.0 * 0.007);
        }
        sweepsDone++;
        simTimeNs += stepTimeNs;
    }
}

void McDock::writeBeads(std::vector<glm::vec4>& out, int nBeads, float segLen) const {
    out.resize(size_t(mols.size()) * nBeads);
    for (size_t m = 0; m < mols.size(); ++m) {
        const Mol& M = mols[m];
        vec3 pos = M.p0;
        int li = 0;
        float base = 0.f;
        for (int i = 0; i < nBeads; ++i) {
            float t = i * segLen;
            while (li < p.nLinks - 1 && t > base + linkLen) {
                pos += M.q[li] * vec3(0, 0, linkLen);
                base += linkLen;
                li++;
            }
            out[m * nBeads + i] =
                glm::vec4(pos + M.q[li] * vec3(0, 0, t - base), 0.f);
        }
    }
}

// enumerate foreign link pairs closer than maxD; fn(mA,iA,mB,iB,dzLocal)
template <class F>
void McDock::forEachContact(float maxD, F&& fn) const {
    int N = (int)mols.size();
    int nL = p.nLinks;
    float hl = linkLen * 0.5f;
    float reach = linkLen + maxD;
    if (cellHead.empty()) return;
    for (int m = 0; m < N; ++m)
        for (int i = 0; i < nL; ++i) {
            vec3 cA = linkC[size_t(m) * nL + i];
            vec3 aA = mols[m].q[i] * vec3(0, 0, 1);
            forEachNear(cA, [&](int j, int jb) {
                if (j <= m) return;
                vec3 dc = linkC[size_t(j) * nL + jb] - cA;
                if (pbc) {
                    vec3 bs = 2.f * boxHalf;
                    dc -= bs * glm::round(dc / bs);
                }
                if (glm::dot(dc, dc) > reach * reach) return;
                float s, t;
                vec3 cB;
                vec3 aB = mols[j].q[jb] * vec3(0, 0, 1);
                float d = segSegDist(cA, aA, linkC[size_t(j) * nL + jb], aB, hl,
                                     pbc, boxHalf, s, t, cB);
                if (d > maxD) return;
                float contA = i * linkLen + s * linkLen;
                float contB = jb * linkLen + t * linkLen;
                fn(m, i, j, jb, contA - contB);
            });
        }
}

float McDock::boundFraction() const {
    std::vector<char> bound(mols.size(), 0);
    forEachContact(p.contactR + 1.f,
                   [&](int m, int, int j, int, float) { bound[m] = bound[j] = 1; });
    int n = 0;
    for (char b : bound) n += b;
    return mols.empty() ? 0.f : float(n) / mols.size();
}

float McDock::bandCoherence() const {
    double cs = 0, sn = 0;
    int n = 0;
    forEachContact(p.contactR + 1.f, [&](int, int, int, int, float dz) {
        double ph = 6.2831853 * dz / dPeriod;
        cs += cos(ph);
        sn += sin(ph);
        n++;
    });
    return n ? float(sqrt(cs * cs + sn * sn) / n) : 0.f;
}

float McDock::dFraction(float tol) const {
    int n = 0, hit = 0;
    forEachContact(p.contactR + 1.f, [&](int, int, int, int, float dz) {
        float r = fmodf(fabsf(dz), dPeriod);
        if (r > dPeriod * 0.5f) r = dPeriod - r;
        n++;
        if (r < tol) hit++;
    });
    return n ? float(hit) / n : 0.f;
}

void McDock::dumpContacts(int maxN) const {
    int n = 0;
    forEachContact(p.contactR + 1.f, [&](int m, int i, int j, int jb, float dz) {
        if (n++ < maxN)
            printf("  contact m%d.l%d - m%d.l%d : dz %+7.1f nm (%.2f D)\n", m, i, j,
                   jb, dz, dz / dPeriod);
    });
    printf("  %d contacts total\n", n);
}

int McDock::clusterCount(float& meanSize) const {
    int N = (int)mols.size();
    std::vector<int> root(N);
    for (int i = 0; i < N; ++i) root[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; }
        return x;
    };
    forEachContact(p.contactR + 1.f,
                   [&](int m, int, int j, int, float) { root[find(m)] = find(j); });
    std::vector<int> size(N, 0);
    for (int i = 0; i < N; ++i) size[find(i)]++;
    int nc = 0, tot = 0;
    for (int i = 0; i < N; ++i)
        if (root[i] == i && size[i] > 1) { nc++; tot += size[i]; }
    meanSize = nc ? float(tot) / nc : 0.f;
    return nc;
}
