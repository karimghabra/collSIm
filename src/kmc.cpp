#include "kmc.h"
#include <algorithm>
#include <cmath>
#include <random>

static double chargeOfT(int type, double pH) {
    switch (type) {
        case 0: return -1.0 / (1.0 + pow(10.0, 3.65 - pH));
        case 1: return -1.0 / (1.0 + pow(10.0, 4.25 - pH));
        case 2: return +1.0 / (1.0 + pow(10.0, pH - 6.00));
        case 3: return +1.0 / (1.0 + pow(10.0, pH - 10.53));
        default: return +1.0 / (1.0 + pow(10.0, pH - 12.48));
    }
}

static void pairWeights(const Basis2D& b, float pH, double* w, int& nPairs) {
    int nt = (int)b.nTypes;
    double qs[8];
    for (int a = 0; a < nt; ++a) qs[a] = chargeOfT(a, pH);
    int k = 0;
    for (int a = 0; a < nt; ++a)
        for (int bb = a; bb < nt; ++bb) w[k++] = qs[a] * qs[bb];
    nPairs = k;
}

KmcProfile kmcProfile1D(const Basis2D& b, const std::vector<float>& parRG,
                        float pH, float epsEl, float epsHy,
                        const Corr2D* corr, float epsCorr) {
    double w[16];
    int nPairs;
    pairWeights(b, pH, w, nPairs);
    size_t npts = size_t(b.nD) * b.nPhi * b.nPhi;
    bool useCorr = corr && corr->nD == b.nD && corr->nPhi == b.nPhi;
    KmcProfile pr;
    pr.dstep = b.dstep;
    pr.D = b.D;
    pr.L = b.L;
    // angular FREE ENERGY, not min: F(D) = -ln < exp(-E) >_angles. Broad
    // basins win entropically -- the correct statistics for a sampling engine
    // (min-over-angles overweights measure-zero coincidental minima)
    pr.E.resize(b.nD);
    for (int iz = 0; iz < (int)b.nD; ++iz) {
        double z = 0;
        int na = (int)b.nPhi * (int)b.nPhi;
        for (int ia = 0; ia < (int)b.nPhi; ++ia)
            for (int ib = 0; ib < (int)b.nPhi; ++ib) {
                size_t idx = (size_t(iz) * b.nPhi + ia) * b.nPhi + ib;
                double el = 0;
                for (int kk = 0; kk < nPairs; ++kk)
                    el += w[kk] * b.par[size_t(kk) * npts + idx];
                el = std::clamp(el, -3.0, 3.0);
                double e = epsEl * el + epsHy * parRG[2 * idx + 1];
                if (useCorr) e += epsCorr * corr->c[idx];
                z += exp(std::clamp(-e, -30.0, 30.0));
            }
        pr.E[iz] = (float)(-log(std::max(z / na, 1e-300)));
    }
    return pr;
}

KmcProfile kmcProfileFunnel(const Profiles& p1, float epsEl, float epsHy,
                            float D, float L) {
    KmcProfile pr;
    pr.dstep = p1.ds;
    pr.D = D;
    pr.L = L;
    pr.E.resize(p1.nPar);
    for (uint32_t i = 0; i < p1.nPar; ++i)
        pr.E[i] = epsEl * p1.elPar[i] + epsHy * p1.hyPar[i];
    return pr;
}

double kmcWellDepth(const Basis2D& b, const std::vector<float>& parRG,
                    float pH, float epsEl, float epsHy,
                    const Corr2D* corr, float epsCorr) {
    KmcProfile pr = kmcProfile1D(b, parRG, pH, epsEl, epsHy, corr, epsCorr);
    int izc = (int)((pr.D / pr.dstep) + (b.nD - 1) * 0.5);
    int win = (int)(6.f / pr.dstep);
    float best = 0;
    for (int iz = std::max(0, izc - win); iz <= std::min((int)b.nD - 1, izc + win); ++iz)
        best = std::min(best, pr.E[iz]);
    return -best;
}

void kmcFrameToBeads(const KmcFrame& f, const KmcProfile& prof, int nMolMax,
                     int nBeads, float segLen, const float boxHalf[3],
                     uint32_t seed, std::vector<glm::vec4>& out, int& nUsed) {
    out.assign(size_t(nMolMax) * nBeads, glm::vec4(0));
    // Molecules are laid onto a quasi-hexagonal microfibril lattice in the
    // order they attached, each at ITS OWN KMC-sampled stagger. The lattice
    // and the attachment order are a reconstruction; the staggers -- the part
    // that produces D-banding -- are the rigorous output of the kinetics.
    const int nFiles = 7;
    const float fileR = 3.8f;          // lateral spacing of files (nm)
    float rodLen = prof.L;
    uint32_t rs = seed * 2654435761u + 1u;
    auto rnd = [&] {
        rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
        return (rs >> 8) * (1.0f / 16777216.0f);
    };
    int izc = (int)prof.E.size() ? ((int)prof.E.size() - 1) / 2 : 0;
    auto stagNm = [&](int iz) { return (iz - izc) * prof.dstep; };

    int m = 0;
    for (size_t fi = 0; fi < f.fibStag.size() && m < nMolMax; ++fi) {
        // each fibril gets its own axis and origin in the box
        float ax = (rnd() * 2 - 1), ay = (rnd() * 2 - 1), az = 1.0f + rnd();
        glm::vec3 axis = glm::normalize(glm::vec3(ax * 0.35f, ay * 0.35f, az));
        glm::vec3 up = fabsf(axis.z) < 0.9f ? glm::vec3(0, 0, 1) : glm::vec3(1, 0, 0);
        glm::vec3 e1 = glm::normalize(glm::cross(axis, up));
        glm::vec3 e2 = glm::cross(axis, e1);
        glm::vec3 org((rnd() * 2 - 1) * boxHalf[0] * 0.55f,
                      (rnd() * 2 - 1) * boxHalf[1] * 0.55f,
                      (rnd() * 2 - 1) * boxHalf[2] * 0.35f);
        float zRun[nFiles] = {0};
        const auto& stg = f.fibStag[fi];
        for (size_t k = 0; k < stg.size() && m < nMolMax; ++k, ++m) {
            int file = int(k % nFiles);
            float ang = 6.2831853f * file / nFiles;
            glm::vec3 lat = (e1 * cosf(ang) + e2 * sinf(ang)) * fileR;
            // Hodge-Petruska: neighbouring files sit one D-period apart
            float z0 = zRun[file] + ((file * 3) % 5) * prof.D;
            zRun[file] += std::max(8.f, fabsf(stagNm(stg[k])));
            glm::vec3 base = org + lat + axis * z0;
            for (int b = 0; b < nBeads; ++b)
                out[size_t(m) * nBeads + b] =
                    glm::vec4(base + axis * (b * segLen), 0.f);
        }
    }
    // remaining slots are free monomers, scattered and randomly oriented
    for (; m < nMolMax; ++m) {
        glm::vec3 d;
        do {
            d = glm::vec3(rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1);
        } while (glm::dot(d, d) > 1.f || glm::dot(d, d) < 1e-4f);
        d = glm::normalize(d);
        glm::vec3 c((rnd() * 2 - 1) * boxHalf[0], (rnd() * 2 - 1) * boxHalf[1],
                    (rnd() * 2 - 1) * boxHalf[2]);
        glm::vec3 base = c - d * (rodLen * 0.5f);
        for (int b = 0; b < nBeads; ++b)
            out[size_t(m) * nBeads + b] = glm::vec4(base + d * (b * segLen), 0.f);
    }
    nUsed = m;
}

KmcResult kmcRun(const KmcParams& p, const KmcProfile& prof, const KmcProfile& profRef,
                 double concMgMl, double mwKda, double tempC, uint32_t seed,
                 std::vector<KmcFrame>* frames, int maxFrames) {
    KmcResult r;
    double kT = (273.15 + tempC) / 310.15;
    int nBins = (int)prof.E.size();
    int izc = (nBins - 1) / 2;                       // stagger 0 index
    auto stagOf = [&](int iz) { return (iz - izc) * prof.dstep; };

    // per-molecule binding energy vs stagger: contactPairs*0.1 is the effective
    // growth-zone coupling (a freshly docked molecule engages only part of its
    // eventual contact while registering), keeping wells in the few-kT regime
    // where kinetic-vs-thermodynamic selection is a real competition
    auto eMol = [&](int iz) { return p.contactPairs * 0.1 * prof.E[iz]; };
    double eMin = 1e9;
    for (int iz = 0; iz < nBins; ++iz) eMin = std::min(eMin, eMol(iz));

    double depthCur = -eMin;                                  // kT, molecule level
    double eMinRef = 1e9;
    for (float e : profRef.E) eMinRef = std::min(eMinRef, (double)(p.contactPairs * 0.1 * e));
    double depthRef = -eMinRef;
    double dd = (depthRef - depthCur) * (p.dGscale / 5.0);    // binding deficit
    r.fOff = exp(std::clamp(dd / kT, -30.0, 30.0));
    r.fNuc = exp(std::clamp(-dd * 0.5 * p.nc / kT, -60.0, 30.0));
    double fOn = std::max(0.2, 1.0 + 0.02 * (tempC - 37.0));

    double volL = p.volUm3 * 1e-15;
    double NAV = 6.022e23 * volL;
    double c0 = concMgMl / (mwKda * 1000.0);
    int64_t nTotal = (int64_t)(c0 * NAV);
    if (nTotal < 100) nTotal = 100;
    int64_t nFree = nTotal, fibMass = 0;

    struct Fib {
        std::vector<int> stag;     // profile bin per molecule (LIFO growth zone)
        double koffTop = 0;
    };
    std::vector<Fib> fib;

    std::mt19937_64 rng(seed);
    auto u01 = [&] { return (rng() >> 11) * (1.0 / 9007199254740992.0); };

    // D-phase coherence accumulators (updated O(1) per event)
    double cSum = 0, sSum = 0;
    auto phase = [&](int iz) { return 6.2831853 * stagOf(iz) / prof.D; };

    auto koffOf = [&](int iz) {
        return p.koff0 * r.fOff * exp(std::clamp((eMol(iz) - eMin) / kT, 0.0, 25.0));
    };

    // Boltzmann stagger sampling within a scan window around a uniform dock
    int winBins = std::max(2, (int)(p.scanWinNm / prof.dstep));
    std::vector<double> wbuf(2 * winBins + 1);
    auto sampleStagger = [&]() {
        double overlap;
        int dock;
        do {
            dock = (int)(u01() * nBins);
            overlap = 1.0 - fabs(stagOf(dock)) / prof.L;   // encounter geometry
        } while (overlap < u01());
        int lo = std::max(0, dock - winBins), hi = std::min(nBins - 1, dock + winBins);
        double Z = 0;
        for (int iz = lo; iz <= hi; ++iz) {
            wbuf[iz - lo] = exp(std::clamp(-eMol(iz) / kT, -40.0, 40.0));
            Z += wbuf[iz - lo];
        }
        double x = u01() * Z;
        for (int iz = lo; iz <= hi; ++iz) {
            x -= wbuf[iz - lo];
            if (x <= 0) return iz;
        }
        return hi;
    };

    double t = 0, tMax = p.tMaxMin * 60.0;
    double record = tMax / 900.0, nextRec = 0;
    double frameEvery = tMax / std::max(1, maxFrames), nextFrame = 0;
    double koffSum = 0;   // sum of top-of-stack koff over fibrils

    auto pushMol = [&](Fib& f, int iz) {
        koffSum -= f.koffTop;
        f.stag.push_back(iz);
        f.koffTop = koffOf(iz);
        koffSum += f.koffTop;
        cSum += cos(phase(iz));
        sSum += sin(phase(iz));
    };
    auto popMol = [&](Fib& f) {
        int iz = f.stag.back();
        f.stag.pop_back();
        koffSum -= f.koffTop;
        f.koffTop = f.stag.empty() ? 0 : koffOf(f.stag.back());
        koffSum += f.koffTop;
        cSum -= cos(phase(iz));
        sSum -= sin(phase(iz));
    };
    auto dissolve = [&](size_t i) {
        Fib& f = fib[i];
        for (int iz : f.stag) { cSum -= cos(phase(iz)); sSum -= sin(phase(iz)); }
        nFree += (int64_t)f.stag.size();
        fibMass -= (int64_t)f.stag.size();
        koffSum -= f.koffTop;
        fib.erase(fib.begin() + i);
    };

    auto recordPt = [&] {
        r.t.push_back(float(t / 60.0));
        r.fibrilMass.push_back(float(double(fibMass) / nTotal));
        r.freeMono.push_back(float(double(nFree) / nTotal));
        r.nFib.push_back(float(fib.size()));
        double coh = fibMass > 0 ? sqrt(cSum * cSum + sSum * sSum) / double(fibMass) : 0;
        r.bandOrder.push_back(float(coh));
    };

    while (t < tMax && r.events < 4'000'000) {
        double cFree = double(nFree) / NAV;
        double Rnuc = (nFree >= p.nc) ? p.kn * r.fNuc * pow(cFree, p.nc - 1) * cFree * NAV * fOn : 0.0;
        double Rel = 2.0 * p.kplus * fOn * cFree * double(fib.size());
        double Roff = koffSum;
        double internal = double(fibMass) - double(fib.size());
        double Rfr = p.kfrag * r.fOff * std::max(0.0, internal);
        double R = Rnuc + Rel + Roff + Rfr;
        if (R < 1e-12) { t = tMax; break; }
        t += -log(std::max(u01(), 1e-300)) / R;
        while (t >= nextRec && nextRec <= tMax) { recordPt(); nextRec += record; }
        if (frames && t >= nextFrame && nextFrame <= tMax) {
            KmcFrame kf;
            kf.tMin = float(t / 60.0);
            kf.freeFrac = float(double(nFree) / nTotal);
            kf.massFrac = float(double(fibMass) / nTotal);
            kf.fibStag.reserve(fib.size());
            for (const auto& f : fib) kf.fibStag.push_back(f.stag);
            frames->push_back(std::move(kf));
            while (t >= nextFrame) nextFrame += frameEvery;
        }
        double x = u01() * R;
        if (x < Rnuc) {
            fib.emplace_back();
            for (int m = 0; m < p.nc; ++m) pushMol(fib.back(), sampleStagger());
            nFree -= p.nc;
            fibMass += p.nc;
        } else if (x < Rnuc + Rel) {
            size_t i = std::min(fib.size() - 1, size_t(u01() * fib.size()));
            pushMol(fib[i], sampleStagger());
            nFree--; fibMass++;
        } else if (x < Rnuc + Rel + Roff) {
            double y = u01() * std::max(koffSum, 1e-300);
            for (size_t i = 0; i < fib.size(); ++i) {
                y -= fib[i].koffTop;
                if (y <= 0 || i == fib.size() - 1) {
                    popMol(fib[i]);
                    nFree++; fibMass--;
                    if ((int)fib[i].stag.size() < p.nc) dissolve(i);
                    break;
                }
            }
        } else {
            double pick = u01() * std::max(1.0, internal);
            for (size_t i = 0; i < fib.size(); ++i) {
                pick -= double(fib[i].stag.size() - 1);
                if (pick <= 0) {
                    size_t cut = 1 + size_t(u01() * (fib[i].stag.size() - 1));
                    Fib rest;
                    rest.stag.assign(fib[i].stag.begin() + cut, fib[i].stag.end());
                    fib[i].stag.resize(cut);
                    koffSum -= fib[i].koffTop;
                    fib[i].koffTop = koffOf(fib[i].stag.back());
                    koffSum += fib[i].koffTop;
                    if ((int)rest.stag.size() >= p.nc) {
                        rest.koffTop = koffOf(rest.stag.back());
                        koffSum += rest.koffTop;
                        fib.push_back(std::move(rest));
                    } else {
                        for (int iz : rest.stag) { cSum -= cos(phase(iz)); sSum -= sin(phase(iz)); }
                        nFree += (int64_t)rest.stag.size();
                        fibMass -= (int64_t)rest.stag.size();
                    }
                    if ((int)fib[i].stag.size() < p.nc) dissolve(i);
                    break;
                }
            }
        }
        r.events++;
    }
    while (nextRec <= tMax) { recordPt(); nextRec += record; }

    // final stagger histogram, mod D
    r.stagHist.assign(24, 0.f);
    for (const Fib& f : fib)
        for (int iz : f.stag) {
            double m = fmod(fmod(stagOf(iz), (double)prof.D) + prof.D, (double)prof.D);
            r.stagHist[std::min(23, (int)(m / prof.D * 24))] += 1.f;
        }

    r.plateau = r.fibrilMass.empty() ? 0 : r.fibrilMass.back();
    r.finalOrder = r.bandOrder.empty() ? 0 : r.bandOrder.back();
    for (size_t i = 0; i < r.t.size(); ++i) {
        if (r.lagMin < 0 && r.fibrilMass[i] > 0.1 * std::max(r.plateau, 1e-6))
            r.lagMin = r.t[i];
        if (r.t50Min < 0 && r.fibrilMass[i] > 0.5 * std::max(r.plateau, 1e-6))
            r.t50Min = r.t[i];
    }
    return r;
}
