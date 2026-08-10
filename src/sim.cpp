#include "sim.h"
#include <cstdio>
#include <random>
#include <unordered_map>
#include <vector>

using glm::vec3;
using glm::vec4;

static GLuint makeSSBO(size_t bytes, const void* data = nullptr) {
    GLuint b;
    glCreateBuffers(1, &b);
    glNamedBufferData(b, bytes, data, GL_DYNAMIC_DRAW);
    return b;
}

static float chargeOf(int type, float pH) {
    switch (type) {
        case 0: return -1.f / (1.f + powf(10.f, 3.65f - pH));    // Asp
        case 1: return -1.f / (1.f + powf(10.f, 4.25f - pH));    // Glu
        case 2: return +1.f / (1.f + powf(10.f, pH - 6.00f));    // His
        case 3: return +1.f / (1.f + powf(10.f, pH - 10.53f));   // Lys
        default: return +1.f / (1.f + powf(10.f, pH - 12.48f));  // Arg
    }
}

float Sim::netCharge() const {
    float q = 0;
    const float* c = (p.atelo && hasAtelo) ? tCounts2 : tCounts;
    for (int a = 0; a < nTypes; ++a) q += c[a] * chargeOf(a, p.pH);
    return q;
}

void Sim::recombinePH() {
    float qs[8];
    for (int a = 0; a < nTypes; ++a) qs[a] = chargeOf(a, p.pH);
    float w[16];
    int k = 0;
    for (int a = 0; a < nTypes; ++a)
        for (int b = a; b < nTypes; ++b) w[k++] = qs[a] * qs[b];
    prRecomb.use();
    glUniform1fv(prRecomb.loc("w"), nPairsB, w);
    prRecomb.set("nPairs", nPairsB);
    bool at = p.atelo == 1 && hasAtelo;
    for (int pass = 0; pass < 2; ++pass) {
        int nz = pass == 0 ? p2nD : p2nR;
        int nTexel = nz * p2nPhi * p2nPhi;
        prRecomb.set("nTexel", nTexel);
        GLuint bb = pass == 0 ? (at ? basisParBuf2 : basisParBuf)
                              : (at ? basisApBuf2 : basisApBuf);
        GLuint hb = pass == 0 ? (at ? hyParBuf2 : hyParBuf)
                              : (at ? hyApBuf2 : hyApBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, bb);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, hb);
        glBindImageTexture(0, pass == 0 ? texPar2 : texAp2, 0, GL_TRUE, 0,
                           GL_WRITE_ONLY, GL_RG32F);
        glDispatchCompute((nTexel + 255) / 256, 1, 1);
    }
    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Sim::init(const AtomTemplate& tmpl, const Profiles& prof, const Profiles2D& prof2,
               const Basis2D& basis, const Profiles2D* prof2A, const Basis2D* basisA) {
    molLen = tmpl.lengthNm;
    dPeriod = tmpl.dPeriodNm;
    profDs = prof.ds;
    nPar = prof.nPar;
    nAp = prof.nAp;
    p2Dstep = prof2.dstep;
    p2S0 = prof2.S0;
    p2nD = prof2.nD;
    p2nPhi = prof2.nPhi;
    p2nR = prof2.nR;

    // 3D azimuthal tables: x = phiB, y = phiA (both periodic), z = stagger
    auto make3d = [&](GLuint& tex, const std::vector<float>& data, int nz) {
        glCreateTextures(GL_TEXTURE_3D, 1, &tex);
        glTextureStorage3D(tex, 1, GL_RG32F, p2nPhi, p2nPhi, nz);
        glTextureSubImage3D(tex, 0, 0, 0, 0, p2nPhi, p2nPhi, nz,
                            GL_RG, GL_FLOAT, data.data());
        glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(tex, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    };
    make3d(texPar2, prof2.par, p2nD);
    make3d(texAp2, prof2.ap, p2nR);

    // pH machinery: el basis tables + static hy channels
    nTypes = basis.nTypes;
    nPairsB = basis.nPairs();
    mwKda = basis.mwKda;
    for (int a = 0; a < nTypes && a < 8; ++a) tCounts[a] = basis.counts[a];
    basisParBuf = makeSSBO(basis.par.size() * 4, basis.par.data());
    basisApBuf = makeSSBO(basis.ap.size() * 4, basis.ap.data());
    {
        std::vector<float> hp(prof2.par.size() / 2), ha(prof2.ap.size() / 2);
        for (size_t i = 0; i < hp.size(); ++i) hp[i] = prof2.par[2 * i + 1];
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = prof2.ap[2 * i + 1];
        hyParBuf = makeSSBO(hp.size() * 4, hp.data());
        hyApBuf = makeSSBO(ha.size() * 4, ha.data());
    }
    if (prof2A && basisA && basisA->nD == basis.nD && basisA->nR == basis.nR) {
        hasAtelo = true;
        for (int a = 0; a < nTypes && a < 8; ++a) tCounts2[a] = basisA->counts[a];
        basisParBuf2 = makeSSBO(basisA->par.size() * 4, basisA->par.data());
        basisApBuf2 = makeSSBO(basisA->ap.size() * 4, basisA->ap.data());
        std::vector<float> hp(prof2A->par.size() / 2), ha(prof2A->ap.size() / 2);
        for (size_t i = 0; i < hp.size(); ++i) hp[i] = prof2A->par[2 * i + 1];
        for (size_t i = 0; i < ha.size(); ++i) ha[i] = prof2A->ap[2 * i + 1];
        hyParBuf2 = makeSSBO(hp.size() * 4, hp.data());
        hyApBuf2 = makeSSBO(ha.size() * 4, ha.data());
    }

    // registry-energy textures: R = electrostatic, G = hydrophobic
    std::vector<float> par(nPar * 2), ap(nAp * 2);
    for (int i = 0; i < nPar; ++i) { par[2*i] = prof.elPar[i]; par[2*i+1] = prof.hyPar[i]; }
    for (int i = 0; i < nAp; ++i)  { ap[2*i] = prof.elAp[i];   ap[2*i+1] = prof.hyAp[i]; }
    glCreateTextures(GL_TEXTURE_1D, 1, &texPar);
    glTextureStorage1D(texPar, 1, GL_RG32F, nPar);
    glTextureSubImage1D(texPar, 0, 0, nPar, GL_RG, GL_FLOAT, par.data());
    glCreateTextures(GL_TEXTURE_1D, 1, &texAp);
    glTextureStorage1D(texAp, 1, GL_RG32F, nAp);
    glTextureSubImage1D(texAp, 0, 0, nAp, GL_RG, GL_FLOAT, ap.data());
    for (GLuint t : {texPar, texAp}) {
        glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTextureParameteri(t, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }

    prCount.computeFromFile("shaders/grid_count.comp");
    prScan1.computeFromFile("shaders/grid_scan1.comp");
    prScan2.computeFromFile("shaders/grid_scan2.comp");
    prScan3.computeFromFile("shaders/grid_scan3.comp");
    prFill.computeFromFile("shaders/grid_fill.comp");
    prForces.computeFromFile("shaders/forces.comp");
    prInteg.computeFromFile("shaders/integrate.comp");
    prXpbd.computeFromFile("shaders/xpbd.comp");
    prFrames.computeFromFile("shaders/frames.comp");
    prRot.computeFromFile("shaders/rot_integrate.comp");
    prRecomb.computeFromFile("shaders/recombine.comp");
    prSmooth.computeFromFile("shaders/smooth.comp");
    prCenters.computeFromFile("shaders/mol_centers.comp");
    prNbr.computeFromFile("shaders/nbr_collect.comp");
    prRecenter.computeFromFile("shaders/recenter.comp");
    prMc.computeFromFile("shaders/mc_moves.comp");

    restart();
    recombinePH();
}

void Sim::restart() {
    segLen = molLen / float(p.nBeads - 1);
    if (p.scenario == 2) { p.nMol = 2; p.boxHalf = {60, 60, 220}; }
    if (p.scenario == 3) { p.nMol = 7; p.boxHalf = {60, 60, 260}; }
    std::vector<glm::vec4> fibrilInit;
    if (p.scenario == 4) {
        // Hodge-Petruska quasi-hexagonal fibril: 5D axial period per file,
        // neighbor files staggered by ((i + 2j) mod 5) * D
        float lenNm = p.fibrilLenUm * 1000.f;
        int nPer = 0;
        if (p.pbc) {
            // snap to a whole number of 5D lattice periods -> infinite fibril
            float per = 5.f * dPeriod;
            nPer = std::max(1, (int)std::lround(lenNm / per));
            lenNm = nPer * per;
        }
        float hex = 1.55f;
        for (int j = -40; j <= 40; ++j) {
            for (int i = -40; i <= 40; ++i) {
                float x = (i + 0.5f * (j & 1)) * hex;
                float y = j * hex * 0.866f;
                if (x * x + y * y > p.fibrilRad * p.fibrilRad) continue;
                int phase = ((i % 5) + 2 * (j % 5) + 25) % 5;
                float z0 = -0.5f * lenNm + phase * dPeriod;
                if (p.pbc) {
                    // exactly one molecule per 5D period per file, no images doubled
                    for (int n = 0; n < nPer; ++n) {
                        if ((int)fibrilInit.size() >= 4000) break;
                        fibrilInit.push_back(glm::vec4(x, y, z0 + n * 5.f * dPeriod, 0));
                    }
                } else {
                    for (float z = z0 - 5.f * dPeriod; z < 0.5f * lenNm; z += 5.f * dPeriod) {
                        if (z + molLen < -0.5f * lenNm) continue;
                        if ((int)fibrilInit.size() >= 4000) break;
                        fibrilInit.push_back(glm::vec4(x, y, z, 0));
                    }
                }
            }
        }
        p.nMol = (int)fibrilInit.size();
        p.boxHalf = {p.fibrilRad + 45.f, p.fibrilRad + 45.f,
                     p.pbc ? 0.5f * lenNm : 0.5f * lenNm + molLen};
        printf("fibril scenario: %d molecules, r=%.1f nm, L=%.2f um\n",
               p.nMol, p.fibrilRad, p.fibrilLenUm);
    }

    // grid sizing: the neighbor-collect stencil must capture midpoints out to
    // cutoff + segment length + Verlet skin; cells tile the box exactly so
    // the periodic stencil can wrap with modulo (>= 4 cells/axis under pbc)
    float maxCut = p.cutoff + (p.epsDep > 0.f ? p.depR : 0.f);
    float cellMin = std::max(6.f, segLen + maxCut + NBR_SKIN + 0.2f);
    if (p.pbc) {
        float minBox = 4.f * cellMin;
        p.boxHalf = glm::max(p.boxHalf, glm::vec3(0.5f * minBox));
    }
    glm::vec3 boxSize = 2.f * p.boxHalf;
    for (;;) {
        gridDim = glm::max(glm::ivec3(glm::floor(boxSize / cellMin)), glm::ivec3(1));
        if (p.pbc) gridDim = glm::max(gridDim, glm::ivec3(4));
        nCells = gridDim.x * gridDim.y * gridDim.z;
        if (nCells <= (1 << 20)) break;
        cellMin *= 1.15f;
    }
    cellSize = boxSize / glm::vec3(gridDim);
    scanBlocks = (nCells + 1023) / 1024;

    std::mt19937 rng(p.seed);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::normal_distribution<float> gauss(0.f, 1.f);

    if (p.scenario == 0) {
        // packing sanity: cap the rod volume fraction so no input path
        // (concentration slider, CLI, stale boxes) can request a jammed start
        double V = 8.0 * p.boxHalf.x * p.boxHalf.y * p.boxHalf.z;
        double vRod = 3.14159 * 0.75 * 0.75 * molLen;
        int nMax = std::max(2, (int)(0.15 * V / vRod));
        if (p.nMol > nMax) {
            printf("packing guard: %d molecules exceeds 15%% volume fraction in this box, clamping to %d\n",
                   p.nMol, nMax);
            p.nMol = nMax;
        }
    }

    std::vector<vec4> pos(nBead());
    if (p.scenario == 4) {
        for (int m = 0; m < p.nMol; ++m) {
            glm::vec4 f = fibrilInit[m];
            for (int i = 0; i < p.nBeads; ++i)
                pos[m * p.nBeads + i] = vec4(f.x, f.y, f.z + i * segLen, 0.f);
        }
    } else if (p.scenario == 2 || p.scenario == 3) {
        // straight rods along z at prescribed lateral offsets and staggers
        const float hexR = 2.2f;
        const float hx[7] = {0, hexR, 0.5f * hexR, -0.5f * hexR, -hexR, -0.5f * hexR, 0.5f * hexR};
        const float hy[7] = {0, 0, 0.866f * hexR, 0.866f * hexR, 0, -0.866f * hexR, -0.866f * hexR};
        for (int m = 0; m < p.nMol; ++m) {
            float ox, oy, stag;
            if (p.scenario == 2) {
                ox = (m == 0) ? -1.5f : 1.5f;
                oy = 0.f;
                stag = (m == 0) ? -0.5f * p.pairStag : 0.5f * p.pairStag;
            } else {
                ox = hx[m]; oy = hy[m];
                stag = (m == 0) ? 0.f : 80.f * uni(rng);
            }
            for (int i = 0; i < p.nBeads; ++i) {
                float t = (i - (p.nBeads - 1) * 0.5f) * segLen;
                pos[m * p.nBeads + i] = vec4(ox, oy, t + stag, 0.f);
            }
        }
    } else
    for (int m = 0; m < p.nMol; ++m) {
        // random direction, center placed so the rod stays inside the box
        vec3 d;
        if (p.initNematic) {
            d = glm::normalize(vec3(gauss(rng) * 0.12f, gauss(rng) * 0.12f, 1.f));
        } else {
            do { d = vec3(uni(rng), uni(rng), uni(rng)); } while (glm::dot(d, d) > 1.f || glm::dot(d, d) < 1e-4f);
            d = glm::normalize(d);
        }
        vec3 span = glm::abs(d) * (molLen * 0.5f);
        vec3 c;
        for (int k = 0; k < 3; ++k) {
            if (p.pbc) {
                // periodic: fill the whole box, molecules may cross the seam
                c[k] = uni(rng) * p.boxHalf[k];
            } else {
                float room = std::max(5.f, p.boxHalf[k] - span[k] - 5.f);
                c[k] = uni(rng) * room;
            }
        }
        // gentle random bow so molecules don't look laser-straight
        vec3 perp = glm::normalize(glm::cross(d, vec3(gauss(rng), gauss(rng), gauss(rng))));
        float bow = 6.f * uni(rng);
        for (int i = 0; i < p.nBeads; ++i) {
            float t = (i - (p.nBeads - 1) * 0.5f) * segLen;
            float u = float(i) / (p.nBeads - 1) * 2.f - 1.f;
            vec3 x = c + d * t + perp * (bow * (1.f - u * u));
            pos[m * p.nBeads + i] = vec4(x, 0.f);
        }
    }

    auto del = [](GLuint& b) { if (b) { glDeleteBuffers(1, &b); b = 0; } };
    del(posBuf); del(posBuf2); del(frameBuf); del(segFBuf);
    del(gridCount); del(gridOffset); del(gridCursor); del(gridIds); del(gridBlockSums);
    del(statsBuf); del(thetaBuf); del(segTqBuf);
    del(renderPosBuf); del(centersBuf); del(nbrListBuf); del(nbrCntBuf); del(ageBuf);
    del(attMagBuf);

    posBuf = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    posBuf2 = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    frameBuf = makeSSBO(sizeof(vec4) * nBead());
    segFBuf = makeSSBO(sizeof(vec4) * 2 * nSeg());
    gridCount = makeSSBO(4u * nCells);
    gridOffset = makeSSBO(4u * (nCells + 1));
    gridCursor = makeSSBO(4u * nCells);
    gridIds = makeSSBO(4u * nSeg());
    gridBlockSums = makeSSBO(4u * scanBlocks);
    statsBuf = makeSSBO(32u);
    {
        std::vector<float> th(p.nMol);
        for (auto& t : th) t = uni(rng) * 3.14159265f;
        thetaBuf = makeSSBO(4u * p.nMol, th.data());
    }
    segTqBuf = makeSSBO(4u * nSeg());
    renderPosBuf = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    centersBuf = makeSSBO(sizeof(vec4) * p.nMol);
    nbrListBuf = makeSSBO(4u * size_t(nSeg()) * NBR_CAP);
    nbrCntBuf = makeSSBO(4u * nSeg());
    {
        std::vector<float> ages(p.nMol, -1e9f);   // existing molecules: no glow
        ageBuf = makeSSBO(4u * p.nMol, ages.data());
    }
    attMagBuf = makeSSBO(4u * nSeg());
    glClearNamedBufferData(attMagBuf, GL_R32F, GL_RED, GL_FLOAT, nullptr);

    if (xlinkBuf) { glDeleteBuffers(1, &xlinkBuf); xlinkBuf = 0; }
    {
        std::vector<glm::vec4> xl(nBead(), glm::vec4(glm::intBitsToFloat(-1), 0,
                                                     glm::intBitsToFloat(-1), 0));
        nXlinks = 0;
        if (p.scenario == 4 && p.xlink) {
            // covalent end tethers: each molecule's terminal beads bond to the
            // nearest bead of a neighboring molecule (lysyl-oxidase analog)
            auto slotOf = [&](int b) -> int {
                if (glm::floatBitsToInt(xl[b].x) < 0) return 0;
                if (glm::floatBitsToInt(xl[b].z) < 0) return 1;
                return -1;
            };
            for (int m = 0; m < p.nMol; ++m) {
                for (int e : {0, p.nBeads - 1}) {
                    int be = m * p.nBeads + e;
                    if (slotOf(be) < 0) continue;
                    int bestB = -1;
                    float bestD = p.xlinkCut;
                    for (int b = 0; b < nBead(); ++b) {
                        if (b / p.nBeads == m) continue;
                        vec3 dv = vec3(pos[b]) - vec3(pos[be]);
                        if (p.pbc) {
                            vec3 bs = 2.f * p.boxHalf;
                            dv -= bs * glm::round(dv / bs);
                        }
                        float d = glm::length(dv);
                        if (d < bestD && slotOf(b) >= 0) { bestD = d; bestB = b; }
                    }
                    if (bestB >= 0) {
                        int sa = slotOf(be), sb = slotOf(bestB);
                        float* fa = &xl[be].x;
                        fa[sa * 2] = glm::intBitsToFloat(bestB);
                        fa[sa * 2 + 1] = bestD;
                        float* fb = &xl[bestB].x;
                        fb[sb * 2] = glm::intBitsToFloat(be);
                        fb[sb * 2 + 1] = bestD;
                        nXlinks++;
                    }
                }
            }
            printf("crosslinks: %d end tethers\n", nXlinks);
        }
        xlinkBuf = makeSSBO(sizeof(glm::vec4) * nBead(), xl.data());
    }

    simTimeNs = 0;
    totalSteps = 0;
    burnin = 1500;    // soft-start: relax initial overlaps before attractions
    computeFrames();
}

static void barrier() { glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); }

void Sim::buildGrid() {
    glClearNamedBufferData(gridCount, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
    barrier();
    prCount.use();
    prCount.set("nSeg", nSeg());
    prCount.set("nBeads", p.nBeads);
    prCount.set("boxHalf", p.boxHalf);
    prCount.set("cellSize", cellSize);
    prCount.set("gridDim", glm::vec3(gridDim));
    prCount.set("pbc", p.pbc);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridCount);
    glDispatchCompute((nSeg() + 255) / 256, 1, 1);
    barrier();
    prScan1.use();
    prScan1.set("n", nCells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, gridCount);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridOffset);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridBlockSums);
    glDispatchCompute(scanBlocks, 1, 1);
    barrier();
    prScan2.use();
    prScan2.set("n", scanBlocks);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridBlockSums);
    glDispatchCompute(1, 1, 1);
    barrier();
    prScan3.use();
    prScan3.set("n", nCells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridOffset);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridBlockSums);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, gridCursor);
    glDispatchCompute(scanBlocks, 1, 1);
    barrier();
    prFill.use();
    prFill.set("nSeg", nSeg());
    prFill.set("nBeads", p.nBeads);
    prFill.set("boxHalf", p.boxHalf);
    prFill.set("cellSize", cellSize);
    prFill.set("gridDim", glm::vec3(gridDim));
    prFill.set("pbc", p.pbc);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, gridCursor);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, gridIds);
    glDispatchCompute((nSeg() + 255) / 256, 1, 1);
    barrier();

    prNbr.use();
    prNbr.set("nSeg", nSeg());
    prNbr.set("nBeads", p.nBeads);
    prNbr.set("segLen", segLen);
    prNbr.set("boxHalf", p.boxHalf);
    prNbr.set("cellSize", cellSize);
    prNbr.set("gridDim", glm::vec3(gridDim));
    prNbr.set("pbc", p.pbc);
    float maxCut = p.cutoff + (p.epsDep > 0.f ? p.depR : 0.f);
    prNbr.set("reach", maxCut + segLen + NBR_SKIN);
    prNbr.set("nbrCap", NBR_CAP);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridOffset);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridIds);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, nbrListBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, nbrCntBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, statsBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, gridCount);
    glDispatchCompute((nSeg() + 127) / 128, 1, 1);
    barrier();
}

void Sim::step(int nSteps) {
    float tK = 273.15f + p.tempC;
    float kTeff = p.kT * tK / 310.15f;
    // hydrophobic effect is entropic: strengthens warm, weakens cold
    float epsHyEff = p.epsHy * std::max(0.05f, 1.f + 0.013f * (tK - 310.15f));
    for (int s = 0; s < nSteps; ++s) {
        if (totalSteps % 4 == 0) buildGrid();
        if (p.registryMode == 2) computeFrames();   // forces need material frames

        // soft-start ramp: gentle repulsion first, attractions fade in
        float ramp = burnin > 0 ? 1.f - float(burnin) / 1500.f : 1.f;
        if (burnin > 0) burnin--;
        float repRamp = 0.25f + 0.75f * ramp;

        prForces.use();
        prForces.set("nSeg", nSeg());
        prForces.set("nBeads", p.nBeads);
        prForces.set("segLen", segLen);
        prForces.set("boxHalf", p.boxHalf);
        prForces.set("rRep", p.rRep);
        prForces.set("kRep", p.kRep * repRamp);
        prForces.set("attR0", p.attR0);
        prForces.set("attW", p.attW);
        prForces.set("cutoff", p.cutoff + (p.epsDep > 0.f ? p.depR : 0.f));
        prForces.set("epsEl", p.epsEl * ramp);
        prForces.set("epsHy", epsHyEff * ramp);
        prForces.set("epsNs", p.epsNs * ramp);
        prForces.set("epsDep", p.epsDep * ramp);
        prForces.set("depR", p.depR);
        prForces.set("apMix", p.apMix);
        prForces.set("specific", p.specific);
        prForces.set("molLen", molLen);
        prForces.set("profDs", profDs);
        prForces.set("nPar", nPar);
        prForces.set("nAp", nAp);
        prForces.set("fMax", p.fMax);
        prForces.set("dPeriod", dPeriod);
        prForces.set("registryMode", p.registryMode);
        prForces.set("p2Dstep", p2Dstep);
        prForces.set("p2S0", p2S0);
        prForces.set("p2nD", p2nD);
        prForces.set("p2nR", p2nR);
        prForces.set("p2nPhi", p2nPhi);
        prForces.set("nbrCap", NBR_CAP);
        prForces.set("pbc", p.pbc);
        int statsOn = (totalSteps % 16 == 0) ? 1 : 0;
        prForces.set("statsOn", statsOn);
        if (statsOn)
            glClearNamedBufferData(statsBuf, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, nbrListBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, nbrCntBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, segFBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, attMagBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, frameBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, statsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, segTqBuf);
        glBindTextureUnit(0, texPar);
        glBindTextureUnit(1, texAp);
        glBindTextureUnit(2, texPar2);
        glBindTextureUnit(3, texAp2);
        glDispatchCompute((nSeg() + 127) / 128, 1, 1);
        barrier();

        prInteg.use();
        prInteg.set("nBead", nBead());
        prInteg.set("nBeads", p.nBeads);
        prInteg.set("dt", p.dt);
        prInteg.set("gamma", p.gamma / p.speed);
        prInteg.set("kT", kTeff);
        prInteg.set("kBend", p.persistLen * p.kT / segLen);
        prInteg.set("boxHalf", p.boxHalf);
        prInteg.set("kWall", p.kWall);
        prInteg.set("pbc", p.pbc);
        prInteg.set("fMax", p.fMax);
        prInteg.set("seed", (unsigned)(p.seed + (totalSteps & 0x7fffffff)));
        prInteg.set("xlinkK", nXlinks > 0 ? p.xlinkK : 0.f);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, xlinkBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, segFBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, posBuf2);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, statsBuf);
        glDispatchCompute((nBead() + 255) / 256, 1, 1);
        barrier();

        if (p.registryMode == 2) {
            prRot.use();
            prRot.set("nMol", p.nMol);
            prRot.set("nBeads", p.nBeads);
            prRot.set("dt", p.dt);
            prRot.set("gammaRot", p.gammaRot / p.speed);
            prRot.set("kT", kTeff);
            prRot.set("seed", (unsigned)(p.seed * 7u + (totalSteps & 0x7fffffff)));
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, segTqBuf);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, thetaBuf);
            glDispatchCompute((p.nMol + 63) / 64, 1, 1);
            barrier();
        }

        prXpbd.use();
        prXpbd.set("nSeg", nSeg());
        prXpbd.set("nBeads", p.nBeads);
        prXpbd.set("segLen", segLen);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, posBuf2);
        for (int it = 0; it < 4; ++it) {
            prXpbd.set("parity", it & 1);
            glDispatchCompute((nSeg() + 255) / 256, 1, 1);
            barrier();
        }

        std::swap(posBuf, posBuf2);
        totalSteps++;
        simTimeNs += p.dt * p.speed;   // physical time at physical drag

        if (p.pbc && totalSteps % 16 == 0) {
            prRecenter.use();
            prRecenter.set("nMol", p.nMol);
            prRecenter.set("nBeads", p.nBeads);
            prRecenter.set("boxHalf", p.boxHalf);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, renderPosBuf);
            glDispatchCompute((p.nMol + 63) / 64, 1, 1);
            barrier();
        }
    }
}

void Sim::insertMolecules(int nNew, float nowSec) {
    if (nNew <= 0 || p.scenario != 0) return;
    int oldN = p.nMol;
    int oldBeads = oldN * p.nBeads;

    // read current state
    std::vector<vec4> pos(oldBeads);
    std::vector<float> th(oldN), ages(oldN);
    glGetNamedBufferSubData(posBuf, 0, sizeof(vec4) * oldBeads, pos.data());
    glGetNamedBufferSubData(thetaBuf, 0, 4 * oldN, th.data());
    glGetNamedBufferSubData(ageBuf, 0, 4 * oldN, ages.data());

    // occupancy hash of existing beads for overlap-free placement
    const float hcell = 4.f;
    auto key = [&](const vec3& v) {
        return (int64_t(floorf(v.x / hcell)) & 0xfffff) |
               ((int64_t(floorf(v.y / hcell)) & 0xfffff) << 20) |
               ((int64_t(floorf(v.z / hcell)) & 0xfffff) << 40);
    };
    std::unordered_map<int64_t, std::vector<vec3>> occ;
    for (int b = 0; b < oldBeads; ++b) occ[key(vec3(pos[b]))].push_back(vec3(pos[b]));
    auto clear2 = [&](const vec3& q, float rMin) {
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            auto it = occ.find(key(q + vec3(dx, dy, dz) * hcell));
            if (it == occ.end()) continue;
            for (const vec3& w : it->second) {
                vec3 dqw = q - w;
                if (glm::dot(dqw, dqw) < rMin * rMin) return false;
            }
        }
        return true;
    };

    std::mt19937 rng(p.seed * 977u + (uint32_t)oldN);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::normal_distribution<float> gauss(0.f, 1.f);
    pos.resize(size_t(oldN + nNew) * p.nBeads);
    for (int m = oldN; m < oldN + nNew; ++m) {
        bool placed = false;
        for (int attempt = 0; attempt < 60 && !placed; ++attempt) {
            vec3 d;
            if (p.initNematic) d = glm::normalize(vec3(gauss(rng) * 0.12f, gauss(rng) * 0.12f, 1.f));
            else {
                do { d = vec3(uni(rng), uni(rng), uni(rng)); } while (glm::dot(d, d) > 1.f || glm::dot(d, d) < 1e-4f);
                d = glm::normalize(d);
            }
            vec3 span = glm::abs(d) * (molLen * 0.5f);
            vec3 c;
            for (int k = 0; k < 3; ++k)
                c[k] = p.pbc ? uni(rng) * p.boxHalf[k]
                             : uni(rng) * std::max(5.f, p.boxHalf[k] - span[k] - 5.f);
            bool ok = true;
            for (int i = 0; i < p.nBeads && ok; i += 2) {
                float t = (i - (p.nBeads - 1) * 0.5f) * segLen;
                if (!clear2(c + d * t, 1.6f)) ok = false;
            }
            if (ok || attempt == 59) {
                for (int i = 0; i < p.nBeads; ++i) {
                    float t = (i - (p.nBeads - 1) * 0.5f) * segLen;
                    vec3 x = c + d * t;
                    pos[size_t(m) * p.nBeads + i] = vec4(x, 0.f);
                    occ[key(x)].push_back(x);
                }
                placed = true;
            }
        }
        th.push_back(uni(rng) * 3.14159265f);
        ages.push_back(nowSec);
    }
    p.nMol = oldN + nNew;

    // recreate N-dependent buffers (grid cell buffers are box-sized: unchanged)
    auto del = [](GLuint& b) { if (b) { glDeleteBuffers(1, &b); b = 0; } };
    del(posBuf); del(posBuf2); del(frameBuf); del(segFBuf); del(gridIds);
    del(thetaBuf); del(segTqBuf); del(renderPosBuf); del(centersBuf);
    del(nbrListBuf); del(nbrCntBuf); del(ageBuf);
    posBuf = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    posBuf2 = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    frameBuf = makeSSBO(sizeof(vec4) * nBead());
    segFBuf = makeSSBO(sizeof(vec4) * 2 * nSeg());
    gridIds = makeSSBO(4u * nSeg());
    thetaBuf = makeSSBO(4u * p.nMol, th.data());
    segTqBuf = makeSSBO(4u * nSeg());
    renderPosBuf = makeSSBO(sizeof(vec4) * nBead(), pos.data());
    centersBuf = makeSSBO(sizeof(vec4) * p.nMol);
    nbrListBuf = makeSSBO(4u * size_t(nSeg()) * NBR_CAP);
    nbrCntBuf = makeSSBO(4u * nSeg());
    ageBuf = makeSSBO(4u * p.nMol, ages.data());
    attMagBuf = makeSSBO(4u * nSeg());
    glClearNamedBufferData(attMagBuf, GL_R32F, GL_RED, GL_FLOAT, nullptr);
    if (xlinkBuf) { glDeleteBuffers(1, &xlinkBuf); xlinkBuf = 0; }
    {
        std::vector<glm::vec4> xl(nBead(), glm::vec4(glm::intBitsToFloat(-1), 0,
                                                     glm::intBitsToFloat(-1), 0));
        xlinkBuf = makeSSBO(sizeof(glm::vec4) * nBead(), xl.data());
        nXlinks = 0;
    }
    burnin = 400;
    computeFrames();
    printf("inserted %d molecules (N = %d)\n", nNew, p.nMol);
}

void Sim::readStats() {
    glGetNamedBufferSubData(statsBuf, 0, 32, stats);
}

void Sim::mcPass(int nPasses) {
    float tK = 273.15f + p.tempC;
    float kTeff = p.kT * tK / 310.15f;
    float epsHyEff = p.epsHy * std::max(0.05f, 1.f + 0.013f * (tK - 310.15f));
    for (int pass = 0; pass < nPasses; ++pass) {
        buildGrid();
        if (p.registryMode == 2) computeFrames();   // funnel mode needs no frames
        prMc.use();
        prMc.set("nMol", p.nMol);
        prMc.set("nBeads", p.nBeads);
        prMc.set("segLen", segLen);
        prMc.set("boxHalf", p.boxHalf);
        prMc.set("cellSize", cellSize);
        prMc.set("gridDim", glm::vec3(gridDim));
        prMc.set("pbc", p.pbc);
        prMc.set("rRep", p.rRep);
        prMc.set("kRep", p.kRep);
        prMc.set("attR0", p.attR0);
        prMc.set("attW", p.attW);
        prMc.set("cutoff", p.cutoff + (p.epsDep > 0.f ? p.depR : 0.f));
        prMc.set("epsEl", p.epsEl);
        prMc.set("epsHy", epsHyEff);
        prMc.set("epsNs", p.epsNs);
        prMc.set("apMix", p.apMix);
        prMc.set("kWall", p.kWall);
        prMc.set("kT", kTeff);
        prMc.set("registryMode", p.registryMode);
        prMc.set("profDs", profDs);
        prMc.set("nPar", nPar);
        prMc.set("nAp", nAp);
        prMc.set("p2Dstep", p2Dstep);
        prMc.set("p2S0", p2S0);
        prMc.set("p2nD", p2nD);
        prMc.set("p2nR", p2nR);
        prMc.set("p2nPhi", p2nPhi);
        prMc.set("seed", p.seed * 31u + 7u);
        prMc.set("mcStep", mcStepCounter++);
        prMc.set("stride", p.mcStride);
        prMc.set("nbrCap", NBR_CAP);
        prMc.set("ampT", p.mcAmpT);
        prMc.set("ampR", p.mcAmpR);
        prMc.set("ampS", p.mcAmpS);
        prMc.set("ampTw", p.mcAmpTw);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, gridOffset);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, gridIds);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, frameBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, statsBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, nbrListBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, nbrCntBuf);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, gridCount);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, thetaBuf);
        glBindTextureUnit(0, texPar);
        glBindTextureUnit(1, texAp);
        glBindTextureUnit(2, texPar2);
        glBindTextureUnit(3, texAp2);
        glDispatchCompute(p.nMol, 1, 1);   // one workgroup per molecule
        barrier();
    }
}

void Sim::computeFrames(GLuint posBuffer) {
    prFrames.use();
    prFrames.set("nMol", p.nMol);
    prFrames.set("nBeads", p.nBeads);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuffer ? posBuffer : posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, frameBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, thetaBuf);
    glDispatchCompute((p.nMol + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
}

void Sim::smoothForDisplay(float alpha, bool reset) {
    prSmooth.use();
    prSmooth.set("nBead", nBead());
    prSmooth.set("alpha", alpha);
    prSmooth.set("reset", reset ? 1 : 0);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, renderPosBuf);
    glDispatchCompute((nBead() + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Sim::updateCenters(GLuint posBuffer) {
    prCenters.use();
    prCenters.set("nMol", p.nMol);
    prCenters.set("nBeads", p.nBeads);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, posBuffer ? posBuffer : posBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, centersBuf);
    glDispatchCompute((p.nMol + 63) / 64, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
}
