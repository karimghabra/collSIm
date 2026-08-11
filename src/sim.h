#pragma once
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <cstdint>
#include "io.h"
#include "shader.h"

// GPU Brownian dynamics of semiflexible tropocollagen filaments.
// Units: length nm, energy kT, time ns. Overdamped Langevin; XPBD bond
// constraints; segment-segment interactions with sequence-derived registry
// energies sampled from 1D textures.
struct SimParams {
    int nMol = 400;
    int nBeads = 96;              // beads per molecule
    glm::vec3 boxHalf = {250.f, 250.f, 330.f};
    float dt = 0.03f;             // ns
    float gamma = 7.0f;           // physical drag, kT ns / nm^2 per bead
    float speed = 10.f;           // kinetic time-compression (divides gamma)
    float kT = 1.0f;              // temperature (1 = 300 K)
    float persistLen = 60.f;      // nm
    float epsEl = 2.5f;           // electrostatic registry well depth (kT/pair)
    float epsHy = 1.0f;           // hydrophobic registry well depth (kT/pair)
    float epsNs = 0.15f;          // nonspecific contact attraction (kT)
    float epsCorr = 1.0f;         // measured Delta-learning correction weight
    float wExp = 1.0f;            // alignment gate exponent (tilt verdict:
                                  // geometry attenuates; 1 = polarity gating)
    float tempC = 37.f;           // temperature; kT and hydrophobic strength follow
    float pH = 7.4f;              // protonation states via basis recombination
    float epsDep = 0.f;           // depletion/crowding attraction (kT)
    float depR = 3.0f;            // depletion range (nm beyond contact)
    float apMix = 1.0f;           // antiparallel channel multiplier
    int specific = 1;             // legacy flag for mode 1 (funnel)
    int registryMode = 2;         // 0 off, 1 funnel-1D, 2 azimuthal-2D
    int atelo = 0;                // 1: telopeptide-free electrostatics (2D mode)
    float gammaRot = 500.f;       // rotational drag, kT ns (whole molecule)
    int initNematic = 0;          // 1: quench preset (aligned start)
    int scenario = 0;             // 0 normal, 2 pair, 3 hex-7, 4 preassembled fibril
    float pairStag = 40.f;        // scenario 2: initial stagger (nm)
    float fibrilRad = 5.5f;       // scenario 4: fibril radius (nm)
    float fibrilLenUm = 1.5f;     // scenario 4: fibril length (um)
    int xlink = 0;                // scenario 4: covalent end crosslinks
    float xlinkK = 20.f;          // tether stiffness, kT/nm^2
    float xlinkCut = 2.6f;        // pairing cutoff at initialization (nm)
    float rRep = 1.6f;            // core repulsion onset (nm, center-center)
    float kRep = 15.f;            // kT/nm^2 (barrier ~19 kT, no crossings)
    float attR0 = 1.65f;          // preferred contact distance (PMF gap scan)
    float attW = 0.75f;           // envelope width (fit 0.60; softened 25%
                                  // pending replicate confirmation)
    float cutoff = 3.5f;          // interaction cutoff (auto-raised by depletion)
    float kWall = 4.f;
    float fMax = 15.f;            // force clamp kT/nm
    int pbc = 0;                  // periodic boundaries (minimum image)
    // Metropolis rigid-body boost (interleaved with BD): whole-molecule
    // trial moves accepted against the registry energetics
    int mcBoost = 0;
    int mcStride = 16;            // ~nMol/stride movers per pass
    int mcPasses = 3;             // passes per rendered frame
    float mcAmpT = 2.f;           // nm translation
    float mcAmpR = 0.15f;         // rad rotation
    float mcAmpS = 80.f;          // nm axial slide (covers > 1 D)
    float mcAmpTw = 0.6f;         // rad twist
    uint32_t seed = 1234;
};

class Sim {
public:
    void init(const AtomTemplate& tmpl, const Profiles& prof, const Profiles2D& prof2,
              const Basis2D& basis, const Profiles2D* prof2Atelo = nullptr,
              const Basis2D* basisAtelo = nullptr, const Corr2D* corr = nullptr);
    bool hasAtelo = false;
    bool hasCorr = false;
    void recombinePH();           // rebuild electrostatic tables at p.pH
    float netCharge() const;      // per molecule, elementary charges, at p.pH
    float mwKda = 285.f;
    void restart();               // re-place molecules using current params
    void step(int nSteps);
    void computeFrames(GLuint posBuffer = 0);   // frames from given positions
    void smoothForDisplay(float alpha, bool reset);  // EMA into renderPosBuf
    void updateCenters(GLuint posBuffer);       // molecule centers for LOD
    // live insertion (scenario 0): place nNew molecules avoiding overlap,
    // highlight via ageBuf until the glow fades
    void insertMolecules(int nNew, float nowSec);
    void mcPass(int nPasses);     // Metropolis whole-molecule move passes

    SimParams p;
    float segLen = 0;             // a, nm
    float molLen = 0;             // L, nm
    float dPeriod = 0;
    double simTimeNs = 0;
    int64_t totalSteps = 0;
    int burnin = 0;               // soft-start steps remaining after restart

    // GPU objects (bound by renderer too)
    GLuint posBuf = 0, posBuf2 = 0, frameBuf = 0, segFBuf = 0;
    GLuint gridCount = 0, gridOffset = 0, gridCursor = 0, gridIds = 0, gridBlockSums = 0;
    GLuint texPar = 0, texAp = 0, statsBuf = 0;
    GLuint texPar2 = 0, texAp2 = 0;           // 3D azimuthal tables
    GLuint thetaBuf = 0, segTqBuf = 0;        // molecular twist DOF
    GLuint basisParBuf = 0, basisApBuf = 0;   // pH basis (el), hy statics
    GLuint hyParBuf = 0, hyApBuf = 0;
    GLuint basisParBuf2 = 0, basisApBuf2 = 0; // atelo variants
    GLuint hyParBuf2 = 0, hyApBuf2 = 0;
    GLuint corrParBuf = 0, corrZeroBuf = 0;   // measured correction (par only)
    float tCounts2[8] = {0};
    GLuint renderPosBuf = 0, centersBuf = 0;  // display smoothing + LOD
    GLuint nbrListBuf = 0, nbrCntBuf = 0;     // Verlet neighbor lists
    GLuint ageBuf = 0;                        // spawn wall-time per molecule
    GLuint attMagBuf = 0;                     // per-segment attraction sums
    GLuint xlinkBuf = 0;                      // 2 tether slots per bead
    int nXlinks = 0;
    // contact statistics (from last readStats): [0] hard overlaps (d<1.0nm),
    // [1] inter-molecular contacts (d<2.5), [2] contacts within 3nm of a
    // D-stagger multiple, [3] neighbor-list overflow, [4] displacement-clamp
    // events (accumulated over the 16-step stats window). Pairs double-counted.
    uint32_t stats[8] = {0};
    void readStats();
    int nSeg() const { return p.nMol * (p.nBeads - 1); }
    int nBead() const { return p.nMol * p.nBeads; }

private:
    void buildGrid();
    glm::ivec3 gridDim{};
    glm::vec3 cellSize{6.f};
    int nCells = 0, scanBlocks = 0;
    float profDs = 0.1f;
    int nPar = 0, nAp = 0;
    float p2Dstep = 0.5f, p2S0 = 8.f;
    int p2nD = 0, p2nPhi = 0, p2nR = 0;
    Program prCount, prScan1, prScan2, prScan3, prFill, prForces, prInteg, prXpbd,
        prFrames, prRot, prRecomb, prSmooth, prCenters, prNbr, prRecenter, prMc;
    uint32_t mcStepCounter = 0;
    static constexpr int NBR_CAP = 128;
    static constexpr float NBR_SKIN = 3.0f;
    int nTypes = 5, nPairsB = 15;
    float tCounts[8] = {0};
    int stepParity = 0;
};
