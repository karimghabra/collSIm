#pragma once
#include <cstdint>
#include <string>
#include <vector>

// mirrors the record written by tools/build_tropocollagen.py (20 bytes)
#pragma pack(push, 1)
struct AtomRec {
    float s, x, y;      // axial contour (nm), local cross-section offsets (nm)
    uint8_t elem;       // 0 C, 1 N, 2 O, 3 S
    uint8_t cls;        // 0 nonpolar, 1 polar, 2 pos, 3 neg, 4 gly
    uint16_t res;
    uint8_t chain;
    uint8_t pad;
    uint16_t pad2;
};
#pragma pack(pop)
static_assert(sizeof(AtomRec) == 20);

struct AtomTemplate {
    uint32_t natoms = 0;
    float lengthNm = 0, dPeriodNm = 0;
    uint32_t nresPerChain = 0;
    std::vector<AtomRec> atoms;
};

struct Profiles2D {
    float dstep = 0, L = 0, D = 0, S0 = 0;
    uint32_t nD = 0, nPhi = 0, nR = 0;
    // interleaved RG (el, hy), layout [n][phiA][phiB]
    std::vector<float> par, ap;
};

struct Basis2D {
    uint32_t nD = 0, nPhi = 0, nR = 0, nTypes = 0;
    float dstep = 0, L = 0, D = 0, S0 = 0, elScale = 1, mwKda = 285;
    float counts[8] = {0};
    std::vector<float> par, ap;   // nPairs consecutive tables, [n][phiA][phiB]
    int nPairs() const { return nTypes * (nTypes + 1) / 2; }
};

// measured Delta-learning correction (kT per engine segment, additive on G)
struct Corr2D {
    uint32_t nD = 0, nPhi = 0;
    float dstep = 0, L = 0, D = 0;
    std::vector<float> c;         // [nD][phiA][phiB]
};

AtomTemplate loadAtoms(const std::string& path);
Profiles2D loadProfiles2D(const std::string& path);
Basis2D loadBasis2D(const std::string& path);
Corr2D loadCorr2D(const std::string& path);
