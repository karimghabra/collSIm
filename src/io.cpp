#include "io.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>

static FILE* openBin(const std::string& path, char magic[4]) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    if (fread(magic, 1, 4, f) != 4) throw std::runtime_error("short read " + path);
    return f;
}

AtomTemplate loadAtoms(const std::string& path) {
    char magic[4];
    FILE* f = openBin(path, magic);
    if (memcmp(magic, "TROP", 4)) throw std::runtime_error("bad magic in " + path);
    uint32_t verN[2];
    float fdims[2];
    uint32_t rest[3];
    fread(verN, 4, 2, f);
    fread(fdims, 4, 2, f);
    fread(rest, 4, 3, f);
    AtomTemplate t;
    t.natoms = verN[1];
    t.lengthNm = fdims[0];
    t.dPeriodNm = fdims[1];
    t.nresPerChain = rest[0];
    t.atoms.resize(t.natoms);
    if (fread(t.atoms.data(), sizeof(AtomRec), t.natoms, f) != t.natoms)
        throw std::runtime_error("truncated " + path);
    fclose(f);
    return t;
}

Profiles2D loadProfiles2D(const std::string& path) {
    char magic[4];
    FILE* f = openBin(path, magic);
    if (memcmp(magic, "PRF2", 4)) throw std::runtime_error("bad magic in " + path);
    uint32_t hdr[4];
    float fh[4];
    fread(hdr, 4, 4, f);
    fread(fh, 4, 4, f);
    Profiles2D p;
    p.nD = hdr[1]; p.nPhi = hdr[2]; p.nR = hdr[3];
    p.dstep = fh[0]; p.L = fh[1]; p.D = fh[2]; p.S0 = fh[3];
    size_t np = size_t(p.nD) * p.nPhi * p.nPhi;
    size_t na = size_t(p.nR) * p.nPhi * p.nPhi;
    std::vector<float> elp(np), hyp(np), ela(na), hya(na);
    fread(elp.data(), 4, np, f);
    fread(hyp.data(), 4, np, f);
    fread(ela.data(), 4, na, f);
    fread(hya.data(), 4, na, f);
    fclose(f);
    p.par.resize(np * 2);
    p.ap.resize(na * 2);
    for (size_t i = 0; i < np; ++i) { p.par[2*i] = elp[i]; p.par[2*i+1] = hyp[i]; }
    for (size_t i = 0; i < na; ++i) { p.ap[2*i] = ela[i]; p.ap[2*i+1] = hya[i]; }
    return p;
}

Basis2D loadBasis2D(const std::string& path) {
    char magic[4];
    FILE* f = openBin(path, magic);
    if (memcmp(magic, "PRFB", 4)) throw std::runtime_error("bad magic in " + path);
    uint32_t hdr[5];
    float fh[6];
    fread(hdr, 4, 5, f);
    fread(fh, 4, 6, f);
    Basis2D b;
    b.nD = hdr[1]; b.nPhi = hdr[2]; b.nR = hdr[3]; b.nTypes = hdr[4];
    b.dstep = fh[0]; b.L = fh[1]; b.D = fh[2]; b.S0 = fh[3];
    b.elScale = fh[4]; b.mwKda = fh[5];
    fread(b.counts, 4, b.nTypes, f);
    size_t np = size_t(b.nD) * b.nPhi * b.nPhi * b.nPairs();
    size_t na = size_t(b.nR) * b.nPhi * b.nPhi * b.nPairs();
    b.par.resize(np);
    b.ap.resize(na);
    fread(b.par.data(), 4, np, f);
    fread(b.ap.data(), 4, na, f);
    fclose(f);
    return b;
}

Corr2D loadCorr2D(const std::string& path) {
    char magic[4];
    FILE* f = openBin(path, magic);
    if (memcmp(magic, "CORR", 4)) throw std::runtime_error("bad magic in " + path);
    uint32_t hdr[4];
    float fh[4];
    fread(hdr, 4, 4, f);
    fread(fh, 4, 4, f);
    Corr2D c;
    c.nD = hdr[1];
    c.nPhi = hdr[2];
    c.dstep = fh[0];
    c.L = fh[1];
    c.D = fh[2];
    c.c.resize(size_t(c.nD) * c.nPhi * c.nPhi);
    fread(c.c.data(), 4, c.c.size(), f);
    fclose(f);
    return c;
}

Profiles loadProfiles(const std::string& path) {
    char magic[4];
    FILE* f = openBin(path, magic);
    if (memcmp(magic, "PROF", 4)) throw std::runtime_error("bad magic in " + path);
    uint32_t hdr[3];
    float fh[3];
    fread(hdr, 4, 3, f);
    fread(fh, 4, 3, f);
    Profiles p;
    p.nPar = hdr[1];
    p.nAp = hdr[2];
    p.ds = fh[0];
    p.L = fh[1];
    p.D = fh[2];
    p.elPar.resize(p.nPar); p.hyPar.resize(p.nPar);
    p.elAp.resize(p.nAp);   p.hyAp.resize(p.nAp);
    fread(p.elPar.data(), 4, p.nPar, f);
    fread(p.hyPar.data(), 4, p.nPar, f);
    fread(p.elAp.data(), 4, p.nAp, f);
    fread(p.hyAp.data(), 4, p.nAp, f);
    fclose(f);
    return p;
}
