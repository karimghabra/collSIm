#pragma once
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "io.h"

// Docking-based Monte Carlo after Vakser et al., PNAS 2022: the pairwise
// interaction landscape (our sequence-derived registry tables = the docking
// energy landscape) is mined for its discrete low-energy docking poses, and
// the system evolves by Metropolis minima-hopping between docked states,
// with the neighbor-count (Ni/Nj) detailed-balance correction and MC steps
// calibrated to physical time through the rod diffusion coefficient.
// v1: rigid straight tropocollagens (paper-faithful). v2 path: per-segment
// local docking frames to restore flexibility.
struct DockPose {
    float dz;          // stagger (nm)
    float pa, pb;      // facing angles (rad)
    float ePair;       // kT per contact pair at this pose (negative)
};

struct McDockParams {
    float sigmaFree = 10.f;    // free-move displacement scale (nm) -> sets step time
    float tempFactor = 1.f;    // multiplies kT
    float collTol = 1.1f;      // capsule-capsule minimum distance (nm)
    float captureR = 6.f;      // docking neighborhood beyond contact (nm)
    float contactR = 1.5f;     // docked center-line spacing (nm)
    int maxPoses = 4096;
    float poseCutoff = -0.15f; // per-pair energy threshold for library entry
    float contactPairs = 25.f; // kT scale: pairs engaged per unit overlap (x0.1)
};

class McDock {
public:
    struct Mol {
        glm::vec3 c;
        glm::quat q;           // axis = q * +z, material normal = q * +x
    };
    struct Match {
        int a, b, pose;
        float E;
    };

    void buildPoseLibrary(const Basis2D& b, const std::vector<float>& parRG,
                          float pH, float epsEl, float epsHy);
    void initFromScatter(int nMol, glm::vec3 boxHalf, float molLen, bool pbcOn,
                         uint32_t seed);
    void sweep(int nSweeps);
    // bead positions for the shared renderer
    void writeBeads(std::vector<glm::vec4>& out, int nBeads, float segLen) const;

    McDockParams p;
    std::vector<Mol> mols;
    std::vector<Match> matches;
    std::vector<DockPose> poses;
    glm::vec3 boxHalf{200.f};
    float molLen = 290.f, dPeriod = 66.9f;
    bool pbc = true;
    double simTimeNs = 0;
    double stepTimeNs = 0;     // physical time per sweep (from sigmaFree & D)
    int64_t sweepsDone = 0, proposed = 0, accepted = 0;
    // observables
    float boundFraction() const;
    float bandCoherence() const;
    int clusterCount(float& meanSize) const;

private:
    std::vector<std::vector<int>> molMatch;   // match indices per molecule
    uint64_t rngState = 88172645463325252ull;
    double rnd();
    float capsuleDist(int i, const glm::vec3& c, const glm::quat& q, int j) const;
    int neighborCount(int i, const glm::vec3& c, const glm::quat& q) const;
    void computeStepTime();
};
