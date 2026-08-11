#pragma once
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "io.h"

// Segmental docking Monte Carlo (docs/segmental_docking_mc.md): the Vakser
// minima-hopping methodology with persistence-length LINKS as the rigid
// docking bodies. Molecules are chains of rigid links (base point + link
// quaternions, exactly connected by construction) with WLC bending at the
// joints. Interaction energy is the CONTACT INTEGRAL of the merged registry
// kernel over each close link pair — tilted contacts price themselves via
// the stagger ramp (tilt-series verdict), and "docked" is emergent from
// proximity rather than bookkept. Moves: whole-molecule rigid, dock-guided
// rigid placement of a chosen link onto a neighbor pose, and joint pivots.
struct DockPose {
    float dz;          // stagger (nm)
    float pa, pb;      // facing angles (rad)
    float ePair;       // kernel value at the pose (kT per engine segment)
};

struct McDockParams {
    int nLinks = 5;            // links per molecule (persistence-length sized)
    float sigmaFree = 10.f;    // free-move displacement scale (nm) -> step time
    float tempFactor = 1.f;
    float collTol = 1.1f;      // capsule-capsule minimum distance (nm)
    float captureR = 6.f;      // docking neighborhood beyond contact (nm)
    float contactR = 1.65f;    // docked center-line spacing (nm, PMF radial)
    float envW = 0.75f;        // radial envelope width for contact integrals
    float cutoff = 4.0f;       // integral evaluated for local gaps below this
    int maxPoses = 4096;
    float poseCutoff = -0.15f;
    float pivotAmp = 0.25f;    // rad, joint pivot moves
    float bendOn = 1.f;        // scales WLC joint stiffness
    // Dock hops are the paper's biased basin-finding move: the placement is
    // deterministic, so its exact reverse is not a library placement. Turn
    // them off for a strictly reversible chain (transport/slide/pivot only).
    bool dockMoves = true;
};

class McDock {
public:
    struct Mol {
        glm::vec3 p0;                  // start of link 0
        std::vector<glm::quat> q;      // per-link orientation (axis = q*z)
    };

    // build the CPU kernel grid + pose library (call on pH/form/eps change)
    void buildPoseLibrary(const Basis2D& b, const std::vector<float>& parRG,
                          float pH, float epsEl, float epsHy,
                          const Corr2D* corrTab = nullptr, float epsCorr = 1.f);
    void initFromScatter(int nMol, glm::vec3 boxHalf, float molLen, bool pbcOn,
                         uint32_t seed);
    void sweep(int nSweeps);
    void writeBeads(std::vector<glm::vec4>& out, int nBeads, float segLen) const;

    McDockParams p;
    std::vector<Mol> mols;
    std::vector<DockPose> poses;
    glm::vec3 boxHalf{200.f};
    float molLen = 290.f, dPeriod = 66.9f, linkLen = 58.f;
    float persistLen = 60.f;
    bool pbc = true;
    double simTimeNs = 0;
    // Clock: sigma^2/6D credits a full free-diffusion hop every sweep, but
    // most proposals are rejected, so it overstates elapsed time by ~1/acc.
    // stepTimeNs is calibrated instead on the MEASURED mean-square COM
    // displacement of molecules that were free at the start of the sweep --
    // free molecules keep diffusing at the physical rate no matter what the
    // sampler accepts, so they are the honest clock. With none left (fully
    // assembled) the last calibration is held and flagged as extrapolated.
    double stepTimeNs = 0;
    double stepTimeNominalNs = 0;   // sigma^2/6D, for comparison
    double lastMsdFree = 0;         // nm^2 per sweep
    int lastFreeCount = 0;
    bool clockExtrapolated = false;
    int64_t sweepsDone = 0, proposed = 0, accepted = 0;
    int64_t accWhole = 0, accDock = 0, accPivot = 0, accSlide = 0;

    float boundFraction() const;
    float bandCoherence() const;
    // fraction of contacts whose local stagger is within tol of a D-multiple
    // (same estimator as the BD registry-model D-fraction)
    float dFraction(float tol = 3.f) const;
    int clusterCount(float& meanSize) const;
    void dumpContacts(int maxN) const;

private:
    // merged kernel grid (Delta, phiA, phiB), kT per engine segment
    std::vector<float> kernel;
    int knD = 0, knPhi = 0;
    float kdstep = 0.5f;
    float segNorm = 3.056f;            // engine segment length (kernel units)

    std::vector<glm::vec3> linkC;      // cached link centers [mol*nLinks+i]
    uint64_t rngState = 88172645463325252ull;

    // uniform cell grid over link centers (cell >= interaction reach)
    std::vector<int> cellHead, cellNext;
    int gDim[3] = {1, 1, 1};
    glm::vec3 gCell{1.f};
    float gReach = 0.f;
    void rebuildGrid();
    int cellOf(const glm::vec3& c) const;
    template <class F> void forEachNear(const glm::vec3& c, F&& fn) const;

    double rnd();
    glm::vec3 chainCenter(const Mol& st, int i) const;
    void cacheMol(int m);
    float kernelAt(float dz, float pa, float pb) const;
    // contact-integral energy of one (possibly proposed) link of mol mA
    // against link iB of mol mB (cached state); contA = contour of link start
    float linkPairE(const glm::vec3& cA, const glm::quat& qA, float contA,
                    int mB, int iB) const;
    float molEnergy(int m, const Mol& state, bool& clash) const;
    float bendEnergy(const Mol& state) const;
    void computeStepTime();

    template <class F> void forEachContact(float maxD, F&& fn) const;
};
