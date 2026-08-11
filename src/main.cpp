// collagenSim â€” real-time GPU Brownian dynamics of type I collagen fibril
// self-assembly with full-atom rendering of every tropocollagen.
#include <glad/gl.h>

// hybrid-graphics laptops: request the discrete GPU
extern "C" {
__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "camera.h"
#include "io.h"
#include "kmc.h"
#include "mcdock.h"
#include "we.h"
#include "render.h"
#include "shader.h"
#include "sim.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static OrbitCamera g_cam;
static bool g_rotating = false, g_panning = false;
static double g_mx = 0, g_my = 0;
static int g_appMode = 0;          // 0 BD, 1 MC docking, 2 KMC
static McDock g_mcd;
static int g_mcdTestN = 300;
static int g_mcdTestSweeps = 2000;
static bool g_mcdAnneal = false;
// weighted-ensemble test knobs (pass BEFORE --wetest; the arg loop returns on it)
static int g_weNmol = 40, g_weTarget = 3, g_weTau = 1500, g_weIters = 60;
static int g_weBrute = 24, g_weBruteSteps = 30000, g_weWalkers = 4;
static bool g_weTest = false;
static WeRun g_we;
static WeParams g_weUi;
static bool g_weActive = false;
static int g_weIdx = 0;
static int g_mcdSweeps = 4;

static void mouseButton(GLFWwindow* w, int b, int a, int) {
    if (ImGui::GetIO().WantCaptureMouse) { g_rotating = g_panning = false; return; }
    if (b == GLFW_MOUSE_BUTTON_LEFT) g_rotating = (a == GLFW_PRESS);
    if (b == GLFW_MOUSE_BUTTON_MIDDLE || b == GLFW_MOUSE_BUTTON_RIGHT) g_panning = (a == GLFW_PRESS);
    glfwGetCursorPos(w, &g_mx, &g_my);
}
static void mouseMove(GLFWwindow*, double x, double y) {
    float dx = float(x - g_mx), dy = float(y - g_my);
    g_mx = x; g_my = y;
    if (g_rotating) g_cam.rotate(dx, dy);
    if (g_panning) g_cam.pan(dx, dy);
}
static void mouseWheel(GLFWwindow*, double, double dy) {
    if (!ImGui::GetIO().WantCaptureMouse) g_cam.zoom(float(dy));
}

struct Args {
    int frames = -1, every = 60, steps = -1, nmol = -1;
    std::string shot;
    bool hidden = false;
    uint32_t seed = 0;
    float camDist = 0, yaw = 999, pitch = 999, lod = -1;
    float epsEl = -1, epsHy = -1, epsNs = -1;
    glm::vec3 box{0};
    int preset = -1;    // 0 dilute, 1 crowded quench, 2 dense nucleation
    int colorMode = -1;
    int scenario = 0;
    bool bench = false;
    float pairStag = 0;
    int registry = -1;
    float pH = -1, temp = -999, fibRad = -1, fibLen = -1;
    int pbc = 0;
    int mc = 0;
    int xlink = 0;
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](int def) { return (i + 1 < argc) ? atoi(argv[++i]) : def; };
        if (s == "--frames") a.frames = next(600);
        else if (s == "--every") a.every = next(60);
        else if (s == "--steps") a.steps = next(200);
        else if (s == "--nmol") a.nmol = next(400);
        else if (s == "--seed") a.seed = (uint32_t)next(1234);
        else if (s == "--shot" && i + 1 < argc) a.shot = argv[++i];
        else if (s == "--hidden") a.hidden = true;
        else if (s == "--camdist" && i + 1 < argc) a.camDist = (float)atof(argv[++i]);
        else if (s == "--yaw" && i + 1 < argc) a.yaw = (float)atof(argv[++i]);
        else if (s == "--pitch" && i + 1 < argc) a.pitch = (float)atof(argv[++i]);
        else if (s == "--lod" && i + 1 < argc) a.lod = (float)atof(argv[++i]);
        else if (s == "--epsel" && i + 1 < argc) a.epsEl = (float)atof(argv[++i]);
        else if (s == "--epshy" && i + 1 < argc) a.epsHy = (float)atof(argv[++i]);
        else if (s == "--epsns" && i + 1 < argc) a.epsNs = (float)atof(argv[++i]);
        else if (s == "--preset" && i + 1 < argc) a.preset = next(0);
        else if (s == "--colormode" && i + 1 < argc) a.colorMode = next(0);
        else if (s == "--scenario" && i + 1 < argc) a.scenario = next(0);
        else if (s == "--bench") a.bench = true;
        else if (s == "--pairstag" && i + 1 < argc) a.pairStag = (float)atof(argv[++i]);
        else if (s == "--registry" && i + 1 < argc) a.registry = next(2);
        else if (s == "--ph" && i + 1 < argc) a.pH = (float)atof(argv[++i]);
        else if (s == "--temp" && i + 1 < argc) a.temp = (float)atof(argv[++i]);
        else if (s == "--fibrad" && i + 1 < argc) a.fibRad = (float)atof(argv[++i]);
        else if (s == "--fiblen" && i + 1 < argc) a.fibLen = (float)atof(argv[++i]);
        else if (s == "--pbc") a.pbc = 1;
        else if (s == "--mc") a.mc = 1;
        else if (s == "--xlink") a.xlink = 1;
        else if (s == "--box" && i + 3 < argc) {
            a.box.x = (float)atof(argv[++i]);
            a.box.y = (float)atof(argv[++i]);
            a.box.z = (float)atof(argv[++i]);
        }
    }
    return a;
}

static void savePng(const std::string& path, int w, int h) {
    std::vector<unsigned char> px(size_t(w) * h * 3), flip(px.size());
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    for (int y = 0; y < h; ++y)
        memcpy(&flip[size_t(y) * w * 3], &px[size_t(h - 1 - y) * w * 3], size_t(w) * 3);
    stbi_write_png(path.c_str(), w, h, 3, flip.data(), w * 3);
    printf("wrote %s\n", path.c_str());
}

static void applyPreset(SimParams& p, int which) {
    p.scenario = 0;
    switch (which) {
        case 0:  // dilute solution: purist mode, slow nucleation
            p.nMol = 400; p.boxHalf = {250, 250, 330};
            p.initNematic = 0; p.epsDep = 0.f; p.speed = 10.f;
            p.epsEl = 2.5f; p.epsHy = 1.0f; p.epsNs = 0.15f;
            break;
        case 1:  // crowded quench: aligned start, registry wells do the work
            p.nMol = 700; p.boxHalf = {140, 140, 240};
            p.initNematic = 1; p.epsDep = 0.f; p.speed = 10.f;
            p.epsEl = 3.0f; p.epsHy = 1.2f; p.epsNs = 0.2f;
            break;
        case 2:  // dense nucleation: isotropic, mild crowding
            p.nMol = 700; p.boxHalf = {170, 170, 260};
            p.initNematic = 0; p.epsDep = 0.3f; p.depR = 1.8f; p.speed = 15.f;
            p.epsEl = 3.0f; p.epsHy = 1.2f; p.epsNs = 0.2f;
            break;
    }
}

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    if (!glfwInit()) { fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (args.hidden) glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(1600, 1000, "collagenSim â€” type I collagen self-assembly", nullptr, nullptr);
    if (!win) { fprintf(stderr, "window creation failed\n"); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(glfwGetProcAddress)) { fprintf(stderr, "gladLoadGL failed\n"); return 1; }
    glfwSwapInterval(args.frames > 0 ? 0 : 1);
    printf("GL: %s | %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));

    glfwSetMouseButtonCallback(win, mouseButton);
    glfwSetCursorPosCallback(win, mouseMove);
    glfwSetScrollCallback(win, mouseWheel);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    {
        ImGuiStyle& st = ImGui::GetStyle();
        st.FrameRounding = 4.f; st.GrabRounding = 4.f; st.WindowRounding = 0.f;
        st.WindowPadding = {14, 12}; st.ItemSpacing = {8, 6}; st.ScrollbarSize = 12.f;
        ImVec4* c = st.Colors;
        c[ImGuiCol_WindowBg] = {0.055f, 0.065f, 0.09f, 0.98f};
        c[ImGuiCol_Header] = {0.10f, 0.30f, 0.38f, 1.f};
        c[ImGuiCol_HeaderHovered] = {0.13f, 0.38f, 0.48f, 1.f};
        c[ImGuiCol_HeaderActive] = {0.16f, 0.45f, 0.56f, 1.f};
        c[ImGuiCol_FrameBg] = {0.10f, 0.13f, 0.18f, 1.f};
        c[ImGuiCol_FrameBgHovered] = {0.13f, 0.17f, 0.24f, 1.f};
        c[ImGuiCol_FrameBgActive] = {0.16f, 0.21f, 0.29f, 1.f};
        c[ImGuiCol_SliderGrab] = {0.22f, 0.62f, 0.78f, 1.f};
        c[ImGuiCol_SliderGrabActive] = {0.30f, 0.75f, 0.92f, 1.f};
        c[ImGuiCol_Button] = {0.12f, 0.32f, 0.42f, 1.f};
        c[ImGuiCol_ButtonHovered] = {0.16f, 0.42f, 0.55f, 1.f};
        c[ImGuiCol_ButtonActive] = {0.20f, 0.52f, 0.68f, 1.f};
        c[ImGuiCol_CheckMark] = {0.30f, 0.78f, 0.95f, 1.f};
        c[ImGuiCol_Separator] = {0.20f, 0.28f, 0.36f, 1.f};
    }
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 460");

    AtomTemplate tmpl = loadAtoms(resolveAsset("assets/atoms.bin"));
    Profiles prof = loadProfiles(resolveAsset("assets/profiles.bin"));
    Profiles2D prof2 = loadProfiles2D(resolveAsset("assets/profiles2d.bin"));
    Basis2D basis = loadBasis2D(resolveAsset("assets/profiles2d_basis.bin"));
    Profiles2D prof2A;
    Basis2D basisA;
    bool haveAtelo = false;
    try {
        prof2A = loadProfiles2D(resolveAsset("assets/profiles2d_atelo.bin"));
        basisA = loadBasis2D(resolveAsset("assets/profiles2d_basis_atelo.bin"));
        haveAtelo = true;
    } catch (...) {
        printf("no atelo tables (run tools/azimuthal_profile.py --atelo)\n");
    }
    Corr2D corrTab;
    bool haveCorr = false;
    try {
        corrTab = loadCorr2D(resolveAsset("assets/correction2d.bin"));
        haveCorr = true;
        printf("measured correction table loaded (%ux%ux%u)\n",
               corrTab.nD, corrTab.nPhi, corrTab.nPhi);
    } catch (...) {
        printf("no measured corrections (run tools/pmf_merge.py)\n");
    }
    printf("template: %u atoms, L=%.1f nm, D=%.2f nm | 2D registry %ux%ux%u\n",
           tmpl.natoms, tmpl.lengthNm, tmpl.dPeriodNm, prof2.nD, prof2.nPhi, prof2.nPhi);

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--mclinks" && i + 1 < argc)
            g_mcd.p.nLinks = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--mcbend" && i + 1 < argc)
            g_mcd.p.bendOn = (float)atof(argv[i + 1]);
        if (std::string(argv[i]) == "--mcnmol" && i + 1 < argc)
            g_mcdTestN = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--mcsweeps" && i + 1 < argc)
            g_mcdTestSweeps = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--mcnodock") g_mcd.p.dockMoves = false;
        if (std::string(argv[i]) == "--mcanneal") g_mcdAnneal = true;
        if (std::string(argv[i]) == "--mcdocktest") {
            g_mcd.buildPoseLibrary(basis, prof2.par, 7.4f, 2.5f, 1.0f,
                                   haveCorr ? &corrTab : nullptr);
            float bxy = g_mcdTestN <= 8 ? 20.f : 120.f;   // tight box for pair tests
            g_mcd.initFromScatter(g_mcdTestN, {bxy, bxy, 210}, tmpl.lengthNm, true, 1234);
            printf("links %d (%.0f nm), bend x%.2f -> Lp %.0f nm, N %d\n",
                   g_mcd.p.nLinks, g_mcd.linkLen, g_mcd.p.bendOn,
                   g_mcd.p.bendOn * g_mcd.persistLen, g_mcdTestN);
            double t0 = glfwGetTime();
            int nrep = std::max(1, g_mcdTestSweeps / 100);
            for (int rep = 0; rep < nrep; ++rep) {
                if (g_mcdAnneal && nrep > 1)   // simulated anneal 3 kT -> 1 kT
                    g_mcd.p.tempFactor = 1.f + 2.f * (1.f - float(rep) / (nrep - 1));
                g_mcd.sweep(100);
                float ms = 0;
                int nc = g_mcd.clusterCount(ms);
                printf("sweep %5lld | t=%.3f ms | bound %.0f%% | clusters %d (%.1f) | "
                       "band %.2f Dfrac %.0f%% | acc %.1f%% (T%.1f D%.1f S%.1f P%.1f)\n",
                       (long long)g_mcd.sweepsDone, g_mcd.simTimeNs * 1e-6,
                       g_mcd.boundFraction() * 100.f, nc, ms, g_mcd.bandCoherence(),
                       g_mcd.dFraction() * 100.f,
                       g_mcd.proposed ? 100.0 * g_mcd.accepted / g_mcd.proposed : 0.0,
                       100.0 * g_mcd.accWhole / std::max(1.0, (double)g_mcd.proposed),
                       100.0 * g_mcd.accDock / std::max(1.0, (double)g_mcd.proposed),
                       100.0 * g_mcd.accSlide / std::max(1.0, (double)g_mcd.proposed),
                       100.0 * g_mcd.accPivot / std::max(1.0, (double)g_mcd.proposed));
            }
            double el = glfwGetTime() - t0;
            if (g_mcdTestN <= 8) g_mcd.dumpContacts(40);
            {   // render path: bond spacing must hold and chains must bend
                std::vector<glm::vec4> bt;
                const int nb = 96;
                float sl = tmpl.lengthNm / (nb - 1);
                g_mcd.writeBeads(bt, nb, sl);
                float maxDev = 0, sumR = 0;
                int nm = std::min((int)g_mcd.mols.size(), 50);
                for (int m = 0; m < nm; ++m) {
                    for (int i = 0; i + 1 < nb; ++i) {
                        glm::vec3 a(bt[m * nb + i]), b(bt[m * nb + i + 1]);
                        maxDev = std::max(maxDev, fabsf(glm::length(b - a) - sl));
                    }
                    glm::vec3 e0(bt[m * nb]), e1(bt[m * nb + nb - 1]);
                    sumR += glm::length(e1 - e0);
                }
                printf("beads: max bond deviation %.4f nm | <Ree>/L %.2f (WLC ~0.57 free)\n",
                       maxDev, sumR / nm / tmpl.lengthNm);
            }
            double nsw = nrep * 100.0;
            printf("clock: %.2f us/sweep calibrated on %d free mols (MSD %.3f nm^2), "
                   "uncorrected sigma^2/6D would say %.1f us%s\n",
                   g_mcd.stepTimeNs * 1e-3, g_mcd.lastFreeCount, g_mcd.lastMsdFree,
                   g_mcd.stepTimeNominalNs * 1e-3,
                   g_mcd.clockExtrapolated ? " [EXTRAPOLATED]" : "");
            printf("%.0f sweeps in %.1f s = %.0f sweeps/s -> %.3g sim-seconds per wall-hour\n",
                   nsw, el, nsw / el, nsw / el * g_mcd.stepTimeNs * 1e-9 * 3600.0);
            return 0;
        }
        if (std::string(argv[i]) == "--wetest") g_weTest = true;
        if (std::string(argv[i]) == "--wenmol" && i + 1 < argc) g_weNmol = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--wetarget" && i + 1 < argc) g_weTarget = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--wetau" && i + 1 < argc) g_weTau = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--weiters" && i + 1 < argc) g_weIters = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--webrute" && i + 1 < argc) g_weBrute = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--webrutesteps" && i + 1 < argc)
            g_weBruteSteps = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--wewalkers" && i + 1 < argc)
            g_weWalkers = atoi(argv[i + 1]);
        if (std::string(argv[i]) == "--kmctest") {
            KmcParams kp;
            const Corr2D* ctt = haveCorr ? &corrTab : nullptr;
            KmcProfile prT = kmcProfile1D(basis, prof2.par, 7.4f, 2.5f, 1.0f, ctt);
            KmcProfile prAcid = kmcProfile1D(basis, prof2.par, 3.5f, 2.5f, 1.0f, ctt);
            for (double c : {0.3, 1.0, 3.0}) {
                KmcResult r = kmcRun(kp, prT, prT, c, 284.7, 37.0, 7);
                printf("telo %.1f mg/mL 37C: lag %5.1f min, t50 %5.1f, plateau %3.0f%%, "
                       "band order %.2f\n",
                       c, r.lagMin, r.t50Min, r.plateau * 100, r.finalOrder);
            }
            KmcParams slow = kp; slow.kplus = 5e4;   // slow growth: anneal better
            KmcResult rs = kmcRun(slow, prT, prT, 1.0, 284.7, 37.0, 7);
            printf("slow growth (k+ /10): lag %5.1f, band order %.2f\n",
                   rs.lagMin, rs.finalOrder);
            KmcParams narrow = kp; narrow.scanWinNm = 3.0;
            KmcResult rn = kmcRun(narrow, prT, prT, 1.0, 284.7, 37.0, 7);
            printf("narrow scan (3 nm):  lag %5.1f, band order %.2f\n",
                   rn.lagMin, rn.finalOrder);
            KmcParams wide = kp; wide.scanWinNm = 70.0;
            KmcResult rw = kmcRun(wide, prT, prT, 1.0, 284.7, 37.0, 7);
            printf("wide scan (70 nm):   lag %5.1f, band order %.2f\n",
                   rw.lagMin, rw.finalOrder);
            KmcResult rp = kmcRun(kp, prAcid, prT, 1.0, 284.7, 37.0, 7);
            printf("telo pH3.5 37C: lag %.1f min, plateau %.0f%%\n",
                   rp.lagMin, rp.plateau * 100);
            KmcProfile prF = kmcProfileFunnel(prof, 2.5f, 1.0f,
                                              tmpl.dPeriodNm, tmpl.lengthNm);
            KmcResult rf = kmcRun(kp, prF, prF, 1.0, 284.7, 37.0, 7);
            printf("funnel (dock-and-lock) 1.0 mg/mL: lag %5.1f, band order %.2f\n",
                   rf.lagMin, rf.finalOrder);
            KmcParams slowF = kp; slowF.kplus = 5e4;
            KmcResult rf2 = kmcRun(slowF, prF, prF, 1.0, 284.7, 37.0, 7);
            printf("funnel slow growth:               lag %5.1f, band order %.2f\n",
                   rf2.lagMin, rf2.finalOrder);
            return 0;
        }
    }

    Sim sim;
    if (args.preset >= 0) applyPreset(sim.p, args.preset);
    if (args.scenario) sim.p.scenario = args.scenario;
    if (args.pairStag != 0) sim.p.pairStag = args.pairStag;
    if (args.nmol > 0) sim.p.nMol = args.nmol;
    if (args.seed) sim.p.seed = args.seed;
    if (args.epsEl >= 0) sim.p.epsEl = args.epsEl;
    if (args.epsHy >= 0) sim.p.epsHy = args.epsHy;
    if (args.epsNs >= 0) sim.p.epsNs = args.epsNs;
    if (args.box.x > 0) sim.p.boxHalf = args.box;
    if (args.registry >= 0) sim.p.registryMode = args.registry;
    if (args.pbc) sim.p.pbc = 1;
    if (args.mc) sim.p.mcBoost = 1;
    if (args.xlink) sim.p.xlink = 1;
    if (args.pH > 0) sim.p.pH = args.pH;
    if (args.temp > -900) sim.p.tempC = args.temp;
    if (args.scenario == 4) {
        if (args.fibRad > 0) sim.p.fibrilRad = args.fibRad;
        if (args.fibLen > 0) sim.p.fibrilLenUm = args.fibLen;
    }
    sim.init(tmpl, prof, prof2, basis, haveAtelo ? &prof2A : nullptr,
             haveAtelo ? &basisA : nullptr, haveCorr ? &corrTab : nullptr);
    if (g_weTest) {
            WeParams wp;
            wp.targetSize = g_weTarget;
            wp.tauSteps = g_weTau;
            wp.walkersPerBin = g_weWalkers;
            sim.p.nMol = g_weNmol;
            sim.p.boxHalf = {60.f, 60.f, 200.f};
            sim.p.pbc = 1;
            sim.p.scenario = 0;
            sim.p.registryMode = 2;
            sim.restart();
            double volNm3 = 8.0 * 60.0 * 60.0 * 200.0;
            // g per molecule -> g/mL over the box volume -> mg/mL
            double mgml = g_weNmol * sim.mwKda * 1000.0 / 6.022e23 /
                          (volNm3 * 1e-21) * 1000.0;
            printf("WE system: %d molecules in 120x120x400 nm (%.2f mg/mL), "
                   "target cluster %d, tau %d steps\n",
                   g_weNmol, mgml, wp.targetSize, wp.tauSteps);

            // --- brute force reference: events / total observation time ---
            double bruteTimeNs = 0;
            int bruteEvents = 0;
            double t0 = glfwGetTime();
            for (int r = 0; r < g_weBrute; ++r) {
                sim.p.seed = 9000u + r;
                sim.restart();
                sim.step(1500);      // finish the soft-start ramp, then clear it
                sim.burnin = 0;      // so WE walkers and this run see identical
                sim.step(300);       // interaction strength from the clock's t=0
                double t0ns = sim.simTimeNs;
                SimState st;
                sim.snapshot(st);
                if (largestCluster(st.pos, sim.p.nMol, sim.p.nBeads, wp.contactNm,
                                   true, sim.p.boxHalf) >= wp.targetSize) {
                    printf("  run %d started already nucleated -- reactant state is "
                           "not clean; lower the concentration\n", r);
                    continue;
                }
                for (int s = 0; s < g_weBruteSteps; s += 500) {
                    sim.step(500);
                    sim.snapshot(st);
                    int lam = largestCluster(st.pos, sim.p.nMol, sim.p.nBeads,
                                             wp.contactNm, true, sim.p.boxHalf);
                    if (lam >= wp.targetSize) { bruteEvents++; break; }
                }
                bruteTimeNs += sim.simTimeNs - t0ns;
            }
            double bruteRate = bruteTimeNs > 0 ? bruteEvents / bruteTimeNs : 0.0;
            double bruteWall = glfwGetTime() - t0;
            printf("brute force: %d events in %.1f ns observed (%d runs, %.0f s wall)"
                   " -> k = %.4g /ns\n",
                   bruteEvents, bruteTimeNs, g_weBrute, bruteWall, bruteRate);
            if (bruteEvents > 0)
                printf("             +/- %.4g (Poisson, 1/sqrt(N))\n",
                       bruteRate / sqrt((double)bruteEvents));

            // --- weighted ensemble on the same system ---
            t0 = glfwGetTime();
            WeRun we;
            we.init(sim, wp, 4242u);
            we.burnInIters = std::max(2, g_weIters / 4);
            printf("WE init: %zu clean starts (%d rejected as pre-nucleated)\n",
                   we.initPool.size(), we.poolRejects);
            if (we.initPool.empty()) {
                printf("ABORT: no clean reactant state at this concentration\n");
                return 0;
            }
            for (int it = 0; it < g_weIters; ++it) {
                we.iterate(sim);
                if ((it + 1) % 5 == 0 || it == g_weIters - 1) {
                    printf("  WE iter %3d | t %.1f ns | walkers %2zu | recycles %3d "
                           "| k %.4g /ns | bins",
                           it + 1, we.timeNs, we.walkers.size(), we.recycles,
                           we.ratePerNs());
                    for (int b = 0; b < (int)we.binPop.size(); ++b)
                        printf(" %d:%.1e", b + 1, we.binWeight[b]);
                    printf("\n");
                    fflush(stdout);
                }
            }
            double weWall = glfwGetTime() - t0;
            printf("WE: k = %.4g /ns steady-state (%.4g including burn-in) from "
                   "%.1f ns post-burn-in x %zu walkers, %d/%d recycles, %.0f s wall\n",
                   we.rateSSPerNs(), we.ratePerNs(), we.timeNsSS, we.walkers.size(),
                   we.recyclesSS, we.recycles, weWall);
            if (we.recyclesSS > 0)
                printf("    +/- %.4g (Poisson on %d steady-state crossings)\n",
                       we.rateSSPerNs() / sqrt((double)we.recyclesSS), we.recyclesSS);
            // WE can only resolve events rarer than one per tau: a walker is
            // harvested at most once per iteration, so the estimator saturates
            // at 1/tau and any faster true rate is reported as 1/tau.
            double mfptBrute = bruteRate > 0 ? 1.0 / bruteRate : 0.0;
            if (mfptBrute > 0 && mfptBrute < 5.0 * we.tauNs)
                printf("WARNING: true MFPT %.0f ns is only %.1f tau -- WE saturates "
                       "at 1/tau = %.4g /ns here. Lower the concentration or raise "
                       "the target to test it properly.\n",
                       mfptBrute, mfptBrute / we.tauNs, 1.0 / we.tauNs);
            if (bruteRate > 0 && we.rateSSPerNs() > 0)
                printf("VERDICT: WE/brute = %.2fx  (agreement within ~2x is the bar "
                       "for a rare-event estimator)\n",
                       we.rateSSPerNs() / bruteRate);
            else
                printf("VERDICT: inconclusive (one estimator saw no events)\n");
            return 0;
    }

    Renderer ren;
    ren.init(tmpl);
    if (args.camDist > 0) g_cam.dist = args.camDist;
    if (args.yaw < 900) g_cam.yaw = args.yaw;
    if (args.pitch < 900) g_cam.pitch = args.pitch;
    if (args.lod >= 0) ren.opts.lodDist = args.lod;
    if (args.colorMode >= 0) ren.opts.colorMode = args.colorMode;

    // offscreen target so screenshots work even hidden; recreated on resize
    int fbw = 0, fbh = 0;
    GLuint fbo = 0, colorTex = 0, depthRb = 0;
    auto ensureFbo = [&](int w, int h) {
        if (w == fbw && h == fbh && fbo) return;
        if (fbo) {
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &colorTex);
            glDeleteRenderbuffers(1, &depthRb);
        }
        fbw = w; fbh = h;
        glCreateFramebuffers(1, &fbo);
        glCreateTextures(GL_TEXTURE_2D, 1, &colorTex);
        glTextureStorage2D(colorTex, 1, GL_RGBA8, fbw, fbh);
        glCreateRenderbuffers(1, &depthRb);
        glNamedRenderbufferStorage(depthRb, GL_DEPTH_COMPONENT32F, fbw, fbh);
        glNamedFramebufferTexture(fbo, GL_COLOR_ATTACHMENT0, colorTex, 0);
        glNamedFramebufferRenderbuffer(fbo, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRb);
    };
    {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        ensureFbo(w, h);
    }

    // bench mode: pure physics, console metrics, zero rendering
    if (args.bench) {
        int frames = args.frames > 0 ? args.frames : 400;
        int steps = args.steps > 0 ? args.steps : 400;
        std::vector<float> buf(size_t(sim.nBead()) * 4);
        double t0 = glfwGetTime();
        for (int f = 0; f <= frames; ++f) {
            sim.step(steps);
            if (sim.p.mcBoost) sim.mcPass(sim.p.mcPasses);
            if (f % args.every == 0) {
                glGetNamedBufferSubData(sim.posBuf, 0, sizeof(float) * buf.size(), buf.data());
                int M = sim.p.nBeads;
                printf("t=%8.2f us  ", sim.simTimeNs * 1e-3);
                double z0 = 0, x0 = 0, y0 = 0;
                for (int m = 0; m < sim.p.nMol && m < 7; ++m) {
                    double zm = 0, xm = 0, ym = 0;
                    for (int i = 0; i < M; ++i) {
                        const float* p = &buf[(size_t(m) * M + i) * 4];
                        xm += p[0]; ym += p[1]; zm += p[2];
                    }
                    xm /= M; ym /= M; zm /= M;
                    if (m == 0) { z0 = zm; x0 = xm; y0 = ym; continue; }
                    double dz = zm - z0;
                    double dperp = sqrt((xm - x0) * (xm - x0) + (ym - y0) * (ym - y0));
                    printf("| dz %7.2f dperp %5.2f ", dz, dperp);
                }
                sim.readStats();
                printf("| ov %u\n", sim.stats[0] / 2);
                fflush(stdout);
            }
        }
        double el = glfwGetTime() - t0;
        printf("bench: %d steps in %.1f s = %.0f steps/s\n",
               (frames + 1) * steps, el, (frames + 1) * steps / el);
        return 0;
    }

    bool running = true;
    bool autoDt = true;
    int stepsPerFrame = args.steps > 0 ? args.steps : 60;
    bool autoPace = args.steps <= 0;      // adapt steps to the frame budget
    float targetFps = 30.f;
    float smoothAlpha = 0.35f;
    int frameIdx = 0, shotIdx = 0;
    double lastT = glfwGetTime(), fps = 0, spsAvg = 0;

    while (!glfwWindowShouldClose(win)) {
        double frameStart = glfwGetTime();
        glfwPollEvents();

        if (running && g_appMode == 0) sim.step(stepsPerFrame);
        if (running && g_appMode == 3 && g_weActive && !g_we.walkers.empty()) {
            g_we.advanceWalker(sim, g_weIdx);
            if (++g_weIdx >= (int)g_we.walkers.size()) {
                g_we.finishIteration(sim);
                g_weIdx = 0;
            }
        }
        if (running && g_appMode == 1 && g_mcd.mols.size() == (size_t)sim.p.nMol) {
            g_mcd.sweep(g_mcdSweeps);
            static std::vector<glm::vec4> beadTmp;
            g_mcd.writeBeads(beadTmp, sim.p.nBeads, sim.segLen);
            glNamedBufferSubData(sim.posBuf, 0, sizeof(glm::vec4) * beadTmp.size(),
                                 beadTmp.data());
        }
        sim.smoothForDisplay(smoothAlpha, frameIdx == 0);
        sim.computeFrames(sim.renderPosBuf);
        if (frameIdx % 2 == 0) {
            sim.updateCenters(sim.renderPosBuf);
            ren.updateLod(sim, g_cam.eye());
        }

        {
            int w, h;
            glfwGetFramebufferSize(win, &w, &h);
            if (w <= 0 || h <= 0) { frameIdx++; continue; }   // minimized
            ensureFbo(w, h);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, fbw, fbh);
        glClearColor(ren.opts.fogColor.r, ren.opts.fogColor.g, ren.opts.fogColor.b, 1.f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        float aspect = float(fbw) / float(fbh);
        ren.draw(sim, g_cam.view(), g_cam.proj(aspect), (float)glfwGetTime());

        if (!args.shot.empty() && frameIdx % args.every == 0) {
            char buf[512];
            snprintf(buf, sizeof buf, "%s_%04d.png", args.shot.c_str(), shotIdx++);
            savePng(resolveAsset(buf), fbw, fbh);
            sim.readStats();
            printf("  t=%.2f us  contacts=%u  D-staggered=%u  overlaps=%u\n",
                   sim.simTimeNs * 1e-3, sim.stats[1] / 2, sim.stats[2] / 2, sim.stats[0] / 2);
            fflush(stdout);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, fbw, fbh);
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBlitNamedFramebuffer(fbo, 0, 0, 0, fbw, fbh, 0, 0, fbw, fbh,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);

        double now = glfwGetTime();
        double dtw = now - lastT;
        lastT = now;
        fps = 0.9 * fps + 0.1 / std::max(dtw, 1e-4);
        spsAvg = 0.9 * spsAvg + 0.1 * (running ? stepsPerFrame / std::max(dtw, 1e-4) : 0);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGuiIO& gio = ImGui::GetIO();
            const float panelW = 430.f;
            ImGui::SetNextWindowPos({gio.DisplaySize.x - panelW, 0}, ImGuiCond_Always);
            ImGui::SetNextWindowSize({panelW, gio.DisplaySize.y}, ImGuiCond_Always);
        }
        ImGui::Begin("collagenSim", nullptr,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar);
        ImGui::TextColored({0.35f, 0.8f, 0.95f, 1.f}, "collagenSim");
        ImGui::SameLine();
        ImGui::TextDisabled("type I fibrillogenesis");
        ImGui::Separator();

        static KmcParams kmcP;
        static KmcResult kmcR, kmcRef;
        static bool kmcDirty = true, kmcHaveRef = false;
        static float kmcConc = 1.0f;
        static int kmcSeed = 7;
        int& appMode = g_appMode;
        if (ImGui::BeginTabBar("mode")) {
            if (ImGui::BeginTabItem("BD (Langevin)")) { appMode = 0; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("MC (docking)")) { appMode = 1; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("KMC (kinetics)")) { appMode = 2; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("WE (rare events)")) { appMode = 3; ImGui::EndTabItem(); }
            ImGui::EndTabBar();
        }
        if (appMode == 3) {
            ImGui::TextWrapped("Weighted ensemble over the BD engine. Every walker is "
                               "plain unadjusted Brownian dynamics -- WE only moves "
                               "computational effort, splitting walkers that climb the "
                               "coordinate and merging ones that fall back, carrying "
                               "weights that keep the ensemble exact. Rates come out "
                               "unbiased, so this reaches rare events without giving up "
                               "the one engine here whose clock is trustworthy.");
            ImGui::Separator();
            ImGui::SliderInt("target cluster", &g_weUi.targetSize, 2, 12);
            ImGui::SliderInt("tau (BD steps)", &g_weUi.tauSteps, 100, 5000);
            ImGui::SliderInt("walkers per bin", &g_weUi.walkersPerBin, 2, 12);
            ImGui::SliderFloat("approach range (nm)", &g_weUi.approachNm, 5.f, 60.f);
            if (ImGui::Button(g_weActive ? "restart WE" : "start WE")) {
                g_we.init(sim, g_weUi, (uint32_t)glfwGetTime() * 977u + 13u);
                g_we.burnInIters = 10;
                g_weActive = true;
                g_weIdx = 0;
                running = true;
            }
            ImGui::SameLine();
            if (ImGui::Button(running ? "pause##we" : "run##we")) running = !running;
            if (g_weActive) {
                ImGui::Separator();
                ImGui::Text("iteration %d | walkers %zu | walker %d advancing",
                            g_we.iters, g_we.walkers.size(), g_weIdx);
                ImGui::Text("ensemble time %.1f ns | crossings %d (%d steady-state)",
                            g_we.timeNs, g_we.recycles, g_we.recyclesSS);
                double kss = g_we.rateSSPerNs();
                if (kss > 0) {
                    double mfptNs = 1.0 / kss;
                    const char* u = "ns";
                    double v = mfptNs;
                    if (v > 6e10) { v /= 6e10; u = "min"; }
                    else if (v > 1e9) { v /= 1e9; u = "s"; }
                    else if (v > 1e6) { v /= 1e6; u = "ms"; }
                    else if (v > 1e3) { v /= 1e3; u = "us"; }
                    ImGui::Text("rate %.4g /ns  ->  mean first passage %.2f %s", kss, v, u);
                    if (g_we.recyclesSS > 0)
                        ImGui::TextDisabled("  +/- %.0f%% (Poisson, %d crossings)",
                                            100.0 / sqrt((double)g_we.recyclesSS),
                                            g_we.recyclesSS);
                    if (g_we.iters <= g_we.burnInIters)
                        ImGui::TextDisabled("  still in burn-in; rate not yet meaningful");
                } else {
                    ImGui::TextDisabled("no crossings yet");
                }
                ImGui::Text("weight by coordinate (target cluster %d):",
                            g_weUi.targetSize);
                for (int b = 0; b < (int)g_we.binWeight.size(); ++b) {
                    if (g_we.binPop[b] == 0) continue;
                    float lam = 1.f + float(b) / std::max(1, g_weUi.subBins);
                    ImGui::Text("  lambda %.2f : w %.2e  (%d walkers)", lam,
                                g_we.binWeight[b], g_we.binPop[b]);
                }
            }
            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(win);
            glFinish();
            frameIdx++;
            continue;
        }
        if (appMode == 1) {
            // docking-based Monte Carlo (Vakser et al., PNAS 2022 adapted to rods)
            static bool mcdInit = false;
            static float lastPH = -1, lastEl = -1, lastHy = -1;
            static int lastAtelo = -1;
            ImGui::TextWrapped("Segmental docking MC: each molecule is a chain of "
                               "persistence-length links. Moves are adaptive rigid "
                               "transport, link-dock hops onto pose-library minima "
                               "(chain dragged along), and joint pivots; energies are "
                               "contact integrals of the measured registry kernel, "
                               "Metropolis with Hastings/Ni-Nj corrections.");
            if (ImGui::Button(mcdInit ? "reset MC system" : "start MC system") || !mcdInit) {
                if (sim.p.pH != lastPH || sim.p.epsEl != lastEl || sim.p.epsHy != lastHy ||
                    sim.p.atelo != lastAtelo || g_mcd.poses.empty()) {
                    const Basis2D& bAct = (sim.p.atelo && haveAtelo) ? basisA : basis;
                    const Profiles2D& pAct = (sim.p.atelo && haveAtelo) ? prof2A : prof2;
                    g_mcd.buildPoseLibrary(bAct, pAct.par, sim.p.pH, sim.p.epsEl,
                                           sim.p.epsHy, haveCorr ? &corrTab : nullptr,
                                           sim.p.epsCorr);
                    lastPH = sim.p.pH; lastEl = sim.p.epsEl; lastHy = sim.p.epsHy;
                    lastAtelo = sim.p.atelo;
                }
                g_mcd.initFromScatter(sim.p.nMol, sim.p.boxHalf, sim.molLen,
                                      sim.p.pbc == 1, sim.p.seed);
                mcdInit = true;
            }
            if (ImGui::Button(running ? "pause##mcd" : "run##mcd")) running = !running;
            ImGui::SliderInt("sweeps/frame", &g_mcdSweeps, 1, 40);
            ImGui::SliderInt("links/molecule", &g_mcd.p.nLinks, 1, 12);
            ImGui::TextDisabled("link %.0f nm (Lp %.0f nm) - reset to apply",
                                g_mcd.linkLen, g_mcd.persistLen);
            ImGui::SliderFloat("hop scale (nm)", &g_mcd.p.sigmaFree, 2.f, 40.f);
            ImGui::SliderFloat("temperature factor", &g_mcd.p.tempFactor, 0.2f, 4.f);
            ImGui::SliderFloat("capture radius (nm)", &g_mcd.p.captureR, 3.f, 15.f);
            ImGui::SliderFloat("pivot amplitude (rad)", &g_mcd.p.pivotAmp, 0.02f, 0.8f);
            ImGui::SliderFloat("bending stiffness x", &g_mcd.p.bendOn, 0.f, 6.f);
            ImGui::TextDisabled("  -> effective Lp %.0f nm", g_mcd.p.bendOn * g_mcd.persistLen);
            ImGui::Checkbox("dock hops (biased basin finding)", &g_mcd.p.dockMoves);
            if (!g_mcd.p.dockMoves)
                ImGui::TextDisabled("  strictly reversible chain (transport/slide/pivot)");
            double tNs = g_mcd.simTimeNs;
            const char* unit = "ns";
            double tv = tNs;
            if (tv > 3.6e12) { tv /= 3.6e12; unit = "h"; }
            else if (tv > 6e10) { tv /= 6e10; unit = "min"; }
            else if (tv > 1e9) { tv /= 1e9; unit = "s"; }
            else if (tv > 1e6) { tv /= 1e6; unit = "ms"; }
            else if (tv > 1e3) { tv /= 1e3; unit = "us"; }
            ImGui::Separator();
            ImGui::Text("sim time: %.2f %s | step %.2f us | %lld sweeps",
                        tv, unit, g_mcd.stepTimeNs * 1e-3, (long long)g_mcd.sweepsDone);
            if (g_mcd.clockExtrapolated)
                ImGui::TextDisabled("  clock EXTRAPOLATED (no free molecules left)");
            else
                ImGui::TextDisabled("  calibrated on %d free mols, MSD %.3f nm^2/sweep"
                                    " (uncorrected would be %.1f us)",
                                    g_mcd.lastFreeCount, g_mcd.lastMsdFree,
                                    g_mcd.stepTimeNominalNs * 1e-3);
            ImGui::Text("acceptance %.1f%% | poses %zu",
                        g_mcd.proposed ? 100.0 * g_mcd.accepted / g_mcd.proposed : 0.0,
                        g_mcd.poses.size());
            double pr = std::max(1.0, (double)g_mcd.proposed);
            ImGui::Text("accepted: transport %.1f | dock %.1f | slide %.1f | pivot %.1f (%%all)",
                        100.0 * g_mcd.accWhole / pr, 100.0 * g_mcd.accDock / pr,
                        100.0 * g_mcd.accSlide / pr, 100.0 * g_mcd.accPivot / pr);
            float meanSz = 0;
            int nc = g_mcd.clusterCount(meanSz);
            ImGui::Text("bound %.0f%% | clusters %d (mean %.1f)",
                        g_mcd.boundFraction() * 100.f, nc, meanSz);
            ImGui::Text("band coherence %.2f | D-fraction %.0f%%",
                        g_mcd.bandCoherence(), g_mcd.dFraction() * 100.f);
            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(win);
            glFinish();
            double remainK = 1.0 / targetFps - (glfwGetTime() - frameStart);
            if (args.frames <= 0 && remainK > 0.002)
                std::this_thread::sleep_for(std::chrono::duration<double>(remainK));
            frameIdx++;
            if (args.frames > 0 && frameIdx >= args.frames) break;
            continue;
        }
        if (appMode == 2) {
            ImGui::TextWrapped("Gillespie kinetics: whole-molecule events with rates tied "
                               "to the registry energetics at the current environment. "
                               "Curves recompute instantly.");
            {
                const char* regModes[] = {"off", "1D funnel (dock-and-lock)",
                                          "2D azimuthal (mean-field)"};
                int rm = sim.p.registryMode;
                if (ImGui::Combo("registry##kmc", &rm, regModes, 3)) {
                    sim.p.registryMode = std::max(1, rm);   // KMC needs a landscape
                    kmcDirty = true;
                }
            }
            bool d = false;
            d |= ImGui::SliderFloat("conc (mg/mL)##kmc", &kmcConc, 0.05f, 5.f, "%.2f",
                                    ImGuiSliderFlags_Logarithmic);
            d |= ImGui::SliderFloat("temperature (C)##kmc", &sim.p.tempC, 4.f, 45.f, "%.1f");
            if (ImGui::SliderFloat("pH##kmc", &sim.p.pH, 2.5f, 11.f, "%.2f")) {
                sim.recombinePH();
                d = true;
            }
            if (sim.hasAtelo) {
                int at = sim.p.atelo;
                const char* forms[] = {"telocollagen", "atelocollagen"};
                if (ImGui::Combo("form##kmc", &at, forms, 2)) {
                    sim.p.atelo = at;
                    sim.recombinePH();
                    d = true;
                }
            }
            if (ImGui::TreeNode("rate model")) {
                float sw = (float)kmcP.scanWinNm;
                if (ImGui::SliderFloat("scan window (nm)", &sw, 2.f, 80.f)) {
                    kmcP.scanWinNm = sw; d = true;
                }
                ImGui::SetItemTooltip("how far a docked molecule explores registry\n"
                                      "before locking: small = kinetic trapping,\n"
                                      "large = thermodynamic (sharp banding)");
                float cp = (float)kmcP.contactPairs;
                if (ImGui::SliderFloat("contact pairs", &cp, 5.f, 60.f)) {
                    kmcP.contactPairs = cp; d = true;
                }
                d |= ImGui::SliderInt("nucleus size", &kmcP.nc, 2, 6);
                float kn = (float)log10(kmcP.kn), kp = (float)log10(kmcP.kplus);
                float ko = (float)log10(kmcP.koff0), kf = (float)log10(kmcP.kfrag);
                if (ImGui::SliderFloat("log10 k_nuc", &kn, 0.f, 8.f)) { kmcP.kn = pow(10, kn); d = true; }
                if (ImGui::SliderFloat("log10 k_plus", &kp, 3.f, 8.f)) { kmcP.kplus = pow(10, kp); d = true; }
                if (ImGui::SliderFloat("log10 k_off", &ko, -6.f, 0.f)) { kmcP.koff0 = pow(10, ko); d = true; }
                if (ImGui::SliderFloat("log10 k_frag", &kf, -10.f, -4.f)) { kmcP.kfrag = pow(10, kf); d = true; }
                float dg = (float)kmcP.dGscale;
                if (ImGui::SliderFloat("dG scale", &dg, 0.f, 15.f)) { kmcP.dGscale = dg; d = true; }
                float tm = (float)kmcP.tMaxMin;
                if (ImGui::SliderFloat("span (min)", &tm, 30.f, 600.f)) { kmcP.tMaxMin = tm; d = true; }
                ImGui::TreePop();
            }
            if (ImGui::Button("re-roll")) { kmcSeed++; d = true; }
            ImGui::SameLine();
            if (ImGui::Button("pin as reference")) { kmcRef = kmcR; kmcHaveRef = true; }
            if (d) kmcDirty = true;

            if (kmcDirty) {
                const Basis2D& bAct = (sim.p.atelo && haveAtelo) ? basisA : basis;
                const Profiles2D& pAct = (sim.p.atelo && haveAtelo) ? prof2A : prof2;
                float ehyT = sim.p.epsHy * std::max(0.05f, 1.f + 0.013f * (sim.p.tempC - 37.f));
                KmcProfile kprof, kprofRef;
                const Corr2D* ct = haveCorr ? &corrTab : nullptr;
                if (sim.p.registryMode == 1) {
                    kprof = kmcProfileFunnel(prof, sim.p.epsEl, ehyT,
                                             tmpl.dPeriodNm, tmpl.lengthNm);
                    kprofRef = kmcProfileFunnel(prof, sim.p.epsEl, sim.p.epsHy,
                                                tmpl.dPeriodNm, tmpl.lengthNm);
                } else {
                    kprof = kmcProfile1D(bAct, pAct.par, sim.p.pH, sim.p.epsEl,
                                         ehyT, ct, sim.p.epsCorr);
                    kprofRef = kmcProfile1D(basis, prof2.par, 7.4f, sim.p.epsEl,
                                            sim.p.epsHy, ct, sim.p.epsCorr);
                }
                kmcR = kmcRun(kmcP, kprof, kprofRef, kmcConc, sim.mwKda, sim.p.tempC,
                              (uint32_t)kmcSeed);
                kmcDirty = false;
            }
            ImGui::Separator();
            if (kmcHaveRef && !kmcRef.fibrilMass.empty())
                ImGui::PlotLines("##ref", kmcRef.fibrilMass.data(), (int)kmcRef.fibrilMass.size(),
                                 0, "reference (pinned)", 0.f, 1.f, ImVec2(-1, 60));
            if (!kmcR.fibrilMass.empty()) {
                ImGui::PlotLines("##fm", kmcR.fibrilMass.data(), (int)kmcR.fibrilMass.size(),
                                 0, "fibril mass fraction (turbidity)", 0.f, 1.f, ImVec2(-1, 160));
                ImGui::PlotLines("##fn", kmcR.nFib.data(), (int)kmcR.nFib.size(),
                                 0, "fibril count", FLT_MAX, FLT_MAX, ImVec2(-1, 70));
                ImGui::PlotLines("##bo", kmcR.bandOrder.data(), (int)kmcR.bandOrder.size(),
                                 0, "D-banding order (phase coherence)", 0.f, 1.f,
                                 ImVec2(-1, 90));
                if (!kmcR.stagHist.empty())
                    ImGui::PlotHistogram("##sh", kmcR.stagHist.data(),
                                         (int)kmcR.stagHist.size(), 0,
                                         "final stagger distribution (mod D)",
                                         FLT_MAX, FLT_MAX, ImVec2(-1, 80));
                ImGui::Text("lag %.1f min | t50 %.1f min | plateau %.0f%% | band %.2f",
                            kmcR.lagMin, kmcR.t50Min, kmcR.plateau * 100.0,
                            kmcR.finalOrder);
                ImGui::Text("koff x%.2g | nuc x%.2g | %d events | span %.0f min",
                            kmcR.fOff, kmcR.fNuc, kmcR.events, kmcP.tMaxMin);
            }
            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(win);
            glFinish();
            double remainK = 1.0 / targetFps - (glfwGetTime() - frameStart);
            if (args.frames <= 0 && remainK > 0.002)
                std::this_thread::sleep_for(std::chrono::duration<double>(remainK));
            frameIdx++;
            if (args.frames > 0 && frameIdx >= args.frames) break;
            continue;
        }
        ImGui::Text("%.0f fps | %.0f steps/s | t = %.3f us", fps, spsAvg, sim.simTimeNs * 1e-3);
        ImGui::Text("molecules %d | atoms/mol %u | near-LOD %d", sim.p.nMol, tmpl.natoms, ren.nearCount);
        if (frameIdx % 30 == 0) {
            sim.readStats();
            // physicality guard: clamped moves mean dt outruns the forces
            if (autoDt && running) {
                uint32_t ce = sim.stats[4];
                if (ce > (uint32_t)(sim.nBead() / 100))
                    sim.p.dt = std::max(0.004f, sim.p.dt * 0.93f);
                else if (ce == 0 && sim.p.dt < 0.03f)
                    sim.p.dt *= 1.02f;
            }
        }
        ImGui::Text("contacts %u | D-staggered %u | hard overlaps %u%s",
                    sim.stats[1] / 2, sim.stats[2] / 2, sim.stats[0] / 2,
                    sim.stats[3] ? "  [nbr overflow!]" : "");
        ImGui::Text("clamped moves/16 steps: %u %s", sim.stats[4],
                    sim.stats[4] ? "(dt adapting)" : "(all moves physical)");
        ImGui::Checkbox("auto-pace", &autoPace);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120);
        ImGui::SliderFloat("target fps", &targetFps, 15.f, 60.f, "%.0f");
        ImGui::SliderFloat("display smoothing", &smoothAlpha, 0.1f, 1.f);
        if (ImGui::Button(running ? "pause" : "run")) running = !running;
        ImGui::SameLine();
        if (ImGui::Button("restart")) sim.restart();
        ImGui::SameLine();
        ImGui::TextDisabled("presets:");
        ImGui::SameLine();
        if (ImGui::SmallButton("dilute")) { applyPreset(sim.p, 0); sim.restart(); }
        ImGui::SameLine();
        if (ImGui::SmallButton("quench")) { applyPreset(sim.p, 1); sim.restart(); }
        ImGui::SameLine();
        if (ImGui::SmallButton("dense")) { applyPreset(sim.p, 2); sim.restart(); }
        ImGui::SameLine();
        if (ImGui::SmallButton("fibril")) { sim.p.scenario = 4; sim.restart(); }
        ImGui::SetItemTooltip("preassembled Hodge-Petruska fibril (radius/length below)");
        ImGui::SliderFloat("fibril radius (nm)", &sim.p.fibrilRad, 3.f, 12.f);
        ImGui::SliderFloat("fibril length (um)", &sim.p.fibrilLenUm, 0.5f, 4.f);
        bool xk = sim.p.xlink == 1;
        if (ImGui::Checkbox("crosslinks (mature fibril)", &xk)) sim.p.xlink = xk ? 1 : 0;
        ImGui::SameLine();
        ImGui::TextDisabled("%d tethers", sim.nXlinks);
        ImGui::SetItemTooltip("lysyl-oxidase analog: terminal beads tether to the\n"
                              "nearest neighbor molecule at fibril initialization");
        ImGui::SliderInt("steps/frame", &stepsPerFrame, 10, 1500);
        ImGui::SliderFloat("dt (ns)", &sim.p.dt, 0.002f, 0.08f, "%.3f");
        ImGui::SameLine();
        ImGui::Checkbox("auto dt", &autoDt);
        ImGui::SliderFloat("kinetic speed-up x", &sim.p.speed, 1.f, 50.f, "%.0f");

        if (ImGui::CollapsingHeader("environment", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("temperature (C)", &sim.p.tempC, 0.f, 60.f, "%.1f");
            if (ImGui::SliderFloat("pH", &sim.p.pH, 2.5f, 11.f, "%.2f"))
                sim.recombinePH();
            ImGui::Text("net charge/molecule %+.0f e | MW %.0f kDa",
                        sim.netCharge(), sim.mwKda);
            // concentration from N and box volume (mg/mL and % w/v)
            // mass per molecule: mwKda*1000 Da * 1.66054e-21 mg/Da; V nm3 * 1e-21 mL
            double Vnm3 = 8.0 * sim.p.boxHalf.x * sim.p.boxHalf.y * sim.p.boxHalf.z;
            double mgPerMol = sim.mwKda * 1000.0 * 1.66054e-21;
            float conc = float(sim.p.nMol * mgPerMol / (Vnm3 * 1e-21));
            float phi = float(sim.p.nMol * (3.14159 * 0.75 * 0.75 * sim.molLen) / Vnm3);
            if (sim.p.scenario == 4) {
                ImGui::TextDisabled("concentration set by fibril geometry (%.2f mg/mL, phi %.1f%%)",
                                    conc, phi * 100.f);
            } else {
                static float uiConc = -1.f;
                if (uiConc < 0.f) uiConc = conc;
                ImGui::SliderFloat("conc (mg/mL)", &uiConc, 0.02f, 12.f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic);
                int want = std::min(4000, std::max(2,
                    (int)std::lround(uiConc * Vnm3 * 1e-21 / mgPerMol)));
                ImGui::Text("now: %.2f mg/mL (%.4f%% w/v) N=%d phi %.1f%%",
                            conc, conc * 0.1f, sim.p.nMol, phi * 100.f);
                if (want > sim.p.nMol) {
                    char lbl[64];
                    snprintf(lbl, sizeof lbl, "insert +%d now (glow)", want - sim.p.nMol);
                    if (ImGui::Button(lbl))
                        sim.insertMolecules(want - sim.p.nMol, (float)glfwGetTime());
                    ImGui::SameLine();
                }
                if (want != sim.p.nMol && ImGui::Button("apply at restart")) {
                    sim.p.nMol = want;
                    sim.restart();
                }
            }
            if (sim.hasAtelo) {
                int at = sim.p.atelo;
                const char* forms[] = {"telocollagen (intact)", "atelocollagen (pepsin-treated)"};
                if (ImGui::Combo("collagen form", &at, forms, 2)) {
                    sim.p.atelo = at;
                    sim.recombinePH();
                }
            }
        }
        if (ImGui::CollapsingHeader("physics", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("persistence (nm)", &sim.p.persistLen, 20.f, 200.f);
            ImGui::SliderFloat("eps electrostatic", &sim.p.epsEl, 0.f, 20.f);
            ImGui::SliderFloat("eps hydrophobic", &sim.p.epsHy, 0.f, 20.f);
            ImGui::SliderFloat("eps nonspecific", &sim.p.epsNs, 0.f, 4.f);
            if (sim.hasCorr) {
                ImGui::SliderFloat("measured correction", &sim.p.epsCorr, 0.f, 2.f);
                ImGui::SetItemTooltip("Delta-learned wells from the all-atom PMF\n"
                                      "campaign (1x = as measured, shrunk+capped)");
            } else {
                ImGui::TextDisabled("no measured corrections loaded");
            }
            ImGui::SliderFloat("alignment gate exp", &sim.p.wExp, 0.5f, 2.f);
            ImGui::SliderFloat("eps depletion (crowding)", &sim.p.epsDep, 0.f, 4.f);
            ImGui::SliderFloat("depletion range (nm)", &sim.p.depR, 1.f, 8.f);
            ImGui::SliderFloat("antiparallel mix", &sim.p.apMix, 0.f, 1.f);
            const char* regModes[] = {"off (nonspecific)", "1D funnel (idealized)",
                                      "2D azimuthal (sequence-raw)"};
            ImGui::Combo("registry model", &sim.p.registryMode, regModes, 3);
            ImGui::SliderFloat("gamma (drag)", &sim.p.gamma, 1.f, 30.f);
            ImGui::SliderFloat("gamma rot", &sim.p.gammaRot, 50.f, 2000.f);
        }
        if (ImGui::CollapsingHeader("system (restart applies)")) {
            ImGui::SliderInt("N molecules", &sim.p.nMol, 20, 2000);
            ImGui::SliderFloat("box x (nm)", &sim.p.boxHalf.x, 100.f, 600.f);
            ImGui::SliderFloat("box y (nm)", &sim.p.boxHalf.y, 100.f, 600.f);
            ImGui::SliderFloat("box z (nm)", &sim.p.boxHalf.z, 100.f, 600.f);
            bool pb = sim.p.pbc == 1;
            if (ImGui::Checkbox("periodic boundaries", &pb)) sim.p.pbc = pb ? 1 : 0;
            ImGui::SetItemTooltip("minimum-image forces, wrapped grid; with the fibril\n"
                                  "scenario the box snaps to the 5D lattice period\n"
                                  "-> an effectively infinite fibril");
            int s = (int)sim.p.seed;
            if (ImGui::InputInt("seed", &s)) sim.p.seed = (uint32_t)std::max(1, s);
        }
        if (ImGui::CollapsingHeader("render", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* modes[] = {"element (CPK)", "residue class", "chain", "molecule", "D-phase"};
            ImGui::Combo("color", &ren.opts.colorMode, modes, 5);
            const char* reps[] = {"off", "z axis", "full 27"};
            ImGui::Combo("periodic repeats", &ren.opts.repeats, reps, 3);
            ImGui::SliderFloat("atom radius x", &ren.opts.radScale, 0.5f, 2.f);
            ImGui::SliderFloat("tube radius (nm)", &ren.opts.tubeR, 0.3f, 1.5f);
            ImGui::SliderFloat("atom LOD dist (nm)", &ren.opts.lodDist, 0.f, 600.f);
            ImGui::SliderFloat("fog", &ren.opts.fogDensity, 0.f, 0.004f, "%.4f");
        }
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
        // hard-bound the GPU queue: never let compute batches pile up behind
        // the compositor (this is what kept the desktop responsive-less before)
        glFinish();

        double ft = glfwGetTime() - frameStart;
        double target = 1.0 / targetFps;
        if (autoPace && running) {
            if (ft > target * 1.05)
                stepsPerFrame = std::max(5, int(stepsPerFrame * 0.8));
            else if (ft < target * 0.7)
                stepsPerFrame = std::min(2000, int(stepsPerFrame * 1.12) + 1);
        }
        if (args.frames <= 0) {           // interactive: cap the frame rate
            double remain = target - (glfwGetTime() - frameStart);
            if (remain > 0.002)
                std::this_thread::sleep_for(std::chrono::duration<double>(remain));
        }
        frameIdx++;
        if (args.frames > 0 && frameIdx >= args.frames) break;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
