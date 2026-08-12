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
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "camera.h"
#include "io.h"
#include "we.h"
#include "render.h"
#include "shader.h"
#include "sim.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static OrbitCamera g_cam;
static bool g_rotating = false, g_panning = false;
static double g_mx = 0, g_my = 0;
// 0 = BD fast (Euler-Maruyama), 1 = BD rigorous (MALA), 3 = WE (--advanced only)
static int g_appMode = 0;
static bool g_advanced = false;    // reveals the weighted-ensemble tab
// weighted-ensemble test knobs (pass BEFORE --wetest; the arg loop returns on it)
static int g_weNmol = 40, g_weTarget = 3, g_weTau = 1500, g_weIters = 60;
static int g_weBrute = 24, g_weBruteSteps = 30000, g_weWalkers = 4;
static bool g_weTest = false;
static WeRun g_we;
static WeParams g_weUi;
static bool g_weActive = false;
static int g_weIdx = 0;
static bool g_gradTest = false;    // finite-difference check of U vs the drift
static bool g_dtScan = false;      // MALA acceptance vs timestep
static bool g_malaProbe = false;   // dump the Metropolis ratio's components
static bool g_equipart = false;    // equipartition test across engines
static double g_equipNs = 300.0;   // simulated ns sampled per engine
static int g_gradNmol = 12, g_gradSettle = 3000;

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
    int engine = 0;
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
        else if (s == "--advanced") g_advanced = true;
        else if (s == "--gradtest") g_gradTest = true;
        else if (s == "--dtscan") g_dtScan = true;
        else if (s == "--equipart") g_equipart = true;
        else if (s == "--equipns" && i + 1 < argc) g_equipNs = atof(argv[++i]);
        else if (s == "--malaprobe") { g_dtScan = true; g_malaProbe = true; }
        else if (s == "--engine" && i + 1 < argc) a.engine = next(0);
        else if (s == "--gradnmol" && i + 1 < argc) g_gradNmol = atoi(argv[++i]);
        else if (s == "--gradsettle" && i + 1 < argc) g_gradSettle = atoi(argv[++i]);
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
    if (args.pbc) sim.p.pbc = 1;
    if (args.engine) sim.p.engine = args.engine;
    if (args.xlink) sim.p.xlink = 1;
    if (args.pH > 0) sim.p.pH = args.pH;
    if (args.temp > -900) sim.p.tempC = args.temp;
    if (args.scenario == 4) {
        if (args.fibRad > 0) sim.p.fibrilRad = args.fibRad;
        if (args.fibLen > 0) sim.p.fibrilLenUm = args.fibLen;
    }
    sim.init(tmpl, prof2, basis, haveAtelo ? &prof2A : nullptr,
             haveAtelo ? &basisA : nullptr, haveCorr ? &corrTab : nullptr);

    // --------------------------------------------------------------------
    // --gradtest: is the rigorous engine's drift the gradient of its U?
    //
    // Differentiate U numerically along random directions and compare with
    // -F.dir. This is a DIAGNOSTIC, not a pass/fail gate: Metropolis-Hastings
    // needs an exact, evaluable U but tolerates an approximate drift, which
    // only costs acceptance rate. What the number tells us is how much
    // acceptance we are giving away, and it localises which term is off.
    // --------------------------------------------------------------------
    if (g_gradTest) {
        sim.p.engine = 1;
        sim.p.nMol = g_gradNmol;
        sim.p.boxHalf = {40.f, 40.f, 180.f};
        sim.p.pbc = 1;
        sim.p.scenario = 0;
        sim.restart();
        printf("gradtest: %d molecules, %d beads, dtRig %.4f ns, kBond %.0f kT/nm^2 "
               "(rms strain %.1f%% of a = %.3f nm)\n",
               sim.p.nMol, sim.nBead(), sim.p.dtRig, sim.kBond(),
               sim.p.bondStrain * 100.f, sim.segLen);
        sim.step(g_gradSettle);            // settle into real contacts
        sim.burnin = 0;                    // measure the unramped U only

        const int nB = sim.nBead();
        std::vector<glm::vec4> x0(nB), xp(nB);
        std::vector<glm::vec4> F(nB);
        glGetNamedBufferSubData(sim.posBuf, 0, sizeof(glm::vec4) * nB, x0.data());

        double u0 = sim.computeU(sim.posBuf);
        sim.driftAt(sim.posBuf);
        glGetNamedBufferSubData(sim.beadFBuf, 0, sizeof(glm::vec4) * nB, F.data());
        printf("  U = %.4f kT total, %.5f kT/bead | <|F|> ", u0, u0 / nB);
        {
            double fs = 0;
            for (int i = 0; i < nB; ++i) fs += glm::length(glm::vec3(F[i]));
            printf("%.4f kT/nm\n", fs / nB);
        }

        // scratch buffer so the base positions are never disturbed
        GLuint probe = 0;
        glCreateBuffers(1, &probe);
        glNamedBufferData(probe, sizeof(glm::vec4) * nB, x0.data(), GL_DYNAMIC_DRAW);

        // Two models over the SAME configuration. Turning the registry off
        // leaves only terms whose gradients are written analytically (bonds,
        // bending, soft core, walls); whatever error survives there is a bug,
        // and whatever appears only in the full model is the price of the
        // omitted d(facing angle)/dx and d(gate)/dx terms.
        const int nTrial = 48;
        struct Case { const char* name; bool registry; };
        for (Case cs : {Case{"bonded + steric only", false}, Case{"full model", true}}) {
            float sEl = sim.p.epsEl, sHy = sim.p.epsHy, sNs = sim.p.epsNs;
            float sCo = sim.p.epsCorr, sDe = sim.p.epsDep;
            if (!cs.registry) {
                sim.p.epsEl = sim.p.epsHy = sim.p.epsNs = 0.f;
                sim.p.epsCorr = sim.p.epsDep = 0.f;
            }
            sim.computeU(sim.posBuf);
            sim.driftAt(sim.posBuf);
            glGetNamedBufferSubData(sim.beadFBuf, 0, sizeof(glm::vec4) * nB, F.data());
            double meanF = 0;
            for (int i = 0; i < nB; ++i) meanF += glm::length(glm::vec3(F[i]));
            meanF /= nB;
            printf("\n  [%s]  U = %.3f kT, <|F|> = %.3f kT/nm\n",
                   cs.name, sim.lastU, meanF);
            // Errors are ABSOLUTE (kT/nm). Normalising per bead is misleading:
            // a bead near force balance has |F| ~ 0 and reports a huge relative
            // error for an utterly negligible absolute one.
            printf("  %-9s %11s %11s %11s %11s %8s\n", "eps(nm)", "FD dU/ds",
                   "-F.dir", "med |err|", "p95 |err|", "med/<|F|>");
            for (float eps : {3e-2f, 1e-2f, 3e-3f, 1e-3f}) {
                // ONE bead at a time. A direction spread over all 3N
                // coordinates displaces each bead by only eps/sqrt(3N) ~ 2 ulp
                // of a float32 coordinate at 40 nm, so the finite difference
                // measures rounding rather than physics. Moving a single bead
                // by the full eps also makes every untouched energy element
                // bitwise identical between the two evaluations, so they cancel
                // exactly in the double reduction.
                std::mt19937 rg(20260811u);          // same probes every eps
                std::normal_distribution<float> nd(0.f, 1.f);
                std::vector<double> rels;
                double fd0 = 0, an0 = 0;
                for (int t = 0; t < nTrial; ++t) {
                    int b = (int)(rg() % (unsigned)nB);
                    glm::vec3 dir(nd(rg), nd(rg), nd(rg));
                    dir = glm::normalize(dir);
                    double uPM[2] = {0, 0};
                    for (int s = 0; s < 2; ++s) {
                        glm::vec4 xb = x0[b] + glm::vec4(dir * (s == 0 ? -eps : eps), 0.f);
                        glNamedBufferSubData(probe, sizeof(glm::vec4) * b,
                                             sizeof(glm::vec4), &xb);
                        uPM[s] = sim.computeU(probe);
                    }
                    glNamedBufferSubData(probe, sizeof(glm::vec4) * b,
                                         sizeof(glm::vec4), &x0[b]);
                    double fd = (uPM[1] - uPM[0]) / (2.0 * eps);
                    double an = -glm::dot(glm::vec3(F[b]), dir);
                    rels.push_back(fabs(fd - an));       // kT/nm
                    if (t == 0) { fd0 = fd; an0 = an; }
                }
                std::sort(rels.begin(), rels.end());
                double med = rels[rels.size() / 2];
                double p95 = rels[(size_t)(rels.size() * 0.95)];
                printf("  %-9.0e %11.5f %11.5f %11.2e %11.2e %7.3f%%\n",
                       eps, fd0, an0, med, p95, 100.0 * med / meanF);
            }
            sim.p.epsEl = sEl; sim.p.epsHy = sHy; sim.p.epsNs = sNs;
            sim.p.epsCorr = sCo; sim.p.epsDep = sDe;
        }
        glDeleteBuffers(1, &probe);
        printf("\n  Reading: error that shrinks with eps is finite-difference\n"
               "  truncation. Error that plateaus is a real inconsistency\n"
               "  between U and the drift. The drift omits d(facing angle)/dx\n"
               "  and d(alignment gate)/dx, and does not differentiate the\n"
               "  parallel-transport frames at all -- that omission costs MALA\n"
               "  acceptance, not correctness, since Metropolis-Hastings needs\n"
               "  an exact U but tolerates an approximate proposal drift.\n");
        return 0;
    }

    // --------------------------------------------------------------------
    // --dtscan: the certificate. Metropolis makes the sampled distribution
    // exactly exp(-U/kT) at ANY dt, so this is not an accuracy curve -- it is
    // a KINETICS curve. Every rejection freezes the system for a step, which
    // real Brownian motion never does, so the trajectory is only interpretable
    // as dynamics while acceptance is near 1. This finds where that stops.
    // --------------------------------------------------------------------
    if (g_dtScan) {
        sim.p.engine = 1;
        sim.p.mala = 1;
        sim.p.nMol = g_gradNmol;
        sim.p.boxHalf = {40.f, 40.f, 180.f};
        sim.p.pbc = 1;
        sim.p.scenario = args.scenario ? args.scenario : 0;
        sim.restart();
        printf("dtscan: %d molecules (%d beads, %d DOF), kBond %.0f kT/nm^2, "
               "tau_bond = gamma/k = %.4f ns\n",
               sim.p.nMol, sim.nBead(), 3 * sim.nBead() + sim.p.nMol,
               sim.kBond(), sim.p.gamma / sim.kBond());
        // Equilibrate with MALA OFF. Settling with it on is circular: if the
        // chosen dt happens to reject everything, the "equilibrated" snapshot
        // is just the cold straight-rod start, and the scan then measures how
        // hard it is to thermalise rather than the acceptance at equilibrium.
        // U has to plateau before any of these numbers mean anything -- the
        // slowest local mode is bending, tau = gamma/(kBend/a^2) ~ 3.3 ns.
        sim.p.mala = 0;
        sim.p.dtRig = 0.005f;
        printf("  equilibrating (unadjusted, dt %.3f ns):", sim.p.dtRig);
        for (int w = 0; w < 6; ++w) {
            sim.step(g_gradSettle);
            sim.burnin = 0;
            printf(" %.0f", sim.computeU());
            fflush(stdout);
        }
        printf("  kT  (%.1f ns simulated)\n", 6.0 * g_gradSettle * sim.p.dtRig);
        SimState eq;
        sim.snapshot(eq);

        // Three nested models over the same configuration. Bonds alone are a
        // near-exact Gaussian target, for which MALA's acceptance is known in
        // closed form -- so that row is a machinery check, not a measurement.
        // Whatever only appears in the richer rows is a property of the
        // potential, not of the sampler.
        struct MCase { const char* name; bool bendCore; bool registry; };
        float sEl = sim.p.epsEl, sHy = sim.p.epsHy, sNs = sim.p.epsNs;
        float sCo = sim.p.epsCorr, sDe = sim.p.epsDep;
        float sRep = sim.p.kRep, sLp = sim.p.persistLen, sWall = sim.p.kWall;
        int nBonds = sim.p.nMol * (sim.p.nBeads - 1);
        for (MCase mc : {MCase{"bonds only (analytic reference)", false, false},
                         MCase{"+ bending + soft core", true, false},
                         MCase{"+ registry tables (full)", true, true}}) {
            sim.restore(eq);
            sim.p.epsEl = mc.registry ? sEl : 0.f;
            sim.p.epsHy = mc.registry ? sHy : 0.f;
            sim.p.epsNs = mc.registry ? sNs : 0.f;
            sim.p.epsCorr = mc.registry ? sCo : 0.f;
            sim.p.epsDep = mc.registry ? sDe : 0.f;
            sim.p.kRep = mc.bendCore ? sRep : 0.f;
            sim.p.persistLen = mc.bendCore ? sLp : 1e-6f;
            sim.p.kWall = mc.bendCore ? sWall : 0.f;
            sim.p.mala = 0;
            sim.p.dtRig = 0.005f;
            sim.step(g_gradSettle);          // re-equilibrate under THIS U
            sim.burnin = 0;
            sim.p.mala = 1;
            SimState eqc;
            sim.snapshot(eqc);
            printf("\n  [%s]  U = %.1f kT\n", mc.name, sim.computeU());
            printf("  %-10s %-8s %10s %10s %13s %s\n", "dtRig(ns)", "dt/tau",
                   "accept", "predicted", "ns per wall-s", "verdict");
        for (float dt : {0.0005f, 0.001f, 0.002f, 0.005f, 0.01f, 0.02f, 0.05f}) {
            sim.restore(eqc);
            sim.p.dtRig = dt;
            sim.malaTries = sim.malaAccepted = 0;
            glClearNamedBufferData(sim.totalBuf, GL_R32UI, GL_RED_INTEGER,
                                   GL_UNSIGNED_INT, nullptr);
            int nst = 400;
            double t0 = glfwGetTime();
            sim.step(nst);
            double el = glfwGetTime() - t0;
            double acc = sim.malaAccept;
            const char* verdict = acc >= 0.99 ? "dynamics OK"
                                : acc >= 0.90 ? "sampling OK, kinetics suspect"
                                : acc >= 0.50 ? "sampler only"
                                              : "frozen";
            // Gaussian-target prediction. Per stiff mode the MALA log-ratio has
            // variance 0.5*a_m^3 with a_m = k_m*dt/gamma. The modes of a
            // bead-spring CHAIN are k_m = 2k(1 - cos q), q in [0,pi] -- so the
            // stiffest is 4k, not k, and <(2-2cos q)^3> = 20 over the band.
            // Using the bare spring constant here (as the naive estimate does)
            // understates the variance by 20x and overstates acceptance badly.
            double a = dt / (sim.p.gamma / sim.kBond());
            double sg = sqrt(10.0 * nBonds) * pow(a, 1.5);
            double pred = erfc(sg / (2.0 * 1.41421356));
            printf("  %-10.4f %-8.3f %9.3f %10.3f %13.1f  %s\n",
                   dt, a, acc, pred, nst * dt / el, verdict);
            if (g_malaProbe) {
                double t[8];
                glGetNamedBufferSubData(sim.totalBuf, 0, 64, t);
                double D = 4.0 * sim.p.kT * dt / sim.p.gamma;
                double Dr = 4.0 * sim.p.kT * dt / sim.p.gammaRot;
                printf("       U %.4f -> %.4f  (dU %+.4f)\n", t[0], t[3], t[3] - t[0]);
                printf("       pos: fwd %.6f rev %.6f  d/(4Ddt) %+.4f\n",
                       t[1], t[4], (t[1] - t[4]) / D);
                printf("       th : fwd %.3e rev %.3e  d/(4Ddt) %+.4f\n",
                       t[2], t[5], (t[2] - t[5]) / Dr);
                printf("       log a = %+.4f\n",
                       -(t[3] - t[0]) / sim.p.kT + (t[1] - t[4]) / D + (t[2] - t[5]) / Dr);
            }
        }
        }
        sim.p.epsEl = sEl; sim.p.epsHy = sHy; sim.p.epsNs = sNs;
        sim.p.epsCorr = sCo; sim.p.epsDep = sDe;
        sim.p.kRep = sRep; sim.p.persistLen = sLp; sim.p.kWall = sWall;
        printf("\n  Acceptance is the falsifiable answer to \"is my timestep\n"
               "  honest?\" -- a question the fast engine cannot answer at all.\n"
               "  Row 1 should track its prediction; if it does, the sampler is\n"
               "  sound and any shortfall below is the potential's own doing.\n");
        return 0;
    }

    // --------------------------------------------------------------------
    // --equipart: does each engine sample the distribution it claims to?
    //
    // With pair interactions off, U = bonds + bending only, and in bond-vector
    // coordinates (l_i, direction_i) the Jacobian is prod l_i^2, so both terms
    // separate and have closed-form equilibria:
    //
    //   p(l)      ~ l^2 exp(-k(l-a)^2 / 2kT)   =>  <l> = a + 2 sigma^2 / a
    //   p(cos t)  ~ exp(-kappa (1 - cos t))    =>  <cos t> = coth k - 1/k
    //
    // The second is the Langevin function and depends on NOTHING but kappa =
    // kBend/kT. Any engine that samples exp(-U/kT) must reproduce it; an engine
    // that clamps, projects or truncates need not, and cannot say by how much.
    // --------------------------------------------------------------------
    if (g_equipart) {
        // The decisive comparison is the dt pair at the bottom. MALA samples
        // exp(-U/kT) at ANY dt, so its answer MUST be dt-independent; the
        // unadjusted propagator's must not be. If MALA drifts with dt it is not
        // correcting; if it holds steady but misses the analytic value, then
        // the analytic value is what is wrong.
        struct ECase { const char* name; int engine; int mala; float dt; };
        const int nm = g_gradNmol > 0 ? g_gradNmol : 8;
        double simNs = g_equipNs;
        printf("equipart: %d molecules, pair interactions OFF, %.0f ns sampled "
               "per engine\n", nm, simNs);
        // Shared starting state. restart() lays molecules down as straight
        // rods, and bending relaxes slowly, so every engine would otherwise
        // spend its whole budget crawling away from cos t = 1 and the answer
        // would be an equilibration artefact. Tab 1 covers 60x more simulated
        // time per step, so it does the pre-relaxation; each engine then only
        // has to travel the short distance from tab 1's equilibrium to its own.
        sim.p.nMol = nm;
        sim.p.boxHalf = {60.f, 60.f, 200.f};
        sim.p.pbc = 1;
        sim.p.scenario = 0;
        sim.p.epsEl = sim.p.epsHy = sim.p.epsNs = 0.f;
        sim.p.epsCorr = sim.p.epsDep = 0.f;
        sim.p.kRep = 0.f;
        sim.p.engine = 0;
        sim.restart();
        sim.burnin = 0;
        sim.step((int)(20000.0 / (sim.p.dt * sim.p.speed)));   // 20 us, cheap
        SimState start;
        sim.snapshot(start);

        bool first = true;
        float dt0 = sim.p.dtRig;
        // MALA must be run where it actually MOVES. Its nominal simulated time
        // is not a comparable budget: a rejected step advances the clock while
        // the system stays put, so a low-acceptance run reports thousands of
        // nanoseconds having explored almost nothing. The acceptance column
        // below says whether a row is worth reading.
        for (ECase ec : {ECase{"tab 1  fast (EM + XPBD + clamps)", 0, 0, 0.f},
                         ECase{"tab 2  unadjusted   dt 1e-3", 1, 0, 1e-3f},
                         ECase{"tab 2  + MALA       dt 1e-3", 1, 1, 1e-3f},
                         ECase{"tab 3  constrained  dt 2e-2", 2, 0, 0.f}}) {
            sim.restore(start);
            sim.p.engine = ec.engine;
            sim.p.mala = ec.mala;
            if (ec.dt > 0) sim.p.dtRig = ec.dt; else sim.p.dtRig = dt0;

            double dtEff = ec.engine == 2 ? sim.p.dtCon
                         : ec.engine == 1 ? sim.p.dtRig
                                          : sim.p.dt * sim.p.speed;
            int totSteps = (int)(simNs / dtEff);
            int nSnap = 100;
            int per = std::max(1, totSteps / nSnap);
            sim.step(totSteps / 4);                          // settle into THIS engine

            if (first) {
                double kb = sim.kBond(), a = sim.segLen;
                double sg = sqrt(1.0 / kb);
                double kap = sim.p.persistLen / sim.segLen;
                printf("  kBond %.0f kT/nm^2, a %.4f nm, kappa = Lp/a = %.3f\n", kb, a, kap);
                printf("  EXACT: <l> %.5f  sd(l) %.5f  <cos t> %.5f  -> Lp %.2f nm\n\n",
                       a + 2 * sg * sg / a, sg,
                       1.0 / tanh(kap) - 1.0 / kap,
                       -a / log(1.0 / tanh(kap) - 1.0 / kap));
                printf("  %-34s %9s %9s %10s %8s %7s %7s\n", "engine",
                       "<l>", "sd(l)", "<cos t>", "Lp(nm)", "clamp", "accept");
                first = false;
            }

            std::vector<float> buf(size_t(sim.nBead()) * 4);
            double sl = 0, sl2 = 0, sc = 0;
            long long nl = 0, nc = 0;
            // running <cos t> per quarter: if it is still drifting the number
            // below is an equilibration artefact, not a property of the engine
            double qc[4] = {0, 0, 0, 0};
            long long qn[4] = {0, 0, 0, 0};
            for (int s = 0; s < nSnap; ++s) {
                int q = std::min(3, s * 4 / nSnap);
                double scPrev = sc;
                long long ncPrev = nc;
                sim.step(per);
                glGetNamedBufferSubData(sim.posBuf, 0, sizeof(float) * buf.size(),
                                        buf.data());
                int M = sim.p.nBeads;
                for (int m = 0; m < sim.p.nMol; ++m) {
                    for (int i = 0; i + 1 < M; ++i) {
                        const float* p0 = &buf[(size_t(m) * M + i) * 4];
                        const float* p1 = &buf[(size_t(m) * M + i + 1) * 4];
                        glm::vec3 d(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
                        double L = glm::length(d);
                        sl += L; sl2 += L * L; nl++;
                        if (i + 2 < M) {
                            const float* p2 = &buf[(size_t(m) * M + i + 2) * 4];
                            glm::vec3 e(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
                            double le = glm::length(e);
                            if (L > 1e-9 && le > 1e-9) {
                                sc += glm::dot(d, e) / (L * le);
                                nc++;
                            }
                        }
                    }
                }
                qc[q] += sc - scPrev;
                qn[q] += nc - ncPrev;
            }
            double ml = sl / nl, vl = sl2 / nl - ml * ml;
            double mc = sc / nc;
            double lp = mc > 0 && mc < 1 ? -sim.segLen / log(mc) : 0.0;
            // Clamp telemetry. The force pass clears stats every 16 steps, so a
            // single read is one 16-step window and swings wildly (13-74% in
            // practice). Average over many windows or the number is noise.
            char clamp[16] = "   -";
            if (ec.engine == 0) {
                // MUST align to the clear boundary first. The force pass zeroes
                // stats when totalSteps % 16 == 0; starting off-phase counts
                // fewer than 16 steps of accumulation while still dividing by
                // 16, and since stepping 16 preserves the phase, averaging does
                // not wash it out -- it just repeats the same wrong window.
                while (sim.totalSteps % 16 != 0) sim.step(1);
                double acc2 = 0;
                const int nWin = 40;
                for (int w = 0; w < nWin; ++w) {
                    sim.step(16);
                    sim.readStats();
                    acc2 += sim.stats[4] / (16.0 * sim.nBead());
                }
                snprintf(clamp, sizeof clamp, "%5.1f%%", 100.0 * acc2 / nWin);
            }
            char accs[16] = "   -";
            if (ec.mala) snprintf(accs, sizeof accs, "%6.3f", sim.malaAccept);
            printf("  %-34s %9.5f %9.5f %10.5f %8.2f %7s %7s\n",
                   ec.name, ml, sqrt(std::max(0.0, vl)), mc, lp, clamp, accs);
            printf("  %-34s <cos t> by quarter: %.5f %.5f %.5f %.5f\n", "",
                   qn[0] ? qc[0] / qn[0] : 0, qn[1] ? qc[1] / qn[1] : 0,
                   qn[2] ? qc[2] / qn[2] : 0, qn[3] ? qc[3] / qn[3] : 0);
        }
        printf("\n  sd(l) ~ 0 for tab 1 is XPBD holding the bond rigid -- a\n"
               "  different model, not an error. <cos t> is the shared test:\n"
               "  both engines have the same kBend and must give the same answer.\n");
        return 0;
    }

    if (g_weTest) {
            WeParams wp;
            wp.targetSize = g_weTarget;
            wp.tauSteps = g_weTau;
            wp.walkersPerBin = g_weWalkers;
            sim.p.nMol = g_weNmol;
            sim.p.boxHalf = {60.f, 60.f, 200.f};
            sim.p.pbc = 1;
            sim.p.scenario = 0;
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
                printf("| ov %u", sim.stats[0] / 2);
                if (sim.p.engine == 1) {
                    // evaluate first: argument order is unspecified, so
                    // passing computeU() and lastU together reads a stale U
                    double u = sim.computeU();
                    printf(" | U %.1f kT (%.4f/bead)", u, u / sim.nBead());
                    if (sim.p.mala) printf(" | acc %.3f", sim.malaAccept);
                }
                printf("\n");
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

        // both BD tabs drive the same Sim; the tab only selects the propagator
        if (running && (g_appMode == 0 || g_appMode == 1)) sim.step(stepsPerFrame);
        if (running && g_appMode == 3 && g_weActive && !g_we.walkers.empty()) {
            g_we.advanceWalker(sim, g_weIdx);
            if (++g_weIdx >= (int)g_we.walkers.size()) {
                g_we.finishIteration(sim);
                g_weIdx = 0;
            }
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

        int& appMode = g_appMode;
        if (ImGui::BeginTabBar("mode")) {
            if (ImGui::BeginTabItem("BD - fast")) { appMode = 0; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("BD - rigorous")) { appMode = 1; ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("BD - constrained")) { appMode = 2; ImGui::EndTabItem(); }
            if (g_advanced && ImGui::BeginTabItem("WE (rare events)")) {
                appMode = 3;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
        // the tab IS the engine: same Sim, same state, different propagator
        sim.p.engine = (appMode == 1) ? 1 : (appMode == 2) ? 2 : 0;
        if (appMode == 2) {
            ImGui::TextWrapped("Bonds are rigid holonomic constraints (SHAKE), not "
                               "springs. That removes the stiffest mode from the "
                               "spectrum, so dt can be ~10x larger than the rigorous "
                               "tab -- and it is the more faithful limit: collagen's "
                               "real axial stiffness is ~675 kT/nm^2, stiffer than any "
                               "spring we would dare integrate.");
            ImGui::TextColored({1.f, 0.65f, 0.2f, 1.f},
                               "NOT EXACT YET: no Fixman term and no Metropolis "
                               "adjustment. Constraining changes the equilibrium "
                               "measure; --equipart measures by how much.");
            ImGui::Separator();
            ImGui::SliderFloat("dt constrained (ns)", &sim.p.dtCon, 0.0005f, 0.2f,
                               "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderInt("SHAKE sweeps", &sim.p.shakeIters, 2, 128);
            ImGui::TextDisabled("  tab 1 uses 4 and does not converge: its 0.052 nm\n"
                                "  bond spread is residual projection error, not physics.");
            if (ImGui::Button("measure U")) sim.computeU();
            ImGui::SameLine();
            ImGui::Text("U = %.1f kT", sim.lastU);
            ImGui::Separator();
        }
        if (appMode == 1) {
            ImGui::TextWrapped("Rigorous engine: explicit potential energy U, harmonic "
                               "bonds, no force or displacement clamps, and a Metropolis "
                               "accept/reject on every step, so the sampled distribution "
                               "is exactly exp(-U/kT) rather than whatever the clamps "
                               "leave behind. Costs roughly 10-100x per nanosecond.");
            ImGui::Checkbox("Metropolis-adjusted (MALA)", (bool*)&sim.p.mala);
            if (sim.p.mala) {
                double a = sim.malaAccept;
                ImVec4 col = a >= 0.99 ? ImVec4(0.4f, 0.9f, 0.5f, 1.f)
                           : a >= 0.90 ? ImVec4(0.9f, 0.9f, 0.4f, 1.f)
                                       : ImVec4(1.f, 0.5f, 0.35f, 1.f);
                ImGui::TextColored(col, "acceptance %.3f  --  %s", a,
                    a >= 0.99 ? "dt is honest: rejections rare, still dynamics"
                  : a >= 0.90 ? "equilibrium exact, kinetics distorted"
                  : a >= 0.50 ? "sampler only -- do not read rates off this"
                              : "frozen: lower dt");
                ImGui::TextDisabled("  Equilibrium is exactly exp(-U/kT) at ANY dt. "
                                    "Acceptance is about KINETICS: a rejection freezes\n"
                                    "  the system for a step, which Brownian motion "
                                    "never does. Falls as N^(-1/3).");
            } else {
                ImGui::TextColored({1.f, 0.65f, 0.2f, 1.f},
                                   "Unadjusted: conservative force field, but the "
                                   "sampled equilibrium is dt-dependent.");
            }
            ImGui::Separator();
            ImGui::SliderFloat("dt rigorous (ns)", &sim.p.dtRig, 0.00001f, 0.05f,
                               "%.5f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("bond strain (rms)", &sim.p.bondStrain, 0.005f, 0.1f, "%.3f");
            ImGui::TextDisabled("  -> kBond %.0f kT/nm^2 (a = %.3f nm). Stability needs "
                                "dt < 2*gamma/kBond = %.4f ns",
                                sim.kBond(), sim.segLen, 2.f * sim.p.gamma / sim.kBond());
            ImGui::Checkbox("Leimkuhler-Matthews noise", (bool*)&sim.p.lmNoise);
            if (ImGui::Button("measure U")) sim.computeU();
            ImGui::SameLine();
            ImGui::Text("U = %.1f kT  (%.4f kT/bead)", sim.lastU,
                        sim.lastU / std::max(1, sim.nBead()));
            ImGui::Separator();
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
            ImGui::SliderInt("tau (BD steps)", &g_weUi.tauSteps, 50, 5000);
            ImGui::TextDisabled("  = %.0f ns. Flux is only counted at iteration "
                                "boundaries, so a crossing that reverts within\n"
                                "  one tau is missed. Measured: halving tau did NOT "
                                "close the ~2x deficit vs brute force.",
                                g_weUi.tauSteps * sim.p.dt * sim.p.speed);
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
                        ImGui::TextDisabled("  +/- %.0f%% nominal (Poisson on %d "
                                            "crossings -- OPTIMISTIC: crossings carry\n"
                                            "  unequal weights, and observed run-to-run "
                                            "scatter is far wider than this)",
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
            ImGui::TextDisabled("registry: measured 2D azimuthal + Delta-correction");
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
