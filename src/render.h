#pragma once
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <vector>
#include "io.h"
#include "shader.h"

class Sim;

struct RenderOpts {
    int colorMode = 0;          // 0 element 1 class 2 chain 3 molecule 4 D-phase
    int repeats = 1;            // pbc images: 0 off, 1 z-axis (3), 2 full (27)
    float radScale = 1.0f;
    float tubeR = 0.75f;        // nm
    float lodDist = 150.f;      // molecules closer than this get atoms
    int atomBudget = 6'000'000; // max sphere instances
    glm::vec3 fogColor = {0.043f, 0.055f, 0.078f};
    float fogDensity = 0.0009f;
};

class Renderer {
public:
    void init(const AtomTemplate& tmpl);
    // refresh near/far LOD lists from current GPU positions (call sparingly)
    void updateLod(Sim& sim, const glm::vec3& camPos);
    void draw(Sim& sim, const glm::mat4& view, const glm::mat4& proj, float nowSec);

    RenderOpts opts;
    int nearCount = 0;

private:
    Program atomProg, capProg;
    GLuint atomsSSBO = 0, nearIds = 0, nearMask = 0, nearOffs = 0, dummyVao = 0;
    int natoms = 0;
    float dPeriod = 0;
    std::vector<glm::vec3> molCenters;
    std::vector<float> scratch;
    std::vector<uint32_t> nearList, maskList;
    std::vector<glm::vec4> offList;
};
