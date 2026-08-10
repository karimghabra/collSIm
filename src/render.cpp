#include "render.h"
#include "sim.h"
#include <algorithm>
#include <cstring>
#include <numeric>

void Renderer::init(const AtomTemplate& tmpl) {
    natoms = tmpl.natoms;
    dPeriod = tmpl.dPeriodNm;

    // pack template: vec4(s, x, y, bits(elem | cls<<4 | chain<<8 | res<<16))
    std::vector<glm::vec4> packed(natoms);
    for (int i = 0; i < natoms; ++i) {
        const AtomRec& a = tmpl.atoms[i];
        uint32_t pk = uint32_t(a.elem) | (uint32_t(a.cls) << 4) |
                      (uint32_t(a.chain) << 8) | (uint32_t(a.res) << 16);
        float pf;
        memcpy(&pf, &pk, 4);
        packed[i] = glm::vec4(a.s, a.x, a.y, pf);
    }
    glCreateBuffers(1, &atomsSSBO);
    glNamedBufferData(atomsSSBO, sizeof(glm::vec4) * natoms, packed.data(), GL_STATIC_DRAW);

    atomProg.graphicsFromFiles("shaders/atoms.vert", "shaders/atoms.frag");
    capProg.graphicsFromFiles("shaders/caps.vert", "shaders/caps.frag");
    glCreateVertexArrays(1, &dummyVao);
}

void Renderer::updateLod(Sim& sim, const glm::vec3& camPos) {
    int N = sim.p.nMol;
    molCenters.resize(N);
    std::vector<glm::vec4> cs(N);
    glGetNamedBufferSubData(sim.centersBuf, 0, sizeof(glm::vec4) * N, cs.data());
    for (int m = 0; m < N; ++m) molCenters[m] = glm::vec3(cs[m]);
    // under pbc, judge each molecule by its camera-nearest periodic image
    glm::vec3 boxSize = 2.f * sim.p.boxHalf;
    bool usePbc = sim.p.pbc == 1 && opts.repeats > 0;
    std::vector<glm::vec3> offs(N, glm::vec3(0));
    if (usePbc) {
        for (int m = 0; m < N; ++m) {
            glm::vec3 o = glm::round((camPos - molCenters[m]) / boxSize);
            o = glm::clamp(o, glm::vec3(-1), glm::vec3(1));
            if (opts.repeats == 1) o.x = o.y = 0.f;
            offs[m] = o * boxSize;
        }
    }
    std::vector<std::pair<float, uint32_t>> order(N);
    for (int m = 0; m < N; ++m)
        order[m] = {glm::distance(molCenters[m] + offs[m], camPos), uint32_t(m)};
    std::sort(order.begin(), order.end());

    nearList.clear();
    offList.clear();
    maskList.assign(N, 0u);
    int maxNear = std::max(0, opts.atomBudget / std::max(natoms, 1));
    for (auto& [d, m] : order) {
        if (d > opts.lodDist || (int)nearList.size() >= maxNear) break;
        nearList.push_back(m);
        offList.push_back(glm::vec4(offs[m], 0.f));
        maskList[m] = 1u;
    }
    nearCount = (int)nearList.size();

    if (!nearIds) glCreateBuffers(1, &nearIds);
    if (!nearMask) glCreateBuffers(1, &nearMask);
    if (!nearOffs) glCreateBuffers(1, &nearOffs);
    glNamedBufferData(nearIds, sizeof(uint32_t) * std::max<size_t>(1, nearList.size()),
                      nearList.empty() ? nullptr : nearList.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(nearOffs, sizeof(glm::vec4) * std::max<size_t>(1, offList.size()),
                      offList.empty() ? nullptr : offList.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(nearMask, sizeof(uint32_t) * N, maskList.data(), GL_DYNAMIC_DRAW);
}

void Renderer::draw(Sim& sim, const glm::mat4& view, const glm::mat4& proj, float nowSec) {
    glBindVertexArray(dummyVao);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sim.renderPosBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, sim.frameBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, atomsSSBO);
    if (nearIds) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, nearIds);
    if (nearMask) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, nearMask);
    if (nearOffs) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, nearOffs);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, sim.ageBuf);

    int nImg = 1, imgMode = 0;
    if (sim.p.pbc == 1 && opts.repeats == 1) { nImg = 3; imgMode = 1; }
    if (sim.p.pbc == 1 && opts.repeats == 2) { nImg = 27; imgMode = 2; }

    capProg.use();
    capProg.set("view", view);
    capProg.set("proj", proj);
    capProg.set("nBeads", sim.p.nBeads);
    capProg.set("tubeR", opts.tubeR);
    capProg.set("dPeriod", dPeriod);
    capProg.set("segLen", sim.segLen);
    capProg.set("colorMode", opts.colorMode);
    capProg.set("fogColor", opts.fogColor);
    capProg.set("fogDensity", opts.fogDensity);
    capProg.set("nSegTotal", sim.nSeg());
    capProg.set("imgMode", imgMode);
    capProg.set("boxSize", 2.f * sim.p.boxHalf);
    capProg.set("nowSec", nowSec);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, GLsizei(size_t(sim.nSeg()) * nImg));

    if (nearCount > 0) {
        atomProg.use();
        atomProg.set("view", view);
        atomProg.set("proj", proj);
        atomProg.set("natoms", natoms);
        atomProg.set("nBeads", sim.p.nBeads);
        atomProg.set("segLen", sim.segLen);
        atomProg.set("radScale", opts.radScale);
        atomProg.set("dPeriod", dPeriod);
        atomProg.set("colorMode", opts.colorMode);
        atomProg.set("fogColor", opts.fogColor);
        atomProg.set("fogDensity", opts.fogDensity);
        atomProg.set("nowSec", nowSec);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, GLsizei(size_t(nearCount) * natoms));
    }
}
