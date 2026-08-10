#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct OrbitCamera {
    glm::vec3 target{0, 0, 0};
    float yaw = 0.6f, pitch = 0.35f, dist = 900.f;
    float fovDeg = 40.f;

    glm::vec3 eye() const {
        float cp = cosf(pitch), sp = sinf(pitch);
        return target + dist * glm::vec3(cp * cosf(yaw), sp, cp * sinf(yaw));
    }
    glm::mat4 view() const { return glm::lookAt(eye(), target, {0, 1, 0}); }
    glm::mat4 proj(float aspect) const {
        return glm::perspective(glm::radians(fovDeg), aspect, 2.f, 8000.f);
    }
    void rotate(float dx, float dy) {
        yaw += dx * 0.006f;
        pitch = glm::clamp(pitch + dy * 0.006f, -1.5f, 1.5f);
    }
    void pan(float dx, float dy) {
        glm::vec3 f = glm::normalize(target - eye());
        glm::vec3 r = glm::normalize(glm::cross(f, {0, 1, 0}));
        glm::vec3 u = glm::cross(r, f);
        target += (-r * dx + u * dy) * dist * 0.0012f;
    }
    void zoom(float d) { dist = glm::clamp(dist * powf(0.9f, d), 20.f, 5000.f); }
};
