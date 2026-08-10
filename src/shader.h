#pragma once
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <string>

// Compile-from-file shader helpers. Paths are resolved against the project
// root found by searching upward from the executable for a "shaders" dir.
std::string resolveAsset(const std::string& rel);

struct Program {
    GLuint id = 0;
    void computeFromFile(const std::string& path);
    void graphicsFromFiles(const std::string& vsPath, const std::string& fsPath);
    void use() const { glUseProgram(id); }
    GLint loc(const char* name) const { return glGetUniformLocation(id, name); }
    void set(const char* n, float v) const { glUniform1f(loc(n), v); }
    void set(const char* n, int v) const { glUniform1i(loc(n), v); }
    void set(const char* n, unsigned v) const { glUniform1ui(loc(n), v); }
    void set(const char* n, const glm::vec3& v) const { glUniform3fv(loc(n), 1, &v.x); }
    void set(const char* n, const glm::vec4& v) const { glUniform4fv(loc(n), 1, &v.x); }
    void set(const char* n, const glm::mat4& m) const { glUniformMatrix4fv(loc(n), 1, GL_FALSE, &m[0][0]); }
};
