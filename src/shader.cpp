#include "shader.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

static fs::path g_root;

#ifdef _WIN32
#include <windows.h>
#endif

static void findRoot() {
    if (!g_root.empty()) return;
    auto search = [](fs::path p) -> fs::path {
        for (int i = 0; i < 6; ++i) {
            if (fs::exists(p / "shaders") && fs::exists(p / "assets")) return p;
            if (!p.has_parent_path() || p == p.parent_path()) break;
            p = p.parent_path();
        }
        return {};
    };
    g_root = search(fs::current_path());
#ifdef _WIN32
    if (g_root.empty()) {
        char buf[MAX_PATH];
        if (GetModuleFileNameA(nullptr, buf, MAX_PATH))
            g_root = search(fs::path(buf).parent_path());
    }
#endif
    if (g_root.empty())
        throw std::runtime_error("could not locate shaders/ + assets/ near " +
                                 fs::current_path().string());
}

std::string resolveAsset(const std::string& rel) {
    findRoot();
    return (g_root / rel).string();
}

static std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static GLuint compile(GLenum type, const std::string& src, const std::string& name) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "=== shader compile error in %s ===\n%s\n", name.c_str(), log);
        throw std::runtime_error("shader compile failed: " + name);
    }
    return s;
}

static void link(GLuint prog, const std::string& name) {
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[8192];
        glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        fprintf(stderr, "=== link error %s ===\n%s\n", name.c_str(), log);
        throw std::runtime_error("link failed: " + name);
    }
}

void Program::computeFromFile(const std::string& path) {
    std::string full = resolveAsset(path);
    GLuint s = compile(GL_COMPUTE_SHADER, readFile(full), path);
    id = glCreateProgram();
    glAttachShader(id, s);
    link(id, path);
    glDeleteShader(s);
}

void Program::graphicsFromFiles(const std::string& vsPath, const std::string& fsPath) {
    GLuint v = compile(GL_VERTEX_SHADER, readFile(resolveAsset(vsPath)), vsPath);
    GLuint f = compile(GL_FRAGMENT_SHADER, readFile(resolveAsset(fsPath)), fsPath);
    id = glCreateProgram();
    glAttachShader(id, v);
    glAttachShader(id, f);
    link(id, vsPath + "+" + fsPath);
    glDeleteShader(v);
    glDeleteShader(f);
}
