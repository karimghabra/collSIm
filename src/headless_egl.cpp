// Headless OpenGL context for machines with no window system.
//
// GLFW 3.4 cannot provide one: its null platform asks EGL for the default
// display and then for a *window* surface on a null native window, which
// fails. So the context is made here directly, on Mesa's surfaceless EGL
// platform, with no surface at all -- every render target the engine draws
// into is a framebuffer object of its own.
//
// libEGL is opened at run time and the few entry points and constants are
// declared by hand, so building this needs neither EGL headers nor a
// link-time dependency on a library that a headless machine may only have
// in its runtime form.
#include "headless_egl.h"

#if defined(__linux__)
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

namespace {
using EGLDisplay = void*;
using EGLConfig = void*;
using EGLContext = void*;
using EGLSurface = void*;
using EGLint = int32_t;
using EGLBoolean = unsigned int;
using EGLenum = unsigned int;
using EGLAttrib = intptr_t;

const EGLint kNone = 0x3038;                    // EGL_NONE
const EGLint kExtensions = 0x3055;              // EGL_EXTENSIONS
const EGLint kSurfaceType = 0x3033;             // EGL_SURFACE_TYPE
const EGLint kPbufferBit = 0x0001;              // EGL_PBUFFER_BIT
const EGLint kRenderableType = 0x3040;          // EGL_RENDERABLE_TYPE
const EGLint kOpenGLBit = 0x0008;               // EGL_OPENGL_BIT
const EGLint kContextMajor = 0x3098;            // EGL_CONTEXT_MAJOR_VERSION
const EGLint kContextMinor = 0x30FB;            // EGL_CONTEXT_MINOR_VERSION
const EGLint kContextProfileMask = 0x30FD;      // EGL_CONTEXT_OPENGL_PROFILE_MASK
const EGLint kContextCoreBit = 0x0001;          // EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT
const EGLenum kOpenGLApi = 0x30A2;              // EGL_OPENGL_API
const EGLenum kPlatformSurfaceless = 0x31DD;    // EGL_PLATFORM_SURFACELESS_MESA

void* (*g_getProcAddress)(const char*) = nullptr;

void (*loadSymbol(const char* name))() {
    return reinterpret_cast<void (*)()>(g_getProcAddress(name));
}
}  // namespace

HeadlessLoader headlessEglInit() {
    void* lib = dlopen("libEGL.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
        fprintf(stderr, "headless: cannot open libEGL.so.1 (%s)\n", dlerror());
        return nullptr;
    }
#define EGL_SYM(type, name)                                              \
    auto name = reinterpret_cast<type>(dlsym(lib, #name));               \
    if (!name) {                                                         \
        fprintf(stderr, "headless: libEGL.so.1 has no " #name "\n");     \
        return nullptr;                                                  \
    }
    EGL_SYM(const char* (*)(EGLDisplay, EGLint), eglQueryString)
    EGL_SYM(EGLDisplay (*)(EGLenum, void*, const EGLAttrib*), eglGetPlatformDisplay)
    EGL_SYM(EGLBoolean (*)(EGLDisplay, EGLint*, EGLint*), eglInitialize)
    EGL_SYM(EGLBoolean (*)(EGLenum), eglBindAPI)
    EGL_SYM(EGLBoolean (*)(EGLDisplay, const EGLint*, EGLConfig*, EGLint, EGLint*),
            eglChooseConfig)
    EGL_SYM(EGLContext (*)(EGLDisplay, EGLConfig, EGLContext, const EGLint*),
            eglCreateContext)
    EGL_SYM(EGLBoolean (*)(EGLDisplay, EGLSurface, EGLSurface, EGLContext), eglMakeCurrent)
    EGL_SYM(EGLint (*)(), eglGetError)
    EGL_SYM(void* (*)(const char*), eglGetProcAddress)
#undef EGL_SYM

    const char* clientExt = eglQueryString(nullptr, kExtensions);
    if (!clientExt || !strstr(clientExt, "EGL_MESA_platform_surfaceless")) {
        fprintf(stderr, "headless: this EGL has no surfaceless platform "
                        "(EGL_MESA_platform_surfaceless)\n");
        return nullptr;
    }
    EGLDisplay dpy = eglGetPlatformDisplay(kPlatformSurfaceless, nullptr, nullptr);
    EGLint eglMajor = 0, eglMinor = 0;
    if (!dpy || !eglInitialize(dpy, &eglMajor, &eglMinor)) {
        fprintf(stderr, "headless: eglInitialize failed (EGL error 0x%x)\n", eglGetError());
        return nullptr;
    }
    if (!eglBindAPI(kOpenGLApi)) {
        fprintf(stderr, "headless: this EGL does not offer desktop OpenGL\n");
        return nullptr;
    }
    // eglChooseConfig defaults EGL_SURFACE_TYPE to WINDOW, which a surfaceless
    // display never offers; ask for pbuffer-capable configs instead.
    const EGLint configAttribs[] = {kRenderableType, kOpenGLBit,
                                    kSurfaceType, kPbufferBit, kNone};
    EGLConfig config = nullptr;
    EGLint nConfig = 0;
    if (!eglChooseConfig(dpy, configAttribs, &config, 1, &nConfig) || nConfig < 1) {
        fprintf(stderr, "headless: no OpenGL-capable EGL config (EGL error 0x%x)\n",
                eglGetError());
        return nullptr;
    }
    EGLContext ctx = nullptr;
    for (EGLint minor : {6, 5}) {
        const EGLint contextAttribs[] = {kContextMajor, 4, kContextMinor, minor,
                                         kContextProfileMask, kContextCoreBit, kNone};
        ctx = eglCreateContext(dpy, config, nullptr, contextAttribs);
        if (ctx) break;
    }
    if (!ctx) {
        fprintf(stderr, "headless: no OpenGL 4.6 or 4.5 core context (EGL error 0x%x)\n",
                eglGetError());
        return nullptr;
    }
    if (!eglMakeCurrent(dpy, nullptr, nullptr, ctx)) {
        fprintf(stderr, "headless: eglMakeCurrent without a surface failed "
                        "(EGL error 0x%x)\n", eglGetError());
        return nullptr;
    }
    g_getProcAddress = eglGetProcAddress;
    return loadSymbol;
}

#else

HeadlessLoader headlessEglInit() { return nullptr; }

#endif
