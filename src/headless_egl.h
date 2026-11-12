#pragma once

// OpenGL context for a machine with no window system (--headless).
//
// Makes a surfaceless core-profile context current on the calling thread and
// returns a symbol loader for it, suitable for gladLoadGL, or nullptr if none
// can be made (the reason is printed to stderr). OpenGL 4.6 is requested
// first; 4.5 is accepted as a fallback, because that is where Mesa's software
// renderer stops and it is the only OpenGL a machine without a GPU has.
//
// Linux only. On every other platform this returns nullptr.
using HeadlessLoader = void (*(*)(const char*))();
HeadlessLoader headlessEglInit();
