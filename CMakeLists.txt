cmake_minimum_required(VERSION 3.24)
project(collagenSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# Every download is pinned by hash: a truncated or altered archive stops the
# configure step instead of being unpacked and built.
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.6.zip
  URL_HASH SHA256=549963a124817c03514ceb9d3f49d4a5a4c10bdc747fdc62d3797b9b2030135f
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(glm
  URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
  URL_HASH SHA256=09c5716296787e1f7fcb87b1cbdbf26814ec1288ed6259ccd30d5d9795809fa5
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(glfw imgui glm)

add_library(glad STATIC extern/glad/src/gl.c)
target_include_directories(glad PUBLIC extern/glad/include)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC glfw)
# The GLFW backend never calls OpenGL. Without this, glfw3.h includes the
# system <GL/gl.h>, which a Linux machine without the Mesa development
# package does not have.
target_compile_definitions(imgui_lib PRIVATE GLFW_INCLUDE_NONE)

add_executable(collagenSim
  src/main.cpp
  src/shader.cpp
  src/io.cpp
  src/sim.cpp
  src/we.cpp
  src/render.cpp
  src/headless_egl.cpp)
target_include_directories(collagenSim PRIVATE extern src)
target_link_libraries(collagenSim PRIVATE glad glfw imgui_lib glm::glm)
if (WIN32)
  target_link_libraries(collagenSim PRIVATE opengl32)
else()
  # glad resolves every GL entry point at run time through the context's own
  # loader, so nothing is linked; libEGL is opened with dlopen for --headless
  target_link_libraries(collagenSim PRIVATE ${CMAKE_DL_LIBS})
endif()
if (MSVC)
  target_compile_options(collagenSim PRIVATE /W3 /wd4267 /wd4244)
  target_compile_definitions(collagenSim PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# Scale ladder (docs/multiscale_framework.md). Deliberately free of GL, GLM and
# every other dependency: a rung is a free energy, a mobility and a propagator,
# and none of those need a graphics context. It also means the validation gates
# run anywhere, including in CI on a machine with no GPU.
set(SCALE_SOURCES
  src/scale/level.cpp
  src/scale/mala.cpp
  src/scale/kramers.cpp
  src/scale/microfibril.cpp)

add_library(scale STATIC ${SCALE_SOURCES})
target_include_directories(scale PUBLIC src)

add_executable(scale_selftest tests/scale_selftest.cpp)
target_link_libraries(scale_selftest PRIVATE scale)
if (MSVC)
  target_compile_options(scale PRIVATE /W3)
  target_compile_options(scale_selftest PRIVATE /W3)
endif()
enable_testing()
add_test(NAME scale_selftest COMMAND scale_selftest)
