cmake_minimum_required(VERSION 3.24)
project(collagenSim LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.91.6.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(glm
  URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.zip
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

add_executable(collagenSim
  src/main.cpp
  src/shader.cpp
  src/io.cpp
  src/sim.cpp
  src/we.cpp
  src/render.cpp)
target_include_directories(collagenSim PRIVATE extern src)
target_link_libraries(collagenSim PRIVATE glad glfw imgui_lib glm::glm opengl32)
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
