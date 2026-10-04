cmake_minimum_required(VERSION 3.22)
project(rigid_backend LANGUAGES C CXX)
include(FetchContent)
set(BOX2D_SAMPLES OFF CACHE BOOL "" FORCE)
set(BOX2D_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BOX2D_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BOX2D_VALIDATE OFF CACHE BOOL "" FORCE)
option(RIGID_BLOCK_BACKEND "Build the Box2D 2.4.1 two-point block solver comparator" OFF)
if(RIGID_BLOCK_BACKEND)
set(BOX2D_BUILD_TESTBED OFF CACHE BOOL "" FORCE)
set(BOX2D_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.8)
FetchContent_Declare(box2d
  URL https://codeload.github.com/erincatto/box2d/tar.gz/v2.4.1
  URL_HASH SHA256=d6b4650ff897ee1ead27cf77a5933ea197cbeef6705638dd181adc2e816b23c2
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
else()
FetchContent_Declare(box2d
  URL https://codeload.github.com/erincatto/box2d/tar.gz/8c661469c9507d3ad6fbd2fea3f1aa71669c2fe3
  URL_HASH SHA256=dee7bcf0f50b3dcccf67678ce14ffe3ce02583b51b8b03e447ab30a84c3436ef
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
endif()
FetchContent_MakeAvailable(box2d)
add_executable(rigid_runner runner.cpp)
target_compile_features(rigid_runner PRIVATE cxx_std_17)
target_link_libraries(rigid_runner PRIVATE box2d)
if(RIGID_BLOCK_BACKEND)
  target_compile_definitions(rigid_runner PRIVATE RIGID_BLOCK_BACKEND)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(rigid_runner PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
endif()
