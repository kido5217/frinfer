# Configure-time build identity for the product binaries' `--version` output.
#
# The version is the exact git description (`git describe --tags --always --dirty`) so it matches
# the release tag, falling back to the CMake project version for a git-less source tree. The
# generated header is exposed by the `ninfer_build_info` interface target and read by
# `src/product/build_info/build_info.h`.

set(NINFER_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
set(NINFER_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
set(NINFER_CUDA_ARCH "sm_${CMAKE_CUDA_ARCHITECTURES}")
set(NINFER_VERSION "${PROJECT_VERSION}")
set(NINFER_GIT_SHA "unknown")

find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE NINFER_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _git_describe_status)
  if(NOT _git_describe_status EQUAL 0 OR NINFER_VERSION STREQUAL "")
    set(NINFER_VERSION "${PROJECT_VERSION}")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE NINFER_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _git_sha_status)
  if(NOT _git_sha_status EQUAL 0 OR NINFER_GIT_SHA STREQUAL "")
    set(NINFER_GIT_SHA "unknown")
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/build_info.h.in"
  "${CMAKE_BINARY_DIR}/generated/ninfer/build_info.h"
  @ONLY)

add_library(ninfer_build_info INTERFACE)
target_include_directories(ninfer_build_info INTERFACE "${CMAKE_BINARY_DIR}/generated")
