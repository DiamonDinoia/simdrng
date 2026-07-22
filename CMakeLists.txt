cmake_minimum_required(VERSION 3.29)

# ---------------------------------------------------------------------------
# Version. The top-level VERSION file holds the canonical "x.y.z"; project()
# consumes it, and GenerateVersion.cmake derives a git-aware full string that
# is baked into include/simdrng/version.hpp.
# ---------------------------------------------------------------------------
file(
    STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
    SIMDRNG_VERSION_BASE
    LIMIT_COUNT 1
)
string(STRIP "${SIMDRNG_VERSION_BASE}" SIMDRNG_VERSION_BASE)

project(simdrng VERSION ${SIMDRNG_VERSION_BASE} LANGUAGES CXX)

# if no build type was specified, default to Release — but only as the top-level
# project. When FetchContent'd/add_subdirectory'd, forcing the parent's build
# type would be pollution.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# ---------------------------------------------------------------------------
# Options (all SIMDRNG_*-prefixed).
# ---------------------------------------------------------------------------
# CTest injects dashboard targets (Nightly/Continuous/…) and defines
# BUILD_TESTING into whatever project includes it — keep it to the top level.
if(PROJECT_IS_TOP_LEVEL)
    include(CTest) # defines BUILD_TESTING (default ON)
endif()

option(
    SIMDRNG_BUILD_TESTS
    "Build the Catch2 test suite and benchmarks"
    ${PROJECT_IS_TOP_LEVEL}
)
option(
    SIMDRNG_INSTALL
    "Generate the install/export rules for simdrng"
    ${PROJECT_IS_TOP_LEVEL}
)
option(
    SIMDRNG_BUILD_C_API
    "Build the C ABI (shared + static) that language bindings link against"
    ${PROJECT_IS_TOP_LEVEL}
)
option(SIMDRNG_BUILD_PYTHON "Build the nanobind Python bindings" OFF)
option(
    SIMDRNG_DISPATCH
    "Build every SIMD tier and select at runtime; simdrng::Xoshiro aliases the \
portable dispatch type (binaries to ship). OFF (default) builds no tier and \
aliases XoshiroNative (xsimd::best_arch): fastest for the build host, NOT \
portable — for local/run-only builds."
    OFF
)
option(SIMDRNG_BUILD_MATLAB "Build the mwrap MATLAB/Octave binding" OFF)
option(SIMDRNG_BUILD_FORTRAN "Build the iso_c_binding Fortran binding" OFF)
option(
    SIMDRNG_BUILD_JULIA
    "Register the Julia binding's ctest (needs julia)"
    OFF
)
option(SIMDRNG_BUILD_JS "Build the JS/TS binding (N-API addon or WASM)" OFF)
option(SIMDRNG_BUILD_EXAMPLES "Build the C++ examples under examples/cpp" OFF)
option(SIMDRNG_BUILD_DOCS "Build Sphinx/Doxygen documentation" OFF)
option(SIMDRNG_MARCH_NATIVE "Build benchmarks with -march=native" OFF)
option(
    SIMDRNG_ENABLE_CODSPEED
    "Link the benchmark harness against codspeed-cpp for CodSpeed instrumentation"
    OFF
)
option(
    SIMDRNG_DEVEL
    "Enable development tooling (CPM, warnings helpers, coverage, analysis)"
    OFF
)

# ---------------------------------------------------------------------------
# Version header. version.hpp is committed and also refreshed by the pre-commit
# hook, so consumers get a valid header from the checkout. Only regenerate it
# into the source tree as the top-level project — a FetchContent consumer must
# not have its fetched source dirtied on every configure.
# ---------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL)
    include(cmake/GenerateVersion.cmake)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/simdrng/version.hpp.in"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/simdrng/version.hpp"
        @ONLY
    )
endif()

# ---------------------------------------------------------------------------
# Developer helpers / CPM bootstrap (needed to fetch xsimd/poet).
# ---------------------------------------------------------------------------
include(cmake/dev_helpers.cmake)

# ---------------------------------------------------------------------------
# Dependencies.
#   xsimd : SIMD backend.
#   poet  : compile-time helpers used by the SIMD headers and Python bindings.
# Pinned for reproducibility (xsimd by release tag, poet by SHA).
# ---------------------------------------------------------------------------
set(XSIMD_DOWNLOAD_VERSION 14.3.0)
set(POET_DOWNLOAD_VERSION d5a86a2de13cbe2283bb6c51e4dc2f5170196278) # DiamonDinoia/poet main (GenerateVersion add_subdirectory fix)

# SYSTEM YES marks the fetched headers as system includes (-isystem) so their
# warnings (e.g. xsimd's -Wpedantic __int128) don't surface in our build.
# Guard on the target so a parent project that already provides xsimd/poet
# (by any means, not just CPM) wins instead of clashing with our fetch.
if(NOT TARGET xsimd)
    cpmaddpackage(
            NAME xsimd
            GITHUB_REPOSITORY xtensor-stack/xsimd
            GIT_TAG ${XSIMD_DOWNLOAD_VERSION}
            GIT_SHALLOW NO
            GIT_PROGRESS YES
            EXCLUDE_FROM_ALL YES
            SYSTEM YES
    )
endif()

# poet defines the poet::poet INTERFACE target itself; add_subdirectory it
# and disable its tests/benchmarks/examples so we only pull the headers.
if(NOT TARGET poet::poet)
    cpmaddpackage(
            NAME poet
            GITHUB_REPOSITORY DiamonDinoia/poet
            GIT_TAG ${POET_DOWNLOAD_VERSION}
            GIT_SHALLOW NO
            GIT_PROGRESS YES
            EXCLUDE_FROM_ALL YES
            SYSTEM YES
            OPTIONS "POET_BUILD_TESTS OFF" "POET_BUILD_BENCHMARKS OFF"
                    "POET_BUILD_EXAMPLES OFF" "POET_BUILD_DEVEL OFF"
    )
endif()

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckIPOSupported)
include(cmake/CompilerOptions.cmake)

check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)

# When IPO is on under GCC, build the library with *fat* LTO objects (machine code
# alongside the GIMPLE bytecode). The library and benchmarks still link with full
# LTO, but the test executables link with -fno-lto (see tests/CMakeLists.txt) and
# so consume the embedded machine code directly. That keeps lto-wrapper out of the
# test link entirely, which sidesteps a GCC aarch64 LTO ICE
# (internal compiler error in lto_tag_to_tree_code) hit while streaming some of the
# dispatch template instantiations that only testBulkAPI pulls in — a known GCC LTO
# bug, not a code defect (clang/aarch64 and gcc/x86 link the same objects cleanly).
set(_ipo_fat_flag "")
if(_ipo_ok AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(_ipo_fat_flag "-ffat-lto-objects")
endif()

if(
    NOT DEFINED CMAKE_LINKER_TYPE
    AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
)
    set(CMAKE_LINKER_TYPE LLD)
endif()

# List candidate standards in descending order.
# Use CMAKE_CXX_COMPILE_FEATURES (portable across MSVC/GCC/Clang) rather than
# probing raw -std=c++XX flags, which MSVC doesn't accept.
set(CXX_STANDARDS "26" "23" "20" "17")
set(SUPPORTED_CXX_STANDARD "")

foreach(std IN LISTS CXX_STANDARDS)
    if("cxx_std_${std}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(SUPPORTED_CXX_STANDARD ${std})
        break()
    endif()
endforeach()

if(NOT SUPPORTED_CXX_STANDARD)
    message(FATAL_ERROR "The compiler does not support C++17 or higher.")
endif()

# ---------------------------------------------------------------------------
# Library target: a STATIC library carrying the runtime SIMD dispatch TUs.
# ---------------------------------------------------------------------------
add_library(simdrng STATIC src/xoshiro_simd.cpp)
add_library(simdrng::simdrng ALIAS simdrng)
target_include_directories(
    simdrng
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
# poet is consumed header-only and not part of our export set, so reference
# it only in the build tree; installed consumers obtain it via the
# find_dependency(poet) call emitted into simdrngConfig.cmake.
target_link_libraries(simdrng PUBLIC xsimd $<BUILD_INTERFACE:poet::poet>)
set_target_properties(
    simdrng
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ${_ipo_ok}
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${_ipo_ok}
        INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${_ipo_ok}
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
# Suppress GCC 15 -Wtemplate-body: xsimd template definitions reference
# SIMD intrinsics that don't exist without a SIMD arch, but these templates
# are only instantiated in dispatch TUs compiled with the correct -march flags.
add_compile_flags_if_supported(TEMPLATE_BODY_FLAGS -Wno-template-body)
target_compile_options(simdrng PUBLIC ${TEMPLATE_BODY_FLAGS})
if(_ipo_fat_flag)
    target_compile_options(simdrng PRIVATE ${_ipo_fat_flag})
endif()
target_compile_features(simdrng PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
target_compile_features(simdrng INTERFACE cxx_std_20)

# Propagate the dispatch choice to consumers: it selects what simdrng::Xoshiro
# aliases (XoshiroSIMD when ON, XoshiroNative/best_arch when OFF) and must match
# the tiers actually compiled into the library. PUBLIC so it rides the installed
# INTERFACE and downstream sees the same default.
if(SIMDRNG_DISPATCH)
    set(SIMDRNG_DISPATCH_VALUE 1)
else()
    set(SIMDRNG_DISPATCH_VALUE 0)
endif()
target_compile_definitions(
    simdrng
    PUBLIC SIMDRNG_DISPATCH=${SIMDRNG_DISPATCH_VALUE}
)

# ---------------------------------------------------------------------------
# Architecture-specific SIMD dispatch variants (SIMDRNG_DISPATCH=ON only).
# Each variant is compiled in its own TU with flags enabling a specific SIMD
# level; xsimd::dispatch picks the best available at runtime — portable binaries
# safe on any CPU down to the baseline (ship path). With SIMDRNG_DISPATCH=OFF
# (default) no tier is built: SIMDRNG_DISPATCH_VARIANTS stays unset, the loop
# below is a no-op, and simdrng::Xoshiro aliases XoshiroNative (xsimd::best_arch)
# — fastest for the build host, non-portable, for local/run-only builds.
# ---------------------------------------------------------------------------
if(SIMDRNG_DISPATCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(SIMDRNG_DISPATCH_VARIANTS "x86-64;x86-64-v3;x86-64-v4")
        if(MSVC)
            # MSVC has no -march equivalent and doesn't define __SSE2__ even
            # though x64 is SSE2 baseline; define it explicitly so the dispatch
            # TUs pick the right template instantiation. MSVC does define
            # __AVX2__ under /arch:AVX2 and the AVX-512 family under /arch:AVX512.
            set(SIMDRNG_DISPATCH_FLAGS_x86_64 /D__SSE2__)
            set(SIMDRNG_DISPATCH_FLAGS_x86_64_v3 /arch:AVX2)
            set(SIMDRNG_DISPATCH_FLAGS_x86_64_v4 /arch:AVX512)
        else()
            set(SIMDRNG_DISPATCH_FLAGS_x86_64 -march=x86-64 -mtune=generic)
            set(SIMDRNG_DISPATCH_FLAGS_x86_64_v3
                -march=x86-64-v3
                -mtune=generic
            )
            set(SIMDRNG_DISPATCH_FLAGS_x86_64_v4
                -march=x86-64-v4
                -mtune=generic
            )
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(SIMDRNG_DISPATCH_VARIANTS "neon64")
        set(SIMDRNG_DISPATCH_FLAGS_neon64 "") # NEON is baseline on aarch64
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
        set(SIMDRNG_DISPATCH_VARIANTS "rvv")
        set(SIMDRNG_DISPATCH_FLAGS_rvv -march=rv64gcv -mrvv-vector-bits=zvl)
    else()
        message(
            FATAL_ERROR
            "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}"
        )
    endif()
endif()

foreach(SIMDRNG_TYPE IN ITEMS xoshiro chacha philox)
    foreach(VARIANT IN LISTS SIMDRNG_DISPATCH_VARIANTS)
        string(REPLACE "-" "_" TARGET_SUFFIX "${VARIANT}")
        set(SIMD_SOURCE_TARGET "${SIMDRNG_TYPE}_simd_source_${TARGET_SUFFIX}")

        add_library(
            ${SIMD_SOURCE_TARGET}
            OBJECT
            src/${SIMDRNG_TYPE}_simd_dispatch.cpp
        )
        target_include_directories(
            ${SIMD_SOURCE_TARGET}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        target_link_libraries(${SIMD_SOURCE_TARGET} PRIVATE xsimd poet::poet)
        target_compile_options(
            ${SIMD_SOURCE_TARGET}
            PRIVATE ${SIMDRNG_DISPATCH_FLAGS_${TARGET_SUFFIX}} ${_ipo_fat_flag}
        )
        set_target_properties(
            ${SIMD_SOURCE_TARGET}
            PROPERTIES
                POSITION_INDEPENDENT_CODE ON
                INTERPROCEDURAL_OPTIMIZATION_RELEASE ${_ipo_ok}
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${_ipo_ok}
                INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${_ipo_ok}
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
        )
        target_compile_features(
            ${SIMD_SOURCE_TARGET}
            PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD}
        )
        target_sources(simdrng PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
    endforeach()
endforeach()

# ---------------------------------------------------------------------------
# C ABI. A thin extern-"C" wrapper (src/capi.cpp) over the simdrng static lib,
# shipped as both a shared library (simdrng_c — for dlopen/Julia/Fortran) and a
# static archive (simdrng_c_static — for self-contained MEX/N-API bundles).
# Both come from one OBJECT library so the glue is compiled once. Compiled at
# baseline arch: it only calls the runtime-dispatched public API, whose per-ISA
# code already lives in the simdrng archive.
# ---------------------------------------------------------------------------
if(SIMDRNG_BUILD_C_API)
    add_library(simdrng_c_objects OBJECT src/capi.cpp)
    target_link_libraries(simdrng_c_objects PRIVATE simdrng) # usage requirements
    target_compile_definitions(simdrng_c_objects PRIVATE SIMDRNG_C_BUILD)
    target_compile_features(
        simdrng_c_objects
        PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD}
    )
    set_target_properties(
        simdrng_c_objects
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            C_VISIBILITY_PRESET hidden
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
            # capi.cpp uses no C++20 modules. Disable scanning so CMake doesn't
            # put an @...modmap flag in compile_commands.json for a build-time
            # file that clang-tidy (run without building this target) can't find.
            CXX_SCAN_FOR_MODULES OFF
    )

    add_library(simdrng_c SHARED $<TARGET_OBJECTS:simdrng_c_objects>)
    add_library(simdrng::simdrng_c ALIAS simdrng_c)
    target_link_libraries(simdrng_c PRIVATE simdrng)
    target_include_directories(
        simdrng_c
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    set_target_properties(
        simdrng_c
        PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR}
    )

    add_library(simdrng_c_static STATIC $<TARGET_OBJECTS:simdrng_c_objects>)
    add_library(simdrng::simdrng_c_static ALIAS simdrng_c_static)
    target_link_libraries(simdrng_c_static PRIVATE simdrng)
    target_include_directories(
        simdrng_c_static
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    # Consumers of the static lib see a plain (non-dllimport) header.
    target_compile_definitions(simdrng_c_static INTERFACE SIMDRNG_STATIC)
endif()

if(SIMDRNG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SIMDRNG_BUILD_EXAMPLES)
    if(NOT SIMDRNG_BUILD_TESTS)
        enable_testing()
    endif()
    add_subdirectory(examples/cpp)
endif()

if(SIMDRNG_BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(SIMDRNG_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# MATLAB/Octave binding links simdrng_c_static, so it needs the C ABI.
if(SIMDRNG_BUILD_MATLAB)
    if(NOT SIMDRNG_BUILD_C_API)
        message(
            FATAL_ERROR
            "SIMDRNG_BUILD_MATLAB requires SIMDRNG_BUILD_C_API (the binding links simdrng_c_static)."
        )
    endif()
    if(NOT SIMDRNG_BUILD_TESTS)
        enable_testing()
    endif()
    add_subdirectory(bindings/matlab)
endif()

# Fortran binding links simdrng_c_static, so it needs the C ABI.
if(SIMDRNG_BUILD_FORTRAN)
    if(NOT SIMDRNG_BUILD_C_API)
        message(
            FATAL_ERROR
            "SIMDRNG_BUILD_FORTRAN requires SIMDRNG_BUILD_C_API (the binding links simdrng_c_static)."
        )
    endif()
    if(NOT SIMDRNG_BUILD_TESTS)
        enable_testing()
    endif()
    add_subdirectory(bindings/fortran)
endif()

# Julia binding dlopen()s the shared simdrng_c at test time, so it needs the C API.
if(SIMDRNG_BUILD_JULIA)
    if(NOT SIMDRNG_BUILD_C_API)
        message(
            FATAL_ERROR
            "SIMDRNG_BUILD_JULIA requires SIMDRNG_BUILD_C_API (the package dlopen()s simdrng_c)."
        )
    endif()
    if(NOT SIMDRNG_BUILD_TESTS)
        enable_testing()
    endif()
    add_subdirectory(bindings/julia)
endif()

# JS binding links simdrng_c_static (native addon) / exports its symbols (WASM).
if(SIMDRNG_BUILD_JS)
    if(NOT SIMDRNG_BUILD_C_API)
        message(
            FATAL_ERROR
            "SIMDRNG_BUILD_JS requires SIMDRNG_BUILD_C_API (the binding links simdrng_c_static)."
        )
    endif()
    if(NOT SIMDRNG_BUILD_TESTS)
        enable_testing()
    endif()
    add_subdirectory(bindings/js)
endif()

# ---------------------------------------------------------------------------
# Installation. Gated on SIMDRNG_INSTALL (default: top-level only) so a
# FetchContent consumer's `cmake --install` doesn't also install simdrng's
# headers and package config.
# ---------------------------------------------------------------------------
if(SIMDRNG_INSTALL)
    set(SIMDRNG_INSTALL_TARGETS simdrng)
    if(SIMDRNG_BUILD_C_API)
        list(APPEND SIMDRNG_INSTALL_TARGETS simdrng_c simdrng_c_static)
    endif()
    install(
        TARGETS ${SIMDRNG_INSTALL_TARGETS}
        EXPORT simdrngTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )

    install(
        DIRECTORY include/
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
    )

    install(
        EXPORT simdrngTargets
        FILE simdrngTargets.cmake
        NAMESPACE simdrng::
        DESTINATION lib/cmake/simdrng
    )

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    configure_file(
        cmake/simdrngConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfig.cmake"
        @ONLY
    )

    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfigVersion.cmake"
        DESTINATION lib/cmake/simdrng
    )
endif()
