cmake_minimum_required(VERSION 3.29)

# ---------------------------------------------------------------------------
# Version. The top-level VERSION file holds the canonical "x.y.z"; project()
# consumes it, and GenerateVersion.cmake derives a git-aware full string that
# is baked into include/simdrng/version.hpp.
# ---------------------------------------------------------------------------
file(
    STRINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
    SIMDRNG_VERSION_BASE
    LIMIT_COUNT 1
)
string(STRIP "${SIMDRNG_VERSION_BASE}" SIMDRNG_VERSION_BASE)

project(simdrng VERSION ${SIMDRNG_VERSION_BASE} LANGUAGES CXX)

# if no build type was specified, default to Release
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# ---------------------------------------------------------------------------
# Options (all SIMDRNG_*-prefixed).
# ---------------------------------------------------------------------------
include(CTest) # defines BUILD_TESTING (default ON)

option(
    SIMDRNG_WITH_XSIMD
    "Build the SIMD backends with xsimd (OFF = scalar-only header library)"
    ON
)
option(
    SIMDRNG_BUILD_TESTS
    "Build the Catch2 test suite and benchmarks"
    ${BUILD_TESTING}
)
option(
    SIMDRNG_BUILD_PYTHON
    "Build the nanobind Python bindings (requires SIMDRNG_WITH_XSIMD)"
    OFF
)
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

if(SIMDRNG_BUILD_PYTHON AND NOT SIMDRNG_WITH_XSIMD)
    message(
        FATAL_ERROR
        "SIMDRNG_BUILD_PYTHON requires SIMDRNG_WITH_XSIMD=ON (the Python bindings expose the SIMD generators)."
    )
endif()

# ---------------------------------------------------------------------------
# Version header (committed; regenerated here and by the pre-commit hook).
# ---------------------------------------------------------------------------
include(cmake/GenerateVersion.cmake)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/simdrng/version.hpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/simdrng/version.hpp"
    @ONLY
)

# ---------------------------------------------------------------------------
# Developer helpers / CPM bootstrap. Needed whenever we fetch a dependency
# (xsimd/poet) or build any dev artifact. A bare scalar install pulls in
# nothing extra.
# ---------------------------------------------------------------------------
if(
    SIMDRNG_WITH_XSIMD
    OR SIMDRNG_BUILD_TESTS
    OR SIMDRNG_BUILD_PYTHON
    OR SIMDRNG_BUILD_EXAMPLES
    OR SIMDRNG_DEVEL
)
    include(cmake/dev_helpers.cmake)
endif()

# ---------------------------------------------------------------------------
# Dependencies.
#   xsimd : SIMD backend (optional).
#   poet  : compile-time helpers used by the SIMD headers and Python bindings.
# Pinned by SHA for reproducibility.
# ---------------------------------------------------------------------------
set(XSIMD_DOWNLOAD_VERSION 15603d4560a210018202ae7541c5b71e611e1eeb) # DiamonDinoia/xsimd fused-mul-hilo — fused mul_hilo for 64-bit batches (shared 32x32->64 partials); revert to xtensor-stack/xsimd once merged upstream (base: master 01f17ec)
set(POET_DOWNLOAD_VERSION 7562dbdab569739d526cc3174bea25f618ba74f7) # DiamonDinoia/poet v0.0.0

if(SIMDRNG_WITH_XSIMD)
    cpmaddpackage(
            NAME xsimd
            GITHUB_REPOSITORY DiamonDinoia/xsimd
            GIT_TAG ${XSIMD_DOWNLOAD_VERSION}
            GIT_SHALLOW NO
            GIT_PROGRESS YES
            EXCLUDE_FROM_ALL YES
    )
endif()

if(SIMDRNG_WITH_XSIMD OR SIMDRNG_BUILD_PYTHON)
    # poet is header-only (its version.hpp is committed). We fetch sources only
    # and define the interface target ourselves rather than add_subdirectory'ing
    # poet: poet's CMakeLists runs a GenerateVersion step that resolves the
    # source dir to the *consumer* whenever the consumer has its own top-level
    # VERSION file (which we now do), breaking the configure. DOWNLOAD_ONLY
    # avoids running poet's CMakeLists entirely.
    cpmaddpackage(
            NAME poet
            GITHUB_REPOSITORY DiamonDinoia/poet
            GIT_TAG ${POET_DOWNLOAD_VERSION}
            GIT_SHALLOW NO
            GIT_PROGRESS YES
            DOWNLOAD_ONLY YES
    )
    if(poet_ADDED AND NOT TARGET poet::poet)
        add_library(poet INTERFACE)
        add_library(poet::poet ALIAS poet)
        target_include_directories(
            poet
            INTERFACE $<BUILD_INTERFACE:${poet_SOURCE_DIR}/include>
        )
        target_compile_features(poet INTERFACE cxx_std_17)
    endif()
endif()

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckIPOSupported)
include(cmake/CompilerOptions.cmake)

check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)

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
# Library target.
#   With xsimd : a STATIC library carrying the runtime SIMD dispatch TUs.
#   Without    : a header-only INTERFACE library (scalar generators only).
# ---------------------------------------------------------------------------
if(SIMDRNG_WITH_XSIMD)
    add_library(simdrng STATIC src/xoshiro_simd.cpp)
    add_library(simdrng::simdrng ALIAS simdrng)
    target_include_directories(
        simdrng
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(simdrng PUBLIC SIMDRNG_WITH_XSIMD=1)
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
    target_compile_features(simdrng PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
    target_compile_features(simdrng INTERFACE cxx_std_20)
else()
    add_library(simdrng INTERFACE)
    add_library(simdrng::simdrng ALIAS simdrng)
    target_include_directories(
        simdrng
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_compile_definitions(simdrng INTERFACE SIMDRNG_WITH_XSIMD=0)
    target_compile_features(simdrng INTERFACE cxx_std_20)
endif()

# ---------------------------------------------------------------------------
# Architecture-specific SIMD dispatch variants (xsimd only).
# Each variant is compiled with flags that enable a specific SIMD level;
# xsimd::dispatch picks the best available at runtime.
# ---------------------------------------------------------------------------
if(SIMDRNG_WITH_XSIMD)
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

    foreach(SIMDRNG_TYPE IN ITEMS xoshiro chacha philox)
        foreach(VARIANT IN LISTS SIMDRNG_DISPATCH_VARIANTS)
            string(REPLACE "-" "_" TARGET_SUFFIX "${VARIANT}")
            set(SIMD_SOURCE_TARGET
                "${SIMDRNG_TYPE}_simd_source_${TARGET_SUFFIX}"
            )

            add_library(
                ${SIMD_SOURCE_TARGET}
                OBJECT
                src/${SIMDRNG_TYPE}_simd_dispatch.cpp
            )
            target_include_directories(
                ${SIMD_SOURCE_TARGET}
                PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
            )
            target_compile_definitions(
                ${SIMD_SOURCE_TARGET}
                PRIVATE SIMDRNG_WITH_XSIMD=1
            )
            target_link_libraries(
                ${SIMD_SOURCE_TARGET}
                PRIVATE xsimd poet::poet
            )
            target_compile_options(
                ${SIMD_SOURCE_TARGET}
                PRIVATE ${SIMDRNG_DISPATCH_FLAGS_${TARGET_SUFFIX}}
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
            target_sources(
                simdrng
                PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>
            )
        endforeach()
    endforeach()
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

# ---------------------------------------------------------------------------
# Installation (always available, independent of Python).
# ---------------------------------------------------------------------------
install(
    TARGETS simdrng
    EXPORT simdrngTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.hpp")

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
