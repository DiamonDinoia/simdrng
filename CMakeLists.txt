cmake_minimum_required(VERSION 3.29)

project(simdrng VERSION 0.0 LANGUAGES CXX)

# if no build type was specified, default to Release
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

set(CPM_DOWNLOAD_VERSION 0.42.0)
set(XSIMD_DOWNLOAD_VERSION feat/mulhilo)
set(POET_DOWNLOAD_VERSION 4e0c50b75224b9916cb8062a75d81ab7cc403df1)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_PYTHON "Enable Python bindings" ON)
option(MARCH_NATIVE "Build benchmarks with -march=native" OFF)
option(BUILD_EXAMPLES "Build C++ examples under examples/cpp" OFF)
option(BUILD_DOCS "Build Sphinx/Doxygen documentation" OFF)
option(ENABLE_CODSPEED "Link the benchmark harness against codspeed-cpp for CodSpeed instrumentation" OFF)

if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
    )
endif ()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY DiamonDinoia/xsimd
        GIT_TAG ${XSIMD_DOWNLOAD_VERSION}
        GIT_SHALLOW NO
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
)

CPMAddPackage(
        NAME poet
        GITHUB_REPOSITORY DiamonDinoia/poet
        GIT_TAG ${POET_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "POET_BUILD_TESTS OFF"
            "POET_BUILD_BENCHMARKS OFF"
            "POET_STRICT_WARNINGS OFF"
)

include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckIPOSupported)
include(cmake/CompilerOptions.cmake)

check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg LANGUAGES CXX)

if (NOT DEFINED CMAKE_LINKER_TYPE
    AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_LINKER_TYPE LLD)
endif ()

# List candidate standards in descending order.
# Use CMAKE_CXX_COMPILE_FEATURES (portable across MSVC/GCC/Clang) rather than
# probing raw -std=c++XX flags, which MSVC doesn't accept.
set(CXX_STANDARDS "26" "23" "20" "17")
set(SUPPORTED_CXX_STANDARD "")

foreach (std IN LISTS CXX_STANDARDS)
    if ("cxx_std_${std}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        set(SUPPORTED_CXX_STANDARD ${std})
        break()
    endif ()
endforeach ()

if (NOT SUPPORTED_CXX_STANDARD)
    message(FATAL_ERROR "The compiler does not support C++17 or higher.")
endif ()

add_library(simdrng STATIC src/xoshiro_simd.cpp)
add_library(simdrng::simdrng ALIAS simdrng)
target_include_directories(simdrng PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(simdrng PUBLIC xsimd poet::poet)
set_target_properties(simdrng PROPERTIES
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

# Architecture-specific SIMD dispatch variants.
# Each variant is compiled with flags that enable a specific SIMD level;
# xsimd::dispatch picks the best available at runtime.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(PRNG_DISPATCH_VARIANTS "x86-64;x86-64-v3;x86-64-v4")
    if (MSVC)
        # MSVC has no -march equivalent and doesn't define __SSE2__ even
        # though x64 is SSE2 baseline; define it explicitly so the dispatch
        # TUs pick the right template instantiation. MSVC does define
        # __AVX2__ under /arch:AVX2 and __AVX512F__ under /arch:AVX512.
        set(PRNG_DISPATCH_FLAGS_x86_64     /D__SSE2__)
        set(PRNG_DISPATCH_FLAGS_x86_64_v3  /arch:AVX2)
        set(PRNG_DISPATCH_FLAGS_x86_64_v4  /arch:AVX512)
    else ()
        set(PRNG_DISPATCH_FLAGS_x86_64     -march=x86-64   -mtune=generic)
        set(PRNG_DISPATCH_FLAGS_x86_64_v3  -march=x86-64-v3 -mtune=generic)
        set(PRNG_DISPATCH_FLAGS_x86_64_v4  -march=x86-64-v4 -mtune=generic)
    endif ()
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(PRNG_DISPATCH_VARIANTS "neon64")
    set(PRNG_DISPATCH_FLAGS_neon64 "")  # NEON is baseline on aarch64
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64$")
    set(PRNG_DISPATCH_VARIANTS "rvv")
    set(PRNG_DISPATCH_FLAGS_rvv -march=rv64gcv -mrvv-vector-bits=zvl)
else ()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

foreach (PRNG_TYPE IN ITEMS xoshiro chacha philox)
    foreach (VARIANT IN LISTS PRNG_DISPATCH_VARIANTS)
        string(REPLACE "-" "_" TARGET_SUFFIX "${VARIANT}")
        set(SIMD_SOURCE_TARGET "${PRNG_TYPE}_simd_source_${TARGET_SUFFIX}")

        add_library(${SIMD_SOURCE_TARGET} OBJECT src/${PRNG_TYPE}_simd_dispatch.cpp)
        target_include_directories(${SIMD_SOURCE_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(${SIMD_SOURCE_TARGET} PRIVATE xsimd poet::poet)
        target_compile_options(${SIMD_SOURCE_TARGET} PRIVATE ${PRNG_DISPATCH_FLAGS_${TARGET_SUFFIX}})
        set_target_properties(${SIMD_SOURCE_TARGET} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            INTERPROCEDURAL_OPTIMIZATION_RELEASE ${_ipo_ok}
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ${_ipo_ok}
            INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ${_ipo_ok}
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
        )
        target_compile_features(${SIMD_SOURCE_TARGET} PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
        target_sources(simdrng PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
    endforeach ()
endforeach ()

if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (BUILD_EXAMPLES)
    if (NOT ENABLE_TESTS)
        enable_testing()
    endif()
    add_subdirectory(examples/cpp)
endif()

if (BUILD_DOCS)
    add_subdirectory(docs)
endif()

if (ENABLE_PYTHON)
    add_subdirectory(python)
endif()

# Installation rules (Always available, independent of Python)
install(TARGETS simdrng
        EXPORT simdrngTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT simdrngTargets
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

configure_file(cmake/simdrngConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfig.cmake"
        @ONLY
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/simdrngConfigVersion.cmake"
        DESTINATION lib/cmake/simdrng
)
