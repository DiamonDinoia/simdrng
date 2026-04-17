cmake_minimum_required(VERSION 3.16)

project(random VERSION 0.1 LANGUAGES CXX)

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
include(cmake/CompilerOptions.cmake)

# List candidate standards in descending order.
set(CXX_STANDARDS "26" "23" "20" "17")
set(SUPPORTED_CXX_STANDARD "")

foreach (std IN LISTS CXX_STANDARDS)
    set(flag "-std=c++${std}")
    # Create a unique variable name for each check.
    string(CONCAT var "COMPILER_SUPPORTS_CXX" std)
    check_cxx_compiler_flag(${flag} ${var})
    if (${var})
        set(SUPPORTED_CXX_STANDARD ${std})
        break()  # Use the highest supported standard.
    endif ()
endforeach ()

if (NOT SUPPORTED_CXX_STANDARD)
    message(FATAL_ERROR "The compiler does not support C++17 or higher.")
endif ()

# Get performance flags
get_performance_flags(PERF_COMPILE_FLAGS PERF_LINK_FLAGS)

# --- wire into your existing lists (only for optimized configs) ---
list(APPEND COMPILE_OPTIONS
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${PERF_COMPILE_FLAGS}>
)

set(LINK_OPTIONS "")
list(APPEND LINK_OPTIONS
        $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:${PERF_LINK_FLAGS}>
)

add_library(random STATIC src/xoshiro_simd.cpp)
target_include_directories(random PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(random PUBLIC xsimd poet::poet)
set_target_properties(random PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
    INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON
    INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
# Suppress GCC 15 -Wtemplate-body: xsimd template definitions reference
# SIMD intrinsics that don't exist without a SIMD arch, but these templates
# are only instantiated in dispatch TUs compiled with the correct -march flags.
add_compile_flags_if_supported(TEMPLATE_BODY_FLAGS -Wno-template-body)
target_compile_options(random PRIVATE ${COMPILE_OPTIONS})
target_compile_options(random PUBLIC ${TEMPLATE_BODY_FLAGS})
target_link_options(random PRIVATE ${LINK_OPTIONS})
target_compile_features(random PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
target_compile_features(random INTERFACE cxx_std_20)

# Architecture-specific SIMD dispatch variants.
# Each variant is compiled with flags that enable a specific SIMD level;
# xsimd::dispatch picks the best available at runtime.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(PRNG_DISPATCH_VARIANTS "x86-64;x86-64-v3;x86-64-v4")
    set(PRNG_DISPATCH_FLAGS_x86_64     -march=x86-64   -mtune=generic)
    set(PRNG_DISPATCH_FLAGS_x86_64_v3  -march=x86-64-v3 -mtune=generic)
    set(PRNG_DISPATCH_FLAGS_x86_64_v4  -march=x86-64-v4 -mtune=generic)
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
        target_compile_options(${SIMD_SOURCE_TARGET} PRIVATE ${COMPILE_OPTIONS} ${PRNG_DISPATCH_FLAGS_${TARGET_SUFFIX}})
        set_target_properties(${SIMD_SOURCE_TARGET} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON
            INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON
        )
        target_link_options(${SIMD_SOURCE_TARGET} PRIVATE ${LINK_OPTIONS})
        target_compile_features(${SIMD_SOURCE_TARGET} PRIVATE cxx_std_${SUPPORTED_CXX_STANDARD})
        target_sources(random PRIVATE $<TARGET_OBJECTS:${SIMD_SOURCE_TARGET}>)
    endforeach ()
endforeach ()

if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()


if (ENABLE_PYTHON)
    add_subdirectory(python)
endif()

# Installation rules (Always available, independent of Python)
install(TARGETS random
        EXPORT randomTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT randomTargets
        FILE randomTargets.cmake
        NAMESPACE prng::
        DESTINATION lib/cmake/random
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/randomConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_file(cmake/randomConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/randomConfig.cmake"
        @ONLY
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/randomConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/randomConfigVersion.cmake"
        DESTINATION lib/cmake/random
)
