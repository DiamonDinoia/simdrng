CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.12.0
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)

# Google Benchmark. Linux builds enable libpfm for hardware counters
# (IPC / cache-miss / branch-miss) to match the columns we previously
# surfaced via nanobench.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT SIMDRNG_ENABLE_CODSPEED)
    # Detect libpfm4 (header + .so). When both are present, enable hardware
    # counters in Google Benchmark; otherwise fall back silently so local
    # builds without libpfm4-dev still work.
    find_path(PFM_INCLUDE_DIR perfmon/pfmlib.h)
    find_library(PFM_LIBRARY pfm)
    if (PFM_INCLUDE_DIR AND PFM_LIBRARY)
        set(BENCHMARK_ENABLE_LIBPFM ON CACHE BOOL "" FORCE)
        message(STATUS "libpfm4 found; Google Benchmark hardware counters enabled")
    else()
        message(STATUS "libpfm4 not found; hardware counters disabled (install libpfm4-dev)")
    endif()
endif()

if (SIMDRNG_ENABLE_CODSPEED)
    # codspeed-cpp ships its own fork of google/benchmark under google_benchmark/.
    # The fork's CMakeLists.txt unconditionally pulls in codspeed-cpp's `core/`,
    # which defines an option(ENABLE_TESTS) that clashes with CMake's BUILD_TESTING
    # convention. Shield it via a local ENABLE_TESTS=OFF (CMP0077 NEW -> option()
    # respects normal vars).
    CPMAddPackage(
            NAME codspeed
            GITHUB_REPOSITORY CodSpeedHQ/codspeed-cpp
            GIT_TAG main
            GIT_SHALLOW YES
            GIT_PROGRESS YES
            DOWNLOAD_ONLY YES
    )
    set(ENABLE_TESTS OFF)
    add_subdirectory(${codspeed_SOURCE_DIR}/google_benchmark
                     ${CMAKE_BINARY_DIR}/_deps/codspeed-benchmark-build
                     EXCLUDE_FROM_ALL)
    set(ENABLE_TESTS ON)
else()
    CPMAddPackage(
            NAME benchmark
            GITHUB_REPOSITORY google/benchmark
            VERSION 1.9.1
            GIT_SHALLOW YES
            GIT_PROGRESS YES
            EXCLUDE_FROM_ALL YES
            SYSTEM
    )
endif()
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/contrib)

enable_language(C)

# ---------------------------------------------------------------------------
# Reference generators from https://prng.di.unimi.it/ (public domain / CC0).
# Used by the SplitMix/Xoshiro tests and benchmarks as a ground-truth oracle.
#
# Cache them in the CPM source cache when one is configured (so CI restores
# them alongside xsimd/poet/Catch2/...), else in a stable build-tree dir. The
# download is STATUS-checked with a short retry and is a HARD ERROR on failure:
# a silent empty include previously surfaced as `s`/`next` undeclared in the
# benchmark build (the original CodSpeed failure). ChaCha uses Monocypher
# (fetched via CPM below), not a downloaded reference.
# ---------------------------------------------------------------------------
if (DEFINED ENV{CPM_SOURCE_CACHE})
    set(TEST_INCLUDE_DIR "$ENV{CPM_SOURCE_CACHE}/simdrng-reference")
else ()
    set(TEST_INCLUDE_DIR "${CMAKE_BINARY_DIR}/_refcache")
endif ()
file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})

function(simdrng_fetch_reference url dest)
    if (EXISTS "${dest}")
        return()
    endif ()
    set(_msg "no attempt made")
    foreach (attempt RANGE 1 3)
        file(DOWNLOAD "${url}" "${dest}" STATUS _st TLS_VERIFY ON)
        list(GET _st 0 _code)
        list(GET _st 1 _msg)
        if (_code EQUAL 0)
            return()
        endif ()
        message(STATUS "Reference download attempt ${attempt}/3 for ${url} failed: ${_msg}")
        file(REMOVE "${dest}")
    endforeach ()
    message(FATAL_ERROR "Failed to download reference file ${url} after 3 attempts: ${_msg}")
endfunction()

simdrng_fetch_reference(https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)
simdrng_fetch_reference(https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)

# Use Monocypher for ChaCha20 (fetched via CPM)
CPMAddPackage(
    NAME monocypher
    GITHUB_REPOSITORY LoupVaillant/Monocypher
    GIT_TAG master
    GIT_SHALLOW YES
    GIT_PROGRESS YES
    EXCLUDE_FROM_ALL YES
)
if (NOT TARGET monocypher)
    add_library(monocypher STATIC ${monocypher_SOURCE_DIR}/src/monocypher.c)
    target_include_directories(monocypher PUBLIC ${monocypher_SOURCE_DIR}/src)
    set_target_properties(monocypher PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

include(CTest)
enable_testing()

# Scalar tests build with or without xsimd.
add_executable(testSplitMix splitmix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testSplitMix PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testSplitMix COMMAND testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testXoshiro PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testXoshiro COMMAND testXoshiro)

add_executable(testChaCha test_chacha.cpp)
target_link_libraries(testChaCha PRIVATE simdrng monocypher Catch2::Catch2WithMain)
add_test(NAME testChaCha COMMAND testChaCha)

add_executable(testPhilox test_philox.cpp)
target_link_libraries(testPhilox PRIVATE simdrng Catch2::Catch2WithMain)
add_test(NAME testPhilox COMMAND testPhilox)

set(SIMDRNG_SCALAR_TEST_TARGETS testSplitMix testXoshiro testChaCha testPhilox)
foreach (_t IN LISTS SIMDRNG_SCALAR_TEST_TARGETS)
    simdrng_enable_warnings(${_t})
endforeach ()

# SIMD tests and benchmarks require the xsimd backend.
if (SIMDRNG_WITH_XSIMD)
    add_executable(testXoshiroSIMD test_xoshiro_simd.cpp)
    target_link_libraries(testXoshiroSIMD PRIVATE simdrng Catch2::Catch2WithMain)
    target_include_directories(testXoshiroSIMD PRIVATE ${TEST_INCLUDE_DIR})
    target_link_libraries(testXoshiroSIMD PRIVATE xsimd)
    add_test(NAME testXoshiroSIMD COMMAND testXoshiroSIMD)

    add_executable(testChaChaSIMD test_chacha_simd.cpp)
    target_link_libraries(testChaChaSIMD PRIVATE simdrng xsimd Catch2::Catch2WithMain)
    target_include_directories(testChaChaSIMD PRIVATE ${TEST_INCLUDE_DIR})
    add_test(NAME testChaChaSIMD COMMAND testChaChaSIMD)

    add_executable(testPhiloxSIMD test_philox_simd.cpp)
    target_link_libraries(testPhiloxSIMD PRIVATE simdrng xsimd Catch2::Catch2WithMain)
    target_include_directories(testPhiloxSIMD PRIVATE ${TEST_INCLUDE_DIR})
    add_test(NAME testPhiloxSIMD COMMAND testPhiloxSIMD)

    add_executable(benchmarks benchmarks.cpp)
    target_include_directories(benchmarks PRIVATE ${TEST_INCLUDE_DIR})
    target_link_libraries(benchmarks PRIVATE simdrng xsimd benchmark::benchmark_main)
    if (SIMDRNG_ENABLE_CODSPEED)
        target_compile_definitions(benchmarks PRIVATE SIMDRNG_CODSPEED=1)
    endif()

    if (SIMDRNG_MARCH_NATIVE)
        target_compile_options(benchmarks PRIVATE -march=native)
    endif ()

    foreach (_t IN ITEMS testXoshiroSIMD testChaChaSIMD testPhiloxSIMD benchmarks)
        simdrng_enable_warnings(${_t})
    endforeach ()

    # Convenience target to run the benchmarks with hardware counters enabled.
    add_custom_target(run_benchmarks
        COMMAND benchmarks
                --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES
        DEPENDS benchmarks
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running benchmarks..."
        VERBATIM
    )
endif ()
