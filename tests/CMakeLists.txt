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
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ENABLE_CODSPEED)
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

if (ENABLE_CODSPEED)
    # codspeed-cpp ships its own fork of google/benchmark under google_benchmark/.
    # The fork's CMakeLists.txt unconditionally pulls in codspeed-cpp's `core/`,
    # which defines an option(ENABLE_TESTS) that clashes with our own. Shield it
    # via a local ENABLE_TESTS=OFF (CMP0077 NEW -> option() respects normal vars).
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
set(TEST_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

enable_language(C)

# Download reference implementations here. The ChaCha reference implementation
# requires multiple files because it transitively includes macros from them
# (see https://cr.yp.to/chacha.html). Although the linked page links to ChaCha8
# implementations, we download ChaCha20 versions instead. The files
# ecrypt-{portable,config,machine}.h are taken from the Salsa20 reference
# implementation rather than ChaCha20, since no ChaCha20 versions of these
# appear to exist. Since ChaCha20 is a subdir in the Salsa20 directory in
# EBACS, and given ChaCha20 is structured very similar to Salsa20, I can only
# assume these header files are also the ones used for ChaCha20.
file(MAKE_DIRECTORY ${TEST_INCLUDE_DIR})
file(DOWNLOAD https://prng.di.unimi.it/splitmix64.c ${TEST_INCLUDE_DIR}/splitmix64.c)
file(DOWNLOAD https://prng.di.unimi.it/xoshiro256plusplus.c ${TEST_INCLUDE_DIR}/xoshiro256plusplus.c)

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

add_executable(testSplitMix splitmix_tests.cpp)
target_link_libraries(testSplitMix PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testSplitMix PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testSplitMix COMMAND testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testXoshiro PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testXoshiro COMMAND testXoshiro)

add_executable(testXoshiroSIMD test_xoshiro_simd.cpp)
target_link_libraries(testXoshiroSIMD PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testXoshiroSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
add_test(NAME testXoshiroSIMD COMMAND testXoshiroSIMD)

# Use Monocypher for ChaCha20 and link it into the chacha test.
# monocypher is fetched above via CPM; create a target if the package didn't.
if (NOT TARGET monocypher)
    add_library(monocypher STATIC ${monocypher_SOURCE_DIR}/src/monocypher.c)
    target_include_directories(monocypher PUBLIC ${monocypher_SOURCE_DIR}/src)
    set_target_properties(monocypher PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_executable(testChaCha test_chacha.cpp)
target_link_libraries(testChaCha PRIVATE simdrng monocypher Catch2::Catch2WithMain)
add_test(NAME testChaCha COMMAND testChaCha)

add_executable(testChaChaSIMD test_chacha_simd.cpp)
target_link_libraries(testChaChaSIMD PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testChaChaSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
add_test(NAME testChaChaSIMD COMMAND testChaChaSIMD)

add_executable(testPhilox test_philox.cpp)
target_link_libraries(testPhilox PRIVATE simdrng Catch2::Catch2WithMain)
add_test(NAME testPhilox COMMAND testPhilox)

add_executable(testPhiloxSIMD test_philox_simd.cpp)
target_link_libraries(testPhiloxSIMD PRIVATE simdrng Catch2::Catch2WithMain)
target_include_directories(testPhiloxSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
add_test(NAME testPhiloxSIMD COMMAND testPhiloxSIMD)

add_executable(benchmarks benchmarks.cpp)
target_include_directories(benchmarks PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
target_link_libraries(benchmarks PRIVATE simdrng benchmark::benchmark_main)
if (ENABLE_CODSPEED)
    target_compile_definitions(benchmarks PRIVATE SIMDRNG_CODSPEED=1)
endif()

if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()

# Convenience target to run the benchmarks with hardware counters enabled.
add_custom_target(run_benchmarks
    COMMAND benchmarks
            --benchmark_perf_counters=CYCLES,INSTRUCTIONS,CACHE-MISSES,BRANCH-MISSES,BRANCHES
    DEPENDS benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
    VERBATIM
)
