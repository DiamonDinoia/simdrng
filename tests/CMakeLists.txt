CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.12.0
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)

CPMAddPackage(
        NAME nanobench
        GITHUB_REPOSITORY martinus/nanobench
        VERSION 4.3.11
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)
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
target_link_libraries(testSplitMix PRIVATE random Catch2::Catch2WithMain)
target_include_directories(testSplitMix PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testSplitMix COMMAND testSplitMix)

add_executable(testXoshiro test_xoshiro.cpp)
target_link_libraries(testXoshiro PRIVATE random Catch2::Catch2WithMain)
target_include_directories(testXoshiro PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME testXoshiro COMMAND testXoshiro)

add_executable(testXoshiroSIMD test_xoshiro_simd.cpp)
target_link_libraries(testXoshiroSIMD PRIVATE random Catch2::Catch2WithMain)
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
target_link_libraries(testChaCha PRIVATE random monocypher Catch2::Catch2WithMain)
add_test(NAME testChaCha COMMAND testChaCha)

add_executable(testChaChaSIMD test_chacha_simd.cpp)
target_link_libraries(testChaChaSIMD PRIVATE random Catch2::Catch2WithMain)
target_include_directories(testChaChaSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
add_test(NAME testChaChaSIMD COMMAND testChaChaSIMD)

add_executable(testPhilox test_philox.cpp)
target_link_libraries(testPhilox PRIVATE random Catch2::Catch2WithMain)
add_test(NAME testPhilox COMMAND testPhilox)

add_executable(testPhiloxSIMD test_philox_simd.cpp)
target_link_libraries(testPhiloxSIMD PRIVATE random Catch2::Catch2WithMain)
target_include_directories(testPhiloxSIMD PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
add_test(NAME testPhiloxSIMD COMMAND testPhiloxSIMD)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE random nanobench)
target_include_directories(benchmarks PRIVATE ${TEST_INCLUDE_DIR} xsimd::xsimd)
target_compile_options(benchmarks PRIVATE ${COMPILE_OPTIONS})
target_link_options(benchmarks PRIVATE ${LINK_OPTIONS})

if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()

# Add a custom target to run the benchmarks
add_custom_target(benchmark
    COMMAND benchmarks
    DEPENDS benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks..."
    VERBATIM
)


if (MARCH_NATIVE)
    target_compile_options(benchmarks PRIVATE -march=native)
endif ()
