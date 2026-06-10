# C++ examples. Each example is also registered as a ctest target so CI
# verifies they build and run.

find_package(OpenMP QUIET)

# Examples that only use the portable `Xoshiro` umbrella alias build in both
# scalar-only and SIMD configurations.
set(SIMDRNG_SCALAR_EXAMPLES
    basic_xoshiro
    threaded_openmp
)

# Examples that use the SIMD generators directly require the xsimd backend.
set(SIMDRNG_SIMD_EXAMPLES
    simd_bulk_fill
    xoshiro_native
    philox_counter_based
)

set(SIMDRNG_EXAMPLES ${SIMDRNG_SCALAR_EXAMPLES})
if (SIMDRNG_WITH_XSIMD)
    list(APPEND SIMDRNG_EXAMPLES ${SIMDRNG_SIMD_EXAMPLES})
endif ()

foreach(example IN LISTS SIMDRNG_EXAMPLES)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE simdrng)
    target_compile_features(${example} PRIVATE cxx_std_20)
    set_target_properties(${example} PROPERTIES FOLDER "examples")
    simdrng_enable_warnings(${example})

    if(example STREQUAL "xoshiro_native")
        target_compile_options(${example} PRIVATE -march=native)
    endif()

    if(example STREQUAL "threaded_openmp" AND OpenMP_CXX_FOUND)
        target_link_libraries(${example} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME example_${example} COMMAND ${example})
endforeach()
