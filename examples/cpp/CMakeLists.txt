# C++ examples. Each example is also registered as a ctest target so CI
# verifies they build and run.

set(SIMDRNG_EXAMPLES
    basic_xoshiro
    simd_bulk_fill
    xoshiro_native
    threaded_openmp
    philox_counter_based
)

find_package(OpenMP QUIET)

foreach(example IN LISTS SIMDRNG_EXAMPLES)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE simdrng)
    target_compile_features(${example} PRIVATE cxx_std_20)
    set_target_properties(${example} PROPERTIES FOLDER "examples")

    if(example STREQUAL "xoshiro_native")
        target_compile_options(${example} PRIVATE -march=native)
    endif()

    if(example STREQUAL "threaded_openmp" AND OpenMP_CXX_FOUND)
        target_link_libraries(${example} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(NAME example_${example} COMMAND ${example})
endforeach()
