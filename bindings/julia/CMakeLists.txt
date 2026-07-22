# bindings/julia — no build step; the package dlopen()s libsimdrng_c at test
# time. We just point it at the freshly-built shared library via LIBSIMDRNG_C.
# Added by the top-level CMakeLists when SIMDRNG_BUILD_JULIA=ON.

find_program(SIMDRNG_JULIA_EXECUTABLE julia)
if(NOT SIMDRNG_JULIA_EXECUTABLE)
    message(STATUS "simdrng[julia]: julia not on PATH — skipping.")
    return()
endif()

add_test(
    NAME julia_simdrng
    COMMAND
        "${SIMDRNG_JULIA_EXECUTABLE}"
        "--project=${CMAKE_CURRENT_SOURCE_DIR}/SimdRNG" -e
        "import Pkg; Pkg.test(coverage = false)"
)
set_tests_properties(
    julia_simdrng
    PROPERTIES ENVIRONMENT "LIBSIMDRNG_C=$<TARGET_FILE:simdrng_c>"
)
message(STATUS "simdrng[julia]: test 'julia_simdrng' registered.")
