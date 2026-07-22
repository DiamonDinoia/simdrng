# bindings/fortran — a thin iso_c_binding module (simdrng.f90) plus a
# self-checking test, linking the C ABI (simdrng_c_static). A missing Fortran
# compiler degrades to a STATUS message and no target. Added by the top-level
# CMakeLists when SIMDRNG_BUILD_FORTRAN=ON.
#
# COMPILER SELECTION FOOTGUN: check_language/enable_language picks whichever
# Fortran compiler is first in PATH. On RHEL 8 the system default is gfortran 8.5;
# this binding uses `error stop <integer-expression>` (Fortran 2018), which
# gfortran 8 miscompiles silently. Pass a modern compiler explicitly:
#
#   cmake ... -DCMAKE_Fortran_COMPILER=$(which gfortran)   # after module load gcc/13
#
# or set FC before configuring. Verify: gfortran --version >= 10 (>= 13 preferred).

include(CheckLanguage)
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
    message(STATUS "simdrng[fortran]: no Fortran compiler found — skipping.")
    return()
endif()

enable_language(Fortran)

# The test ends with `error stop <count>`, whose non-constant stop-code form is a
# Fortran 2018 feature; declare 2018 so violations diagnose correctly.
set(CMAKE_Fortran_STANDARD 2018)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Compile the binding module once into a small static library; linking it
# propagates its .mod directory to consumers. (Compiling simdrng.f90 in two
# targets would make Ninja's Fortran dyndep see two rules for simdrng.mod.)
add_library(simdrng_fortran STATIC "${CMAKE_CURRENT_SOURCE_DIR}/simdrng.f90")
target_link_libraries(simdrng_fortran PUBLIC simdrng_c_static)

add_executable(
    simdrng_fortran_test
    "${CMAKE_CURRENT_SOURCE_DIR}/test_simdrng.f90"
)
target_link_libraries(simdrng_fortran_test PRIVATE simdrng_fortran)

add_test(NAME fortran_simdrng COMMAND simdrng_fortran_test)
message(
    STATUS
    "simdrng[fortran]: test 'fortran_simdrng' registered (${CMAKE_Fortran_COMPILER})."
)
