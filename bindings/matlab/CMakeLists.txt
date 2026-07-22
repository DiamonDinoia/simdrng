# bindings/matlab — MATLAB/Octave MEX binding. Pregenerated gateway by default
# (SIMDRNG_MATLAB_USE_PREGENERATED=ON); set OFF to fetch+run mwrap via CPM.
# Backend auto-detected: mkoctfile (license-free) or MATLAB; both if present.

find_program(SIMDRNG_MKOCTFILE_EXECUTABLE mkoctfile)
find_program(SIMDRNG_OCTAVE_EXECUTABLE NAMES octave-cli octave)
find_package(Matlab QUIET COMPONENTS MAIN_PROGRAM MEX_COMPILER MX_LIBRARY)

if(NOT SIMDRNG_MKOCTFILE_EXECUTABLE AND NOT Matlab_FOUND)
    message(
        STATUS
        "simdrng[matlab]: neither mkoctfile (Octave) nor MATLAB found — skipping."
    )
    return()
endif()

option(
    SIMDRNG_MATLAB_USE_PREGENERATED
    "Compile the checked-in mwrap gateway (generated/) instead of running mwrap"
    ON
)
# Static-link libstdc++/libgcc into the GNU MATLAB MEX (portable shipped binary);
# OFF links them dynamically (smaller, but needs a compatible system libstdc++).
option(
    SIMDRNG_MATLAB_STATIC_LIBSTDCXX
    "Static-link the C++ runtime into the GNU MATLAB MEX (portable .mexa64)"
    ON
)
set(_simdrng_inc "${PROJECT_SOURCE_DIR}/include")

if(SIMDRNG_MATLAB_USE_PREGENERATED)
    # Stage sr_*.m into build dir so a single addpath imports the binding;
    # source tree never gets a compiled MEX written into it.
    set(_simdrng_mex_src
        "${CMAKE_CURRENT_SOURCE_DIR}/generated/simdrng_mex_gen.cpp"
    )
    set(_simdrng_gen_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(
        COPY "${CMAKE_CURRENT_SOURCE_DIR}/generated/"
        DESTINATION "${_simdrng_gen_dir}"
        FILES_MATCHING
        PATTERN "sr_*.m"
    )
    set(_simdrng_mw_dep "") # no generator target
else()
    # Regen path: MWRAP_COMPILE_MEX OFF so we drive compile/link ourselves.
    cpmaddpackage(
        NAME mwrap
        GITHUB_REPOSITORY zgimbutas/mwrap
        GIT_TAG a53713c34b09b74221ff7220c77ee1e12583804d
        OPTIONS "MWRAP_BUILD_EXAMPLES OFF" "MWRAP_COMPILE_MEX OFF"
    )
    if(NOT TARGET mwrap)
        message(
            WARNING
            "simdrng[matlab]: mwrap generator unavailable after CPM — skipping."
        )
        return()
    endif()

    # mwrap's flex-generated scanner #includes <unistd.h>, which MSVC lacks.
    if(MSVC)
        target_compile_definitions(mwrap PRIVATE YY_NO_UNISTD_H)
    endif()

    include("${mwrap_SOURCE_DIR}/cmake/MwrapAddMex.cmake")

    mwrap_add_mex(
        simdrng_mw
        MEX_NAME simdrng_mex
        CC_FILENAME simdrng_mex_gen.cpp
        MW_FILES "${CMAKE_CURRENT_SOURCE_DIR}/simdrng.mw"
        MWRAP_FLAGS -mb
        OUTPUT_VAR _simdrng_mex_src
    )
    get_target_property(_simdrng_gen_dir simdrng_mw MWRAP_OUTPUT_DIRECTORY)
    set(_simdrng_mw_dep simdrng_mw)

    # Refresh the checked-in copy from mwrap's fresh output:
    #   cmake --build <dir> --target simdrng_mw_regen
    # then `git diff bindings/matlab/generated/` shows the .mw edit's effect.
    set(_simdrng_regen_script "${CMAKE_CURRENT_BINARY_DIR}/regen_copy.cmake")
    file(
        WRITE "${_simdrng_regen_script}"
        "file(COPY \"${_simdrng_gen_dir}/\" DESTINATION \"${CMAKE_CURRENT_SOURCE_DIR}/generated\" FILES_MATCHING PATTERN \"sr_*.m\")\nfile(COPY \"${_simdrng_mex_src}\" DESTINATION \"${CMAKE_CURRENT_SOURCE_DIR}/generated\")\n"
    )
    add_custom_target(
        simdrng_mw_regen
        COMMAND ${CMAKE_COMMAND} -P "${_simdrng_regen_script}"
        DEPENDS simdrng_mw
        COMMENT
            "simdrng[matlab]: refreshing checked-in mwrap output in generated/"
        VERBATIM
    )
endif()

if(Matlab_FOUND)
    matlab_add_mex(
        NAME simdrng_mex_matlab
        SRC "${_simdrng_mex_src}"
        OUTPUT_NAME simdrng_mex
    )
    if(_simdrng_mw_dep)
        add_dependencies(simdrng_mex_matlab ${_simdrng_mw_dep})
    endif()
    target_include_directories(simdrng_mex_matlab PRIVATE "${_simdrng_inc}")
    target_compile_definitions(simdrng_mex_matlab PRIVATE R2008OO)
    # matlab_add_mex() uses the plain TLL signature; must match to avoid CMake error.
    target_link_libraries(simdrng_mex_matlab simdrng_c_static)
    # -static-libstdc++/-static-libgcc: .mexa64 must not need a newer GLIBCXX than
    # MATLAB's bundled libstdc++. Standard portable-MEX fix.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SIMDRNG_MATLAB_STATIC_LIBSTDCXX)
        target_link_options(
            simdrng_mex_matlab
            PRIVATE -static-libstdc++ -static-libgcc
        )
    endif()
    set_target_properties(
        simdrng_mex_matlab
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${_simdrng_gen_dir}"
            RUNTIME_OUTPUT_DIRECTORY "${_simdrng_gen_dir}"
    )
endif()

# octave_compat/ shim satisfies mwrap's #include <matrix.h> (MATLAB-only header).
if(SIMDRNG_MKOCTFILE_EXECUTABLE)
    set(_simdrng_oct_mex "${_simdrng_gen_dir}/simdrng_mex.mex")
    # The C ABI glue (simdrng_c_static) references the runtime-dispatched SIMD
    # kernels, which are archived in libsimdrng. mkoctfile takes plain archive
    # paths (no CMake transitive propagation), so pass both, glue first so the
    # linker resolves capi.cpp's references against libsimdrng after it.
    add_custom_command(
        OUTPUT "${_simdrng_oct_mex}"
        COMMAND
            "${SIMDRNG_MKOCTFILE_EXECUTABLE}" --mex -DR2008OO
            "-I${_simdrng_inc}" "-I${CMAKE_CURRENT_SOURCE_DIR}/octave_compat"
            "${_simdrng_mex_src}" "$<TARGET_FILE:simdrng_c_static>"
            "$<TARGET_FILE:simdrng>" -o "${_simdrng_oct_mex}"
        DEPENDS
            ${_simdrng_mw_dep}
            simdrng_c_static
            simdrng
            "${_simdrng_mex_src}"
        WORKING_DIRECTORY "${_simdrng_gen_dir}"
        COMMENT "simdrng[matlab]: building Octave MEX simdrng_mex.mex"
        VERBATIM
    )
    add_custom_target(simdrng_mex_octave ALL DEPENDS "${_simdrng_oct_mex}")
endif()

if(SIMDRNG_MKOCTFILE_EXECUTABLE AND SIMDRNG_OCTAVE_EXECUTABLE)
    add_test(
        NAME matlab_simdrng
        COMMAND
            "${SIMDRNG_OCTAVE_EXECUTABLE}" --no-gui --quiet --eval
            "addpath('${CMAKE_CURRENT_SOURCE_DIR}'); addpath('${_simdrng_gen_dir}'); test_simdrng"
    )
    message(
        STATUS
        "simdrng[matlab]: test 'matlab_simdrng' registered (Octave: ${SIMDRNG_OCTAVE_EXECUTABLE})."
    )
elseif(Matlab_FOUND AND Matlab_MAIN_PROGRAM)
    add_test(
        NAME matlab_simdrng
        COMMAND
            "${Matlab_MAIN_PROGRAM}" -batch
            "addpath('${CMAKE_CURRENT_SOURCE_DIR}'); addpath('${_simdrng_gen_dir}'); test_simdrng"
    )
    message(
        STATUS
        "simdrng[matlab]: test 'matlab_simdrng' registered (MATLAB: ${Matlab_MAIN_PROGRAM})."
    )
else()
    message(
        STATUS
        "simdrng[matlab]: MEX built, but no interpreter found to run the test."
    )
endif()
