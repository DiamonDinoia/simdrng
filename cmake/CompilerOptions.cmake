include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckIPOSupported)

function(add_compile_flags_if_supported OUTVAR)
    foreach (flag ${ARGN})
        string(MAKE_C_IDENTIFIER "HAS_${flag}" var)
        check_cxx_compiler_flag("${flag}" ${var})
        if (${var})
            list(APPEND ${OUTVAR} "${flag}")
        endif ()
    endforeach ()
    set(${OUTVAR} "${${OUTVAR}}" PARENT_SCOPE)
endfunction()

function(add_link_flags_if_supported OUTVAR)
    foreach (flag ${ARGN})
        string(MAKE_C_IDENTIFIER "HAS_LNK_${flag}" var)
        check_linker_flag(CXX "${flag}" ${var})
        if (${var})
            list(APPEND ${OUTVAR} "${flag}")
        endif ()
    endforeach ()
    set(${OUTVAR} "${${OUTVAR}}" PARENT_SCOPE)
endfunction()

function(get_performance_flags COMPILE_FLAGS_VAR LINK_FLAGS_VAR)
    set(PERF_COMPILE_FLAGS "")
    set(PERF_LINK_FLAGS "")

    # LTO / IPO
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        # We will handle IPO via target property INTERPROCEDURAL_OPTIMIZATION in the main CMakeLists.txt
        # But we can still check for specific flags if needed, though CMake handles it mostly.
        # For now, let's stick to the manual flags if we want specific control (like thin vs full),
        # but modern CMake prefers the property.
        # However, the user's existing code had specific checks for thin/full.
        # Let's try to use the property where possible, but for flags like -fno-plt, we keep them.
    endif()

    # MSVC compile-side LTCG
    add_compile_flags_if_supported(PERF_COMPILE_FLAGS /GL)

    # Devirtualization helpers + DCE/ICF enablers
    add_compile_flags_if_supported(PERF_COMPILE_FLAGS
            -fno-semantic-interposition
            -fwhole-program-vtables
            -fstrict-vtable-pointers
            -fipa-pta
            -fdevirtualize-at-ltrans
            -ffunction-sections -fdata-sections
            -funroll-loops
            -fpeel-loops
            -funswitch-loops
            -Ofast
            -fimplicit-constexpr
    )

    # ELF niceties
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_compile_flags_if_supported(PERF_COMPILE_FLAGS -fno-plt -fuse-ld=lld)
    endif ()

    # Link-time flags
    add_link_flags_if_supported(PERF_LINK_FLAGS
            -Wl,--gc-sections
            -Wl,--icf=all
            -Wl,-dead_strip
            /LTCG /OPT:ICF /OPT:REF /INCREMENTAL:NO
    )
    
    set(${COMPILE_FLAGS_VAR} "${PERF_COMPILE_FLAGS}" PARENT_SCOPE)
    set(${LINK_FLAGS_VAR} "${PERF_LINK_FLAGS}" PARENT_SCOPE)
endfunction()
