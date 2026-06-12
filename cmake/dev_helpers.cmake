include_guard(GLOBAL)

# ==============================================================================
# simdrng development helpers
# ==============================================================================
# Pulled in only for developer / test / benchmark / example / dependency-fetch
# builds. Provides:
#   - CPM bootstrap (used to fetch xsimd, poet, Catch2, google-benchmark, ...)
#   - simdrng_enable_warnings(target)          -- strict warning profile
#   - simdrng_configure_static_analysis(target)-- clang-tidy / cppcheck hooks
#   - a `coverage` target (lcov+genhtml, gcovr fallback)
#
# The library target itself never gets the strict warning profile; only the
# tests/examples/benchmarks do (see the respective CMakeLists).
# ==============================================================================

# -------------------------
# CPM bootstrap (pinned by SHA256)
# -------------------------
include(FetchContent)

set(CPM_DOWNLOAD_VERSION 0.42.0)
FetchContent_Declare(
    CPM
    URL
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    URL_HASH
        SHA256=2020b4fc42dba44817983e06342e682ecfc3d2f484a581f11cc5731fbe4dce8a
    DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_GetProperties(CPM)
if(NOT CPM_POPULATED)
    FetchContent_MakeAvailable(CPM)
endif()
include(${cpm_SOURCE_DIR}/CPM.cmake)

# -------------------------
# Warnings
# -------------------------
# Off by default so a normal build stays green; CI flips it on for the strict gate.
option(SIMDRNG_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Apply a strict warning profile to a single target (PRIVATE for normal targets,
# INTERFACE for interface libraries).
function(simdrng_enable_warnings target)
    if(NOT TARGET "${target}")
        message(
            FATAL_ERROR
            "simdrng_enable_warnings called with non-existent target '${target}'"
        )
    endif()
    get_target_property(_type "${target}" TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        set(_scope INTERFACE)
    else()
        set(_scope PRIVATE)
    endif()

    set(_cxx $<COMPILE_LANGUAGE:CXX>)
    set(_clang $<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>)
    set(_gnu $<CXX_COMPILER_ID:GNU>)
    set(_gnu_or_clang $<OR:${_gnu},${_clang}>)
    set(_msvc $<CXX_COMPILER_ID:MSVC>)

    set(_warns
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wnull-dereference
        -Woverloaded-virtual
        -Wcast-align
        -Wunused
        -Wimplicit-fallthrough
        -Wformat=2
    )

    set(_opts)
    foreach(flag IN LISTS _warns)
        list(APPEND _opts $<$<AND:${_cxx},${_gnu_or_clang}>:${flag}>)
    endforeach()
    list(
        APPEND _opts
        $<$<AND:${_cxx},${_msvc}>:/W4>
        $<$<AND:${_cxx},${_msvc}>:/permissive->
    )

    if(SIMDRNG_WARNINGS_AS_ERRORS)
        list(
            APPEND _opts
            $<$<AND:${_cxx},${_gnu_or_clang}>:-Werror>
            $<$<AND:${_cxx},${_msvc}>:/WX>
        )
    endif()

    target_compile_options(${target} ${_scope} ${_opts})
endfunction()

# -------------------------
# Static analysis (clang-tidy / cppcheck via target properties)
# -------------------------
option(SIMDRNG_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)
option(SIMDRNG_ENABLE_CPPCHECK "Run cppcheck as part of the build" OFF)

function(simdrng_configure_static_analysis target)
    if(NOT TARGET "${target}")
        message(
            FATAL_ERROR
            "simdrng_configure_static_analysis called with non-existent target '${target}'"
        )
    endif()
    if(SIMDRNG_ENABLE_CLANG_TIDY)
        find_program(_clang_tidy NAMES clang-tidy)
        if(_clang_tidy)
            set_property(
                TARGET ${target}
                PROPERTY
                    CXX_CLANG_TIDY "${_clang_tidy};--extra-arg=-fsyntax-only"
            )
        else()
            message(
                WARNING
                "SIMDRNG_ENABLE_CLANG_TIDY is ON but clang-tidy was not found"
            )
        endif()
    endif()
    if(SIMDRNG_ENABLE_CPPCHECK)
        find_program(_cppcheck NAMES cppcheck)
        if(_cppcheck)
            set_property(
                TARGET ${target}
                PROPERTY
                    CXX_CPPCHECK
                        "${_cppcheck};--enable=warning,style,performance,portability;--inline-suppr"
            )
        else()
            message(
                WARNING
                "SIMDRNG_ENABLE_CPPCHECK is ON but cppcheck was not found"
            )
        endif()
    endif()
endfunction()

# -------------------------
# Sanitizers
# -------------------------
# OFF | ON (ASan + UBSan) | TSAN. Applied globally (compile + link) so the
# library, dispatch TUs, and tests are all instrumented. No -march guard is
# needed: xsimd::dispatch only ever calls the SIMD tier the runtime CPU
# supports, so the AVX-512 TU's instructions stay dormant on lesser CPUs.
set(SIMDRNG_USE_SANITIZERS
    "OFF"
    CACHE STRING
    "Sanitizers: OFF, ON (ASan+UBSan), TSAN"
)
set_property(CACHE SIMDRNG_USE_SANITIZERS PROPERTY STRINGS OFF ON TSAN)
string(TOUPPER "${SIMDRNG_USE_SANITIZERS}" _simdrng_san_mode)
set(_simdrng_san_flags)
if(_simdrng_san_mode STREQUAL "OFF")
elseif(_simdrng_san_mode STREQUAL "ON")
    set(_simdrng_san_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
elseif(_simdrng_san_mode STREQUAL "TSAN")
    set(_simdrng_san_flags -fsanitize=thread -fno-omit-frame-pointer)
else()
    message(
        FATAL_ERROR
        "Unsupported SIMDRNG_USE_SANITIZERS value '${SIMDRNG_USE_SANITIZERS}'. Use one of: OFF, ON, TSAN."
    )
endif()
if(_simdrng_san_flags)
    # Applied to all languages (the flags are valid for both the C and C++
    # compilers); mirrors the global coverage instrumentation below.
    add_compile_options(${_simdrng_san_flags})
    add_link_options(${_simdrng_san_flags})
endif()

# -------------------------
# Coverage
# -------------------------
option(
    SIMDRNG_ENABLE_COVERAGE
    "Instrument the library and tests for coverage"
    OFF
)
if(SIMDRNG_ENABLE_COVERAGE)
    # --coverage instruments line/branch counters. -fprofile-update=atomic makes
    # those updates atomic so the multithreaded std::thread test can't race them
    # into negative counts (geninfo rejects "Unexpected negative count").
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:--coverage>
        $<$<COMPILE_LANGUAGE:CXX>:-fprofile-update=atomic>
    )
    add_link_options(--coverage)

    find_program(LCOV_EXECUTABLE lcov)
    find_program(GENHTML_EXECUTABLE genhtml)
    find_program(GCOVR_EXECUTABLE gcovr)
    if(LCOV_EXECUTABLE AND GENHTML_EXECUTABLE)
        add_custom_target(
            coverage
            COMMAND
                ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}
                --output-on-failure
            COMMAND
                ${LCOV_EXECUTABLE} --capture --directory ${CMAKE_BINARY_DIR}
                --output-file ${CMAKE_BINARY_DIR}/coverage.info --ignore-errors
                inconsistent,unused
            COMMAND
                ${LCOV_EXECUTABLE} --remove ${CMAKE_BINARY_DIR}/coverage.info
                "/usr/*" "*/_deps/*" --output-file
                ${CMAKE_BINARY_DIR}/coverage.filtered.info --ignore-errors
                inconsistent,unused
            COMMAND
                ${GENHTML_EXECUTABLE} -o ${CMAKE_BINARY_DIR}/coverage
                ${CMAKE_BINARY_DIR}/coverage.filtered.info
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT
                "Generating coverage report -> ${CMAKE_BINARY_DIR}/coverage/index.html"
            VERBATIM
        )
    elseif(GCOVR_EXECUTABLE)
        add_custom_target(
            coverage
            COMMAND
                ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR}
                --output-on-failure
            COMMAND
                ${GCOVR_EXECUTABLE} -r ${CMAKE_SOURCE_DIR} --filter
                "include/simdrng/|src/|tests/" --exclude ".*/_deps/.*" --html
                --html-details -o ${CMAKE_BINARY_DIR}/coverage-report.html
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating coverage report (gcovr)"
            VERBATIM
        )
    endif()
endif()
