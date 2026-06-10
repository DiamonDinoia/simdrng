# Derives a git-aware full version string.
#
# Input : SIMDRNG_VERSION_BASE  -- "x.y.z", read from the top-level VERSION file.
# Output: SIMDRNG_VERSION       -- "x.y.z" on an exact tag, otherwise
#                                  "x.y.z-dev.<commits>+g<short-sha>".
#
# The base x.y.z is what project(... VERSION ...) consumes; the full string is
# baked into include/simdrng/version.hpp for runtime/library identification.

set(SIMDRNG_VERSION "${SIMDRNG_VERSION_BASE}")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE _simdrng_tag_rc
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(NOT _simdrng_tag_rc EQUAL 0)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _simdrng_commits
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _simdrng_sha
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_simdrng_commits AND _simdrng_sha)
            set(SIMDRNG_VERSION
                "${SIMDRNG_VERSION_BASE}-dev.${_simdrng_commits}+g${_simdrng_sha}"
            )
        endif()
    endif()
endif()

message(STATUS "simdrng version: ${SIMDRNG_VERSION}")
