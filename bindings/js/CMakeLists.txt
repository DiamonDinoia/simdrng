# bindings/js — JavaScript/TypeScript binding over the C ABI, with two
# toolchain-selected backends:
#   * native (default): a Node-API `.node` addon linking simdrng_c_static.
#     Headers via node-addon-api (header-only, over the ABI-stable node_api.h
#     shipped with Node) — no cmake-js, so the preset/CTest flow is unchanged.
#   * WASM (under emcc): the same C ABI compiled to a single WASM module; the C
#     ABI symbols are exported by name and the Emscripten runtime drives them.
# Added by the top-level CMakeLists when SIMDRNG_BUILD_JS=ON. A missing toolchain
# (node / node-addon-api / tsc) degrades to a STATUS message.
#
# Artifacts land in this package's dist/ (gitignored): simdrng.node (native),
# simdrng.mjs + simdrng.wasm (WASM), and the tsc output (*.js + *.d.ts).

include_guard()

find_program(SIMDRNG_NODE_EXECUTABLE NAMES node)
if(NOT SIMDRNG_NODE_EXECUTABLE)
    message(STATUS "simdrng[js]: node not found — skipping.")
    return()
endif()

set(_js_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(_js_dist "${_js_dir}/dist")
set(_js_inc "${PROJECT_SOURCE_DIR}/include")
file(MAKE_DIRECTORY "${_js_dist}")

if(EMSCRIPTEN)
    # ---- WASM backend --------------------------------------------------------
    # The public surface is the C ABI, exported by name (underscore-prefixed).
    # The exported-functions list doubles as the linker's set of roots, pulling
    # the needed members out of the simdrng_c_static archive.
    add_executable(simdrng_wasm "${_js_dir}/src/wasm_glue.cpp")
    target_include_directories(simdrng_wasm PRIVATE "${_js_inc}")
    target_link_libraries(simdrng_wasm PRIVATE simdrng_c_static)

    set(_sr_exported_functions
        "_simdrng_create,_simdrng_free,_simdrng_get_kind,_simdrng_next_u64,_simdrng_next_double,_simdrng_fill_u64,_simdrng_fill_u32,_simdrng_fill_double,_simdrng_fill_float,_simdrng_version,_simdrng_last_error,_sr_create_split,_malloc,_free"
    )
    set(_sr_runtime_methods "ccall,cwrap,HEAPF64,HEAPF32,HEAP32,HEAPU8")
    target_link_options(
        simdrng_wasm
        PRIVATE
            "-sMODULARIZE=1"
            "-sEXPORT_ES6=1"
            "-sALLOW_MEMORY_GROWTH=1"
            "-sENVIRONMENT=web,node"
            "--no-entry"
            "-sEXPORTED_FUNCTIONS=[${_sr_exported_functions}]"
            "-sEXPORTED_RUNTIME_METHODS=[${_sr_runtime_methods}]"
    )
    set_target_properties(
        simdrng_wasm
        PROPERTIES
            OUTPUT_NAME simdrng
            SUFFIX ".mjs" # emcc emits simdrng.mjs (+ simdrng.wasm) here
            RUNTIME_OUTPUT_DIRECTORY "${_js_dist}"
    )
    message(STATUS "simdrng[js]: WASM module -> ${_js_dist}/simdrng.mjs")
else()
    # ---- native N-API addon --------------------------------------------------
    # node-addon-api's include_dir is relative to cwd, so resolve it from the
    # package dir (where node_modules lives) to an absolute path.
    execute_process(
        COMMAND
            "${SIMDRNG_NODE_EXECUTABLE}" -p
            "require('path').resolve(require('node-addon-api').include_dir)"
        WORKING_DIRECTORY "${_js_dir}"
        OUTPUT_VARIABLE _napi_inc
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _napi_res
        ERROR_QUIET
    )
    if(NOT _napi_res EQUAL 0 OR NOT EXISTS "${_napi_inc}")
        message(
            STATUS
            "simdrng[js]: node-addon-api not found (run 'npm install' in bindings/js) — skipping native addon."
        )
    else()
        # node_api.h ships inside the Node distribution's include/node directory.
        # Windows node has no bundled headers, so the prebuild workflow downloads
        # them and passes SIMDRNG_NODE_INCLUDE; otherwise derive from execPath.
        if(SIMDRNG_NODE_INCLUDE)
            set(_node_inc "${SIMDRNG_NODE_INCLUDE}")
        else()
            execute_process(
                COMMAND
                    "${SIMDRNG_NODE_EXECUTABLE}" -e
                    "const p=require('path');process.stdout.write(p.join(p.dirname(process.execPath),'..','include','node'))"
                OUTPUT_VARIABLE _node_inc
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _node_res
            )
        endif()
        if(NOT EXISTS "${_node_inc}")
            message(
                WARNING
                "simdrng[js]: could not resolve node include dir — skipping native addon."
            )
        else()
            add_library(simdrng_node MODULE "${_js_dir}/src/simdrng_napi.cpp")
            target_include_directories(
                simdrng_node
                PRIVATE "${_napi_inc}" "${_node_inc}" "${_js_inc}"
            )
            target_link_libraries(simdrng_node PRIVATE simdrng_c_static)
            target_compile_definitions(
                simdrng_node
                PRIVATE NAPI_VERSION=8 NAPI_CPP_EXCEPTIONS
            )
            set_target_properties(
                simdrng_node
                PROPERTIES
                    PREFIX ""
                    SUFFIX ".node"
                    OUTPUT_NAME simdrng
                    LIBRARY_OUTPUT_DIRECTORY "${_js_dist}"
            )
            if(APPLE)
                # node resolves the napi_* symbols at load time on macOS.
                target_link_options(
                    simdrng_node
                    PRIVATE -undefined dynamic_lookup
                )
            elseif(WIN32)
                # MSVC cannot leave the napi_* imports unresolved (unlike ELF /
                # dyld). Link node's import library and delay-load node.exe so they
                # bind to the host at load time — the node-gyp approach.
                # SIMDRNG_NODE_LIB is node.lib for the target node.
                if(NOT SIMDRNG_NODE_LIB)
                    message(
                        FATAL_ERROR
                        "simdrng[js]: Windows native addon needs -DSIMDRNG_NODE_LIB=<path to node.lib>"
                    )
                endif()
                target_sources(
                    simdrng_node
                    PRIVATE "${_js_dir}/src/win_delay_load_hook.cc"
                )
                target_link_libraries(
                    simdrng_node
                    PRIVATE "${SIMDRNG_NODE_LIB}" delayimp
                )
                target_link_options(simdrng_node PRIVATE /DELAYLOAD:node.exe)
            endif()
            message(
                STATUS
                "simdrng[js]: native addon -> ${_js_dist}/simdrng.node"
            )
        endif()
    endif()
endif()

# ---- TypeScript -> dist/*.js (+ .d.ts), built as part of the default target --
find_program(
    SIMDRNG_TSC
    NAMES tsc
    PATHS "${_js_dir}/node_modules/.bin"
    NO_DEFAULT_PATH
)
if(SIMDRNG_TSC)
    add_custom_target(
        simdrng_js_ts
        ALL
        COMMAND "${SIMDRNG_TSC}" -p "${_js_dir}/tsconfig.json"
        WORKING_DIRECTORY "${_js_dir}"
        COMMENT "simdrng[js]: compiling TypeScript -> dist/"
        VERBATIM
    )
else()
    message(
        STATUS
        "simdrng[js]: tsc not found in node_modules/.bin (run 'npm install') — dist/*.js not built here."
    )
endif()

# ---- CTest: the node:test suite, exercising whichever backends are in dist/ --
add_test(
    NAME js_simdrng
    COMMAND
        "${SIMDRNG_NODE_EXECUTABLE}" --test "${_js_dir}/test/simdrng.test.mjs"
)
message(STATUS "simdrng[js]: test 'js_simdrng' registered.")
