find_package(Doxygen REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(DOXYGEN_INPUT_DIR "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")
set(DOXYGEN_XML_DIR "${DOXYGEN_OUTPUT_DIR}/xml")
set(SPHINX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPHINX_BUILD "${CMAKE_CURRENT_BINARY_DIR}/sphinx")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY
)

add_custom_target(doxygen
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}"
    COMMAND Doxygen::doxygen "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Doxygen to emit XML for Breathe"
    VERBATIM
)

add_custom_target(docs
    COMMAND ${CMAKE_COMMAND} -E env
            "BREATHE_PROJECT_XML=${DOXYGEN_XML_DIR}"
            ${Python3_EXECUTABLE} -m sphinx -b html -W --keep-going
            "${SPHINX_SOURCE}" "${SPHINX_BUILD}"
    DEPENDS doxygen
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building Sphinx HTML docs in ${SPHINX_BUILD}"
    VERBATIM
)
