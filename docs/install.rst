Installation
============

Requirements
------------

- C++20 compiler (GCC 13+, Clang 17+, MSVC 19.38+, Apple Clang 15+)
- CMake 3.16+
- For the Python bindings: Python 3.10+, numpy 1.23+

C++ via CMake presets
---------------------

.. code-block:: sh

   git clone https://github.com/DiamonDinoia/simdrng.git
   cd simdrng
   cmake --preset release
   cmake --build build/release
   ctest --test-dir build/release --output-on-failure

C++ via FetchContent
--------------------

.. code-block:: cmake

   include(FetchContent)
   FetchContent_Declare(
     simdrng
     GIT_REPOSITORY https://github.com/DiamonDinoia/simdrng.git
     GIT_TAG        main
   )
   FetchContent_MakeAvailable(simdrng)
   target_link_libraries(my_target PRIVATE simdrng::simdrng)

Python bindings
---------------

.. code-block:: sh

   pip install .            # from a clone of the repo
   # or, for development:
   pip install -e '.[dev]'

Options
-------

+-----------------------+---------+---------------------------------------+
| CMake option          | Default | Purpose                               |
+=======================+=========+=======================================+
| ``ENABLE_TESTS``      | ON      | Build Catch2 tests and benchmarks     |
+-----------------------+---------+---------------------------------------+
| ``ENABLE_PYTHON``     | ON      | Build nanobind Python extension       |
+-----------------------+---------+---------------------------------------+
| ``BUILD_EXAMPLES``    | OFF     | Build C++ examples under examples/cpp |
+-----------------------+---------+---------------------------------------+
| ``BUILD_DOCS``        | OFF     | Generate Sphinx/Doxygen docs          |
+-----------------------+---------+---------------------------------------+
| ``MARCH_NATIVE``      | OFF     | Compile benchmarks with ``-march=native``|
+-----------------------+---------+---------------------------------------+
| ``ENABLE_CODSPEED``   | OFF     | Link codspeed-cpp into the bench harness|
+-----------------------+---------+---------------------------------------+
