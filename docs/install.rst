Installation
============

Requirements
------------

- C++20 compiler (GCC 13+, Clang 17+, MSVC 19.38+, Apple Clang 15+)
- CMake 3.16+
- For the Python bindings: Python 3.10+, numpy 1.23+

Dependencies
------------

The SIMD backends use `xsimd <https://github.com/xtensor-stack/xsimd>`_ (pinned to
release ``14.3.0``) and `poet <https://github.com/DiamonDinoia/poet>`_ for
compile-time helpers. Both are fetched automatically at configure time, so no
manual installation is required.

C++ via CMake presets
---------------------

.. code-block:: sh

   git clone https://github.com/DiamonDinoia/simdrng.git
   cd simdrng
   cmake --preset release
   cmake --build build/release
   ctest --test-dir build/release --output-on-failure

C++ via install + ``find_package``
----------------------------------

.. code-block:: sh

   cmake --preset release
   cmake --build build/release
   cmake --install build/release --prefix /path/to/prefix

Then, from a downstream project:

.. code-block:: cmake

   find_package(simdrng CONFIG REQUIRED)
   target_link_libraries(my_target PRIVATE simdrng::simdrng)

The installed package config resolves its dependencies (``xsimd``, ``poet``) via
``find_dependency``.

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

+----------------------------+---------+------------------------------------------+
| CMake option               | Default | Purpose                                  |
+============================+=========+==========================================+
| ``SIMDRNG_BUILD_TESTS``    | ON      | Build Catch2 tests and benchmarks        |
|                            |         | (follows ``BUILD_TESTING``)              |
+----------------------------+---------+------------------------------------------+
| ``SIMDRNG_BUILD_PYTHON``   | OFF     | Build nanobind Python extension          |
+----------------------------+---------+------------------------------------------+
| ``SIMDRNG_BUILD_EXAMPLES`` | OFF     | Build C++ examples under examples/cpp    |
+----------------------------+---------+------------------------------------------+
| ``SIMDRNG_BUILD_DOCS``     | OFF     | Generate Sphinx/Doxygen docs             |
+----------------------------+---------+------------------------------------------+
| ``SIMDRNG_MARCH_NATIVE``   | OFF     | Compile benchmarks with ``-march=native``|
+----------------------------+---------+------------------------------------------+
| ``SIMDRNG_ENABLE_CODSPEED``| OFF     | Link codspeed-cpp into the bench harness |
+----------------------------+---------+------------------------------------------+
