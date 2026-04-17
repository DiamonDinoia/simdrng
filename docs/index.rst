simdrng
=======

``simdrng`` is a C++20 header-only library of scalar and SIMD-accelerated
random number generators, with nanobind-powered Python bindings that drop
straight into ``numpy`` and ``scipy``.

**Generator families**

- Xoshiro256++ (scalar, SIMD dispatch, ``-march=native``)
- SplitMix64
- ChaCha (8 / 12 / 20 rounds) — scalar, SIMD dispatch, native
- Philox (2x32, 4x32, 2x64, 4x64) — scalar, SIMD dispatch, native

.. toctree::
   :maxdepth: 2
   :caption: Guide

   install
   examples
   benchmarks

.. toctree::
   :maxdepth: 2
   :caption: API

   api_cpp
   api_python

Indices
-------

* :ref:`genindex`
* :ref:`search`
