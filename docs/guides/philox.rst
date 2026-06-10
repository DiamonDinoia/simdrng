Philox
======

What it is
----------

Philox (Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3*) is a
counter-based generator: each output block is a keyed bijection of an
incrementing counter. There is no sequential state to advance — output *N* is
computed directly from counter *N* — which makes Philox **stateless and
trivially parallel**. simdrng provides the four standard shapes ``2x32``,
``4x32``, ``2x64`` and ``4x64`` (counter words × word width), each with the
reference 10 rounds.

When to use it
--------------

Choose Philox when work items must each produce an independent, reproducible
sub-stream **with no coordination** — GPU-style kernels, embarrassingly parallel
Monte Carlo, or any case where you want to seek to an arbitrary stream position
in O(1). If you instead want the fastest single sequential stream, prefer
:doc:`xoshiro`.

Scalar, SIMD dispatch, and native
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 30 38

   * - Type
     - Header
     - Use when
   * - ``Philox<N,W,R>`` / ``Philox4x32`` …
     - ``<simdrng/philox.hpp>``
     - Portable, dependency-free, ``constexpr``-friendly scalar core.
   * - ``PhiloxSIMD<…>`` / ``Philox4x32SIMD`` …
     - ``<simdrng/philox_simd.hpp>``
     - Runtime SIMD dispatch across CPUs from one binary.
   * - ``PhiloxNative<…>`` / ``Philox4x64Native`` …
     - ``<simdrng/philox_simd.hpp>``
     - Built with ``-march=native`` for the widest width with no dispatch.

.. code-block:: cpp

   #include <simdrng/philox_simd.hpp>

   simdrng::Philox4x32SIMD rng(seed, counter);   // (seed, starting counter)
   std::uint64_t x = rng();
   double u = rng.uniform();

The 64-bit seed is expanded into the Philox key via SplitMix64 (see
:doc:`splitmix`); the counter selects the starting stream position.

Periods and parallel streams
----------------------------

A ``Philox<N,W>`` block spans a counter space of 2\ :sup:`W·N`, so distinct
counters give independent, non-overlapping streams with no setup. The common
pattern is to fix the seed and assign each work item a disjoint counter range:

.. code-block:: cpp

   simdrng::Philox4x64SIMD rng(seed, /*counter=*/ work_item_id);

See :doc:`/references` for the full period table and the counter-based design
reference.
