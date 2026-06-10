Xoshiro256++
============

What it is
----------

xoshiro256++ is Blackman & Vigna's all-purpose 64-bit generator: a small,
extremely fast scrambled-linear engine with a period of
2\ :sup:`256` − 1 that passes the full BigCrush / PractRand batteries. It is the
default ``simdrng::Xoshiro``.

When to use it
--------------

Reach for xoshiro256++ as the **general-purpose default** — simulation,
sampling, games, procedural generation — anywhere you want maximum throughput
of high-quality 64-bit values and do not need cryptographic guarantees or
stateless counter-based seeking (use :doc:`philox` for the latter).

Scalar, SIMD dispatch, and native
----------------------------------

Three implementations share one interface; pick by include and type:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Type
     - Header
     - Use when
   * - ``XoshiroScalar``
     - ``<simdrng/xoshiro_scalar.hpp>``
     - Portable, dependency-free, ``constexpr``-friendly single-stream use.
   * - ``XoshiroSIMD``
     - ``<simdrng/xoshiro_simd.hpp>``
     - One binary, many CPUs: the SIMD tier is chosen at runtime via xsimd
       dispatch (AVX-512 / AVX2 / SSE2 / NEON / …).
   * - ``XoshiroNative``
     - ``<simdrng/xoshiro_simd.hpp>``
     - You build with ``-march=native`` and want the widest width inlined with
       no dispatch indirection.

``simdrng::Xoshiro`` (from ``<simdrng/xoshiro.hpp>``) aliases ``XoshiroSIMD``
when built with xsimd, and ``XoshiroScalar`` otherwise.

.. code-block:: cpp

   #include <simdrng/xoshiro.hpp>

   simdrng::Xoshiro rng(42);          // seed (SplitMix64-expanded internally)
   std::uint64_t x = rng();           // next 64-bit value
   double u = rng.uniform();          // double in [0, 1)

The SIMD types refill an internal cache with wide batches, so an ordinary
generate loop runs vectorised — there is no separate bulk-fill API.

Periods and streams
-------------------

xoshiro256++ has a period of 2\ :sup:`256` − 1. The seed is expanded through
SplitMix64 (see :doc:`splitmix`) so even low-entropy seeds give sound state. For
independent parallel streams, pass ``thread_id`` and ``cluster_id``:

.. code-block:: cpp

   simdrng::Xoshiro rng(seed, thread_id, cluster_id);

Internally ``jump()`` carves out non-overlapping per-thread subsequences and
``long_jump()`` per-cluster starting points. See :ref:`gen-properties` for the
exact jump spacing and the full period table.

References
----------

xoshiro256++ and its SplitMix64 seeding follow D. Blackman and S. Vigna,
*Scrambled Linear Pseudorandom Number Generators* (the design and the ``++``
scrambler) and S. Vigna's https://prng.di.unimi.it/ (seeding and the shootout).
See :doc:`/references`.
