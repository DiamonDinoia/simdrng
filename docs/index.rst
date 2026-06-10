simdrng
=======

``simdrng`` is a library of C++20 random number generators that run scalar
**and** SIMD-accelerated, with nanobind-powered Python bindings that drop
straight into ``numpy`` and ``scipy``.

It pairs best-in-class scalar generators (xoshiro256++, Philox, the ChaCha
cipher core) with hand-tuned SIMD backends and picks the right implementation
**at runtime** via `xsimd <https://github.com/xtensor-stack/xsimd>`_ dispatch —
so one binary uses AVX-512 on a server and NEON on an ARM laptop, with no
recompilation.

Project links
-------------

- Source: https://github.com/DiamonDinoia/simdrng
- Benchmarks (CI-generated charts): https://github.com/DiamonDinoia/simdrng/tree/benchmark-results
- Compiler Explorer snippets: https://github.com/DiamonDinoia/simdrng/tree/godbolt-links

Core generator families
------------------------

- **Xoshiro256++** — all-purpose 64-bit generator (scalar, SIMD dispatch, ``-march=native``)
- **SplitMix64** — seeding helper used to expand a 64-bit seed into engine state
- **ChaCha 8 / 12 / 20** — counter-based cipher core (scalar, SIMD dispatch, native)
- **Philox 2x32 / 4x32 / 2x64 / 4x64** — stateless counter-based, trivially parallel (scalar, SIMD dispatch, native)

Choosing a generator
--------------------

Following Vigna's all-purpose-versus-special-purpose framing (extended to the
counter-based families):

- **Default to xoshiro256++** — the all-purpose 64-bit generator: large state,
  passes all known tests, very fast, not cryptographically secure. See
  :doc:`guides/xoshiro`.
- **Need stateless, seekable, trivially-parallel streams?** Use **Philox** —
  each work item derives its own sub-stream from ``(seed, counter)`` with no
  coordination. See :doc:`guides/philox`.
- **Need cryptographic-grade quality / a CSPRNG?** Use **ChaCha** (8 / 12 / 20).
  See :doc:`guides/chacha`.
- **Seeding** is always done with **SplitMix64** (Vigna's recommendation); it is
  a seeding helper, not a stream generator. See :doc:`guides/splitmix`.

The rationale behind this guidance is in :ref:`choosing-a-generator`.

Quick start
-----------

.. note::

   Ready-to-run `Compiler Explorer links
   <https://github.com/DiamonDinoia/simdrng/tree/godbolt-links>`_ let you try
   simdrng without cloning anything.

.. code-block:: cpp

   #include <simdrng/xoshiro.hpp>

   simdrng::Xoshiro rng(42);          // seed
   std::uint64_t x = rng();           // next 64-bit value
   double u = rng.uniform();          // double in [0, 1)

Every generator satisfies the standard ``UniformRandomBitGenerator``
requirements, so it composes with ``std::uniform_int_distribution`` and friends;
``uniform()`` is a faster path to a ``double`` in ``[0, 1)``.

Parallel streams
----------------

Pass optional ``thread_id`` / ``cluster_id`` to carve out independent,
non-overlapping streams per thread and per node (via xoshiro's ``jump()`` /
``long_jump()``); Philox instead derives each work item's sub-stream from
``(seed, counter)`` with no coordination. See the per-family **Guides** below.

Next reads
----------

- New here? Start with :doc:`install`, then the :doc:`examples`.
- Choosing a generator? See the per-family **Guides**.
- Reproducibility, periods and the ``uniform()`` rationale: :doc:`references`.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   install
   examples
   benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Guides

   guides/xoshiro
   guides/chacha
   guides/philox
   guides/splitmix

.. toctree::
   :maxdepth: 2
   :caption: API

   api_cpp
   api_python

.. toctree::
   :maxdepth: 1
   :caption: Reference

   references

Indices
-------

* :ref:`genindex`
* :ref:`search`
