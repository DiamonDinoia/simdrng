ChaCha
======

What it is
----------

ChaCha is Bernstein's stream-cipher core (a variant of Salsa20) repurposed here
as a counter-based RNG. Each 64-byte block is a pure function of a 256-bit key,
a 64-bit block counter and a 64-bit nonce, run through ``R`` quarter-round
rounds. simdrng ships the three standard strengths: **ChaCha8**, **ChaCha12**
and **ChaCha20**.

When to use it
--------------

Use ChaCha when you want **cryptographic-grade statistical quality** or a
CSPRNG-style stream: the higher round counts (12, 20) are the cryptographic
choices, while ChaCha8 trades margin for speed. Because the block is a function
of ``(key, counter, nonce)``, streams are reproducible and seekable. For raw
throughput of non-cryptographic randomness, :doc:`xoshiro` is faster.

Scalar, SIMD dispatch, and native
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Type
     - Header
     - Use when
   * - ``ChaCha<R>``
     - ``<simdrng/chacha.hpp>``
     - Portable, dependency-free scalar core (constructed from an explicit
       256-bit key).
   * - ``ChaChaSIMD<R>`` / ``ChaCha8SIMD`` …
     - ``<simdrng/chacha_simd.hpp>``
     - Runtime SIMD dispatch; also offers a seed constructor.
   * - ``ChaChaNative<R>`` / ``ChaCha20Native`` …
     - ``<simdrng/chacha_simd.hpp>``
     - Built with ``-march=native`` for the widest width with no dispatch.

.. code-block:: cpp

   #include <simdrng/chacha_simd.hpp>

   simdrng::ChaCha20SIMD rng(42);     // seed, counter = 0, nonce = 0
   std::uint64_t x = rng();
   double u = rng.uniform();

The scalar ``ChaCha<R>`` is constructed from an explicit key, counter and nonce
(``ChaCha<20> rng(key, counter, nonce)``); the SIMD/native types add a
seed-based constructor that derives the key for you.

Rounds, periods and reproducibility
-----------------------------------

``R`` is always rounded up to an even number (odd and even rounds are issued in
pairs). Within a fixed ``(key, nonce)`` the 64-bit block counter gives a period
of 2\ :sup:`64` blocks. Re-running with the same ``(seed/key, counter, nonce)``
reproduces the stream exactly, and bumping the counter or nonce yields
independent streams. ChaCha20 is cross-checked against `Monocypher
<https://monocypher.org/>`_ in the test suite; see :doc:`/references`.
