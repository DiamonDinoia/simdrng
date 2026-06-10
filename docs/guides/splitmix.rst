SplitMix64
==========

What it is
----------

SplitMix64 (Steele; recommended by Vigna for seeding) is a tiny, fast 64-bit
generator built from a Weyl sequence (the state is bumped by the golden-ratio
constant each call) followed by an avalanche mix. Its period is 2\ :sup:`64`.

When to use it
--------------

SplitMix64 is **a seeding helper, not a general-purpose engine**. simdrng uses
it internally to expand a single 64-bit seed into the larger state of the other
generators, so that even low-entropy seeds (``0``, ``1``, …) produce
well-distributed starting states. Use it directly when you need to derive
several independent 64-bit seeds from one master seed; for actual random
streams use :doc:`xoshiro`, :doc:`philox`, or :doc:`chacha`.

Scalar only
-----------

SplitMix64 has a single scalar implementation — there is no SIMD-dispatch or
``-march=native`` variant, because it exists to seed the other engines rather
than to produce bulk output.

.. code-block:: cpp

   #include <simdrng/splitmix.hpp>

   simdrng::SplitMix sm(seed);
   std::uint64_t a = sm();            // a derived seed
   std::uint64_t b = sm();            // another, independent of a

Each call advances the state and returns the next mixed value, so successive
calls yield independent 64-bit seeds.

Periods and streams
-------------------

SplitMix64 has a period of 2\ :sup:`64`. simdrng uses it only to expand a 64-bit
seed into the larger state of the other engines. See :ref:`gen-properties` for
the full period table.

References
----------

Using SplitMix64 to seed the other generators follows S. Vigna's explicit
recommendation (https://prng.di.unimi.it/), quoted under :ref:`seeding`. See
:doc:`/references`.
