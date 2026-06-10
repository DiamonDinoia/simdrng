References
==========

Sources by generator family
----------------------------

Each generator family in simdrng is implemented from a single canonical source
(plus, for ChaCha, an independent oracle used only to cross-check the tests).
This is the map from family to paper:

**Xoshiro256++ & SplitMix64**

- D. Blackman and S. Vigna, *Scrambled Linear Pseudorandom Number Generators*,
  ACM Transactions on Mathematical Software (2021).
  https://vigna.di.unimi.it/papers.php#BlVSLPNG — the design and analysis of the
  scrambled-linear family and the ``++`` scrambler.
- S. Vigna, *xoshiro / xoroshiro generators and the PRNG shootout*.
  https://prng.di.unimi.it/ — the shootout, the SplitMix64 **seeding**
  recommendation, and the ``uniform()`` conversion rationale.

**ChaCha 8 / 12 / 20**

- D. J. Bernstein, *ChaCha, a variant of Salsa20*.
  https://cr.yp.to/chacha.html — the cipher core simdrng uses as a CSPRNG.
- `Monocypher <https://monocypher.org/>`_ — an independent ChaCha20
  implementation used as a test oracle (cross-check only; not a dependency).

**Philox (2×32, 4×32, 2×64, 4×64)**

- J. K. Salmon, M. A. Moraes, R. O. Dror, D. E. Shaw,
  *Parallel Random Numbers: As Easy as 1, 2, 3* (Philox), SC '11 — the
  counter-based design behind the stateless, trivially parallel Philox family.

The annotated period, scrambling, seeding and conversion details below are the
authoritative statement of each fact; the per-family **Guides** link here rather
than restating the numbers.

.. _choosing-a-generator:

Choosing a generator
--------------------

The all-purpose-versus-special-purpose framing below is Vigna's
(https://prng.di.unimi.it/), extended here to the two counter-based families he
does not cover:

- **xoshiro256++ is the all-purpose default.** Vigna's recommended general
  64-bit generator: large state, passes all known statistical tests, very fast —
  but *not* cryptographically secure. Reach for it unless you have a specific
  reason not to.
- **Seed with SplitMix64.** Vigna's explicit recommendation (quoted under
  :ref:`seeding`); SplitMix is a *seeding helper only*, not a general-purpose
  engine.
- **Philox** (Salmon et al.) when you need **stateless, seekable,
  trivially-parallel** streams — GPU-style kernels and embarrassingly parallel
  Monte Carlo, where each work item derives its own sub-stream from
  ``(seed, counter)`` with no coordination.
- **ChaCha** (Bernstein) when you want **cryptographic-grade** statistical
  quality or a CSPRNG-style stream; pick the round count (8 / 12 / 20) to trade
  margin against speed.

.. _gen-properties:

Generator properties
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26 18 56

   * - Generator
     - Period
     - Notes
   * - xoshiro256++
     - 2\ :sup:`256` − 1
     - All-purpose 64-bit generator; ``jump()`` / ``long_jump()`` carve out
       2\ :sup:`128` / 2\ :sup:`192`-spaced non-overlapping subsequences for
       parallel streams.
   * - SplitMix64
     - 2\ :sup:`64`
     - Seeding only. Per Vigna: *"We suggest to use SplitMix64 to initialize
       the state of our generators starting from a 64-bit seed."* simdrng uses
       it to expand a scalar seed into the xoshiro state.
   * - ChaCha (8 / 12 / 20)
     - 2\ :sup:`64` per (key, nonce)
     - Counter-based stream cipher used as a CSPRNG; the 64-bit block counter
       sets the period within a (key, nonce) pair.
   * - Philox (2×32, 4×32, 2×64, 4×64)
     - 2\ :sup:`W·N`
     - Counter-based; stateless and trivially parallel — distinct counters give
       independent streams with no coordination.

.. _scrambling:

Scrambling and equidistribution
--------------------------------

xoshiro256++ is a **scrambled-linear** generator: a fast linear engine over
GF(2) (the ``xoshiro`` state transition) supplies excellent
equidistribution and a known period, while a non-linear *scrambler* applied to
the output hides the linear artifacts that batteries like BigCrush detect. The
``++`` scrambler — ``rotl(s0 + s3, 23) + s0`` — is a sum-and-rotate that, per
Blackman & Vigna, makes the output pass the full test suites while keeping the
per-call cost to a couple of instructions. The underlying linear engine is
**equidistributed**, i.e. over a full period every output value appears the same
number of times (minus the all-zero state), which is what gives the family its
strong low-dimensional uniformity.

.. _seeding:

Seeding
-------

Vigna recommends never seeding a scrambled-linear generator directly from a
small integer, because a low-entropy state propagates slowly through the linear
map. simdrng follows the recommendation from https://prng.di.unimi.it/ and
expands every 64-bit seed through **SplitMix64** before filling engine state:

   *"We suggest to use a SplitMix64 generator … to fill the state of our
   generators starting from a 64-bit seed, as research has shown that
   initialization must be performed with a generator radically different in
   nature from the one initialized."*

This is why ``0`` and ``1`` are perfectly good seeds in simdrng. The same
SplitMix64 expansion derives the Philox key from its seed.

.. _uniform-doubles:

Uniform doubles in ``[0, 1)``
-----------------------------

``uniform()`` converts a 64-bit output ``x`` to a double with

.. code-block:: cpp

   static_cast<double>(x >> 11) * 0x1.0p-53;   // (x >> 11) · 2⁻⁵³

following the rationale on https://prng.di.unimi.it/#remarks:

   *"This conversion guarantees that all dyadic rationals of the form k / 2⁻⁵³
   will be equally likely."*

The right shift takes the top 53 bits, because the conversion
*"prefers the high bits of x (usually, a good idea)"* — the high bits of
xoshiro-class generators have the best statistical quality.

.. _reference-impls:

Reference implementations
-------------------------

The test suite validates simdrng against the authors' own reference code:

- ``splitmix64.c`` and ``xoshiro256plusplus.c`` are downloaded from
  https://prng.di.unimi.it/ and are released to the **public domain (CC0)** by
  David Blackman and Sebastiano Vigna.
- ChaCha20 is cross-checked against `Monocypher <https://monocypher.org/>`_.

These files are used only by the tests/benchmarks as a ground-truth oracle and
are not part of the installed library.
