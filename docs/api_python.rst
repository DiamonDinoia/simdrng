Python API
==========

The ``simdrng`` module exposes every generator as a factory function that
returns a ``numpy.random.Generator`` subclass. Bulk methods (``random``,
``integers``, ``standard_normal``, …) transparently dispatch to the native
C++ fast path with the GIL released.

.. code-block:: python

   import simdrng
   import numpy as np

   rng = simdrng.XoshiroSIMD(seed=42)
   assert isinstance(rng, np.random.Generator)
   samples = rng.random(10_000_000)        # GIL released, bulk fill

Factory functions
-----------------

.. autofunction:: simdrng.SplitMix
.. autofunction:: simdrng.Xoshiro
.. autofunction:: simdrng.XoshiroSIMD
.. autofunction:: simdrng.XoshiroNative

.. autofunction:: simdrng.ChaCha8
.. autofunction:: simdrng.ChaCha12
.. autofunction:: simdrng.ChaCha20
.. autofunction:: simdrng.ChaCha8Native
.. autofunction:: simdrng.ChaCha12Native
.. autofunction:: simdrng.ChaCha20Native

.. autofunction:: simdrng.Philox4x32
.. autofunction:: simdrng.Philox2x32
.. autofunction:: simdrng.Philox4x64
.. autofunction:: simdrng.Philox2x64
.. autofunction:: simdrng.Philox4x32Native
.. autofunction:: simdrng.Philox2x32Native
.. autofunction:: simdrng.Philox4x64Native
.. autofunction:: simdrng.Philox2x64Native
