"""High-performance random number generators with seamless NumPy/SciPy integration.

Every factory function returns a ``Generator`` (subclass of
``np.random.Generator``) whose ``random()`` method transparently dispatches
to C++ bulk fill — zero copies, GIL released, no special API.

    >>> import pyrandom
    >>> rng = pyrandom.XoshiroSIMD(42)
    >>> rng.random(10_000_000)              # bulk float64 via C++ fast path
    >>> rng.random(10_000_000, dtype=np.float32)  # bulk float32, 2 per uint64
    >>> isinstance(rng, np.random.Generator)      # True
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

import numpy as np

from pyrandom_ext import (
    _SplitMix,
    _Xoshiro,
    _XoshiroSIMD,
    _ChaCha8SIMD,
    _ChaCha12SIMD,
    _ChaCha20SIMD,
    _Philox4x32SIMD,
    _Philox2x32SIMD,
    _Philox4x64SIMD,
    _Philox2x64SIMD,
)

# Conditional imports for native (compile-time arch) generators
try:
    from pyrandom_ext import _XoshiroNative
except ImportError:
    _XoshiroNative = None

try:
    from pyrandom_ext import _ChaCha8Native, _ChaCha12Native, _ChaCha20Native
except ImportError:
    _ChaCha8Native = _ChaCha12Native = _ChaCha20Native = None

try:
    from pyrandom_ext import (
        _Philox4x32Native, _Philox2x32Native,
        _Philox4x64Native, _Philox2x64Native,
    )
except ImportError:
    _Philox4x32Native = _Philox2x32Native = None
    _Philox4x64Native = _Philox2x64Native = None

# ── Registry for pickle restoration ─────────────────────────────────────────

_REGISTRY: dict[str, type[BitGeneratorBase]] = {}


def _register(cls: type[BitGeneratorBase]) -> type[BitGeneratorBase]:
    _REGISTRY[cls._bit_generator_name] = cls
    return cls


def _restore_bit_generator(
    name: str, state: dict,
) -> BitGeneratorBase:
    return _REGISTRY[name]._from_state(state)


# ── Base class ──────────────────────────────────────────────────────────────


class BitGeneratorBase:
    """Base class for all pyrandom bit generators.

    Provides the ``capsule`` / ``lock`` / ``state`` interface that
    ``np.random.Generator`` requires, plus pickle support and bulk fill.
    """

    __slots__ = ("_core", "_capsule", "lock")
    _bit_generator_name: str

    def __init__(self, core: object) -> None:
        self._core = core
        self._capsule = core.capsule()
        self.lock = threading.Lock()

    @property
    def capsule(self) -> object:
        return self._capsule

    @property
    def state(self) -> dict:
        return {
            "bit_generator": self._bit_generator_name,
            "state": self._core.get_state(),
        }

    @state.setter
    def state(self, value: dict) -> None:
        if value["bit_generator"] != self._bit_generator_name:
            raise ValueError(
                f"state for {value['bit_generator']}, "
                f"expected {self._bit_generator_name}"
            )
        self._core.set_state(value["state"])

    def __reduce__(self) -> tuple:
        return (_restore_bit_generator, (self._bit_generator_name, self.state))

    def fill_uniform(self, out: np.ndarray) -> None:
        """Bulk fill float64 array.  Releases GIL."""
        self._core.fill_uniform(np.ascontiguousarray(out, dtype=np.float64).ravel())

    def fill_float32(self, out: np.ndarray) -> None:
        """Bulk fill float32 array (2 samples per uint64).  Releases GIL."""
        self._core.fill_float32(np.ascontiguousarray(out, dtype=np.float32).ravel())

    def fill_uint64(self, out: np.ndarray) -> None:
        """Bulk fill uint64 array.  Releases GIL."""
        self._core.fill_uint64(np.ascontiguousarray(out, dtype=np.uint64).ravel())

    def fill_uint32(self, out: np.ndarray) -> None:
        """Bulk fill uint32 array (2 samples per uint64).  Releases GIL."""
        self._core.fill_uint32(np.ascontiguousarray(out, dtype=np.uint32).ravel())

    def random_raw(self, size: int = 1) -> np.ndarray:
        """Return *size* raw uint64 values."""
        out = np.empty(size, dtype=np.uint64)
        self._core.fill_uint64(out)
        return out


class Generator(np.random.Generator):
    """``np.random.Generator`` subclass with transparent C++ bulk fill.

    ``random()`` is overridden: when the output is a contiguous float64 or
    float32 array, it dispatches directly to the C++ bulk fill (GIL released,
    zero copies).  All other calls fall through to NumPy's standard path.
    """

    def random(
        self,
        size: int | tuple[int, ...] | None = None,
        dtype: np.dtype | type = np.float64,
        out: np.ndarray | None = None,
    ) -> np.ndarray | float:
        if size is None and out is None:
            return super().random()

        dt = np.dtype(dtype)

        if out is not None:
            target = out
        else:
            shape = size if isinstance(size, tuple) else (size,)
            target = np.empty(shape, dtype=dt)

        if target.flags.c_contiguous:
            flat = target.reshape(-1)  # view, zero-copy for C-contiguous
            core = self.bit_generator._core
            if dt == np.float64:
                core.fill_uniform(flat)
                return target
            if dt == np.float32:
                core.fill_float32(flat)
                return target

        return super().random(size=size, dtype=dtype, out=out)


# ── Per-generator subclasses ────────────────────────────────────────────────


@_register
class SplitMixBitGenerator(BitGeneratorBase):
    _bit_generator_name = "SplitMix"

    def __init__(self, seed: int) -> None:
        super().__init__(_SplitMix(seed))

    @classmethod
    def _from_state(cls, state: dict) -> SplitMixBitGenerator:
        bg = cls(0)
        bg.state = state
        return bg


@_register
class XoshiroBitGenerator(BitGeneratorBase):
    _bit_generator_name = "Xoshiro"

    def __init__(
        self, seed: int, thread: int = 0, cluster: int = 0,
    ) -> None:
        super().__init__(_Xoshiro(seed, thread, cluster))

    @classmethod
    def _from_state(cls, state: dict) -> XoshiroBitGenerator:
        bg = cls(0)
        bg.state = state
        return bg

    def jump(self) -> None:
        self._core.jump()

    def long_jump(self) -> None:
        self._core.long_jump()


@_register
class XoshiroSIMDBitGenerator(BitGeneratorBase):
    _bit_generator_name = "XoshiroSIMD"

    def __init__(
        self, seed: int, thread: int = 0, cluster: int = 0,
    ) -> None:
        super().__init__(_XoshiroSIMD(seed, thread, cluster))

    @classmethod
    def _from_state(cls, state: dict) -> XoshiroSIMDBitGenerator:
        bg = cls(0)
        bg.state = state
        return bg

    def jump(self) -> None:
        self._core.jump()

    def long_jump(self) -> None:
        self._core.long_jump()


if _XoshiroNative is not None:

    @_register
    class XoshiroNativeBitGenerator(BitGeneratorBase):
        _bit_generator_name = "XoshiroNative"

        def __init__(
            self, seed: int, thread: int = 0, cluster: int = 0,
        ) -> None:
            super().__init__(_XoshiroNative(seed, thread, cluster))

        @classmethod
        def _from_state(cls, state: dict) -> XoshiroNativeBitGenerator:
            bg = cls(0)
            bg.state = state
            return bg

        def jump(self) -> None:
            self._core.jump()

        def long_jump(self) -> None:
            self._core.long_jump()


# ── ChaCha helpers ──────────────────────────────────────────────────────────



def _make_chacha_bg_class(
    name: str,
    simd_cls: type,
    native_cls: type | None,
) -> tuple[type[BitGeneratorBase], type[BitGeneratorBase] | None]:
    """Factory that creates SIMD and Native BitGenerator subclasses for a ChaCha variant."""

    @_register
    class _SIMDClass(BitGeneratorBase):
        _bit_generator_name = f"{name}SIMD"

        def __init__(
            self,
            seed_or_key: int | Sequence[int] = 0,
            counter: int = 0,
            nonce: int = 0,
        ) -> None:
            if isinstance(seed_or_key, int):
                super().__init__(simd_cls(seed_or_key, counter, nonce))
            else:
                super().__init__(simd_cls(list(seed_or_key), counter, nonce))

        @classmethod
        def _from_state(cls, state: dict) -> _SIMDClass:
            bg = cls(0)
            bg.state = state
            return bg

    _SIMDClass.__name__ = _SIMDClass.__qualname__ = f"{name}SIMDBitGenerator"

    native_klass = None
    if native_cls is not None:

        @_register
        class _NativeClass(BitGeneratorBase):
            _bit_generator_name = f"{name}Native"

            def __init__(
                self,
                seed_or_key: int | Sequence[int] = 0,
                counter: int = 0,
                nonce: int = 0,
            ) -> None:
                if isinstance(seed_or_key, int):
                    super().__init__(native_cls(seed_or_key, counter, nonce))
                else:
                    super().__init__(native_cls(list(seed_or_key), counter, nonce))

            @classmethod
            def _from_state(cls, state: dict) -> _NativeClass:
                bg = cls(0)
                bg.state = state
                return bg

        _NativeClass.__name__ = _NativeClass.__qualname__ = (
            f"{name}NativeBitGenerator"
        )
        native_klass = _NativeClass

    return _SIMDClass, native_klass


ChaCha8SIMDBitGenerator, ChaCha8NativeBitGenerator = _make_chacha_bg_class(
    "ChaCha8", _ChaCha8SIMD, _ChaCha8Native,
)
ChaCha12SIMDBitGenerator, ChaCha12NativeBitGenerator = _make_chacha_bg_class(
    "ChaCha12", _ChaCha12SIMD, _ChaCha12Native,
)
ChaCha20SIMDBitGenerator, ChaCha20NativeBitGenerator = _make_chacha_bg_class(
    "ChaCha20", _ChaCha20SIMD, _ChaCha20Native,
)


# ── Philox helpers ─────────────────────────────────────────────────────────


def _make_philox_bg_class(
    name: str,
    simd_cls: type,
    native_cls: type | None,
) -> tuple[type[BitGeneratorBase], type[BitGeneratorBase] | None]:
    """Factory that creates SIMD and Native BitGenerator subclasses for a Philox variant."""

    @_register
    class _SIMDClass(BitGeneratorBase):
        _bit_generator_name = f"{name}SIMD"

        def __init__(
            self,
            seed_or_key: int | Sequence[int] = 0,
            counter: int | Sequence[int] = 0,
        ) -> None:
            if isinstance(seed_or_key, int):
                if isinstance(counter, int):
                    super().__init__(simd_cls(seed_or_key, counter))
                else:
                    super().__init__(simd_cls(seed_or_key, 0))
            else:
                super().__init__(simd_cls(list(seed_or_key), list(counter)))

        @classmethod
        def _from_state(cls, state: dict) -> _SIMDClass:
            bg = cls(0)
            bg.state = state
            return bg

    _SIMDClass.__name__ = _SIMDClass.__qualname__ = f"{name}SIMDBitGenerator"

    native_klass = None
    if native_cls is not None:

        @_register
        class _NativeClass(BitGeneratorBase):
            _bit_generator_name = f"{name}Native"

            def __init__(
                self,
                seed_or_key: int | Sequence[int] = 0,
                counter: int | Sequence[int] = 0,
            ) -> None:
                if isinstance(seed_or_key, int):
                    if isinstance(counter, int):
                        super().__init__(native_cls(seed_or_key, counter))
                    else:
                        super().__init__(native_cls(seed_or_key, 0))
                else:
                    super().__init__(native_cls(list(seed_or_key), list(counter)))

            @classmethod
            def _from_state(cls, state: dict) -> _NativeClass:
                bg = cls(0)
                bg.state = state
                return bg

        _NativeClass.__name__ = _NativeClass.__qualname__ = (
            f"{name}NativeBitGenerator"
        )
        native_klass = _NativeClass

    return _SIMDClass, native_klass


Philox4x32SIMDBitGenerator, Philox4x32NativeBitGenerator = _make_philox_bg_class(
    "Philox4x32", _Philox4x32SIMD, _Philox4x32Native,
)
Philox2x32SIMDBitGenerator, Philox2x32NativeBitGenerator = _make_philox_bg_class(
    "Philox2x32", _Philox2x32SIMD, _Philox2x32Native,
)
Philox4x64SIMDBitGenerator, Philox4x64NativeBitGenerator = _make_philox_bg_class(
    "Philox4x64", _Philox4x64SIMD, _Philox4x64Native,
)
Philox2x64SIMDBitGenerator, Philox2x64NativeBitGenerator = _make_philox_bg_class(
    "Philox2x64", _Philox2x64SIMD, _Philox2x64Native,
)


# ── Factory functions ───────────────────────────────────────────────────────
# Each returns a real np.random.Generator.


def SplitMix(seed: int) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by SplitMix."""
    return Generator(SplitMixBitGenerator(seed))


def Xoshiro(seed: int, thread: int = 0, cluster: int = 0) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by scalar Xoshiro256++."""
    return Generator(XoshiroBitGenerator(seed, thread, cluster))


def XoshiroSIMD(
    seed: int, thread: int = 0, cluster: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by XoshiroSIMD (runtime SIMD dispatch)."""
    return Generator(XoshiroSIMDBitGenerator(seed, thread, cluster))


def XoshiroNative(
    seed: int, thread: int = 0, cluster: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by XoshiroNative (compile-time best arch)."""
    if _XoshiroNative is None:
        raise RuntimeError("XoshiroNative not available on this platform")
    return Generator(XoshiroNativeBitGenerator(seed, thread, cluster))


def ChaCha8(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha8 (SIMD dispatch).

    Accepts an integer seed (expanded to 256-bit key in C++) or a list of 8 uint32 key words.
    """
    return Generator(ChaCha8SIMDBitGenerator(seed_or_key, counter, nonce))


def ChaCha12(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha12 (SIMD dispatch)."""
    return Generator(ChaCha12SIMDBitGenerator(seed_or_key, counter, nonce))


def ChaCha20(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha20 (SIMD dispatch)."""
    return Generator(ChaCha20SIMDBitGenerator(seed_or_key, counter, nonce))


def ChaCha8Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha8 (compile-time best arch)."""
    if ChaCha8NativeBitGenerator is None:
        raise RuntimeError("ChaCha8Native not available on this platform")
    return Generator(ChaCha8NativeBitGenerator(seed_or_key, counter, nonce))


def ChaCha12Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha12 (compile-time best arch)."""
    if ChaCha12NativeBitGenerator is None:
        raise RuntimeError("ChaCha12Native not available on this platform")
    return Generator(ChaCha12NativeBitGenerator(seed_or_key, counter, nonce))


def ChaCha20Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int = 0,
    nonce: int = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by ChaCha20 (compile-time best arch)."""
    if ChaCha20NativeBitGenerator is None:
        raise RuntimeError("ChaCha20Native not available on this platform")
    return Generator(ChaCha20NativeBitGenerator(seed_or_key, counter, nonce))


def Philox4x32(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox4x32-10 (SIMD dispatch)."""
    return Generator(Philox4x32SIMDBitGenerator(seed_or_key, counter))


def Philox2x32(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox2x32-10 (SIMD dispatch)."""
    return Generator(Philox2x32SIMDBitGenerator(seed_or_key, counter))


def Philox4x64(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox4x64-10 (SIMD dispatch)."""
    return Generator(Philox4x64SIMDBitGenerator(seed_or_key, counter))


def Philox2x64(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox2x64-10 (SIMD dispatch)."""
    return Generator(Philox2x64SIMDBitGenerator(seed_or_key, counter))


def Philox4x32Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox4x32-10 (compile-time best arch)."""
    if Philox4x32NativeBitGenerator is None:
        raise RuntimeError("Philox4x32Native not available on this platform")
    return Generator(Philox4x32NativeBitGenerator(seed_or_key, counter))


def Philox2x32Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox2x32-10 (compile-time best arch)."""
    if Philox2x32NativeBitGenerator is None:
        raise RuntimeError("Philox2x32Native not available on this platform")
    return Generator(Philox2x32NativeBitGenerator(seed_or_key, counter))


def Philox4x64Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox4x64-10 (compile-time best arch)."""
    if Philox4x64NativeBitGenerator is None:
        raise RuntimeError("Philox4x64Native not available on this platform")
    return Generator(Philox4x64NativeBitGenerator(seed_or_key, counter))


def Philox2x64Native(
    seed_or_key: int | Sequence[int] = 0,
    counter: int | Sequence[int] = 0,
) -> np.random.Generator:
    """Return ``np.random.Generator`` backed by Philox2x64-10 (compile-time best arch)."""
    if Philox2x64NativeBitGenerator is None:
        raise RuntimeError("Philox2x64Native not available on this platform")
    return Generator(Philox2x64NativeBitGenerator(seed_or_key, counter))


# ── Global default generator ───────────────────────────────────────────────
# Provides a module-level RNG that can be used as a drop-in replacement for
# numpy.random functions.  Seed once, use everywhere — zero code changes.

_global_rng: Optional[np.random.Generator] = None
_global_lock = threading.Lock()


def seed(s: int = 42, *, generator: str = "XoshiroSIMD") -> None:
    """Set the global default generator.

    After calling ``pyrandom.seed(42)``, the module-level convenience
    functions (``pyrandom.random``, ``pyrandom.normal``, etc.) use this
    generator.

    Parameters
    ----------
    s : int
        Seed value.
    generator : str
        Generator backend.  One of ``"XoshiroSIMD"`` (default),
        ``"XoshiroNative"``, ``"Xoshiro"``, ``"SplitMix"``,
        ``"ChaCha20"``, etc.
    """
    global _global_rng
    factories = {
        "XoshiroSIMD": lambda: XoshiroSIMD(s),
        "XoshiroNative": lambda: XoshiroNative(s),
        "Xoshiro": lambda: Xoshiro(s),
        "SplitMix": lambda: SplitMix(s),
        "ChaCha8": lambda: ChaCha8(s),
        "ChaCha12": lambda: ChaCha12(s),
        "ChaCha20": lambda: ChaCha20(s),
        "Philox4x32": lambda: Philox4x32(s),
        "Philox2x32": lambda: Philox2x32(s),
        "Philox4x64": lambda: Philox4x64(s),
        "Philox2x64": lambda: Philox2x64(s),
    }
    if generator not in factories:
        raise ValueError(f"Unknown generator {generator!r}")
    with _global_lock:
        _global_rng = factories[generator]()


def default_rng(s: Optional[int] = None) -> np.random.Generator:
    """Return the global generator, creating one if needed.

    If *s* is given, re-seeds the global generator.
    """
    global _global_rng
    if s is not None:
        seed(s)
    with _global_lock:
        if _global_rng is None:
            _global_rng = XoshiroSIMD(42)
        return _global_rng


def random(
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Draw uniform [0, 1) samples from the global generator."""
    return default_rng().random(size)


def normal(
    loc: float = 0.0,
    scale: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Draw normal samples from the global generator."""
    return default_rng().normal(loc, scale, size)


def integers(
    low: int,
    high: int | None = None,
    size: int | tuple[int, ...] | None = None,
    dtype: np.dtype = np.int64,
    endpoint: bool = False,
) -> np.ndarray | int:
    """Draw integer samples from the global generator."""
    return default_rng().integers(low, high, size=size, dtype=dtype, endpoint=endpoint)


def uniform(
    low: float = 0.0,
    high: float = 1.0,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray | float:
    """Draw uniform [low, high) samples from the global generator."""
    return default_rng().uniform(low, high, size)


# ── Public API ──────────────────────────────────────────────────────────────

__all__ = [
    # Factory functions (return np.random.Generator)
    "SplitMix",
    "Xoshiro",
    "XoshiroSIMD",
    "XoshiroNative",
    "ChaCha8",
    "ChaCha12",
    "ChaCha20",
    "ChaCha8Native",
    "ChaCha12Native",
    "ChaCha20Native",
    "Philox4x32",
    "Philox2x32",
    "Philox4x64",
    "Philox2x64",
    "Philox4x32Native",
    "Philox2x32Native",
    "Philox4x64Native",
    "Philox2x64Native",
    # BitGenerator classes (for direct use / isinstance checks)
    "SplitMixBitGenerator",
    "XoshiroBitGenerator",
    "XoshiroSIMDBitGenerator",
    "XoshiroNativeBitGenerator",
    "ChaCha8SIMDBitGenerator",
    "ChaCha12SIMDBitGenerator",
    "ChaCha20SIMDBitGenerator",
    "ChaCha8NativeBitGenerator",
    "ChaCha12NativeBitGenerator",
    "ChaCha20NativeBitGenerator",
    "Philox4x32SIMDBitGenerator",
    "Philox2x32SIMDBitGenerator",
    "Philox4x64SIMDBitGenerator",
    "Philox2x64SIMDBitGenerator",
    "Philox4x32NativeBitGenerator",
    "Philox2x32NativeBitGenerator",
    "Philox4x64NativeBitGenerator",
    "Philox2x64NativeBitGenerator",
    # Global generator
    "seed",
    "default_rng",
    "random",
    "normal",
    "integers",
    "uniform",
]
