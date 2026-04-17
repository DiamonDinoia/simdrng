"""Parametrized test suite for all pyrandom generators.

Every generator gets the same battery of tests:
- isinstance(rng, np.random.Generator)
- shape / dtype correctness
- SciPy compatibility
- State serialization round-trip
- Pickle round-trip
- Bulk fill correctness
- Jump reproducibility (Xoshiro family)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import stats

import pyrandom

# ── Generator matrix ────────────────────────────────────────────────────────

ALL_GENERATORS = [
    ("SplitMix", lambda: pyrandom.SplitMix(42)),
    ("Xoshiro", lambda: pyrandom.Xoshiro(42)),
    ("XoshiroSIMD", lambda: pyrandom.XoshiroSIMD(42)),
    ("XoshiroNative", lambda: pyrandom.XoshiroNative(42)),
    ("ChaCha8", lambda: pyrandom.ChaCha8(42)),
    ("ChaCha12", lambda: pyrandom.ChaCha12(42)),
    ("ChaCha20", lambda: pyrandom.ChaCha20(42)),
    ("Philox4x32", lambda: pyrandom.Philox4x32(42)),
    ("Philox2x32", lambda: pyrandom.Philox2x32(42)),
    ("Philox4x64", lambda: pyrandom.Philox4x64(42)),
    ("Philox2x64", lambda: pyrandom.Philox2x64(42)),
]


@pytest.fixture(params=ALL_GENERATORS, ids=[g[0] for g in ALL_GENERATORS])
def gen_factory(request):
    return request.param


# ── Basic functionality ─────────────────────────────────────────────────────


class TestGenerator:
    def test_returns_numpy_generator(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        assert isinstance(rng, np.random.Generator)

    def test_random_shape(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        assert rng.random((3, 4)).shape == (3, 4)

    def test_float32_dtype(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(100, dtype=np.float32)
        rng.random(out.shape, dtype=np.float32, out=out)
        assert out.dtype == np.float32
        assert np.all((out >= 0.0) & (out < 1.0))

    def test_integers(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        c = rng.integers(0, 10, size=(2, 5))
        assert c.shape == (2, 5)
        assert np.all((c >= 0) & (c < 10))

    def test_normal(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        d = rng.normal(0.0, 1.0, size=(3, 3))
        assert d.shape == (3, 3)

    def test_scipy_compat(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        s_norm = stats.norm.rvs(size=5, random_state=rng)
        assert s_norm.shape == (5,)
        s_int = stats.randint.rvs(0, 10, size=5, random_state=rng)
        assert s_int.max() < 10 and s_int.min() >= 0

    def test_random_values_in_range(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        vals = rng.random(10_000)
        assert np.all((vals >= 0.0) & (vals < 1.0))

    def test_different_seeds_differ(self, gen_factory):
        _name, factory = gen_factory
        rng1 = factory()
        # Create a second generator with a different seed
        rng2 = factory()  # same seed
        # Same seed should produce same sequence
        np.testing.assert_array_equal(rng1.random(10), rng2.random(10))


# ── State serialization ────────────────────────────────────────────────────


class TestStateSerialization:
    def test_state_roundtrip(self, gen_factory):
        """get_state -> set_state preserves sequence."""
        _name, factory = gen_factory
        rng = factory()
        rng.random(100)  # advance state
        state = rng.bit_generator.state
        expected = rng.random(10)
        rng.bit_generator.state = state
        actual = rng.random(10)
        np.testing.assert_array_equal(actual, expected)

    def test_pickle_roundtrip(self, gen_factory):
        """Pickle/unpickle preserves sequence."""
        _name, factory = gen_factory
        rng = factory()
        rng.random(100)
        state = rng.bit_generator.state
        bg = pickle.loads(pickle.dumps(rng.bit_generator))
        rng2 = np.random.Generator(bg)
        rng.bit_generator.state = state
        np.testing.assert_array_equal(rng.random(10), rng2.random(10))

    def test_state_wrong_generator_raises(self, gen_factory):
        """Setting state from a different generator type raises ValueError."""
        _name, factory = gen_factory
        rng = factory()
        state = rng.bit_generator.state
        state["bit_generator"] = "__nonexistent__"
        with pytest.raises(ValueError):
            rng.bit_generator.state = state


# ── Bulk fill ───────────────────────────────────────────────────────────────


class TestBulkFill:
    def test_fill_uniform_range(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(10_000, dtype=np.float64)
        rng.bit_generator.fill_uniform(out)
        assert np.all((out >= 0.0) & (out < 1.0))

    def test_fill_uint64_not_all_zero(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(1_000, dtype=np.uint64)
        rng.bit_generator.fill_uint64(out)
        assert not np.all(out == 0)

    def test_fill_uniform_matches_sequential(self, gen_factory):
        """Bulk fill produces same values as per-sample operator() path."""
        _name, factory = gen_factory
        rng1 = factory()
        rng2 = factory()

        # Per-sample via random_raw -> manual convert
        raw = rng1.bit_generator.random_raw(1000)
        expected = (raw >> np.uint64(11)).astype(np.float64) * (2.0**-53)

        out = np.empty(1000, dtype=np.float64)
        rng2.bit_generator.fill_uniform(out)
        np.testing.assert_array_equal(out, expected)


# ── Jump tests (Xoshiro family) ────────────────────────────────────────────

JUMP_GENERATORS = [
    ("Xoshiro", lambda: pyrandom.Xoshiro(42)),
    ("XoshiroSIMD", lambda: pyrandom.XoshiroSIMD(42)),
    ("XoshiroNative", lambda: pyrandom.XoshiroNative(42)),
]


@pytest.fixture(
    params=JUMP_GENERATORS, ids=[g[0] for g in JUMP_GENERATORS],
)
def jump_factory(request):
    return request.param


class TestJump:
    def test_jump_produces_different_sequence(self, jump_factory):
        _name, factory = jump_factory
        rng1 = factory()
        rng2 = factory()
        rng2.bit_generator.jump()
        # After jump, sequences should differ
        a = rng1.random(100)
        b = rng2.random(100)
        assert not np.array_equal(a, b)

    def test_jump_reproducible(self, jump_factory):
        _name, factory = jump_factory
        rng1 = factory()
        rng2 = factory()
        rng1.bit_generator.jump()
        rng2.bit_generator.jump()
        np.testing.assert_array_equal(rng1.random(100), rng2.random(100))


# ── Global generator ───────────────────────────────────────────────────────


class TestGlobalGenerator:
    def test_seed_and_random(self):
        pyrandom.seed(123)
        a = pyrandom.random(10)
        pyrandom.seed(123)
        b = pyrandom.random(10)
        np.testing.assert_array_equal(a, b)

    def test_default_rng_returns_generator(self):
        rng = pyrandom.default_rng(99)
        assert isinstance(rng, np.random.Generator)

    def test_normal(self):
        pyrandom.seed(42)
        vals = pyrandom.normal(size=1000)
        assert vals.shape == (1000,)

    def test_integers(self):
        pyrandom.seed(42)
        vals = pyrandom.integers(0, 100, size=50)
        assert np.all((vals >= 0) & (vals < 100))
