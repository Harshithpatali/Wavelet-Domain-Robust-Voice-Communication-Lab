import numpy as np

from src.audio import synthetic_voice_like
from src.wavelet_codec import (
    decompose,
    descramble,
    reconstruct,
    scramble,
    threshold,
)


def test_round_trip_is_close():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    y = reconstruct(p)
    assert y.shape == x.shape
    assert np.max(np.abs(x - y)) < 1e-10


def test_threshold_reduces_or_preserves_coefficients():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    _, retained = threshold(p, 0.05)
    assert 0 < retained <= 1


def test_scramble_changes_coefficient_order():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    scrambled, permutation = scramble(p, key=2026)
    assert scrambled.coeffs[0].shape == p.coeffs[0].shape
    assert np.array_equal(np.sort(scrambled.coeffs[0]), np.sort(p.coeffs[0]))
    assert not np.array_equal(scrambled.coeffs[0], p.coeffs[0])
    assert permutation.shape == p.coeffs[0].shape


def test_scramble_descramble_is_exact():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    scrambled, permutation = scramble(p, key=2026)
    restored = descramble(scrambled, permutation)
    assert np.array_equal(restored.coeffs[0], p.coeffs[0])
    y = reconstruct(restored)
    assert np.max(np.abs(x - y)) < 1e-10


def test_same_key_recreates_same_permutation():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    scrambled_a, perm_a = scramble(p, key=12345)
    scrambled_b, perm_b = scramble(p, key=12345)
    assert np.array_equal(perm_a, perm_b)
    assert np.array_equal(scrambled_a.coeffs[0], scrambled_b.coeffs[0])


def test_different_key_does_not_restore_coefficients():
    x = synthetic_voice_like(duration=1.0)
    p = decompose(x, "db4", 4)
    scrambled, correct_permutation = scramble(p, key=2026)
    wrong_permutation = scramble(p, key=2027)[1]
    wrong = descramble(scrambled, wrong_permutation)
    assert not np.array_equal(wrong.coeffs[0], p.coeffs[0])
    restored = descramble(scrambled, correct_permutation)
    assert np.array_equal(restored.coeffs[0], p.coeffs[0])
