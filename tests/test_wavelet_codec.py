import numpy as np
from src.audio import synthetic_voice_like
from src.wavelet_codec import decompose, threshold, reconstruct


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
