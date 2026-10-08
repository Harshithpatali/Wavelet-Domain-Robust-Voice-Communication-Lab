from __future__ import annotations

import numpy as np


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add zero-mean AWGN at the requested signal-to-noise ratio."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(snr_db):
        return x.copy()
    power = float(np.mean(x * x))
    if power <= 0:
        return x.copy()
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def packet_loss(signal: np.ndarray, loss_probability: float,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate sample/packet erasures as zeroed samples."""
    if not 0 <= loss_probability <= 1:
        raise ValueError("loss_probability must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(signal, dtype=np.float64).copy()
    mask = rng.random(x.shape) < loss_probability
    x[mask] = 0.0
    return x
