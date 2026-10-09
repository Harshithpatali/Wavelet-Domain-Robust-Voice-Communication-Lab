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


def gilbert_elliott_mask(
    size: int,
    *,
    p_good_to_bad: float = 0.02,
    p_bad_to_good: float = 0.25,
    loss_good: float = 0.001,
    loss_bad: float = 0.8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a seeded Markov mask for burst-loss experiments.

    True means that the corresponding sample or packet is erased. Parameters
    are probabilities in [0, 1]; consecutive losses become correlated while
    the process remains in the bad state.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    for name, probability in (
        ("p_good_to_bad", p_good_to_bad),
        ("p_bad_to_good", p_bad_to_good),
        ("loss_good", loss_good),
        ("loss_bad", loss_bad),
    ):
        if not 0 <= probability <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)

    mask = np.zeros(size, dtype=bool)
    bad_state = False
    for i in range(size):
        if bad_state:
            mask[i] = rng.random() < loss_bad
            if rng.random() < p_bad_to_good:
                bad_state = False
        else:
            mask[i] = rng.random() < loss_good
            if rng.random() < p_good_to_bad:
                bad_state = True
    return mask


def gilbert_elliott_loss(
    signal: np.ndarray,
    *,
    p_good_to_bad: float = 0.02,
    p_bad_to_good: float = 0.25,
    loss_good: float = 0.001,
    loss_bad: float = 0.8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply correlated Gilbert–Elliott erasures by zeroing lost values."""
    x = np.asarray(signal, dtype=np.float64).copy()
    mask = gilbert_elliott_mask(
        x.size,
        p_good_to_bad=p_good_to_bad,
        p_bad_to_good=p_bad_to_good,
        loss_good=loss_good,
        loss_bad=loss_bad,
        rng=rng,
    )
    x.reshape(-1)[mask] = 0.0
    return x
