from __future__ import annotations

import numpy as np


def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean((a[:n] - b[:n]) ** 2))


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    signal_power = np.mean(a[:n] ** 2)
    error_power = np.mean((a[:n] - b[:n]) ** 2)
    if error_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10 * np.log10(signal_power / error_power))


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(estimate, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 2 or np.std(a[:n]) == 0 or np.std(b[:n]) == 0:
        return 0.0
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def segmental_snr_db(
    reference: np.ndarray,
    estimate: np.ndarray,
    frame_length: int = 320,
    hop_length: int = 160,
    min_snr_db: float = -10.0,
    max_snr_db: float = 35.0,
) -> float:
    """Compute mean clipped frame-wise SNR, useful as an audio diagnostic."""
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    if frame_length < 1 or hop_length < 1:
        raise ValueError("frame_length and hop_length must be positive")
    values: list[float] = []
    for start in range(0, n, hop_length):
        x = a[start : min(start + frame_length, n)]
        y = b[start : min(start + frame_length, n)]
        if x.size == 0:
            continue
        signal_power = float(np.mean(x * x))
        error_power = float(np.mean((x - y) ** 2))
        if error_power <= 1e-15:
            value = max_snr_db
        elif signal_power <= 1e-15:
            value = min_snr_db
        else:
            value = 10.0 * np.log10(signal_power / error_power)
        values.append(float(np.clip(value, min_snr_db, max_snr_db)))
    return float(np.mean(values)) if values else float("nan")


def optional_stoi(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int = 16000,
) -> float | None:
    """Return STOI when the optional pystoi package is installed, else None."""
    try:
        from pystoi import stoi
    except ImportError:
        return None
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < sample_rate // 4 or sample_rate not in (10000, 16000):
        return None
    try:
        return float(stoi(a[:n], b[:n], sample_rate, extended=False))
    except (ValueError, FloatingPointError):
        return None
