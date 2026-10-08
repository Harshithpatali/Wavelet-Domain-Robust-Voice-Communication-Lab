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
