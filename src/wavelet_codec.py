from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt


@dataclass
class WaveletPacket:
    coeffs: list[np.ndarray]
    slices: object
    original_length: int
    wavelet: str
    level: int


def validate_wavelet(wavelet: str) -> None:
    try:
        pywt.Wavelet(wavelet)
    except Exception as exc:
        raise ValueError(f"Unknown wavelet: {wavelet}") from exc


def decompose(signal: np.ndarray, wavelet: str = "db4", level: int = 4) -> WaveletPacket:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        raise ValueError("Signal is empty")
    validate_wavelet(wavelet)
    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    if level < 1 or level > max_level:
        raise ValueError(f"level must be between 1 and {max_level} for this signal/wavelet")
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode="symmetric")
    flat, slices = pywt.coeffs_to_array(coeffs)
    return WaveletPacket(
        coeffs=[flat],
        slices=slices,
        original_length=signal.size,
        wavelet=wavelet,
        level=level,
    )


def threshold(packet: WaveletPacket, fraction: float = 0.0) -> tuple[WaveletPacket, float]:
    """Zero coefficients below fraction * max(abs(coeff))."""
    if not 0 <= fraction <= 1:
        raise ValueError("threshold fraction must be in [0, 1]")
    flat = packet.coeffs[0].copy()
    if fraction == 0:
        return packet, 1.0
    cutoff = fraction * (np.max(np.abs(flat)) + 1e-12)
    mask = np.abs(flat) >= cutoff
    flat[~mask] = 0.0
    retained = float(np.count_nonzero(flat) / flat.size)
    return (
        WaveletPacket(
            [flat],
            packet.slices,
            packet.original_length,
            packet.wavelet,
            packet.level,
        ),
        retained,
    )


def make_permutation(size: int, key: int) -> np.ndarray:
    """Create a deterministic coefficient permutation from a shared integer key.

    This is intentionally a reversible research obfuscation mechanism, not
    cryptographic encryption. Both transmitter and receiver must know the key.
    """
    if size < 0:
        raise ValueError("size must be non-negative")
    if key < 0:
        raise ValueError("key must be non-negative")
    return np.random.default_rng(int(key)).permutation(size)


def scramble(packet: WaveletPacket, key: int) -> tuple[WaveletPacket, np.ndarray]:
    """Permute flattened wavelet coefficients using a shared deterministic key."""
    flat = np.asarray(packet.coeffs[0], dtype=np.float64).reshape(-1)
    permutation = make_permutation(flat.size, key)
    scrambled = flat[permutation]
    return (
        WaveletPacket(
            [scrambled],
            packet.slices,
            packet.original_length,
            packet.wavelet,
            packet.level,
        ),
        permutation,
    )


def descramble(packet: WaveletPacket, permutation: np.ndarray) -> WaveletPacket:
    """Restore the original coefficient ordering using the transmitter permutation."""
    scrambled = np.asarray(packet.coeffs[0], dtype=np.float64).reshape(-1)
    permutation = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if scrambled.size != permutation.size:
        raise ValueError("Coefficient and permutation sizes do not match")
    restored = np.empty_like(scrambled)
    restored[permutation] = scrambled
    return WaveletPacket(
        [restored],
        packet.slices,
        packet.original_length,
        packet.wavelet,
        packet.level,
    )


def reconstruct(packet: WaveletPacket) -> np.ndarray:
    flat = packet.coeffs[0]
    coeffs = pywt.array_to_coeffs(flat, packet.slices, output_format="wavedec")
    out = pywt.waverec(coeffs, packet.wavelet, mode="symmetric")
    return np.asarray(out[:packet.original_length], dtype=np.float64)
