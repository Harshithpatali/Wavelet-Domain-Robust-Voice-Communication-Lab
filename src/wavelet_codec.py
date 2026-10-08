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
    return WaveletPacket(coeffs=[flat], slices=slices, original_length=signal.size,
                         wavelet=wavelet, level=level)


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
    return WaveletPacket([flat], packet.slices, packet.original_length,
                         packet.wavelet, packet.level), retained


def reconstruct(packet: WaveletPacket) -> np.ndarray:
    flat = packet.coeffs[0]
    coeffs = pywt.array_to_coeffs(flat, packet.slices, output_format="wavedec")
    out = pywt.waverec(coeffs, packet.wavelet, mode="symmetric")
    return np.asarray(out[:packet.original_length], dtype=np.float64)
