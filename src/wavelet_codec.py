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


# =====================================================================
# PORTABLE WAV TRANSMISSION
# =====================================================================
TRANSMISSION_MAGIC = 0.31415927
TRANSMISSION_HEADER_SIZE = 5
DEFAULT_CARRIER_GAIN = 1e-3


def make_transmission_wav(
    scrambled_packet: WaveletPacket,
    scrambled_audio: np.ndarray,
    sample_rate: int,
) -> bytes:
    """Create the receiver-compatible stereo WAV transmission."""
    import io
    import soundfile as sf

    coeffs = np.asarray(scrambled_packet.coeffs[0], dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(coeffs))) if coeffs.size else 0.0
    coeff_scale = peak if peak > 0 else 1.0
    coeff_stream = (coeffs / coeff_scale).astype(np.float32)

    carrier_gain = DEFAULT_CARRIER_GAIN
    header = np.array(
        [TRANSMISSION_MAGIC, float(scrambled_packet.original_length),
         float(coeffs.size), float(peak), carrier_gain],
        dtype=np.float32,
    )
    coeff_channel = np.concatenate([header, coeff_stream * carrier_gain])

    audible = np.asarray(scrambled_audio, dtype=np.float32).reshape(-1)
    frames = max(audible.size, coeff_channel.size)
    stereo = np.zeros((frames, 2), dtype=np.float32)
    stereo[:audible.size, 0] = np.clip(audible, -1.0, 1.0)
    stereo[:coeff_channel.size, 1] = np.clip(coeff_channel, -1.0, 1.0)

    out = io.BytesIO()
    sf.write(out, stereo, int(sample_rate), format="WAV", subtype="FLOAT")
    return out.getvalue()


def load_transmission_wav(
    data: bytes,
    wavelet: str,
    level: int,
) -> tuple[WaveletPacket, np.ndarray, int]:
    """Load and validate a receiver-compatible transmission WAV."""
    import io
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(
            io.BytesIO(data), always_2d=True, dtype="float64"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable WAV audio file.") from exc

    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError("This transmission must be the stereo WAV generated by the transmitter.")

    scrambled_audio = np.asarray(audio[:, 0], dtype=np.float64)
    coeff_channel = np.asarray(audio[:, 1], dtype=np.float64).reshape(-1)
    if coeff_channel.size < TRANSMISSION_HEADER_SIZE:
        raise ValueError("Transmission WAV is missing its internal transmission header.")
    if not np.isclose(coeff_channel[0], TRANSMISSION_MAGIC, atol=1e-5, rtol=0.0):
        raise ValueError("This WAV is stereo, but it is not a Wavelet Voice Lab transmission.")

    original_length = int(round(coeff_channel[1]))
    coefficient_count = int(round(coeff_channel[2]))
    coefficient_scale = float(coeff_channel[3])
    carrier_gain = float(coeff_channel[4])
    if original_length <= 0 or coefficient_count <= 0:
        raise ValueError("Transmission WAV contains invalid transmission metadata.")
    if coefficient_scale <= 0 or carrier_gain <= 0:
        raise ValueError("Transmission WAV contains invalid coefficient metadata.")

    end = TRANSMISSION_HEADER_SIZE + coefficient_count
    if end > coeff_channel.size:
        raise ValueError("Transmission WAV coefficient payload is incomplete.")
    coeffs = coeff_channel[TRANSMISSION_HEADER_SIZE:end] / carrier_gain

    try:
        layout = decompose(np.zeros(original_length, dtype=np.float64), wavelet, level)
    except ValueError as exc:
        raise ValueError(
            f"The selected wavelet/DWT level cannot decode this transmission: {exc}"
        ) from exc
    if layout.coeffs[0].size != coefficient_count:
        raise ValueError("The selected wavelet/DWT level does not match the transmission WAV.")

    packet = WaveletPacket(
        coeffs=[coeffs], slices=layout.slices,
        original_length=original_length, wavelet=wavelet, level=level,
    )
    return packet, scrambled_audio[:original_length], int(sample_rate)
