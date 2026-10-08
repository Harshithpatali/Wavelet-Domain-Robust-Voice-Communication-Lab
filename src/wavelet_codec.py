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
TRANSMISSION_VERSION = 2.0
TRANSMISSION_V3_VERSION = 3.0
TRANSMISSION_HEADER_SIZE_V1 = 5
TRANSMISSION_HEADER_SIZE_V2 = 8
DEFAULT_CARRIER_GAIN = 1e-3
TRANSMISSION_WAVELETS = ("haar", "db2", "db4", "db8", "sym4", "coif1")
TRANSMISSION_CHUNK_ID = b"WVTP"


def _wavelet_id(wavelet: str) -> int:
    if wavelet not in TRANSMISSION_WAVELETS:
        raise ValueError(f"Wavelet '{wavelet}' is not supported by the transmission format.")
    return TRANSMISSION_WAVELETS.index(wavelet) + 1


def _wavelet_from_id(value: float) -> str:
    idx = int(round(value))
    if not np.isclose(value, idx, atol=1e-4, rtol=0.0):
        raise ValueError("Transmission WAV contains an invalid wavelet identifier.")
    if not 1 <= idx <= len(TRANSMISSION_WAVELETS):
        raise ValueError("Transmission WAV contains an unknown wavelet identifier.")
    return TRANSMISSION_WAVELETS[idx - 1]


def _append_transmission_chunk(wav_bytes: bytes, chunk_payload: bytes) -> bytes:
    """Append a private RIFF chunk without changing the playable WAV audio."""
    import struct

    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Could not create a valid RIFF/WAV transmission.")
    padding = b"\x00" if len(chunk_payload) % 2 else b""
    chunk = TRANSMISSION_CHUNK_ID + struct.pack("<I", len(chunk_payload)) + chunk_payload + padding
    out = bytearray(wav_bytes)
    riff_size = struct.unpack_from("<I", out, 4)[0]
    struct.pack_into("<I", out, 4, riff_size + len(chunk))
    out.extend(chunk)
    return bytes(out)


def _find_transmission_chunk(wav_bytes: bytes) -> bytes | None:
    """Find the private WVTP chunk in a RIFF/WAV file."""
    import struct

    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return None
    pos = 12
    limit = min(len(wav_bytes), 8 + struct.unpack_from("<I", wav_bytes, 4)[0])
    while pos + 8 <= limit:
        chunk_id = wav_bytes[pos:pos + 4]
        size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        start = pos + 8
        end = start + size
        if end > len(wav_bytes):
            return None
        if chunk_id == TRANSMISSION_CHUNK_ID:
            return wav_bytes[start:end]
        pos = end + (size % 2)
    return None


def make_transmission_wav(scrambled_packet: WaveletPacket, scrambled_audio: np.ndarray, sample_rate: int) -> bytes:
    """Create a portable mono WAV with a private RIFF recovery payload.

    V3 keeps the spoken transmission as ordinary mono WAV audio. The scrambled
    coefficient stream is stored in a private RIFF chunk, so media players,
    Streamlit previews, and normal WAV handling cannot silently drop the
    recovery payload by downmixing or exposing a second audio channel.
    """
    import io
    import struct
    import soundfile as sf

    coeffs = np.asarray(scrambled_packet.coeffs[0], dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(coeffs))) if coeffs.size else 1.0
    scale = peak if peak > 0 else 1.0
    normalized = (coeffs / scale).astype(np.float32)

    # V3 payload is self-describing and does not contain the secret key.
    header = struct.pack(
        "<8f",
        float(TRANSMISSION_MAGIC),
        float(TRANSMISSION_V3_VERSION),
        float(scrambled_packet.original_length),
        float(coeffs.size),
        float(scale),
        float(DEFAULT_CARRIER_GAIN),
        float(scrambled_packet.level),
        float(_wavelet_id(scrambled_packet.wavelet)),
    )
    payload = header + normalized.tobytes(order="C")

    audible = np.asarray(scrambled_audio, dtype=np.float32).reshape(-1)
    audible = np.clip(audible, -1.0, 1.0)

    out = io.BytesIO()
    # The WAV itself contains only the playable scrambled voice.
    sf.write(out, audible, int(sample_rate), format="WAV", subtype="FLOAT")
    return _append_transmission_chunk(out.getvalue(), payload)


def _decode_v3_payload(payload: bytes) -> tuple[WaveletPacket, int]:
    import struct

    header_bytes = 8 * 4
    if len(payload) < header_bytes:
        raise ValueError("Transmission WAV has an incomplete V3 recovery payload.")

    values = struct.unpack("<8f", payload[:header_bytes])
    magic, version, original_length_f, count_f, scale, gain, level_f, wavelet_id = values

    if not np.isclose(magic, TRANSMISSION_MAGIC, atol=1e-5, rtol=0.0):
        raise ValueError("This WAV contains an invalid Wavelet Voice Lab transmission payload.")
    if not np.isclose(version, TRANSMISSION_V3_VERSION, atol=1e-5, rtol=0.0):
        raise ValueError("Transmission WAV contains an unsupported transmission version.")

    original_length = int(round(original_length_f))
    count = int(round(count_f))
    level = int(round(level_f))
    if original_length <= 0 or count <= 0 or scale <= 0 or gain <= 0:
        raise ValueError("Transmission WAV contains invalid transmission metadata.")

    expected = header_bytes + count * 4
    if len(payload) < expected:
        raise ValueError("Transmission WAV coefficient payload is incomplete.")

    coeffs_normalized = np.frombuffer(
        payload[header_bytes:expected], dtype="<f4", count=count
    ).astype(np.float64)
    coeffs = (coeffs_normalized / gain) * scale
    wavelet = _wavelet_from_id(wavelet_id)

    layout = decompose(np.zeros(original_length, dtype=np.float64), wavelet, level)
    if layout.coeffs[0].size != count:
        raise ValueError("Transmission WAV codec metadata does not match the coefficient payload.")

    return (
        WaveletPacket(
            coeffs=[coeffs], slices=layout.slices,
            original_length=original_length, wavelet=wavelet, level=level,
        ),
        original_length,
    )


def load_transmission_wav(
    data: bytes, wavelet: str | None = None, level: int | None = None
) -> tuple[WaveletPacket, np.ndarray, int]:
    """Load V3 mono RIFF transmissions and retain V2/V1 compatibility."""
    import io
    import soundfile as sf

    # First look for V3's private RIFF payload. This works regardless of
    # whether a WAV has one or two audio channels.
    chunk = _find_transmission_chunk(data)
    if chunk is not None:
        packet, original_length = _decode_v3_payload(chunk)
        try:
            audio, sample_rate = sf.read(
                io.BytesIO(data), always_2d=True, dtype="float64"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise ValueError("The uploaded file is not a readable WAV audio file.") from exc
        scrambled_audio = np.asarray(audio[:, 0], dtype=np.float64)[:original_length]
        return packet, scrambled_audio, int(sample_rate)

    # Backward-compatible V2/V1 reader for transmissions created before V3.
    try:
        audio, sample_rate = sf.read(io.BytesIO(data), always_2d=True, dtype="float64")
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable WAV audio file.") from exc

    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError(
            "This WAV does not contain a Wavelet Voice Lab recovery payload. "
            "Upload the sender's **voice_transmission.wav** generated by the Download transmission WAV button."
        )

    carrier = np.asarray(audio[:, 1], dtype=np.float64).reshape(-1)
    if carrier.size < TRANSMISSION_HEADER_SIZE_V1:
        raise ValueError("Transmission WAV is missing its internal transmission header.")
    if not np.isclose(carrier[0], TRANSMISSION_MAGIC, atol=1e-5, rtol=0.0):
        raise ValueError("This WAV is stereo, but it is not a Wavelet Voice Lab transmission.")

    is_v2 = np.isclose(carrier[1], TRANSMISSION_VERSION, atol=1e-5, rtol=0.0)
    if is_v2:
        if carrier.size < TRANSMISSION_HEADER_SIZE_V2:
            raise ValueError("Transmission WAV has an incomplete V2 header.")
        original_length = int(round(carrier[2]))
        count = int(round(carrier[3]))
        scale = float(carrier[4])
        gain = float(carrier[5])
        level = int(round(carrier[6]))
        wavelet = _wavelet_from_id(carrier[7])
        header_size = TRANSMISSION_HEADER_SIZE_V2
    else:
        if wavelet is None or level is None:
            raise ValueError(
                "This is a legacy transmission WAV. Generate a new transmission WAV with the current app."
            )
        original_length = int(round(carrier[1]))
        count = int(round(carrier[2]))
        scale = float(carrier[3])
        gain = float(carrier[4])
        header_size = TRANSMISSION_HEADER_SIZE_V1

    if original_length <= 0 or count <= 0 or scale <= 0 or gain <= 0:
        raise ValueError("Transmission WAV contains invalid transmission metadata.")
    end = header_size + count
    if end > carrier.size:
        raise ValueError("Transmission WAV coefficient payload is incomplete.")

    coeffs = (carrier[header_size:end] / gain) * scale
    layout = decompose(np.zeros(original_length, dtype=np.float64), wavelet, level)
    if layout.coeffs[0].size != count:
        raise ValueError("Transmission WAV codec metadata does not match the coefficient payload.")

    packet = WaveletPacket(
        coeffs=[coeffs], slices=layout.slices,
        original_length=original_length, wavelet=wavelet, level=level,
    )
    scrambled_audio = np.asarray(audio[:, 0], dtype=np.float64)[:original_length]
    return packet, scrambled_audio, int(sample_rate)
