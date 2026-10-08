"""Backward-compatible transmission codec.

This module keeps the historical src.transmission import path working even
when an older Streamlit worker still imports it. The implementation is
self-contained and does not depend on newer transmission symbols being
present in src.wavelet_codec.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import soundfile as sf

from src.wavelet_codec import WaveletPacket, decompose

TRANSMISSION_MAGIC = 0.31415927
TRANSMISSION_VERSION_V2 = 2.0
TRANSMISSION_VERSION_V3 = 3.0
TRANSMISSION_HEADER_FLOATS = 8
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


def _append_chunk(wav_bytes: bytes, payload: bytes) -> bytes:
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Could not create a valid RIFF/WAV transmission.")
    padding = b"\x00" if len(payload) % 2 else b""
    chunk = TRANSMISSION_CHUNK_ID + struct.pack("<I", len(payload)) + payload + padding
    out = bytearray(wav_bytes)
    riff_size = struct.unpack_from("<I", out, 4)[0]
    struct.pack_into("<I", out, 4, riff_size + len(chunk))
    out.extend(chunk)
    return bytes(out)


def _find_chunk(wav_bytes: bytes) -> bytes | None:
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return None
    riff_size = struct.unpack_from("<I", wav_bytes, 4)[0]
    limit = min(len(wav_bytes), 8 + riff_size)
    pos = 12
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
    coeffs = np.asarray(scrambled_packet.coeffs[0], dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(coeffs))) if coeffs.size else 1.0
    scale = peak if peak > 0 else 1.0
    normalized = (coeffs / scale).astype("<f4")

    header = struct.pack(
        "<8f",
        float(TRANSMISSION_MAGIC),
        float(TRANSMISSION_VERSION_V3),
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
    sf.write(out, audible, int(sample_rate), format="WAV", subtype="FLOAT")
    return _append_chunk(out.getvalue(), payload)


def _decode_v3(chunk: bytes) -> tuple[WaveletPacket, int]:
    header_bytes = TRANSMISSION_HEADER_FLOATS * 4
    if len(chunk) < header_bytes:
        raise ValueError("Transmission WAV has an incomplete V3 recovery payload.")

    magic, version, length_f, count_f, scale, gain, level_f, wavelet_id = struct.unpack(
        "<8f", chunk[:header_bytes]
    )
    if not np.isclose(magic, TRANSMISSION_MAGIC, atol=1e-5, rtol=0.0):
        raise ValueError("This WAV contains an invalid Wavelet Voice Lab transmission payload.")
    if not np.isclose(version, TRANSMISSION_VERSION_V3, atol=1e-5, rtol=0.0):
        raise ValueError("Transmission WAV contains an unsupported transmission version.")

    original_length = int(round(length_f))
    count = int(round(count_f))
    level = int(round(level_f))
    if original_length <= 0 or count <= 0 or scale <= 0 or gain <= 0:
        raise ValueError("Transmission WAV contains invalid transmission metadata.")

    expected = header_bytes + count * 4
    if len(chunk) < expected:
        raise ValueError("Transmission WAV coefficient payload is incomplete.")

    normalized = np.frombuffer(
        chunk[header_bytes:expected], dtype="<f4", count=count
    ).astype(np.float64)
    coeffs = (normalized / gain) * scale
    wavelet = _wavelet_from_id(wavelet_id)

    layout = decompose(np.zeros(original_length, dtype=np.float64), wavelet, level)
    if layout.coeffs[0].size != count:
        raise ValueError("Transmission WAV codec metadata does not match the coefficient payload.")

    return (
        WaveletPacket(
            coeffs=[coeffs],
            slices=layout.slices,
            original_length=original_length,
            wavelet=wavelet,
            level=level,
        ),
        original_length,
    )


def _load_v2(audio: np.ndarray) -> tuple[WaveletPacket, np.ndarray]:
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError(
            "This WAV does not contain a Wavelet Voice Lab recovery payload. "
            "Upload the sender's voice_transmission.wav generated by the Download transmission WAV button."
        )

    carrier = np.asarray(audio[:, 1], dtype=np.float64).reshape(-1)
    if carrier.size < 5:
        raise ValueError("Transmission WAV is missing its internal transmission header.")
    if not np.isclose(carrier[0], TRANSMISSION_MAGIC, atol=1e-5, rtol=0.0):
        raise ValueError("This WAV is stereo, but it is not a Wavelet Voice Lab transmission.")

    if not np.isclose(carrier[1], TRANSMISSION_VERSION_V2, atol=1e-5, rtol=0.0):
        raise ValueError(
            "This is a legacy transmission WAV. Generate a new transmission WAV with the current app."
        )

    if carrier.size < 8:
        raise ValueError("Transmission WAV has an incomplete V2 header.")

    original_length = int(round(carrier[2]))
    count = int(round(carrier[3]))
    scale = float(carrier[4])
    gain = float(carrier[5])
    level = int(round(carrier[6]))
    wavelet = _wavelet_from_id(carrier[7])

    if original_length <= 0 or count <= 0 or scale <= 0 or gain <= 0:
        raise ValueError("Transmission WAV contains invalid transmission metadata.")

    end = 8 + count
    if end > carrier.size:
        raise ValueError("Transmission WAV coefficient payload is incomplete.")

    coeffs = (carrier[8:end] / gain) * scale
    layout = decompose(np.zeros(original_length, dtype=np.float64), wavelet, level)
    if layout.coeffs[0].size != count:
        raise ValueError("Transmission WAV codec metadata does not match the coefficient payload.")

    packet = WaveletPacket(
        coeffs=[coeffs],
        slices=layout.slices,
        original_length=original_length,
        wavelet=wavelet,
        level=level,
    )
    return packet, np.asarray(audio[:, 0], dtype=np.float64)[:original_length]


def load_transmission_wav(
    data: bytes, wavelet: str | None = None, level: int | None = None
) -> tuple[WaveletPacket, np.ndarray, int]:
    """Load V3 RIFF-payload WAVs and retain V2 stereo compatibility."""
    chunk = _find_chunk(data)
    if chunk is not None:
        packet, original_length = _decode_v3(chunk)
        try:
            audio, sample_rate = sf.read(
                io.BytesIO(data), always_2d=True, dtype="float64"
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise ValueError("The uploaded file is not a readable WAV audio file.") from exc
        return (
            packet,
            np.asarray(audio[:, 0], dtype=np.float64)[:original_length],
            int(sample_rate),
        )

    try:
        audio, sample_rate = sf.read(
            io.BytesIO(data), always_2d=True, dtype="float64"
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable WAV audio file.") from exc

    packet, scrambled_audio = _load_v2(audio)
    return packet, scrambled_audio, int(sample_rate)


__all__ = ["load_transmission_wav", "make_transmission_wav"]
