from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _normalize_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio file is empty or has an unsupported shape")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float64)


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr != target_sr:
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr
    return audio.astype(np.float64), sr


def load_wav(path: str | os.PathLike[str] | io.BytesIO, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load WAV audio as mono float64 and resample to target_sr."""
    audio, sr = sf.read(path, always_2d=False)
    audio = _normalize_mono(audio)
    return _resample(audio, int(sr), target_sr)


def load_audio_bytes(data: bytes, filename: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load WAV/FLAC directly; decode M4A/MP3 and other FFmpeg formats via a bundled FFmpeg binary.

    The application normalizes everything to mono float64 PCM at target_sr before wavelet processing.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        try:
            audio, sr = sf.read(io.BytesIO(data), always_2d=False)
            audio = _normalize_mono(audio)
            return _resample(audio, int(sr), target_sr)
        except Exception:
            # Fall through to FFmpeg for formats supported by FFmpeg but not libsndfile.
            pass

    if suffix not in {".m4a", ".mp3", ".aac", ".mp4", ".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        raise ValueError(f"Unsupported audio format: {suffix or 'unknown'}")

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Audio decoding for M4A/MP3 requires imageio-ffmpeg. "
            "Run: pip install -r requirements.txt"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
        src.write(data)
        source_path = src.name

    try:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-i", source_path,
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-f", "wav", "-acodec", "pcm_s16le", "pipe:1",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"FFmpeg could not decode {filename}: {detail or 'unknown error'}")
        audio, sr = sf.read(io.BytesIO(result.stdout), always_2d=False)
        audio = _normalize_mono(audio)
        return _resample(audio, int(sr), target_sr)
    finally:
        try:
            os.unlink(source_path)
        except OSError:
            pass


def save_wav(path: str | os.PathLike[str], audio: np.ndarray, sr: int = 16000) -> None:
    """Save normalized floating-point mono audio."""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    sf.write(path, audio, sr)


def synthetic_voice_like(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate a deterministic speech-like signal for tests/demo use."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.1 * t)
    carrier = (
        0.55 * np.sin(2 * np.pi * 180 * t)
        + 0.28 * np.sin(2 * np.pi * 360 * t)
        + 0.14 * np.sin(2 * np.pi * 720 * t)
    )
    mod = 0.25 * np.sin(2 * np.pi * 4.7 * t)
    x = (envelope * carrier) + mod
    x *= np.hanning(n)
    return (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float64)
