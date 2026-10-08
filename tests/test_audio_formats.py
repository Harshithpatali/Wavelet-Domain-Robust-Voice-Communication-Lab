import io
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from src.audio import load_audio_bytes, synthetic_voice_like


def test_wav_bytes_are_loaded_and_normalized():
    x = synthetic_voice_like(duration=0.25, sr=8000)
    buf = io.BytesIO()
    sf.write(buf, x, 8000, format="WAV")
    y, sr = load_audio_bytes(buf.getvalue(), "sample.wav", target_sr=16000)
    assert sr == 16000
    assert y.ndim == 1
    assert y.size > 0


def test_m4a_bytes_are_decoded_when_ffmpeg_is_available():
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    x = synthetic_voice_like(duration=0.25, sr=16000)
    wav = io.BytesIO()
    sf.write(wav, x, 16000, format="WAV")

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as out_file:
        m4a_path = out_file.name

    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-c:a", "aac", "-b:a", "96k", "-f", "ipod", m4a_path],
            input=wav.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
        with open(m4a_path, "rb") as f:
            m4a_bytes = f.read()
    finally:
        import os
        try:
            os.unlink(m4a_path)
        except OSError:
            pass

    y, sr = load_audio_bytes(m4a_bytes, "sample.m4a", target_sr=16000)
    assert sr == 16000
    assert y.ndim == 1
    assert y.size > 0
    assert np.max(np.abs(y)) <= 1.0 + 1e-9
