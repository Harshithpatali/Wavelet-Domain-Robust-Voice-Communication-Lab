import io

import numpy as np
import pytest
import soundfile as sf

from src.wavelet_codec import load_transmission_wav, make_transmission_wav
from src.wavelet_codec import decompose, descramble, make_permutation, reconstruct, scramble, threshold


def test_transmission_wav_round_trip():
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    original = 0.4 * np.sin(2 * np.pi * 220 * t)

    packet = decompose(original, "db4", 4)
    thresholded, _ = threshold(packet, 0.02)
    scrambled_packet, permutation = scramble(thresholded, 2026)
    scrambled_audio = reconstruct(scrambled_packet)

    data = make_transmission_wav(scrambled_packet, scrambled_audio, sr)
    raw, read_sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float64")

    assert read_sr == sr
    # V3 is intentionally mono: the recovery payload lives in a private RIFF chunk.
    assert raw.shape[1] == 1

    received_packet, received_scrambled, received_sr = load_transmission_wav(
        data, "db4", 4
    )
    assert received_sr == sr
    assert received_packet.coeffs[0].size == scrambled_packet.coeffs[0].size
    assert received_scrambled.size == original.size

    recovered_packet = descramble(
        received_packet,
        make_permutation(received_packet.coeffs[0].size, 2026),
    )
    recovered = reconstruct(recovered_packet)
    corr = np.corrcoef(original, recovered)[0, 1]
    assert corr > 0.999


def test_transmission_rejects_plain_mono_audio():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(8000, dtype=np.float32), 16000, format="WAV")

    with pytest.raises(ValueError, match="stereo WAV"):
        load_transmission_wav(buf.getvalue(), "db4", 4)


def test_transmission_rejects_stereo_non_transmission_wav():
    buf = io.BytesIO()
    stereo = np.zeros((8000, 2), dtype=np.float32)
    sf.write(buf, stereo, 16000, format="WAV")

    with pytest.raises(ValueError, match="not a Wavelet Voice Lab transmission"):
        load_transmission_wav(buf.getvalue(), "db4", 4)
