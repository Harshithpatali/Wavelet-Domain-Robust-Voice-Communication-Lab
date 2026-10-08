import numpy as np

from run_experiment import run
from src.audio import synthetic_voice_like


def test_cli_experiment_uses_keyed_round_trip():
    x = synthetic_voice_like(duration=0.25)
    result = run(
        x,
        wavelet="db4",
        level=4,
        threshold_fraction=0.02,
        channel_snr=40.0,
        loss_probability=0.0,
        key=2026,
        seed=42,
    )

    assert result["reconstruction_correlation"] > 0.99
    assert result["reconstruction_snr_db"] > 20.0
    assert 0 <= result["coefficients_reordered_fraction"] <= 1
    assert result["wrong_key_correlation"] < result["reconstruction_correlation"]
