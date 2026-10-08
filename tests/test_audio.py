from src.audio import synthetic_voice_like


def test_synthetic_signal():
    x = synthetic_voice_like(duration=1.0, sr=16000)
    assert len(x) == 16000
    assert x.max() <= 1.0
    assert x.min() >= -1.0
