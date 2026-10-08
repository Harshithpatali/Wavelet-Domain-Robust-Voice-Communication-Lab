import numpy as np
from src.channel import add_awgn, packet_loss


def test_awgn_is_deterministic_with_seed():
    x = np.ones(1000)
    a = add_awgn(x, 10, np.random.default_rng(1))
    b = add_awgn(x, 10, np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_packet_loss_zero_probability_is_identity():
    x = np.arange(100.0)
    y = packet_loss(x, 0.0, np.random.default_rng(1))
    assert np.array_equal(x, y)
