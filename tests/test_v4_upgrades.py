import numpy as np
import pytest

from src.crypto import decrypt_transmission, encrypt_transmission
from src.experiment_lab import run_parameter_sweep, simulate_coefficient_channel
from src.fec import encode_xor, recover_xor_packets, transmit_with_xor_fec
from src.packet import DataPacket, packetize, reassemble


def test_packet_round_trip_and_out_of_order_frames():
    raw = bytes(range(256)) * 17
    frames = packetize(raw, payload_size=333)
    assert reassemble(list(reversed(frames))) == raw


def test_packet_checksum_detects_corruption():
    frame = bytearray(packetize(b"hello world", payload_size=32)[0])
    frame[-1] ^= 0x01
    with pytest.raises(ValueError, match="checksum"):
        DataPacket.decode(bytes(frame))


def test_packet_reassembly_rejects_missing_packet():
    frames = packetize(b"x" * 500, payload_size=100)
    with pytest.raises(ValueError, match="incomplete"):
        reassemble(frames[:-1])


def test_xor_fec_recovers_one_lost_packet_per_group():
    raw = bytes((i * 7) % 256 for i in range(8192))
    frame = encode_xor(raw, packet_size=256, group_size=4)
    erased_data = [False] * len(frame.data_packets)
    erased_data[1] = True
    erased_data[6] = True
    erased_parity = [False] * len(frame.parity_packets)

    recovered, stats = recover_xor_packets(frame, erased_data, erased_parity)
    assert recovered == raw
    assert stats["recovered_packets"] == 2
    assert stats["unrecovered_packets"] == 0

def test_channel_simulation_is_reproducible_and_preserves_length():
    raw = bytes(range(251)) * 12
    a, stats_a = transmit_with_xor_fec(
        raw, 0.25, use_fec=True, packet_size=251, group_size=4,
        rng=np.random.default_rng(11),
    )
    b, stats_b = transmit_with_xor_fec(
        raw, 0.25, use_fec=True, packet_size=251, group_size=4,
        rng=np.random.default_rng(11),
    )
    assert len(a) == len(raw)
    assert a == b
    assert stats_a == stats_b


def test_aes_gcm_round_trip_and_authentication():
    raw = b"RIFF" + bytes(range(128)) * 4
    encrypted = encrypt_transmission(raw, "separate receiver passphrase")
    assert encrypted != raw
    assert decrypt_transmission(encrypted, "separate receiver passphrase") == raw
    with pytest.raises(ValueError, match="Authentication failed"):
        decrypt_transmission(encrypted, "wrong passphrase")


def test_aes_gcm_detects_tampering():
    encrypted = bytearray(encrypt_transmission(b"payload", "correct horse"))
    encrypted[-1] ^= 1
    with pytest.raises(ValueError, match="Authentication failed"):
        decrypt_transmission(bytes(encrypted), "correct horse")


def test_parameter_sweep_is_bounded_and_returns_metrics():
    signal = np.sin(np.linspace(0, 30, 16000))
    rows = run_parameter_sweep(
        signal,
        wavelets=["db4"],
        levels=[2],
        thresholds=[0.0, 0.02],
        snr_values=[20.0],
        loss_rates=[0.0, 0.05],
        seed=7,
        use_fec=True,
        max_runs=8,
    )
    assert len(rows) == 4
    assert all("recovered_snr_db" in row for row in rows)
    assert all(row["fec_enabled"] for row in rows)


def test_parameter_sweep_rejects_excessive_grid():
    with pytest.raises(ValueError, match="safety limit"):
        run_parameter_sweep(
            np.ones(1000),
            wavelets=["haar", "db2"],
            levels=[1, 2],
            thresholds=[0.0, 0.1],
            snr_values=[0.0, 10.0],
            loss_rates=[0.0, 0.1],
            max_runs=4,
        )


def test_coefficient_channel_returns_same_length_and_packet_stats():
    values = np.linspace(-1, 1, 4096)
    noisy, stats = simulate_coefficient_channel(
        values, 20.0, 0.05, seed=5, use_fec=True, channel_model="burst"
    )
    assert noisy.shape == values.shape
    assert stats["fec_enabled"] is True
    assert stats["redundancy_percent"] > 0
