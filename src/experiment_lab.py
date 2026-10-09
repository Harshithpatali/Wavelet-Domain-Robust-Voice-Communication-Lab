"""Bounded, reproducible channel and parameter experiments for the research tab."""
from __future__ import annotations

from itertools import product

import numpy as np

from src.audio import synthetic_voice_like
from src.channel import add_awgn
from src.fec import transmit_with_xor_fec
from src.metrics import correlation, mse, snr_db
from src.wavelet_codec import (
    decompose,
    descramble,
    make_permutation,
    reconstruct,
    scramble,
    threshold,
)


def simulate_coefficient_channel(
    coefficients: np.ndarray,
    snr_db_value: float,
    loss_probability: float,
    *,
    seed: int = 42,
    use_fec: bool = False,
    channel_model: str = "random",
    packet_size: int = 2048,
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Apply AWGN then packet erasures to float32 coefficients.

    FEC adds one XOR parity packet per group of four data packets. The payload
    is byte-framed only inside this simulation; it does not alter V3 WAV files.
    """
    if channel_model not in {"random", "burst"}:
        raise ValueError("channel_model must be 'random' or 'burst'")
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Coefficient stream is empty")
    if packet_size < 4 or packet_size % 4:
        raise ValueError("packet_size must be a positive multiple of four bytes")

    rng = np.random.default_rng(seed)
    noisy = add_awgn(values, float(snr_db_value), rng)
    wire_bytes = noisy.astype("<f4").tobytes(order="C")
    received_bytes, stats = transmit_with_xor_fec(
        wire_bytes,
        float(loss_probability),
        use_fec=use_fec,
        packet_size=packet_size,
        group_size=4,
        burst=(channel_model == "burst"),
        rng=rng,
    )
    received = np.frombuffer(received_bytes, dtype="<f4").astype(np.float64)
    if received.size < values.size:
        received = np.pad(received, (0, values.size - received.size))
    return received[: values.size], stats


def run_parameter_sweep(
    signal: np.ndarray,
    *,
    wavelets: tuple[str, ...] | list[str] = ("haar", "db4", "sym4"),
    levels: tuple[int, ...] | list[int] = (2, 3, 4),
    thresholds: tuple[float, ...] | list[float] = (0.0, 0.02),
    snr_values: tuple[float, ...] | list[float] = (0.0, 10.0, 20.0),
    loss_rates: tuple[float, ...] | list[float] = (0.0, 0.05),
    key: int = 2026,
    seed: int = 42,
    channel_model: str = "random",
    use_fec: bool = False,
    max_runs: int = 180,
    max_samples: int = 80000,
) -> list[dict[str, int | float | bool | str]]:
    """Run a capped parameter grid and return one metrics row per valid run."""
    arrays = [wavelets, levels, thresholds, snr_values, loss_rates]
    if any(len(values) == 0 for values in arrays):
        raise ValueError("Every sweep dimension must contain at least one value")
    count = int(np.prod([len(values) for values in arrays], dtype=np.int64))
    if count > max_runs:
        raise ValueError(
            f"This sweep contains {count} configurations; the safety limit is {max_runs}."
        )
    if channel_model not in {"random", "burst"}:
        raise ValueError("channel_model must be 'random' or 'burst'")
    if key < 0 or seed < 0:
        raise ValueError("key and seed must be non-negative")

    source = np.asarray(signal, dtype=np.float64).reshape(-1)
    if source.size == 0:
        source = synthetic_voice_like()
    source = source[:max_samples]
    rows: list[dict[str, int | float | bool | str]] = []

    for run_index, (wavelet, level, threshold_fraction, snr_value, loss_rate) in enumerate(
        product(wavelets, levels, thresholds, snr_values, loss_rates)
    ):
        if not 0 <= float(threshold_fraction) <= 1:
            raise ValueError("threshold fractions must be between 0 and 1")
        if not 0 <= float(loss_rate) <= 1:
            raise ValueError("loss rates must be between 0 and 1")
        try:
            packet = decompose(source, str(wavelet), int(level))
            thresholded, retained_fraction = threshold(packet, float(threshold_fraction))
        except ValueError:
            # A wavelet/level combination that is invalid for this signal is
            # skipped rather than terminating the entire parameter sweep.
            continue

        scrambled, permutation = scramble(thresholded, key=int(key))
        noisy_received, channel_stats = simulate_coefficient_channel(
            scrambled.coeffs[0],
            float(snr_value),
            float(loss_rate),
            seed=int(seed) + run_index,
            use_fec=use_fec,
            channel_model=channel_model,
        )
        received_packet = type(scrambled)(
            coeffs=[noisy_received],
            slices=scrambled.slices,
            original_length=scrambled.original_length,
            wavelet=scrambled.wavelet,
            level=scrambled.level,
        )
        recovered_packet = descramble(received_packet, permutation)
        estimate = reconstruct(recovered_packet)
        rows.append(
            {
                "wavelet": str(wavelet),
                "level": int(level),
                "threshold": float(threshold_fraction),
                "snr_db": float(snr_value),
                "loss_probability": float(loss_rate),
                "channel_model": channel_model,
                "fec_enabled": bool(use_fec),
                "retained_coefficients": float(retained_fraction),
                "recovered_snr_db": snr_db(source, estimate),
                "recovered_mse": mse(source, estimate),
                "recovered_correlation": correlation(source, estimate),
                "data_packets": int(channel_stats["data_packets"]),
                "lost_data_packets": int(channel_stats["lost_data_packets"]),
                "recovered_packets": int(channel_stats["recovered_packets"]),
                "unrecovered_packets": int(channel_stats["unrecovered_packets"]),
                "redundancy_percent": float(channel_stats["redundancy_percent"]),
            }
        )

    if not rows:
        raise ValueError(
            "No valid sweep configurations. Try a lower DWT level or a longer input."
        )
    return rows
