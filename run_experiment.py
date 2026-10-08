from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.audio import load_wav, synthetic_voice_like
from src.channel import add_awgn, packet_loss
from src.metrics import mse, snr_db, correlation
from src.wavelet_codec import decompose, descramble, make_permutation, reconstruct, scramble, threshold


def run(signal: np.ndarray, wavelet: str, level: int, threshold_fraction: float,
        channel_snr: float, loss_probability: float, key: int = 2026,
        seed: int = 42) -> dict:
    packet = decompose(signal, wavelet, level)
    packet, retained = threshold(packet, threshold_fraction)
    scrambled_packet, permutation = scramble(packet, key)

    rng = np.random.default_rng(seed)
    received_coeffs = add_awgn(scrambled_packet.coeffs[0], channel_snr, rng)
    received_coeffs = packet_loss(received_coeffs, loss_probability, rng)

    received_packet = type(scrambled_packet)(
        coeffs=[received_coeffs],
        slices=scrambled_packet.slices,
        original_length=scrambled_packet.original_length,
        wavelet=scrambled_packet.wavelet,
        level=scrambled_packet.level,
    )
    recovered_packet = descramble(
        received_packet,
        make_permutation(received_coeffs.size, key),
    )
    wrong_packet = descramble(
        received_packet,
        make_permutation(received_coeffs.size, (key + 1) % 2_147_483_648),
    )
    reconstructed = reconstruct(recovered_packet)
    wrong_reconstructed = reconstruct(wrong_packet)

    return {
        "wavelet": wavelet,
        "level": level,
        "threshold_fraction": threshold_fraction,
        "key": int(key),
        "channel_snr_db": channel_snr,
        "packet_loss_probability": loss_probability,
        "retained_coefficients": retained,
        "reconstruction_mse": mse(signal, reconstructed),
        "reconstruction_snr_db": snr_db(signal, reconstructed),
        "reconstruction_correlation": correlation(signal, reconstructed),
        "wrong_key_snr_db": snr_db(signal, wrong_reconstructed),
        "wrong_key_correlation": correlation(signal, wrong_reconstructed),
        "coefficients_reordered_fraction": float(
            np.mean(permutation != np.arange(permutation.size))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--key", type=int, default=2026)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="outputs/results.json")
    args = parser.parse_args()

    if args.input:
        signal, sr = load_wav(args.input)
    else:
        sr = 16000
        signal = synthetic_voice_like(sr=sr)

    if args.key < 0:
        parser.error("--key must be non-negative")
    result = run(
        signal,
        args.wavelet,
        args.level,
        args.threshold,
        args.snr_db,
        args.packet_loss,
        key=args.key,
    )
    result["sample_rate"] = sr
    result["samples"] = int(signal.size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
