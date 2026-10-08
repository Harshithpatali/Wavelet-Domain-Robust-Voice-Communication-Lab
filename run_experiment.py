from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.audio import load_wav, synthetic_voice_like
from src.channel import add_awgn, packet_loss
from src.metrics import mse, snr_db, correlation
from src.wavelet_codec import decompose, threshold, reconstruct


def run(signal: np.ndarray, wavelet: str, level: int, threshold_fraction: float,
        channel_snr: float, loss_probability: float, seed: int = 42) -> dict:
    packet = decompose(signal, wavelet, level)
    packet, retained = threshold(packet, threshold_fraction)
    coeffs = packet.coeffs[0]
    rng = np.random.default_rng(seed)
    received = add_awgn(coeffs, channel_snr, rng)
    received = packet_loss(received, loss_probability, rng)
    packet.coeffs[0] = received
    reconstructed = reconstruct(packet)
    return {
        "wavelet": wavelet,
        "level": level,
        "threshold_fraction": threshold_fraction,
        "channel_snr_db": channel_snr,
        "packet_loss_probability": loss_probability,
        "retained_coefficients": retained,
        "mse": mse(signal, reconstructed),
        "reconstruction_snr_db": snr_db(signal, reconstructed),
        "correlation": correlation(signal, reconstructed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--level", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--packet-loss", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="outputs/results.json")
    args = parser.parse_args()

    if args.input:
        signal, sr = load_wav(args.input)
    else:
        sr = 16000
        signal = synthetic_voice_like(sr=sr)

    result = run(signal, args.wavelet, args.level, args.threshold,
                 args.snr_db, args.packet_loss)
    result["sample_rate"] = sr
    result["samples"] = int(signal.size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
