from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from src.audio import load_audio_bytes, synthetic_voice_like
from src.channel import add_awgn, packet_loss
from src.metrics import correlation, mse, snr_db
from src.wavelet_codec import (
    WaveletPacket,
    decompose,
    descramble,
    make_permutation,
    reconstruct,
    scramble,
    threshold,
)

st.set_page_config(page_title="Wavelet Voice Lab", layout="wide")
st.title("Wavelet-Domain Robust Voice Communication Lab")
st.caption(
    "Academic simulation: transmitter → DWT → keyed coefficient scrambling → "
    "noisy channel → receiver descrambling → IDWT"
)


def audio_bytes(signal: np.ndarray, sr: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, np.asarray(signal, dtype=np.float32), sr, format="WAV")
    return out.getvalue()


def safe_normalize(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / peak if peak > 1.0 else x




def coefficient_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    return float(np.dot(a0, b0) / denom) if denom > 0 else 0.0


def permutation_change_rate(permutation: np.ndarray) -> float:
    permutation = np.asarray(permutation, dtype=np.int64).reshape(-1)
    if permutation.size == 0:
        return 0.0
    return float(np.mean(permutation != np.arange(permutation.size)))


def wrong_key_reconstruction(
    scrambled_packet: WaveletPacket,
    key: int,
) -> np.ndarray:
    wrong_permutation = make_permutation(
        scrambled_packet.coeffs[0].size,
        int(key),
    )
    wrong_packet = descramble(scrambled_packet, wrong_permutation)
    return safe_normalize(reconstruct(wrong_packet))


def packet_with_coeffs(packet: WaveletPacket, coeffs: np.ndarray) -> WaveletPacket:
    return WaveletPacket(
        coeffs=[np.asarray(coeffs, dtype=np.float64)],
        slices=packet.slices,
        original_length=packet.original_length,
        wavelet=packet.wavelet,
        level=packet.level,
    )


with st.sidebar:
    st.header("Shared Transmitter / Receiver Configuration")
    wavelet = st.selectbox(
        "Wavelet",
        ["haar", "db2", "db4", "db8", "sym4", "coif1"],
        index=2,
    )
    level = st.slider("DWT level", 1, 6, 4)
    threshold_fraction = st.slider(
        "Coefficient threshold",
        0.0,
        0.20,
        0.02,
        0.005,
    )
    key = st.number_input(
        "Shared scrambling key",
        min_value=0,
        max_value=2_147_483_647,
        value=2026,
        step=1,
        help="Both transmitter and receiver must use the same key.",
    )

    st.divider()
    st.subheader("Simulated Channel")
    snr = st.slider("Channel SNR (dB)", -5.0, 40.0, 20.0, 1.0)
    loss = st.slider(
        "Packet/sample loss probability",
        0.0,
        0.20,
        0.0,
        0.005,
    )
    seed = st.number_input("Channel random seed", min_value=0, value=42, step=1)

uploaded = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "m4a", "mp3", "flac", "aac"],
    help="M4A/MP3/AAC are decoded automatically and normalized to 16 kHz mono PCM.",
)

if uploaded is None:
    original = synthetic_voice_like()
    sr = 16000
    source_name = "Synthetic demo signal"
    st.info("No file uploaded. A deterministic speech-like signal is being used.")
else:
    try:
        original, sr = load_audio_bytes(
            uploaded.getvalue(),
            uploaded.name,
            target_sr=16000,
        )
        source_name = uploaded.name
        st.success(f"Loaded {uploaded.name} → {sr:,} Hz mono PCM")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

original = safe_normalize(original)

try:
    # -------------------------
    # TRANSMITTER
    # -------------------------
    packet = decompose(original, wavelet, level)
    thresholded_packet, retained = threshold(packet, threshold_fraction)

    # Scramble the flattened wavelet coefficient stream.
    scrambled_packet, permutation = scramble(thresholded_packet, int(key))

    # This is intentionally NOT reconstructed before scrambling is reversed.
    # Reconstructing the original thresholded coefficients here would produce
    # speech-like audio and would not represent the intercepted signal.
    scrambled_audio = safe_normalize(reconstruct(scrambled_packet))

    # -------------------------
    # CHANNEL
    # -------------------------
    rng = np.random.default_rng(int(seed))
    received_scrambled_coeffs = add_awgn(
        scrambled_packet.coeffs[0],
        snr,
        rng,
    )
    received_scrambled_coeffs = packet_loss(
        received_scrambled_coeffs,
        loss,
        rng,
    )
    received_scrambled_packet = packet_with_coeffs(
        scrambled_packet,
        received_scrambled_coeffs,
    )

    # -------------------------
    # RECEIVER
    # -------------------------
    recovered_coeff_packet = descramble(
        received_scrambled_packet,
        permutation,
    )
    reconstructed = safe_normalize(reconstruct(recovered_coeff_packet))

except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("End-to-end communication stages")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 1. Original")
    st.audio(audio_bytes(original, sr), format="audio/wav")
    st.caption("Clear speech before transmission.")
    st.download_button(
        "Download original WAV",
        audio_bytes(original, sr),
        "original_normalized.wav",
        "audio/wav",
        key="download_original",
    )

with c2:
    st.markdown("### 2. Transmitted / scrambled")
    st.audio(audio_bytes(scrambled_audio, sr), format="audio/wav")
    st.caption(
        "What an interceptor hears: wavelet coefficients are permuted "
        "with the shared key before transmission."
    )
    st.download_button(
        "Download scrambled WAV",
        audio_bytes(scrambled_audio, sr),
        "transmitted_scrambled.wav",
        "audio/wav",
        key="download_scrambled",
    )

with c3:
    st.markdown("### 3. Received scrambled")
    st.audio(audio_bytes(
        safe_normalize(reconstruct(received_scrambled_packet)),
        sr,
    ), format="audio/wav")
    st.caption(
        f"Scrambled signal after channel: {snr:.1f} dB SNR, "
        f"loss={loss:.3f}."
    )
    st.download_button(
        "Download received WAV",
        audio_bytes(
            safe_normalize(reconstruct(received_scrambled_packet)),
            sr,
        ),
        "received_scrambled.wav",
        "audio/wav",
        key="download_received",
    )

with c4:
    st.markdown("### 4. Receiver recovered")
    st.audio(audio_bytes(reconstructed, sr), format="audio/wav")
    st.caption(
        "Receiver reverses the coefficient permutation and applies IDWT "
        "using the same wavelet configuration."
    )
    st.download_button(
        "Download recovered WAV",
        audio_bytes(reconstructed, sr),
        "receiver_recovered.wav",
        "audio/wav",
        key="download_recovered",
    )

st.success(
    f"Receiver configuration: wavelet={wavelet}, level={level}, "
    f"threshold={threshold_fraction:.3f}, shared key={int(key)}"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recovered SNR", f"{snr_db(original, reconstructed):.2f} dB")
m2.metric("Recovered MSE", f"{mse(original, reconstructed):.6g}")
m3.metric("Recovered correlation", f"{correlation(original, reconstructed):.4f}")
m4.metric("Coefficients retained", f"{retained * 100:.1f}%")

st.subheader("Waveform comparison")
preview_len = min(len(original), sr * 2)
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axes[0].plot(original[:preview_len])
axes[0].set_title("1. Original speech waveform")
axes[1].plot(scrambled_audio[:preview_len])
axes[1].set_title("2. Transmitted / scrambled waveform")
axes[2].plot(
    safe_normalize(reconstruct(received_scrambled_packet))[:preview_len]
)
axes[2].set_title("3. Received scrambled waveform")
axes[3].plot(reconstructed[:preview_len])
axes[3].set_title("4. Receiver-recovered waveform")
axes[3].set_xlabel("Sample")
for ax in axes:
    ax.set_ylabel("Amplitude")
st.pyplot(fig)
plt.close(fig)

st.subheader("Received wavelet coefficient distribution")
fig2, ax2 = plt.subplots(figsize=(11, 3))
ax2.plot(
    np.abs(
        received_scrambled_coeffs[
            : min(5000, received_scrambled_coeffs.size)
        ]
    )
)
ax2.set_title("Absolute magnitude of received scrambled coefficients")
ax2.set_xlabel("Scrambled coefficient index")
ax2.set_ylabel("Magnitude")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Experiment summary")
summary = pd.DataFrame(
    [
        {
            "input_file": source_name,
            "sample_rate_hz": sr,
            "wavelet": wavelet,
            "level": level,
            "threshold": threshold_fraction,
            "shared_key": int(key),
            "channel_snr_db": snr,
            "loss_probability": loss,
            "recovered_snr_db": snr_db(original, reconstructed),
            "recovered_mse": mse(original, reconstructed),
            "recovered_correlation": correlation(original, reconstructed),
            "retained_coefficients": retained,
        }
    ]
)
st.dataframe(summary, use_container_width=True)

st.info(
    "Research note: the keyed permutation is reversible signal obfuscation, "
    "not cryptographic encryption. It is included to demonstrate why the "
    "same transmitter/receiver configuration and key are required to recover "
    "the speech. For real confidentiality, use a standard authenticated "
    "encryption scheme."
)
