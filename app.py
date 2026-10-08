from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf

from src.audio import load_audio_bytes, synthetic_voice_like
from src.channel import add_awgn, packet_loss
from src.metrics import mse, snr_db, correlation
from src.wavelet_codec import decompose, threshold, reconstruct

st.set_page_config(page_title="Wavelet Voice Lab", layout="wide")
st.title("Wavelet-Domain Robust Voice Communication Lab")
st.caption("Academic simulation: speech → DWT → simulated channel → IDWT")

def audio_bytes(signal: np.ndarray, sr: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, np.asarray(signal, dtype=np.float32), sr, format="WAV")
    return out.getvalue()

def safe_normalize(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return x / peak if peak > 1.0 else x

with st.sidebar:
    st.header("Experiment")
    wavelet = st.selectbox("Wavelet", ["haar", "db2", "db4", "db8", "sym4", "coif1"], index=2)
    level = st.slider("DWT level", 1, 6, 4)
    threshold_fraction = st.slider("Coefficient threshold", 0.0, 0.20, 0.02, 0.005)
    snr = st.slider("Channel SNR (dB)", -5.0, 40.0, 20.0, 1.0)
    loss = st.slider("Packet/sample loss probability", 0.0, 0.20, 0.0, 0.005)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

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
        original, sr = load_audio_bytes(uploaded.getvalue(), uploaded.name, target_sr=16000)
        source_name = uploaded.name
        st.success(f"Loaded {uploaded.name} → {sr:,} Hz mono PCM")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

original = safe_normalize(original)

try:
    packet = decompose(original, wavelet, level)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# A direct reconstruction of the processed wavelet coefficients gives the
# intermediate "decomposed-domain audio" that can be auditioned.
thresholded_packet, retained = threshold(packet, threshold_fraction)
decomposed_audio = safe_normalize(reconstruct(thresholded_packet))

rng = np.random.default_rng(int(seed))
received_packet = thresholded_packet
received_coeffs = add_awgn(received_packet.coeffs[0], snr, rng)
received_coeffs = packet_loss(received_coeffs, loss, rng)

# Receiver-side inverse transform after the simulated channel.
received_packet = type(thresholded_packet)(
    coeffs=[received_coeffs],
    slices=thresholded_packet.slices,
    original_length=thresholded_packet.original_length,
    wavelet=thresholded_packet.wavelet,
    level=thresholded_packet.level,
)
reconstructed = safe_normalize(reconstruct(received_packet))

st.subheader("Listen to every processing stage")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1. Original audio**")
    st.audio(audio_bytes(original, sr), format="audio/wav")
    st.caption(source_name)
    st.download_button(
        "Download original as WAV",
        audio_bytes(original, sr),
        "original_normalized.wav",
        "audio/wav",
        key="download_original",
    )

with c2:
    st.markdown("**2. Wavelet-domain processed audio**")
    st.audio(audio_bytes(decomposed_audio, sr), format="audio/wav")
    st.caption(f"DWT: {wavelet}, level {level}, threshold {threshold_fraction:.3f}")
    st.download_button(
        "Download decomposed-stage WAV",
        audio_bytes(decomposed_audio, sr),
        "decomposed_stage.wav",
        "audio/wav",
        key="download_decomposed",
    )

with c3:
    st.markdown("**3. Final reconstructed audio**")
    st.audio(audio_bytes(reconstructed, sr), format="audio/wav")
    st.caption(f"Channel: {snr:.1f} dB SNR, loss={loss:.3f}")
    st.download_button(
        "Download final reconstructed WAV",
        audio_bytes(reconstructed, sr),
        "reconstructed.wav",
        "audio/wav",
        key="download_final",
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Reconstruction SNR", f"{snr_db(original, reconstructed):.2f} dB")
m2.metric("MSE", f"{mse(original, reconstructed):.6g}")
m3.metric("Correlation", f"{correlation(original, reconstructed):.4f}")
m4.metric("Coefficients retained", f"{retained*100:.1f}%")

st.subheader("Waveform comparison")
preview_len = min(len(original), sr * 2)
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
axes[0].plot(original[:preview_len])
axes[0].set_title("Original waveform")
axes[1].plot(decomposed_audio[:preview_len])
axes[1].set_title("Wavelet-domain processed / decomposed-stage waveform")
axes[2].plot(reconstructed[:preview_len])
axes[2].set_title("Final reconstructed waveform")
axes[2].set_xlabel("Sample")
for ax in axes:
    ax.set_ylabel("Amplitude")
st.pyplot(fig)
plt.close(fig)

st.subheader("Wavelet coefficient distribution")
fig2, ax2 = plt.subplots(figsize=(11, 3))
ax2.plot(np.abs(received_coeffs[: min(5000, received_coeffs.size)]))
ax2.set_title("Absolute received coefficient magnitude")
ax2.set_xlabel("Coefficient index")
ax2.set_ylabel("Magnitude")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Experiment summary")
summary = pd.DataFrame([{
    "input_file": source_name,
    "sample_rate_hz": sr,
    "wavelet": wavelet,
    "level": level,
    "threshold": threshold_fraction,
    "channel_snr_db": snr,
    "loss_probability": loss,
    "reconstruction_snr_db": snr_db(original, reconstructed),
    "mse": mse(original, reconstructed),
    "correlation": correlation(original, reconstructed),
    "retained_coefficients": retained,
}])
st.dataframe(summary, use_container_width=True)
