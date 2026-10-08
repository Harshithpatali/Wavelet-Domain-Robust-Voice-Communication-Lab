from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.audio import load_audio_bytes, synthetic_voice_like
from src.channel import add_awgn, packet_loss
from src.metrics import mse, snr_db, correlation
from src.wavelet_codec import decompose, threshold, reconstruct

st.set_page_config(page_title="Wavelet Voice Lab", layout="wide")
st.title("Wavelet-Domain Robust Voice Communication Lab")
st.caption("Academic simulation: speech → DWT → simulated channel → IDWT")

with st.sidebar:
    st.header("Experiment")
    wavelet = st.selectbox("Wavelet", ["haar", "db2", "db4", "db8", "sym4", "coif1"], index=2)
    level = st.slider("DWT level", 1, 6, 4)
    threshold_fraction = st.slider("Coefficient threshold", 0.0, 0.20, 0.02, 0.005)
    snr = st.slider("Channel SNR (dB)", -5.0, 40.0, 20.0, 1.0)
    loss = st.slider("Packet/sample loss probability", 0.0, 0.20, 0.0, 0.005)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

uploaded = st.file_uploader(
    "Upload a voice recording (optional)",
    type=["wav", "m4a", "mp3", "flac", "aac"],
    help="M4A/MP3/AAC files are decoded automatically with the bundled FFmpeg runtime; processing is normalized to 16 kHz mono PCM.",
)

if uploaded is None:
    signal = synthetic_voice_like()
    sr = 16000
    st.info("No file uploaded. A deterministic synthetic speech-like signal is being used.")
else:
    try:
        signal, sr = load_audio_bytes(uploaded.getvalue(), uploaded.name, target_sr=16000)
        st.success(f"Loaded {uploaded.name} → {sr:,} Hz mono PCM")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

try:
    packet = decompose(signal, wavelet, level)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

packet, retained = threshold(packet, threshold_fraction)
rng = np.random.default_rng(int(seed))
received = add_awgn(packet.coeffs[0], snr, rng)
received = packet_loss(received, loss, rng)
packet.coeffs[0] = received
reconstructed = reconstruct(packet)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Reconstruction SNR", f"{snr_db(signal, reconstructed):.2f} dB")
m2.metric("MSE", f"{mse(signal, reconstructed):.6g}")
m3.metric("Correlation", f"{correlation(signal, reconstructed):.4f}")
m4.metric("Coefficients retained", f"{retained*100:.1f}%")

fig1, ax1 = plt.subplots(figsize=(11, 3))
ax1.plot(signal[: min(len(signal), sr * 2)])
ax1.set_title("Original waveform")
ax1.set_xlabel("Sample")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(11, 3))
ax2.plot(reconstructed[: min(len(reconstructed), sr * 2)])
ax2.set_title("Reconstructed waveform")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Amplitude")
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Wavelet coefficient distribution")
fig3, ax3 = plt.subplots(figsize=(11, 3))
ax3.plot(np.abs(packet.coeffs[0][: min(5000, packet.coeffs[0].size)]))
ax3.set_title("Absolute received coefficient magnitude")
ax3.set_xlabel("Coefficient index")
ax3.set_ylabel("Magnitude")
st.pyplot(fig3)
plt.close(fig3)

out = io.BytesIO()
import soundfile as sf
sf.write(out, reconstructed, sr, format="WAV")
st.download_button("Download reconstructed WAV", out.getvalue(), "reconstructed.wav", "audio/wav")

st.subheader("Experiment summary")
summary = pd.DataFrame([{
    "wavelet": wavelet,
    "level": level,
    "threshold": threshold_fraction,
    "channel_snr_db": snr,
    "loss_probability": loss,
    "reconstruction_snr_db": snr_db(signal, reconstructed),
    "mse": mse(signal, reconstructed),
    "correlation": correlation(signal, reconstructed),
    "retained_coefficients": retained,
}])
st.dataframe(summary, use_container_width=True)
