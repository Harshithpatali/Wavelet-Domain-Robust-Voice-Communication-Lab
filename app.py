from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

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


def make_transmission_package(
    scrambled_packet: WaveletPacket,
    sample_rate: int,
    threshold_value: float,
) -> bytes:
    """Create a portable receiver package without storing the secret key."""
    scrambled_coeffs = np.asarray(
        scrambled_packet.coeffs[0], dtype=np.float64
    ).reshape(-1)
    coeff_buffer = io.BytesIO()
    np.save(coeff_buffer, scrambled_coeffs)
    coeff_bytes = coeff_buffer.getvalue()
    metadata = {
        "format": "wavelet_voice_transmission_v1",
        "sample_rate": int(sample_rate),
        "original_length": int(scrambled_packet.original_length),
        "wavelet": scrambled_packet.wavelet,
        "level": int(scrambled_packet.level),
        "threshold": float(threshold_value),
        "coefficient_count": int(scrambled_coeffs.size),
        "coefficient_dtype": "float64",
        "coefficient_sha256": hashlib.sha256(coeff_bytes).hexdigest(),
        "note": "The receiver must enter the transmitter key separately.",
    }

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        zf.writestr("scrambled_coefficients.npy", coeff_bytes)
    return out.getvalue()


def load_transmission_package(data: bytes) -> tuple[WaveletPacket, dict]:
    """Load a receiver package and rebuild its deterministic coefficient layout."""
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        if "metadata.json" not in zf.namelist() or "scrambled_coefficients.npy" not in zf.namelist():
            raise ValueError("Invalid transmission package.")
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        coeffs = np.load(io.BytesIO(zf.read("scrambled_coefficients.npy")), allow_pickle=False)

    required = {
        "format", "sample_rate", "original_length", "wavelet", "level",
        "threshold", "coefficient_count", "coefficient_dtype",
        "coefficient_sha256",
    }
    if not required.issubset(metadata):
        raise ValueError("Transmission package metadata is incomplete.")
    if metadata["format"] != "wavelet_voice_transmission_v1":
        raise ValueError("Unsupported transmission package format.")
    if metadata["coefficient_dtype"] != "float64":
        raise ValueError("Unsupported coefficient data type.")
    if int(metadata["sample_rate"]) <= 0 or int(metadata["original_length"]) <= 0:
        raise ValueError("Transmission package contains invalid audio metadata.")
    if int(metadata["level"]) < 1 or not 0 <= float(metadata["threshold"]) <= 1:
        raise ValueError("Transmission package contains invalid DWT metadata.")
    payload_buffer = io.BytesIO()
    np.save(payload_buffer, np.asarray(coeffs, dtype=np.float64))
    if hashlib.sha256(payload_buffer.getvalue()).hexdigest() != metadata["coefficient_sha256"]:
        raise ValueError("Transmission package coefficient checksum failed.")

    # Recreate the exact coefficient-array layout from deterministic DWT metadata.
    dummy = np.zeros(int(metadata["original_length"]), dtype=np.float64)
    layout = decompose(dummy, metadata["wavelet"], int(metadata["level"]))
    if coeffs.size != int(metadata["coefficient_count"]) or coeffs.size != layout.coeffs[0].size:
        raise ValueError("Transmission package coefficient size does not match its metadata.")

    packet = WaveletPacket(
        coeffs=[np.asarray(coeffs, dtype=np.float64)],
        slices=layout.slices,
        original_length=int(metadata["original_length"]),
        wavelet=metadata["wavelet"],
        level=int(metadata["level"]),
    )
    return packet, metadata



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


def reconstruct_with_key(
    scrambled_packet: WaveletPacket,
    key: int,
) -> np.ndarray:
    """Receiver reconstruction using only its locally configured key."""
    receiver_permutation = make_permutation(
        scrambled_packet.coeffs[0].size,
        int(key),
    )
    recovered_packet = descramble(scrambled_packet, receiver_permutation)
    return safe_normalize(reconstruct(recovered_packet))


def packet_with_coeffs(packet: WaveletPacket, coeffs: np.ndarray) -> WaveletPacket:
    return WaveletPacket(
        coeffs=[np.asarray(coeffs, dtype=np.float64)],
        slices=packet.slices,
        original_length=packet.original_length,
        wavelet=packet.wavelet,
        level=packet.level,
    )


if "sender_key_locked" not in st.session_state:
    st.session_state.sender_key_locked = False
if "locked_transmitter_key" not in st.session_state:
    st.session_state.locked_transmitter_key = None

with st.sidebar:
    st.header("Communication Mode")
    app_mode = st.radio(
        "Mode",
        ["Transmit & Send", "Receive Shared Transmission"],
        index=0,
    )
    st.divider()
    st.header("Communication Configuration")
    st.caption(
        "Choose and lock the transmitter key before uploading audio. "
        "Share that same key with the receiver separately."
        if app_mode == "Transmit & Send"
        else "Upload a .wvt package and enter the key provided separately by the sender."
    )
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
    if app_mode == "Transmit & Send":
        st.subheader("Transmitter key")
        transmitter_key_input = st.number_input(
            "Choose your secret key",
            min_value=0,
            max_value=2_147_483_647,
            value=2026,
            step=1,
            disabled=st.session_state.sender_key_locked,
            help="Choose this BEFORE uploading audio. Send the same key to the receiver separately.",
        )
        if not st.session_state.sender_key_locked:
            if st.button("Set & Lock Key", type="primary", width="stretch"):
                st.session_state.locked_transmitter_key = int(transmitter_key_input)
                st.session_state.sender_key_locked = True
                st.rerun()
        else:
            st.success(f"Key locked: {st.session_state.locked_transmitter_key}")
            if st.button("Change Key", width="stretch"):
                st.session_state.sender_key_locked = False
                st.session_state.locked_transmitter_key = None
                st.rerun()
    
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

uploaded = None
if app_mode == "Transmit & Send" and st.session_state.sender_key_locked:
    uploaded = st.file_uploader(
        "Upload a voice recording",
        type=["wav", "m4a", "mp3", "flac", "aac"],
        help="M4A/MP3/AAC are decoded automatically and normalized to 16 kHz mono PCM.",
    )

if app_mode == "Transmit & Send" and not st.session_state.sender_key_locked:
    st.warning(
        "Set and lock the transmitter key first. "
        "Audio upload and scrambling are disabled until a key is locked."
    )
    st.info(
        "Workflow: 1) choose key → 2) lock key → 3) upload audio → "
        "4) scramble → 5) send the same key separately to the receiver."
    )
    st.stop()

if app_mode == "Receive Shared Transmission":
    st.title("Receive Shared Transmission")
    st.caption("The attachment contains scrambled coefficients, not the recovery key.")
    package_upload = st.file_uploader(
        "Upload the .wvt transmission package",
        type=["wvt", "zip"],
    )
    receiver_key_input = st.number_input(
        "Receiver key",
        min_value=0,
        max_value=2_147_483_647,
        value=2026,
        step=1,
        help="Enter the key provided separately by the sender.",
    )
    if package_upload is None:
        st.info("Upload the transmission package and enter the receiver key.")
        st.stop()
    try:
        receive_packet, receive_metadata = load_transmission_package(package_upload.getvalue())
        receive_permutation = make_permutation(
            receive_packet.coeffs[0].size,
            int(receiver_key_input),
        )
        recovered_receive = safe_normalize(
            reconstruct(descramble(receive_packet, receive_permutation))
        )
        scrambled_receive = safe_normalize(reconstruct(receive_packet))
        st.success(
            "Recovery completed using the key you entered. "
            "This prototype cannot authenticate whether the key is correct; "
            "compare the recovered voice and metrics to verify the result."
        )
        st.subheader("Received transmission")
        st.audio(audio_bytes(scrambled_receive, int(receive_metadata["sample_rate"])), format="audio/wav")
        st.subheader("Recovered original voice")
        st.audio(audio_bytes(recovered_receive, int(receive_metadata["sample_rate"])), format="audio/wav")
        st.download_button(
            "Download recovered voice",
            audio_bytes(recovered_receive, int(receive_metadata["sample_rate"])),
            "recovered_voice.wav",
            "audio/wav",
        )
        st.caption(
            f"Wavelet: {receive_metadata['wavelet']} · "
            f"Level: {receive_metadata['level']} · "
            "The secret key is not stored in the package."
        )
    except (ValueError, zipfile.BadZipFile, KeyError, OSError) as exc:
        st.error(f"Could not open this transmission package: {exc}")
    st.stop()

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

    # Scramble the flattened wavelet coefficient stream using ONLY the
    # transmitter key that was explicitly locked before upload.
    transmitter_key = int(st.session_state.locked_transmitter_key)
    scrambled_packet, transmitter_permutation = scramble(
        thresholded_packet,
        transmitter_key,
    )
    transmission_package = make_transmission_package(
        scrambled_packet,
        sr,
        threshold_fraction,
    )

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
    receiver_key = int(transmitter_key)
    receiver_permutation = make_permutation(
        received_scrambled_packet.coeffs[0].size,
        receiver_key,
    )
    recovered_coeff_packet = descramble(
        received_scrambled_packet,
        receiver_permutation,
    )
    reconstructed = safe_normalize(reconstruct(recovered_coeff_packet))

    # Verification receiver: intentionally use a different key.
    wrong_key = (int(transmitter_key) + 1) % 2_147_483_648
    wrong_key_audio = reconstruct_with_key(
        received_scrambled_packet,
        wrong_key,
    )

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
    f"Transmitter key locked before upload: {int(transmitter_key)}. "
    "The receiver must enter this same key to recover the voice."
)

st.info(
    f"Wavelet={wavelet}, level={level}, threshold={threshold_fraction:.3f}. "
    "The key itself is never embedded in the transmission package or email."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recovered SNR", f"{snr_db(original, reconstructed):.2f} dB")
m2.metric("Recovered MSE", f"{mse(original, reconstructed):.6g}")
m3.metric("Recovered correlation", f"{correlation(original, reconstructed):.4f}")
m4.metric("Coefficients retained", f"{retained * 100:.1f}%")

st.subheader("Transmission Verification Dashboard")
original_coeffs = thresholded_packet.coeffs[0]
scrambled_coeffs = scrambled_packet.coeffs[0]
scrambled_coeff_corr = coefficient_correlation(original_coeffs, scrambled_coeffs)
change_rate = permutation_change_rate(transmitter_permutation)
same_key_corr = correlation(original, reconstructed)
wrong_key_corr = correlation(original, wrong_key_audio)
same_key_snr = snr_db(original, reconstructed)
wrong_key_snr = snr_db(original, wrong_key_audio)

v1, v2, v3, v4 = st.columns(4)
v1.metric("Coefficients reordered", f"{change_rate * 100:.2f}%")
v2.metric("Original ↔ scrambled coeff. corr.", f"{scrambled_coeff_corr:.4f}")
v3.metric("Original ↔ correct-key corr.", f"{same_key_corr:.4f}")
v4.metric("Original ↔ wrong-key corr.", f"{wrong_key_corr:.4f}")

st.caption(
    "A successful transmission should have a large coefficient reorder rate, "
    "low original-to-scrambled similarity, and high original-to-correct-key similarity."
)

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Correct-key receiver**")
    st.audio(audio_bytes(reconstructed, sr), format="audio/wav")
    st.metric("Correct-key reconstruction SNR", f"{same_key_snr:.2f} dB")
with d2:
    st.markdown("**Wrong-key receiver**")
    st.audio(audio_bytes(wrong_key_audio, sr), format="audio/wav")
    st.metric("Wrong-key reconstruction SNR", f"{wrong_key_snr:.2f} dB")
    st.caption(f"Intentionally incorrect key: {wrong_key}")

st.subheader("Coefficient-order visualization")
show_n = min(120, original_coeffs.size)
fig_order, ax_order = plt.subplots(figsize=(11, 4))
ax_order.plot(np.arange(show_n), original_coeffs[:show_n], label="Original coefficient order")
ax_order.plot(
    np.arange(show_n),
    scrambled_coeffs[:show_n],
    label="Transmitted scrambled order",
    alpha=0.8,
)
ax_order.set_title("First coefficients: original vs transmitted ordering")
ax_order.set_xlabel("Coefficient position")
ax_order.set_ylabel("Coefficient value")
ax_order.legend()
st.pyplot(fig_order)
plt.close(fig_order)

fig_perm, ax_perm = plt.subplots(figsize=(11, 3))
ax_perm.scatter(np.arange(show_n), transmitter_permutation[:show_n], s=12)
ax_perm.plot(
    np.arange(show_n),
    np.arange(show_n),
    linestyle="--",
    label="No scrambling reference",
)
ax_perm.set_title("Permutation map: transmitted position → source coefficient index")
ax_perm.set_xlabel("Transmitted coefficient position")
ax_perm.set_ylabel("Source coefficient index")
ax_perm.legend()
st.pyplot(fig_perm)
plt.close(fig_perm)

st.success(
    "Verification result: the transmitted coefficient stream is reordered before "
    "the channel. The correct key restores the ordering; the intentionally wrong "
    "key does not."
)

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
            "transmitter_key": int(transmitter_key),
            "receiver_key": "entered by receiver",
            "keys_match": "checked during receiver recovery",
            "channel_snr_db": snr,
            "loss_probability": loss,
            "recovered_snr_db": snr_db(original, reconstructed),
            "recovered_mse": mse(original, reconstructed),
            "recovered_correlation": correlation(original, reconstructed),
            "retained_coefficients": retained,
        }
    ]
)
st.dataframe(summary, width="stretch")

st.info(
    "Research note: the keyed permutation is reversible signal obfuscation, "
    "not cryptographic encryption. It is included to demonstrate why the "
    "same transmitter/receiver configuration and key are required to recover "
    "the speech. For real confidentiality, use a standard authenticated "
    "encryption scheme."
)
