from __future__ import annotations

import hashlib
import io
import json
import zipfile

import matplotlib

matplotlib.use("Agg")

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

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Wavelet Voice Lab · Secure Voice Link",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# THEME
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#070b14;
  --surface:#0d1424;
  --surface-2:#111a2e;
  --border:#1e2a44;
  --text:#e6edf7;
  --muted:#8ea3c0;
  --cyan:#22d3ee;
  --violet:#a78bfa;
  --emerald:#34d399;
  --amber:#fbbf24;
  --rose:#fb7185;
}

html, body, .stApp{
  font-family:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ---------- app background ---------- */
.stApp{
  background:
    radial-gradient(1100px 520px at 12% -12%, rgba(34,211,238,.12), transparent 62%),
    radial-gradient(950px 480px at 102% -4%, rgba(167,139,250,.13), transparent 58%),
    radial-gradient(900px 620px at 50% 120%, rgba(52,211,153,.07), transparent 60%),
    var(--bg);
  color:var(--text);
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stToolbar"]{ right:1rem; }
#MainMenu, footer{ visibility:hidden; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a1020 0%,#070b14 100%);
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .block-container{ padding-top:1.2rem; }

.side-brand{
  display:flex; align-items:center; gap:.55rem;
  font-family:'JetBrains Mono',monospace;
  font-size:.78rem; letter-spacing:.18em; font-weight:600;
  color:var(--cyan); margin-bottom:.9rem;
}
.side-brand .dot{
  width:8px;height:8px;border-radius:50%;
  background:var(--cyan);
  box-shadow:0 0 10px var(--cyan), 0 0 22px rgba(34,211,238,.6);
  animation:pulse 2s infinite;
}
@keyframes pulse{ 0%,100%{opacity:1} 50%{opacity:.35} }

.side-sec{
  font-family:'JetBrains Mono',monospace;
  font-size:.68rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--muted); margin:1rem 0 .45rem 0;
  padding-bottom:.3rem; border-bottom:1px solid var(--border);
}

/* radio → card selector */
[data-testid="stSidebar"] div[role="radiogroup"] label{
  background:rgba(255,255,255,.02);
  border:1px solid var(--border);
  border-radius:12px;
  padding:.55rem .7rem;
  margin-bottom:.4rem;
  transition:all .15s ease;
  cursor:pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  border-color:rgba(34,211,238,.55);
  background:rgba(34,211,238,.07);
}
[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] > div:first-child{
  display:none;
}
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  font-size:.86rem !important; font-weight:600 !important;
}

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden;
  padding:1.7rem 1.9rem 1.5rem 1.9rem;
  border-radius:22px;
  border:1px solid var(--border);
  background:
    linear-gradient(135deg, rgba(34,211,238,.11), rgba(167,139,250,.09) 46%, rgba(52,211,153,.06));
  margin-bottom:1.1rem;
}
.hero::after{
  content:""; position:absolute; inset:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
  background-size:30px 30px;
  mask-image:radial-gradient(circle at 20% 0%, #000 10%, transparent 78%);
}
.hero-badge{
  display:inline-flex; align-items:center; gap:.5rem;
  font-family:'JetBrains Mono',monospace;
  font-size:.68rem; letter-spacing:.22em; text-transform:uppercase;
  color:var(--emerald);
  border:1px solid rgba(52,211,153,.35);
  background:rgba(52,211,153,.08);
  border-radius:999px; padding:.28rem .8rem; margin-bottom:.9rem;
}
.hero h1{
  margin:0 0 .35rem 0; font-size:2.15rem; font-weight:700; letter-spacing:-.02em;
  background:linear-gradient(92deg,#e6edf7 10%, #22d3ee 55%, #a78bfa 92%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{ margin:0; color:var(--muted); font-size:.94rem; max-width:80ch; }
.hero-chips{ display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.05rem; }

.chip{
  display:inline-flex; align-items:center; gap:.4rem;
  font-family:'JetBrains Mono',monospace; font-size:.7rem;
  padding:.3rem .72rem; border-radius:999px;
  border:1px solid var(--border); background:rgba(255,255,255,.03);
  color:var(--muted);
}
.chip b{ color:var(--text); font-weight:600; }
.chip.cy{ border-color:rgba(34,211,238,.4); background:rgba(34,211,238,.08); color:var(--cyan); }
.chip.vi{ border-color:rgba(167,139,250,.4); background:rgba(167,139,250,.08); color:var(--violet); }
.chip.em{ border-color:rgba(52,211,153,.4); background:rgba(52,211,153,.08); color:var(--emerald); }
.chip.am{ border-color:rgba(251,191,36,.4); background:rgba(251,191,36,.08); color:var(--amber); }
.chip.ro{ border-color:rgba(251,113,133,.4); background:rgba(251,113,133,.08); color:var(--rose); }

/* ---------- pipeline strip ---------- */
.pipe{
  display:flex; flex-wrap:wrap; align-items:center; gap:.4rem;
  padding:.7rem .9rem; margin-bottom:1.1rem;
  border:1px solid var(--border); border-radius:14px;
  background:rgba(255,255,255,.02);
}
.pipe-node{
  font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.1em;
  padding:.3rem .65rem; border-radius:9px; border:1px solid var(--border);
  background:rgba(255,255,255,.03); color:var(--muted); white-space:nowrap;
}
.pipe-node.tx{ color:var(--cyan); border-color:rgba(34,211,238,.35); background:rgba(34,211,238,.07); }
.pipe-node.key{ color:var(--violet); border-color:rgba(167,139,250,.35); background:rgba(167,139,250,.07); }
.pipe-node.ch{ color:var(--amber); border-color:rgba(251,191,36,.35); background:rgba(251,191,36,.07); }
.pipe-node.rx{ color:var(--emerald); border-color:rgba(52,211,153,.35); background:rgba(52,211,153,.07); }
.pipe-arrow{ color:#33415c; font-size:.85rem; }

/* ---------- containers as cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  border-radius:16px;
  border:1px solid var(--border);
  background:linear-gradient(180deg, rgba(255,255,255,.028), rgba(255,255,255,.008));
  padding:.35rem .25rem;
}

/* ---------- stage headers ---------- */
.stage-head{ display:flex; align-items:flex-start; gap:.6rem; margin-bottom:.55rem; }
.stage-num{
  flex:0 0 auto;
  width:26px; height:26px; border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-family:'JetBrains Mono',monospace; font-size:.76rem; font-weight:600;
  border:1px solid;
}
.stage-title{ font-size:.9rem; font-weight:600; color:var(--text); line-height:1.15; }
.stage-sub{
  font-family:'JetBrains Mono',monospace;
  font-size:.62rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin-top:.16rem;
}

/* ---------- metrics ---------- */
div[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.008));
  border:1px solid var(--border);
  border-radius:14px;
  padding:.85rem 1rem;
}
div[data-testid="stMetricLabel"] p{
  color:var(--muted) !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:.64rem !important;
  letter-spacing:.13em; text-transform:uppercase;
}
div[data-testid="stMetricValue"]{
  font-family:'JetBrains Mono',monospace;
  color:var(--text); font-size:1.35rem;
}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button{
  width:100%;
  border-radius:11px;
  border:1px solid var(--border);
  background:rgba(34,211,238,.07);
  color:var(--text);
  font-weight:600; font-size:.83rem;
  transition:all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:rgba(34,211,238,.65);
  background:rgba(34,211,238,.15);
  color:#ffffff;
}
.stButton > button[kind="primary"]{
  background:linear-gradient(135deg,#22d3ee,#6366f1);
  border:none; color:#04121a; font-weight:700;
}
.stButton > button[kind="primary"]:hover{
  filter:brightness(1.12); color:#04121a;
}

/* ---------- audio ---------- */
audio{
  width:100%; height:38px; border-radius:10px;
  background:rgba(255,255,255,.04);
  border:1px solid var(--border);
}

/* ---------- tabs ---------- */
div[data-baseweb="tab-list"]{
  gap:.35rem; background:transparent; border-bottom:1px solid var(--border);
}
button[data-baseweb="tab"]{
  border-radius:10px 10px 0 0;
  font-family:'JetBrains Mono',monospace; font-size:.74rem; letter-spacing:.06em;
  color:var(--muted); padding:.5rem .9rem;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color:var(--cyan);
  background:rgba(34,211,238,.07);
}

/* ---------- expander ---------- */
div[data-testid="stExpander"]{
  border:1px solid var(--border) !important;
  border-radius:14px !important;
  background:rgba(255,255,255,.02);
}

/* ---------- dataframe ---------- */
div[data-testid="stDataFrame"]{ border-radius:14px; overflow:hidden; border:1px solid var(--border); }

/* ---------- alerts ---------- */
div[data-testid="stAlert"]{ border-radius:14px; border:1px solid var(--border); }

/* ---------- footer ---------- */
.foot{
  margin-top:2rem; padding:1rem 1.2rem;
  border:1px dashed var(--border); border-radius:14px;
  font-size:.78rem; color:var(--muted); line-height:1.6;
}
.foot b{ color:var(--amber); }
</style>
""",
    unsafe_allow_html=True,
)

# Dark matplotlib styling
plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.facecolor": "#0b1220",
        "axes.edgecolor": "#1e2a44",
        "axes.labelcolor": "#8ea3c0",
        "axes.titlecolor": "#e6edf7",
        "text.color": "#e6edf7",
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "grid.color": "#1e2a44",
        "grid.alpha": 0.5,
        "axes.grid": True,
        "legend.facecolor": "#0d1424",
        "legend.edgecolor": "#1e2a44",
        "legend.labelcolor": "#e6edf7",
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.autolayout": True,
    }
)

ACCENT = {
    "tx": "#22d3ee",
    "key": "#a78bfa",
    "ch": "#fbbf24",
    "rx": "#34d399",
    "bad": "#fb7185",
}


# =====================================================================
# HELPERS
# =====================================================================
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


def reconstruct_with_key(scrambled_packet: WaveletPacket, key: int) -> np.ndarray:
    """Receiver reconstruction using only its locally configured key."""
    receiver_permutation = make_permutation(scrambled_packet.coeffs[0].size, int(key))
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


def make_transmission_package(
    scrambled_packet: WaveletPacket,
    sample_rate: int,
    threshold_value: float,
) -> bytes:
    """Create a portable receiver package without storing the secret key."""
    scrambled_coeffs = np.asarray(scrambled_packet.coeffs[0], dtype=np.float64).reshape(-1)
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
        if (
            "metadata.json" not in zf.namelist()
            or "scrambled_coefficients.npy" not in zf.namelist()
        ):
            raise ValueError("Invalid transmission package.")
        metadata = json.loads(zf.read("metadata.json").decode("utf-8"))
        coeffs = np.load(
            io.BytesIO(zf.read("scrambled_coefficients.npy")), allow_pickle=False
        )

    required = {
        "format",
        "sample_rate",
        "original_length",
        "wavelet",
        "level",
        "threshold",
        "coefficient_count",
        "coefficient_dtype",
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


# ---------- UI building blocks ----------
def hero(mode: str, key_locked: bool, key_value, wavelet: str, level: int, snr: float, loss: float):
    key_chip = (
        f'<span class="chip vi">🔑 KEY <b>{key_value}</b></span>'
        if key_locked
        else '<span class="chip ro">🔒 KEY NOT SET</span>'
    )
    link_chip = (
        '<span class="chip em">● LINK READY</span>'
        if key_locked or mode.startswith("Receive")
        else '<span class="chip am">● STANDBY</span>'
    )
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-badge">● SECURE VOICE LINK · WAVELET DOMAIN</div>
          <h1>Wavelet Voice Lab</h1>
          <p>
            Wavelet-domain voice scrambling over a simulated noisy channel —
            DWT decomposition, key-driven coefficient permutation, AWGN / packet-loss
            channel, and keyed reconstruction at the receiver.
          </p>
          <div class="hero-chips">
            {link_chip}
            <span class="chip cy">MODE <b>{mode}</b></span>
            <span class="chip cy">CODEC <b>{wavelet.upper()} · L{level}</b></span>
            {key_chip}
            <span class="chip am">CHANNEL <b>{snr:.1f} dB · loss {loss:.3f}</b></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_strip(stage: int = 0):
    nodes = [
        ("① SOURCE", "tx"),
        ("② DWT", "tx"),
        ("③ THRESHOLD", "tx"),
        ("④ KEY PERMUTE", "key"),
        ("⑤ CHANNEL", "ch"),
        ("⑥ DE-PERMUTE", "rx"),
        ("⑦ IDWT", "rx"),
        ("⑧ VOICE OUT", "rx"),
    ]
    html = '<div class="pipe">'
    for i, (label, cls) in enumerate(nodes):
        html += f'<span class="pipe-node {cls}">{label}</span>'
        if i < len(nodes) - 1:
            html += '<span class="pipe-arrow">→</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def stage_head(number: str, title: str, subtitle: str, color: str):
    st.markdown(
        f"""
        <div class="stage-head">
          <div class="stage-num" style="color:{color};border-color:{color}55;background:{color}1a">{number}</div>
          <div>
            <div class="stage-title">{title}</div>
            <div class="stage-sub">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stage_card(
    col,
    number: str,
    title: str,
    subtitle: str,
    color: str,
    signal: np.ndarray,
    sr: int,
    caption: str,
    download_label: str,
    file_name: str,
    key: str,
):
    with col:
        with st.container(border=True):
            stage_head(number, title, subtitle, color)
            st.audio(audio_bytes(signal, sr), format="audio/wav")
            st.caption(caption)
            st.download_button(
                download_label,
                audio_bytes(signal, sr),
                file_name,
                "audio/wav",
                key=key,
            )


# =====================================================================
# SESSION STATE
# =====================================================================
if "sender_key_locked" not in st.session_state:
    st.session_state.sender_key_locked = False
if "locked_transmitter_key" not in st.session_state:
    st.session_state.locked_transmitter_key = None

# =====================================================================
# SIDEBAR — MISSION CONTROL
# =====================================================================
with st.sidebar:
    st.markdown(
        '<div class="side-brand"><span class="dot"></span> MISSION CONTROL</div>',
        unsafe_allow_html=True,
    )

    app_mode = st.radio(
        "Mode",
        ["Transmit & Send", "Receive Shared Transmission"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-sec">Codec Configuration</div>', unsafe_allow_html=True)
    if app_mode.startswith("Receive"):
        st.caption("Codec parameters are read from the uploaded package.")
    wavelet = st.selectbox("Wavelet", ["haar", "db2", "db4", "db8", "sym4", "coif1"], index=2)
    level = st.slider("DWT level", 1, 6, 4)
    threshold_fraction = st.slider("Coefficient threshold", 0.0, 0.20, 0.02, 0.005)

    if app_mode == "Transmit & Send":
        st.markdown('<div class="side-sec">🔑 Transmitter Key</div>', unsafe_allow_html=True)
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
            if st.button("🔒 Set & Lock Key", type="primary"):
                st.session_state.locked_transmitter_key = int(transmitter_key_input)
                st.session_state.sender_key_locked = True
                st.rerun()
        else:
            st.success(f"Key locked: {st.session_state.locked_transmitter_key}")
            if st.button("↺ Change Key"):
                st.session_state.sender_key_locked = False
                st.session_state.locked_transmitter_key = None
                st.rerun()

    st.markdown('<div class="side-sec">📡 Simulated Channel</div>', unsafe_allow_html=True)
    snr = st.slider("Channel SNR (dB)", -5.0, 40.0, 20.0, 1.0)
    loss = st.slider("Packet/sample loss probability", 0.0, 0.20, 0.0, 0.005)
    seed = st.number_input("Channel random seed", min_value=0, value=42, step=1)

    st.markdown(
        '<div class="foot" style="margin-top:1.4rem;padding:.7rem .8rem;font-size:.7rem;">'
        "The key is <b>never</b> embedded in the transmission package. "
        "Deliver it out-of-band."
        "</div>",
        unsafe_allow_html=True,
    )

# =====================================================================
# HERO
# =====================================================================
hero(
    app_mode,
    st.session_state.sender_key_locked,
    st.session_state.locked_transmitter_key,
    wavelet,
    level,
    snr,
    loss,
)

# =====================================================================
# RECEIVER MODE
# =====================================================================
if app_mode == "Receive Shared Transmission":
    pipeline_strip()

    st.markdown("### 🛰️ Ground Station · Receive")
    st.caption(
        "The attachment contains scrambled wavelet coefficients only — the recovery key "
        "must arrive through a separate trusted channel."
    )

    c_up, c_key = st.columns([2, 1], gap="large")

    with c_up:
        with st.container(border=True):
            stage_head("01", "Transmission Package", "UPLOAD .WVT", ACCENT["key"])
            package_upload = st.file_uploader(
                "Upload the .wvt transmission package",
                type=["wvt", "zip"],
                label_visibility="collapsed",
            )

    with c_key:
        with st.container(border=True):
            stage_head("02", "Receiver Key", "OUT-OF-BAND SECRET", ACCENT["vi"] if False else ACCENT["key"])
            receiver_key_input = st.number_input(
                "Receiver key",
                min_value=0,
                max_value=2_147_483_647,
                value=2026,
                step=1,
                label_visibility="collapsed",
                help="Enter the key provided separately by the sender.",
            )

    if package_upload is None:
        st.info("Upload the transmission package and enter the receiver key to begin recovery.")
        st.stop()

    try:
        receive_packet, receive_metadata = load_transmission_package(package_upload.getvalue())
        receive_permutation = make_permutation(
            receive_packet.coeffs[0].size, int(receiver_key_input)
        )
        recovered_receive = safe_normalize(
            reconstruct(descramble(receive_packet, receive_permutation))
        )
        scrambled_receive = safe_normalize(reconstruct(receive_packet))

        st.success(
            "Recovery completed with the key you entered. This prototype cannot "
            "authenticate the key — compare the recovered voice and metrics."
        )

        st.markdown("### Output")
        r1, r2 = st.columns(2, gap="large")
        stage_card(
            r1,
            "A",
            "Received Transmission",
            "INTERCEPTED / SCRAMBLED",
            ACCENT["ch"],
            scrambled_receive,
            int(receive_metadata["sample_rate"]),
            "What the channel delivered — still permuted.",
            "Download received WAV",
            "received_scrambled.wav",
            "rx_dl_scrambled",
        )
        stage_card(
            r2,
            "B",
            "Recovered Original Voice",
            "DE-PERMUTED + IDWT",
            ACCENT["rx"],
            recovered_receive,
            int(receive_metadata["sample_rate"]),
            "Reconstructed after reversing the keyed coefficient permutation.",
            "Download recovered WAV",
            "recovered_voice.wav",
            "rx_dl_recovered",
        )

        with st.expander("📦 Package metadata", expanded=False):
            st.json(receive_metadata)

        st.caption(
            f"Wavelet: **{receive_metadata['wavelet']}** · "
            f"Level: **{receive_metadata['level']}** · "
            f"Threshold: **{receive_metadata['threshold']:.3f}** · "
            "The secret key is not stored in the package."
        )
    except (ValueError, zipfile.BadZipFile, KeyError, OSError) as exc:
        st.error(f"Could not open this transmission package: {exc}")

    st.markdown(
        '<div class="foot"><b>Security note:</b> the keyed permutation is reversible '
        "signal obfuscation, not cryptographic encryption. Use a standard authenticated "
        "encryption scheme for real confidentiality.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# =====================================================================
# TRANSMITTER GATE
# =====================================================================
if not st.session_state.sender_key_locked:
    pipeline_strip()
    st.warning(
        "Set and lock the transmitter key before uploading audio. "
        "Scrambling is disabled until a key is locked."
    )
    st.markdown(
        """
        <div class="foot" style="border-style:solid;">
          <b>Secure workflow</b><br>
          1 &nbsp;·&nbsp; Choose the secret key in the sidebar<br>
          2 &nbsp;·&nbsp; Lock the key (this freezes it before any audio is read)<br>
          3 &nbsp;·&nbsp; Upload the voice recording<br>
          4 &nbsp;·&nbsp; Scramble the wavelet coefficients<br>
          5 &nbsp;·&nbsp; Send the <b>.wvt</b> package to the receiver<br>
          6 &nbsp;·&nbsp; Deliver the same key through a separate channel
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# =====================================================================
# TRANSMITTER — SOURCE
# =====================================================================
pipeline_strip()

st.markdown("### 🎙️ Transmitter · Source")
uploaded = st.file_uploader(
    "Upload a voice recording",
    type=["wav", "m4a", "mp3", "flac", "aac"],
    help="M4A/MP3/AAC are decoded automatically and normalized to 16 kHz mono PCM.",
)

if uploaded is None:
    original = synthetic_voice_like()
    sr = 16000
    source_name = "Synthetic demo signal"
    st.info("No file uploaded — a deterministic speech-like signal is being used.")
else:
    try:
        original, sr = load_audio_bytes(uploaded.getvalue(), uploaded.name, target_sr=16000)
        source_name = uploaded.name
        st.success(f"Loaded {uploaded.name} → {sr:,} Hz mono PCM")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

original = safe_normalize(original)

# =====================================================================
# DSP PIPELINE
# =====================================================================
try:
    # ------------------------- TRANSMITTER -------------------------
    packet = decompose(original, wavelet, level)
    thresholded_packet, retained = threshold(packet, threshold_fraction)

    transmitter_key = int(st.session_state.locked_transmitter_key)
    scrambled_packet, transmitter_permutation = scramble(thresholded_packet, transmitter_key)

    transmission_package = make_transmission_package(scrambled_packet, sr, threshold_fraction)

    # Interceptor view: never reconstruct the unscrambled coefficients.
    scrambled_audio = safe_normalize(reconstruct(scrambled_packet))

    # ------------------------- CHANNEL -----------------------------
    rng = np.random.default_rng(int(seed))
    received_scrambled_coeffs = add_awgn(scrambled_packet.coeffs[0], snr, rng)
    received_scrambled_coeffs = packet_loss(received_scrambled_coeffs, loss, rng)
    received_scrambled_packet = packet_with_coeffs(
        scrambled_packet, received_scrambled_coeffs
    )

    # ------------------------- RECEIVER ----------------------------
    receiver_key = int(transmitter_key)
    receiver_permutation = make_permutation(
        received_scrambled_packet.coeffs[0].size, receiver_key
    )
    recovered_coeff_packet = descramble(received_scrambled_packet, receiver_permutation)
    reconstructed = safe_normalize(reconstruct(recovered_coeff_packet))

    received_scrambled_audio = safe_normalize(reconstruct(received_scrambled_packet))

    # Verification receiver: intentionally wrong key
    wrong_key = (int(transmitter_key) + 1) % 2_147_483_648
    wrong_key_audio = reconstruct_with_key(received_scrambled_packet, wrong_key)

except ValueError as exc:
    st.error(str(exc))
    st.stop()

# =====================================================================
# RESULTS
# =====================================================================
tab_pipe, tab_verify, tab_signal, tab_report = st.tabs(
    ["▶  Pipeline", "🔐  Key Verification", "📈  Signal Analysis", "🧾  Report"]
)

# ---------------------------------------------------------------- PIPE
with tab_pipe:
    st.markdown("#### End-to-end communication stages")
    c1, c2, c3, c4 = st.columns(4, gap="small")

    stage_card(
        c1, "1", "Original", "CLEAR SOURCE", ACCENT["tx"],
        original, sr,
        "Clean speech before transmission.",
        "Download original WAV", "original_normalized.wav", "dl_original",
    )
    stage_card(
        c2, "2", "Transmitted", "SCRAMBLED · KEY PERMUTED", ACCENT["key"],
        scrambled_audio, sr,
        "What an interceptor hears: coefficients are permuted with the shared key.",
        "Download scrambled WAV", "transmitted_scrambled.wav", "dl_scrambled",
    )
    stage_card(
        c3, "3", "Received", f"CHANNEL {snr:.1f} dB · LOSS {loss:.3f}", ACCENT["ch"],
        received_scrambled_audio, sr,
        "Scrambled signal after AWGN and packet loss.",
        "Download received WAV", "received_scrambled.wav", "dl_received",
    )
    stage_card(
        c4, "4", "Recovered", "DE-PERMUTED · IDWT", ACCENT["rx"],
        reconstructed, sr,
        "Receiver reverses the permutation and applies the inverse DWT.",
        "Download recovered WAV", "receiver_recovered.wav", "dl_recovered",
    )

    st.markdown("#### Link quality")
    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Recovered SNR", f"{snr_db(original, reconstructed):.2f} dB")
    m2.metric("Recovered MSE", f"{mse(original, reconstructed):.6g}")
    m3.metric("Recovered correlation", f"{correlation(original, reconstructed):.4f}")
    m4.metric("Coefficients retained", f"{retained * 100:.1f}%")

    st.success(
        f"🔑 Transmitter key locked before upload: **{transmitter_key}**. "
        "The receiver must enter this same key to recover the voice."
    )
    st.info(
        f"Codec: wavelet = **{wavelet}**, level = **{level}**, "
        f"threshold = **{threshold_fraction:.3f}**. The key is never embedded in the "
        "transmission package or email."
    )

    st.markdown("#### Portable transmission package")
    st.caption(
        "Download this `.wvt` file and send it to the receiver. It contains scrambled "
        "coefficients and metadata — never the key."
    )
    st.download_button(
        "📦 Download .wvt transmission package",
        transmission_package,
        "voice_transmission.wvt",
        "application/zip",
        key="dl_package",
    )

# -------------------------------------------------------------- VERIFY
with tab_verify:
    original_coeffs = thresholded_packet.coeffs[0]
    scrambled_coeffs = scrambled_packet.coeffs[0]
    scrambled_coeff_corr = coefficient_correlation(original_coeffs, scrambled_coeffs)
    change_rate = permutation_change_rate(transmitter_permutation)
    same_key_corr = correlation(original, reconstructed)
    wrong_key_corr = correlation(original, wrong_key_audio)
    same_key_snr = snr_db(original, reconstructed)
    wrong_key_snr = snr_db(original, wrong_key_audio)

    st.markdown("#### Transmission verification dashboard")
    v1, v2, v3, v4 = st.columns(4, gap="small")
    v1.metric("Coefficients reordered", f"{change_rate * 100:.2f}%")
    v2.metric("Original ↔ scrambled corr.", f"{scrambled_coeff_corr:.4f}")
    v3.metric("Original ↔ correct key corr.", f"{same_key_corr:.4f}")
    v4.metric("Original ↔ wrong key corr.", f"{wrong_key_corr:.4f}")

    st.caption(
        "A successful transmission shows a high coefficient reorder rate, low "
        "original-to-scrambled similarity, and high original-to-correct-key similarity."
    )

    st.markdown("#### Receiver comparison")
    d1, d2 = st.columns(2, gap="large")

    with d1:
        with st.container(border=True):
            stage_head("✓", "Correct-key receiver", f"KEY {transmitter_key}", ACCENT["rx"])
            st.audio(audio_bytes(reconstructed, sr), format="audio/wav")
            st.metric("Reconstruction SNR", f"{same_key_snr:.2f} dB")
            st.metric("Correlation with original", f"{same_key_corr:.4f}")

    with d2:
        with st.container(border=True):
            stage_head("✕", "Wrong-key receiver", f"KEY {wrong_key}", ACCENT["bad"])
            st.audio(audio_bytes(wrong_key_audio, sr), format="audio/wav")
            st.metric("Reconstruction SNR", f"{wrong_key_snr:.2f} dB")
            st.metric("Correlation with original", f"{wrong_key_corr:.4f}")

    st.markdown("#### Coefficient-order visualization")
    show_n = min(120, original_coeffs.size)

    fig_order, ax_order = plt.subplots(figsize=(11, 3.6))
    ax_order.plot(
        np.arange(show_n), original_coeffs[:show_n],
        label="Original coefficient order", color=ACCENT["tx"], linewidth=1.4,
    )
    ax_order.plot(
        np.arange(show_n), scrambled_coeffs[:show_n],
        label="Transmitted scrambled order", color=ACCENT["key"],
        alpha=0.85, linewidth=1.2,
    )
    ax_order.set_title("First coefficients · original vs transmitted ordering")
    ax_order.set_xlabel("Coefficient position")
    ax_order.set_ylabel("Coefficient value")
    ax_order.legend(frameon=True)
    st.pyplot(fig_order)
    plt.close(fig_order)

    fig_perm, ax_perm = plt.subplots(figsize=(11, 2.9))
    ax_perm.scatter(
        np.arange(show_n), transmitter_permutation[:show_n],
        s=14, color=ACCENT["key"], alpha=0.85, label="Keyed permutation",
    )
    ax_perm.plot(
        np.arange(show_n), np.arange(show_n),
        linestyle="--", color=ACCENT["muted"] if "muted" in ACCENT else "#8ea3c0",
        linewidth=1, label="No scrambling reference",
    )
    ax_perm.set_title("Permutation map · transmitted position → source coefficient index")
    ax_perm.set_xlabel("Transmitted coefficient position")
    ax_perm.set_ylabel("Source index")
    ax_perm.legend(frameon=True)
    st.pyplot(fig_perm)
    plt.close(fig_perm)

    st.success(
        "Verification result: the coefficient stream is reordered before the channel. "
        "The correct key restores the ordering; the intentionally wrong key does not."
    )

# -------------------------------------------------------------- SIGNAL
with tab_signal:
    st.markdown("#### Waveform comparison")
    preview_len = min(len(original), sr * 2)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8.2), sharex=True)

    series = [
        (original[:preview_len], "1 · Original speech waveform", ACCENT["tx"]),
        (scrambled_audio[:preview_len], "2 · Transmitted / scrambled waveform", ACCENT["key"]),
        (received_scrambled_audio[:preview_len], "3 · Received scrambled waveform", ACCENT["ch"]),
        (reconstructed[:preview_len], "4 · Receiver-recovered waveform", ACCENT["rx"]),
    ]
    for ax, (data, title, color) in zip(axes, series):
        ax.plot(data, color=color, linewidth=0.85)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Amplitude")
    axes[-1].set_xlabel("Sample")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Received wavelet coefficient distribution")
    fig2, ax2 = plt.subplots(figsize=(11, 2.9))
    ax2.plot(
        np.abs(received_scrambled_coeffs[: min(5000, received_scrambled_coeffs.size)]),
        color=ACCENT["ch"], linewidth=0.7,
    )
    ax2.set_title("Absolute magnitude of received scrambled coefficients")
    ax2.set_xlabel("Scrambled coefficient index")
    ax2.set_ylabel("Magnitude")
    st.pyplot(fig2)
    plt.close(fig2)

    with st.expander("🔢 Coefficient statistics", expanded=False):
        stats = pd.DataFrame(
            {
                "metric": [
                    "Original coefficient count",
                    "Retained after threshold",
                    "Transmitted coefficient count",
                    "Permutation change rate",
                    "Original ↔ scrambled coefficient corr.",
                ],
                "value": [
                    f"{original_coeffs.size:,}",
                    f"{retained * 100:.2f}%",
                    f"{scrambled_coeffs.size:,}",
                    f"{change_rate * 100:.2f}%",
                    f"{scrambled_coeff_corr:.4f}",
                ],
            }
        )
        st.dataframe(stats, width="stretch", hide_index=True)

# -------------------------------------------------------------- REPORT
with tab_report:
    st.markdown("#### Experiment summary")
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
                "channel_seed": int(seed),
                "recovered_snr_db": snr_db(original, reconstructed),
                "recovered_mse": mse(original, reconstructed),
                "recovered_correlation": correlation(original, reconstructed),
                "retained_coefficients": retained,
            }
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    st.download_button(
        "⬇️ Download summary CSV",
        summary.to_csv(index=False).encode("utf-8"),
        "wavelet_voice_lab_summary.csv",
        "text/csv",
        key="dl_summary_csv",
    )

    with st.expander("📦 Transmission package metadata", expanded=False):
        st.json(
            {
                "format": "wavelet_voice_transmission_v1",
                "sample_rate": sr,
                "original_length": int(scrambled_packet.original_length),
                "wavelet": wavelet,
                "level": level,
                "threshold": threshold_fraction,
                "coefficient_count": int(scrambled_packet.coeffs[0].size),
                "coefficient_dtype": "float64",
                "note": "The receiver must enter the transmitter key separately.",
            }
        )

    st.markdown(
        """
        <div class="foot">
          <b>Research note:</b> the keyed permutation is reversible signal obfuscation,
          not cryptographic encryption. It is included to demonstrate why the same
          transmitter/receiver configuration and key are required to recover the speech.
          For real confidentiality, use a standard authenticated encryption scheme.
        </div>
        """,
        unsafe_allow_html=True,
    )
