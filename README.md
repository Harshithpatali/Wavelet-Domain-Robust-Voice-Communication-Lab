<div align="center">

# 🛰️ Wavelet Voice Lab

### Wavelet-Domain Robust Voice Communication Simulator

**DWT → thresholding → keyed coefficient scrambling → portable WAV transmission → keyed reconstruction**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="PyWavelets" src="https://img.shields.io/badge/PyWavelets-1.5%2B-22d3ee">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-a78bfa">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Prototype-34d399">
</p>

[Overview](#-overview) ·
[Quick Start](#-quick-start) ·
[How It Works](#-how-it-works) ·
[Usage](#-usage) ·
[Transmission WAV](#-the-transmission-wav-format) ·
[V4 Research Lab](#-v4-advanced-research-lab) ·
[Private Live Calling](#-private-live-calling) ·
[Metrics](#-verification-dashboard) ·
[Security](#-security-limitations) ·
[FAQ](#-faq)

</div>

> ⚠️ **Not cryptography.** This project demonstrates *reversible signal obfuscation*, not encryption. Read [Security Limitations](#-security-limitations) before using it for anything beyond learning.

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Highlights](#-highlights)
3. [Quick Start](#-quick-start)
4. [How It Works](#-how-it-works)
5. [Usage](#-usage)
6. [The transmission WAV format](#-the-transmission-wav-format)
7. [V4 Advanced Research Lab](#-v4-advanced-research-lab)
8. [Private Live Calling](#-private-live-calling)
9. [Verification Dashboard](#-verification-dashboard)
8. [Experiment Matrix](#-experiment-matrix)
9. [Interpreting Results](#-interpreting-results)
10. [Security Limitations](#-security-limitations)
11. [Project Structure](#-project-structure)
12. [Testing](#-testing)
13. [Troubleshooting](#-troubleshooting)
14. [FAQ](#-faq)
15. [Scope](#-scope)
16. [Learning Outcomes](#-learning-outcomes)
17. [Roadmap](#-roadmap)
18. [Contributing](#-contributing)
19. [Citation](#-citation)
20. [License](#-license)

---

## 📡 Overview

**Wavelet Voice Lab** is an interactive, end-to-end simulation of a voice link built on the **Discrete Wavelet Transform (DWT)**. It lets you:

- Decompose speech into multi-resolution wavelet coefficients
- Threshold and **permute those coefficients with a secret key**
- Push the scrambled stream through a simulated noisy channel (**AWGN + packet loss**)
- Recover the voice at the receiver using the **same key and codec configuration**

The result is a hands-on lab for exploring **wavelet representation, reversible keyed scrambling, and channel robustness**, delivered as one Streamlit app plus a headless CLI.

---

## ✨ Highlights

| | Feature | Details |
|---|---|---|
| 🎛️ | **Full DWT control** | Haar, db2–db8, sym4, coif1 · decomposition levels 1–6 |
| 🔑 | **Key-first workflow** | Key is locked *before* audio upload and never stored in the package |
| 🎵 | **Portable WAV transmission** | Normal playable mono WAV audio + private `WVTP` recovery payload; key is never embedded |
| 📡 | **Channel simulation** | Adjustable AWGN, random loss, and reproducible Gilbert–Elliott burst-loss experiments |
| 📦 | **Packet transport** | Optional .wvp stream with sequence numbers, CRC32, and receiver reassembly |
| 🛟 | **Forward error correction** | XOR parity can repair one erased packet per group of four |
| 🔐 | **Authenticated-encryption comparison** | Optional AES-256-GCM .wve package; separate passphrase, never stored in the package |
| 🧪 | **Research Lab** | Bounded multi-parameter sweeps, recovery metrics, and CSV output |
| 🔗 | **Receiver handoff** | QR links, seven-day expiry, and configurable receiver-session limits |
| 🛰️ | **Multi-mode UI** | Transmit, receive, and an optional **Private Live Call** mode (requires the separate signaling service) |
| 📊 | **Verification dashboard** | Reorder rate, correlation metrics, correct- vs. wrong-key comparison |
| 📈 | **Visual analysis** | Waveform stages, coefficient order, permutation map, magnitude distribution |
| 🧾 | **Exports** | Transmission WAV, recovered WAV, summary CSV |
| 🧪 | **Test suite** | Pytest coverage for codec round-trip, channel, and permutation reversibility |
| 💻 | **CLI + Web** | Headless experiments or interactive Streamlit |
| 🎧 | **Format support** | WAV out of the box; M4A / MP3 / AAC via `ffmpeg` |
| 🧬 | **Zero-setup demo** | Falls back to a deterministic synthetic speech-like signal |

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/Wavelet-Domain-Robust-Voice-Communication-Lab.git
cd Wavelet-Domain-Robust-Voice-Communication-Lab

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# 3. Install
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

No audio file? No problem. Skip the upload and the app uses a built-in synthetic signal so you can experiment immediately.

### System dependencies (for M4A / MP3 / AAC decoding)

```bash
# Debian / Ubuntu
sudo apt-get install -y ffmpeg libsndfile1

# macOS
brew install ffmpeg libsndfile
```

> Streamlit Community Cloud already ships with `ffmpeg` and `libsndfile`.

---

## 📞 Private Live Calling

The optional **Private Live Call** mode lets two people open an invitation link, enter an 8-digit access code, and talk live using browser microphone audio.

### Call flow

1. Deploy the signaling service described in [call_service/README.md](call_service/README.md).
2. Configure `CALL_SIGNALING_URL` and `CALL_CREATION_TOKEN` in Streamlit Secrets. Set the same creation token in the signaling service environment.
3. Open **Private Live Call** in the sidebar and create a room with an expiry.
4. Email the invitation link to the recipient and share the 8-digit access code separately where possible.
5. Both callers open the app link, enter the code, and press **Join call**. The interface includes mute/unmute, end call, connection state, a round-trip estimate, and a peer-verification code.

### Architecture

```mermaid
flowchart LR
    A[Caller browser] <-->|WebRTC DTLS-SRTP media| B[Recipient browser]
    A -->|Offer, answer, ICE| C[FastAPI signaling]
    C -->|Offer, answer, ICE| B
    D[Streamlit app] -->|Create invitation using server-only token| C
    C --> E[(Neon invitation metadata)]
```

The signaling service relays connection setup messages only. Browser media travels through WebRTC and is encrypted using DTLS-SRTP; no recording is implemented. For some network configurations, a TURN service is needed. Configure `ICE_SERVERS_JSON` in the service to provide TURN as appropriate.

**Important distinction:** the current live-call mode does **not** transform audio into wavelet coefficients. It sends normal browser microphone audio through WebRTC; the recipient's browser decrypts the media transport and plays the voice normally. The invitation link identifies the room, while the separate eight-digit access code controls room entry. That code is not the media-encryption key—WebRTC negotiates the media keys automatically.

The room access code is stored as a salted PBKDF2 hash, not plaintext. The server-to-server `CALL_CREATION_TOKEN` must remain a Streamlit secret and a signaling-service environment variable; it is never sent to the browser. Invitations are short-lived and allow two participants. Persistent invitation metadata requires `DATABASE_URL` on the signaling service.

Each browser displays a peer-verification code derived from both DTLS fingerprints. Compare those codes through a trusted, independent channel; if they differ, end the call. WebRTC protects media in transit, but a room code by itself does not verify the real-world identity of the other person. This prototype is not security-audited, and the wavelet coefficient permutation is **not** the live-call encryption mechanism.

For standard Docker Web Service deployment and limitations, see [call_service/README.md](call_service/README.md).

---

## 🔬 How It Works

### Pipeline

```mermaid
flowchart TD
    subgraph TX["📤 Transmitter"]
        A["🎙️ Original speech"] --> B["DWT<br/>(wavelet, level)"]
        B --> C["Coefficient thresholding"]
        C --> D["🔐 Keyed coefficient permutation"]
        D --> E["Transmitted signal<br/><i>intentionally unintelligible</i>"]
    end

    E --> F["📡 Channel<br/>AWGN + packet loss"]

    subgraph RX["📥 Receiver"]
        F --> G["Same wavelet · level · threshold<br/>🔑 shared key"]
        G --> H["Reverse the permutation"]
        H --> I["IDWT"]
        I --> J["🔊 Recovered speech"]
    end
```

The middle signal is **deliberately unintelligible**: coefficient ordering is scrambled *before* the intermediate audio is reconstructed. Only the correct key (with matching codec parameters) restores it.

### The math, briefly

**1. Multi-level DWT.** A signal $x[n]$ is decomposed into one approximation band and $L$ detail bands:

$$x \;\longrightarrow\; \{\,cA_L,\; cD_L,\; cD_{L-1},\; \dots,\; cD_1\,\}$$

The bands are flattened into a single coefficient vector $\mathbf{c} \in \mathbb{R}^N$.

**2. Thresholding.** Small coefficients are zeroed:

$$\tilde{c}_i = \begin{cases} c_i & |c_i| \ge \tau \\ 0 & \text{otherwise} \end{cases}$$

**3. Keyed permutation.** A key $K$ deterministically seeds a pseudo-random permutation $\pi_K$ of $\{0,\dots,N-1\}$:

$$s_i = \tilde{c}_{\pi_K(i)}$$

**4. Channel.** Additive white Gaussian noise at a target SNR, plus random loss of samples/packets:

$$r_i = m_i \cdot (s_i + w_i), \qquad w_i \sim \mathcal{N}(0,\sigma^2), \quad m_i \in \{0,1\}$$

**5. Recovery.** The receiver applies $\pi_K^{-1}$ and then the inverse DWT. With a wrong key $K' \ne K$, the inverse permutation is incorrect and the output is noise-like.

### Why the receiver needs *both* the key and the config

| Mismatch | Effect |
|---|---|
| Wrong key | Coefficients land in wrong positions → garbled audio |
| Wrong wavelet | Reconstruction filters don't match analysis filters |
| Wrong level | Band boundaries differ → permutation applies to a differently-laid-out vector |
| Wrong length | Permutation length mismatch / padding errors |

The transmission WAV carries the non-secret length and coefficient metadata; the receiver currently selects the same wavelet and DWT level separately.

---

## 🖥️ Usage

### 🟢 Transmitter (web app)

```text
1.  Choose the secret key
2.  Click  🔒 Set & Lock Key
3.  Only then upload the voice recording
4.  The app performs:  DWT → threshold → keyed permutation
5.  Download  voice_transmission.wav
6.  Send the WAV and the key through SEPARATE channels
```

> **Important:** use the **Download transmission WAV** button. The analysis preview of the scrambled waveform is not a receiver package because it does not contain the coefficient payload.

### 🔵 Receiver (web app)

The sender can create a **one-click receiver link** from the app. The transmission is temporarily stored in the project's Neon Postgres database and the receiver link carries a random transmission ID.

```text
1.  Sender locks the secret key
2.  Sender uploads the voice
3.  Sender downloads voice_transmission.wav if a local copy is desired
4.  Sender clicks 🔗 Create receiver link
5.  Sender sends the generated receiver link to the receiver
6.  Receiver clicks the link
7.  The app opens directly in Receive Shared Transmission mode
8.  The shared WAV loads automatically — no manual file upload
9.  Receiver listens to the scrambled voice BEFORE entering the key
10. Receiver enters the key provided separately by the sender
11. Receiver clicks 🔑 Recover original voice
12. Receiver listens to / downloads recovered_voice.wav
```

> The receiver link does not contain the secret key. Shared transmissions are temporary and expire automatically. Send the key separately from the link.

### ⌨️ Command-line experiment

```bash
python run_experiment.py \
  --input path/to/voice.wav \
  --wavelet db4 \
  --level 4 \
  --threshold 0.02 \
  --key 2026 \
  --snr-db 10 \
  --packet-loss 0.02
```

| Flag | Description | Example |
|---|---|---|
| `--input` | Input audio file. Omit to use the synthetic signal | `voice.wav` |
| `--wavelet` | Wavelet family | `db4` |
| `--level` | DWT decomposition level (1–6) | `4` |
| `--threshold` | Coefficient magnitude threshold | `0.02` |
| `--key` | Non-negative integer shared key | `2026` |
| `--snr-db` | Channel SNR in dB | `10` |
| `--packet-loss` | Packet/sample loss probability (0–1) | `0.02` |

### 🐍 Programmatic usage (illustrative)

The core logic lives in `src/wavelet_codec.py`. The sketch below shows the shape of the workflow; check the module for the exact function names and signatures.

```python
# Illustrative only: adapt names to match src/wavelet_codec.py
from src import wavelet_codec as codec, channel, metrics

# Transmitter
coeffs = codec.dwt(signal, wavelet="db4", level=4)
coeffs = codec.threshold(coeffs, 0.02)
scrambled = codec.scramble(coeffs, key=2026)

# Channel
received = channel.apply(scrambled, snr_db=10, packet_loss=0.02, seed=0)

# Receiver
recovered_coeffs = codec.descramble(received, key=2026)
recovered = codec.idwt(recovered_coeffs, wavelet="db4", level=4)

print(metrics.snr(signal, recovered))
```

---

## 🎵 The transmission WAV format

The transmitter exports a portable WAV named **`voice_transmission.wav`**.

V3 uses a normal mono WAV for the playable scrambled voice and stores the recovery payload in a private RIFF chunk named **`WVTP`**.

| Part | Contents |
|---|---|
| **WAV audio stream** | Audible scrambled voice reconstruction |
| **`WVTP` RIFF chunk** | Non-secret codec metadata + scrambled wavelet coefficients |

The private payload stores:

- transmission magic/version marker
- original sample count
- flattened coefficient count
- coefficient scale
- carrier metadata
- DWT level
- wavelet identifier
- scrambled wavelet coefficients

The **secret key is never stored in the WAV**.


### Streamlit secret for one-click sharing

Add your Neon Postgres connection string to the Streamlit app secrets as:

```toml
NEON_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
```

The app creates the `wavelet_transmissions` table automatically and safely adds access-limit columns to an existing table. It stores the WAV bytes, a random UUID, creation/expiry timestamps, and access counters. The recovery key and AES-GCM passphrase are never stored in Neon. Expired rows are cleaned when a new transmission is stored.

### Receiver Link

A generated link has this shape (the token is unique per transmission):

`https://wavelet-domain-robust-voice-communication-lab.streamlit.app/?mode=receive&tx=<random-token>`

Opening the link selects **Receive Shared Transmission**, loads the WAV from Neon, and extracts the wavelet name, DWT level, original length, and coefficient count from its `WVTP` metadata. The receiver does not need to choose codec settings. Each link expires after seven days and can be limited to one, five, or twenty receiver sessions. The access counter is incremented on a new session load, not each Streamlit rerun in that session.

The URL token is a bearer link: share it only with the intended recipient. Send the wavelet key separately. The token does not contain that key.

> A plain audio preview such as `received_scrambled.wav` or `channel_received_preview.wav` is not a receiver package because it does not contain the `WVTP` recovery payload.

## 🧪 V4 Advanced Research Lab

The **Research Lab** tab adds optional experiments without replacing the default WAV/Neon workflow:

- **Sequence-numbered packet transport:** download a `.wvp` stream with packet sequence numbers and CRC32 checksums. The receiver checks the stream and rebuilds the WAV before parsing its `WVTP` payload.
- **Authenticated-encryption comparison:** download an optional `.wve` package protected by AES-256-GCM. Scrypt derives the encryption key from a separate passphrase. The receiver authenticates and decrypts the package before reading the WAV. Share both the wavelet key and package passphrase out of band; they serve different purposes.
- **Burst-loss channel:** compare independent random packet erasures with a Gilbert–Elliott two-state channel that creates correlated losses.
- **Forward-error correction:** one XOR parity packet per group of four data packets can recover one erased packet within a group. Multiple losses in the same group may remain unrecoverable. The dashboard reports redundancy and recovery counts.
- **Automated parameter sweep:** vary wavelet, decomposition level, threshold, SNR, and loss rate. Results include SNR, MSE, correlation, retained coefficients, packet recovery, and redundancy. The interface caps a sweep at 180 combinations and analyzes up to five seconds of audio.
- **Speech-oriented evaluation:** segmental SNR is included; STOI is displayed when the optional `pystoi` package is installed.

The `.wvp` and `.wve` exports are optional. Existing `voice_transmission.wav` files remain supported. CRC32 is for accidental corruption detection, not cryptographic authentication; AES-GCM authentication applies only to the optional encrypted wrapper.

## 📊 Verification Dashboard

Results shouldn't rest on listening alone. The Streamlit UI includes a dedicated verification section.

| Metric | Meaning | Expected |
|---|---|---|
| **Coefficients reordered (%)** | Fraction of positions changed by the keyed permutation | High |
| **Original ↔ scrambled corr.** | Similarity of coefficient streams before/after scrambling | **Low** |
| **Original ↔ correct-key corr.** | Recovery quality with the correct key | **High** |
| **Original ↔ wrong-key corr.** | Recovery quality with a deliberately wrong key | **Low** |
| **Recovered SNR / MSE** | Standard audio reconstruction metrics | SNR high, MSE low |

### Visualizations

- **Waveform comparison**: Original · Transmitted · Received · Recovered
- **Coefficient-order plot**: original vs. transmitted coefficient values
- **Permutation map**: transmitted position → source coefficient index (a diagonal means no scrambling)
- **Coefficient distribution**: absolute magnitude of received scrambled coefficients

### Wrong-key demonstration

```text
Transmitter key = K    Receiver key = K       →  ✅ correct recovery
Transmitter key = K    Receiver key = K + 1   →  ❌ incorrect recovery
```

Both outputs are playable side by side, which gives a practical demonstration that the shared key is required.

---

## 🧪 Experiment Matrix

A useful sweep of parameters for your research:

| Parameter | Suggested values |
|---|---|
| Wavelet | `haar`, `db2`, `db4`, `db8`, `sym4`, `coif1` |
| DWT level | `1 – 6` |
| Threshold | `0.00 – 0.20` |
| Shared key | Any non-negative integer |
| Channel SNR | `-5 – 40 dB` |
| Packet / sample loss | `0 – 20 %` |

For every configuration, compare:

- Original vs. recovered waveform
- Recovered SNR, MSE, correlation
- Retained coefficient percentage
- Subjective intelligibility of the scrambled intermediate audio

### Suggested starter studies

| # | Question | Sweep | Watch |
|---|---|---|---|
| 1 | Which wavelet reconstructs speech best under noise? | wavelet × SNR | Recovered SNR |
| 2 | How much can we threshold before quality collapses? | threshold 0 → 0.2 | Retained %, MSE |
| 3 | Does deeper decomposition help or hurt under packet loss? | level 1 → 6 × loss | Correlation |
| 4 | How fragile is recovery to a near-miss key? | key ± 1 | Wrong-key corr. |
| 5 | What is the noise floor at which the correct key stops helping? | SNR −5 → 40 dB | Recovered SNR |

---

## 🧠 Interpreting Results

The lab demonstrates **three distinct ideas**:

1. **Wavelet representation.** DWT decomposes speech into multi-resolution coefficients. Different wavelets and levels expose different time-frequency structure.
2. **Reversible keyed scrambling.** A keyed permutation of the flattened coefficient stream makes the intermediate representation unintelligible while remaining **exactly reversible** when the correct key is available.
3. **Channel robustness.** AWGN and packet loss are applied to the scrambled coefficients. Even with the correct key, poor channel conditions degrade recovery, which motivates error-resilient coding in real systems.

A useful side effect to notice: because the permutation is applied to the *coefficient* stream, a lost packet doesn't remove a contiguous chunk of audio. After de-permutation, its damage is **spread across the signal** as scattered missing coefficients. Compare this with loss on the raw waveform.

---

## 🔒 Security Limitations

> **The default coefficient permutation is not cryptography.**

The keyed permutation is **reversible signal obfuscation**, not encryption. It is:

- ❌ Not resistant to cryptanalysis
- ❌ Not cryptographically authenticated
- ❌ Not a substitute for authenticated encryption
- ✅ Useful for teaching why shared configuration and a shared key are required

The optional `.wve` wrapper uses AES-256-GCM with Scrypt-derived keys to encrypt and authenticate a package. This is a separate mode and does not turn the default scrambled WAV, Neon bearer link, hosting environment, or surrounding application into an audited secure communications system.

### Known weaknesses (by design)

| Weakness | Why it matters |
|---|---|
| **Small key space** | The key is a non-negative integer used to seed a pseudo-random permutation. Its search space is tiny compared with a 128/256-bit cryptographic key. |
| **Magnitude statistics preserved** | A permutation reorders values but doesn't change them. The coefficient magnitude histogram (visible in the distribution plot) is identical before and after scrambling. |
| **Known-plaintext exposure** | If an attacker knows or can guess a portion of the original coefficients, the permutation can be partially recovered. |
| **No authentication** | A modified transmission WAV can be altered without proving origin or authenticity. |
| **Structure leakage** | Speech coefficients have exploitable structure (e.g., energy concentrated in low bands), which can guide an attacker toward the correct ordering. |

For real confidentiality and integrity, use a **standard, reviewed authenticated-encryption construction**.

---

## 📁 Project Structure

```text
Wavelet-Domain-Robust-Voice-Communication-Lab/
├── app.py                     # Sender/receiver UI and Research Lab
├── run_experiment.py          # CLI experiment runner
├── requirements.txt
├── pytest.ini
├── README.md
├── src/
│   ├── audio.py               # Loading, decoding, resampling
│   ├── channel.py             # AWGN, random loss, Gilbert–Elliott loss
│   ├── crypto.py              # Optional AES-GCM wrapper
│   ├── experiment_lab.py      # Bounded sweeps and FEC channel experiments
│   ├── fec.py                 # XOR parity and erasure recovery
│   ├── metrics.py             # SNR, segmental SNR, MSE, correlation, optional STOI
│   ├── packet.py              # Sequence-numbered frames, CRC, .wvp pack/reassemble
│   ├── share_store.py         # Expiring Neon shares and access limits
│   └── wavelet_codec.py       # DWT, thresholding, permutation, IDWT
└── tests/
    ├── test_audio_formats.py
    ├── test_channel.py
    ├── test_experiment.py
    ├── test_v4_upgrades.py
    └── test_wavelet_codec.py
```

| Module | Responsibility |
|---|---|
| `audio.py` | File loading, format decoding (via `ffmpeg`/`libsndfile`), resampling |
| `wavelet_codec.py` | DWT/IDWT, thresholding, keyed permutation and its inverse |
| `channel.py` | AWGN, independent losses, and Gilbert–Elliott burst losses |
| `packet.py` | Sequence numbering, CRC32, packet-stream serialization |
| `fec.py` | XOR parity packet generation and one-erasure-per-group recovery |
| `crypto.py` | Optional Scrypt + AES-256-GCM package encryption/authentication |
| `experiment_lab.py` | Reproducible packet experiments and capped parameter sweeps |
| `share_store.py` | Expiring Neon shares, per-link access limits, and cleanup |
| `metrics.py` | SNR, segmental SNR, MSE, correlation, optional STOI |
| `app.py` | Sender/receiver app, QR links, encrypted uploads, Research Lab |
| `run_experiment.py` | End-to-end CLI pipeline |

---

## ✅ Testing

```bash
pytest
```

The suite verifies:

- ✅ Exact DWT / IDWT reconstruction
- ✅ Coefficient thresholding behavior
- ✅ Scrambling changes coefficient ordering
- ✅ `scramble → descramble` is exactly reversible
- ✅ The same key produces the same permutation
- ✅ Channel behavior (noise level and packet loss)
- ✅ Audio format loading
- ✅ End-to-end experiment run
- ✅ Portable WAV transmission round-trip and rejection of ordinary WAV previews
- ✅ Sequence and CRC validation for packetized transmission files
- ✅ XOR FEC behavior and deterministic channel loss simulation
- ✅ AES-GCM round-trip, wrong-passphrase rejection, and tamper detection
- ✅ Bounded parameter sweeps and research-channel metrics

---

## 🛠️ Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| M4A / MP3 / AAC upload fails | `ffmpeg` missing | Install `ffmpeg` and `libsndfile1` (see [Quick Start](#-quick-start)) |
| Receiver says the WAV has no recovery payload | The uploaded file is a plain audio preview | Download **voice_transmission.wav**, **voice_transmission.wvp**, or an encrypted **voice_transmission.wve** from the sender |
| Receiver says the WAV is not a Wavelet Voice Lab transmission | The upload is not an exported transmission or its payload is corrupted | Re-export the .wav/.wvp/.wve package from the sender |
| Old V1/V2 transmission needs specific settings | Legacy formats may have less metadata than V3 | Prefer the current self-describing V3 WAV export; V3 settings load automatically |
| Encrypted .wve package fails | The AES-GCM passphrase is wrong or the package was modified | Verify the separate package passphrase; keep the wavelet key distinct |
| Packetized .wvp package fails | A packet is missing/corrupted or the stream was truncated | Re-export the .wvp package; CRC and sequence checks reject incomplete streams |
| Recovered audio is noise | Wrong key or damaged payload | Verify the separately shared key; V3 codec settings are read from the WAV metadata |
| Recovery is poor even with the right key | Channel too harsh in the experiment | Raise SNR or lower packet-loss; try a different wavelet or level |
| `streamlit: command not found` | Virtual environment not activated | Activate `.venv` and reinstall requirements |
| PowerShell blocks activation script | Execution policy | `Set-ExecutionPolicy -Scope Process RemoteSigned` |
| Results differ between runs | Different channel seed | Fix the seed for reproducible experiments |

---

## ❓ FAQ

**Is the default scrambling mode secure?**
No. The keyed permutation is an educational obfuscation method, not encryption. The optional .wve export uses AES-GCM authenticated encryption, but the overall service is still a research prototype. See [Security Limitations](#-security-limitations).

**Why send the key and the WAV separately?**
If both travel together, the obfuscation provides nothing at all. The WAV contains the scrambled signal and non-secret recovery metadata, while the key is delivered separately.

**Does the receiver need to select the same wavelet and level?**
No manual selection is needed for V3 transmissions. The WVTP header carries the wavelet and DWT level, and the receiver builds the correct coefficient layout from that metadata. Older V1/V2 WAVs remain supported through the compatibility reader.

**Why is the scrambled audio so loud and noisy?**
After permutation, energy that was concentrated in low-frequency bands is spread across the whole spectrum, so the intermediate signal sounds like broadband noise.

**Can I use my own recordings?**
Yes. WAV works directly; M4A / MP3 / AAC need `ffmpeg`.

**Can I make results reproducible?**
Yes. The key determines the permutation and the channel simulation accepts a seed.

---

## 🎯 Scope

This repository is intentionally limited to an **academic simulation** of:

- Digital signal processing
- Wavelet-domain representation
- Reversible coefficient scrambling
- Simulated communication-channel degradation
- Receiver-side reconstruction
- Quantitative audio-quality evaluation

It does **not** implement military radio protocols, tactical communication systems, frequency hopping, anti-interception procedures, or operational deployment.

---

## 📚 Learning Outcomes

By working through this lab, you will be able to:

- Explain how DWT produces a multi-resolution representation of speech
- Implement and reason about a deterministic keyed permutation
- Quantify reconstruction quality across codec and channel configurations
- Demonstrate why the receiver needs the same key **and** the same codec parameters
- Distinguish *signal obfuscation* from *cryptographic security*

---

## 🗺️ Roadmap

Completed in the V4 research branch:

- [x] Self-describing receiver metadata and one-click receive mode
- [x] Sequence-numbered packet transport with CRC32 checksums
- [x] XOR parity FEC and Gilbert–Elliott burst-loss experiments
- [x] Segmental SNR and optional STOI
- [x] AES-GCM comparison mode for exported packages
- [x] Bounded experiment sweeps with CSV export
- [x] QR receiver links and expiring/access-limited Neon shares

Potential future research directions:

- [ ] Stronger block FEC (Reed–Solomon or LDPC) and interleaving across burst losses
- [ ] Optional PESQ/POLQA evaluation where access and licensing permit
- [ ] Band-wise permutation vs. global permutation comparison
- [ ] Live streaming with bounded chunk buffering and sequence-aware replay
- [ ] Benchmark sweeps across larger datasets using a separate worker process

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-idea`
3. Add or update tests, then run `pytest`
4. Commit and push your branch
5. Open a pull request describing the change

---

## 📖 Citation

If you use this lab in coursework or research, you can cite it as:

```bibtex
@software{wavelet_voice_lab,
  title  = {Wavelet Voice Lab: Wavelet-Domain Robust Voice Communication Simulator},
  author = {<Your Name>},
  year   = {2026},
  url    = {https://github.com/<your-username>/Wavelet-Domain-Robust-Voice-Communication-Lab}
}
```

---

## 📄 License

Released under the **MIT License**. See `LICENSE` for details.

---

<div align="center">

**Wavelet Voice Lab** · Built for curious signal-processing minds 🛰️

</div>

