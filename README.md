# Wavelet-Domain Robust Voice Communication Lab

An academic signal-processing prototype that demonstrates a **transmitter → scrambled channel → receiver** workflow for voice.

The central experiment is:

```text
TRANSMITTER
Original speech
      ↓
DWT (selected wavelet + level)
      ↓
Coefficient thresholding
      ↓
KEYED COEFFICIENT SCRAMBLING
      ↓
────────────────────────────────
     TRANSMITTED SIGNAL
     intentionally unintelligible
────────────────────────────────
      ↓
AWGN / packet-loss simulation
      ↓
RECEIVER
Same wavelet + level + threshold + shared key
      ↓
Reverse coefficient scrambling
      ↓
IDWT
      ↓
Recovered speech
```

## Important distinction

A wavelet transform by itself is **not encryption**. DWT/IDWT is a reversible representation of the signal.

This project adds a **deterministic keyed permutation of the flattened wavelet coefficients** so that the signal reconstructed from the transmitted representation is intentionally not recognizable as speech. The receiver uses the same key to restore the coefficient ordering before IDWT.

This keyed permutation is an **academic obfuscation/signal-processing demonstration, not cryptographic security**. For actual confidentiality, a standard authenticated encryption scheme should be used.

## What the Streamlit app demonstrates

The UI exposes the complete communication path:

1. **Original Audio** — clear input speech.
2. **Transmitted / Scrambled Audio** — the intentionally unintelligible intermediate representation.
3. **Received Scrambled Audio** — the scrambled representation after simulated channel noise/loss.
4. **Receiver Recovered Audio** — coefficient descrambling followed by IDWT.

The sidebar contains the transmission configuration:

- Wavelet: Haar, Daubechies, Symlet, or Coiflet options.
- DWT decomposition level.
- Coefficient threshold.
- Transmitter scrambling key (locked before upload).
- Channel SNR.
- Packet/sample loss probability.
- Channel random seed.

The same wavelet, level, threshold and scrambling key are used by the receiver in the experiment.

## Why the middle audio is different from the original

The earlier version of this project performed:

```text
DWT → threshold → IDWT
```

That naturally produces speech-like audio because the inverse transform reconstructs the signal.

The current version performs:

```text
DWT
  ↓
threshold
  ↓
keyed coefficient permutation
  ↓
IDWT
```

The coefficient ordering is intentionally changed before the intermediate audio is reconstructed. Therefore the **transmitted/scrambled audio is not expected to sound like the original speech**.

At the receiver:

```text
received scrambled coefficients
  ↓
reverse the same permutation
  ↓
original coefficient ordering
  ↓
IDWT
```

This restores the signal, subject to any simulated channel corruption.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

If no file is uploaded, the app uses a deterministic synthetic speech-like signal.

## Supported audio files

The app accepts:

- WAV
- M4A
- MP3
- FLAC
- AAC

M4A/MP3/AAC are decoded through `imageio-ffmpeg` and normalized to 16 kHz mono PCM before DWT processing. Recovered audio can be downloaded as WAV.

## Run tests

```bash
pytest
```

The test suite verifies:

- Exact DWT/IDWT reconstruction.
- Coefficient thresholding.
- Scrambling changes coefficient ordering.
- Scramble → descramble is exactly reversible.
- The same key produces the same permutation.

## Run the command-line experiment

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

If `--input` is omitted, a synthetic speech-like test signal is generated.

## Project structure

```text
Wavelet-Domain-Robust-Voice-Communication-Lab/
├── app.py
├── run_experiment.py
├── requirements.txt
├── pytest.ini
├── README.md
├── src/
│   ├── audio.py
│   ├── channel.py
│   ├── metrics.py
│   └── wavelet_codec.py
└── tests/
    ├── test_audio_formats.py
    ├── test_channel.py
    ├── test_experiment.py
    └── test_wavelet_codec.py
```



## Receiver WAV workflow

Email/SMTP is intentionally deferred. The current app uses a single WAV transmission file.

### Sender

1. Choose the secret key.
2. Click **Set & Lock Key**.
3. Only then upload the original voice.
4. The app performs DWT, thresholding, and keyed coefficient scrambling.
5. Download **voice_transmission.wav**.
6. Send that WAV and the same key separately to the receiver.

The transmission WAV is stereo:

- **Channel 1:** the audible scrambled voice, so the receiver can listen before entering the key.
- **Channel 2:** the scrambled wavelet coefficient payload plus a small non-secret header required to reconstruct the coefficient layout.

The secret key is **never stored in the WAV**.

### Receiver

1. Select **Receive Shared Transmission**.
2. Upload **voice_transmission.wav**.
3. The app immediately presents the **scrambled transmission** for listening.
4. Enter the key supplied separately by the sender.
5. Click **Recover original voice**.
6. The app reverses the keyed coefficient permutation and performs IDWT.
7. Listen to or download **recovered_voice.wav**.

The receiver does not need .wvt, .zip, or SMTP/email configuration.

## Transmission Verification Dashboard

The Streamlit app now includes a dedicated verification section so the experiment is not based only on listening.

It reports:

- **Coefficients reordered (%)** — fraction of coefficient positions changed by the key-driven permutation.
- **Original ↔ scrambled coefficient correlation** — similarity between the coefficient stream before and after scrambling.
- **Original ↔ correct-key correlation** — similarity between the original speech and correctly recovered speech.
- **Original ↔ wrong-key correlation** — similarity after deliberately using a different receiver key.
- Correct-key and wrong-key reconstruction SNR.

### Coefficient-order visualization

The dashboard plots:

1. Original coefficient values versus the transmitted scrambled ordering.
2. The permutation map for the first coefficients.

A diagonal permutation map would indicate no reordering. A scrambled map deviates from that diagonal.

### Wrong-key demonstration

The app automatically tests:

```text
Transmitter key = K
Receiver key    = K      → correct recovery ✓

Transmitter key = K
Receiver key    = K + 1  → incorrect recovery ✗
```

The wrong-key audio is included as a separate playable stage. This provides a practical demonstration that the receiver needs the shared scrambling key to restore the coefficient ordering.

The wrong-key test is a **demonstration of the implemented reversible permutation**, not a cryptographic security proof.

## Research experiment

A useful experiment matrix is:

| Parameter | Example values |
|---|---|
| Wavelet | haar, db2, db4, db8, sym4, coif1 |
| DWT level | 1–6 |
| Threshold | 0.00–0.20 |
| Shared key | Any non-negative integer |
| Channel SNR | -5–40 dB |
| Packet/sample loss | 0–20% |

For every configuration, compare:

- Original vs recovered waveform.
- Recovered SNR.
- MSE.
- Correlation.
- Retained coefficient percentage.
- Subjective intelligibility of the scrambled intermediate audio.

## Interpretation

The experiment demonstrates three separate ideas:

### 1. Wavelet representation

DWT decomposes the speech signal into multi-resolution coefficients. Different wavelets and decomposition levels produce different coefficient structures.

### 2. Reversible keyed scrambling

The coefficient permutation makes the intermediate representation difficult to recognize as ordinary speech while remaining exactly reversible when the correct key is available.

### 3. Channel robustness

AWGN and packet/sample loss are applied to the scrambled coefficient stream. The receiver then attempts to restore the original coefficient ordering and reconstruct the speech.

A poor channel can therefore reduce recovered audio quality even when the correct key is available.

## Security limitation

The scrambling mechanism in this project should **not** be described as military-grade, secure encryption, or a replacement for cryptography.

The key-driven permutation is useful for demonstrating the signal-processing concept:

```text
same configuration + correct key
        → recover

different key / unknown key
        → coefficient ordering remains incorrect
```

For a production communication system requiring confidentiality and authentication, use a standard, reviewed authenticated-encryption construction rather than relying on wavelet transforms or coefficient permutations.

## Scope

This repository is intentionally limited to an academic simulation of:

- Digital signal processing.
- Wavelet-domain representation.
- Reversible coefficient scrambling.
- Simulated communication-channel degradation.
- Receiver-side reconstruction.
- Quantitative audio-quality evaluation.

It does not implement military radio protocols, tactical communication systems, frequency hopping, anti-interception procedures, or operational deployment.

## Key-first transmission workflow

Email delivery is intentionally **deferred** for now. The current app focuses on the core transmitter/receiver experiment and uses a portable `.wav` package.

### Sender

Choose:

    Communication Mode → Transmit & Send

Then:

1. Choose the secret **transmitter key**.
2. Click **Set & Lock Key**.
3. Only after the key is locked does audio upload become available.
4. Upload the voice recording.
5. The locked key drives the deterministic coefficient permutation.
6. Download the generated `.wav` transmission WAV.
7. Share the `.wav` package and the same key with the intended receiver through separate channels.

The transmission WAV contains:

- scrambled wavelet coefficients;
- wavelet name;
- DWT level;
- sample rate;
- original signal length;
- threshold metadata;
- a coefficient-payload SHA-256 checksum.

The **secret key is never stored in the WAV**.

### Receiver

Choose:

    Communication Mode → Receive Shared Transmission

Then:

1. Upload the `.wav` transmission.
2. Enter the key supplied separately by the sender.
3. The app derives the same deterministic permutation locally.
4. The receiver reverses the coefficient ordering and applies IDWT.
5. Listen to or download the recovered WAV.

The receiver UI deliberately does not claim that an entered integer is cryptographically authenticated. A correct key is verified experimentally by the recovered audio and similarity metrics.

### Why a checksum is included

The current WAV transmission uses a non-secret header in its second audio channel to describe the coefficient payload. The secret key is never embedded in that header. Production systems should add authenticated integrity protection.

## Scope

This repository is intentionally limited to an academic simulation of:

- Digital signal processing.
- Wavelet-domain representation.
- Reversible coefficient scrambling.
- Simulated communication-channel degradation.
- Receiver-side reconstruction.
- Quantitative audio-quality evaluation.

It does not implement military radio protocols, tactical communication systems, frequency hopping, anti-interception procedures, or operational deployment.

## Security note

The keyed coefficient permutation is **reversible signal obfuscation, not cryptographic encryption**. It should not be described as military-grade security or as a replacement for cryptography.

For real confidentiality, integrity, and authentication, use a standard reviewed authenticated-encryption construction.

