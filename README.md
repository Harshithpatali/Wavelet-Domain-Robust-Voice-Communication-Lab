# Wavelet-Domain Robust Voice Communication

A research prototype for studying wavelet-domain speech representation and reconstruction over simulated noisy wireless channels.

## What it does

- Loads a WAV speech recording or generates a synthetic test signal.
- Resamples audio to a configurable sample rate.
- Applies Discrete Wavelet Transform (DWT).
- Optionally thresholds small wavelet coefficients for compression.
- Simulates additive white Gaussian noise (AWGN) and packet loss.
- Reconstructs speech with inverse DWT (IDWT).
- Calculates MSE, SNR, retained-coefficient ratio, compression proxy, and correlation.
- Provides a Streamlit interface for interactive experiments.
- Includes automated tests.

## Safety / scope

This is an academic signal-processing and communications simulation. It does not implement military radio protocols, frequency hopping, anti-interception mechanisms, tactical communications, or operational encryption.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the web app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Run the command-line experiment

```bash
python run_experiment.py --input path/to/voice.wav --wavelet db4 --level 4 --threshold 0.02 --snr-db 10 --packet-loss 0.02
```

If `--input` is omitted, a synthetic speech-like test signal is generated.

## Project structure

```text
wavelet_voice_project/
├── app.py
├── run_experiment.py
├── requirements.txt
├── README.md
├── src/
│   ├── audio.py
│   ├── metrics.py
│   ├── wavelet_codec.py
│   └── channel.py
└── tests/
    ├── test_audio.py
    ├── test_channel.py
    └── test_wavelet_codec.py
```

## Research workflow

1. Start with no noise and no coefficient thresholding.
2. Compare Haar, db2, db4, db8, sym4, and coif1.
3. Vary decomposition level.
4. Vary threshold from 0 to larger values.
5. Vary channel SNR.
6. Vary packet-loss probability.
7. Plot the trade-off between retained coefficients and reconstruction SNR.

## Interpretation

Wavelet transformation is a reversible signal representation, not encryption. Anyone who knows or estimates the transform can reconstruct the signal. Security is intentionally outside the scope of this prototype.


## Audio input support

The Streamlit app accepts `.wav`, `.m4a`, `.mp3`, `.flac`, and `.aac`. WAV/FLAC-style files are read directly when possible. M4A/MP3/AAC are decoded through the bundled `imageio-ffmpeg` runtime, then normalized to **16 kHz, mono, float PCM** before DWT processing. The reconstructed result is always available as a standard WAV file.

No system-wide FFmpeg installation is required for the supported M4A/MP3 path because `imageio-ffmpeg` supplies an FFmpeg executable.
