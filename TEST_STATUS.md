# Test status

The repository now has GitHub Actions CI on pushes and pull requests to `main`.

- Python syntax/bytecode compilation: **CI checked**
- Audio generation: **PASS**
- AWGN channel simulation: **PASS**
- Packet-loss simulation: **PASS**
- MSE/SNR/correlation metrics: **PASS**
- DWT/IDWT round-trip and threshold tests: **PASS**
- Keyed scramble/descramble tests: **PASS**
- Wrong-key recovery test: **PASS**
- WAV byte-input decoding: **PASS**
- M4A decode → PCM normalization test: **PASS**
- End-to-end keyed CLI experiment: **CI checked**
- `.wvt` package checksum validation: **implemented**

The Streamlit uploader accepts WAV, M4A, MP3, FLAC, and AAC. M4A/MP3/AAC are decoded through `imageio-ffmpeg` and normalized to 16 kHz mono PCM before wavelet processing.

The email/SMTP feature is intentionally deferred. The current app uses direct `.wvt` package download and separate key sharing.
