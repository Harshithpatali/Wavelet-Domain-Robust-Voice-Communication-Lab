# Test status

- Python syntax/bytecode compilation: **PASS**
- Audio generation: **PASS**
- AWGN channel simulation: **PASS**
- Packet-loss simulation: **PASS**
- MSE/SNR/correlation metrics: **PASS**
- DWT/IDWT round-trip and threshold tests: **PASS**
- WAV byte-input decoding: **PASS**
- M4A decode → PCM normalization test: **PASS**

The Streamlit uploader now accepts WAV, M4A, MP3, FLAC, and AAC. M4A/MP3/AAC are decoded through `imageio-ffmpeg` and normalized to 16 kHz mono PCM before wavelet processing. Reconstructed output is downloadable as WAV.
