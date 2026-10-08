# Test status

The repository has GitHub Actions CI on pushes and pull requests to main.

## Automated checks

- Python syntax/bytecode compilation: configured in CI
- Audio generation: PASS
- AWGN channel simulation: PASS
- Packet-loss simulation: PASS
- MSE/SNR/correlation metrics: PASS
- DWT/IDWT round-trip and threshold tests: PASS
- Keyed scramble/descramble tests: PASS
- Wrong-key recovery test: PASS
- WAV byte-input decoding: PASS
- M4A decode → PCM normalization test: PASS
- End-to-end keyed CLI experiment: configured in CI

## Current transmission workflow

- Sender key must be locked before audio upload.
- Sender exports a stereo voice_transmission.wav.
- Receiver accepts WAV only.
- Receiver can listen to the scrambled voice before entering the key.
- Receiver enters the separately shared key and recovers the original voice.
- The secret key is never stored in the WAV.

## Important limitation

The WAV header carries only non-secret reconstruction metadata. The keyed permutation is reversible signal obfuscation, not cryptographic encryption. Production confidentiality and integrity should use authenticated encryption.

