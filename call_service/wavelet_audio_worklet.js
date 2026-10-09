"use strict";

/*
 * Experimental, frame-based Haar-DWT voice scrambling for Web Audio.
 *
 * This is reversible DSP obfuscation, not cryptography. WebRTC's DTLS-SRTP
 * remains the network media security layer. Blocks are 960 samples at 48 kHz
 * (~20 ms) to keep processing latency bounded and align with common Opus frames.
 */

const BLOCK_SIZE = 960;
const WAVELET_LEVELS = 5;
const SQRT_HALF = Math.SQRT1_2;

function seedFromKey(key) {
  let hash = 2166136261;
  const value = String(key);
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash || 0x9e3779b9;
}

function makePermutation(size, key) {
  const permutation = new Uint16Array(size);
  for (let i = 0; i < size; i += 1) permutation[i] = i;

  let state = seedFromKey(key);
  const next = () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4294967296;
  };

  for (let i = size - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1));
    const temp = permutation[i];
    permutation[i] = permutation[j];
    permutation[j] = temp;
  }
  return permutation;
}

/* Orthogonal, multi-level Haar DWT. Layout: cA_L, cD_L, ..., cD_1. */
function haarForward(samples, levels = WAVELET_LEVELS) {
  const input = Float32Array.from(samples);
  if (!input.length || input.length % (1 << levels) !== 0) {
    throw new Error("Audio block length must be divisible by 2^levels.");
  }

  let approximation = input;
  const details = [];
  for (let level = 1; level <= levels; level += 1) {
    const half = approximation.length / 2;
    const nextApproximation = new Float32Array(half);
    const detail = new Float32Array(half);
    for (let i = 0; i < half; i += 1) {
      const even = approximation[2 * i];
      const odd = approximation[2 * i + 1];
      nextApproximation[i] = (even + odd) * SQRT_HALF;
      detail[i] = (even - odd) * SQRT_HALF;
    }
    details.push(detail);
    approximation = nextApproximation;
  }

  const coefficients = new Float32Array(input.length);
  coefficients.set(approximation, 0);
  let offset = approximation.length;
  for (let i = details.length - 1; i >= 0; i -= 1) {
    coefficients.set(details[i], offset);
    offset += details[i].length;
  }
  return coefficients;
}

/* Inverse of haarForward, using the same coefficient layout. */
function haarInverse(coefficients, levels = WAVELET_LEVELS) {
  const input = Float32Array.from(coefficients);
  if (!input.length || input.length % (1 << levels) !== 0) {
    throw new Error("Coefficient length must be divisible by 2^levels.");
  }

  const blockLength = input.length;
  let approximationLength = blockLength / (1 << levels);
  let approximation = input.slice(0, approximationLength);
  let offset = approximationLength;

  for (let level = levels; level >= 1; level -= 1) {
    const detailLength = blockLength / (1 << level);
    const detail = input.slice(offset, offset + detailLength);
    offset += detailLength;
    const reconstructed = new Float32Array(approximation.length * 2);
    for (let i = 0; i < approximation.length; i += 1) {
      reconstructed[2 * i] = (approximation[i] + detail[i]) * SQRT_HALF;
      reconstructed[2 * i + 1] = (approximation[i] - detail[i]) * SQRT_HALF;
    }
    approximation = reconstructed;
  }
  return approximation;
}

/*
 * Transform one audio-time block:
 * scramble = IDWT(permute(DWT(audio)))
 * recover  = IDWT(inverse_permute(DWT(scrambled_audio)))
 */
function processBlock(samples, mode, permutation) {
  const coefficients = haarForward(samples);
  const mapped = new Float32Array(coefficients.length);

  if (mode === "scramble") {
    for (let i = 0; i < permutation.length; i += 1) {
      mapped[i] = coefficients[permutation[i]];
    }
  } else if (mode === "recover") {
    for (let i = 0; i < permutation.length; i += 1) {
      mapped[permutation[i]] = coefficients[i];
    }
  } else {
    throw new Error("Unknown wavelet audio processor mode.");
  }
  return haarInverse(mapped);
}

if (typeof module === "object" && module.exports) {
  module.exports = {
    BLOCK_SIZE,
    WAVELET_LEVELS,
    seedFromKey,
    makePermutation,
    haarForward,
    haarInverse,
    processBlock,
  };
} else {
  class WaveletBlockProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const processorOptions = options.processorOptions || {};
      this.mode = processorOptions.mode || "scramble";
      this.permutation = makePermutation(BLOCK_SIZE, processorOptions.key || "");
      this.inputBlock = new Float32Array(BLOCK_SIZE);
      this.inputFill = 0;
      this.outputRing = new Float32Array(BLOCK_SIZE * 4);
      this.readIndex = 0;
      this.writeIndex = 0;
      this.outputCount = 0;
    }

    process(inputs, outputs) {
      const input = inputs[0] && inputs[0][0];
      const output = outputs[0] && outputs[0][0];
      if (!output) return true;

      // Collect and transform before draining so the 960-sample block size
      // remains continuous even when AudioWorklet quanta are 128 samples.
      if (input) {
        for (let i = 0; i < input.length; i += 1) {
          this.inputBlock[this.inputFill] = input[i];
          this.inputFill += 1;
          if (this.inputFill === BLOCK_SIZE) {
            const transformed = processBlock(this.inputBlock, this.mode, this.permutation);
            for (let j = 0; j < transformed.length; j += 1) {
              if (this.outputCount >= this.outputRing.length) {
                // A persistent overrun indicates a browser scheduling problem;
                // drop the oldest queued sample rather than blocking the audio thread.
                this.readIndex = (this.readIndex + 1) % this.outputRing.length;
                this.outputCount -= 1;
              }
              this.outputRing[this.writeIndex] = transformed[j];
              this.writeIndex = (this.writeIndex + 1) % this.outputRing.length;
              this.outputCount += 1;
            }
            this.inputFill = 0;
          }
        }
      }

      for (let i = 0; i < output.length; i += 1) {
        if (this.outputCount > 0) {
          output[i] = this.outputRing[this.readIndex];
          this.readIndex = (this.readIndex + 1) % this.outputRing.length;
          this.outputCount -= 1;
        } else {
          // One block of buffering is necessary before the first transformed block.
          output[i] = 0;
        }
      }
      return true;
    }
  }

  registerProcessor("wavelet-block-processor", WaveletBlockProcessor);
}
