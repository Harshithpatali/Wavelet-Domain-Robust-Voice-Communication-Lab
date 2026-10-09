"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  BLOCK_SIZE,
  makePermutation,
  haarForward,
  haarInverse,
  processBlock,
} = require("../call_service/wavelet_audio_worklet.js");

function sampleSignal() {
  return Float32Array.from({ length: BLOCK_SIZE }, (_, i) =>
    0.42 * Math.sin(2 * Math.PI * 440 * i / 48000) +
    0.17 * Math.sin(2 * Math.PI * 880 * i / 48000) +
    0.04 * Math.cos(2 * Math.PI * 127 * i / 48000)
  );
}

function maxAbsDifference(a, b) {
  let maximum = 0;
  for (let i = 0; i < a.length; i += 1) {
    maximum = Math.max(maximum, Math.abs(a[i] - b[i]));
  }
  return maximum;
}

test("five-level Haar DWT round-trips a 20 ms audio block", () => {
  const input = sampleSignal();
  assert.ok(maxAbsDifference(input, haarInverse(haarForward(input))) < 2e-6);
});

test("same wavelet key creates the same permutation", () => {
  assert.deepEqual(
    Array.from(makePermutation(BLOCK_SIZE, "shared test passphrase")),
    Array.from(makePermutation(BLOCK_SIZE, "shared test passphrase"))
  );
});

test("correct wavelet key recovers a scrambled block", () => {
  const input = sampleSignal();
  const scrambled = processBlock(input, "scramble", makePermutation(BLOCK_SIZE, "test code A"));
  assert.ok(maxAbsDifference(input, scrambled) > 0.05);
  const recovered = processBlock(scrambled, "recover", makePermutation(BLOCK_SIZE, "test code A"));
  assert.ok(maxAbsDifference(input, recovered) < 3e-6);
});

test("wrong wavelet key does not recover the original block", () => {
  const input = sampleSignal();
  const scrambled = processBlock(input, "scramble", makePermutation(BLOCK_SIZE, "correct code"));
  const recovered = processBlock(scrambled, "recover", makePermutation(BLOCK_SIZE, "wrong code"));
  assert.ok(maxAbsDifference(input, recovered) > 0.02);
});
