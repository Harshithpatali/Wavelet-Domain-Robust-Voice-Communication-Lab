"""Simple XOR forward-error-correction helpers for lab experiments.

One parity packet can recover one erased data packet in each group. This is a
teaching implementation; it is not a replacement for production FEC codes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.channel import gilbert_elliott_mask


@dataclass(frozen=True)
class XORFrame:
    data_packets: tuple[bytes, ...]
    parity_packets: tuple[bytes, ...]
    packet_lengths: tuple[int, ...]
    original_length: int
    packet_size: int
    group_size: int


def _padded(payload: bytes, size: int) -> bytes:
    if len(payload) > size:
        raise ValueError("Payload is larger than the configured packet size")
    return payload + (b"\x00" * (size - len(payload)))


def _xor_group(packets: list[bytes], packet_size: int) -> bytes:
    result = bytearray(packet_size)
    for payload in packets:
        padded = _padded(payload, packet_size)
        for index, value in enumerate(padded):
            result[index] ^= value
    return bytes(result)


def encode_xor(data: bytes, packet_size: int = 1024, group_size: int = 4) -> XORFrame:
    """Packetize bytes and create one XOR parity packet per data-packet group."""
    raw = bytes(data)
    if not raw:
        raise ValueError("Cannot encode an empty payload")
    if packet_size < 1:
        raise ValueError("packet_size must be positive")
    if group_size < 2:
        raise ValueError("group_size must be at least two")

    packets = [raw[i : i + packet_size] for i in range(0, len(raw), packet_size)]
    parity = [
        _xor_group(packets[i : i + group_size], packet_size)
        for i in range(0, len(packets), group_size)
    ]
    return XORFrame(
        data_packets=tuple(packets),
        parity_packets=tuple(parity),
        packet_lengths=tuple(map(len, packets)),
        original_length=len(raw),
        packet_size=packet_size,
        group_size=group_size,
    )


def _loss_mask(
    count: int,
    loss_probability: float,
    rng: np.random.Generator,
    burst: bool,
) -> np.ndarray:
    if not 0 <= loss_probability <= 1:
        raise ValueError("loss_probability must be in [0, 1]")
    if count <= 0 or loss_probability == 0:
        return np.zeros(count, dtype=bool)
    if not burst:
        return rng.random(count) < loss_probability
    # A two-state Markov model deliberately produces correlated/burst losses.
    return gilbert_elliott_mask(
        count,
        p_good_to_bad=min(0.5, max(0.01, loss_probability * 0.2)),
        p_bad_to_good=0.25,
        loss_good=min(0.1, loss_probability * 0.15),
        loss_bad=max(0.5, loss_probability),
        rng=rng,
    )


def transmit_with_xor_fec(
    data: bytes,
    loss_probability: float,
    *,
    use_fec: bool = True,
    packet_size: int = 1024,
    group_size: int = 4,
    burst: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[bytes, dict[str, int | float | bool]]:
    """Simulate packet erasures and optionally repair one loss per group.

    Missing/unrecoverable bytes are zero-filled so the receiver gets a
    deterministic-length payload instead of malformed/truncated data.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    frame = encode_xor(data, packet_size=packet_size, group_size=group_size)
    parity_count = len(frame.parity_packets) if use_fec else 0
    mask = _loss_mask(
        len(frame.data_packets) + parity_count,
        loss_probability,
        rng,
        burst,
    )
    data_missing = mask[: len(frame.data_packets)].tolist()
    parity_missing = (
        mask[len(frame.data_packets) :].tolist() if use_fec else []
    )
    received: list[bytes | None] = [
        None if missing else payload
        for payload, missing in zip(frame.data_packets, data_missing)
    ]
    parity_received: list[bytes | None] = [
        None if missing else payload
        for payload, missing in zip(frame.parity_packets, parity_missing)
    ] if use_fec else []

    recovered_count = 0
    group_count = 0
    for group_index, start in enumerate(range(0, len(received), frame.group_size)):
        group_count += 1
        end = min(start + frame.group_size, len(received))
        missing_indexes = [idx for idx in range(start, end) if received[idx] is None]
        if (
            use_fec
            and len(missing_indexes) == 1
            and group_index < len(parity_received)
            and parity_received[group_index] is not None
        ):
            target = missing_indexes[0]
            restored = bytearray(parity_received[group_index] or b"")
            for idx in range(start, end):
                if idx == target:
                    continue
                known = received[idx]
                if known is None:
                    break
                padded = _padded(known, frame.packet_size)
                for byte_index, value in enumerate(padded):
                    restored[byte_index] ^= value
            else:
                received[target] = bytes(restored[: frame.packet_lengths[target]])
                recovered_count += 1

    unrecovered_count = sum(payload is None for payload in received)
    safe_packets = [
        payload if payload is not None else (b"\x00" * length)
        for payload, length in zip(received, frame.packet_lengths)
    ]
    output = b"".join(safe_packets)[: frame.original_length]
    lost_count = sum(data_missing)
    stats: dict[str, int | float | bool] = {
        "data_packets": len(frame.data_packets),
        "parity_packets": len(frame.parity_packets) if use_fec else 0,
        "lost_data_packets": int(lost_count),
        "recovered_packets": int(recovered_count),
        "unrecovered_packets": int(unrecovered_count),
        "total_groups": int(group_count),
        "fec_enabled": bool(use_fec),
        "burst_channel": bool(burst),
        "redundancy_percent": (
            100.0 * len(frame.parity_packets) / len(frame.data_packets)
            if use_fec else 0.0
        ),
        "data_recovery_rate": (
            1.0 if lost_count == 0 else float(recovered_count / lost_count)
        ),
    }
    return output, stats
