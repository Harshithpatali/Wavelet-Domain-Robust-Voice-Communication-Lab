"""Small, versioned packet framing for communication experiments.

This framing is a research/demo protocol, not a production network protocol.
It detects accidental corruption and missing/reordered packets; it does not
authenticate senders or encrypt payloads.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

MAGIC = b"WVP1"
_HEADER = struct.Struct("!4sIIII")
MAX_PACKET_PAYLOAD = 16 * 1024 * 1024


@dataclass(frozen=True)
class DataPacket:
    sequence_number: int
    total_packets: int
    payload: bytes

    def encode(self) -> bytes:
        payload = bytes(self.payload)
        if self.total_packets < 1:
            raise ValueError("total_packets must be positive")
        if not 0 <= self.sequence_number < self.total_packets:
            raise ValueError("sequence_number must be within the packet sequence")
        if len(payload) > MAX_PACKET_PAYLOAD:
            raise ValueError("Packet payload exceeds the safety limit")
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        return _HEADER.pack(
            MAGIC, self.sequence_number, self.total_packets, len(payload), checksum
        ) + payload

    @classmethod
    def decode(cls, frame: bytes) -> "DataPacket":
        if len(frame) < _HEADER.size:
            raise ValueError("Packet frame is shorter than its header")
        magic, sequence, total, payload_length, checksum = _HEADER.unpack(
            frame[: _HEADER.size]
        )
        if magic != MAGIC:
            raise ValueError("Unknown packet framing version")
        if total < 1 or sequence >= total:
            raise ValueError("Packet sequence metadata is invalid")
        if payload_length > MAX_PACKET_PAYLOAD:
            raise ValueError("Packet payload exceeds the safety limit")
        payload = frame[_HEADER.size :]
        if len(payload) != payload_length:
            raise ValueError("Packet payload length does not match its header")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != checksum:
            raise ValueError("Packet checksum failed; payload may be corrupted")
        return cls(sequence, total, payload)


def packetize(data: bytes, payload_size: int = 1024) -> list[bytes]:
    """Split bytes into ordered frames containing sequence and CRC metadata."""
    data = bytes(data)
    if not data:
        raise ValueError("Cannot packetize an empty transmission")
    if payload_size < 1 or payload_size > MAX_PACKET_PAYLOAD:
        raise ValueError("payload_size is outside the supported range")
    chunks = [data[i : i + payload_size] for i in range(0, len(data), payload_size)]
    total = len(chunks)
    return [DataPacket(i, total, chunk).encode() for i, chunk in enumerate(chunks)]


def reassemble(frames: list[bytes] | tuple[bytes, ...]) -> bytes:
    """Validate, reorder, and join a complete set of packet frames."""
    if not frames:
        raise ValueError("No packet frames were supplied")
    packets = [DataPacket.decode(frame) for frame in frames]
    totals = {packet.total_packets for packet in packets}
    if len(totals) != 1:
        raise ValueError("Packet frames disagree about the transmission length")
    total = totals.pop()
    by_sequence: dict[int, bytes] = {}
    for packet in packets:
        if packet.sequence_number in by_sequence:
            raise ValueError("Duplicate packet sequence number")
        by_sequence[packet.sequence_number] = packet.payload
    if len(by_sequence) != total:
        raise ValueError("Transmission is incomplete: one or more packets are missing")
    return b"".join(by_sequence[i] for i in range(total))


_STREAM_MAGIC = b"WVS1"
_STREAM_HEADER = struct.Struct("!4sI")


def pack_transmission(data: bytes, payload_size: int = 1024) -> bytes:
    """Serialize a WAV or other byte payload as a sequence-numbered .wvp stream."""
    frames = packetize(data, payload_size=payload_size)
    if len(frames) > 1_000_000:
        raise ValueError("Transmission would require too many packets")
    return _STREAM_HEADER.pack(_STREAM_MAGIC, len(frames)) + b"".join(frames)


def unpack_transmission(stream: bytes) -> bytes:
    """Validate a packed stream and reassemble all packets in sequence order."""
    raw = bytes(stream)
    if len(raw) < _STREAM_HEADER.size:
        raise ValueError("Packed transmission header is incomplete")
    magic, declared_count = _STREAM_HEADER.unpack(raw[: _STREAM_HEADER.size])
    if magic != _STREAM_MAGIC:
        raise ValueError("Unknown packed transmission format")
    if declared_count < 1 or declared_count > 1_000_000:
        raise ValueError("Packed transmission packet count is invalid")

    pos = _STREAM_HEADER.size
    frames: list[bytes] = []
    while pos < len(raw):
        if pos + _HEADER.size > len(raw):
            raise ValueError("Packed transmission ends inside a packet header")
        header = raw[pos : pos + _HEADER.size]
        magic_packet, sequence, total, payload_length, _checksum = _HEADER.unpack(header)
        if magic_packet != MAGIC:
            raise ValueError("Packed transmission contains an invalid packet frame")
        if payload_length > MAX_PACKET_PAYLOAD:
            raise ValueError("Packet payload exceeds the safety limit")
        frame_end = pos + _HEADER.size + payload_length
        if frame_end > len(raw):
            raise ValueError("Packed transmission ends inside a packet payload")
        frame = raw[pos:frame_end]
        packet = DataPacket.decode(frame)
        if packet.sequence_number != sequence or packet.total_packets != total:
            raise ValueError("Packet header validation failed")
        frames.append(frame)
        pos = frame_end

    if len(frames) != declared_count:
        raise ValueError("Packed transmission packet count does not match its header")
    return reassemble(frames)
