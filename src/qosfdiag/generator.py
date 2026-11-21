"""Deterministic packet generator and a versioned loopback datagram codec."""

from __future__ import annotations

import socket
import struct
from dataclasses import replace
from typing import Any, Iterable, Mapping

import numpy as np

from .canonical import digest
from .models import GeneratorConfig, Packet


MAGIC = b"QFD1"
HEADER = struct.Struct(">4sQQQIH")


def _require_int(value: Any, field: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def parse_generator_config(value: Mapping[str, Any]) -> GeneratorConfig:
    seed = _require_int(value.get("seed"), "seed", 0)
    count = _require_int(value.get("count"), "count", 1)
    packet_interval_ns = _require_int(
        value.get("packet_interval_ns"), "packet_interval_ns", 1
    )
    start_time_ns = _require_int(value.get("start_time_ns", 0), "start_time_ns", 0)
    min_packet_bytes = _require_int(
        value.get("min_packet_bytes"), "min_packet_bytes", HEADER.size + 8
    )
    max_packet_bytes = _require_int(
        value.get("max_packet_bytes"), "max_packet_bytes", min_packet_bytes
    )
    if max_packet_bytes > 65507:
        raise ValueError("max_packet_bytes must be <= 65507 for IPv4 UDP")
    dst_port_start = _require_int(
        value.get("dst_port_start"), "dst_port_start", 1, 65535
    )
    dst_port_span = _require_int(value.get("dst_port_span", 1), "dst_port_span", 1)
    if dst_port_start + dst_port_span - 1 > 65535:
        raise ValueError("dst_port_start + dst_port_span - 1 must be <= 65535")
    batch_size = _require_int(value.get("batch_size", 4), "batch_size", 1)
    return GeneratorConfig(
        seed=seed,
        count=count,
        packet_interval_ns=packet_interval_ns,
        start_time_ns=start_time_ns,
        min_packet_bytes=min_packet_bytes,
        max_packet_bytes=max_packet_bytes,
        dst_port_start=dst_port_start,
        dst_port_span=dst_port_span,
        batch_size=batch_size,
    )


def encode_packet(packet: Packet) -> bytes:
    payload = bytes(packet.payload)
    total_size = HEADER.size + len(payload)
    if total_size != packet.size_bytes:
        raise ValueError(
            f"packet {packet.packet_id} size mismatch: header+payload={total_size}, "
            f"declared={packet.size_bytes}"
        )
    header = HEADER.pack(
        MAGIC,
        packet.seq,
        packet.timestamp_ns,
        packet.size_bytes,
        packet.dst_port,
        len(payload),
    )
    return header + payload


def decode_packet(datagram: bytes) -> Packet:
    if len(datagram) < HEADER.size:
        raise ValueError("truncated datagram")
    magic, seq, timestamp_ns, size_bytes, dst_port, payload_len = HEADER.unpack_from(
        datagram
    )
    if magic != MAGIC:
        raise ValueError("invalid datagram magic")
    if HEADER.size + payload_len != len(datagram):
        raise ValueError("invalid payload length")
    if size_bytes != len(datagram):
        raise ValueError("invalid declared datagram size")
    payload = datagram[HEADER.size :]
    return Packet(
        packet_id=f"p{seq:08d}",
        seq=seq,
        timestamp_ns=timestamp_ns,
        size_bytes=size_bytes,
        dst_port=dst_port,
        payload=payload,
    )


class DeterministicUdpGenerator:
    """Build a deterministic packet sequence from a seed and stream it via UDP."""

    def __init__(self, config: GeneratorConfig | Mapping[str, Any]) -> None:
        self.config = (
            config if isinstance(config, GeneratorConfig) else parse_generator_config(config)
        )
        overhead = HEADER.size
        for field in (self.config.min_packet_bytes, self.config.max_packet_bytes):
            if field < overhead + 1:
                raise ValueError(f"packet size must exceed codec overhead {overhead}")
        if self.config.packet_interval_ns < 1:
            raise ValueError("packet interval must be positive")
        self.packets = self._build_packets()

    def _build_packets(self) -> tuple[Packet, ...]:
        rng = np.random.Generator(np.random.PCG64(self.config.seed))
        packets: list[Packet] = []
        for seq in range(self.config.count):
            size_bytes = int(
                rng.integers(
                    self.config.min_packet_bytes,
                    self.config.max_packet_bytes + 1,
                    endpoint=False,
                )
            )
            port_offset = int(
                rng.integers(0, self.config.dst_port_span, endpoint=False)
            )
            payload_len = size_bytes - HEADER.size
            payload = rng.integers(0, 256, size=payload_len, dtype=np.uint8).tobytes()
            packet = Packet(
                packet_id=f"p{seq:08d}",
                seq=seq,
                timestamp_ns=self.config.start_time_ns
                + seq * self.config.packet_interval_ns,
                size_bytes=size_bytes,
                dst_port=self.config.dst_port_start + port_offset,
                payload=payload,
            )
            packets.append(packet)
        return tuple(packets)

    @property
    def manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(packet.manifest() for packet in self.packets)

    @property
    def sequence_digest(self) -> str:
        return digest(list(self.manifest))

    def iter_packets(self) -> Iterable[Packet]:
        return iter(self.packets)

    def datagrams(self) -> Iterable[bytes]:
        for packet in self.packets:
            yield encode_packet(packet)

    def send_one(self, packet: Packet, destination_port: int) -> int:
        datagram = encode_packet(packet)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            return sock.sendto(datagram, ("127.0.0.1", destination_port))

    def send_all(self, destination_port: int) -> int:
        total = 0
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            for datagram in self.datagrams():
                total += sock.sendto(datagram, ("127.0.0.1", destination_port))
        return total

    def with_packet_payload(self, packet: Packet, payload: bytes) -> Packet:
        return replace(packet, payload=payload)
