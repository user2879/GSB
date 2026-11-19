"""Binary v1/v2 device frame codecs and protocol migration."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .constants import (
    CHANNEL_INDEX,
    CHANNELS,
    E_PROTO_FORMAT,
    NEW_CHANNEL_BITS,
    OLD_CHANNEL_BITS,
    OLD_PROTOCOL_FREQUENCY_HZ,
    PROTOCOL_FLAG_LOSSY_FREQUENCY,
    PROTOCOL_FLAG_MIGRATED,
    PROTOCOL_MAGIC,
    PROTOCOL_V1,
    PROTOCOL_V2,
    SUPPORTED_PROTOCOLS,
)
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile

HEADER = struct.Struct(">2sBBIQB")
CRC = struct.Struct(">I")
U16 = struct.Struct(">H")


class ProtocolFormatError(ValueError):
    """A typed protocol failure with the required frame/field location."""

    def __init__(self, frame_seq: int, field: str, detail: str):
        self.code = E_PROTO_FORMAT
        self.frame_seq = int(frame_seq)
        self.field = field
        self.detail = detail
        super().__init__(f"{self.code}: frame_seq={self.frame_seq} field={self.field} {detail}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "frame_seq": self.frame_seq,
            "field": self.field,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WireChannel:
    channel: str
    amplitude_code: int
    frequency_code: int | None

    def __post_init__(self) -> None:
        if self.channel not in CHANNEL_INDEX:
            raise ValueError(f"unknown wire channel: {self.channel!r}")
        if not 0 <= int(self.amplitude_code) <= 65535:
            raise ValueError("amplitude code outside uint16")
        if self.frequency_code is not None and not 0 <= int(self.frequency_code) <= 65535:
            raise ValueError("frequency code outside uint16")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "amplitude_code": self.amplitude_code,
            "frequency_code": self.frequency_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WireChannel":
        frequency_code = value.get("frequency_code")
        return cls(
            channel=str(value["channel"]),
            amplitude_code=int(value["amplitude_code"]),
            frequency_code=None if frequency_code is None else int(frequency_code),
        )


@dataclass(frozen=True)
class WireFrame:
    seq: int
    timestamp_us: int
    protocol_version: int
    channels: tuple[WireChannel, ...]
    flags: int = 0

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("frame sequence must be non-negative")
        if self.timestamp_us < 0:
            raise ValueError("timestamp must be non-negative")
        if self.protocol_version not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        channels = [item.channel for item in self.channels]
        if len(set(channels)) != len(channels):
            raise ValueError("duplicate channels in wire frame")
        if channels != sorted(channels, key=CHANNEL_INDEX.__getitem__):
            raise ValueError("wire channels must use canonical order")
        if self.protocol_version == PROTOCOL_V1:
            if any(item.frequency_code is not None for item in self.channels):
                raise ValueError("v1 wire frames cannot encode frequency codes")

    @property
    def mask(self) -> int:
        bit_table = NEW_CHANNEL_BITS if self.protocol_version == PROTOCOL_V2 else OLD_CHANNEL_BITS
        result = 0
        for item in self.channels:
            result |= bit_table[item.channel]
        return result

    @property
    def channel_by_name(self) -> dict[str, WireChannel]:
        return {item.channel: item for item in self.channels}

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_us": self.timestamp_us,
            "protocol_version": self.protocol_version,
            "flags": self.flags,
            "channels": [item.to_dict() for item in self.channels],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WireFrame":
        return cls(
            seq=int(value["seq"]),
            timestamp_us=int(value["timestamp_us"]),
            protocol_version=int(value["protocol_version"]),
            flags=int(value.get("flags", 0)),
            channels=tuple(WireChannel.from_dict(item) for item in value["channels"]),
        )


def _frame_seq_from_prefix(data: bytes) -> int:
    if len(data) >= 8:
        return struct.unpack(">I", data[4:8])[0]
    return -1


def _active_channels_for_mask(mask: int, version: int) -> list[str]:
    table = NEW_CHANNEL_BITS if version == PROTOCOL_V2 else OLD_CHANNEL_BITS
    valid_mask = 0
    for bit in table.values():
        valid_mask |= bit
    invalid = mask & ~valid_mask
    if invalid:
        raise ValueError(f"invalid channel mask bits 0x{invalid:02x}")
    return [name for name in CHANNELS if mask & table[name]]


def encode_frame(frame: WireFrame) -> bytes:
    header = HEADER.pack(
        PROTOCOL_MAGIC,
        frame.protocol_version,
        frame.flags & 0xFF,
        frame.seq,
        frame.timestamp_us,
        frame.mask,
    )
    body = bytearray()
    for item in frame.channels:
        body.extend(U16.pack(item.amplitude_code))
        if frame.protocol_version == PROTOCOL_V2:
            if item.frequency_code is None:
                raise ProtocolFormatError(frame.seq, "frequency_code", "v2 frame requires frequency code")
            body.extend(U16.pack(item.frequency_code))
    checksum = zlib.crc32(header + bytes(body)) & 0xFFFFFFFF
    return header + bytes(body) + CRC.pack(checksum)


def decode_frame(data: bytes) -> WireFrame:
    seq = _frame_seq_from_prefix(data)
    if len(data) < HEADER.size + CRC.size:
        raise ProtocolFormatError(seq, "frame_length", f"frame is {len(data)} bytes, minimum is {HEADER.size + CRC.size}")
    magic, version, flags, seq, timestamp_us, mask = HEADER.unpack(data[: HEADER.size])
    if magic != PROTOCOL_MAGIC:
        raise ProtocolFormatError(seq, "magic", f"expected {PROTOCOL_MAGIC!r}, got {magic!r}")
    if version not in SUPPORTED_PROTOCOLS:
        raise ProtocolFormatError(seq, "protocol_version", f"unsupported version {version}")
    try:
        active_channels = _active_channels_for_mask(mask, version)
    except ValueError as exc:
        raise ProtocolFormatError(seq, "channel_mask", str(exc)) from exc
    width = 4 if version == PROTOCOL_V2 else 2
    expected_length = HEADER.size + width * len(active_channels) + CRC.size
    if len(data) != expected_length:
        actual_body_length = max(0, len(data) - HEADER.size - CRC.size)
        field = "frame_length"
        if actual_body_length < width * len(active_channels):
            offset = 0
            for _channel in active_channels:
                if actual_body_length - offset < 2:
                    field = "amplitude"
                    break
                offset += 2
                if version == PROTOCOL_V2:
                    if actual_body_length - offset < 2:
                        field = "frequency"
                        break
                    offset += 2
        elif actual_body_length > width * len(active_channels):
            field = "trailing_bytes"
        raise ProtocolFormatError(seq, field, f"expected frame length {expected_length}, got {len(data)}")
    body = data[HEADER.size : -CRC.size]
    (expected_crc,) = CRC.unpack(data[-CRC.size :])
    actual_crc = zlib.crc32(data[:-CRC.size]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolFormatError(seq, "crc", f"expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}")
    channels: list[WireChannel] = []
    offset = 0
    for channel in active_channels:
        (amplitude_code,) = U16.unpack(body[offset : offset + 2])
        offset += 2
        frequency_code: int | None = None
        if version == PROTOCOL_V2:
            (frequency_code,) = U16.unpack(body[offset : offset + 2])
            offset += 2
        channels.append(
            WireChannel(
                channel=channel,
                amplitude_code=amplitude_code,
                frequency_code=frequency_code,
            )
        )
    return WireFrame(
        seq=seq,
        timestamp_us=timestamp_us,
        protocol_version=version,
        channels=tuple(channels),
        flags=flags,
    )


def migrate_to_v2(
    frame: WireFrame,
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> WireFrame:
    """Migrate a v1 frame, or return a v2 frame unchanged (idempotent)."""
    if frame.protocol_version == PROTOCOL_V2:
        return frame
    migrated_channels = tuple(
        WireChannel(
            channel=item.channel,
            amplitude_code=item.amplitude_code,
            frequency_code=profile.encode_frequency(OLD_PROTOCOL_FREQUENCY_HZ[item.channel]),
        )
        for item in frame.channels
    )
    return replace(
        frame,
        protocol_version=PROTOCOL_V2,
        channels=migrated_channels,
        flags=frame.flags | PROTOCOL_FLAG_MIGRATED,
    )


def downgrade_to_v1(frame: WireFrame) -> WireFrame:
    """Downgrade to v1. Frequencies are necessarily lost and later filled by
    the fixed migration table."""
    if frame.protocol_version == PROTOCOL_V1:
        return frame
    return replace(
        frame,
        protocol_version=PROTOCOL_V1,
        channels=tuple(
            WireChannel(item.channel, item.amplitude_code, None) for item in frame.channels
        ),
        flags=frame.flags | PROTOCOL_FLAG_LOSSY_FREQUENCY,
    )


def transcode_frame(frame: WireFrame, target_version: int) -> WireFrame:
    if target_version == frame.protocol_version:
        return frame
    if target_version == PROTOCOL_V2:
        return migrate_to_v2(frame)
    if target_version == PROTOCOL_V1:
        return downgrade_to_v1(frame)
    raise ProtocolFormatError(frame.seq, "protocol_version", f"unsupported target version {target_version}")


def migrate_wire_frames(frames: Iterable[WireFrame]) -> list[WireFrame]:
    return [migrate_to_v2(frame) for frame in frames]


def describe_frame_values(
    frame: WireFrame,
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> list[dict[str, Any]]:
    return [
        {
            "channel": item.channel,
            "amplitude": profile.decode_amplitude(item.amplitude_code),
            "frequency_hz": (
                None
                if item.frequency_code is None
                else profile.decode_frequency(item.frequency_code)
            ),
        }
        for item in frame.channels
    ]
