"""Compiled frame records, content hashes, and batch digests."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import sha256_bytes
from .codec import WireChannel, WireFrame, encode_frame
from .constants import CHANNEL_INDEX, PROTOCOL_V2
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile


@dataclass(frozen=True)
class ChannelRecord:
    channel: str
    amplitude: float
    frequency_hz: float
    amplitude_code: int
    frequency_code: int | None
    amplitude_error: float
    frequency_error: float | None

    def __post_init__(self) -> None:
        if self.channel not in CHANNEL_INDEX:
            raise ValueError(f"unknown channel in record: {self.channel!r}")

    def wire_channel(self) -> WireChannel:
        return WireChannel(
            channel=self.channel,
            amplitude_code=self.amplitude_code,
            frequency_code=self.frequency_code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "amplitude_code": self.amplitude_code,
            "frequency_code": self.frequency_code,
            "amplitude_error": self.amplitude_error,
            "frequency_error": self.frequency_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelRecord":
        return cls(
            channel=str(value["channel"]),
            amplitude=float(value["amplitude"]),
            frequency_hz=float(value["frequency_hz"]),
            amplitude_code=int(value["amplitude_code"]),
            frequency_code=(
                None if value.get("frequency_code") is None else int(value["frequency_code"])
            ),
            amplitude_error=float(value["amplitude_error"]),
            frequency_error=(
                None if value.get("frequency_error") is None else float(value["frequency_error"])
            ),
        )


@dataclass(frozen=True)
class FrameRecord:
    seq: int
    timestamp_us: int
    protocol_version: int
    channels: tuple[ChannelRecord, ...]
    flags: int = 0

    def __post_init__(self) -> None:
        if self.seq < 0 or self.timestamp_us < 0:
            raise ValueError("invalid frame sequence or timestamp")
        channel_names = [item.channel for item in self.channels]
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("duplicate channels in frame record")
        if channel_names != sorted(channel_names, key=CHANNEL_INDEX.__getitem__):
            raise ValueError("frame channels must use canonical order")

    @property
    def mask(self) -> int:
        return self.wire_frame().mask

    @property
    def channel_by_name(self) -> dict[str, ChannelRecord]:
        return {item.channel: item for item in self.channels}

    def wire_frame(self) -> WireFrame:
        return WireFrame(
            seq=self.seq,
            timestamp_us=self.timestamp_us,
            protocol_version=self.protocol_version,
            channels=tuple(item.wire_channel() for item in self.channels),
            flags=self.flags,
        )

    def encode(self) -> bytes:
        return encode_frame(self.wire_frame())

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.encode())

    @property
    def frame_key(self) -> str:
        return f"frame:{self.seq:06d}"

    @property
    def version_stamp(self) -> str:
        return f"v{self.protocol_version}:{self.content_hash[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_us": self.timestamp_us,
            "protocol_version": self.protocol_version,
            "flags": self.flags,
            "channels": [item.to_dict() for item in self.channels],
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameRecord":
        frame = cls(
            seq=int(value["seq"]),
            timestamp_us=int(value["timestamp_us"]),
            protocol_version=int(value["protocol_version"]),
            flags=int(value.get("flags", 0)),
            channels=tuple(ChannelRecord.from_dict(item) for item in value["channels"]),
        )
        expected_hash = value.get("content_hash")
        if expected_hash is not None and expected_hash != frame.content_hash:
            raise ValueError(
                f"frame {frame.seq} content hash mismatch during restore: "
                f"stored={expected_hash} actual={frame.content_hash}"
            )
        return frame


@dataclass(frozen=True)
class BatchDigest:
    batch_key: str
    first_seq: int
    last_seq: int
    protocol_version: int
    frame_count: int
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_key": self.batch_key,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "protocol_version": self.protocol_version,
            "frame_count": self.frame_count,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BatchDigest":
        return cls(
            batch_key=str(value["batch_key"]),
            first_seq=int(value["first_seq"]),
            last_seq=int(value["last_seq"]),
            protocol_version=int(value["protocol_version"]),
            frame_count=int(value["frame_count"]),
            digest=str(value["digest"]),
        )


def batch_key_for_range(first_seq: int, last_seq: int) -> str:
    return f"batch:{first_seq:06d}-{last_seq:06d}"


def digest_encoded_frames(
    frames: Sequence[tuple[int, bytes]],
    protocol_version: int,
) -> str:
    """Digest over the on-wire bytes, canonical sequence, and protocol stamp."""
    ordered = sorted(frames, key=lambda item: item[0])
    digest_input = bytearray(b"HF-BATCH\x02")
    digest_input.extend(struct.pack(">BII", protocol_version & 0xFF, len(ordered), ordered[0][0] if ordered else 0))
    for seq, data in ordered:
        digest_input.extend(struct.pack(">II", seq, len(data)))
        digest_input.extend(data)
    return sha256_bytes(bytes(digest_input))


def make_batch_digest(
    frames: Sequence[FrameRecord],
    batch_key: str,
) -> BatchDigest:
    if not frames:
        raise ValueError("cannot digest an empty batch")
    protocols = {frame.protocol_version for frame in frames}
    if len(protocols) != 1:
        raise ValueError("mixed protocol versions in a batch")
    version = next(iter(protocols))
    encoded = [(frame.seq, frame.encode()) for frame in frames]
    return BatchDigest(
        batch_key=batch_key,
        first_seq=frames[0].seq,
        last_seq=frames[-1].seq,
        protocol_version=version,
        frame_count=len(frames),
        digest=digest_encoded_frames(encoded, version),
    )


def records_by_seq(frames: Iterable[FrameRecord]) -> dict[int, FrameRecord]:
    result: dict[int, FrameRecord] = {}
    for frame in frames:
        if frame.seq in result:
            raise ValueError(f"duplicate frame sequence {frame.seq}")
        result[frame.seq] = frame
    return result


def channel_codes(channels: Iterable[ChannelRecord], protocol_version: int) -> dict[str, tuple[int, int | None]]:
    return {
        item.channel: (
            item.amplitude_code,
            item.frequency_code if protocol_version == PROTOCOL_V2 else None,
        )
        for item in channels
    }

