"""Custom compact binary per-packet observation codec."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable

from .errors import format_error, wire_error
from .model import (
    LEGAL_FLAG_COMBINATIONS,
    KNOWN_FLAGS,
    FiveTuple,
    FrameBundle,
    FrameObservation,
    TimeBasis,
)
from .util import content_digest, sha256_hex


MAGIC = b"PKO1"
HEADER_FORMAT = "<4sHIIQIIHHBBIH"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_PAYLOAD = 65535


def _validate_flags(flags: int, *, byte_offset: int, frame_index: int) -> None:
    if flags & ~KNOWN_FLAGS:
        raise wire_error(
            "frame contains unknown flag bits",
            byte_offset=byte_offset,
            frame_index=frame_index,
            flags=flags,
        )
    if flags not in LEGAL_FLAG_COMBINATIONS:
        raise wire_error(
            "frame flag combination is not in the legal set",
            byte_offset=byte_offset,
            frame_index=frame_index,
            flags=flags,
            legal_flags=sorted(LEGAL_FLAG_COMBINATIONS),
        )


def encode_frame(
    *,
    five_tuple: FiveTuple,
    seq: int,
    send_ms: int,
    flags: int,
    arrival_ordinal: int,
    payload: bytes = b"",
    payload_len: int | None = None,
    wire_flow_id: int | None = None,
) -> bytes:
    """Encode one observation. This is deliberately strict like the decoder."""
    if not 0 <= seq <= 0xFFFFFFFF:
        raise ValueError("seq outside uint32")
    if not 0 <= send_ms <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("send_ms outside uint64")
    if not 0 <= arrival_ordinal <= 0xFFFFFFFF:
        raise ValueError("arrival_ordinal outside uint32")
    _validate_flags(flags, byte_offset=0, frame_index=1)
    if payload_len is None:
        payload_len = len(payload)
    if payload_len != len(payload):
        raise ValueError("payload_len does not match payload bytes")
    if not 0 <= payload_len <= MAX_PAYLOAD:
        raise ValueError("payload_len outside uint16")
    header = struct.pack(
        HEADER_FORMAT,
        MAGIC,
        HEADER_SIZE + payload_len,
        five_tuple.wire_flow_id if wire_flow_id is None else wire_flow_id,
        seq,
        send_ms,
        int.from_bytes(bytes(map(int, five_tuple.src_ip.split("."))), "big"),
        int.from_bytes(bytes(map(int, five_tuple.dst_ip.split("."))), "big"),
        five_tuple.src_port,
        five_tuple.dst_port,
        five_tuple.protocol_number,
        flags,
        arrival_ordinal,
        payload_len,
    )
    return header + payload


def _ipv4_text(value: int) -> str:
    return ".".join(str((value >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def decode_wire(data: bytes, *, source_id: str) -> list[FrameObservation]:
    frames: list[FrameObservation] = []
    offset = 0
    frame_index = 0
    while offset < len(data):
        frame_index += 1
        remaining = len(data) - offset
        if remaining < HEADER_SIZE:
            raise wire_error(
                "truncated frame header",
                byte_offset=offset,
                frame_index=frame_index,
                remaining_bytes=remaining,
                required_bytes=HEADER_SIZE,
            )
        fields = struct.unpack_from(HEADER_FORMAT, data, offset)
        (
            magic,
            frame_len,
            wire_flow_id,
            seq,
            send_ms,
            src_ip_num,
            dst_ip_num,
            src_port,
            dst_port,
            protocol_num,
            flags,
            arrival_ordinal,
            payload_len,
        ) = fields
        if magic != MAGIC:
            raise wire_error(
                "frame magic mismatch",
                byte_offset=offset,
                frame_index=frame_index,
                magic=magic.hex(),
            )
        if frame_len < HEADER_SIZE:
            raise wire_error(
                "declared frame length is smaller than the header",
                byte_offset=offset,
                frame_index=frame_index,
                declared_frame_len=frame_len,
                header_len=HEADER_SIZE,
            )
        if frame_len > remaining:
            raise wire_error(
                "truncated frame body",
                byte_offset=offset,
                frame_index=frame_index,
                declared_frame_len=frame_len,
                remaining_bytes=remaining,
            )
        if frame_len != HEADER_SIZE + payload_len:
            raise wire_error(
                "declared frame length and payload length disagree",
                byte_offset=offset,
                frame_index=frame_index,
                declared_frame_len=frame_len,
                payload_len=payload_len,
                expected_frame_len=HEADER_SIZE + payload_len,
            )
        _validate_flags(flags, byte_offset=offset, frame_index=frame_index)
        if protocol_num not in (6, 17):
            raise wire_error(
                "unsupported protocol number",
                byte_offset=offset,
                frame_index=frame_index,
                protocol=protocol_num,
            )
        try:
            five_tuple = FiveTuple(
                src_ip=_ipv4_text(src_ip_num),
                dst_ip=_ipv4_text(dst_ip_num),
                src_port=src_port,
                dst_port=dst_port,
                protocol="tcp" if protocol_num == 6 else "udp",
            )
        except ValueError as exc:
            raise wire_error(
                "invalid five tuple in frame",
                byte_offset=offset,
                frame_index=frame_index,
                reason=str(exc),
            ) from exc
        if wire_flow_id != five_tuple.wire_flow_id:
            raise wire_error(
                "frame flow identifier does not match its five tuple",
                byte_offset=offset,
                frame_index=frame_index,
                wire_flow_id=wire_flow_id,
                expected_wire_flow_id=five_tuple.wire_flow_id,
                flow_id=five_tuple.flow_id,
            )
        frames.append(
            FrameObservation(
                frame_index=frame_index,
                byte_offset=offset,
                five_tuple=five_tuple,
                seq=seq,
                send_ms=send_ms,
                flags=flags,
                arrival_ordinal=arrival_ordinal,
                payload_len=payload_len,
                source_id=source_id,
            )
        )
        offset += frame_len
    return frames


def frame_content(frames: Iterable[FrameObservation]) -> list[dict[str, Any]]:
    """Traversal-independent frame content for a wire digest."""
    result = [
        {
            "flow_id": frame.flow_id,
            "seq": frame.seq,
            "send_ms": frame.send_ms,
            "flags": frame.flags,
            "arrival_ordinal": frame.arrival_ordinal,
            "payload_len": frame.payload_len,
            "source_id": frame.source_id,
        }
        for frame in frames
    ]
    result.sort(
        key=lambda item: (
            item["flow_id"],
            item["seq"],
            item["arrival_ordinal"],
            item["send_ms"],
            item["flags"],
            item["payload_len"],
        )
    )
    return result


def load_wire_manifest(path: str | Path) -> dict[str, Any]:
    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise format_error(
            f"cannot read wire manifest: {exc}",
            record_number=0,
            field="wire_manifest",
        ) from exc
    if not isinstance(parsed, dict):
        raise format_error(
            "wire manifest root must be an object",
            record_number=0,
            field="wire_manifest",
        )
    for field_name in ("source_id", "model_version", "time_basis", "alignment_version"):
        if field_name not in parsed:
            raise format_error(
                "wire manifest is missing a required field",
                record_number=0,
                field=f"wire_manifest.{field_name}",
            )
    try:
        basis = TimeBasis.from_dict(parsed["time_basis"], field_name="wire_manifest.time_basis")
    except ValueError as exc:
        raise format_error(
            "wire manifest time basis is invalid",
            record_number=0,
            field=str(exc),
        ) from exc
    if basis.clock != "sender_monotonic_ms" or basis.unit != "ms":
        raise format_error(
            "wire time basis must explicitly be sender_monotonic_ms in ms",
            record_number=0,
            field="wire_manifest.time_basis",
            actual=basis.to_dict(),
        )
    if not isinstance(parsed["alignment_version"], str) or not parsed["alignment_version"]:
        raise format_error(
            "wire alignment_version must be a non-empty string",
            record_number=0,
            field="wire_manifest.alignment_version",
        )
    return parsed


def read_wire_bundle(
    wire_path: str | Path,
    manifest_path: str | Path,
    *,
    corrupt: bool = False,
) -> FrameBundle:
    manifest = load_wire_manifest(manifest_path)
    try:
        data = Path(wire_path).read_bytes()
    except OSError as exc:
        raise wire_error(
            f"cannot read wire file: {exc}",
            byte_offset=0,
            frame_index=0,
        ) from exc
    raw_digest = sha256_hex(data)
    if corrupt:
        data = data[:-1]
    frames = decode_wire(data, source_id=str(manifest["source_id"]))
    try:
        basis = TimeBasis.from_dict(manifest["time_basis"], field_name="wire_manifest.time_basis")
    except ValueError as exc:
        raise format_error(
            "wire manifest time basis is invalid",
            record_number=0,
            field=str(exc),
        ) from exc
    return FrameBundle(
        source_id=str(manifest["source_id"]),
        model_version=str(manifest["model_version"]),
        time_basis=basis,
        alignment_version=str(manifest["alignment_version"]),
        frames=frames,
        raw_digest=raw_digest,
        wire_digest=content_digest(frame_content(frames)),
    )
