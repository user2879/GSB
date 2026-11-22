from __future__ import annotations

import struct
import zlib
import numpy as np

from .errors import DiagnosticError
from .model import Track


MAGIC = b"STV3"
HEADER = struct.Struct("<4sHHIIHHQI")
RECORD = struct.Struct("<16sHHIII")
HEADER_SIZE = HEADER.size
RECORD_SIZE = RECORD.size
NAME_BYTES = 16


def encode_v3(tracks: list[Track], sample_rate: int, frame_count: int | None = None) -> bytes:
    if not tracks:
        raise ValueError("v3 package needs at least one track")
    if frame_count is None:
        frame_count = tracks[0].frames
    seen: set[str] = set()
    encoded_names: list[bytes] = []
    for track in tracks:
        if track.name in seen:
            raise ValueError(f"duplicate track name {track.name!r}")
        seen.add(track.name)
        if track.frames != frame_count:
            raise ValueError("v3 tracks must share frame count")
        name_bytes = track.name.encode("utf-8")
        if not name_bytes or len(name_bytes) > NAME_BYTES or b"\x00" in name_bytes:
            raise ValueError(f"v3 track name {track.name!r} must fit in 16 non-NUL bytes")
        encoded_names.append(name_bytes.ljust(NAME_BYTES, b"\x00"))

    data_offset = HEADER_SIZE + RECORD_SIZE * len(tracks)
    records: list[bytes] = []
    blocks: list[bytes] = []
    offset = data_offset
    for channel, (track, name_bytes) in enumerate(zip(tracks, encoded_names, strict=True)):
        block = np.ascontiguousarray(track.pcm, dtype="<i2").tobytes(order="C")
        checksum = zlib.crc32(block) & 0xFFFFFFFF
        records.append(
            RECORD.pack(name_bytes, channel, 0, offset, track.frames, checksum)
        )
        blocks.append(block)
        offset += len(block)

    header = HEADER.pack(
        MAGIC,
        3,
        HEADER_SIZE,
        sample_rate,
        frame_count,
        len(tracks),
        0,
        data_offset,
        0,
    )
    return header + b"".join(records) + b"".join(blocks)


def decode_v3(
    data: bytes,
    source_path: str,
    clip_id: str = "",
) -> tuple[dict[str, Track], tuple[str, ...], list[str], int]:
    failures: list[DiagnosticError] = []
    try:
        if len(data) < HEADER_SIZE:
            raise ValueError("file shorter than fixed header")
        magic, version, header_size, sample_rate, frame_count, track_count, flags, data_offset, reserved = (
            HEADER.unpack_from(data, 0)
        )
        if magic != MAGIC:
            raise ValueError("bad v3 magic")
        if version != 3:
            raise ValueError(f"unsupported v3 version {version}")
        if header_size != HEADER_SIZE or header_size > len(data):
            raise ValueError("invalid v3 header size")
        if reserved != 0 or flags != 0:
            raise ValueError("unsupported v3 flags/reserved fields")
        if sample_rate <= 0 or frame_count < 0 or track_count <= 0:
            raise ValueError("invalid v3 header values")
        expected_offset = HEADER_SIZE + RECORD_SIZE * track_count
        if data_offset != expected_offset or data_offset > len(data):
            raise ValueError("invalid v3 data offset")
        records: list[tuple[bytes, int, int, int, int, int]] = []
        names: set[str] = set()
        for index in range(track_count):
            position = HEADER_SIZE + RECORD_SIZE * index
            name_raw, channel, record_reserved, offset, frames, checksum = RECORD.unpack_from(data, position)
            name = name_raw.rstrip(b"\x00").decode("utf-8", errors="strict")
            if not name or name in names:
                raise ValueError(f"invalid or duplicate track name at record {index}")
            names.add(name)
            if channel != index or record_reserved != 0:
                raise ValueError(f"invalid channel mapping for track {name}")
            if frames != frame_count:
                raise ValueError(f"track {name} frame count {frames} != package {frame_count}")
            if offset < data_offset or offset + frames * 2 > len(data):
                raise ValueError(f"track {name} PCM block is out of bounds")
            if offset % 2:
                raise ValueError(f"track {name} PCM block is not sample aligned")
            records.append((name_raw, channel, record_reserved, offset, frames, checksum))

        ranges = []
        for name_raw, _, _, offset, frames, _ in records:
            name = name_raw.rstrip(b"\x00").decode("utf-8")
            ranges.append((offset, offset + frames * 2, name))
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if left[1] > right[0]:
                raise ValueError(f"overlapping PCM blocks {left[2]!r} and {right[2]!r}")

        tracks: dict[str, Track] = {}
        for name_raw, channel, _, offset, frames, checksum in records:
            name = name_raw.rstrip(b"\x00").decode("utf-8")
            block = data[offset : offset + frames * 2]
            actual_checksum = zlib.crc32(block) & 0xFFFFFFFF
            if checksum != actual_checksum:
                failures.append(
                    DiagnosticError(
                        "E_TRACK_PARSE_FAILED",
                        f"checksum mismatch for v3 track {name}",
                        stage="parse",
                        format_version="v3",
                        clip_id=clip_id,
                        track=name,
                        details={"expected": checksum, "actual": actual_checksum},
                    )
                )
                continue
            pcm = np.frombuffer(block, dtype="<i2").copy()
            tracks[name] = Track(
                name=name,
                pcm=pcm,
                sample_rate=sample_rate,
                source_format="v3",
                metadata={"block_checksum_crc32": f"{checksum:08x}"},
            )
        order = tuple(
            name_raw.rstrip(b"\x00").decode("utf-8")
            for name_raw, _, _, _, _, _ in records
        )
        return tracks, order, failures, sample_rate
    except (ValueError, struct.error, UnicodeDecodeError) as exc:
        return {}, (), [
            DiagnosticError(
                "E_TRACK_PARSE_FAILED",
                f"cannot parse v3 package {source_path}: {exc}",
                stage="parse",
                format_version="v3",
                clip_id=clip_id,
                details={"path": source_path},
            )
        ], 0
