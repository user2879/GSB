"""Parsers and serializers for v1 text, v2 JSON and v3 binary partitions."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from .compat import fixed_to_micro, micro_to_fixed
from .config import DIMENSIONS, FILE_ID_PATTERN
from .errors import (
    E_EVAL_FORMAT,
    E_LEVEL_UNMAPPED,
    Failure,
    FormatError,
    LevelMapError,
)
from .levels import LevelMap
from .models import FrameSummary, UnifiedResult, sorted_degradations
from .util import (
    MICRO_SCALE,
    canonical_json_bytes,
    decimal_to_micro,
    micro_to_float,
    pretty_json_bytes,
    round_half_up_fraction,
    sha256_bytes,
)


V1_FILE_FIELD = 24
V1_SCORE_FIELD = 4
V1_LEVEL_FIELD = 2
V1_LINE_WIDTH = V1_FILE_FIELD + 5 * V1_SCORE_FIELD + V1_LEVEL_FIELD
V1_FIELDS = ("total",) + DIMENSIONS

V3_MAGIC = b"AQV3"
V3_FORMAT_VERSION = 3
V3_HEADER_SIZE = 64
V3_HEADER = struct.Struct("<4sHHIIIIQQQQII")
V3_RECORD = struct.Struct("<24s9I4s20s8s")
V3_FRAME = struct.Struct("<IIihHHHHI")


@dataclass
class ParseOutcome:
    version: int
    records: list[UnifiedResult] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "records": [record.to_dict() for record in self.records],
            "failures": [failure.to_dict() for failure in sorted(self.failures, key=Failure.sort_key)],
        }


def _failure(
    message: str,
    *,
    version: int,
    location: str,
    file_id: str = "",
    code: str = E_EVAL_FORMAT,
    detail: dict[str, Any] | None = None,
) -> Failure:
    return Failure(
        code=code,
        message=message,
        file_id=file_id,
        version=version,
        stage="parse",
        location=location,
        task_key=f"{file_id or '?'}|parse|v{version}",
        detail=detail or {},
    )


def parse_v1(data: bytes, location: str, level_map: LevelMap) -> ParseOutcome:
    outcome = ParseOutcome(version=1)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        outcome.failures.append(
            _failure(f"v1 text is not UTF-8: {exc}", version=1, location=location)
        )
        return outcome
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        line_location = f"{location}:{line_number}"
        if len(line) != V1_LINE_WIDTH:
            outcome.failures.append(
                _failure(
                    f"v1 fixed-width line must be {V1_LINE_WIDTH} characters, got {len(line)}",
                    version=1,
                    location=line_location,
                    detail={"line": line, "expected_width": V1_LINE_WIDTH, "actual_width": len(line)},
                )
            )
            continue
        file_id = line[0:V1_FILE_FIELD].rstrip()
        if not FILE_ID_PATTERN.fullmatch(file_id):
            outcome.failures.append(
                _failure(
                    "invalid v1 file identifier",
                    version=1,
                    location=line_location,
                    file_id=file_id,
                )
            )
            continue
        score_fields: dict[str, int] = {}
        valid = True
        for index, field_name in enumerate(V1_FIELDS):
            start = V1_FILE_FIELD + index * V1_SCORE_FIELD
            raw = line[start : start + V1_SCORE_FIELD]
            if not raw.isdigit() or not 0 <= int(raw) <= 1000:
                outcome.failures.append(
                    _failure(
                        f"invalid v1 {field_name} score field",
                        version=1,
                        location=line_location,
                        file_id=file_id,
                        detail={"field": field_name, "raw": raw},
                    )
                )
                valid = False
                break
            score_fields[field_name] = int(raw)
        if not valid:
            continue
        code_raw = line[V1_FILE_FIELD + 5 * V1_SCORE_FIELD : V1_LINE_WIDTH]
        code = code_raw.strip()
        if not code:
            outcome.failures.append(
                _failure(
                    "v1 level code is empty",
                    version=1,
                    location=line_location,
                    file_id=file_id,
                )
            )
            continue
        if file_id in seen:
            outcome.failures.append(
                _failure(
                    "duplicate file identifier in v1 exchange",
                    version=1,
                    location=line_location,
                    file_id=file_id,
                )
            )
            continue
        seen.add(file_id)
        entry = level_map.optional_by_code(code)
        if entry is None:
            outcome.failures.append(
                _failure(
                    f"level code {code!r} is not present in the level map",
                    version=1,
                    location=line_location,
                    file_id=file_id,
                    code=E_LEVEL_UNMAPPED,
                )
            )
            continue
        outcome.records.append(
            UnifiedResult(
                file_id=file_id,
                source_version=1,
                total_micro=score_fields["total"] * 1000,
                dimensions_micro={name: score_fields[name] * 1000 for name in DIMENSIONS},
                level_code=code,
                level_name=entry.name,
                precision_tag="v1_permille",
                bypass={"source_version": 1, "precision_tag": "v1_permille"},
                location=line_location,
            )
        )
    outcome.records.sort(key=lambda record: record.file_id)
    return outcome


def parse_v2(data: bytes, location: str, level_map: LevelMap) -> ParseOutcome:
    outcome = ParseOutcome(version=2)
    try:
        raw = json.loads(data.decode("utf-8"), parse_float=Decimal)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        outcome.failures.append(
            _failure(f"invalid v2 JSON: {exc}", version=2, location=location)
        )
        return outcome
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        outcome.failures.append(
            _failure("v2 document must contain a results array", version=2, location=location)
        )
        return outcome
    top_precision = raw.get("precision_tag")
    seen: set[str] = set()
    for index, item in enumerate(raw["results"]):
        item_location = f"{location}#results[{index}]"
        if not isinstance(item, dict):
            outcome.failures.append(
                _failure("v2 result must be an object", version=2, location=item_location)
            )
            continue
        file_id = str(item.get("file_id", ""))
        if not FILE_ID_PATTERN.fullmatch(file_id):
            outcome.failures.append(
                _failure(
                    "invalid v2 file identifier",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                )
            )
            continue
        if file_id in seen:
            outcome.failures.append(
                _failure(
                    "duplicate file identifier in v2 exchange",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                )
            )
            continue
        total_raw = item.get("total_score")
        dims_raw = item.get("dimensions")
        if isinstance(total_raw, bool) or not isinstance(total_raw, (int, float, Decimal)):
            outcome.failures.append(
                _failure(
                    "v2 total_score must be numeric",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                )
            )
            continue
        if not isinstance(dims_raw, dict) or set(dims_raw) != set(DIMENSIONS):
            outcome.failures.append(
                _failure(
                    "v2 dimensions must contain exactly the four score dimensions",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                )
            )
            continue
        try:
            total_decimal = Decimal(str(total_raw))
            dims_decimal = {name: Decimal(str(dims_raw[name])) for name in DIMENSIONS}
            if not Decimal(0) <= total_decimal <= Decimal(1):
                raise ValueError("total_score outside 0..1")
            for name, value in dims_decimal.items():
                if not Decimal(0) <= value <= Decimal(1):
                    raise ValueError(f"dimension {name} outside 0..1")
            total_micro = decimal_to_micro(total_decimal)
            dims_micro = {name: decimal_to_micro(value) for name, value in dims_decimal.items()}
        except (ValueError, ArithmeticError) as exc:
            outcome.failures.append(
                _failure(
                    f"invalid v2 numeric score: {exc}",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                )
            )
            continue
        level_name = str(item.get("level", item.get("level_name", "")))
        entry = level_map.optional_by_name(level_name)
        if entry is None:
            outcome.failures.append(
                _failure(
                    f"level name {level_name!r} is not present in the level map",
                    version=2,
                    location=item_location,
                    file_id=file_id,
                    code=E_LEVEL_UNMAPPED,
                )
            )
            continue
        seen.add(file_id)
        precision = item.get("precision_tag", top_precision)
        outcome.records.append(
            UnifiedResult(
                file_id=file_id,
                source_version=2,
                total_micro=total_micro,
                dimensions_micro=dims_micro,
                level_code=entry.code,
                level_name=entry.name,
                precision_tag=str(precision) if precision is not None else None,
                bypass=dict(item.get("bypass", {})),
                location=item_location,
            )
        )
    outcome.records.sort(key=lambda record: record.file_id)
    return outcome


def parse_v3(data: bytes, location: str, level_map: LevelMap) -> ParseOutcome:
    outcome = ParseOutcome(version=3)
    if len(data) < V3_HEADER_SIZE:
        outcome.failures.append(
            _failure("v3 file is shorter than its 64-byte header", version=3, location=location)
        )
        return outcome
    try:
        (
            magic,
            version,
            header_size,
            record_count,
            frame_count,
            fixed_bits,
            _flags,
            index_offset,
            frame_offset,
            metadata_offset,
            metadata_size,
            checksum,
            _reserved,
        ) = V3_HEADER.unpack_from(data, 0)
    except struct.error as exc:
        outcome.failures.append(
            _failure(f"cannot decode v3 header: {exc}", version=3, location=location)
        )
        return outcome
    if magic != V3_MAGIC or version != V3_FORMAT_VERSION or header_size != V3_HEADER_SIZE:
        outcome.failures.append(
            _failure(
                "v3 magic/version/header-size mismatch",
                version=3,
                location=location,
                detail={"magic": magic.decode("latin1"), "version": version, "header_size": header_size},
            )
        )
        return outcome
    if not 0 <= fixed_bits <= 6:
        outcome.failures.append(
            _failure("v3 fixed_decimal_bits outside 0..6", version=3, location=location)
        )
        return outcome
    expected_index = V3_HEADER_SIZE
    expected_frame = index_offset + record_count * V3_RECORD.size
    expected_metadata = frame_offset + frame_count * V3_FRAME.size
    if (
        index_offset != expected_index
        or frame_offset != expected_frame
        or metadata_offset != expected_metadata
        or metadata_offset + metadata_size != len(data)
    ):
        outcome.failures.append(
            _failure(
                "v3 partition offsets are inconsistent with section sizes",
                version=3,
                location=location,
                detail={
                    "index_offset": index_offset,
                    "frame_offset": frame_offset,
                    "metadata_offset": metadata_offset,
                    "metadata_size": metadata_size,
                    "file_size": len(data),
                },
            )
        )
        return outcome
    body = data[V3_HEADER_SIZE:]
    actual_checksum = int.from_bytes(bytes.fromhex(sha256_bytes(body))[:4], "big")
    if actual_checksum != checksum:
        outcome.failures.append(
            _failure("v3 body checksum mismatch", version=3, location=location)
        )
        return outcome
    all_frames: list[tuple[int, FrameSummary]] = []
    for index in range(frame_count):
        offset = frame_offset + index * V3_FRAME.size
        file_idx, frame_idx, rms_centi, zcr_milli, peak_q, zero_run, zero_ratio, clipped, _ = (
            V3_FRAME.unpack_from(data, offset)
        )
        if file_idx >= record_count or frame_idx > 10_000_000:
            outcome.failures.append(
                _failure(
                    "v3 frame summary references an invalid file/frame index",
                    version=3,
                    location=f"{location}#frame[{index}]",
                )
            )
            continue
        all_frames.append(
            (
                file_idx,
                FrameSummary(
                frame_index=frame_idx,
                rms_db=rms_centi / 100.0,
                zcr=zcr_milli / 1000.0,
                peak=peak_q / 32767.0,
                max_zero_run=zero_run,
                zero_run_ratio=zero_ratio / 1000.0,
                clipped_samples=clipped,
                frame_fingerprint="",
                ),
            )
        )
    seen: set[str] = set()
    for index in range(record_count):
        offset = index_offset + index * V3_RECORD.size
        (
            file_id_raw,
            total_fixed,
            level_fixed,
            noise_fixed,
            clipping_fixed,
            continuity_fixed,
            frame_start,
            record_frames,
            meta_offset,
            meta_len,
            code_raw,
            precision_raw,
            _reserved_record,
        ) = V3_RECORD.unpack_from(data, offset)
        try:
            file_id = file_id_raw.rstrip(b"\x00").decode("utf-8")
        except UnicodeDecodeError:
            file_id = ""
        record_location = f"{location}#record[{index}]"
        if not FILE_ID_PATTERN.fullmatch(file_id):
            outcome.failures.append(
                _failure("invalid v3 file identifier", version=3, location=record_location, file_id=file_id)
            )
            continue
        if file_id in seen:
            outcome.failures.append(
                _failure(
                    "duplicate file identifier in v3 exchange",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                )
            )
            continue
        if frame_start + record_frames > len(all_frames):
            outcome.failures.append(
                _failure(
                    "v3 record frame range exceeds its frame partition",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                )
            )
            continue
        record_frames_slice = all_frames[frame_start : frame_start + record_frames]
        if any(frame_file_index != index for frame_file_index, _ in record_frames_slice):
            outcome.failures.append(
                _failure(
                    "v3 record frame range includes frames owned by another file",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                )
            )
            continue
        if meta_offset + meta_len > metadata_size:
            outcome.failures.append(
                _failure(
                    "v3 bypass metadata range exceeds metadata partition",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                )
            )
            continue
        metadata: dict[str, Any] = {}
        if meta_len:
            raw_meta = data[metadata_offset + meta_offset : metadata_offset + meta_offset + meta_len]
            try:
                decoded = json.loads(raw_meta.decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("metadata is not an object")
                metadata = decoded
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                outcome.failures.append(
                    _failure(
                        f"invalid v3 bypass metadata: {exc}",
                        version=3,
                        location=record_location,
                        file_id=file_id,
                    )
                )
                continue
        code = code_raw.rstrip(b"\x00").decode("ascii", errors="replace")
        entry = level_map.optional_by_code(code)
        if entry is None:
            outcome.failures.append(
                _failure(
                    f"level code {code!r} is not present in the level map",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                    code=E_LEVEL_UNMAPPED,
                )
            )
            continue
        try:
            dims_fixed = (level_fixed, noise_fixed, clipping_fixed, continuity_fixed)
            total_micro = fixed_to_micro(total_fixed, fixed_bits)
            dims_micro = {
                name: fixed_to_micro(value, fixed_bits)
                for name, value in zip(DIMENSIONS, dims_fixed)
            }
            if total_micro > MICRO_SCALE or any(value > MICRO_SCALE for value in dims_micro.values()):
                raise ValueError("fixed score exceeds full scale")
        except ValueError as exc:
            outcome.failures.append(
                _failure(
                    f"invalid v3 fixed score: {exc}",
                    version=3,
                    location=record_location,
                    file_id=file_id,
                )
            )
            continue
        seen.add(file_id)
        record_precision = precision_raw.rstrip(b"\x00").decode("ascii", errors="replace")
        precision = str(
            metadata.get(
                "precision_tag",
                record_precision or f"v3_fixed_{fixed_bits}",
            )
        )
        level_name = str(metadata.get("level_name", entry.name))
        frames = [frame for _, frame in record_frames_slice]
        outcome.records.append(
            UnifiedResult(
                file_id=file_id,
                source_version=3,
                total_micro=total_micro,
                dimensions_micro=dims_micro,
                level_code=code,
                level_name=level_name,
                precision_tag=precision,
                frame_summaries=frames,
                bypass=dict(metadata.get("bypass", {})),
                location=record_location,
                source_fixed_bits=fixed_bits,
            )
        )
    outcome.records.sort(key=lambda record: record.file_id)
    return outcome


def read_exchange(version: int, path: str | Path, level_map: LevelMap) -> ParseOutcome:
    location = str(path)
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return ParseOutcome(
            version=version,
            failures=[_failure(f"cannot read exchange file: {exc}", version=version, location=location)],
        )
    if version == 1:
        return parse_v1(data, location, level_map)
    if version == 2:
        return parse_v2(data, location, level_map)
    if version == 3:
        return parse_v3(data, location, level_map)
    return ParseOutcome(
        version=version,
        failures=[_failure(f"unsupported exchange version: {version}", version=version, location=location)],
    )


def write_v1_bytes(records: list[UnifiedResult]) -> bytes:
    lines: list[str] = []
    for record in sorted(records, key=lambda item: item.file_id):
        if record.level_code is None:
            raise FormatError(f"{record.file_id}: v1 requires a level code")
        line = (
            f"{record.file_id:<{V1_FILE_FIELD}}"
            f"{record.total_permille:0{V1_SCORE_FIELD}d}"
            + "".join(
                f"{record.dimension_permille(name):0{V1_SCORE_FIELD}d}" for name in DIMENSIONS
            )
            + f"{record.level_code:<{V1_LEVEL_FIELD}}"
        )
        if len(line) != V1_LINE_WIDTH:
            raise FormatError(f"{record.file_id}: generated v1 line has width {len(line)}")
        lines.append(line)
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_v2_bytes(records: list[UnifiedResult]) -> bytes:
    items: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.file_id):
        if record.level_name is None:
            raise FormatError(f"{record.file_id}: v2 requires a named level")
        item: dict[str, Any] = {
            "file_id": record.file_id,
            "total_score": micro_to_float(record.total_micro),
            "dimensions": {
                name: micro_to_float(record.dimensions_micro[name]) for name in DIMENSIONS
            },
            "level": record.level_name,
            "precision_tag": record.precision_tag or "float6",
        }
        if record.bypass:
            item["bypass"] = record.bypass
        items.append(item)
    document = {
        "schema": "aqeval.v2",
        "precision_tag": "float6",
        "results": items,
    }
    return pretty_json_bytes(document)


def write_v3_bytes(records: list[UnifiedResult], fixed_bits: int) -> bytes:
    ordered = sorted(records, key=lambda item: item.file_id)
    metadata_parts: list[bytes] = []
    metadata_ranges: list[tuple[int, int]] = []
    frame_values: list[tuple[int, int, int, int, int, int, int, int, int]] = []
    fixed_rows: list[tuple[Any, ...]] = []
    frame_cursor = 0
    for file_index, record in enumerate(ordered):
        if record.level_code is None:
            raise FormatError(f"{record.file_id}: v3 requires a level code")
        metadata = {
            "precision_tag": record.precision_tag or f"v3_fixed_{fixed_bits}",
            "level_name": record.level_name,
            "bypass": record.bypass,
        }
        meta_bytes = canonical_json_bytes(metadata)
        meta_offset = sum(len(part) for part in metadata_parts)
        metadata_parts.append(meta_bytes)
        metadata_ranges.append((meta_offset, len(meta_bytes)))
        for frame in sorted(record.frame_summaries, key=lambda item: item.frame_index):
            frame_values.append(
                (
                    file_index,
                    frame.frame_index,
                    int(round(frame.rms_db * 100)),
                    max(0, min(1000, int(round(frame.zcr * 1000)))),
                    max(0, min(32767, int(round(frame.peak * 32767)))),
                    max(0, min(65535, frame.max_zero_run)),
                    max(0, min(1000, int(round(frame.zero_run_ratio * 1000)))),
                    max(0, min(65535, frame.clipped_samples)),
                    0,
                )
            )
        fixed_rows.append(
            (
                record.file_id.encode("utf-8")[:24].ljust(24, b"\x00"),
                micro_to_fixed(record.total_micro, fixed_bits),
                *[micro_to_fixed(record.dimensions_micro[name], fixed_bits) for name in DIMENSIONS],
                frame_cursor,
                len(record.frame_summaries),
                meta_offset,
                len(meta_bytes),
                record.level_code.encode("ascii")[:4].ljust(4, b"\x00"),
                (record.precision_tag or f"v3_fixed_{fixed_bits}").encode("ascii", errors="replace")[:20].ljust(20, b"\x00"),
                b"\x00" * 8,
            )
        )
        frame_cursor += len(record.frame_summaries)
    index_offset = V3_HEADER_SIZE
    frame_offset = index_offset + len(fixed_rows) * V3_RECORD.size
    metadata_offset = frame_offset + len(frame_values) * V3_FRAME.size
    index_part = b"".join(V3_RECORD.pack(*row) for row in fixed_rows)
    frame_part = b"".join(V3_FRAME.pack(*row) for row in frame_values)
    metadata_part = b"".join(metadata_parts)
    body = index_part + frame_part + metadata_part
    checksum = int.from_bytes(bytes.fromhex(sha256_bytes(body))[:4], "big")
    header = V3_HEADER.pack(
        V3_MAGIC,
        V3_FORMAT_VERSION,
        V3_HEADER_SIZE,
        len(fixed_rows),
        len(frame_values),
        fixed_bits,
        0,
        index_offset,
        frame_offset,
        metadata_offset,
        len(metadata_part),
        checksum,
        0,
    )
    return header + body


def write_format_bytes(version: int, records: list[UnifiedResult], fixed_bits: int) -> bytes:
    if version == 1:
        return write_v1_bytes(records)
    if version == 2:
        return write_v2_bytes(records)
    if version == 3:
        return write_v3_bytes(records, fixed_bits)
    raise FormatError(f"unsupported exchange version: {version}")


def parse_bytes(version: int, data: bytes, location: str, level_map: LevelMap) -> ParseOutcome:
    if version == 1:
        return parse_v1(data, location, level_map)
    if version == 2:
        return parse_v2(data, location, level_map)
    if version == 3:
        return parse_v3(data, location, level_map)
    return ParseOutcome(
        version=version,
        failures=[_failure(f"unsupported exchange version: {version}", version=version, location=location)],
    )


def write_exchange(version: int, records: list[UnifiedResult], fixed_bits: int, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(write_format_bytes(version, records, fixed_bits))
