"""Codecs and cross-version normalization for c1/c2/c3 (requirement 2).

Formats
-------
c1: fixed-width text lines, only object key, shard index, shard digest.
c2: JSON lines, adding merge id, integer shard bytes, meta version.
c3: JSON lines, adding content check, parent merge reference, commit cursor.

Normalization never overwrites a real higher-version value; only fields absent
from the lower source format are backfilled, and every backfill is recorded
with its basis. All structural failures raise E_FORMAT with a line number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .errors import FormatError
from .models import (
    FORMAT_C1,
    FORMAT_C2,
    FORMAT_C3,
    FORMAT_NAMES,
    NormalizedRecord,
)

# Fixed-width layout for c1 (exact byte columns).
KEY_WIDTH = 24
INDEX_WIDTH = 6
DIGEST_WIDTH = 16
C1_LINE_WIDTH = KEY_WIDTH + INDEX_WIDTH + DIGEST_WIDTH  # 46

C2_FIELDS = ("object_key", "shard_index", "shard_digest",
             "merge_id", "shard_bytes", "meta_version")
C3_EXTRA_FIELDS = ("content_check", "parent_merge_id", "commit_cursor")


# --------------------------------------------------------------------------- #
# c1 fixed width
# --------------------------------------------------------------------------- #
def encode_c1_line(record: NormalizedRecord) -> str:
    key = record.object_key
    if len(key) > KEY_WIDTH:
        raise FormatError("object_key exceeds fixed width", key=key, field="object_key",
                          line=record.source_line)
    digest = record.shard_digest
    if len(digest) != DIGEST_WIDTH:
        raise FormatError("shard_digest must be 16 chars", key=key, field="shard_digest",
                          line=record.source_line)
    return f"{key:<{KEY_WIDTH}}{record.shard_index:0{INDEX_WIDTH}d}{digest}"


def decode_c1_line(line: str, line_number: int) -> NormalizedRecord:
    if len(line) != C1_LINE_WIDTH:
        raise FormatError(
            f"c1 line width {len(line)} != {C1_LINE_WIDTH}", line=line_number,
            field="fixed_width",
        )
    key = line[:KEY_WIDTH].rstrip(" ")
    index_text = line[KEY_WIDTH:KEY_WIDTH + INDEX_WIDTH]
    digest = line[KEY_WIDTH + INDEX_WIDTH:]
    if not key:
        raise FormatError("blank object_key", line=line_number, field="object_key")
    if not index_text.isdigit():
        raise FormatError("shard_index column must be digits", line=line_number,
                          field="shard_index")
    if len(digest) != DIGEST_WIDTH or any(c not in "0123456789abcdef" for c in digest):
        raise FormatError("shard_digest column must be 16 hex chars", line=line_number,
                          field="shard_digest")
    return NormalizedRecord(
        object_key=key,
        shard_index=int(index_text),
        shard_digest=digest,
        merge_id=None,
        shard_bytes=None,
        meta_version=None,
        content_check=None,
        parent_merge_id=None,
        commit_cursor=None,
        source_format=FORMAT_C1,
        source_line=line_number,
    )


# --------------------------------------------------------------------------- #
# c2 / c3 JSON
# --------------------------------------------------------------------------- #
def _require(mapping: dict, name: str, line_number: int) -> Any:
    if name not in mapping:
        raise FormatError(f"missing field {name}", line=line_number, field=name)
    return mapping[name]


def _as_int(value: Any, name: str, line_number: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FormatError(f"field {name} must be an integer", line=line_number, field=name)
    return int(value)


def _as_str(value: Any, name: str, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise FormatError(f"field {name} must be a non-empty string",
                          line=line_number, field=name)
    return value


def decode_json_line(line: str, line_number: int, fmt: int) -> NormalizedRecord:
    try:
        mapping = json.loads(line)
    except json.JSONDecodeError as exc:
        raise FormatError(f"invalid JSON: {exc.msg}", line=line_number,
                          field="json") from exc
    if not isinstance(mapping, dict):
        raise FormatError("metadata line must be a JSON object", line=line_number)

    key = _as_str(_require(mapping, "object_key", line_number), "object_key", line_number)
    shard_index = _as_int(_require(mapping, "shard_index", line_number),
                          "shard_index", line_number)
    digest = _as_str(_require(mapping, "shard_digest", line_number),
                     "shard_digest", line_number)
    if len(digest) != DIGEST_WIDTH:
        raise FormatError("shard_digest must be 16 chars", line=line_number,
                          field="shard_digest", key=key)

    merge_id = shard_bytes = meta_version = None
    content_check = parent_merge_id = commit_cursor = None

    if fmt == FORMAT_C2:
        merge_id = _as_str(_require(mapping, "merge_id", line_number),
                           "merge_id", line_number)
        shard_bytes = _as_int(_require(mapping, "shard_bytes", line_number),
                              "shard_bytes", line_number)
        meta_version = _as_int(_require(mapping, "meta_version", line_number),
                               "meta_version", line_number)
    elif fmt == FORMAT_C3:
        merge_id = _as_str(_require(mapping, "merge_id", line_number),
                           "merge_id", line_number)
        shard_bytes = _as_int(_require(mapping, "shard_bytes", line_number),
                              "shard_bytes", line_number)
        meta_version = _as_int(_require(mapping, "meta_version", line_number),
                               "meta_version", line_number)
        content_check = _as_str(_require(mapping, "content_check", line_number),
                                "content_check", line_number)
        parent_value = mapping.get("parent_merge_id")
        if parent_value is not None and not isinstance(parent_value, str):
            raise FormatError("parent_merge_id must be a string or null",
                              line=line_number, field="parent_merge_id")
        parent_merge_id = parent_value
        commit_cursor = _as_int(_require(mapping, "commit_cursor", line_number),
                                "commit_cursor", line_number)
    else:  # pragma: no cover - guarded by public decoders
        raise FormatError(f"unknown format {fmt}", line=line_number)

    return NormalizedRecord(
        object_key=key,
        shard_index=shard_index,
        shard_digest=digest,
        merge_id=merge_id,
        shard_bytes=shard_bytes,
        meta_version=meta_version,
        content_check=content_check,
        parent_merge_id=parent_merge_id,
        commit_cursor=commit_cursor,
        source_format=fmt,
        source_line=line_number,
    )


def encode_json_line(record: NormalizedRecord, fmt: int) -> str:
    payload: dict[str, Any] = {
        "object_key": record.object_key,
        "shard_index": record.shard_index,
        "shard_digest": record.shard_digest,
    }
    if fmt >= FORMAT_C2:
        payload.update({
            "merge_id": record.merge_id,
            "shard_bytes": record.shard_bytes,
            "meta_version": record.meta_version,
        })
    if fmt >= FORMAT_C3:
        payload.update({
            "content_check": record.content_check,
            "parent_merge_id": record.parent_merge_id,
            "commit_cursor": record.commit_cursor,
        })
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# File decoding + cross-version normalization
# --------------------------------------------------------------------------- #
@dataclass
class DecodedFile:
    fmt: int
    source_name: str
    records: list[NormalizedRecord]


def detect_format(path: str) -> int:
    lowered = path.lower()
    for fmt, name in FORMAT_NAMES.items():
        if f".{name}." in lowered or lowered.endswith(f".{name}") or f".{name}-" in lowered:
            return fmt
    raise FormatError(f"cannot detect c1/c2/c3 format from path {path}", field="path")


def decode_file(path: str, fmt: Optional[int] = None) -> DecodedFile:
    if fmt is None:
        fmt = detect_format(path)
    records: list[NormalizedRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if fmt == FORMAT_C1:
                records.append(decode_c1_line(line, line_number))
            else:
                records.append(decode_json_line(line, line_number, fmt))
    return DecodedFile(fmt=fmt, source_name=path, records=records)


def _backfill_record(record: NormalizedRecord, line_number: int) -> NormalizedRecord:
    """Backfill fields absent in lower source formats, recording the basis."""
    filled: set[str] = set()
    basis: dict[str, str] = {}
    merge_id = record.merge_id
    shard_bytes = record.shard_bytes
    meta_version = record.meta_version
    content_check = record.content_check
    parent_merge_id = record.parent_merge_id
    commit_cursor = record.commit_cursor

    if merge_id is None:
        merge_id = f"F{record.source_format}-{record.source_line:06d}-{record.shard_digest[:8]}"
        filled.add("merge_id")
        basis["merge_id"] = "synthetic:line+digest"
    if shard_bytes is None:
        shard_bytes = 0
        filled.add("shard_bytes")
        basis["shard_bytes"] = "default-zero:absent-in-c1"
    if meta_version is None:
        meta_version = 0
        filled.add("meta_version")
        basis["meta_version"] = "default-zero:absent-in-c1"
    if content_check is None:
        content_check = ""
        filled.add("content_check")
        basis["content_check"] = "empty:absent-below-c3"
    if parent_merge_id is None and record.source_format < FORMAT_C3:
        parent_merge_id = None
        filled.add("parent_merge_id")
        basis["parent_merge_id"] = "root:no-parent-declared"
    if commit_cursor is None:
        commit_cursor = 0
        filled.add("commit_cursor")
        basis["commit_cursor"] = "default-zero:absent-below-c3"

    return NormalizedRecord(
        object_key=record.object_key,
        shard_index=record.shard_index,
        shard_digest=record.shard_digest,
        merge_id=merge_id,
        shard_bytes=shard_bytes,
        meta_version=meta_version,
        content_check=content_check,
        parent_merge_id=parent_merge_id,
        commit_cursor=commit_cursor,
        source_format=record.source_format,
        source_line=line_number,
        filled_fields=frozenset(filled),
        fill_basis=basis,
        merge_seq=record.merge_seq,
    )


def normalize_files(decoded: Iterable[DecodedFile]) -> list[NormalizedRecord]:
    """Decode/backfill and run cross-record structural validation.

    Checks: shard index/key consistency, merge id uniqueness, parent reference
    existence and acyclicity. Returns records ordered deterministically.
    """
    normalized: list[NormalizedRecord] = []
    merge_index: dict[str, NormalizedRecord] = {}
    by_key_digest: dict[tuple[str, int], NormalizedRecord] = {}

    for decoded_file in decoded:
        for record in decoded_file.records:
            filled_record = _backfill_record(record, record.source_line)
            assert filled_record.merge_id is not None
            if filled_record.merge_id in merge_index:
                raise FormatError(
                    f"duplicate merge_id {filled_record.merge_id}",
                    line=filled_record.source_line, field="merge_id",
                    key=filled_record.object_key,
                )
            identity = (filled_record.object_key, filled_record.shard_index)
            if identity in by_key_digest:
                previous = by_key_digest[identity]
                if previous is not None and previous.object_key != filled_record.object_key:
                    raise FormatError("object key mismatch at same shard identity",
                                      line=filled_record.source_line,
                                      field="object_key", key=filled_record.object_key)
            by_key_digest[identity] = filled_record
            merge_index[filled_record.merge_id] = filled_record
            normalized.append(filled_record)

    # Parent references must exist and form no cycles.
    for record in normalized:
        parent = record.parent_merge_id
        seen: set[str] = set()
        current = record
        while parent is not None:
            if parent not in merge_index:
                raise FormatError(
                    f"parent_merge_id {parent} does not exist",
                    line=record.source_line, field="parent_merge_id",
                    key=record.object_key,
                )
            if parent in seen:
                raise FormatError(
                    f"cycle detected at parent_merge_id {parent}",
                    line=record.source_line, field="parent_merge_id",
                    key=record.object_key,
                )
            seen.add(parent)
            current = merge_index[parent]
            parent = current.parent_merge_id
    return normalized


__all__ = [
    "encode_c1_line",
    "decode_c1_line",
    "decode_json_line",
    "encode_json_line",
    "decode_file",
    "detect_format",
    "normalize_files",
    "DecodedFile",
    "C1_LINE_WIDTH",
]
