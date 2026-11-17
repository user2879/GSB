"""On-disk columnar segment format, write/read with whole-segment validation.

File layout (little-endian)::

    magic "CSEG0001" (8 bytes)
    u32 header_len
    header_len bytes: JSON table of contents
    block payloads back-to-back at offsets recorded in the TOC

Each TOC block entry records: column, row group, encoding, compression flag,
row count, width, uncompressed/compressed payload sizes, payload offset and
CRC32.  Compression uses zlib and only the payload bytes are compressed; the
block header remains plain JSON so block-level metadata is always inspectable.
"""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .encoding import decode_block, encode_block
from .errors import EncodingError, PayloadError, SegmentError
from .model import BlockId, ColumnType, Encoding, SegmentDefinition

MAGIC = b"CSEG0001"


@dataclass
class BlockMeta:
    block_id: BlockId
    encoding: Encoding
    compressed: bool
    rows: int
    width: int
    uncompressed_bytes: int
    compressed_bytes: int
    payload_offset: int
    crc32: int
    min_value: int
    max_value: int

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "block": self.block_id.to_dict(),
            "encoding": self.encoding.value,
            "compressed": self.compressed,
            "rows": self.rows,
            "width": self.width,
            "uncompressed_bytes": self.uncompressed_bytes,
            "compressed_bytes": self.compressed_bytes,
            "payload_offset": self.payload_offset,
            "crc32": self.crc32,
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BlockMeta":
        return cls(
            block_id=BlockId.from_dict(raw["block"]),
            encoding=Encoding(raw["encoding"]),
            compressed=bool(raw["compressed"]),
            rows=int(raw["rows"]),
            width=int(raw["width"]),
            uncompressed_bytes=int(raw["uncompressed_bytes"]),
            compressed_bytes=int(raw["compressed_bytes"]),
            payload_offset=int(raw["payload_offset"]),
            crc32=int(raw["crc32"]),
            min_value=int(raw["min_value"]),
            max_value=int(raw["max_value"]),
        )


@dataclass
class LoadedSegment:
    definition: SegmentDefinition
    path: str
    blocks: List[BlockMeta]
    raw_bytes: bytes
    file_hash: str

    def block_meta(self, row_group: int, column: str) -> BlockMeta:
        for meta in self.blocks:
            if (
                meta.block_id.row_group == row_group
                and meta.block_id.column == column
            ):
                return meta
        raise SegmentError(
            "block missing from segment",
            {"column": column, "row_group": row_group, "segment": self.definition.name},
        )

    def file_summary(self) -> Dict[str, Any]:
        return {
            "segment": self.definition.name,
            "path": os.path.abspath(self.path),
            "file_hash": self.file_hash,
            "format_version": self.definition.format_version,
            "total_rows": self.definition.total_rows,
            "num_row_groups": self.definition.num_row_groups,
            "columns": [c.name for c in self.definition.columns],
            "compression": {
                c.name: c.compressed for c in self.definition.columns
            },
        }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_segment(
    definition: SegmentDefinition,
    path: str,
    selected_encodings: Optional[Dict[str, Encoding]] = None,
) -> Dict[str, Encoding]:
    """Encode every (row group, column) block and persist one segment file.

    ``selected_encodings`` maps ``"row_group:column"`` to an explicit encoding;
    missing entries fall back to the definition block specs and finally to
    automatic deterministic selection.  Returns the full encoding map used.
    """
    selected_encodings = dict(selected_encodings or {})
    bounds = definition.row_group_bounds()
    toc_blocks: List[Dict[str, Any]] = []
    payload_blob = bytearray()

    for col in definition.columns:
        full_values = definition.values[col.name]
        for row_group, (start, end) in enumerate(bounds):
            key = f"{row_group}:{col.name}"
            values = list(full_values[start:end])
            explicit = definition.block_encoding(row_group, col.name)
            encoding = (
                selected_encodings.get(key)
                or explicit
                or _auto_select(values, col.dtype)
            )
            selected_encodings[key] = encoding
            raw_payload = encode_block(values, col.dtype, encoding)
            stored = (
                zlib.compress(raw_payload, level=9) if col.compressed else raw_payload
            )
            crc = zlib.crc32(raw_payload) & 0xFFFFFFFF
            offset = len(payload_blob)
            payload_blob.extend(stored)
            meta = BlockMeta(
                block_id=BlockId(definition.name, row_group, col.name),
                encoding=encoding,
                compressed=col.compressed,
                rows=len(values),
                width=col.dtype.width,
                uncompressed_bytes=len(raw_payload),
                compressed_bytes=len(stored),
                payload_offset=offset,
                crc32=crc,
                min_value=min(values),
                max_value=max(values),
            )
            toc_blocks.append(meta.to_dict())

    header = {
        "magic": MAGIC.decode("ascii"),
        "format_version": definition.format_version,
        "segment": definition.name,
        "columns": [c.to_dict() for c in definition.columns],
        "row_group_rows": list(definition.row_group_rows),
        "blocks": toc_blocks,
    }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(MAGIC)
        handle.write(len(header_bytes).to_bytes(4, "little", signed=False))
        handle.write(header_bytes)
        handle.write(bytes(payload_blob))
    return selected_encodings


def _auto_select(values: Sequence[int], dtype: ColumnType) -> Encoding:
    from .encoding import select_encoding

    encoding, _decision = select_encoding(values, dtype)
    return encoding


# ---------------------------------------------------------------------------
# Reading / whole-segment validation
# ---------------------------------------------------------------------------

def load_segment(path: str, definition: Optional[SegmentDefinition] = None) -> LoadedSegment:
    """Read and validate an entire segment.

    Validation is all-or-nothing: any corrupt block raises a domain error
    (``E_PAYLOAD`` / ``E_ENCODING`` / generic ``E_SEGMENT``) carrying column,
    row group and payload offset, and no partially decoded segment is returned.
    """
    if not os.path.exists(path):
        raise SegmentError("segment file not found", {"path": path})
    with open(path, "rb") as handle:
        raw = handle.read()
    file_hash = hashlib.sha256(raw).hexdigest()

    if len(raw) < len(MAGIC) + 4:
        raise SegmentError("file shorter than segment header", {"path": path})
    if raw[: len(MAGIC)] != MAGIC:
        raise SegmentError("bad segment magic", {"path": path, "magic": raw[:8].hex()})
    header_len = int.from_bytes(raw[8:12], "little", signed=False)
    header_end = 12 + header_len
    if header_end > len(raw):
        raise SegmentError(
            "segment header truncated",
            {"header_len": header_len, "file_len": len(raw)},
        )
    try:
        header = json.loads(raw[12:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SegmentError("segment header is not valid JSON", {"path": path}) from exc

    blocks = [BlockMeta.from_dict(item) for item in header["blocks"]]
    if definition is None:
        raise SegmentError(
            "segment definition required for validated loading", {"path": path}
        )
    payload_start = header_end
    _cross_validate(definition, blocks, path, raw, payload_start)

    # Decode every block: structural corruption must fail the whole segment.
    bounds = definition.row_group_bounds()
    for meta in blocks:
        payload = _extract_payload(raw, meta, payload_start)
        raw_payload = _inflate(payload, meta)
        _check_crc(raw_payload, meta)
        try:
            decoded = decode_block(raw_payload, definition.column(meta.block_id.column).dtype, meta.encoding)
        except EncodingError as exc:
            # Re-raise with full block coordinates for whole-segment rejection.
            exc.context.update(
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                    "segment": header["segment"],
                }
            )
            raise
        start, end = bounds[meta.block_id.row_group]
        if len(decoded) != end - start or len(decoded) != meta.rows:
            raise SegmentError(
                "block header row count inconsistent with payload/segment",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                    "header_rows": meta.rows,
                    "decoded_rows": len(decoded),
                },
            )

    return LoadedSegment(
        definition=definition, path=path, blocks=blocks, raw_bytes=raw, file_hash=file_hash
    )


def decode_segment_block(
    segment: LoadedSegment, meta: BlockMeta, payload_start: Optional[int] = None
) -> List[int]:
    """Decode a single validated block (used by scan worker tasks)."""
    if payload_start is None:
        payload_start = _payload_start(segment.raw_bytes)
    payload = _extract_payload(segment.raw_bytes, meta, payload_start)
    raw_payload = _inflate(payload, meta)
    _check_crc(raw_payload, meta)
    dtype = segment.definition.column(meta.block_id.column).dtype
    return decode_block(raw_payload, dtype, meta.encoding)


def _payload_start(raw: bytes) -> int:
    header_len = int.from_bytes(raw[8:12], "little", signed=False)
    return 12 + header_len


def _extract_payload(raw: bytes, meta: BlockMeta, payload_start: int) -> bytes:
    absolute_offset = payload_start + meta.payload_offset
    end = absolute_offset + meta.compressed_bytes
    if end > len(raw):
        raise PayloadError(
            "block payload truncated inside segment file",
            {
                "column": meta.block_id.column,
                "row_group": meta.block_id.row_group,
                "payload_offset": absolute_offset,
                "declared_bytes": meta.compressed_bytes,
                "file_len": len(raw),
            },
        )
    return raw[absolute_offset:end]


def _inflate(payload: bytes, meta: BlockMeta) -> bytes:
    if not meta.compressed:
        if len(payload) != meta.uncompressed_bytes:
            raise PayloadError(
                "uncompressed payload length disagrees with block header",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                    "header_bytes": meta.uncompressed_bytes,
                    "actual_bytes": len(payload),
                },
            )
        return payload
    try:
        inflated = zlib.decompress(payload)
    except zlib.error as exc:
        raise PayloadError(
            "compressed block payload cannot be inflated",
            {
                "column": meta.block_id.column,
                "row_group": meta.block_id.row_group,
                "payload_offset": meta.payload_offset,
            },
        ) from exc
    if len(inflated) != meta.uncompressed_bytes:
        raise PayloadError(
            "inflated payload length disagrees with block header",
            {
                "column": meta.block_id.column,
                "row_group": meta.block_id.row_group,
                "payload_offset": meta.payload_offset,
                "header_bytes": meta.uncompressed_bytes,
                "actual_bytes": len(inflated),
            },
        )
    return inflated


def _check_crc(raw_payload: bytes, meta: BlockMeta) -> None:
    actual = zlib.crc32(raw_payload) & 0xFFFFFFFF
    if actual != meta.crc32:
        raise SegmentError(
            "block payload CRC mismatch",
            {
                "column": meta.block_id.column,
                "row_group": meta.block_id.row_group,
                "payload_offset": meta.payload_offset,
                "expected_crc": meta.crc32,
                "actual_crc": actual,
            },
        )


def _cross_validate(
    definition: SegmentDefinition,
    blocks: List[BlockMeta],
    path: str,
    raw: bytes,
    payload_start: int,
) -> None:
    expected = {
        (rg, col.name)
        for col in definition.columns
        for rg in range(definition.num_row_groups)
    }
    actual = {(m.block_id.row_group, m.block_id.column) for m in blocks}
    if actual != expected:
        raise SegmentError(
            "segment block set does not match definition",
            {"missing": sorted(expected - actual), "extra": sorted(actual - expected)},
        )
    payload_end = len(raw)
    for meta in blocks:
        if meta.encoding not in (Encoding.RAW, Encoding.DICTIONARY, Encoding.RLE):
            raise EncodingError(
                "unsupported encoding marker in block header",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                    "encoding": str(meta.encoding),
                },
            )
        if meta.rows <= 0:
            raise SegmentError(
                "block header reports non-positive row count",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                },
            )
        dtype = definition.column(meta.block_id.column).dtype
        if meta.width != dtype.width:
            raise SegmentError(
                "block width disagrees with schema",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                    "header_width": meta.width,
                    "schema_width": dtype.width,
                },
            )
        if meta.compressed_bytes <= 0 or meta.uncompressed_bytes <= 0:
            raise SegmentError(
                "block header reports non-positive byte counts",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": meta.payload_offset,
                },
            )
        absolute_offset = payload_start + meta.payload_offset
        if (
            meta.payload_offset < 0
            or absolute_offset + meta.compressed_bytes > payload_end
        ):
            raise PayloadError(
                "block payload offset outside segment file",
                {
                    "column": meta.block_id.column,
                    "row_group": meta.block_id.row_group,
                    "payload_offset": absolute_offset,
                },
            )


# ---------------------------------------------------------------------------
# Fault injection (input-only; source code is never modified)
# ---------------------------------------------------------------------------

def inject_payload_truncation(path: str, output_path: str) -> int:
    """Declare a payload one byte shorter than the bytes actually on disk.

    The TOC/load-length self-consistency check then fails with ``E_PAYLOAD``
    while the segment file itself remains fully readable.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    header_len = int.from_bytes(raw[8:12], "little", signed=False)
    header = json.loads(raw[12:12 + header_len].decode("utf-8"))
    blocks = header["blocks"]
    target = max(blocks, key=lambda b: b["payload_offset"])
    target["compressed_bytes"] = int(target["compressed_bytes"]) - 1
    payload_start = 12 + header_len
    new_header = json.dumps(header, sort_keys=True).encode("utf-8")
    rebuilt = (
        raw[:8]
        + len(new_header).to_bytes(4, "little")
        + new_header
        + raw[12 + header_len:]
    )
    with open(output_path, "wb") as handle:
        handle.write(rebuilt)
    return payload_start + target["payload_offset"]


def inject_encoding_mismatch(path: str, output_path: str) -> Dict[str, Any]:
    """Flip a block's encoding marker without touching its payload -> E_ENCODING."""
    with open(path, "rb") as handle:
        raw = handle.read()
    header_len = int.from_bytes(raw[8:12], "little", signed=False)
    header = json.loads(raw[12:12 + header_len].decode("utf-8"))
    target = header["blocks"][0]
    flip = {
        Encoding.RAW.value: Encoding.DICTIONARY.value,
        Encoding.DICTIONARY.value: Encoding.RLE.value,
        Encoding.RLE.value: Encoding.RAW.value,
    }
    old_encoding = target["encoding"]
    target["encoding"] = flip[target["encoding"]]
    new_header = json.dumps(header, sort_keys=True).encode("utf-8")
    delta = len(new_header) - header_len
    if delta != 0:
        # Payload offsets are relative to the payload area; the area start moves
        # but relative offsets stay valid, so only the outer length prefix changes.
        pass
    rebuilt = (
        raw[:8]
        + len(new_header).to_bytes(4, "little")
        + new_header
        + raw[12 + header_len:]
    )
    with open(output_path, "wb") as handle:
        handle.write(rebuilt)
    return {
        "column": target["block"]["column"],
        "row_group": target["block"]["row_group"],
        "payload_offset": target["payload_offset"],
        "flipped_from": old_encoding,
        "flipped_to": target["encoding"],
    }
