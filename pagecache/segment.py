from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import zlib

from .canonical import atomic_write_bytes
from .errors import EngineError, ErrorCode
from .model import CURRENT_FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS, PageData
from .pagefile import PAGE_MAGIC, decode_page_bytes


SEGMENT_MAGIC = b"PGSEG001"
SEGMENT_TRAILER = b"PGSEND01"
_HEADER = struct.Struct("<8sHBBHII")
_CHECKSUM = struct.Struct("<I")
_TRAILER = struct.Struct("<8sI")


class SegmentRecord:
    __slots__ = (
        "version",
        "page_id",
        "row_count",
        "compressed",
        "payload",
        "stored_payload",
        "payload_checksum",
        "raw_bytes",
    )

    def __init__(
        self,
        *,
        version: int,
        page_id: str,
        row_count: int,
        compressed: bool,
        payload: bytes,
        stored_payload: bytes,
        payload_checksum: int | None,
        raw_bytes: bytes,
    ) -> None:
        self.version = version
        self.page_id = page_id
        self.row_count = row_count
        self.compressed = compressed
        self.payload = payload
        self.stored_payload = stored_payload
        self.payload_checksum = payload_checksum
        self.raw_bytes = raw_bytes

    def page(self) -> PageData:
        return decode_page_bytes(self.payload, expected_page_id=self.page_id, source_path=f"segment[{self.page_id}]")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "page_id": self.page_id,
            "row_count": self.row_count,
            "compressed": self.compressed,
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
            "stored_payload_bytes": len(self.stored_payload),
            "payload_checksum": self.payload_checksum,
        }


def encode_segment(
    *,
    page_id: str,
    row_count: int,
    payload: bytes,
    compressed: bool,
    version: int = CURRENT_FORMAT_VERSION,
) -> bytes:
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(ErrorCode.E_FORMAT, "unsupported segment version", path="$segment.version")
    if version < 3 and compressed:
        raise EngineError(ErrorCode.E_FORMAT, "compression is only defined for segment v3", path="$segment.compressed")
    _validate_ref(page_id, path="$segment.page_id")
    payload_id = decode_page_bytes(payload, source_path="$segment.payload").page_id
    if payload_id != page_id:
        raise EngineError(
            ErrorCode.E_PAYLOAD,
            "segment page_id does not match its payload",
            page_id=page_id,
            details={"payload_page_id": payload_id},
        )
    page_id_bytes = page_id.encode("utf-8")
    stored_payload = zlib.compress(payload, level=9) if compressed else payload
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    body = bytearray(
        _HEADER.pack(
            SEGMENT_MAGIC,
            version,
            1 if compressed else 0,
            0,
            len(page_id_bytes),
            row_count,
            len(stored_payload),
        )
    )
    body.extend(page_id_bytes)
    if version >= 2:
        body.extend(_CHECKSUM.pack(checksum))
    body.extend(stored_payload)
    return bytes(body) + _TRAILER.pack(SEGMENT_TRAILER, zlib.crc32(body) & 0xFFFFFFFF)


def write_segment(
    path: str | Path,
    *,
    page_id: str,
    row_count: int,
    payload: bytes,
    compressed: bool,
    version: int = CURRENT_FORMAT_VERSION,
) -> bytes:
    data = encode_segment(page_id=page_id, row_count=row_count, payload=payload, compressed=compressed, version=version)
    atomic_write_bytes(path, data)
    return data


def read_segment_file(path: str | Path, *, expected_page_id: str | None = None) -> SegmentRecord:
    target = Path(path)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise EngineError(ErrorCode.E_INTEGRITY, f"cannot read segment: {exc}", path=str(target), page_id=expected_page_id) from exc
    return decode_segment(data, source_path=str(target), expected_page_id=expected_page_id)


def decode_segment(data: bytes, *, source_path: str = "$segment", expected_page_id: str | None = None) -> SegmentRecord:
    minimum = _HEADER.size + _TRAILER.size
    if len(data) < minimum:
        raise EngineError(ErrorCode.E_FORMAT, "segment is truncated", path=f"{source_path}.header", page_id=expected_page_id)
    magic, version, flags, reserved, page_id_len, row_count, payload_len = _HEADER.unpack_from(data, 0)
    if magic != SEGMENT_MAGIC:
        raise EngineError(ErrorCode.E_FORMAT, "invalid segment magic", path=f"{source_path}.magic", page_id=expected_page_id)
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "unsupported segment format version",
            path=f"{source_path}.version",
            page_id=expected_page_id,
            details={"version": version},
        )
    if flags not in (0, 1):
        raise EngineError(ErrorCode.E_FORMAT, "invalid segment compression flag", path=f"{source_path}.flags", page_id=expected_page_id)
    if version < 3 and flags:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "pre-v3 segment cannot be marked compressed",
            path=f"{source_path}.flags",
            page_id=expected_page_id,
        )
    if reserved != 0:
        raise EngineError(ErrorCode.E_FORMAT, "segment reserved field must be zero", path=f"{source_path}.reserved", page_id=expected_page_id)
    offset = _HEADER.size
    if page_id_len == 0 or offset + page_id_len > len(data) - _TRAILER.size:
        raise EngineError(ErrorCode.E_FORMAT, "invalid segment page_id length", path=f"{source_path}.page_id_len", page_id=expected_page_id)
    try:
        page_id = data[offset : offset + page_id_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EngineError(ErrorCode.E_FORMAT, "segment page_id is not UTF-8", path=f"{source_path}.page_id", page_id=expected_page_id) from exc
    offset += page_id_len
    _validate_ref(page_id, path=f"{source_path}.page_id")
    if expected_page_id is not None and page_id != expected_page_id:
        raise EngineError(
            ErrorCode.E_PAYLOAD,
            "segment identifier does not match the expected page identifier",
            page_id=expected_page_id,
            details={"segment_page_id": page_id},
        )
    if row_count <= 0:
        raise EngineError(ErrorCode.E_FORMAT, "segment row_count must be positive", path=f"{source_path}.row_count", page_id=page_id)

    stored_checksum: int | None = None
    if version >= 2:
        if offset + _CHECKSUM.size > len(data) - _TRAILER.size:
            raise EngineError(ErrorCode.E_FORMAT, "segment payload checksum is missing", path=f"{source_path}.payload_checksum", page_id=page_id)
        stored_checksum = _CHECKSUM.unpack_from(data, offset)[0]
        offset += _CHECKSUM.size
    payload_start = offset
    payload_end = payload_start + payload_len
    trailer_start = len(data) - _TRAILER.size
    if payload_end != trailer_start:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "segment payload length is not self-consistent",
            path=f"{source_path}.payload_length",
            page_id=page_id,
            details={"expected_end": payload_end, "trailer_start": trailer_start},
        )
    stored_payload = data[payload_start:payload_end]
    trailer_magic, expected_crc = _TRAILER.unpack_from(data, trailer_start)
    if trailer_magic != SEGMENT_TRAILER:
        raise EngineError(ErrorCode.E_FORMAT, "invalid segment trailer", path=f"{source_path}.trailer.magic", page_id=page_id)
    actual_crc = zlib.crc32(data[:trailer_start]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise EngineError(
            ErrorCode.E_INTEGRITY,
            "segment file checksum mismatch",
            path=f"{source_path}.trailer.checksum",
            page_id=page_id,
            details={"expected": expected_crc, "actual": actual_crc},
        )

    compressed = bool(flags)
    payload = _materialize_payload(page_id, stored_payload, compressed, source_path)
    if stored_checksum is not None:
        actual_payload_checksum = zlib.crc32(payload) & 0xFFFFFFFF
        if stored_checksum != actual_payload_checksum:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "segment payload checksum mismatch",
                path=f"{source_path}.payload_checksum",
                page_id=page_id,
                details={"expected": stored_checksum, "actual": actual_payload_checksum},
            )
    try:
        page = decode_page_bytes(payload, expected_page_id=page_id, source_path=f"{source_path}.payload")
    except EngineError as exc:
        if exc.code is ErrorCode.E_PAYLOAD:
            raise
        raise EngineError(
            ErrorCode.E_PAYLOAD,
            "segment payload is not a valid page for its compression flag",
            path=f"{source_path}.payload",
            page_id=page_id,
            details={"cause": exc.code.value, "cause_path": exc.path},
        ) from exc
    if page.row_count != row_count:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "segment row_count does not match page payload",
            path=f"{source_path}.row_count",
            page_id=page_id,
            details={"segment_rows": row_count, "page_rows": page.row_count},
        )
    return SegmentRecord(
        version=version,
        page_id=page_id,
        row_count=row_count,
        compressed=compressed,
        payload=payload,
        stored_payload=stored_payload,
        payload_checksum=stored_checksum,
        raw_bytes=bytes(data),
    )


def segment_inventory(overflow_dir: str | Path) -> list[dict[str, object]]:
    directory = Path(overflow_dir)
    if not directory.exists():
        return []
    inventory: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.seg"), key=lambda item: item.name):
        try:
            record = read_segment_file(path)
        except EngineError as exc:
            inventory.append({"name": path.name, "status": "invalid", "error": exc.to_dict()})
            continue
        inventory.append(
            {
                "name": path.name,
                "status": "ok",
                "page_id": record.page_id,
                "version": record.version,
                "compressed": record.compressed,
                "row_count": record.row_count,
                "payload_sha256": hashlib.sha256(record.payload).hexdigest(),
                "file_sha256": hashlib.sha256(record.raw_bytes).hexdigest(),
                "payload_checksum": record.payload_checksum,
                "file_size": len(record.raw_bytes),
            }
        )
    return inventory


def segment_path(overflow_dir: str | Path, ref: str | None, page_id: str) -> Path:
    reference = ref or f"{page_id}.seg"
    _validate_ref(reference, path="$segment.ref")
    return Path(overflow_dir) / reference


def _materialize_payload(page_id: str, stored_payload: bytes, compressed: bool, source_path: str) -> bytes:
    if compressed:
        try:
            return zlib.decompress(stored_payload)
        except zlib.error as exc:
            raise EngineError(
                ErrorCode.E_PAYLOAD,
                "compressed flag is set but the payload is not a zlib stream",
                path=f"{source_path}.payload",
                page_id=page_id,
                details={"compression_flag": True},
            ) from exc

    if stored_payload.startswith(PAGE_MAGIC):
        return stored_payload
    try:
        decoded = zlib.decompress(stored_payload)
    except zlib.error:
        return stored_payload
    if decoded.startswith(PAGE_MAGIC):
        raise EngineError(
            ErrorCode.E_PAYLOAD,
            "compression flag is clear but the payload is a zlib stream",
            path=f"{source_path}.payload",
            page_id=page_id,
            details={"compression_flag": False},
        )
    return stored_payload


def _validate_ref(ref: str, *, path: str) -> None:
    if not ref or ref in {".", ".."}:
        raise EngineError(ErrorCode.E_FORMAT, "invalid file reference", path=path)
    if "/" in ref or "\\" in ref or "\x00" in ref:
        raise EngineError(ErrorCode.E_FORMAT, "file reference must be a base filename", path=path)
