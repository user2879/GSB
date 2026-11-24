from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import PageSpec, WalEntry


MAGIC = b"CLMPAGE1"
CODE_TO_DTYPE = {code: name for name, (_, _, code) in {
    "uint8": ("<u1", 1, 1),
    "int32": ("<i4", 4, 2),
    "int64": ("<i8", 8, 3),
    "float32": ("<f4", 4, 4),
    "float64": ("<f8", 8, 5),
}.items()}
DTYPE_BY_CODE = {
    1: "<u1",
    2: "<i4",
    3: "<i8",
    4: "<f4",
    5: "<f8",
}


@dataclass
class PageContent:
    page_id: str
    version: int
    lsn: int
    arrays: dict[str, np.ndarray]
    null_bitmap: tuple[bool, ...]

    def encode(self, page: PageSpec | None = None) -> bytes:
        return encode_page_bytes(
            self.page_id,
            self.version,
            self.lsn,
            [col.name for col in page.columns] if page else list(self.arrays),
            {name: arr for name, arr in self.arrays.items()},
            self.null_bitmap,
            column_types={col.name: col.type_code for col in page.columns} if page else None,
        )


def sha256_hex(data: bytes | str) -> str:
    digest = hashlib.sha256()
    if isinstance(data, str):
        digest.update(data.encode("utf-8"))
    else:
        digest.update(data)
    return digest.hexdigest()


def content_digest(data: bytes) -> str:
    return sha256_hex(b"colpage-content-v1\n" + data)


def version_digest(page_id: str, version: int, lsn: int, digest: str) -> str:
    return sha256_hex(f"colpage-version-v1|{page_id}|{version}|{lsn}|{digest}")


def initial_content(page: PageSpec) -> PageContent:
    arrays = {
        col.name: np.zeros(page.row_count, dtype=col.canonical_dtype)
        for col in page.columns
    }
    return PageContent(
        page_id=page.page_id,
        version=0,
        lsn=0,
        arrays=arrays,
        null_bitmap=page.null_bitmap,
    )


def _deterministic_mix(
    count: int,
    start: int,
    *,
    page_id: str,
    target_version: int,
    lsn: int,
    seed: int,
    column_index: int,
) -> np.ndarray:
    rows = np.arange(start, start + count, dtype=np.uint64)
    page_material = hashlib.sha256(page_id.encode("utf-8")).digest()
    page_nonce = int.from_bytes(page_material[:8], "little")
    with np.errstate(over="ignore"):
        value = np.full(count, np.uint64(seed % (1 << 64)), dtype=np.uint64)
        value = value * np.uint64(11400714819323198485)
        value = value + np.uint64(lsn & ((1 << 64) - 1)) * np.uint64(14029467366897019727)
        value = value + np.uint64(target_version & ((1 << 64) - 1)) * np.uint64(1609587929392839161)
        value = value + rows * np.uint64(9650029242287828579)
        value = value + np.uint64(column_index + 1) * np.uint64(7046029254386353131)
        value = value + np.uint64(page_nonce)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
        return value


def generate_patch(
    page: PageSpec,
    entry: WalEntry,
    target_version: int,
) -> dict[str, np.ndarray]:
    """Generate the deterministic half-open row patch for one WAL entry.

    The rule intentionally includes page_id, resulting page version and LSN so
    that replay workers never depend on execution order or scheduler timing.
    """
    count = entry.width
    patches: dict[str, np.ndarray] = {}
    for column_index, column in enumerate(page.columns):
        mixed = _deterministic_mix(
            count,
            entry.row_start,
            page_id=page.page_id,
            target_version=target_version,
            lsn=entry.lsn,
            seed=entry.seed,
            column_index=column_index,
        )
        if column.dtype == "uint8":
            values = (mixed & np.uint64(0xFF)).astype(np.uint8)
        elif column.dtype == "int32":
            values = (mixed & np.uint64(0xFFFFFFFF)).astype(np.uint32).view("<i4")
        elif column.dtype == "int64":
            values = mixed.view("<i8")
        elif column.dtype == "float32":
            unit = (mixed >> np.uint64(11)).astype(np.float64) / 9007199254740992.0
            values = ((unit * 2000.0) - 1000.0).astype("<f4")
        elif column.dtype == "float64":
            unit = (mixed >> np.uint64(11)).astype(np.float64) / 9007199254740992.0
            values = ((unit * 2000000.0) - 1000000.0).astype("<f8")
        else:  # pragma: no cover - validated by loader
            raise ValueError(f"unsupported dtype {column.dtype}")
        patches[column.name] = values
    return patches


def patch_byte_count(page: PageSpec, entry: WalEntry) -> int:
    return entry.width * sum(column.item_size for column in page.columns)


def patch_digest(
    page: PageSpec,
    entry: WalEntry,
    target_version: int,
    patches: dict[str, np.ndarray],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"colpage-patch-v1")
    hasher.update(
        f"{page.page_id}|{target_version}|{entry.lsn}|{entry.row_start}|{entry.row_end}|{entry.seed}".encode()
    )
    for column in page.columns:
        hasher.update(column.name.encode())
        hasher.update(column.canonical_dtype.encode())
        hasher.update(patches[column.name].tobytes(order="C"))
    return hasher.hexdigest()


def apply_patch(
    current: PageContent,
    page: PageSpec,
    entry: WalEntry,
    target_version: int,
    patches: dict[str, np.ndarray],
) -> None:
    for column in page.columns:
        current.arrays[column.name][entry.row_start : entry.row_end] = patches[column.name]
    current.version = target_version
    current.lsn = entry.lsn


def encode_page_bytes(
    page_id: str,
    version: int,
    lsn: int,
    column_names: list[str],
    arrays: dict[str, np.ndarray],
    null_bitmap: tuple[bool, ...],
    column_types: dict[str, int] | None = None,
) -> bytes:
    page_bytes = page_id.encode("utf-8")
    output = bytearray()
    output.extend(MAGIC)
    output.extend(struct.pack("<I", len(page_bytes)))
    output.extend(page_bytes)
    output.extend(struct.pack("<IQI", version, lsn, len(null_bitmap)))
    output.extend(struct.pack("<I", len(column_names)))
    for name in column_names:
        name_bytes = name.encode("utf-8")
        output.extend(struct.pack("<H", len(name_bytes)))
        output.extend(name_bytes)
        dtype = arrays[name].dtype.str
        type_code = _type_code_for_dtype(dtype, name, column_types)
        output.extend(struct.pack("<B", type_code))
        contiguous = np.ascontiguousarray(arrays[name])
        output.extend(contiguous.tobytes(order="C"))
    packed = np.packbits(np.asarray(null_bitmap, dtype=bool)).tobytes()
    output.extend(struct.pack("<I", len(packed)))
    output.extend(packed)
    return bytes(output)


def _type_code_for_dtype(
    dtype_str: str,
    name: str,
    column_types: dict[str, int] | None,
) -> int:
    if column_types and name in column_types:
        return column_types[name]
    canonical_codes = {
        "<u1": 1,
        "<i4": 2,
        "<i8": 3,
        "<f4": 4,
        "<f8": 5,
    }
    if dtype_str in canonical_codes:
        return canonical_codes[dtype_str]
    raise ValueError(f"unsupported canonical dtype {dtype_str}")


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def read(self, size: int) -> bytes:
        chunk = self.data[self.offset : self.offset + size]
        if len(chunk) != size:
            raise ValueError("truncated page file")
        self.offset += size
        return chunk

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))


def decode_page_bytes(data: bytes) -> PageContent:
    reader = _Reader(data)
    if reader.read(len(MAGIC)) != MAGIC:
        raise ValueError("bad page magic")
    (page_id_len,) = reader.unpack("<I")
    page_id = reader.read(page_id_len).decode("utf-8")
    version, lsn, row_count = reader.unpack("<IQI")
    (column_count,) = reader.unpack("<I")
    arrays: dict[str, np.ndarray] = {}
    for _ in range(column_count):
        (name_len,) = reader.unpack("<H")
        name = reader.read(name_len).decode("utf-8")
        (type_code,) = reader.unpack("<B")
        dtype = DTYPE_BY_CODE[type_code]
        item_size = np.dtype(dtype).itemsize
        raw = reader.read(row_count * item_size)
        arrays[name] = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()
    (bitmap_len,) = reader.unpack("<I")
    packed = np.frombuffer(reader.read(bitmap_len), dtype=np.uint8)
    bits = np.unpackbits(packed)[:row_count]
    null_bitmap = tuple(bool(value) for value in bits.tolist())
    return PageContent(
        page_id=page_id,
        version=version,
        lsn=lsn,
        arrays=arrays,
        null_bitmap=null_bitmap,
    )


def write_page_file(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)


def read_page_file(path: str | Path, expected_digest: str | None = None) -> bytes:
    data = Path(path).read_bytes()
    actual = content_digest(data)
    if expected_digest is not None and actual != expected_digest:
        from .errors import SnapshotMismatchError

        raise SnapshotMismatchError(
            "page content digest does not match ledger",
            field="content_digest",
            details={"path": str(path), "expected": expected_digest, "actual": actual},
        )
    return data
