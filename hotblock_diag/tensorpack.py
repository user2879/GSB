"""Little-endian binary tensor pack reader/writer.

Pack layout (all integers little-endian)::

    magic   4 bytes  b"HSPK"
    version uint32   1
    n       uint32   number of tensors
    per tensor:
        name_len uint16
        name     utf-8 bytes
        ndim     uint16
        shape    ndim * int64
        count    int64   number of float32 elements
        data     count * float32 (little-endian)
    crc32   uint32   over every byte preceding the crc field

The file ends immediately after the 4-byte CRC. A truncated pack therefore
fails either the structural length checks or the CRC check.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .errors import E_PACK_TRUNCATED, E_TENSOR_LAYOUT, FailureList

MAGIC = b"HSPK"
VERSION = 1
HEADER = struct.Struct("<4sII")
U16 = struct.Struct("<H")
I64 = struct.Struct("<q")
U32 = struct.Struct("<I")


class TensorLayoutError(ValueError):
    pass


class PackTruncatedError(IOError):
    pass


@dataclass
class TensorPack:
    tensors: dict[str, np.ndarray]
    order: list[str]
    raw_fingerprint: str

    def get(self, name: str) -> np.ndarray:
        if name not in self.tensors:
            raise TensorLayoutError(f"missing tensor {name!r}")
        return self.tensors[name]


def write_pack(path: str, tensors: dict[str, np.ndarray], order: list[str] | None = None) -> None:
    names = order if order is not None else list(tensors.keys())
    body = bytearray()
    body += HEADER.pack(MAGIC, VERSION, len(names))
    for name in names:
        array = np.asarray(tensors[name], dtype=np.float32)
        encoded = name.encode("utf-8")
        body += U16.pack(len(encoded))
        body += encoded
        body += U16.pack(array.ndim)
        for dim in array.shape:
            body += I64.pack(int(dim))
        body += I64.pack(array.size)
        body += array.astype("<f4", copy=False).tobytes(order="C")
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF
    body += U32.pack(crc)
    with open(path, "wb") as handle:
        handle.write(bytes(body))


def read_pack(path: str, failures: FailureList) -> TensorPack | None:
    with open(path, "rb") as handle:
        blob = handle.read()
    fingerprint = __import__("hashlib").sha256(blob).hexdigest()[:16]
    cursor = 0

    def take(size: int, what: str) -> bytes:
        nonlocal cursor
        if cursor + size > len(blob):
            if not any(item.code == E_PACK_TRUNCATED and item.target == path
                       for item in failures.failures):
                failures.add(
                    E_PACK_TRUNCATED,
                    "tensor_parse",
                    path,
                    PackTruncatedError(what),
                    detail=f"need {size} bytes at offset {cursor}, have {len(blob) - cursor}",
                )
            raise PackTruncatedError(what)
        chunk = blob[cursor:cursor + size]
        cursor += size
        return chunk

    try:
        magic, version, count = HEADER.unpack(take(HEADER.size, "header"))
        if magic != MAGIC:
            raise TensorLayoutError(f"bad magic {magic!r}")
        if version != VERSION:
            raise TensorLayoutError(f"unsupported version {version}")
        tensors: dict[str, np.ndarray] = {}
        order: list[str] = []
        for _ in range(count):
            name_len = U16.unpack(take(U16.size, "name_len"))[0]
            name = take(name_len, "name").decode("utf-8")
            ndim = U16.unpack(take(U16.size, "ndim"))[0]
            shape = tuple(I64.unpack(take(I64.size, "dim"))[0] for _ in range(ndim))
            elem_count = I64.unpack(take(I64.size, "count"))[0]
            expected = 1
            for dim in shape:
                expected *= dim
            if expected != elem_count:
                raise TensorLayoutError(
                    f"tensor {name!r}: shape {shape} implies {expected} elems, header says {elem_count}"
                )
            raw = take(elem_count * 4, "data")
            array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            if name in tensors:
                raise TensorLayoutError(f"duplicate tensor {name!r}")
            tensors[name] = array
            order.append(name)
        crc_expected = U32.unpack(take(U32.size, "crc"))[0]
        if cursor != len(blob):
            raise TensorLayoutError(f"{len(blob) - cursor} trailing bytes after last tensor")
        crc_actual = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise TensorLayoutError("crc32 mismatch; pack corrupted or truncated")
        for required in ("query", "key", "value"):
            if required not in tensors:
                raise TensorLayoutError(f"required tensor {required!r} absent")
        return TensorPack(tensors=tensors, order=order, raw_fingerprint=fingerprint)
    except PackTruncatedError:
        return None
    except (struct.error, ValueError, UnicodeDecodeError) as exc:
        failures.add(E_TENSOR_LAYOUT, "tensor_parse", path, exc, detail=str(exc))
        return None
