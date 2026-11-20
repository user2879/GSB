"""Little-endian binary quantized data pack and JSON shape descriptor.

Pack layout (all integers little-endian, int32 for meta/int8 payloads
zero-extended into int32 storage)::

    magic 'QPCK' | version int32 (=1) | n_blocks int32
    per block: name_len int32 | name utf8 | ndim int32 | dims int32[ndim]
               | dtype_code int32 (0=int8,1=int16,2=int32) | values[...]
"""
import json
import os
import struct

import numpy as np

from .errors import QCGError, E_QUANT_PACK_INVALID

MAGIC = b"QPCK"
VERSION = 1
_DTYPE = {0: np.int8, 1: np.int16, 2: np.int32}
_CODE = {np.dtype(np.int8): 0, np.dtype(np.int16): 1,
         np.dtype(np.int32): 2}


def write_pack(path: str, blocks: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<ii", VERSION, len(blocks)))
        for name in sorted(blocks):
            arr = np.ascontiguousarray(blocks[name])
            raw_name = name.encode("utf-8")
            fh.write(struct.pack("<i", len(raw_name)))
            fh.write(raw_name)
            fh.write(struct.pack("<i", arr.ndim))
            fh.write(struct.pack("<%di" % arr.ndim, *arr.shape))
            code = _CODE[arr.dtype]
            fh.write(struct.pack("<i", code))
            if code == 0:
                fh.write(arr.astype(np.int8).tobytes())
            else:
                fh.write(arr.astype("<%s" % arr.dtype.name).tobytes())
    os.replace(tmp, path)


def read_pack(path: str) -> dict:
    with open(path, "rb") as fh:
        blob = fh.read()
    pos = 0

    def take(fmt: str, size: int):
        nonlocal pos
        if pos + size > len(blob):
            raise QCGError(E_QUANT_PACK_INVALID,
                           "quant pack truncated at offset %d" % pos)
        values = struct.unpack_from(fmt, blob, pos)
        pos += size
        return values

    if blob[:4] != MAGIC:
        raise QCGError(E_QUANT_PACK_INVALID, "bad pack magic")
    pos = 4
    version, n_blocks = take("<ii", 8)
    if version != VERSION:
        raise QCGError(E_QUANT_PACK_INVALID,
                       "unsupported pack version %d" % version)
    blocks = {}
    for _ in range(n_blocks):
        (name_len,) = take("<i", 4)
        (name,) = take("<%ds" % name_len, name_len)
        (ndim,) = take("<i", 4)
        dims = take("<%di" % ndim, 4 * ndim)
        (code,) = take("<i", 4)
        if code not in _DTYPE:
            raise QCGError(E_QUANT_PACK_INVALID, "unknown dtype code %d" % code)
        dtype = np.dtype(_DTYPE[code]).newbyteorder("<")
        count = int(np.prod(dims)) if dims else 0
        need = count * dtype.itemsize
        if pos + need > len(blob):
            raise QCGError(E_QUANT_PACK_INVALID,
                           "quant pack truncated inside block %r"
                           % name.decode("utf-8", "replace"))
        arr = np.frombuffer(blob, dtype=dtype, count=count, offset=pos)
        arr = arr.reshape(dims).copy()
        pos += need
        blocks[name.decode("utf-8")] = arr
    return blocks


def write_shapes(path: str, blocks: dict) -> None:
    doc = {name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
           for name, arr in sorted(blocks.items())}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def read_shapes(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_sample_blocks(seed: int = 20261121) -> dict:
    """Deterministic sample blocks; same seed => byte-identical pack."""
    rng = np.random.default_rng(seed)
    blocks = {}
    blocks["act:n0000"] = rng.integers(-8, 8, size=(8, 8), dtype=np.int8)
    blocks["w:n0001"] = rng.integers(-4, 4, size=(3, 3), dtype=np.int8)
    blocks["w:n0002"] = rng.integers(-3, 3, size=(2, 2), dtype=np.int8)
    blocks["w:n0003"] = rng.integers(-2, 2, size=(3, 3), dtype=np.int8)
    blocks["w:n0004"] = rng.integers(-2, 2, size=(2, 2), dtype=np.int8)
    return blocks
