"""Deterministic local sample generation and digest-protected column cache.

Samples are produced by a pure-integer LCG (see util.LcgRng). Both "integer"
and "float" columns are generated directly as fixed-point integer ulps at the
source's ingest scale; no float and no network access are involved.

Cache file layout (little-endian, deterministic)
------------------------------------------------
magic ``FPPCOL01`` | 2-byte header length | canonical JSON header |
N * int64 LE values (null encoded as the sentinel below) | 32-byte sha256 of
everything preceding the digest. Corruption (truncation / byte flip / digest
mismatch) raises E_DATA.
"""

from __future__ import annotations

import os
import struct

from .errors import DataError
from .fixedpoint import INT32_MAX, INT32_MIN
from .spec import SourceSpec, _source_payload
from .util import bytes_digest, canonical_json, content_digest
from .util import LcgRng

_MAGIC = b"FPPCOL01"
_NULL_SENTINEL = 1 << 63


def generate_column(src: SourceSpec) -> list[int | None]:
    value_rng = LcgRng(src.seed)
    null_rng = LcgRng(src.null_seed)
    column: list[int | None] = []
    for _ in range(src.rows):
        is_null = null_rng.next_bps() < src.null_rate_bps
        if is_null:
            column.append(None)
            continue
        value = value_rng.next_int(src.low, src.high)
        column.append(value)
    return column


def column_path(workdir: str, sid: str) -> str:
    return os.path.join(workdir, f"source_{sid}.col")


def write_column(path: str, src: SourceSpec, column: list[int | None]) -> None:
    header = canonical_json(_source_payload(src)).encode("utf-8")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = bytearray()
    body += _MAGIC
    body += struct.pack("<H", len(header))
    body += header
    for value in column:
        body += struct.pack("<Q", _NULL_SENTINEL if value is None else value)
    digest = bytes_digest(bytes(body))
    with open(path, "wb") as fh:
        fh.write(body)
        fh.write(bytes.fromhex(digest))


def read_column(path: str, src: SourceSpec) -> list[int | None]:
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError as exc:
        raise DataError(f"cannot read source cache for {src.sid}: {exc}",
                        location=src.sid, field="cache") from exc
    if len(blob) < len(_MAGIC) + 2 + 32 or blob[: len(_MAGIC)] != _MAGIC:
        raise DataError("source cache bad magic/truncated",
                        location=src.sid, field="cache")
    payload = blob[:-32]
    stored_digest = blob[-32:].hex()
    if bytes_digest(payload) != stored_digest:
        raise DataError("source cache content digest mismatch",
                        location=src.sid, field="digest")
    try:
        hlen = struct.unpack("<H", payload[len(_MAGIC): len(_MAGIC) + 2])[0]
        header_bytes = payload[len(_MAGIC) + 2: len(_MAGIC) + 2 + hlen]
        body = payload[len(_MAGIC) + 2 + hlen:]
        expected_header = canonical_json(_source_payload(src)).encode("utf-8")
        if header_bytes != expected_header:
            raise DataError("source cache header does not match source spec",
                            location=src.sid, field="header")
        if len(body) != src.rows * 8:
            raise DataError(
                f"source cache row bytes {len(body)//8} != declared rows {src.rows}",
                location=src.sid, field="rows")
        raw = struct.unpack(f"<{src.rows}Q", body)
    except DataError:
        raise
    except (struct.error, ValueError) as exc:
        raise DataError(f"source cache unparseable: {exc}",
                        location=src.sid, field="cache") from exc
    column: list[int | None] = []
    for v in raw:
        if v == _NULL_SENTINEL:
            column.append(None)
        else:
            if not (INT32_MIN <= v <= INT32_MAX):
                raise DataError("source value outside 32-bit envelope",
                                location=src.sid, field="value")
            column.append(v)
    return column


def load_or_generate(workdir: str, src: SourceSpec) -> list[int | None]:
    path = column_path(workdir, src.sid)
    if os.path.exists(path):
        return read_column(path, src)
    column = generate_column(src)
    write_column(path, src, column)
    return column


def source_file_digest(src: SourceSpec) -> str:
    return content_digest(_source_payload(src))
