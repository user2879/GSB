"""Columnar block formats b1/b2/b3: encoding, decoding, strong validation, upgrade.

On-disk payload is canonical JSON (sorted keys, tight separators, trailing LF).

  b1: {format, block_id, row_count, columns:[{name, values}]}
      - only column identifiers, row count and raw values; no types, no nulls.
  b2: + columns:[{name, type, null_bitmap(hex), values}]
  b3: + quant_scale_exp (session scale = 10**exp) and batch_max_rows.

Upgrade rules (deterministic, so upgraded bytes match a directly-written b3
of the same logical data):
  b1 -> types inferred (all-int values => int64, any float => float64),
        null bitmap all-valid, quant_scale_exp / batch_max_rows = defaults.
  b2 -> quant_scale_exp / batch_max_rows = defaults.
"""

import hashlib
import json
import struct
from dataclasses import dataclass, field

from .errors import FormatError

CURRENT_FORMAT = "b3"
KNOWN_FORMATS = ("b1", "b2", "b3")
SUPPORTED_TYPES = ("int64", "float64")

DEFAULT_QUANT_SCALE_EXP = 3
DEFAULT_BATCH_MAX_ROWS = 1024

INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def bitmap_len_for(row_count):
    return (row_count + 7) // 8


def make_bitmap(null_rows, row_count):
    bmp = bytearray(bitmap_len_for(row_count))
    for r in null_rows:
        bmp[r // 8] |= 1 << (r % 8)
    return bytes(bmp)


def bitmap_null_rows(bmp):
    rows = []
    for r in range(len(bmp) * 8):
        if bmp[r // 8] >> (r % 8) & 1:
            rows.append(r)
    return rows


@dataclass
class Column:
    name: str
    dtype: str
    null_bitmap: bytes
    values: list  # python ints or floats; value at null rows is 0 / 0.0 placeholder


@dataclass
class Block:
    block_id: str
    row_count: int
    columns: list = field(default_factory=list)
    quant_scale_exp: int = DEFAULT_QUANT_SCALE_EXP
    batch_max_rows: int = DEFAULT_BATCH_MAX_ROWS
    format_version: str = CURRENT_FORMAT

    def column(self, name):
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


def _fail(block_id, field_path, msg):
    raise FormatError(msg, block_id=block_id, field=field_path)


def _require(block_id, obj, key, ctx):
    if not isinstance(obj, dict) or key not in obj:
        _fail(block_id, ctx, f"missing required field {ctx!r}")
    return obj[key]


def decode_block(text):
    """Parse + strongly validate a block payload. Raises FormatError (E_FORMAT)."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatError("payload is not valid JSON", block_id="<unknown>",
                          field="<json>", detail=str(exc))
    if not isinstance(doc, dict):
        raise FormatError("payload root must be an object", block_id="<unknown>", field="<root>")
    block_id = doc.get("block_id", "<unknown>")
    fmt = _require(block_id, doc, "format", "format")
    if fmt not in KNOWN_FORMATS:
        _fail(block_id, "format", f"unknown format version {fmt!r}")
    row_count = _require(block_id, doc, "row_count", "row_count")
    if not isinstance(row_count, int) or row_count < 0:
        _fail(block_id, "row_count", "row_count must be a non-negative integer")
    raw_cols = _require(block_id, doc, "columns", "columns")
    if not isinstance(raw_cols, list) or not raw_cols:
        _fail(block_id, "columns", "columns must be a non-empty array")

    columns = []
    for i, rc in enumerate(raw_cols):
        base = f"columns[{i}]"
        name = _require(block_id, rc, "name", f"{base}.name")
        values = _require(block_id, rc, "values", f"{base}.values")
        if not isinstance(values, list) or len(values) != row_count:
            _fail(block_id, f"{base}.values",
                  f"values length {len(values) if isinstance(values, list) else '?'} "
                  f"!= row_count {row_count}")
        if fmt == "b1":
            dtype = None
            bmp = bytes(bitmap_len_for(row_count))
        else:
            dtype = _require(block_id, rc, "type", f"{base}.type")
            if dtype not in SUPPORTED_TYPES:
                _fail(block_id, f"{base}.type", f"unsupported column type {dtype!r}")
            hexmap = _require(block_id, rc, "null_bitmap", f"{base}.null_bitmap")
            try:
                bmp = bytes.fromhex(hexmap)
            except ValueError:
                _fail(block_id, f"{base}.null_bitmap", "null_bitmap is not valid hex")
            want = bitmap_len_for(row_count)
            if len(bmp) != want:
                _fail(block_id, f"{base}.null_bitmap",
                      f"null_bitmap length {len(bmp)} != ceil(row_count/8) {want}")
        for r, v in enumerate(values):
            if v is None:
                if fmt == "b1":
                    _fail(block_id, f"{base}.values",
                          f"b1 cannot represent nulls (row {r})")
                if not (bmp[r // 8] >> (r % 8) & 1):
                    _fail(block_id, f"{base}.values",
                          f"null value at row {r} not marked in null_bitmap")
                values[r] = 0 if dtype == "int64" else 0.0
            elif not isinstance(v, (int, float)) or isinstance(v, bool):
                _fail(block_id, f"{base}.values", f"row {r}: value must be a number")
        if fmt == "b1":
            dtype = "float64" if any(isinstance(v, float) for v in values) else "int64"
        columns.append(Column(name=name, dtype=dtype, null_bitmap=bmp, values=list(values)))

    quant_scale_exp = DEFAULT_QUANT_SCALE_EXP
    batch_max_rows = DEFAULT_BATCH_MAX_ROWS
    if fmt == "b3":
        quant_scale_exp = _require(block_id, doc, "quant_scale_exp", "quant_scale_exp")
        batch_max_rows = _require(block_id, doc, "batch_max_rows", "batch_max_rows")
        if not isinstance(quant_scale_exp, int) or quant_scale_exp < 0:
            _fail(block_id, "quant_scale_exp", "quant_scale_exp must be a non-negative integer")
        if not isinstance(batch_max_rows, int) or batch_max_rows <= 0:
            _fail(block_id, "batch_max_rows", "batch_max_rows must be a positive integer")

    return Block(block_id=block_id, row_count=row_count, columns=columns,
                 quant_scale_exp=quant_scale_exp, batch_max_rows=batch_max_rows,
                 format_version=fmt)


def encode_block(block, version=None):
    version = version or block.format_version
    cols = []
    for c in block.columns:
        entry = {"name": c.name}
        if version != "b1":
            entry["type"] = c.dtype
            entry["null_bitmap"] = c.null_bitmap.hex()
        vals = []
        for r, v in enumerate(c.values):
            is_null = c.null_bitmap[r // 8] >> (r % 8) & 1
            vals.append(None if (is_null and version != "b1") else v)
        entry["values"] = vals
        cols.append(entry)
    doc = {"format": version, "block_id": block.block_id,
           "row_count": block.row_count, "columns": cols}
    if version == "b3":
        doc["quant_scale_exp"] = block.quant_scale_exp
        doc["batch_max_rows"] = block.batch_max_rows
    return canonical_json(doc)


def upgrade_block(block):
    """Return an equivalent block at CURRENT_FORMAT (deterministic defaults)."""
    if block.format_version == CURRENT_FORMAT:
        return block
    return Block(block_id=block.block_id, row_count=block.row_count,
                 columns=[Column(c.name, c.dtype, c.null_bitmap, list(c.values))
                          for c in block.columns],
                 quant_scale_exp=DEFAULT_QUANT_SCALE_EXP,
                 batch_max_rows=DEFAULT_BATCH_MAX_ROWS,
                 format_version=CURRENT_FORMAT)


def content_bytes(block):
    """Canonical binary serialization of logical content (for digests/divergence)."""
    out = bytearray(b"COLVAL3\x00")
    out += struct.pack("<I", len(block.block_id)) + block.block_id.encode()
    out += struct.pack("<qii", block.row_count, block.quant_scale_exp, block.batch_max_rows)
    out += struct.pack("<I", len(block.columns))
    for c in block.columns:
        out += struct.pack("<I", len(c.name)) + c.name.encode()
        out += b"\x01" if c.dtype == "int64" else b"\x02"
        out += struct.pack("<I", len(c.null_bitmap)) + c.null_bitmap
        if c.dtype == "int64":
            for v in c.values:
                out += struct.pack("<q", int(v))
        else:
            for v in c.values:
                out += struct.pack("<d", float(v))
    return bytes(out)


def content_digest(block):
    return hashlib.sha256(content_bytes(block)).hexdigest()
