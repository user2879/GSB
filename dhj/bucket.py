"""Versioned binary bucket files (v1/v2/v3) with strict decode validation."""

import os
import struct
from dataclasses import dataclass, field

from . import fixedpoint as fp
from .errors import BucketError
from .hashing import digest_hex_64, row_digest
from .jsonio import dumps_canonical, sha256_bytes, write_atomic

MAGIC = b"DHJBUCK1"
SUPPORTED_VERSIONS = (1, 2, 3)
CURRENT_VERSION = 3

SEC_SUMMARY = 1
SEC_COLUMNS = 2
SEC_KEYS = 3
SEC_ORDER = 4
SECTION_TAGS = {
    SEC_SUMMARY: "summary",
    SEC_COLUMNS: "columns",
    SEC_KEYS: "keys",
    SEC_ORDER: "order",
}
REQUIRED_SECTIONS = {
    1: (SEC_SUMMARY,),
    2: (SEC_SUMMARY, SEC_COLUMNS),
    3: (SEC_SUMMARY, SEC_COLUMNS, SEC_KEYS, SEC_ORDER),
}
HEADER_FIXED = struct.Struct(">8sHHI")
SECTION_HEAD = struct.Struct(">II")
U64 = struct.Struct(">Q")
I64 = struct.Struct(">q")
D64 = struct.Struct(">d")
I32 = struct.Struct(">i")
I16 = struct.Struct(">H")


@dataclass
class BucketRow:
    row_id: int
    digest: int
    values: dict
    null_bitmap: int
    q_key: int = fp.NULL_Q


@dataclass
class Bucket:
    side: str
    table_name: str
    partition: int
    num_partitions: int
    join_key: str
    fraction_bits: int
    version: int
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)

    def row_count(self):
        return len(self.rows)

    def ordered_rows(self):
        return sorted(
            self.rows,
            key=lambda r: (r.q_key == fp.NULL_Q, r.q_key, r.row_id),
        )

    def digest_list(self):
        return [r.digest for r in self.ordered_rows()]

    def fingerprint_payload(self):
        return {
            "side": self.side,
            "table": self.table_name,
            "partition": self.partition,
            "num_partitions": self.num_partitions,
            "join_key": self.join_key,
            "fraction_bits": self.fraction_bits,
            "version": self.version,
            "row_count": self.row_count(),
            "digests": [digest_hex_64(r.digest) for r in self.ordered_rows()],
            "q_keys": [r.q_key for r in self.ordered_rows()],
        }

    def fingerprint(self):
        return sha256_bytes(dumps_canonical(self.fingerprint_payload()))


def bucket_filename(partition):
    return f"bucket-p{partition:05d}.dhj"


def _header(bucket):
    payload = {
        "side": bucket.side,
        "table": bucket.table_name,
        "partition": bucket.partition,
        "num_partitions": bucket.num_partitions,
        "join_key": bucket.join_key,
        "fraction_bits": bucket.fraction_bits,
    }
    return dumps_canonical(payload)


def _encode_summary(rows_ordered):
    parts = [U64.pack(len(rows_ordered))]
    for row in rows_ordered:
        parts.append(I64.pack(row.row_id))
        parts.append(U64.pack(row.digest))
    return b"".join(parts)


def _encode_columns(bucket, rows_ordered):
    parts = [I16.pack(len(bucket.columns))]
    for col in bucket.columns:
        name = col.name.encode("utf-8")
        parts.append(I16.pack(len(name)))
        parts.append(name)
        parts.append(b"I" if col.dtype == "int" else b"F")
        parts.append(U64.pack(len(rows_ordered)))
        for row in rows_ordered:
            value = row.values[col.name]
            if value is None:
                parts.append(b"\x00")
                parts.append(b"\x00" * 8)
            else:
                parts.append(b"\x01")
                parts.append(
                    I64.pack(value) if col.dtype == "int" else D64.pack(float(value))
                )
    return b"".join(parts)


def _encode_keys(rows_ordered):
    parts = [U64.pack(len(rows_ordered))]
    for row in rows_ordered:
        parts.append(I64.pack(row.q_key))
    return b"".join(parts)


def _encode_order(rows_ordered):
    parts = [U64.pack(len(rows_ordered))]
    for index, row in enumerate(rows_ordered):
        parts.append(I64.pack(row.row_id))
        parts.append(U64.pack(index))
    return b"".join(parts)


def _section(tag, body):
    return SECTION_HEAD.pack(tag, len(body)) + body


def encode_bucket(bucket):
    if bucket.version not in SUPPORTED_VERSIONS:
        raise BucketError(
            f"unsupported bucket version {bucket.version}",
            version=bucket.version,
        )
    ordered = bucket.ordered_rows()
    header_bytes = _header(bucket)
    out = [
        MAGIC,
        struct.pack(">H", bucket.version),
        I32.pack(len(header_bytes)),
        header_bytes,
    ]
    out.append(_section(SEC_SUMMARY, _encode_summary(ordered)))
    if bucket.version >= 2:
        out.append(_section(SEC_COLUMNS, _encode_columns(bucket, ordered)))
    if bucket.version >= 3:
        out.append(_section(SEC_KEYS, _encode_keys(ordered)))
        out.append(_section(SEC_ORDER, _encode_order(ordered)))
    return b"".join(out)


class _Reader:
    def __init__(self, data, path):
        self.data = data
        self.path = path
        self.pos = 0

    def take(self, size, what):
        if self.pos + size > len(self.data):
            raise BucketError(
                f"truncated bucket while reading {what} at offset {self.pos}",
                path=self.path,
                offset=self.pos,
            )
        chunk = self.data[self.pos:self.pos + size]
        self.pos += size
        return chunk

    def unpack(self, st, what):
        return st.unpack(self.take(st.size, what))[0]

    def unpack_tuple(self, st, what):
        return st.unpack(self.take(st.size, what))


def _read_sections(reader, version, path):
    sections = {}
    end = len(reader.data)
    while reader.pos < end:
        section_start = reader.pos
        tag, length = reader.unpack_tuple(SECTION_HEAD, "section header")
        if tag not in SECTION_TAGS:
            raise BucketError(
                f"unknown section tag {tag} at offset {section_start}",
                path=path,
                version=version,
                offset=section_start,
            )
        if tag in sections:
            raise BucketError(
                f"duplicate section {SECTION_TAGS[tag]} at offset {section_start}",
                path=path,
                version=version,
                offset=section_start,
            )
        body = reader.take(length, f"section {SECTION_TAGS[tag]} body")
        sections[tag] = (body, section_start)
    if reader.pos != end:
        raise BucketError(
            "length fields do not cover the whole bucket",
            path=path,
            version=version,
            offset=reader.pos,
        )
    required = REQUIRED_SECTIONS[version]
    missing = [SECTION_TAGS[t] for t in required if t not in sections]
    if missing:
        raise BucketError(
            f"version {version} bucket missing sections: {','.join(missing)}",
            path=path,
            version=version,
        )
    extra = [
        SECTION_TAGS[t]
        for t in sections
        if (t in (SEC_COLUMNS,) and version < 2)
        or (t in (SEC_KEYS, SEC_ORDER) and version < 3)
    ]
    if extra:
        raise BucketError(
            f"version {version} bucket contains future sections: {','.join(extra)}",
            path=path,
            version=version,
        )
    return sections


def _decode_summary(body, path, version):
    if len(body) < U64.size:
        raise BucketError(
            "summary section too short for row count",
            path=path,
            version=version,
            offset=0,
        )
    count = U64.unpack(body[: U64.size])[0]
    expected = U64.size + count * (I64.size + U64.size)
    if len(body) != expected:
        raise BucketError(
            "summary length field is inconsistent with row count",
            path=path,
            version=version,
            offset=0,
        )
    rows = []
    pos = U64.size
    for _ in range(count):
        row_id = I64.unpack(body[pos:pos + I64.size])[0]
        pos += I64.size
        digest = U64.unpack(body[pos:pos + U64.size])[0]
        pos += U64.size
        rows.append(BucketRow(row_id=row_id, digest=digest, values={},
                              null_bitmap=0))
    return rows


def _decode_columns(body, path, version, rows):
    if len(body) < I16.size:
        raise BucketError(
            "columns section too short for column count",
            path=path,
            version=version,
        )
    pos = 0
    col_count = I16.unpack(body[pos:pos + I16.size])[0]
    pos += I16.size
    columns = []
    for _ in range(col_count):
        if pos + I16.size > len(body):
            raise BucketError(
                "truncated column name length", path=path, version=version,
                offset=pos,
            )
        name_len = I16.unpack(body[pos:pos + I16.size])[0]
        pos += I16.size
        if pos + name_len + 1 + U64.size > len(body):
            raise BucketError(
                "truncated column metadata", path=path, version=version,
                offset=pos,
            )
        name = body[pos:pos + name_len].decode("utf-8")
        pos += name_len
        type_flag = body[pos]
        pos += 1
        if type_flag not in (ord("I"), ord("F")):
            raise BucketError(
                f"unknown column type flag {type_flag}",
                path=path,
                version=version,
                offset=pos,
            )
        dtype = "int" if type_flag == ord("I") else "float"
        columns.append(_Col(name, dtype))
        count = U64.unpack(body[pos:pos + U64.size])[0]
        pos += U64.size
        if count != len(rows):
            raise BucketError(
                f"column {name} row count {count} != summary row count {len(rows)}",
                path=path,
                version=version,
                offset=pos,
            )
        stride = 1 + 8
        if pos + count * stride > len(body):
            raise BucketError(
                f"truncated column data for {name}",
                path=path,
                version=version,
                offset=pos,
            )
        for row in rows:
            present = body[pos]
            pos += 1
            raw = body[pos:pos + 8]
            pos += 8
            if present == 1:
                row.values[name] = (
                    I64.unpack(raw)[0] if dtype == "int" else D64.unpack(raw)[0]
                )
            else:
                row.values[name] = None
    if pos != len(body):
        raise BucketError(
            "column section has trailing bytes inconsistent with length fields",
            path=path,
            version=version,
            offset=pos,
        )
    return columns


class _Col:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype


def _decode_keys(body, path, rows):
    if len(body) != U64.size + len(rows) * I64.size:
        raise BucketError(
            "key section length inconsistent with row count",
            path=path,
            version=3,
        )
    count = U64.unpack(body[: U64.size])[0]
    if count != len(rows):
        raise BucketError(
            f"key row count {count} != summary row count {len(rows)}",
            path=path,
            version=3,
        )
    pos = U64.size
    for row in rows:
        row.q_key = I64.unpack(body[pos:pos + I64.size])[0]
        pos += I64.size


def _decode_order(body, path, rows, fraction_bits, table_name, join_key):
    if len(body) != U64.size + len(rows) * (I64.size + U64.size):
        raise BucketError(
            "order section length inconsistent with row count",
            path=path,
            version=3,
        )
    count = U64.unpack(body[: U64.size])[0]
    if count != len(rows):
        raise BucketError(
            f"order row count {count} != summary row count {len(rows)}",
            path=path,
            version=3,
        )
    expected = sorted(
        rows, key=lambda r: (r.q_key == fp.NULL_Q, r.q_key, r.row_id)
    )
    pos = U64.size
    seen = set()
    for index, row in enumerate(rows):
        row_id = I64.unpack(body[pos:pos + I64.size])[0]
        pos += I64.size
        ordinal = U64.unpack(body[pos:pos + U64.size])[0]
        pos += U64.size
        if ordinal != index:
            raise BucketError(
                f"order table ordinal {ordinal} != position {index}",
                path=path,
                version=3,
            )
        if row_id in seen:
            raise BucketError(
                f"order table repeats row id {row_id}",
                path=path,
                version=3,
            )
        seen.add(row_id)
        if row_id != expected[index].row_id:
            raise BucketError(
                "order table does not match (quantized key, row id) ordering",
                path=path,
                version=3,
            )
    if seen != {r.row_id for r in rows}:
        raise BucketError(
            "order table row ids do not cover summary rows",
            path=path,
            version=3,
        )


def decode_bucket(data, path="<bytes>"):
    """Strictly decode a whole bucket. No partial result is returned on error."""
    if not isinstance(data, (bytes, bytearray)):
        raise BucketError("bucket payload must be bytes", path=path)
    if len(data) < HEADER_FIXED.size:
        raise BucketError(
            "bucket shorter than fixed header", path=path, offset=0
        )
    reader = _Reader(bytes(data), path)
    magic = reader.take(8, "magic")
    if magic != MAGIC:
        raise BucketError(
            f"bad magic marker {magic!r}", path=path, offset=0
        )
    version = reader.unpack(struct.Struct(">H"), "version")
    if version not in SUPPORTED_VERSIONS:
        raise BucketError(
            f"unsupported bucket version {version}",
            path=path,
            version=version,
        )
    header_len = reader.unpack(I32, "header length")
    if header_len <= 0 or reader.pos + header_len > len(data):
        raise BucketError(
            "header length field out of range",
            path=path,
            version=version,
            offset=reader.pos,
        )
    try:
        header = read_json_from_bytes(reader.take(header_len, "header"))
    except Exception as exc:
        raise BucketError(
            f"header JSON invalid: {exc}",
            path=path,
            version=version,
            offset=reader.pos,
        )
    sections = _read_sections(reader, version, path)
    rows = _decode_summary(sections[SEC_SUMMARY][0], path, version)
    columns = []
    if version >= 2:
        columns = _decode_columns(
            sections[SEC_COLUMNS][0], path, version, rows
        )
        _verify_summary_digests(rows, columns, header, path, version)
    if version >= 3:
        _decode_keys(sections[SEC_KEYS][0], path, rows)
        _decode_order(
            sections[SEC_ORDER][0], path, rows,
            header["fraction_bits"], header["table"], header["join_key"],
        )
        _verify_keys(rows, columns, header, path)
    bucket = Bucket(
        side=header["side"],
        table_name=header["table"],
        partition=header["partition"],
        num_partitions=header["num_partitions"],
        join_key=header["join_key"],
        fraction_bits=header["fraction_bits"],
        version=version,
        rows=rows,
        columns=columns,
    )
    return bucket


def read_json_from_bytes(raw):
    import json

    return json.loads(raw.decode("utf-8"))


def _verify_summary_digests(rows, columns, header, path, version):
    for row in rows:
        null_bit = 0 if any(row.values[c.name] is None for c in columns) else 1
        want = row_digest(
            header["table"],
            row.row_id,
            header["side"],
            header["join_key"],
            row.values,
            null_bit,
        )
        if want != row.digest:
            raise BucketError(
                f"row summary mismatch for row id {row.row_id}",
                path=path,
                version=version,
            )
        row.null_bitmap = null_bit


def _verify_keys(rows, columns, header, path):
    join_key = header["join_key"]
    fraction_bits = header["fraction_bits"]
    by_name = {c.name: c for c in columns}
    if join_key not in by_name:
        raise BucketError(
            f"join key column {join_key} absent from column section",
            path=path,
            version=3,
        )
    for row in rows:
        value = row.values[join_key]
        want = fp.quantize(value, fraction_bits)
        if want != row.q_key:
            raise BucketError(
                f"quantized key mismatch for row id {row.row_id}",
                path=path,
                version=3,
            )


def write_bucket_file(path, bucket, make_backup=False, mode=0o644):
    """Encode and atomically write a bucket, optionally backing up the old one."""
    data = encode_bucket(bucket)
    if make_backup and os.path.exists(path):
        backup_path = backup_path_for(path, bucket.version)
        if not os.path.exists(backup_path):
            with open(path, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
        os.chmod(backup_path, 0o444)
    write_atomic(path, data, mode=mode)
    return sha256_bytes(data)


def backup_path_for(path, new_version):
    return f"{path}.bak-v{new_version}"


def read_bucket_file(path):
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise BucketError(f"cannot read bucket: {exc}", path=path)
    return decode_bucket(data, path=path)


def make_bucket(side, table, join_key, partition, num_partitions, fraction_bits,
                row_ids, version=CURRENT_VERSION):
    rows = []
    for row_id in row_ids:
        values = table.rows[row_id]
        if values[join_key] is None:
            continue
        null_bit = table.null_bitmap[row_id]
        digest = row_digest(
            table.name, row_id, side, join_key, values, null_bit
        )
        q_key = fp.quantize(values[join_key], fraction_bits)
        rows.append(
            BucketRow(
                row_id=row_id,
                digest=digest,
                values=dict(values),
                null_bitmap=null_bit,
                q_key=q_key,
            )
        )
    return Bucket(
        side=side,
        table_name=table.name,
        partition=partition,
        num_partitions=num_partitions,
        join_key=join_key,
        fraction_bits=fraction_bits,
        version=version,
        rows=rows,
        columns=list(table.columns),
    )


def upgrade_bucket(bucket, target_version=CURRENT_VERSION, source_table=None,
                   join_key=None):
    """Record-level in-memory upgrade. Returns (new_bucket, added_fields).

    Upgrading a v1 bucket (summaries only) requires the validated source
    table so column data and quantized keys can be reconstructed; every v1
    row summary is matched against the source before fields are filled in.
    """
    if target_version not in SUPPORTED_VERSIONS:
        raise BucketError(
            f"unsupported target bucket version {target_version}",
            version=target_version,
        )
    if bucket.version > target_version:
        raise BucketError(
            f"cannot downgrade bucket from v{bucket.version} to v{target_version}",
            version=bucket.version,
        )
    added = []
    if bucket.version == target_version:
        return bucket, added
    if bucket.version == 1 and target_version >= 2:
        added.extend(["columns", "row_count"])
    if target_version >= 3 and bucket.version < 3:
        added.extend(["quantized_join_keys", "row_order_table"])
    upgraded = Bucket(
        side=bucket.side,
        table_name=bucket.table_name,
        partition=bucket.partition,
        num_partitions=bucket.num_partitions,
        join_key=bucket.join_key,
        fraction_bits=bucket.fraction_bits,
        version=target_version,
        rows=[
            BucketRow(
                row_id=r.row_id,
                digest=r.digest,
                values=dict(r.values),
                null_bitmap=r.null_bitmap,
                q_key=r.q_key,
            )
            for r in bucket.rows
        ],
        columns=list(bucket.columns),
    )
    if source_table is not None:
        source_by_id = {
            idx: (row, source_table.null_bitmap[idx])
            for idx, row in enumerate(source_table.rows)
        }
        for row in upgraded.rows:
            if row.row_id not in source_by_id:
                raise BucketError(
                    f"v1 row id {row.row_id} absent from source table",
                    version=bucket.version,
                )
            values, null_bit = source_by_id[row.row_id]
            rebuilt_digest = row_digest(
                source_table.name, row.row_id, bucket.side,
                join_key or bucket.join_key, values, null_bit,
            )
            if rebuilt_digest != row.digest:
                raise BucketError(
                    f"v1 row summary mismatch for row id {row.row_id}",
                    version=bucket.version,
                )
            row.values = dict(values)
            row.null_bitmap = null_bit
        upgraded.columns = list(source_table.columns)
    if target_version >= 3:
        if not upgraded.columns:
            raise BucketError(
                "cannot build v3 key section without source columns",
                version=bucket.version,
            )
        for row in upgraded.rows:
            row.q_key = fp.quantize(
                row.values[bucket.join_key], bucket.fraction_bits
            )
    return upgraded, added


def first_digest_mismatch(rows_a, rows_b):
    """Locate the first row id whose summary differs between ordered lists."""
    for index, (a, b) in enumerate(zip(rows_a, rows_b)):
        if a.row_id != b.row_id or a.digest != b.digest:
            return index, a.row_id if a.row_id == b.row_id else (a.row_id, b.row_id)
    if len(rows_a) != len(rows_b):
        return min(len(rows_a), len(rows_b)), None
    return None


def bucket_status(path):
    """Return ('unknown', None) for unreadable/partial buckets, never raises."""
    try:
        bucket = read_bucket_file(path)
    except (BucketError, OSError, ValueError, KeyError, struct.error):
        return "unknown", None
    return f"v{bucket.version}", bucket
