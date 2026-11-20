"""列式段文件的写入、读取与块表持久化。

段文件（``*.cseg``）布局::

    magic "CSEG" + uint16 format_version + uint32 segment_id_len + id
    uint32 列数；每列：长度前缀列名 + 类型标记 + 压缩标记
    uint32 行组数；每组：index/start/end
    uint32 块数；每块：块头 + 长度前缀载荷
    结束标记 "CEND"

块头记录编码方式、整数行数、压缩前/压缩后字节数、摘要与区域图
（min/max/不同值数），满足“块头记录整数编码方式、整数行数、压缩前与
压缩后字节数”的要求。
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from . import codecs
from .codecs import ENCODINGS, STR, TYPES, decode_block, encode_block
from .errors import EncodingError, PayloadError
from .schema import BlockSpec, SegmentSpec, load_segment_spec

FILE_MAGIC = b"CSEG"
FILE_END = b"CEND"
SEGMENT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BlockHeader:
    row_group: int
    column: str
    block_index: int
    encoding: str
    col_type: str
    row_count: int
    pre_bytes: int
    post_bytes: int
    compressed: bool
    digest: str
    min_value: object
    max_value: object
    distinct_count: int


@dataclass
class Segment:
    spec: SegmentSpec
    path: Path
    headers: dict[tuple[int, str, int], BlockHeader]
    payloads: dict[tuple[int, str, int], bytes]

    # -- 便捷查询 --------------------------------------------------------
    def header(self, row_group: int, column: str,
               block_index: int = 0) -> BlockHeader:
        return self.headers[(row_group, column, block_index)]

    def read_block(self, row_group: int, column: str,
                   block_index: int = 0, *, expect_digest: bool = True) -> list:
        key = (row_group, column, block_index)
        header = self.headers[key]
        return decode_block(
            self.payloads[key],
            expect_encoding=header.encoding,
            expect_type=header.col_type,
            expect_digest=header.digest if expect_digest else None)

    def zone_map(self, row_group: int, column: str,
                 block_index: int = 0) -> tuple[object, object]:
        header = self.header(row_group, column, block_index)
        return header.min_value, header.max_value

    def digest_overview(self) -> dict:
        """段集合级摘要所需的稳定字节视图。"""
        items = []
        for key in sorted(self.headers):
            header = self.headers[key]
            items.append("|".join([
                str(key[0]), key[1], str(key[2]), header.encoding,
                header.col_type, str(header.row_count),
                str(header.pre_bytes), str(header.post_bytes),
                str(header.compressed), header.digest,
                str(header.min_value), str(header.max_value),
                str(header.distinct_count)]))
        return {"segment_id": self.spec.segment_id,
                "format_version": SEGMENT_FORMAT_VERSION,
                "block_digests": items}


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------
def _put_u32(buf: bytearray, value: int) -> None:
    buf.extend(struct.pack("<I", value))


def _put_str(buf: bytearray, text: str) -> None:
    blob = text.encode("utf-8")
    _put_u32(buf, len(blob))
    buf.extend(blob)


def build_segment(spec: SegmentSpec, directory: str | Path) -> Segment:
    """根据段定义编码所有块并写入本地段文件与块表 JSON。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.segment_id}.cseg"
    table_path = directory / f"{spec.segment_id}.blocks.json"

    out = bytearray()
    out.extend(FILE_MAGIC)
    out.extend(struct.pack("<H", SEGMENT_FORMAT_VERSION))
    _put_str(out, spec.segment_id)

    _put_u32(out, len(spec.columns))
    for column in spec.columns:
        _put_str(out, column.name)
        out.append(TYPES.index(column.col_type))
        out.append(1 if column.compressed else 0)

    _put_u32(out, len(spec.row_groups))
    for group in spec.row_groups:
        _put_u32(out, group.index)
        _put_u32(out, group.start)
        _put_u32(out, group.end)

    headers: dict[tuple[int, str, int], BlockHeader] = {}
    payloads: dict[tuple[int, str, int], bytes] = {}
    ordered_blocks = sorted(spec.blocks,
                            key=lambda item: (item.row_group, item.column,
                                              item.block_index))
    _put_u32(out, len(ordered_blocks))
    table_rows = []
    for block in ordered_blocks:
        column = spec.column(block.column)
        if column.col_type == STR and any(not isinstance(v, str)
                                          for v in block.values):
            raise EncodingError("字符串列出现非字符串值",
                                {"column": block.column})
        encoded = encode_block(block.values, block.encoding,
                               compressed=block.compressed)
        if encoded.col_type != column.col_type:
            raise EncodingError("块值类型与列定义不一致",
                                {"column": block.column,
                                 "declared": column.col_type,
                                 "actual": encoded.col_type})
        distinct = len(set(block.values))
        header = BlockHeader(
            row_group=block.row_group, column=block.column,
            block_index=block.block_index, encoding=block.encoding,
            col_type=column.col_type, row_count=encoded.row_count,
            pre_bytes=encoded.pre_bytes, post_bytes=encoded.post_bytes,
            compressed=block.compressed, digest=encoded.digest,
            min_value=min(block.values), max_value=max(block.values),
            distinct_count=distinct)
        key = (block.row_group, block.column, block.block_index)
        headers[key] = header
        payloads[key] = encoded.payload
        _write_block_header(out, header)
        _put_u32(out, len(encoded.payload))
        out.extend(encoded.payload)
        table_rows.append(_header_row(header))

    out.extend(FILE_END)
    path.write_bytes(bytes(out))
    table_path.write_text(
        json.dumps({"segment_id": spec.segment_id,
                    "format_version": SEGMENT_FORMAT_VERSION,
                    "blocks": table_rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return Segment(spec=spec, path=path, headers=headers, payloads=payloads)


def _write_block_header(buf: bytearray, header: BlockHeader) -> None:
    _put_u32(buf, header.row_group)
    _put_str(buf, header.column)
    _put_u32(buf, header.block_index)
    buf.append(ENCODINGS.index(header.encoding))
    buf.append(TYPES.index(header.col_type))
    _put_u32(buf, header.row_count)
    _put_u32(buf, header.pre_bytes)
    _put_u32(buf, header.post_bytes)
    buf.append(1 if header.compressed else 0)
    _put_str(buf, header.digest)
    _put_str(buf, str(header.min_value))
    _put_str(buf, str(header.max_value))
    _put_u32(buf, header.distinct_count)


def _header_row(header: BlockHeader) -> dict:
    return {
        "row_group": header.row_group, "column": header.column,
        "block_index": header.block_index, "encoding": header.encoding,
        "col_type": header.col_type, "row_count": header.row_count,
        "pre_bytes": header.pre_bytes, "post_bytes": header.post_bytes,
        "compressed": header.compressed, "digest": header.digest,
        "min_value": header.min_value, "max_value": header.max_value,
        "distinct_count": header.distinct_count,
    }


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        if self.pos + size > len(self.data):
            raise PayloadError("段文件被截断",
                               {"offset": self.pos, "need": size,
                                "size": len(self.data)})
        chunk = self.data[self.pos:self.pos + size]
        self.pos += size
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def text(self) -> str:
        length = self.u32()
        return self.take(length).decode("utf-8")


def _coerce(value_text: str, col_type: str):
    if col_type == "int64":
        return int(value_text)
    return value_text


def load_segment(spec_path: str | Path, *, directory: str | Path | None = None,
                 spec: SegmentSpec | None = None) -> Segment:
    """从段文件读取块头与载荷，并用段定义交叉强校验。"""
    resolved_spec = spec or load_segment_spec(spec_path)
    directory = Path(directory) if directory else \
        (resolved_spec.source_path.parent if resolved_spec.source_path
         else Path("."))
    path = directory / f"{resolved_spec.segment_id}.cseg"
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise PayloadError(f"段文件不存在：{path}") from exc
    if len(data) < 6 or data[:4] != FILE_MAGIC:
        raise PayloadError("段文件魔数损坏", {"path": str(path)})
    reader = _Reader(data)
    reader.take(4)
    format_version = reader.u16()
    if format_version != SEGMENT_FORMAT_VERSION:
        raise EncodingError("段文件格式版本不受支持",
                            {"expected": SEGMENT_FORMAT_VERSION,
                             "actual": format_version})
    segment_id = reader.text()
    if segment_id != resolved_spec.segment_id:
        raise EncodingError("段文件与段定义的 segment_id 不一致",
                            {"file": segment_id,
                             "spec": resolved_spec.segment_id})

    column_count = reader.u32()
    file_columns = []
    for _ in range(column_count):
        name = reader.text()
        col_type = TYPES[reader.take(1)[0]]
        compressed = reader.take(1)[0] == 1
        file_columns.append((name, col_type, compressed))

    group_count = reader.u32()
    file_groups = []
    for _ in range(group_count):
        index = reader.u32()
        start = reader.u32()
        end = reader.u32()
        file_groups.append((index, start, end))

    block_count = reader.u32()
    headers = {}
    payloads = {}
    for _ in range(block_count):
        header, payload = _read_block(reader)
        key = (header.row_group, header.column, header.block_index)
        if key in headers:
            raise PayloadError("段文件内出现重复块",
                               {"key": list(key)})
        headers[key] = header
        payloads[key] = payload

    if reader.take(4) != FILE_END:
        raise PayloadError("段文件结束标记损坏")
    if reader.pos != len(data):
        raise PayloadError("段文件存在尾部多余字节",
                           {"consumed": reader.pos, "size": len(data)})

    _cross_validate(resolved_spec, file_columns, file_groups, headers)
    return Segment(spec=resolved_spec, path=path,
                   headers=headers, payloads=payloads)


def _read_block(reader: _Reader) -> tuple[BlockHeader, bytes]:
    row_group = reader.u32()
    column = reader.text()
    block_index = reader.u32()
    encoding = ENCODINGS[reader.take(1)[0]]
    col_type = TYPES[reader.take(1)[0]]
    row_count = reader.u32()
    pre_bytes = reader.u32()
    post_bytes = reader.u32()
    compressed = reader.take(1)[0] == 1
    digest = reader.text()
    min_text = reader.text()
    max_text = reader.text()
    distinct_count = reader.u32()
    payload_length = reader.u32()
    payload = reader.take(payload_length)
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != digest:
        raise PayloadError("块摘要不匹配",
                           {"column": column, "block_index": block_index,
                            "expected": digest, "actual": actual_digest})
    header = BlockHeader(
        row_group=row_group, column=column, block_index=block_index,
        encoding=encoding, col_type=col_type, row_count=row_count,
        pre_bytes=pre_bytes, post_bytes=post_bytes, compressed=compressed,
        digest=digest, min_value=_coerce(min_text, col_type),
        max_value=_coerce(max_text, col_type),
        distinct_count=distinct_count)
    return header, payload


def _cross_validate(spec: SegmentSpec, file_columns, file_groups,
                    headers: dict[tuple[int, str, int], BlockHeader]) -> None:
    expected_columns = [(column.name, column.col_type, column.compressed)
                        for column in spec.columns]
    if file_columns != expected_columns:
        raise EncodingError("段文件列定义与 JSON 不一致",
                            {"file": file_columns, "spec": expected_columns})
    expected_groups = [(group.index, group.start, group.end)
                       for group in spec.row_groups]
    if file_groups != expected_groups:
        raise PayloadError("段文件行组划分与 JSON 不一致",
                           {"file": file_groups, "spec": expected_groups})
    for key, spec_block in {(b.row_group, b.column, b.block_index): b
                            for b in spec.blocks}.items():
        if key not in headers:
            raise PayloadError("段文件缺少 JSON 声明的块", {"key": list(key)})
        header = headers[key]
        if header.encoding != spec_block.encoding \
                or header.compressed != spec_block.compressed:
            raise EncodingError("块头编码/压缩标记与 JSON 不一致",
                                {"key": list(key), "header": header.encoding,
                                 "spec": spec_block.encoding})
        if header.row_count != len(spec_block.values):
            raise PayloadError("块头行数与样本值数量不一致",
                               {"key": list(key)})
