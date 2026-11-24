from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .codec import decode_record, encode_block
from .errors import AppError
from .models import ENCODINGS, SEGMENT_FORMAT_VERSION, EncodedBlock, SegmentSpec
from .util import atomic_write_bytes, canonical_bytes, sha256_bytes, sha256_json


MAGIC = b"COLSEG\x01\n"


@dataclass(frozen=True)
class SegmentImage:
    header: dict
    blocks: dict[str, EncodedBlock]
    block_order: tuple[str, ...]
    file_digest: str
    spec_digest: str
    encoding_selection_version: int
    file_bytes: bytes

    def values(self, block_id: str) -> tuple[int, ...]:
        from .codec import decode_block

        return decode_block(self.blocks[block_id])


def spec_digest(spec: SegmentSpec) -> str:
    return sha256_json(spec.to_dict())


def _segment_error(message: str, *, block_id: str | None = None, field: str = "segment") -> AppError:
    return AppError("E_SEGMENT", message, block_id=block_id, field=field)


def _block_from_record(header_bytes: bytes, payload: bytes, *, expected_id: str) -> EncodedBlock:
    try:
        metadata = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _segment_error("块头不是合法 JSON", block_id=expected_id, field="header") from error
    block_id = str(metadata.get("block_id", expected_id))
    if metadata.get("encoding") not in ENCODINGS:
        raise _segment_error("块头编码方式非法", block_id=block_id, field="encoding")
    if not isinstance(metadata.get("compressed"), bool):
        raise _segment_error("块头压缩标记非法", block_id=block_id, field="compressed")
    for field in ("row_count", "uncompressed_bytes", "compressed_bytes"):
        value = metadata.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _segment_error("块头字节/行数字段必须为非负整数", block_id=block_id, field=field)
    if int(metadata["compressed_bytes"]) != len(payload):
        raise _segment_error(
            "块头 compressed_bytes 与实际负载长度不一致",
            block_id=block_id,
            field="compressed_bytes",
        )
    if not metadata["compressed"] and metadata["compressed_bytes"] != metadata["uncompressed_bytes"]:
        raise _segment_error(
            "未压缩块的压缩前后字节数不一致",
            block_id=block_id,
            field="compressed_bytes",
        )
    if not isinstance(metadata.get("column"), str) or not isinstance(metadata.get("row_group"), str):
        raise _segment_error("块头列/行组字段非法", block_id=block_id, field="column")
    try:
        _, values = decode_record(header_bytes, payload)
    except (ValueError, KeyError, UnicodeDecodeError) as error:
        raise _segment_error(f"块解码失败: {error}", block_id=block_id, field="payload") from error
    if len(values) != metadata["row_count"]:
        raise _segment_error("解码行数与块头不一致", block_id=block_id, field="row_count")
    digest = sha256_bytes(header_bytes + payload)
    record_bytes = (
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(payload))
        + payload
    )
    return EncodedBlock(
        block_id=block_id,
        column=str(metadata.get("column", "")),
        row_group=str(metadata.get("row_group", "")),
        encoding=str(metadata["encoding"]),
        compressed=bool(metadata["compressed"]),
        row_count=int(metadata["row_count"]),
        pre_bytes=int(metadata["uncompressed_bytes"]),
        post_bytes=int(metadata["compressed_bytes"]),
        payload=payload,
        header=metadata,
        header_bytes=header_bytes,
        block_digest=digest,
        record_bytes=record_bytes,
    )


def write_segment(
    path: str | Path,
    spec: SegmentSpec,
    blocks: Iterable[EncodedBlock],
    *,
    encoding_selection_version: int,
) -> SegmentImage:
    block_list = list(blocks)
    spec_map = spec.block_map
    if [block.block_id for block in block_list] != [block.id for block in spec.blocks]:
        raise _segment_error("写入块顺序或集合与规范不一致", field="blocks")
    for block in block_list:
        source = spec_map.get(block.block_id)
        if source is None:
            raise _segment_error("规范中不存在待写块", block_id=block.block_id, field="id")
        if (
            block.column != source.column
            or block.row_group != source.row_group
            or block.encoding != source.encoding
            or block.compressed != source.compressed
            or block.row_count != len(source.values)
        ):
            raise _segment_error("编码块元数据与规范不一致", block_id=block.block_id, field="metadata")
    document_digest = spec_digest(spec)
    header = {
        "format_version": SEGMENT_FORMAT_VERSION,
        "segment_id": spec.segment_id,
        "spec_digest": document_digest,
        "encoding_selection_version": int(encoding_selection_version),
        "block_count": len(block_list),
        "blocks": [
            {
                "block_id": block.block_id,
                "column": block.column,
                "row_group": block.row_group,
                "block_digest": block.block_digest,
            }
            for block in block_list
        ],
    }
    header_bytes = canonical_bytes(header)
    pieces: list[bytes] = [MAGIC, struct.pack("<I", len(header_bytes)), header_bytes]
    for block in block_list:
        pieces.append(struct.pack("<I", len(block.record_bytes)))
        pieces.append(block.record_bytes)
    body = b"".join(pieces)
    file_bytes = body + bytes.fromhex(sha256_bytes(body))
    atomic_write_bytes(path, file_bytes)
    return SegmentImage(
        header=header,
        blocks={block.block_id: block for block in block_list},
        block_order=tuple(block.block_id for block in block_list),
        file_digest=sha256_bytes(file_bytes),
        spec_digest=document_digest,
        encoding_selection_version=int(encoding_selection_version),
        file_bytes=file_bytes,
    )


def read_segment(path: str | Path) -> SegmentImage:
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise _segment_error(f"无法读取段文件: {error}", field="path") from error
    if len(data) < len(MAGIC) + 4 + 32:
        raise _segment_error("段文件长度不足", field="file")
    if data[: len(MAGIC)] != MAGIC:
        raise _segment_error("段文件 magic 不匹配", field="magic")
    body, footer = data[:-32], data[-32:]
    if sha256_bytes(body) != footer.hex():
        raise _segment_error("段文件整体摘要校验失败", field="file_digest")
    offset = len(MAGIC)
    header_length = struct.unpack_from("<I", body, offset)[0]
    offset += 4
    if offset + header_length > len(body):
        raise _segment_error("段头越界", field="header")
    header_bytes = body[offset : offset + header_length]
    offset += header_length
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _segment_error("段头不是合法 JSON", field="header") from error
    if header.get("format_version") != SEGMENT_FORMAT_VERSION:
        raise _segment_error("段格式版本不支持", field="format_version")
    block_entries = header.get("blocks")
    if not isinstance(block_entries, list):
        raise _segment_error("段头 blocks 必须为数组", field="blocks")
    blocks: dict[str, EncodedBlock] = {}
    block_order: list[str] = []
    for entry in block_entries:
        if not isinstance(entry, dict):
            raise _segment_error("段头块条目必须为对象", block_id="<segment>", field="blocks")
        expected_id = str(entry.get("block_id", "<unknown>"))
        if offset + 4 > len(body):
            raise _segment_error("块记录长度缺失", block_id=expected_id, field="record_length")
        record_length = struct.unpack_from("<I", body, offset)[0]
        offset += 4
        record = body[offset : offset + record_length]
        if len(record) != record_length:
            raise _segment_error("块记录越界", block_id=expected_id, field="record_length")
        offset += record_length
        if len(record) < 8:
            raise _segment_error("块记录长度不足", block_id=expected_id, field="record")
        block_header_length = struct.unpack_from("<I", record, 0)[0]
        if 4 + block_header_length + 4 > len(record):
            raise _segment_error("块头长度越界", block_id=expected_id, field="header_length")
        block_header_bytes = record[4 : 4 + block_header_length]
        payload_length_offset = 4 + block_header_length
        payload_length = struct.unpack_from("<I", record, payload_length_offset)[0]
        payload = record[payload_length_offset + 4 :]
        if len(payload) != payload_length:
            raise _segment_error("块负载长度不一致", block_id=expected_id, field="payload_length")
        block = _block_from_record(block_header_bytes, payload, expected_id=expected_id)
        if block.block_id != expected_id:
            raise _segment_error("段头与块头 block_id 不一致", block_id=expected_id, field="block_id")
        if block.block_id in blocks:
            raise _segment_error("块 id 重复", block_id=block.block_id, field="block_id")
        expected_digest = entry.get("block_digest")
        if expected_digest != block.block_digest:
            raise _segment_error("段头块摘要不一致", block_id=block.block_id, field="block_digest")
        blocks[block.block_id] = block
        block_order.append(block.block_id)
    if offset != len(body):
        raise _segment_error("段文件存在未消费尾部字节", field="file")
    if len(blocks) != header.get("block_count"):
        raise _segment_error("块数量与段头不一致", field="block_count")
    return SegmentImage(
        header=header,
        blocks=blocks,
        block_order=tuple(block_order),
        file_digest=sha256_bytes(data),
        spec_digest=str(header.get("spec_digest", "")),
        encoding_selection_version=int(header.get("encoding_selection_version", 0)),
        file_bytes=data,
    )


def encode_spec_blocks(spec: SegmentSpec) -> dict[str, EncodedBlock]:
    result: dict[str, EncodedBlock] = {}
    for source in spec.blocks:
        result[source.id] = encode_block(
            block_id=source.id,
            column=source.column,
            row_group=source.row_group,
            encoding=source.encoding,
            compressed=source.compressed,
            values=source.values,
        )
    return result
