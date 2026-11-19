"""紧凑二进制帧格式。

帧布局（偏移从 0 起）：
  0  : magic "GB"            (2 字节)
  2  : schema_version 标记    (1 字节)
  3  : 字节序标记             (1 字节, 0x01=小端, 0x02=大端)
  4  : 索引宽度               (1 字节, 2 或 4 字节)
  5  : 字段位图               (1 字节, bit0..bit5, 高两位必须为 0)
  6..: 按位图 bit 顺序排列的字段：
       tick / local_seq / base_snapshot_version -> 索引宽度字节整数
       peer_id / payload(json 文本)             -> 长度(索引宽度) + UTF-8 字节
       op_type                                  -> 1 字节枚举
"""

from __future__ import annotations

import json
import struct

from ..errors import E_FORMAT_FIELD, PipelineError
from ..model import (
    CURRENT_SCHEMA_VERSION,
    ID_TO_OP_TYPE,
    OP_TYPE_TO_ID,
    Event,
)

MAGIC = b"GB"
ENDIAN_LITTLE = 0x01
ENDIAN_BIG = 0x02
VALID_WIDTHS = (2, 4)

BIT_TICK = 0x01
BIT_PEER_ID = 0x02
BIT_LOCAL_SEQ = 0x04
BIT_OP_TYPE = 0x08
BIT_PAYLOAD = 0x10
BIT_BASE_VERSION = 0x20
KNOWN_BITS = (
    BIT_TICK | BIT_PEER_ID | BIT_LOCAL_SEQ | BIT_OP_TYPE | BIT_PAYLOAD | BIT_BASE_VERSION
)

# 位图未声明的可选字段（解析后由迁移表补齐）
OPTIONAL_WHEN_MISSING = ("local_seq",)


def _fmt_error(message: str, offset: int, field: str) -> PipelineError:
    return PipelineError(E_FORMAT_FIELD, message, offset=offset, field=field)


def encode_frame(event: Event, *, endian: int = ENDIAN_LITTLE, width: int = 4,
                 bitmap: int = KNOWN_BITS) -> bytes:
    if endian not in (ENDIAN_LITTLE, ENDIAN_BIG):
        raise ValueError("bad endian marker")
    if width not in VALID_WIDTHS:
        raise ValueError("bad index width")
    bo = "<" if endian == ENDIAN_LITTLE else ">"
    out = bytearray()
    out += MAGIC
    out.append(event.schema_version & 0xFF)
    out.append(endian)
    out.append(width)
    out.append(bitmap & 0xFF)

    def put_int(value: int) -> None:
        fmt = {2: "H", 4: "I"}[width]
        out.extend(struct.pack(bo + fmt, value))

    def put_str(text: str) -> None:
        raw = text.encode("utf-8")
        put_int(len(raw))
        out.extend(raw)

    if bitmap & BIT_TICK:
        put_int(event.tick)
    if bitmap & BIT_PEER_ID:
        put_str(event.peer_id)
    if bitmap & BIT_LOCAL_SEQ:
        put_int(event.local_seq)
    if bitmap & BIT_OP_TYPE:
        out.append(OP_TYPE_TO_ID[event.op_type])
    if bitmap & BIT_PAYLOAD:
        put_str(json.dumps(event.payload, sort_keys=True, ensure_ascii=False))
    if bitmap & BIT_BASE_VERSION:
        put_int(event.base_snapshot_version)
    return bytes(out)


def decode_frame(frame: bytes) -> Event:
    if len(frame) < 6:
        raise _fmt_error("帧长度不足帧头 6 字节", len(frame), "header")
    if frame[0:2] != MAGIC:
        raise _fmt_error("magic 不匹配", 0, "magic")

    schema_version = frame[2]
    endian = frame[3]
    width = frame[4]
    bitmap = frame[5]

    if endian not in (ENDIAN_LITTLE, ENDIAN_BIG):
        raise _fmt_error(f"非法字节序标记 0x{endian:02x}", 3, "endianness")
    if width not in VALID_WIDTHS:
        raise _fmt_error(f"非法索引宽度 {width}", 4, "index_width")
    if bitmap & ~KNOWN_BITS:
        raise _fmt_error(f"字段位图含未定义位 0x{bitmap & ~KNOWN_BITS:02x}", 5, "field_bitmap")
    for required, name in (
        (BIT_TICK, "tick"),
        (BIT_PEER_ID, "peer_id"),
        (BIT_OP_TYPE, "op_type"),
        (BIT_PAYLOAD, "payload"),
        (BIT_BASE_VERSION, "base_snapshot_version"),
    ):
        if not bitmap & required:
            raise _fmt_error(f"位图未声明必需字段 {name}", 5, "field_bitmap")

    bo = "<" if endian == ENDIAN_LITTLE else ">"
    offset = 6

    def take_int(field: str) -> int:
        nonlocal offset
        end = offset + width
        if end > len(frame):
            raise _fmt_error(f"字段 {field} 整数越界（宽度 {width}）", offset, field)
        fmt = {2: "H", 4: "I"}[width]
        (value,) = struct.unpack_from(bo + fmt, frame, offset)
        offset = end
        return value

    def take_str(field: str) -> str:
        nonlocal offset
        length = take_int(field)
        end = offset + length
        if end > len(frame):
            raise _fmt_error(f"字段 {field} 字符串长度 {length} 越界", offset, field)
        raw = frame[offset:end]
        offset = end
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _fmt_error(f"字段 {field} 非 UTF-8: {exc}", offset - length, field)

    tick = take_int("tick") if bitmap & BIT_TICK else None
    peer_id = take_str("peer_id") if bitmap & BIT_PEER_ID else None
    local_seq = take_int("local_seq") if bitmap & BIT_LOCAL_SEQ else None
    if bitmap & BIT_OP_TYPE:
        if offset >= len(frame):
            raise _fmt_error("op_type 枚举越界", offset, "op_type")
        op_id = frame[offset]
        offset += 1
        if op_id not in ID_TO_OP_TYPE:
            raise _fmt_error(f"未知 op_type 枚举 {op_id}", offset - 1, "op_type")
        op_type = ID_TO_OP_TYPE[op_id]
    else:
        op_type = None
    payload_raw = take_str("payload") if bitmap & BIT_PAYLOAD else None
    base_version = take_int("base_snapshot_version") if bitmap & BIT_BASE_VERSION else None

    if offset != len(frame):
        raise _fmt_error(f"帧尾部存在 {len(frame) - offset} 个未声明字节", offset, "trailing")

    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
        raise _fmt_error(f"payload 非合法 JSON: {exc}", offset, "payload")
    if not isinstance(payload, dict):
        raise _fmt_error("payload 必须为 JSON 对象", offset, "payload")

    event = Event(
        tick=tick,
        peer_id=peer_id,
        local_seq=local_seq,  # 位图未声明时为 None，由迁移表补齐
        op_type=op_type,
        payload=payload,
        base_snapshot_version=base_version,
        schema_version=schema_version,
        payload_encoding=None,  # 帧内不携带，按版本标记由迁移表补齐
        time_base=None,
        source_format="binary",
    )
    return event
