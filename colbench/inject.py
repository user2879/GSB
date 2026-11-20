"""异常注入：在段文件/预算层面制造确定性故障。

支持三类注入，业务码与任务要求一一对应：

* ``truncate``：截断某个块的载荷 -> 解码时 ``E_PAYLOAD``；
* ``encoding_mismatch``：仅篡改块头编码声明（载荷保持另一种编码）->
  解码时 ``E_ENCODING``；
* ``zero_memory``：把每列解码内存预算置零 -> 调度入口 ``E_BUDGET``。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .codecs import ENCODINGS
from .errors import EncodingError
from .segment import FILE_MAGIC, _Reader
from .versioning import BudgetParams


@dataclass(frozen=True)
class Injection:
    kind: str
    segment_id: str
    row_group: int
    column: str
    block_index: int = 0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "segment_id": self.segment_id,
                "row_group": self.row_group, "column": self.column,
                "block_index": self.block_index}


def inject_payload_truncation(path: str | Path, row_group: int,
                              column: str, *, cut_bytes: int = 8) -> Injection:
    """在段文件中截断目标块的载荷尾部（同步破坏摘要）。"""
    path = Path(path)
    data = bytearray(path.read_bytes())
    offset, payload_length, _ = _locate_block(data, row_group, column)
    payload_start = offset
    new_length = max(1, payload_length - cut_bytes)
    struct.pack_into("<I", data, offset - 4, new_length)
    del data[payload_start + new_length: payload_start + payload_length]
    path.write_bytes(bytes(data))
    return Injection("truncate", path.stem, row_group, column)


def inject_encoding_mismatch(path: str | Path, row_group: int,
                             column: str) -> Injection:
    """把块头编码声明改成另一种编码，但载荷保持原样。"""
    path = Path(path)
    data = bytearray(path.read_bytes())
    payload_offset, _, encoding_pos = _locate_block(
        data, row_group, column)
    current = data[encoding_pos]
    replacement = (current + 1) % len(ENCODINGS)
    data[encoding_pos] = replacement
    path.write_bytes(bytes(data))
    return Injection("encoding_mismatch", path.stem, row_group, column)


def inject_zero_memory(params: BudgetParams) -> BudgetParams:
    return BudgetParams(global_parallelism=params.global_parallelism,
                        per_column_memory=0,
                        per_segment_concurrency=params.per_segment_concurrency)


def _locate_block(data: bytes, row_group: int,
                  column: str) -> tuple[int, int]:
    """返回目标块载荷起始偏移与载荷长度。"""
    reader = _Reader(data)
    if reader.take(4) != FILE_MAGIC:
        raise EncodingError("不是合法段文件，无法注入")
    reader.u16()
    reader.text()
    for _ in range(reader.u32()):
        reader.text()
        reader.take(2)
    for _ in range(reader.u32()):
        reader.u32()
        reader.u32()
        reader.u32()
    block_count = reader.u32()
    for _ in range(block_count):
        block_row_group = reader.u32()
        block_column = reader.text()
        block_index = reader.u32()
        encoding_pos = reader.pos
        reader.take(1)  # encoding
        reader.take(1)  # type
        reader.u32()  # row_count
        reader.u32()  # pre_bytes
        reader.u32()  # post_bytes
        reader.take(1)  # compressed
        reader.text()  # digest
        reader.text()  # min
        reader.text()  # max
        reader.u32()  # distinct
        payload_length = reader.u32()
        payload_start = reader.pos
        reader.take(payload_length)
        if (block_row_group == row_group and block_column == column
                and block_index == 0):
            return payload_start, payload_length, encoding_pos
    raise EncodingError("注入目标块不存在",
                        {"row_group": row_group, "column": column})
