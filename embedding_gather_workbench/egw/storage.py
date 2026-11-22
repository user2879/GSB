"""Embedding 表存储、加载校验、内容指纹与编辑命令。"""
import hashlib
import json
import math

import numpy as np

from .errors import (
    E_DIM_MISMATCH,
    E_NONFINITE_VECTOR,
    E_ROW_INVALID,
    failure,
)


def sha256_hex(parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        digest.update(part)
        digest.update(b"\x1e")
    return digest.hexdigest()


class Table:
    def __init__(self, dim):
        self.dim = dim
        self.rows = {}          # row_id -> Row
        self.block_offsets = {}  # block_id -> {offset: row_id}

    def copy(self):
        other = Table(self.dim)
        for rid, row in self.rows.items():
            other.rows[rid] = Row(row.row_id, row.block_id, row.offset, row.vector.copy())
        for bid, mapping in self.block_offsets.items():
            other.block_offsets[bid] = dict(mapping)
        return other


class Row:
    def __init__(self, row_id, block_id, offset, vector):
        self.row_id = row_id
        self.block_id = block_id
        self.offset = offset
        self.vector = np.asarray(vector, dtype=np.float64)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_rows(path, dim):
    """逐行解析行记录 JSONL，返回 (Table, 失败清单)。"""
    table = Table(dim)
    failures = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    failure(E_ROW_INVALID, f"JSON 解析失败: {exc}", source=path, line=line_no)
                )
                continue
            row_id = record.get("row_id")
            block_id = record.get("block_id")
            offset = record.get("offset")
            vector = record.get("vector")
            if row_id is None or block_id is None or offset is None or vector is None:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        "缺少 row_id/block_id/offset/vector 字段",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            if not _is_int(block_id) or block_id < 0 or not _is_int(offset) or offset < 0:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        "block_id 与 offset 必须为非负整数",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            if not isinstance(vector, list) or not vector or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector
            ):
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        "vector 必须是非空数值数组",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            if len(vector) != dim:
                failures.append(
                    failure(
                        E_DIM_MISMATCH,
                        f"向量维度 {len(vector)} 与配置维度 {dim} 不一致",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            if not all(math.isfinite(float(x)) for x in vector):
                failures.append(
                    failure(
                        E_NONFINITE_VECTOR,
                        "向量含 NaN/Inf 等非有限元素",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            if row_id in seen_ids:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        f"词 ID {row_id} 重复",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            slot = table.block_offsets.setdefault(block_id, {})
            if offset in slot:
                failures.append(
                    failure(
                        E_ROW_INVALID,
                        f"(block_id={block_id}, offset={offset}) 组合冲突",
                        source=path,
                        line=line_no,
                        row_id=row_id,
                    )
                )
                continue
            seen_ids.add(row_id)
            row = Row(row_id, block_id, offset, np.asarray(vector, dtype=np.float64))
            table.rows[row_id] = row
            slot[offset] = row_id
    return table, failures


def content_fingerprint(vector):
    """行内容指纹：仅由 float64 原始字节派生。"""
    return sha256_hex([np.asarray(vector, dtype=np.float64).tobytes()])


def row_fingerprint(vector, int_vector, quant_level, dim):
    return sha256_hex(
        [
            content_fingerprint(vector),
            np.asarray(int_vector, dtype=np.int64).tobytes(),
            str(quant_level),
            str(dim),
        ]
    )


def block_fingerprint(table, block_id, quant_cache):
    """块指纹：行内偏移升序的行指纹有序列表 + 块布局。"""
    mapping = table.block_offsets[block_id]
    parts = [f"block={block_id}"]
    for offset in sorted(mapping):
        rid = mapping[offset]
        parts.append(f"off={offset}")
        parts.append(quant_cache.rows[rid]["row_fingerprint"])
    return sha256_hex(parts)


def all_block_fingerprints(table, quant_cache):
    result = {}
    for block_id in sorted(table.block_offsets):
        result[block_id] = block_fingerprint(table, block_id, quant_cache)
    return result


def block_layout_fingerprint(block_fps):
    """块布局指纹：块 ID 升序拼接块指纹。"""
    parts = []
    for block_id in sorted(block_fps):
        parts.append(f"block={block_id}:{block_fps[block_id]}")
    return sha256_hex(parts)


def apply_edit(table, edit):
    """在表副本上应用单条编辑命令，返回新 Table。"""
    new_table = table.copy()
    kind = edit["op"]

    def _check_vector(vector):
        if not isinstance(vector, list) or len(vector) != new_table.dim:
            raise ValueError(f"编辑向量维度非法: {edit}")
        if not all(math.isfinite(float(x)) for x in vector):
            raise ValueError(f"编辑向量含非有限元素: {edit}")

    if kind == "replace_row":
        row_id = edit["row_id"]
        block_id = edit["block_id"]
        offset = edit["offset"]
        _check_vector(edit["vector"])
        old = new_table.rows.get(row_id)
        if old is not None:
            del new_table.block_offsets[old.block_id][old.offset]
            if not new_table.block_offsets[old.block_id]:
                del new_table.block_offsets[old.block_id]
        slot = new_table.block_offsets.setdefault(block_id, {})
        if offset in slot and slot[offset] != row_id:
            raise ValueError(f"目标槽位 (block={block_id}, offset={offset}) 已被占用")
        slot[offset] = row_id
        new_table.rows[row_id] = Row(row_id, block_id, offset, np.asarray(edit["vector"], np.float64))

    elif kind == "replace_block":
        block_id = edit["block_id"]
        rows = edit["rows"]
        victim_ids = set()
        if block_id in new_table.block_offsets:
            victim_ids = set(new_table.block_offsets[block_id].values())
            del new_table.block_offsets[block_id]
        for rid in victim_ids:
            del new_table.rows[rid]
        slot = new_table.block_offsets.setdefault(block_id, {})
        for item in rows:
            rid = item["row_id"]
            offset = item["offset"]
            _check_vector(item["vector"])
            if offset in slot:
                raise ValueError(f"块 {block_id} 内 offset={offset} 冲突")
            existing = new_table.rows.get(rid)
            if existing is not None and existing.block_id != block_id:
                raise ValueError(f"行 {rid} 已属于块 {existing.block_id}，不能跨块重复定义")
            slot[offset] = rid
            new_table.rows[rid] = Row(rid, block_id, offset, np.asarray(item["vector"], np.float64))

    elif kind == "append_block":
        block_id = edit["block_id"]
        if block_id in new_table.block_offsets:
            raise ValueError(f"追加块 {block_id} 已存在")
        slot = new_table.block_offsets.setdefault(block_id, {})
        for item in edit["rows"]:
            rid = item["row_id"]
            offset = item["offset"]
            _check_vector(item["vector"])
            if rid in new_table.rows:
                raise ValueError(f"追加块中行 {rid} 已存在")
            if offset in slot:
                raise ValueError(f"块 {block_id} 内 offset={offset} 冲突")
            slot[offset] = rid
            new_table.rows[rid] = Row(rid, block_id, offset, np.asarray(item["vector"], np.float64))

    else:
        raise ValueError(f"未知编辑命令 {kind}")
    return new_table
