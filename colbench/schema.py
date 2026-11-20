"""段定义（schema）与块表的加载和强校验。

本地 JSON 结构示例见 ``samples/segment_seg_a.json``。校验失败时收集全部
问题后以 ``E_SCHEMA`` 整体拒绝，问题中携带出错列与块序号。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .codecs import DICT, ENCODINGS, RLE, RAW, STR, TYPES
from .errors import Issue, SchemaError

SCHEMA_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    col_type: str
    compressed: bool
    encoding: str  # 缺省编码，块级可覆盖


@dataclass(frozen=True)
class BlockSpec:
    """块表中的一行：某行组内某列的一个编码块。"""

    row_group: int
    column: str
    block_index: int
    encoding: str
    compressed: bool
    values: list


@dataclass(frozen=True)
class RowGroupSpec:
    index: int
    start: int  # 全局行号，含
    end: int    # 全局行号，不含

    @property
    def row_count(self) -> int:
        return self.end - self.start


@dataclass
class SegmentSpec:
    segment_id: str
    format_version: int
    columns: list[ColumnSpec]
    row_groups: list[RowGroupSpec]
    blocks: list[BlockSpec] = field(default_factory=list)
    source_path: Path | None = None

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def column(self, name: str) -> ColumnSpec:
        return next(column for column in self.columns
                    if column.name == name)

    def blocks_of(self, row_group: int, column: str) -> list[BlockSpec]:
        return [block for block in self.blocks
                if block.row_group == row_group and block.column == column]

    def all_block_keys(self) -> list[tuple[int, str, int]]:
        return [(block.row_group, block.column, block.block_index)
                for block in self.blocks]


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_segment_spec(path: str | Path) -> SegmentSpec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaError(f"段定义文件不存在：{path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"段定义 JSON 解析失败：{exc}",
                          {"path": str(path)}) from exc
    spec = parse_segment_spec(raw)
    spec.source_path = path
    return spec


def parse_segment_spec(raw: dict) -> SegmentSpec:
    issues: list[Issue] = []

    segment_id = raw.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id:
        issues.append(Issue("E_SCHEMA", "segment_id 必须为非空字符串"))
        segment_id = segment_id or "<invalid>"

    format_version = raw.get("format_version", SCHEMA_FORMAT_VERSION)
    if not _is_positive_int(format_version):
        issues.append(Issue("E_SCHEMA", "format_version 必须为正整数"))

    # ---- 列定义 -----------------------------------------------------------
    columns: list[ColumnSpec] = []
    raw_columns = raw.get("columns", [])
    if not isinstance(raw_columns, list) or not raw_columns:
        issues.append(Issue("E_SCHEMA", "columns 必须为非空数组"))
        raw_columns = []
    seen_names: set[str] = set()
    for raw_column in raw_columns:
        name = raw_column.get("name") if isinstance(raw_column, dict) else None
        if not isinstance(name, str) or not name:
            issues.append(Issue("E_SCHEMA", "列名必须为非空字符串"))
            continue
        if name in seen_names:
            issues.append(Issue("E_SCHEMA", f"列名 {name!r} 重复",
                                column=name))
            continue
        seen_names.add(name)
        col_type = raw_column.get("type", "int64")
        if col_type not in TYPES:
            issues.append(Issue("E_SCHEMA",
                                f"列类型 {col_type!r} 非法，允许 {TYPES}",
                                column=name))
            continue
        encoding = raw_column.get("encoding", RAW)
        if encoding not in ENCODINGS:
            issues.append(Issue("E_SCHEMA",
                                f"编码 {encoding!r} 非法，允许 {ENCODINGS}",
                                column=name))
        compressed = bool(raw_column.get("compressed", False))
        columns.append(ColumnSpec(name=name, col_type=col_type,
                                  compressed=compressed, encoding=encoding))
    column_map = {column.name: column for column in columns}

    # ---- 行组区间 ---------------------------------------------------------
    row_groups: list[RowGroupSpec] = []
    raw_groups = raw.get("row_groups", [])
    if not isinstance(raw_groups, list) or not raw_groups:
        issues.append(Issue("E_SCHEMA", "row_groups 必须为非空数组"))
        raw_groups = []
    expected_start = 0
    for position, raw_group in enumerate(raw_groups):
        index = raw_group.get("index", position) if isinstance(raw_group, dict) \
            else position
        start = raw_group.get("start") if isinstance(raw_group, dict) else None
        end = raw_group.get("end") if isinstance(raw_group, dict) else None
        if not _is_nonneg_int(start) or not _is_nonneg_int(end) or end <= start:
            issues.append(Issue("E_SCHEMA",
                                f"行组 {index} 区间非法：start={start}, end={end}",
                                row_group=index))
            continue
        if start != expected_start:
            issues.append(Issue(
                "E_SCHEMA",
                f"行组 {index} 未首尾相接：期望 start={expected_start}，"
                f"实际 start={start}", row_group=index))
        row_groups.append(RowGroupSpec(index=index, start=start, end=end))
        expected_start = end
    total_rows = expected_start

    # ---- 块表 -------------------------------------------------------------
    blocks: list[BlockSpec] = []
    raw_blocks = raw.get("blocks", [])
    if not isinstance(raw_blocks, list) or not raw_blocks:
        issues.append(Issue("E_SCHEMA", "blocks 必须为非空数组"))
        raw_blocks = []
    seen_block_keys: set[tuple[int, str, int]] = set()
    group_by_index = {group.index: group for group in row_groups}
    for position, raw_block in enumerate(raw_blocks):
        row_group = raw_block.get("row_group")
        column = raw_block.get("column")
        block_index = raw_block.get("block_index", position)
        ident = (row_group if isinstance(row_group, int) else -1,
                 column if isinstance(column, str) else "<bad>",
                 block_index if isinstance(block_index, int) else -1)
        if not isinstance(row_group, int) or row_group not in group_by_index:
            issues.append(Issue("E_SCHEMA",
                                f"块引用了未知行组 {row_group!r}",
                                column=column, block_index=block_index,
                                row_group=row_group))
            continue
        if column not in column_map:
            issues.append(Issue("E_SCHEMA",
                                f"块引用了未知列 {column!r}", column=column,
                                block_index=block_index, row_group=row_group))
            continue
        if not isinstance(block_index, int) or block_index < 0:
            issues.append(Issue("E_SCHEMA", "block_index 必须为非负整数",
                                column=column, row_group=row_group))
            continue
        if ident in seen_block_keys:
            issues.append(Issue("E_SCHEMA",
                                f"块 (行组={row_group}, 列={column}, "
                                f"块序号={block_index}) 重复",
                                column=column, block_index=block_index,
                                row_group=row_group))
            continue
        seen_block_keys.add(ident)

        encoding = raw_block.get("encoding",
                                 column_map[column].encoding)
        if encoding not in ENCODINGS:
            issues.append(Issue("E_SCHEMA",
                                f"块编码 {encoding!r} 非法，允许 {ENCODINGS}",
                                column=column, block_index=block_index,
                                row_group=row_group))
        compressed = bool(raw_block.get(
            "compressed", column_map[column].compressed))
        values = raw_block.get("values")
        if not isinstance(values, list) or not values:
            issues.append(Issue("E_SCHEMA", "块 values 必须为非空数组",
                                column=column, block_index=block_index,
                                row_group=row_group))
            continue
        col_type = column_map[column].col_type
        type_issue = _check_value_types(values, col_type, column,
                                        block_index, row_group)
        if type_issue is not None:
            issues.append(type_issue)
            continue
        group = group_by_index[row_group]
        if len(values) != group.row_count:
            issues.append(Issue(
                "E_SCHEMA",
                f"块行数 {len(values)} 与行组行数 {group.row_count} 不一致",
                column=column, block_index=block_index, row_group=row_group))
            continue
        # 压缩前后字节数只允许出现在块表元数据里且必须为正整数
        pre = raw_block.get("pre_bytes")
        post = raw_block.get("post_bytes")
        for label, number in (("pre_bytes", pre), ("post_bytes", post)):
            if number is not None and not _is_positive_int(number):
                issues.append(Issue(
                    "E_SCHEMA", f"{label} 必须为正整数，实际 {number!r}",
                    column=column, block_index=block_index,
                    row_group=row_group))
        blocks.append(BlockSpec(row_group=row_group, column=column,
                                block_index=block_index, encoding=encoding,
                                compressed=compressed, values=values))

    # 每个行组 × 每列必须恰好一个块（块序号从 0 连续）
    for group in row_groups:
        for column in columns:
            owned = [block.block_index for block in blocks
                     if block.row_group == group.index
                     and block.column == column.name]
            if not owned:
                issues.append(Issue(
                    "E_SCHEMA",
                    f"行组 {group.index} 列 {column.name!r} 缺少编码块",
                    column=column.name, row_group=group.index))
            elif owned != [0]:
                issues.append(Issue(
                    "E_SCHEMA",
                    f"行组 {group.index} 列 {column.name!r} 块序号必须为 [0]",
                    column=column.name, row_group=group.index))

    if total_rows <= 0:
        issues.append(Issue("E_SCHEMA", "段总行数必须为正整数"))

    if issues:
        raise SchemaError(
            f"段定义强校验失败，共 {len(issues)} 个问题",
            {"segment_id": segment_id,
             "issues": [issue.to_dict() for issue in issues]})

    return SegmentSpec(segment_id=segment_id,
                       format_version=format_version,
                       columns=columns, row_groups=row_groups, blocks=blocks)


def _check_value_types(values: list, col_type: str, column: str,
                       block_index: int, row_group: int) -> Issue | None:
    for offset, value in enumerate(values):
        if col_type == "int64":
            if not isinstance(value, int) or isinstance(value, bool):
                return Issue("E_SCHEMA",
                             f"值 #{offset} 不是 int64：{value!r}",
                             column=column, block_index=block_index,
                             row_group=row_group)
            if not (-(1 << 63) <= value < (1 << 63)):
                return Issue("E_SCHEMA",
                             f"值 #{offset} 超出 int64 范围：{value}",
                             column=column, block_index=block_index,
                             row_group=row_group)
        elif not isinstance(value, str):
            return Issue("E_SCHEMA",
                         f"值 #{offset} 不是 string：{value!r}",
                         column=column, block_index=block_index,
                         row_group=row_group)
    return None


def write_segment_spec(spec: SegmentSpec, path: str | Path) -> None:
    """把段定义（含确定性样本值）写回 JSON，键顺序固定。"""
    payload = {
        "segment_id": spec.segment_id,
        "format_version": spec.format_version,
        "columns": [
            {"name": column.name, "type": column.col_type,
             "encoding": column.encoding,
             "compressed": column.compressed}
            for column in spec.columns],
        "row_groups": [
            {"index": group.index, "start": group.start, "end": group.end}
            for group in spec.row_groups],
        "blocks": [
            {"row_group": block.row_group, "column": block.column,
             "block_index": block.block_index,
             "encoding": block.encoding,
             "compressed": block.compressed, "values": block.values}
            for block in sorted(spec.blocks,
                                key=lambda item: (item.row_group,
                                                  item.column,
                                                  item.block_index))],
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8")
