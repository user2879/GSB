from __future__ import annotations

from pathlib import Path
from typing import Any

from .codec import encode_block
from .errors import spec_error
from .models import (
    ENCODINGS,
    SEGMENT_FORMAT_VERSION,
    BlockSpec,
    ColumnDef,
    QuerySpec,
    RowGroup,
    SegmentSpec,
)
from .predicate import leaf_columns, parse_where
from .util import atomic_write_json, read_json


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _parse_columns(raw: Any) -> tuple[ColumnDef, ...]:
    if not isinstance(raw, list) or not raw:
        raise spec_error(
            "columns 必须为非空数组", block_id="<spec>", field="columns"
        )
    columns: list[ColumnDef] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise spec_error(
                "列定义必须为对象",
                block_id=f"<column:{index}>",
                field="columns",
            )
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise spec_error(
                "列名必须为非空字符串",
                block_id=f"<column:{index}>",
                field="name",
            )
        if name in names:
            raise spec_error(
                f"列名重复: {name}",
                block_id=f"<column:{name}>",
                field="name",
            )
        kind = item.get("type", "int")
        if kind != "int":
            raise spec_error(
                "当前格式仅支持 int 列",
                block_id=f"<column:{name}>",
                field="type",
            )
        names.add(name)
        columns.append(ColumnDef(name=name, kind=kind))
    return tuple(columns)


def _parse_row_groups(raw: Any) -> tuple[RowGroup, ...]:
    if not isinstance(raw, list) or not raw:
        raise spec_error(
            "row_groups 必须为非空数组", block_id="<spec>", field="row_groups"
        )
    groups: list[RowGroup] = []
    ids: set[str] = set()
    expected_start = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise spec_error(
                "行组必须为对象", block_id=f"<row_group:{index}>", field="row_groups"
            )
        group_id = item.get("id")
        if not isinstance(group_id, str) or not group_id.strip():
            raise spec_error(
                "行组 id 必须为非空字符串",
                block_id=f"<row_group:{index}>",
                field="id",
            )
        if group_id in ids:
            raise spec_error(
                f"行组 id 重复: {group_id}",
                block_id=f"<row_group:{group_id}>",
                field="id",
            )
        start = item.get("start_row")
        count = item.get("row_count")
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise spec_error(
                "start_row 必须为非负整数",
                block_id=f"<row_group:{group_id}>",
                field="start_row",
            )
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise spec_error(
                "row_count 必须为正整数",
                block_id=f"<row_group:{group_id}>",
                field="row_count",
            )
        if start != expected_start:
            raise spec_error(
                f"行组必须从 0 开始首尾相接且不重叠，期望 start_row={expected_start}",
                block_id=f"<row_group:{group_id}>",
                field="start_row",
            )
        ids.add(group_id)
        groups.append(RowGroup(id=group_id, start_row=start, row_count=count))
        expected_start = start + count
    return tuple(groups)


def _parse_block(
    raw: Any,
    *,
    index: int,
    columns: dict[str, ColumnDef],
    row_groups: dict[str, RowGroup],
) -> BlockSpec:
    if not isinstance(raw, dict):
        raise spec_error(
            "块必须为对象", block_id=f"<block:{index}>", field="blocks"
        )
    block_id_value = raw.get("id")
    block_id = (
        block_id_value
        if isinstance(block_id_value, str) and block_id_value.strip()
        else f"<block:{index}>"
    )
    if not isinstance(block_id_value, str) or not block_id_value.strip():
        raise spec_error("块 id 必须为非空字符串", block_id=block_id, field="id")
    column = raw.get("column")
    if not isinstance(column, str) or column not in columns:
        raise spec_error(
            f"块引用了不存在的列 {column!r}", block_id=block_id, field="column"
        )
    row_group = raw.get("row_group")
    if not isinstance(row_group, str) or row_group not in row_groups:
        raise spec_error(
            f"块引用了不存在的行组 {row_group!r}",
            block_id=block_id,
            field="row_group",
        )
    encoding = raw.get("encoding")
    if encoding not in ENCODINGS:
        raise spec_error(
            f"encoding 必须属于 {', '.join(ENCODINGS)}",
            block_id=block_id,
            field="encoding",
        )
    compressed = raw.get("compressed")
    if not isinstance(compressed, bool):
        raise spec_error(
            "compressed 必须为布尔值", block_id=block_id, field="compressed"
        )
    values_raw = raw.get("values")
    if not isinstance(values_raw, list):
        raise spec_error("values 必须为数组", block_id=block_id, field="values")
    values: list[int] = []
    for value in values_raw:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < -(2**63)
            or value > 2**63 - 1
        ):
            raise spec_error(
                "样本值必须为有符号 64 位整数",
                block_id=block_id,
                field="values",
            )
        values.append(value)
    if len(values) != row_groups[row_group].row_count:
        raise spec_error(
            f"样本值行数 {len(values)} 与行组行数 {row_groups[row_group].row_count} 不一致",
            block_id=block_id,
            field="values",
        )
    declared_pre = raw.get("uncompressed_bytes")
    declared_post = raw.get("compressed_bytes")
    if declared_pre is not None and not _is_nonnegative_int(declared_pre):
        raise spec_error(
            "uncompressed_bytes 必须为非负整数",
            block_id=block_id,
            field="uncompressed_bytes",
        )
    if declared_post is not None and not _is_nonnegative_int(declared_post):
        raise spec_error(
            "compressed_bytes 必须为非负整数",
            block_id=block_id,
            field="compressed_bytes",
        )
    encoded = encode_block(
        block_id=block_id,
        column=column,
        row_group=row_group,
        encoding=encoding,
        compressed=compressed,
        values=tuple(values),
    )
    if declared_pre is not None and declared_pre != encoded.pre_bytes:
        raise spec_error(
            f"uncompressed_bytes 声明 {declared_pre} 与实际 {encoded.pre_bytes} 不一致",
            block_id=block_id,
            field="uncompressed_bytes",
        )
    if declared_post is not None and declared_post != encoded.post_bytes:
        raise spec_error(
            f"compressed_bytes 声明 {declared_post} 与实际 {encoded.post_bytes} 不一致",
            block_id=block_id,
            field="compressed_bytes",
        )
    return BlockSpec(
        id=block_id,
        column=column,
        row_group=row_group,
        encoding=encoding,
        compressed=compressed,
        values=tuple(values),
        declared_pre_bytes=encoded.pre_bytes,
        declared_post_bytes=encoded.post_bytes,
    )


def _parse_queries(
    raw: Any,
    *,
    column_names: set[str],
) -> tuple[QuerySpec, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise spec_error(
            "queries 必须为数组", block_id="<spec>", field="queries"
        )
    queries: list[QuerySpec] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise spec_error(
                "查询必须为对象", block_id=f"<query:{index}>", field="queries"
            )
        query_id_value = item.get("id")
        query_id = (
            query_id_value
            if isinstance(query_id_value, str) and query_id_value.strip()
            else f"<query:{index}>"
        )
        if not isinstance(query_id_value, str) or not query_id_value.strip():
            raise spec_error(
                "查询 id 必须为非空字符串", block_id=f"query:{query_id}", field="id"
            )
        if query_id in ids:
            raise spec_error(
                f"查询 id 重复: {query_id}", block_id=f"query:{query_id}", field="id"
            )
        select_raw = item.get("select", item.get("columns"))
        if not isinstance(select_raw, list) or not select_raw:
            raise spec_error(
                "select 必须为非空数组", block_id=f"query:{query_id}", field="select"
            )
        select: list[str] = []
        for column in select_raw:
            if not isinstance(column, str) or column not in column_names:
                raise spec_error(
                    f"查询引用了不存在的列 {column!r}",
                    block_id=f"query:{query_id}",
                    field="select",
                )
            if column not in select:
                select.append(column)
        ast = parse_where(item.get("where"), query_id=query_id)
        for column in leaf_columns(ast):
            if column not in column_names:
                raise spec_error(
                    f"谓词引用了不存在的列 {column!r}",
                    block_id=f"query:{query_id}",
                    field="where",
                )
        ids.add(query_id)
        queries.append(
            QuerySpec(id=query_id, select=tuple(select), where=item.get("where"))
        )
    return tuple(queries)


def parse_spec(raw: Any) -> SegmentSpec:
    if not isinstance(raw, dict):
        raise spec_error("规范根节点必须为对象", block_id="<spec>", field="root")
    format_version = raw.get("format_version", SEGMENT_FORMAT_VERSION)
    if format_version != SEGMENT_FORMAT_VERSION:
        raise spec_error(
            f"仅支持 format_version={SEGMENT_FORMAT_VERSION}",
            block_id="<spec>",
            field="format_version",
        )
    segment_id = raw.get("segment_id")
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise spec_error(
            "segment_id 必须为非空字符串", block_id="<spec>", field="segment_id"
        )
    columns = _parse_columns(raw.get("columns"))
    column_map = {column.name: column for column in columns}
    row_groups = _parse_row_groups(raw.get("row_groups"))
    row_group_map = {group.id: group for group in row_groups}
    blocks_raw = raw.get("blocks")
    if not isinstance(blocks_raw, list) or not blocks_raw:
        raise spec_error("blocks 必须为非空数组", block_id="<spec>", field="blocks")
    blocks: list[BlockSpec] = []
    block_ids: set[str] = set()
    locations: dict[tuple[str, str], str] = {}
    for index, item in enumerate(blocks_raw):
        block = _parse_block(
            item, index=index, columns=column_map, row_groups=row_group_map
        )
        if block.id in block_ids:
            raise spec_error(
                f"块 id 重复: {block.id}", block_id=block.id, field="id"
            )
        location = (block.column, block.row_group)
        if location in locations:
            raise spec_error(
                f"列/行组存在重复块: {block.column}/{block.row_group}",
                block_id=block.id,
                field="row_group",
            )
        block_ids.add(block.id)
        locations[location] = block.id
        blocks.append(block)
    for column in columns:
        for group in row_groups:
            if (column.name, group.id) not in locations:
                missing_id = f"{column.name}@{group.id}"
                raise spec_error(
                    "缺少列/行组编码块",
                    block_id=missing_id,
                    field="blocks",
                )
    queries = _parse_queries(raw.get("queries", []), column_names=set(column_map))
    return SegmentSpec(
        segment_id=segment_id,
        columns=columns,
        row_groups=row_groups,
        blocks=tuple(blocks),
        queries=queries,
        format_version=format_version,
    )


def load_spec(path: str | Path) -> SegmentSpec:
    try:
        raw = read_json(path)
    except (OSError, ValueError) as error:
        raise spec_error(
            f"无法读取规范文件: {error}", block_id="<spec>", field="json"
        ) from error
    return parse_spec(raw)


def write_spec(spec: SegmentSpec, path: str | Path) -> None:
    atomic_write_json(path, spec.to_dict())
