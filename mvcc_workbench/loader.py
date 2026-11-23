"""Strict local JSON/NDJSON specification loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import CycleError, SpecError
from .model import (
    EXTERNAL_OUTCOMES,
    OUTCOME_COMMITTED,
    OUTCOME_PENDING,
    Budget,
    Query,
    ReadPoint,
    Spec,
    Transaction,
    VersionRecord,
)

SPEC_FORMAT = "mvcc-diagnostic-spec"
PREDICATE_OPS = frozenset({"exists", "version_eq", "digest_eq", "tx_eq"})

_TOP_LEVEL_KEYS = {
    "format",
    "format_version",
    "name",
    "transactions",
    "records",
    "read_points",
    "queries",
    "budget",
    "notes",
}
_TX_KEYS = {"id", "start_seq", "outcome", "commit_seq"}
_RECORD_KEYS = {"row_key", "tx_id", "version", "value_digest", "prev_version"}
_READ_POINT_KEYS = {"id", "snapshot_seq", "tx_id"}
_QUERY_KEYS = {"id", "read_point_id", "row_key", "predicate"}
_BUDGET_KEYS = {"global_concurrency", "per_read_point_concurrency"}


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_nonempty_str(
    value: Any,
    *,
    entity_type: str,
    entity_id: str | None,
    path: str,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(
            f"{field_name} 必须是非空字符串",
            entity_type=entity_type,
            entity_id=entity_id,
            path=path,
            details={"field": field_name},
        )
    return value


def _require_object(
    value: Any,
    *,
    entity_type: str,
    entity_id: str | None,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(
            f"{entity_type} 必须是 JSON 对象",
            entity_type=entity_type,
            entity_id=entity_id,
            path=path,
        )
    return value


def _reject_unknown_keys(
    value: dict[str, Any],
    allowed: set[str],
    *,
    entity_type: str,
    entity_id: str | None,
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(
            f"{entity_type} 含有未声明字段: {unknown[0]}",
            entity_type=entity_type,
            entity_id=entity_id,
            path=path,
            details={"field": unknown[0], "unknown_fields": unknown},
        )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"不支持的 JSON 常量: {value}")


def _parse_json(text: str, *, path: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SpecError(
            f"JSON 解析失败: {exc}",
            path=path,
            details={"parse_error": str(exc)},
        ) from exc


def _load_ndjson(text: str, *, source_path: str) -> dict[str, Any]:
    tx_items: list[dict[str, Any]] = []
    record_items: list[dict[str, Any]] = []
    read_point_items: list[dict[str, Any]] = []
    query_items: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SpecError(
                f"NDJSON 第 {line_number} 行解析失败: {exc}",
                path=f"{source_path}:{line_number}",
                details={"line": line_number, "parse_error": str(exc)},
            ) from exc
        if not isinstance(item, dict):
            raise SpecError(
                "NDJSON 每行必须是 JSON 对象",
                path=f"{source_path}:{line_number}",
                details={"line": line_number},
            )
        kind = item.get("kind")
        payload = {key: value for key, value in item.items() if key != "kind"}
        if kind == "meta":
            if meta is not None:
                raise SpecError(
                    "NDJSON 只能包含一条 meta 记录",
                    path=f"{source_path}:{line_number}",
                    details={"line": line_number},
                )
            meta = payload
        elif kind == "transaction":
            tx_items.append(payload)
        elif kind == "record":
            record_items.append(payload)
        elif kind == "read_point":
            read_point_items.append(payload)
        elif kind == "query":
            query_items.append(payload)
        else:
            raise SpecError(
                f"NDJSON 第 {line_number} 行 kind 非法: {kind!r}",
                path=f"{source_path}:{line_number}",
                details={"line": line_number, "kind": kind},
            )

    if meta is None:
        raise SpecError(
            "NDJSON 缺少 kind=meta 记录",
            path=source_path,
        )
    assembled = {
        "format": meta.get("format", SPEC_FORMAT),
        "format_version": meta.get("format_version"),
        "name": meta.get("name"),
        "budget": meta.get("budget"),
        "transactions": tx_items,
        "records": record_items,
        "read_points": read_point_items,
        "queries": query_items,
    }
    if "notes" in meta:
        assembled["notes"] = meta["notes"]
    extra_meta = sorted(set(meta) - {"format", "format_version", "name", "budget", "notes"})
    if extra_meta:
        raise SpecError(
            f"NDJSON meta 含有未声明字段: {extra_meta[0]}",
            path=f"{source_path}:meta",
            details={"field": extra_meta[0], "unknown_fields": extra_meta},
        )
    return assembled


def _parse_transactions(
    raw_items: Any,
    *,
    allow_pending: bool,
) -> tuple[Transaction, ...]:
    if not isinstance(raw_items, list):
        raise SpecError("transactions 必须是数组", entity_type="transaction", path="transactions")

    sortable: list[tuple[str, int, Any]] = []
    for index, item in enumerate(raw_items):
        raw_id = item.get("id") if isinstance(item, dict) else None
        sort_id = raw_id if isinstance(raw_id, str) else f"\uffff{index:09d}"
        sortable.append((sort_id, index, item))
    sortable.sort(key=lambda x: (x[0], x[1]))

    parsed: list[Transaction] = []
    seen: set[str] = set()
    for _, index, item in sortable:
        path = f"transactions[{index}]"
        obj = _require_object(item, entity_type="transaction", entity_id=None, path=path)
        tx_id = _require_nonempty_str(
            obj.get("id"),
            entity_type="transaction",
            entity_id=None,
            path=f"{path}.id",
            field_name="id",
        )
        _reject_unknown_keys(
            obj,
            _TX_KEYS,
            entity_type="transaction",
            entity_id=tx_id,
            path=path,
        )
        if tx_id in seen:
            raise SpecError(
                f"事务标识重复: {tx_id}",
                entity_type="transaction",
                entity_id=tx_id,
                path=f"{path}.id",
                details={"field": "id"},
            )
        seen.add(tx_id)

        start_seq = obj.get("start_seq")
        if not _is_int(start_seq) or start_seq < 0:
            raise SpecError(
                "start_seq 必须是非负整数",
                entity_type="transaction",
                entity_id=tx_id,
                path=f"{path}.start_seq",
                details={"field": "start_seq"},
            )
        outcome = obj.get("outcome")
        allowed_outcomes = EXTERNAL_OUTCOMES | ({OUTCOME_PENDING} if allow_pending else set())
        if outcome not in allowed_outcomes:
            raise SpecError(
                "outcome 必须是 committed 或 rolledback",
                entity_type="transaction",
                entity_id=tx_id,
                path=f"{path}.outcome",
                details={"field": "outcome", "allowed": sorted(allowed_outcomes)},
            )

        commit_seq = obj.get("commit_seq")
        if outcome == OUTCOME_COMMITTED:
            if not _is_int(commit_seq) or commit_seq < 0:
                raise SpecError(
                    "已提交事务的 commit_seq 必须是非负整数",
                    entity_type="transaction",
                    entity_id=tx_id,
                    path=f"{path}.commit_seq",
                    details={"field": "commit_seq"},
                )
            if commit_seq < start_seq:
                raise SpecError(
                    "commit_seq 不得小于 start_seq",
                    entity_type="transaction",
                    entity_id=tx_id,
                    path=f"{path}.commit_seq",
                    details={
                        "field": "commit_seq",
                        "start_seq": start_seq,
                        "commit_seq": commit_seq,
                    },
                )
        elif commit_seq is not None:
            raise SpecError(
                "仅 committed 事务可携带 commit_seq",
                entity_type="transaction",
                entity_id=tx_id,
                path=f"{path}.commit_seq",
                details={"field": "commit_seq"},
            )

        parsed.append(
            Transaction(
                id=tx_id,
                start_seq=start_seq,
                outcome=outcome,
                commit_seq=commit_seq,
            )
        )
    parsed.sort(key=lambda x: x.id)
    return tuple(parsed)


def _parse_records(
    raw_items: Any,
    *,
    tx_ids: set[str],
) -> tuple[VersionRecord, ...]:
    if not isinstance(raw_items, list):
        raise SpecError("records 必须是数组", entity_type="row_key", path="records")

    sortable: list[tuple[str, int, Any]] = []
    for index, item in enumerate(raw_items):
        if isinstance(item, dict) and isinstance(item.get("row_key"), str):
            key = item["row_key"]
        else:
            key = f"\uffff{index:09d}"
        sortable.append((key, index, item))
    sortable.sort(key=lambda x: (x[0], x[1]))

    parsed: list[VersionRecord] = []
    versions_by_row: dict[str, set[int]] = {}
    for _, index, item in sortable:
        path = f"records[{index}]"
        obj = _require_object(item, entity_type="row_key", entity_id=None, path=path)
        row_key = _require_nonempty_str(
            obj.get("row_key"),
            entity_type="row_key",
            entity_id=None,
            path=f"{path}.row_key",
            field_name="row_key",
        )
        _reject_unknown_keys(
            obj,
            _RECORD_KEYS,
            entity_type="row_key",
            entity_id=row_key,
            path=path,
        )
        tx_id = _require_nonempty_str(
            obj.get("tx_id"),
            entity_type="row_key",
            entity_id=row_key,
            path=f"{path}.tx_id",
            field_name="tx_id",
        )
        if tx_id not in tx_ids:
            raise SpecError(
                f"版本记录引用不存在的事务: {tx_id}",
                entity_type="row_key",
                entity_id=row_key,
                path=f"{path}.tx_id",
                details={"field": "tx_id", "tx_id": tx_id},
            )
        version = obj.get("version")
        if not _is_int(version) or version < 0:
            raise SpecError(
                "version 必须是非负整数",
                entity_type="row_key",
                entity_id=row_key,
                path=f"{path}.version",
                details={"field": "version"},
            )
        value_digest = _require_nonempty_str(
            obj.get("value_digest"),
            entity_type="row_key",
            entity_id=row_key,
            path=f"{path}.value_digest",
            field_name="value_digest",
        )
        prev_version = obj.get("prev_version")
        if prev_version is not None and (not _is_int(prev_version) or prev_version < 0):
            raise SpecError(
                "prev_version 必须是 null 或非负整数",
                entity_type="row_key",
                entity_id=row_key,
                path=f"{path}.prev_version",
                details={"field": "prev_version"},
            )
        row_versions = versions_by_row.setdefault(row_key, set())
        if version in row_versions:
            raise SpecError(
                f"行键 {row_key} 的版本号重复: {version}",
                entity_type="row_key",
                entity_id=row_key,
                path=f"{path}.version",
                details={"field": "version", "version": version},
            )
        row_versions.add(version)
        parsed.append(
            VersionRecord(
                row_key=row_key,
                tx_id=tx_id,
                version=version,
                value_digest=value_digest,
                prev_version=prev_version,
            )
        )

    parsed.sort(key=lambda x: (x.row_key, x.version))
    by_row: dict[str, list[VersionRecord]] = {}
    for record in parsed:
        by_row.setdefault(record.row_key, []).append(record)
    for row_key in sorted(by_row):
        by_version = {item.version: item for item in by_row[row_key]}
        for record in by_row[row_key]:
            if record.prev_version is not None and record.prev_version not in by_version:
                raise SpecError(
                    f"行键 {row_key} 的前一版本引用不存在: {record.prev_version}",
                    entity_type="row_key",
                    entity_id=row_key,
                    path=f"records[{row_key}/{record.version}].prev_version",
                    details={
                        "field": "prev_version",
                        "version": record.version,
                        "prev_version": record.prev_version,
                    },
                )
        _detect_first_cycle(row_key, by_row[row_key])
    return tuple(parsed)


def _detect_first_cycle(row_key: str, records: Iterable[VersionRecord]) -> None:
    by_version = {record.version: record for record in records}
    checked: set[int] = set()
    for start in sorted(by_version):
        path: list[int] = []
        positions: dict[int, int] = {}
        current: int | None = start
        while current is not None and current not in checked:
            if current in positions:
                cycle = path[positions[current] :]
                raise CycleError(
                    f"行键 {row_key} 的版本链存在环",
                    row_key=row_key,
                    cycle=cycle,
                    path=f"records[{row_key}]",
                )
            positions[current] = len(path)
            path.append(current)
            current = by_version[current].prev_version
        checked.update(path)


def _parse_read_points(
    raw_items: Any,
    *,
    tx_ids: set[str],
) -> tuple[ReadPoint, ...]:
    if not isinstance(raw_items, list):
        raise SpecError("read_points 必须是数组", entity_type="read_point", path="read_points")

    parsed: list[tuple[int, ReadPoint]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        path = f"read_points[{index}]"
        obj = _require_object(item, entity_type="read_point", entity_id=None, path=path)
        read_point_id = _require_nonempty_str(
            obj.get("id"),
            entity_type="read_point",
            entity_id=None,
            path=f"{path}.id",
            field_name="id",
        )
        _reject_unknown_keys(
            obj,
            _READ_POINT_KEYS,
            entity_type="read_point",
            entity_id=read_point_id,
            path=path,
        )
        if read_point_id in seen:
            raise SpecError(
                f"读点标识重复: {read_point_id}",
                entity_type="read_point",
                entity_id=read_point_id,
                path=f"{path}.id",
                details={"field": "id"},
            )
        seen.add(read_point_id)
        snapshot_seq = obj.get("snapshot_seq")
        if not _is_int(snapshot_seq) or snapshot_seq < 0:
            raise SpecError(
                "snapshot_seq 必须是非负整数",
                entity_type="read_point",
                entity_id=read_point_id,
                path=f"{path}.snapshot_seq",
                details={"field": "snapshot_seq"},
            )
        tx_id = _require_nonempty_str(
            obj.get("tx_id"),
            entity_type="read_point",
            entity_id=read_point_id,
            path=f"{path}.tx_id",
            field_name="tx_id",
        )
        if tx_id not in tx_ids:
            raise SpecError(
                f"读点引用不存在的事务: {tx_id}",
                entity_type="read_point",
                entity_id=read_point_id,
                path=f"{path}.tx_id",
                details={"field": "tx_id", "tx_id": tx_id},
            )
        parsed.append(
            (
                index,
                ReadPoint(id=read_point_id, snapshot_seq=snapshot_seq, tx_id=tx_id),
            )
        )
    parsed.sort(key=lambda item: item[1].id)
    return tuple(item[1] for item in parsed)


def _validate_predicate(
    value: Any,
    *,
    query_id: str,
    path: str,
) -> dict[str, Any]:
    obj = _require_object(value, entity_type="query", entity_id=query_id, path=path)
    op = obj.get("op")
    if op not in PREDICATE_OPS:
        raise SpecError(
            "predicate.op 非法",
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.op",
            details={"field": "predicate.op", "allowed": sorted(PREDICATE_OPS)},
        )
    if op == "exists":
        _reject_unknown_keys(
            obj,
            {"op"},
            entity_type="query",
            entity_id=query_id,
            path=path,
        )
        return {"op": "exists"}
    _reject_unknown_keys(
        obj,
        {"op", "value"},
        entity_type="query",
        entity_id=query_id,
        path=path,
    )
    if "value" not in obj:
        raise SpecError(
            "predicate 缺少 value",
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.value",
            details={"field": "predicate.value"},
        )
    predicate_value = obj["value"]
    if op == "version_eq" and (not _is_int(predicate_value) or predicate_value < 0):
        raise SpecError(
            "version_eq 的 value 必须是非负整数",
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.value",
            details={"field": "predicate.value"},
        )
    if op in {"digest_eq", "tx_eq"} and (
        not isinstance(predicate_value, str) or not predicate_value
    ):
        raise SpecError(
            f"{op} 的 value 必须是非空字符串",
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.value",
            details={"field": "predicate.value"},
        )
    return {"op": op, "value": predicate_value}


def _parse_queries(
    raw_items: Any,
    *,
    read_point_ids: set[str],
    row_keys: set[str],
) -> tuple[Query, ...]:
    if not isinstance(raw_items, list):
        raise SpecError("queries 必须是数组", entity_type="query", path="queries")

    parsed: list[tuple[int, Query]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        path = f"queries[{index}]"
        obj = _require_object(item, entity_type="query", entity_id=None, path=path)
        query_id = _require_nonempty_str(
            obj.get("id"),
            entity_type="query",
            entity_id=None,
            path=f"{path}.id",
            field_name="id",
        )
        _reject_unknown_keys(
            obj,
            _QUERY_KEYS,
            entity_type="query",
            entity_id=query_id,
            path=path,
        )
        if query_id in seen:
            raise SpecError(
                f"查询标识重复: {query_id}",
                entity_type="query",
                entity_id=query_id,
                path=f"{path}.id",
                details={"field": "id"},
            )
        seen.add(query_id)
        read_point_id = _require_nonempty_str(
            obj.get("read_point_id"),
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.read_point_id",
            field_name="read_point_id",
        )
        if read_point_id not in read_point_ids:
            raise SpecError(
                f"查询引用不存在的读点: {read_point_id}",
                entity_type="query",
                entity_id=query_id,
                path=f"{path}.read_point_id",
                details={"field": "read_point_id"},
            )
        row_key = _require_nonempty_str(
            obj.get("row_key"),
            entity_type="query",
            entity_id=query_id,
            path=f"{path}.row_key",
            field_name="row_key",
        )
        if row_key not in row_keys:
            raise SpecError(
                f"查询引用不存在的行键: {row_key}",
                entity_type="query",
                entity_id=query_id,
                path=f"{path}.row_key",
                details={"field": "row_key", "row_key": row_key},
            )
        predicate = _validate_predicate(
            obj.get("predicate"), query_id=query_id, path=f"{path}.predicate"
        )
        parsed.append(
            (
                index,
                Query(
                    id=query_id,
                    read_point_id=read_point_id,
                    row_key=row_key,
                    predicate=predicate,
                ),
            )
        )
    parsed.sort(key=lambda item: item[1].id)
    return tuple(item[1] for item in parsed)


def _parse_budget(value: Any) -> Budget:
    obj = _require_object(value, entity_type="budget", entity_id=None, path="budget")
    _reject_unknown_keys(obj, _BUDGET_KEYS, entity_type="budget", entity_id=None, path="budget")
    global_concurrency = obj.get("global_concurrency")
    per_read_point_concurrency = obj.get("per_read_point_concurrency")
    if not _is_int(global_concurrency) or global_concurrency <= 0:
        raise SpecError(
            "global_concurrency 必须是正整数",
            entity_type="budget",
            path="budget.global_concurrency",
            details={"field": "global_concurrency"},
        )
    if not _is_int(per_read_point_concurrency) or per_read_point_concurrency <= 0:
        raise SpecError(
            "per_read_point_concurrency 必须是正整数",
            entity_type="budget",
            path="budget.per_read_point_concurrency",
            details={"field": "per_read_point_concurrency"},
        )
    return Budget(
        global_concurrency=global_concurrency,
        per_read_point_concurrency=per_read_point_concurrency,
    )


def validate_spec_object(
    value: Any,
    *,
    source_path: str = "<memory>",
    allow_pending: bool = False,
) -> Spec:
    obj = _require_object(value, entity_type="spec", entity_id=None, path="$")
    _reject_unknown_keys(obj, _TOP_LEVEL_KEYS, entity_type="spec", entity_id=None, path="$")
    if obj.get("format") != SPEC_FORMAT:
        raise SpecError(
            f"format 必须是 {SPEC_FORMAT}",
            path="$.format",
            details={"field": "format"},
        )
    format_version = obj.get("format_version")
    if not _is_int(format_version) or format_version <= 0:
        raise SpecError(
            "format_version 必须是正整数",
            path="$.format_version",
            details={"field": "format_version"},
        )
    name = _require_nonempty_str(
        obj.get("name"),
        entity_type="spec",
        entity_id=None,
        path="$.name",
        field_name="name",
    )
    notes = obj.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise SpecError(
            "notes 必须是字符串",
            path="$.notes",
            details={"field": "notes"},
        )
    transactions = _parse_transactions(obj.get("transactions"), allow_pending=allow_pending)
    tx_ids = {item.id for item in transactions}
    records = _parse_records(obj.get("records"), tx_ids=tx_ids)
    read_points = _parse_read_points(obj.get("read_points"), tx_ids=tx_ids)
    queries = _parse_queries(
        obj.get("queries"),
        read_point_ids={item.id for item in read_points},
        row_keys={item.row_key for item in records},
    )
    budget = _parse_budget(obj.get("budget"))
    return Spec(
        name=name,
        format_version=format_version,
        transactions=transactions,
        records=records,
        read_points=read_points,
        queries=queries,
        budget=budget,
        notes=notes,
    )


def load_spec(path: str | Path, *, allow_pending: bool = False) -> Spec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise SpecError(
            f"无法读取规格文件: {exc}",
            path=str(source),
            details={"os_error": str(exc)},
        ) from exc
    if source.suffix.lower() == ".ndjson":
        value = _load_ndjson(text, source_path=str(source))
    else:
        value = _parse_json(text, path=str(source))
    return validate_spec_object(value, source_path=str(source), allow_pending=allow_pending)
