"""JSON specification loading, aliases, normalization and strict validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .errors import SpecError
from .hashing import canonical_dumps, canonical_hash


SUPPORTED_ENCODINGS = {"raw", "dict", "rle"}
ENCODING_ALIASES = {
    "plain": "raw",
    "original": "raw",
    "原始": "raw",
    "dictionary": "dict",
    "字典": "dict",
    "run_length": "rle",
    "run-length": "rle",
    "rle": "rle",
    "游程": "rle",
}
SUPPORTED_COMPRESSIONS = {"none"} | {f"zlib-{level}" for level in range(1, 10)}
COMPRESSION_ALIASES = {"zlib": "zlib-6", "zlib:6": "zlib-6", "zlib-default": "zlib-6"}
SUPPORTED_OPS = {
    "=",
    "==",
    "!=",
    "<>",
    "<",
    "<=",
    ">",
    ">=",
    "in",
    "not_in",
    "not in",
    "between",
    "is_null",
    "is null",
    "is_not_null",
    "is not null",
}
NULLARY_OPS = {"is_null", "is null", "is_not_null", "is not null"}
SUPPORTED_AGGREGATES = {"count_star", "count", "sum", "min", "max", "avg", "mean"}

_BG_KEYS = ("block_groups", "blockGroups", "groups")
_ID_KEYS = ("id", "block_group_id", "blockGroupId", "group_id")
_ROWS_KEYS = ("row_count", "rows", "rowCount")
_COLUMNS_KEYS = ("columns", "cols")
_INDEX_KEYS = ("skip_index", "skipIndex", "index", "statistics", "stats")
_QUERIES_KEYS = ("queries", "query_set", "querySet")
_PREDICATE_KEYS = ("predicates", "where", "filters")


class Spec(dict):
    """Dictionary specification with convenience accessors."""

    @property
    def block_groups(self) -> list[dict[str, Any]]:
        return self["block_groups"]

    @property
    def queries(self) -> list[dict[str, Any]]:
        return self["queries"]

    @property
    def scheduler(self) -> dict[str, Any]:
        return self["scheduler"]

    @property
    def spec_digest(self) -> str:
        return spec_digest(self)

    def block_group(self, block_group_id: str) -> dict[str, Any]:
        for group in self["block_groups"]:
            if group["id"] == block_group_id:
                return group
        raise KeyError(block_group_id)


def _first_key(obj: Mapping[str, Any], names: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in names:
        if name in obj:
            return name, obj[name]
    return None, None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _bg_error(group: Mapping[str, Any] | None, field: str, message: str, value: Any = None) -> SpecError:
    bg_id = None
    if group is not None:
        _, raw_id = _first_key(group, _ID_KEYS)
        bg_id = raw_id if isinstance(raw_id, str) else (str(raw_id) if raw_id is not None else None)
    return SpecError(
        f"E_SPEC block_group={bg_id or '<unknown>'} field={field}: {message}",
        block_group_id=bg_id,
        field=field,
        value=value,
    )


def _normalize_op(op: Any) -> str:
    if not isinstance(op, str):
        return ""
    text = op.strip().lower().replace(" ", "_")
    aliases = {
        "eq": "=",
        "equal": "=",
        "==": "=",
        "neq": "!=",
        "ne": "!=",
        "<>": "!=",
        "lt": "<",
        "lte": "<=",
        "le": "<=",
        "gt": ">",
        "gte": ">=",
        "ge": ">=",
        "not_in": "not_in",
        "is_null": "is_null",
        "is_not_null": "is_not_null",
    }
    text = aliases.get(text, text)
    if text == "==":
        text = "="
    return text


def normalize_predicate(predicate: Mapping[str, Any]) -> dict[str, Any]:
    column = predicate.get("column") or predicate.get("name") or predicate.get("col")
    op = _normalize_op(predicate.get("op") or predicate.get("operator"))
    normalized: dict[str, Any] = {"column": column, "op": op}
    if op not in NULLARY_OPS or "value" in predicate:
        value = predicate.get("value", predicate.get("values"))
        if op == "in" and not isinstance(value, list):
            value = [value]
        if op == "not_in" and not isinstance(value, list):
            value = [value]
        if op in {"in", "not_in"} and isinstance(value, list):
            value = sorted(value, key=lambda item: (type(item).__name__, canonical_dumps(item)))
        normalized["value"] = value
    return normalized


def canonical_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, canonical_dumps(value))


def predicate_sort_key(predicate: Mapping[str, Any]) -> tuple[str, str, str]:
    normalized = normalize_predicate(predicate)
    return (
        str(normalized.get("column", "")),
        str(normalized.get("op", "")),
        canonical_hash(normalized.get("value")),
    )


def _normalize_column(column: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(column, str):
        column = {"name": column}
    name = column.get("name") or column.get("column")
    encoding = ENCODING_ALIASES.get(str(column.get("encoding", "raw")).strip().lower(), column.get("encoding", "raw"))
    compression = column.get("compression", "none")
    compression = COMPRESSION_ALIASES.get(str(compression).strip().lower(), compression)
    normalized = {
        "name": name,
        "encoding": encoding,
        "compression": compression,
        "seed": int(column.get("seed", 0)) if column.get("seed") is not None else 0,
        "modulus": column.get("modulus"),
        "offset": column.get("offset", 0),
        "null_every": column.get("null_every", 0),
    }
    for key in ("generator", "values"):
        if key in column:
            normalized[key] = copy.deepcopy(column[key])
    return normalized


def _normalize_columns(raw_columns: Any) -> list[dict[str, Any]]:
    if isinstance(raw_columns, Mapping):
        columns = []
        for name, config in raw_columns.items():
            item = dict(config) if isinstance(config, Mapping) else {"encoding": config}
            item["name"] = name
            columns.append(_normalize_column(item))
        return columns
    return [_normalize_column(item) for item in raw_columns]


def _validate_group(raw_group: Any, seen: set[str]) -> None:
    if not isinstance(raw_group, Mapping):
        raise SpecError("E_SPEC block_group=<unknown> field=block_groups: each block group must be an object", field="block_groups")

    _, bg_id = _first_key(raw_group, _ID_KEYS)
    if not isinstance(bg_id, str) or not bg_id:
        raise _bg_error(raw_group, "id", "block group id must be a non-empty string", bg_id)
    if bg_id in seen:
        raise _bg_error(raw_group, "id", "duplicate block group id", bg_id)
    seen.add(bg_id)

    _, row_count = _first_key(raw_group, _ROWS_KEYS)
    if not _is_int(row_count) or row_count <= 0:
        raise _bg_error(raw_group, "row_count", "row_count must be a positive integer", row_count)

    _, raw_columns = _first_key(raw_group, _COLUMNS_KEYS)
    if not isinstance(raw_columns, (list, Mapping)) or len(raw_columns) == 0:
        raise _bg_error(raw_group, "columns", "columns must be a non-empty list or object", raw_columns)

    columns = _normalize_columns(raw_columns)
    column_names: set[str] = set()
    for column in columns:
        name = column["name"]
        if not isinstance(name, str) or not name:
            raise _bg_error(raw_group, "columns.name", "column name must be a non-empty string", name)
        if name in column_names:
            raise _bg_error(raw_group, f"columns[{name}].name", "duplicate column name", name)
        column_names.add(name)

        if column["encoding"] not in SUPPORTED_ENCODINGS:
            raise _bg_error(raw_group, "encoding", "encoding must be raw, dict or rle", column["encoding"])
        if column["compression"] not in SUPPORTED_COMPRESSIONS:
            raise _bg_error(
                raw_group,
                "compression",
                "compression must be none or zlib-1..zlib-9",
                column["compression"],
            )
        if not _is_int(column["seed"]):
            raise _bg_error(raw_group, f"columns[{name}].seed", "seed must be an integer", column["seed"])
        if column["modulus"] is not None and (not _is_int(column["modulus"]) or column["modulus"] <= 0):
            raise _bg_error(raw_group, f"columns[{name}].modulus", "modulus must be a positive integer", column["modulus"])
        if not _is_int(column["offset"]):
            raise _bg_error(raw_group, f"columns[{name}].offset", "offset must be an integer", column["offset"])
        if not _is_int(column["null_every"]) or column["null_every"] < 0:
            raise _bg_error(raw_group, f"columns[{name}].null_every", "null_every must be a non-negative integer", column["null_every"])

    _, raw_index = _first_key(raw_group, _INDEX_KEYS)
    if raw_index is not None:
        if not isinstance(raw_index, Mapping):
            raise _bg_error(raw_group, "skip_index", "skip_index must be an object", raw_index)
        for name, stats in raw_index.items():
            field = f"skip_index.{name}"
            if name not in column_names:
                raise _bg_error(raw_group, field, "index references unknown column", name)
            if not isinstance(stats, Mapping):
                raise _bg_error(raw_group, field, "column index stats must be an object", stats)
            minimum = stats.get("min")
            maximum = stats.get("max")
            null_count = stats.get("null_count", stats.get("nullCount", 0))
            if minimum is not None and not _is_number(minimum):
                raise _bg_error(raw_group, f"{field}.min", "min must be numeric or null", minimum)
            if maximum is not None and not _is_number(maximum):
                raise _bg_error(raw_group, f"{field}.max", "max must be numeric or null", maximum)
            if minimum is not None and maximum is not None and minimum > maximum:
                raise _bg_error(raw_group, f"{field}.min", "index min must not exceed max", {"min": minimum, "max": maximum})
            if not _is_int(null_count) or null_count < 0 or null_count > row_count:
                raise _bg_error(raw_group, f"{field}.null_count", "null_count must be between 0 and row_count", null_count)


def _validate_query(query: Any, index: int, groups: list[dict[str, Any]]) -> None:
    if not isinstance(query, Mapping):
        raise SpecError(f"E_SPEC queries[{index}] must be an object", field=f"queries[{index}]")
    query_id = query.get("id") or query.get("query_id") or f"query-{index}"
    _, predicates = _first_key(query, _PREDICATE_KEYS)
    if predicates is None:
        predicates = []
    if not isinstance(predicates, list):
        raise SpecError(f"E_SPEC query={query_id} field=predicates must be a list", query_id=query_id, field="predicates")

    for p_index, predicate in enumerate(predicates):
        if not isinstance(predicate, Mapping):
            raise SpecError(f"E_SPEC query={query_id} predicate {p_index} must be object", query_id=query_id, field="predicates")
        normalized = normalize_predicate(predicate)
        column = normalized["column"]
        if not isinstance(column, str) or not column:
            raise SpecError(f"E_SPEC query={query_id} field=predicates[{p_index}].column missing", query_id=query_id, field="predicates.column")
        if normalized["op"] not in SUPPORTED_OPS:
            raise SpecError(
                f"E_SPEC query={query_id} field=predicates[{p_index}].op unsupported {normalized['op']}",
                query_id=query_id,
                field="predicates.op",
            )
        op = normalized["op"]
        value = normalized.get("value")
        if op in {"in", "not_in"} and (not isinstance(value, list) or not value):
            raise SpecError(
                f"E_SPEC query={query_id} field=predicates[{p_index}].value must be a non-empty list",
                query_id=query_id,
                field="predicates.value",
            )
        if op == "between":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not _is_number(value[0])
                or not _is_number(value[1])
                or value[0] > value[1]
            ):
                raise SpecError(
                    f"E_SPEC query={query_id} field=predicates[{p_index}].value must be [low, high]",
                    query_id=query_id,
                    field="predicates.value",
                )
        for group in groups:
            names = {c["name"] for c in group["columns"]}
            if column not in names:
                raise _bg_error(
                    group,
                    f"queries[{query_id}].predicates[{p_index}].column",
                    f"predicate column {column!r} does not exist in block group",
                    column,
                )

    projection = query.get("projection", query.get("project"))
    if projection is not None:
        if not isinstance(projection, list) or not all(isinstance(x, str) for x in projection):
            raise SpecError(f"E_SPEC query={query_id} field=projection invalid", query_id=query_id, field="projection")
        for column in projection:
            for group in groups:
                if column not in {c["name"] for c in group["columns"]}:
                    raise _bg_error(group, f"queries[{query_id}].projection", f"projection column {column!r} does not exist", column)

    aggregations = query.get("aggregations", query.get("aggregates", []))
    if not isinstance(aggregations, list):
        raise SpecError(f"E_SPEC query={query_id} field=aggregations must be list", query_id=query_id, field="aggregations")
    for agg in aggregations:
        function = str(agg.get("function", "")).lower()
        if function not in SUPPORTED_AGGREGATES:
            raise SpecError(f"E_SPEC query={query_id} field=aggregations.function unsupported", query_id=query_id, field="aggregations.function")
        column = agg.get("column")
        if function != "count_star" and not column:
            raise SpecError(f"E_SPEC query={query_id} field=aggregations.column required", query_id=query_id, field="aggregations.column")
        if column:
            for group in groups:
                if column not in {c["name"] for c in group["columns"]}:
                    raise _bg_error(group, f"queries[{query_id}].aggregations.column", f"aggregate column {column!r} does not exist", column)


def validate_spec(raw_spec: Mapping[str, Any]) -> None:
    if not isinstance(raw_spec, Mapping):
        raise SpecError("E_SPEC root: specification must be a JSON object", field="$")
    _, raw_groups = _first_key(raw_spec, _BG_KEYS)
    if not isinstance(raw_groups, list) or not raw_groups:
        raise SpecError("E_SPEC field=block_groups: at least one block group is required", field="block_groups")

    seen: set[str] = set()
    for group in raw_groups:
        _validate_group(group, seen)

    normalized_groups = [normalize_group(group) for group in raw_groups]
    _, raw_queries = _first_key(raw_spec, _QUERIES_KEYS)
    if raw_queries is None:
        raw_queries = []
    if not isinstance(raw_queries, list):
        raise SpecError("E_SPEC field=queries must be a list", field="queries")
    query_ids: set[str] = set()
    for index, query in enumerate(raw_queries):
        if not isinstance(query, Mapping):
            raise SpecError(f"E_SPEC queries[{index}] must be object", field=f"queries[{index}]")
        qid = query.get("id") or query.get("query_id")
        if not isinstance(qid, str) or not qid:
            raise SpecError(f"E_SPEC queries[{index}].id required", field="queries.id")
        if qid in query_ids:
            raise SpecError(f"E_SPEC query={qid} field=id duplicate", query_id=qid, block_group_id=None, field="id")
        query_ids.add(qid)
        _validate_query(query, index, normalized_groups)

    verification = raw_spec.get("verification", {"mode": "full"})
    mode = str((verification or {}).get("mode", "full")).lower()
    if mode not in {"full", "all", "complete"}:
        raise SpecError("E_SPEC field=verification.mode: only full verification is supported; sampling is forbidden", field="verification.mode", value=mode)

    scheduler = raw_spec.get("scheduler", {}) or {}
    for key in ("global_decode_budget",):
        if key in scheduler and (not _is_int(scheduler[key]) or scheduler[key] <= 0):
            raise SpecError(f"E_SPEC field=scheduler.{key} must be positive integer", field=f"scheduler.{key}")
    if "per_column_lease" in scheduler:
        value = scheduler["per_column_lease"]
        if isinstance(value, Mapping):
            for column, lease in value.items():
                if not _is_int(lease) or lease <= 0:
                    raise SpecError(f"E_SPEC field=scheduler.per_column_lease.{column} must be positive", field="scheduler.per_column_lease")
        elif not _is_int(value) or value <= 0:
            raise SpecError("E_SPEC field=scheduler.per_column_lease must be positive integer or map", field="scheduler.per_column_lease")


def normalize_group(raw_group: Mapping[str, Any]) -> dict[str, Any]:
    _, bg_id = _first_key(raw_group, _ID_KEYS)
    _, row_count = _first_key(raw_group, _ROWS_KEYS)
    _, raw_columns = _first_key(raw_group, _COLUMNS_KEYS)
    _, raw_index = _first_key(raw_group, _INDEX_KEYS)
    columns = sorted(_normalize_columns(raw_columns), key=lambda c: c["name"])
    index: dict[str, Any] = {}
    if isinstance(raw_index, Mapping):
        for name in sorted(raw_index):
            stats = raw_index[name]
            index[name] = {
                "min": stats.get("min"),
                "max": stats.get("max"),
                "null_count": int(stats.get("null_count", stats.get("nullCount", 0))),
            }
    return {
        "id": bg_id,
        "row_count": int(row_count),
        "columns": columns,
        "skip_index": index,
    }


def normalize_query(raw_query: Mapping[str, Any]) -> dict[str, Any]:
    qid = raw_query.get("id") or raw_query.get("query_id")
    _, raw_predicates = _first_key(raw_query, _PREDICATE_KEYS)
    predicates = [normalize_predicate(p) for p in (raw_predicates or [])]
    predicates = sorted(predicates, key=predicate_sort_key)
    projection = raw_query.get("projection", raw_query.get("project", [])) or []
    aggregations = []
    for agg in raw_query.get("aggregations", raw_query.get("aggregates", [])) or []:
        normalized = dict(agg)
        normalized["function"] = str(agg.get("function", "")).lower()
        if normalized["function"] == "mean":
            normalized["function"] = "avg"
        if "alias" not in normalized:
            if normalized["function"] == "count_star":
                normalized["alias"] = "rows"
            else:
                normalized["alias"] = f"{normalized['function']}_{normalized.get('column')}"
        aggregations.append(normalized)
    return {
        "id": qid,
        "predicates": predicates,
        "projection": list(projection),
        "aggregations": aggregations,
    }


def normalize_spec(raw_spec: Mapping[str, Any]) -> Spec:
    validate_spec(raw_spec)
    _, raw_groups = _first_key(raw_spec, _BG_KEYS)
    _, raw_queries = _first_key(raw_spec, _QUERIES_KEYS)
    groups = sorted((normalize_group(g) for g in raw_groups), key=lambda g: g["id"])
    queries = sorted((normalize_query(q) for q in (raw_queries or [])), key=lambda q: q["id"])
    scheduler = {
        "global_decode_budget": int((raw_spec.get("scheduler") or {}).get("global_decode_budget", 2)),
        "per_column_lease": (raw_spec.get("scheduler") or {}).get("per_column_lease", 1),
        "socket_timeout_ms": int((raw_spec.get("scheduler") or {}).get("socket_timeout_ms", 2000)),
        "worker_response_delay_ms": int((raw_spec.get("scheduler") or {}).get("worker_response_delay_ms", 0)),
        "worker_start_timeout_ms": int((raw_spec.get("scheduler") or {}).get("worker_start_timeout_ms", 5000)),
    }
    spec = Spec(
        format_version=int(raw_spec.get("format_version", raw_spec.get("version", 1))),
        version=int(raw_spec.get("version", raw_spec.get("format_version", 1))),
        session_id=raw_spec.get("session_id", "session"),
        block_groups=groups,
        queries=queries,
        scheduler=scheduler,
        verification={"mode": "full"},
    )
    return spec


def load_spec(source: str | Path | Mapping[str, Any]) -> Spec:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    elif isinstance(source, Mapping):
        raw = copy.deepcopy(source)
    else:
        raise TypeError("spec source must be a path or mapping")
    return normalize_spec(raw)


def spec_digest(spec: Mapping[str, Any]) -> str:
    return canonical_hash(spec)


load_spec_file = load_spec
load_json_spec = load_spec
parse_spec = normalize_spec
