"""Raw JSONL, group-map, and table validation."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from .errors import (
    E_DATA_ROW_INVALID,
    E_GROUP_MAP_INVALID,
    WorkbenchError,
    diagnostic,
)
from .util import as_plain_cell, read_json, sha256_obj


MISSING_STRING = "__MISSING__"


def read_jsonl_drafts(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read JSON objects without applying schema validation.

    Parsing failures become row diagnostics and do not prevent later lines from
    loading.  Python's JSON parser accepts ``NaN``/``Infinity`` tokens so injected
    non-finite values reach the finite-number validation layer.
    """

    drafts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(E_DATA_ROW_INVALID, f"cannot open raw table: {exc}") from exc
    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    diagnostic(
                        E_DATA_ROW_INVALID,
                        f"invalid JSON: {exc.msg}",
                        line_no=line_no,
                        stage="load",
                    )
                )
                continue
            if not isinstance(value, dict):
                failures.append(
                    diagnostic(
                        E_DATA_ROW_INVALID,
                        "JSONL line must contain an object",
                        line_no=line_no,
                        stage="load",
                    )
                )
                continue
            drafts.append({"line_no": line_no, "object": value})
    return drafts, failures


def _decode_cell(raw: Any, spec: dict[str, Any], column: str) -> tuple[Any, dict[str, Any] | None]:
    if raw is None or raw == MISSING_STRING:
        return None, None
    if isinstance(raw, dict):
        if raw.get("missing") is True:
            additional = set(raw) - {"missing"}
            if additional:
                return None, diagnostic(
                    E_DATA_ROW_INVALID,
                    "missing marker and an original value cannot appear together",
                    column=column,
                    stage="load",
                    conflicting_fields=sorted(additional),
                )
            return None, None
        return None, diagnostic(
            E_DATA_ROW_INVALID,
            "cell objects must use {\"missing\": true} or a scalar value",
            column=column,
            stage="load",
            value=as_plain_cell(raw),
        )

    if spec["type"] == "numeric":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, diagnostic(
                E_DATA_ROW_INVALID,
                "numeric column value must be a number or missing marker",
                column=column,
                stage="load",
                value=as_plain_cell(raw),
            )
        value = float(raw)
        if not math.isfinite(value):
            return None, diagnostic(
                E_DATA_ROW_INVALID,
                "numeric column value must be finite",
                column=column,
                stage="load",
                value=as_plain_cell(raw),
            )
        return value, None

    if not isinstance(raw, str):
        return None, diagnostic(
            E_DATA_ROW_INVALID,
            "category column value must be a string or missing marker",
            column=column,
            stage="load",
            value=as_plain_cell(raw),
        )
    if raw not in spec.get("domain", []):
        return None, diagnostic(
            E_DATA_ROW_INVALID,
            f"category value {raw!r} is outside the declared domain",
            column=column,
            stage="load",
            value=raw,
        )
    return raw, None


def validate_drafts(
    drafts: list[dict[str, Any]],
    config: dict[str, Any],
    initial_failures: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """Validate drafts, returning valid rows, failures, and the observed schema."""

    failures = list(initial_failures or [])
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    schema_columns: set[str] = set()
    id_column = config["id_column"]
    target = config.get("target_column")

    for draft in drafts:
        obj = draft["object"]
        line_no = draft["line_no"]
        schema_columns.update(str(key) for key in obj)
        raw_id = obj.get(id_column)
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)) or str(raw_id) == "":
            failures.append(
                diagnostic(
                    E_DATA_ROW_INVALID,
                    f"row identifier {id_column!r} must be a non-empty string or integer",
                    line_no=line_no,
                    column=id_column,
                    stage="load",
                )
            )
            continue
        row_id = str(raw_id)
        if row_id in seen_ids:
            failures.append(
                diagnostic(
                    E_DATA_ROW_INVALID,
                    "duplicate row identifier",
                    row_id=row_id,
                    line_no=line_no,
                    column=id_column,
                    stage="load",
                )
            )
            continue
        seen_ids.add(row_id)

        row_errors: list[dict[str, Any]] = []
        values: dict[str, Any] = {}
        for column, spec in config["columns"].items():
            if column == id_column:
                continue
            if column not in obj:
                row_errors.append(
                    diagnostic(
                        E_DATA_ROW_INVALID,
                        "row is missing a configured column",
                        row_id=row_id,
                        line_no=line_no,
                        column=column,
                        stage="load",
                    )
                )
                continue
            value, error = _decode_cell(obj[column], spec, column)
            if error is not None:
                error.update({"row_id": row_id, "line_no": line_no})
                row_errors.append(error)
                continue
            values[column] = value
        if target in values and values[target] is not None:
            target_value = values[target]
            if target_value not in (0.0, 1.0):
                row_errors.append(
                    diagnostic(
                        E_DATA_ROW_INVALID,
                        "classification target must be 0 or 1",
                        row_id=row_id,
                        line_no=line_no,
                        column=target,
                        stage="load",
                        value=target_value,
                    )
                )
        if row_errors:
            failures.extend(row_errors)
            continue
        rows.append({"row_id": row_id, "line_no": line_no, "values": values})

    rows.sort(key=lambda row: row["row_id"])
    failures.sort(key=lambda item: (int(item.get("line_no") or 0), item.get("row_id", ""), item.get("column", "")))
    return rows, failures, schema_columns


def load_raw_table(path: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    drafts, parse_failures = read_jsonl_drafts(path)
    return validate_drafts(drafts, config, parse_failures)


def normalise_group_map(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return {"format": 1, "tables": {}}, [
            diagnostic(E_GROUP_MAP_INVALID, "group map root must be an object", stage="group_map")
        ]
    if isinstance(raw.get("tables"), dict):
        raw_tables = raw["tables"]
    elif "mapping" in raw:
        key_column = raw.get("key_column")
        if not isinstance(key_column, str) or not key_column:
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    "flat group map requires key_column",
                    stage="group_map",
                )
            )
            key_column = ""
        raw_tables = {key_column: {"mapping": raw.get("mapping"), "groups": raw.get("groups")}}
    else:
        failures.append(
            diagnostic(E_GROUP_MAP_INVALID, "group map must provide tables or mapping", stage="group_map")
        )
        raw_tables = {}

    tables: dict[str, Any] = {}
    for key_column, table in raw_tables.items():
        if not isinstance(key_column, str) or not key_column:
            failures.append(
                diagnostic(E_GROUP_MAP_INVALID, "group map key column must be a non-empty string", stage="group_map")
            )
            continue
        if not isinstance(table, dict):
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    "group map table must be an object",
                    column=key_column,
                    stage="group_map",
                )
            )
            continue
        mapping = table.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    "group mapping must be a non-empty object",
                    column=key_column,
                    stage="group_map",
                )
            )
            continue
        normalized_mapping: dict[str, str] = {}
        for raw_key, raw_group in mapping.items():
            if not isinstance(raw_key, str) or not raw_key:
                failures.append(
                    diagnostic(
                        E_GROUP_MAP_INVALID,
                        "mapped key values must be non-empty strings",
                        column=key_column,
                        stage="group_map",
                    )
                )
                continue
            if not isinstance(raw_group, str) or not raw_group:
                failures.append(
                    diagnostic(
                        E_GROUP_MAP_INVALID,
                        "group names must be non-empty strings",
                        column=key_column,
                        stage="group_map",
                        value=as_plain_cell(raw_group),
                    )
                )
                continue
            normalized_mapping[raw_key] = raw_group
        groups = sorted(set(normalized_mapping.values()))
        declared_groups = table.get("groups")
        if declared_groups is not None:
            if not isinstance(declared_groups, list) or any(not isinstance(group, str) for group in declared_groups):
                failures.append(
                    diagnostic(
                        E_GROUP_MAP_INVALID,
                        "groups must be a list of strings",
                        column=key_column,
                        stage="group_map",
                    )
                )
            else:
                groups = sorted(set(groups) | set(declared_groups))
        tables[key_column] = {"mapping": dict(sorted(normalized_mapping.items())), "groups": groups}
    return {"format": 1, "tables": dict(sorted(tables.items()))}, failures


def load_group_map(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        raw = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_GROUP_MAP_INVALID, f"cannot load group map: {exc}") from exc
    return normalise_group_map(raw)


def validate_group_map_against_config(
    config: dict[str, Any], group_map: dict[str, Any]
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    required_keys = {
        spec["group_key"]
        for spec in config["columns"].values()
        if spec.get("imputer") == "grouped_median"
    }
    for key in sorted(required_keys):
        if key not in group_map.get("tables", {}):
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    f"no group mapping table for group key {key!r}",
                    column=key,
                    stage="group_map",
                )
            )
    for key_column in sorted(group_map.get("tables", {})):
        spec = config["columns"].get(key_column)
        if spec is None:
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    f"group mapping refers to undeclared column {key_column!r}",
                    column=key_column,
                    stage="group_map",
                )
            )
        elif spec.get("type") != "category":
            failures.append(
                diagnostic(
                    E_GROUP_MAP_INVALID,
                    f"group mapping column {key_column!r} is not categorical",
                    column=key_column,
                    stage="group_map",
                )
            )
    return failures


def group_coverage_failures(
    rows: list[dict[str, Any]], config: dict[str, Any], group_map: dict[str, Any]
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            spec["group_key"]
            for spec in config["columns"].values()
            if spec.get("imputer") == "grouped_median"
        }
    )
    failures: list[dict[str, Any]] = []
    for row in rows:
        for key in keys:
            value = row["values"].get(key)
            table = group_map.get("tables", {}).get(key, {})
            mapping = table.get("mapping", {})
            if value is None:
                failures.append(
                    diagnostic(
                        E_GROUP_MAP_INVALID,
                        "group key is missing and cannot be mapped",
                        row_id=row["row_id"],
                        line_no=row.get("line_no"),
                        column=key,
                        stage="group_map",
                    )
                )
            elif value not in mapping:
                failures.append(
                    diagnostic(
                        E_GROUP_MAP_INVALID,
                        f"group key value {value!r} has no mapping",
                        row_id=row["row_id"],
                        line_no=row.get("line_no"),
                        column=key,
                        stage="group_map",
                        value=value,
                    )
                )
    failures.sort(key=lambda item: (item.get("row_id", ""), item.get("column", "")))
    return failures


def data_fingerprint(rows: list[dict[str, Any]]) -> str:
    return sha256_obj(
        [
            {"row_id": row["row_id"], "values": row["values"]}
            for row in sorted(rows, key=lambda item: item["row_id"])
        ]
    )


def group_map_fingerprint(group_map: dict[str, Any]) -> str:
    return sha256_obj(group_map)


def clone_group_map(group_map: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(group_map)


def group_for_row(
    row: dict[str, Any], group_key: str, group_map: dict[str, Any]
) -> str | None:
    value = row["values"].get(group_key)
    if value is None:
        return None
    return group_map.get("tables", {}).get(group_key, {}).get("mapping", {}).get(value)
