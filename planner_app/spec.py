"""Workspace JSON loading and strong, atomic specification validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes, digest
from .errors import E_SPEC, E_SYNTAX, AppError, spec_error
from .plan import PlanTree, TableSnapshot, apply_edit, compile_base
from .sql_parser import ParsedQuery, parse_sql


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str
    nulls: tuple[bool, ...]
    values: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "nulls": list(self.nulls),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class TableDef:
    name: str
    columns: tuple[ColumnDef, ...]

    @property
    def row_count(self) -> int:
        return len(self.columns[0].values) if self.columns else 0

    def to_snapshot(self) -> TableSnapshot:
        rows = []
        for index in range(self.row_count):
            rows.append(
                {
                    column.name: (None if column.nulls[index] else column.values[index])
                    for column in self.columns
                }
            )
        return TableSnapshot(
            name=self.name,
            schema=tuple((column.name, column.type) for column in self.columns),
            rows=tuple(rows),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "columns": [column.to_dict() for column in self.columns]}


@dataclass(frozen=True)
class QueryDef:
    name: str
    sql: str
    parsed: ParsedQuery


@dataclass(frozen=True)
class EditBatch:
    name: str
    query: str
    edits: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "query": self.query, "edits": list(self.edits)}


@dataclass(frozen=True)
class WorkspaceSpec:
    format: int
    name: str
    workspace_id: str
    scale_factor: float
    interaction_budget_ms: float
    tables: tuple[TableDef, ...]
    queries: tuple[QueryDef, ...]
    edit_batches: tuple[EditBatch, ...]
    versions: dict[str, int]

    def table_map(self) -> dict[str, TableDef]:
        return {table.name: table for table in self.tables}

    def query_map(self) -> dict[str, QueryDef]:
        return {query.name: query for query in self.queries}

    def table_digest(self) -> str:
        return digest([table.to_dict() for table in self.tables])

    def sql_digest(self) -> str:
        return digest([[query.name, query.sql] for query in self.queries])

    def scale_digest(self) -> str:
        return digest({"scale_factor": self.scale_factor})

    def context_digest(self, query_name: str) -> str:
        query = self.query_map()[query_name]
        return digest(
            {
                "table_digest": self.table_digest(),
                "sql": query.sql,
                "scale_digest": self.scale_digest(),
            }
        )

    def spec_digest(self) -> str:
        return digest(
            {
                "format": self.format,
                "name": self.name,
                "workspace_id": self.workspace_id,
                "scale_factor": self.scale_factor,
                "interaction_budget_ms": self.interaction_budget_ms,
                "tables": [table.to_dict() for table in self.tables],
                "queries": [[query.name, query.sql] for query in self.queries],
                "edit_batches": [batch.to_dict() for batch in self.edit_batches],
            }
        )

    def to_workspace_dict(self) -> dict[str, Any]:
        """Serialise the current declared source for crash-safe session handoff."""

        return {
            "format": 1,
            "name": self.name,
            "workspace_id": self.workspace_id,
            "scale_factor": self.scale_factor,
            "interaction_budget_ms": self.interaction_budget_ms,
            "versions": dict(self.versions),
            "tables": [table.to_dict() for table in self.tables],
            "queries": [{"name": query.name, "sql": query.sql} for query in self.queries],
            "edit_batches": [],
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, path: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(f"{label} must be an object", path)
    return value


def _require_list(value: Any, path: str, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise spec_error(f"{label} must be an array", path)
    return value


def _require_name(value: Any, path: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise spec_error(f"{label} must be a non-empty string", path)
    return value.strip().lower()


def _parse_tables(raw_tables: Any, path: str = "$.tables") -> tuple[TableDef, ...]:
    table_values = _require_list(raw_tables, path, "tables")
    if not table_values:
        raise spec_error("at least one table is required", path)
    table_names: set[str] = set()
    tables: list[TableDef] = []
    for table_index, raw_table in enumerate(table_values):
        table_path = f"{path}[{table_index}]"
        table_object = _require_object(raw_table, table_path, "table")
        table_name = _require_name(table_object.get("name"), f"{table_path}.name", "table name")
        if table_name in table_names:
            raise spec_error(f"duplicate table name {table_name!r}", f"{table_path}.name")
        table_names.add(table_name)
        raw_columns = _require_list(table_object.get("columns"), f"{table_path}.columns", "columns")
        if not raw_columns:
            raise spec_error("table must declare at least one column", f"{table_path}.columns")
        column_names: set[str] = set()
        columns: list[ColumnDef] = []
        expected_rows: int | None = None
        for column_index, raw_column in enumerate(raw_columns):
            column_path = f"{table_path}.columns[{column_index}]"
            column_object = _require_object(raw_column, column_path, "column")
            column_name = _require_name(column_object.get("name"), f"{column_path}.name", "column name")
            if column_name in column_names:
                raise spec_error(f"duplicate column name {column_name!r}", f"{column_path}.name")
            column_names.add(column_name)
            column_type = column_object.get("type")
            if column_type not in {"int", "float"}:
                raise spec_error("column type must be 'int' or 'float'", f"{column_path}.type")
            if "values" in column_object and "rows" in column_object:
                raise spec_error("use either 'values' or 'rows', not both", column_path)
            raw_values = column_object.get("values", column_object.get("rows"))
            values_list = _require_list(raw_values, f"{column_path}.values", "column values")
            raw_nulls = _require_list(column_object.get("nulls"), f"{column_path}.nulls", "null bitmap")
            if len(raw_nulls) != len(values_list):
                raise spec_error(
                    f"null bitmap length {len(raw_nulls)} does not match row count {len(values_list)}",
                    f"{column_path}.nulls",
                )
            if expected_rows is None:
                expected_rows = len(values_list)
            elif expected_rows != len(values_list):
                raise spec_error(
                    f"column row count {len(values_list)} does not match table row count {expected_rows}",
                    f"{column_path}.values",
                )
            normalized_nulls: list[bool] = []
            normalized_values: list[Any] = []
            for row_index, (raw_null, raw_value) in enumerate(zip(raw_nulls, values_list)):
                if not isinstance(raw_null, bool):
                    raise spec_error("null bitmap entries must be booleans", f"{column_path}.nulls[{row_index}]")
                normalized_nulls.append(raw_null)
                if raw_null:
                    normalized_values.append(None)
                    continue
                if raw_value is None:
                    raise spec_error(
                        "non-null row cannot contain a null value", f"{column_path}.values[{row_index}]"
                    )
                if column_type == "int":
                    if not _is_int(raw_value):
                        raise spec_error(
                            "int column contains a non-integer value", f"{column_path}.values[{row_index}]"
                        )
                    normalized_values.append(int(raw_value))
                else:
                    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                        raise spec_error(
                            "float column contains a non-numeric value", f"{column_path}.values[{row_index}]"
                        )
                    numeric = float(raw_value)
                    if not math.isfinite(numeric):
                        raise spec_error(
                            "float column contains a non-finite value", f"{column_path}.values[{row_index}]"
                        )
                    normalized_values.append(numeric)
            columns.append(
                ColumnDef(
                    name=column_name,
                    type=column_type,
                    nulls=tuple(normalized_nulls),
                    values=tuple(normalized_values),
                )
            )
        tables.append(TableDef(name=table_name, columns=tuple(columns)))
    return tuple(tables)


def _parse_queries(
    raw_queries: Any,
    tables: tuple[TableDef, ...],
    scale_factor: float,
    path: str = "$.queries",
) -> tuple[QueryDef, ...]:
    query_values = _require_list(raw_queries, path, "queries")
    if not query_values:
        raise spec_error("at least one query is required", path)
    table_map = {table.name: table for table in tables}
    names: set[str] = set()
    result: list[QueryDef] = []
    for index, raw_query in enumerate(query_values):
        query_path = f"{path}[{index}]"
        query_object = _require_object(raw_query, query_path, "query")
        name = _require_name(query_object.get("name"), f"{query_path}.name", "query name")
        if name in names:
            raise spec_error(f"duplicate query name {name!r}", f"{query_path}.name")
        names.add(name)
        sql = query_object.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise spec_error("query SQL must be a non-empty string", f"{query_path}.sql")
        try:
            parsed = parse_sql(sql)
        except AppError as error:
            if error.code != E_SYNTAX:
                raise
            raise AppError(
                E_SYNTAX,
                error.message,
                position=error.position,
                json_path=f"{query_path}.sql",
            ) from error
        table = table_map.get(parsed.table)
        if table is None:
            raise spec_error(
                f"unknown table {parsed.table!r}", f"{query_path}.sql", position=parsed.table_pos
            )
        try:
            compile_base(parsed, table, scale_factor, name, json_path=f"{query_path}.sql")
        except AppError as error:
            if error.code == E_SPEC:
                raise AppError(
                    E_SPEC,
                    error.message,
                    json_path=error.json_path or f"{query_path}.sql",
                    position=error.position,
                ) from error
            raise
        result.append(QueryDef(name=name, sql=sql, parsed=parsed))
    return tuple(result)


def _parse_edit_batches(
    raw_batches: Any,
    tables: tuple[TableDef, ...],
    queries: tuple[QueryDef, ...],
    scale_factor: float,
    path: str = "$.edit_batches",
) -> tuple[EditBatch, ...]:
    if raw_batches is None:
        return ()
    batch_values = _require_list(raw_batches, path, "edit_batches")
    table_map = {table.name: table for table in tables}
    query_map = {query.name: query for query in queries}
    plans: dict[str, PlanTree] = {
        query.name: compile_base(
            query.parsed,
            table_map[query.parsed.table],
            scale_factor,
            query.name,
            json_path="$.queries",
        )
        for query in queries
    }
    edit_sequences: dict[str, int] = {query.name: 0 for query in queries}
    batches: list[EditBatch] = []
    for batch_index, raw_batch in enumerate(batch_values):
        batch_path = f"{path}[{batch_index}]"
        batch_object = _require_object(raw_batch, batch_path, "edit batch")
        name = _require_name(
            batch_object.get("name", f"batch_{batch_index + 1}"),
            f"{batch_path}.name",
            "batch name",
        )
        query_name = _require_name(batch_object.get("query"), f"{batch_path}.query", "batch query")
        if query_name not in query_map:
            raise spec_error(f"unknown query {query_name!r}", f"{batch_path}.query")
        raw_edits = _require_list(batch_object.get("edits"), f"{batch_path}.edits", "edits")
        if not raw_edits:
            raise spec_error("edit batch must contain at least one edit", f"{batch_path}.edits")
        edits: list[dict[str, Any]] = []
        for edit_index, raw_edit in enumerate(raw_edits):
            edit_path = f"{batch_path}.edits[{edit_index}]"
            edit_object = _require_object(raw_edit, edit_path, "edit")
            edit = json.loads(json.dumps(edit_object, ensure_ascii=False))
            if not isinstance(edit.get("type"), str):
                raise spec_error("edit type must be a string", f"{edit_path}.type")
            edit_sequences[query_name] += 1
            try:
                updated, _effect = apply_edit(plans[query_name], edit, edit_sequences[query_name])
            except AppError as error:
                if error.code in {"E_REF", "E_SPEC"}:
                    detail_path = error.json_path if error.json_path and not error.json_path.startswith("$.edits") else edit_path
                    raise AppError(
                        E_SPEC,
                        error.message,
                        json_path=detail_path,
                        edit_seq=edit_sequences[query_name],
                        node_id=error.node_id,
                    ) from error
                raise
            plans[query_name] = updated
            edits.append(edit)
        batches.append(EditBatch(name=name, query=query_name, edits=tuple(edits)))
    return tuple(batches)


def load_workspace(path: str | Path) -> WorkspaceSpec:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise spec_error(f"cannot read workspace JSON: {error}", "$") from error
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise spec_error(f"invalid JSON: {error.msg}", f"$@{error.pos}") from error
    root = _require_object(raw, "$", "workspace")
    if root.get("format") != 1:
        raise spec_error("format must be integer 1", "$.format")
    name = root.get("name", "workspace")
    if not isinstance(name, str) or not name.strip():
        raise spec_error("workspace name must be a non-empty string", "$.name")
    name = name.strip()
    workspace_id = root.get("workspace_id", name.lower().replace(" ", "-"))
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise spec_error("workspace_id must be a non-empty string", "$.workspace_id")
    scale_factor = root.get("scale_factor", 1.0)
    if not isinstance(scale_factor, (int, float)) or isinstance(scale_factor, bool):
        raise spec_error("scale_factor must be numeric", "$.scale_factor")
    scale_factor = float(scale_factor)
    if not math.isfinite(scale_factor) or scale_factor <= 0:
        raise spec_error("scale_factor must be positive and finite", "$.scale_factor")
    budget = root.get("interaction_budget_ms", 50.0)
    if not isinstance(budget, (int, float)) or isinstance(budget, bool):
        raise spec_error("interaction_budget_ms must be numeric", "$.interaction_budget_ms")
    budget = float(budget)
    if not math.isfinite(budget) or budget < 0:
        raise spec_error("interaction_budget_ms must be non-negative and finite", "$.interaction_budget_ms")

    raw_versions = root.get("versions", {})
    if not isinstance(raw_versions, dict):
        raise spec_error("versions must be an object", "$.versions")
    versions: dict[str, int] = {}
    for key in ("tables", "sql", "edits", "scale"):
        raw_version = raw_versions.get(key, 1)
        if not _is_int(raw_version) or raw_version < 1:
            raise spec_error(f"version {key!r} must be a positive integer", f"$.versions.{key}")
        versions[key] = int(raw_version)

    tables = _parse_tables(root.get("tables"), "$.tables")
    queries = _parse_queries(root.get("queries"), tables, scale_factor, "$.queries")
    edit_batches = _parse_edit_batches(root.get("edit_batches"), tables, queries, scale_factor, "$.edit_batches")
    return WorkspaceSpec(
        format=1,
        name=name,
        workspace_id=workspace_id,
        scale_factor=scale_factor,
        interaction_budget_ms=budget,
        tables=tables,
        queries=queries,
        edit_batches=edit_batches,
        versions=versions,
    )


def replace_query_sql(spec: WorkspaceSpec, query_name: str, sql: str) -> WorkspaceSpec:
    """Return a fully validated replacement spec; no partial state escapes."""

    query_map = spec.query_map()
    if query_name not in query_map:
        raise spec_error(f"unknown query {query_name!r}", "$.queries")
    raw_queries = [{"name": query.name, "sql": sql if query.name == query_name else query.sql} for query in spec.queries]
    queries = _parse_queries(raw_queries, spec.tables, spec.scale_factor, "$.queries")
    versions = dict(spec.versions)
    versions["sql"] = versions.get("sql", 1) + 1
    return replace(spec, queries=queries, edit_batches=(), versions=versions)


def replace_tables(spec: WorkspaceSpec, raw_tables: Any) -> WorkspaceSpec:
    tables = _parse_tables(raw_tables, "$.tables")
    raw_queries = [{"name": query.name, "sql": query.sql} for query in spec.queries]
    queries = _parse_queries(raw_queries, tables, spec.scale_factor, "$.queries")
    versions = dict(spec.versions)
    versions["tables"] = versions.get("tables", 1) + 1
    return replace(spec, tables=tables, queries=queries, edit_batches=(), versions=versions)


def replace_scale_factor(spec: WorkspaceSpec, scale_factor: float) -> WorkspaceSpec:
    if not isinstance(scale_factor, (int, float)) or isinstance(scale_factor, bool):
        raise spec_error("scale_factor must be numeric", "$.scale_factor")
    value = float(scale_factor)
    if not math.isfinite(value) or value <= 0:
        raise spec_error("scale_factor must be positive and finite", "$.scale_factor")
    queries = _parse_queries(
        [{"name": query.name, "sql": query.sql} for query in spec.queries],
        spec.tables,
        value,
        "$.queries",
    )
    versions = dict(spec.versions)
    versions["scale"] = versions.get("scale", 1) + 1
    return replace(spec, scale_factor=value, queries=queries, edit_batches=(), versions=versions)


def tables_as_json(spec: WorkspaceSpec) -> list[dict[str, Any]]:
    return [table.to_dict() for table in spec.tables]


def canonical_edit_batch(batch: EditBatch) -> bytes:
    return canonical_bytes(batch.to_dict())
