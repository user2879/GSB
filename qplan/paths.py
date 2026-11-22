from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .data import quote_identifier
from .spec import IndexSpec, QuerySpec, Spec


@dataclass(frozen=True)
class AccessPath:
    id: str
    kind: str
    driving_table: str
    index_name: str | None
    index_columns: tuple[str, ...]
    covering: bool
    join_order: tuple[str, ...]
    sql: str
    params: tuple[Any, ...]
    output_columns: tuple[dict[str, str], ...]

    @property
    def uses_index(self) -> bool:
        return self.index_name is not None

    @property
    def sql_digest(self) -> str:
        payload = json.dumps(
            {"sql": self.sql, "params": list(self.params)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "driving_table": self.driving_table,
            "index_name": self.index_name,
            "index_columns": list(self.index_columns),
            "covering": self.covering,
            "join_order": list(self.join_order),
            "sql": self.sql,
            "params": list(self.params),
            "sql_digest": self.sql_digest,
            "output_columns": list(self.output_columns),
        }


def _table_clause(table: str, index_name: str | None, *, force_no_index: bool) -> str:
    quoted = quote_identifier(table)
    clause = f"{quoted} AS {quoted}"
    if index_name is not None:
        clause += f" INDEXED BY {quote_identifier(index_name)}"
    elif force_no_index:
        clause += " NOT INDEXED"
    return clause


def _filter_sql(filter_spec, params: list[Any]) -> str:
    column = f"{quote_identifier(filter_spec.table)}.{quote_identifier(filter_spec.column)}"
    op = filter_spec.op
    if op == "between":
        params.extend(filter_spec.value)
        return f"{column} BETWEEN ? AND ?"
    if op == "in":
        params.extend(filter_spec.value)
        return f"{column} IN ({','.join('?' for _ in filter_spec.value)})"
    sql_op = {
        "eq": "=",
        "ne": "<>",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
    }[op]
    params.append(filter_spec.value)
    return f"{column} {sql_op} ?"


def build_path(
    spec: Spec,
    query: QuerySpec,
    *,
    kind: str,
    driving_table: str,
    index: IndexSpec | None,
    covering: bool,
) -> AccessPath:
    join_order = (driving_table,) + tuple(table for table in query.tables if table != driving_table)
    select_sql = ", ".join(
        f"{quote_identifier(ref.table)}.{quote_identifier(ref.column)} AS "
        f"{quote_identifier(ref.table + '.' + ref.column)}"
        for ref in query.select
    )
    table_sql = []
    for table in join_order:
        # A full-scan baseline forces a table scan of its driving table.  Join
        # probes may still use declared physical indexes; otherwise a 120k x
        # 20k diagnostic join degenerates into billions of nested scans and is
        # no longer a useful local baseline.
        force_no_index = kind == "full_scan" and table == driving_table
        table_sql.append(
            _table_clause(
                table,
                index.name if index is not None and table == driving_table else None,
                force_no_index=force_no_index,
            )
        )
    params: list[Any] = []
    conditions: list[str] = []
    for join in query.joins:
        conditions.append(
            f"{quote_identifier(join.left_table)}.{quote_identifier(join.left_column)} = "
            f"{quote_identifier(join.right_table)}.{quote_identifier(join.right_column)}"
        )
    for filter_spec in query.filters:
        conditions.append(_filter_sql(filter_spec, params))
    sql = f"SELECT {select_sql} FROM " + " CROSS JOIN ".join(table_sql)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if index is None:
        path_id = f"scan:{driving_table}"
    else:
        suffix = "cover" if covering else ("single" if len(index.columns) == 1 else "composite")
        path_id = f"{index.name}:{suffix}"
    output_columns = tuple(ref.to_dict() for ref in query.select)
    return AccessPath(
        id=path_id,
        kind=kind,
        driving_table=driving_table,
        index_name=index.name if index is not None else None,
        index_columns=index.columns if index is not None else (),
        covering=covering,
        join_order=join_order,
        sql=sql,
        params=tuple(params),
        output_columns=output_columns,
    )


def build_candidate_paths(spec: Spec, query: QuerySpec) -> list[AccessPath]:
    """Enumerate scan, single/composite index and covering alternatives.

    Candidate indexes are sorted by (table, index name) so the path list is
    independent of declaration order.  A covering alternative supersedes the
    non-covering alternative for the same index because SQLite will prefer the
    covering physical plan when it is available.
    """

    paths: list[AccessPath] = []
    for driving_table in query.tables:
        paths.append(
            build_path(
                spec,
                query,
                kind="full_scan",
                driving_table=driving_table,
                index=None,
                covering=False,
            )
        )
    table_order = {name: index for index, name in enumerate(spec.table_names)}
    indexes = [
        index
        for table in spec.tables
        for index in table.candidate_indexes
        if index.table in query.tables
    ]
    indexes.sort(key=lambda item: (table_order[item.table], item.name))
    for index in indexes:
        table = spec.table(index.table)
        needed = set(query.needed_columns(index.table, spec))
        covering = needed.issubset(set(index.columns))
        relevant_columns = {item.column for item in query.filters_for(index.table)}
        for join in query.joins_for(index.table):
            relevant_columns.update(join.columns_for(index.table))
        first_is_usable = bool(index.columns) and index.columns[0] in relevant_columns
        if covering:
            paths.append(
                build_path(
                    spec,
                    query,
                    kind="covering_index",
                    driving_table=index.table,
                    index=index,
                    covering=True,
                )
            )
        elif first_is_usable:
            kind = "single_index" if len(index.columns) == 1 else "composite_index"
            paths.append(
                build_path(
                    spec,
                    query,
                    kind=kind,
                    driving_table=index.table,
                    index=index,
                    covering=False,
                )
            )
    unique: dict[str, AccessPath] = {}
    for path in paths:
        unique[path.id] = path
    return sorted(unique.values(), key=lambda item: item.id)


def full_scan_baseline(spec: Spec, query: QuerySpec) -> AccessPath:
    return build_path(
        spec,
        query,
        kind="full_scan",
        driving_table=query.tables[0],
        index=None,
        covering=False,
    )
