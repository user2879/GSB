from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal


ScalarType = Literal["integer", "float"]


@dataclass(frozen=True)
class Column:
    name: str
    type: ScalarType


@dataclass(frozen=True)
class HistogramBin:
    lo: Decimal
    hi: Decimal
    count: int


@dataclass(frozen=True)
class Histogram:
    version: int
    bins: tuple[HistogramBin, ...]
    digest: str = ""


@dataclass
class Table:
    name: str
    columns: tuple[Column, ...]
    rows: list[tuple[Any, ...]]
    null_bitmap: list[list[int]]
    histograms: dict[str, Histogram]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def column_index(self, name: str) -> int:
        for index, column in enumerate(self.columns):
            if column.name == name:
                return index
        raise KeyError(name)

    def column(self, name: str) -> Column:
        return self.columns[self.column_index(name)]


@dataclass(frozen=True)
class ColumnRef:
    table: str
    column: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.column}"

    def as_list(self) -> list[str]:
        return [self.table, self.column]


@dataclass(frozen=True)
class FilterPredicate:
    column: ColumnRef
    op: str
    value: int | float | None
    is_null_test: bool = False

    @property
    def key(self) -> str:
        value = "NULL" if self.value is None else repr(self.value)
        return f"{self.column.key}:{self.op}:{value}"

    def canonical(self) -> dict[str, Any]:
        return {
            "column": self.column.as_list(),
            "op": self.op,
            "value": self.value,
            "is_null_test": self.is_null_test,
        }


@dataclass(frozen=True)
class JoinPredicate:
    left: ColumnRef
    right: ColumnRef
    op: str = "="

    def canonical(self) -> dict[str, Any]:
        left, right = self.left, self.right
        if (right.table, right.column) < (left.table, left.column):
            left, right = right, left
        return {"left": left.as_list(), "right": right.as_list(), "op": self.op}

    @property
    def key(self) -> str:
        data = self.canonical()
        return f"{data['left'][0]}.{data['left'][1]}{self.op}{data['right'][0]}.{data['right'][1]}"


@dataclass(frozen=True)
class SelectItem:
    column: ColumnRef | None
    aggregate: str | None
    alias: str
    count_star: bool = False

    def canonical(self) -> dict[str, Any]:
        return {
            "aggregate": self.aggregate,
            "alias": self.alias,
            "column": self.column.as_list() if self.column else None,
            "count_star": self.count_star,
        }


@dataclass(frozen=True)
class Query:
    sql: str
    source_order: tuple[str, ...]
    tables: tuple[str, ...]
    joins: tuple[JoinPredicate, ...]
    filters: tuple[FilterPredicate, ...]
    select: tuple[SelectItem, ...]
    group_by: tuple[ColumnRef, ...]
    order_by: tuple[ColumnRef, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "tables": sorted(self.tables),
            "joins": sorted((predicate.canonical() for predicate in self.joins), key=canonical_key),
            "filters": sorted((predicate.canonical() for predicate in self.filters), key=canonical_key),
            "select": [item.canonical() for item in self.select],
            "group_by": [ref.as_list() for ref in self.group_by],
            "order_by": [ref.as_list() for ref in self.order_by],
        }


def canonical_key(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Tuning:
    seed: int = 20240601
    fixed_point_scale: int = 1000
    exact_scan_threshold_rows: int = 100
    sample_size: int = 128
    confidence_level: float = 0.95
    max_candidates: int = 24
    page_size_bytes: int = 4096
    row_overhead_bytes: int = 16
    global_concurrency: int = 4
    per_table_scan_quota: int = 1
    memory_budget_bytes: int = 4 * 1024 * 1024
    bias_threshold_ratio: float = 100000.0
    comparison_weight: int = 1
    io_page_weight: int = 25
    memory_weight: int = 1

    def canonical(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fixed_point_scale": self.fixed_point_scale,
            "exact_scan_threshold_rows": self.exact_scan_threshold_rows,
            "sample_size": self.sample_size,
            "confidence_level": self.confidence_level,
            "max_candidates": self.max_candidates,
            "page_size_bytes": self.page_size_bytes,
            "row_overhead_bytes": self.row_overhead_bytes,
            "global_concurrency": self.global_concurrency,
            "per_table_scan_quota": self.per_table_scan_quota,
            "memory_budget_bytes": self.memory_budget_bytes,
            "bias_threshold_ratio": self.bias_threshold_ratio,
            "comparison_weight": self.comparison_weight,
            "io_page_weight": self.io_page_weight,
            "memory_weight": self.memory_weight,
        }


@dataclass(frozen=True)
class JoinStep:
    right_table: str
    algorithm: str
    predicates: tuple[JoinPredicate, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "right_table": self.right_table,
            "algorithm": self.algorithm,
            "predicates": sorted((item.canonical() for item in self.predicates), key=canonical_key),
        }


@dataclass(frozen=True)
class Plan:
    query_digest: str
    table_order: tuple[str, ...]
    steps: tuple[JoinStep, ...]
    candidate_id: str
    signature: str
    origin: str = "enumerated"
    baseline_reason: str | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "query_digest": self.query_digest,
            "table_order": list(self.table_order),
            "steps": [step.canonical() for step in self.steps],
        }


@dataclass(frozen=True)
class Workload:
    format_version: int
    tables: tuple[Table, ...]
    query: Query
    tuning: Tuning
    digest: str

    def table_map(self) -> dict[str, Table]:
        return {table.name: table for table in self.tables}

    def canonical(self, *, include_rows: bool = True) -> dict[str, Any]:
        tables: list[dict[str, Any]] = []
        for table in self.tables:
            item: dict[str, Any] = {
                "name": table.name,
                "columns": [{"name": col.name, "type": col.type} for col in table.columns],
                "null_bitmap": table.null_bitmap,
                "histograms": {
                    name: {
                        "version": hist.version,
                        "bins": [
                            {"lo": str(bin.lo), "hi": str(bin.hi), "count": bin.count}
                            for bin in hist.bins
                        ],
                    }
                    for name, hist in sorted(table.histograms.items())
                },
            }
            if include_rows:
                item["rows"] = table.rows
            tables.append(item)
        return {
            "format_version": self.format_version,
            "tables": tables,
            "query": {"sql": self.query.sql},
            "tuning": self.tuning.canonical(),
        }

