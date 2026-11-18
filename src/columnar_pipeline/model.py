from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .canonical import canonical_json, sha256_obj


SUPPORTED_COLUMN_TYPES = frozenset({"int", "fixed", "string"})
NUMERIC_COLUMN_TYPES = frozenset({"int", "fixed"})
SUPPORTED_AGGREGATES = frozenset({"count", "sum", "avg"})
SUPPORTED_SPLIT_STRATEGIES = frozenset({"row_range"})


@dataclass(frozen=True)
class Column:
    name: str
    type_name: str
    values: tuple[Any, ...]
    null_bitmap: tuple[int, ...]
    scale: int = 0

    def value_at(self, row: int) -> Any:
        if self.null_bitmap[row]:
            return None
        return self.values[row]

    def memory_width_bytes(self) -> int:
        if self.type_name in NUMERIC_COLUMN_TYPES:
            return 9
        maximum = 1
        for value in self.values:
            if value is not None:
                maximum = max(maximum, len(str(value).encode("utf-8")))
        return maximum + 1


@dataclass(frozen=True)
class Table:
    name: str
    version: int
    row_count: int
    columns: tuple[Column, ...]
    digest: str

    def column_map(self) -> dict[str, Column]:
        return {column.name: column for column in self.columns}

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)


@dataclass(frozen=True)
class AggregateSpec:
    name: str
    op: str
    column: str
    scale: int


@dataclass(frozen=True)
class Query:
    from_tables: tuple[str, ...]
    group_by: tuple[str, ...]
    aggregates: tuple[AggregateSpec, ...]
    where: dict[str, Any] | None
    where_text: str | None
    query_id: str

    @property
    def referenced_columns(self) -> tuple[str, ...]:
        names = set(self.group_by)
        names.update(aggregate.column for aggregate in self.aggregates if aggregate.column != "*")
        if self.where is not None:
            _collect_filter_columns(self.where, names)
        return tuple(sorted(names))

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "from": list(self.from_tables),
            "where": self.where,
            "group_by": list(self.group_by),
            "aggregates": [
                {
                    "name": aggregate.name,
                    "op": aggregate.op,
                    "column": aggregate.column,
                }
                for aggregate in self.aggregates
            ],
        }


def _collect_filter_columns(node: Mapping[str, Any], target: set[str]) -> None:
    if node.get("op") == "and":
        for term in node["terms"]:
            _collect_filter_columns(term, target)
        return
    target.add(str(node["column"]))


@dataclass(frozen=True)
class Dataset:
    tables: tuple[Table, ...]
    query: Query
    data_digest: str
    injection: dict[str, Any] = field(default_factory=dict)

    def table_map(self) -> dict[str, Table]:
        return {table.name: table for table in self.tables}

    def table(self, name: str) -> Table:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)

    def table_digests(self) -> dict[str, str]:
        return {table.name: table.digest for table in self.tables}

    def table_versions(self) -> dict[str, int]:
        return {table.name: table.version for table in self.tables}


@dataclass(frozen=True)
class RunConfig:
    parallelism: int = 2
    memory_budget_bytes: int = 16_384
    budget_version: int = 1
    split_threshold: int = 8
    split_parts: int = 2
    row_group_size: int = 32
    ledger_batch_size: int = 4
    split_strategy: str = "row_range"
    skew_safety_limit: int = 100_000
    inject_skew_amplify: int = 1
    inject_budget_shortfall: bool = False
    inject_digest_task: str | None = None

    def validate(self) -> None:
        from .errors import data_error

        if not isinstance(self.parallelism, int) or isinstance(self.parallelism, bool) or self.parallelism < 1:
            raise data_error("parallelism must be an integer >= 1", "config.parallelism")
        if (
            not isinstance(self.memory_budget_bytes, int)
            or isinstance(self.memory_budget_bytes, bool)
            or self.memory_budget_bytes < 1
        ):
            raise data_error("memory_budget_bytes must be an integer >= 1", "config.memory_budget_bytes")
        if not isinstance(self.budget_version, int) or isinstance(self.budget_version, bool) or self.budget_version < 1:
            raise data_error("budget_version must be an integer >= 1", "config.budget_version")
        if (
            not isinstance(self.split_threshold, int)
            or isinstance(self.split_threshold, bool)
            or self.split_threshold < 1
        ):
            raise data_error("split_threshold must be an integer >= 1", "config.split_threshold")
        if not isinstance(self.split_parts, int) or isinstance(self.split_parts, bool) or self.split_parts < 2:
            raise data_error("split_parts must be an integer >= 2", "config.split_parts")
        if (
            not isinstance(self.row_group_size, int)
            or isinstance(self.row_group_size, bool)
            or self.row_group_size < 2
        ):
            raise data_error("row_group_size must be an integer >= 2", "config.row_group_size")
        if (
            not isinstance(self.ledger_batch_size, int)
            or isinstance(self.ledger_batch_size, bool)
            or self.ledger_batch_size < 1
        ):
            raise data_error("ledger_batch_size must be an integer >= 1", "config.ledger_batch_size")
        if self.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
            raise data_error(
                f"unsupported split strategy: {self.split_strategy!r}",
                "config.split_strategy",
                details={"supported": sorted(SUPPORTED_SPLIT_STRATEGIES)},
            )
        if (
            not isinstance(self.skew_safety_limit, int)
            or isinstance(self.skew_safety_limit, bool)
            or self.skew_safety_limit < 1
        ):
            raise data_error("skew_safety_limit must be an integer >= 1", "config.skew_safety_limit")
        if (
            not isinstance(self.inject_skew_amplify, int)
            or isinstance(self.inject_skew_amplify, bool)
            or self.inject_skew_amplify < 1
        ):
            raise data_error("inject_skew_amplify must be an integer >= 1", "config.inject_skew_amplify")
        if self.inject_digest_task is not None and not isinstance(self.inject_digest_task, str):
            raise data_error("inject_digest_task must be a string or null", "config.inject_digest_task")

    def effective_memory_budget(self) -> int:
        return 1 if self.inject_budget_shortfall else self.memory_budget_bytes

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "parallelism": self.parallelism,
            "memory_budget_bytes": self.memory_budget_bytes,
            "effective_memory_budget_bytes": self.effective_memory_budget(),
            "budget_version": self.budget_version,
            "split_threshold": self.split_threshold,
            "split_parts": self.split_parts,
            "row_group_size": self.row_group_size,
            "ledger_batch_size": self.ledger_batch_size,
            "split_strategy": self.split_strategy,
            "skew_safety_limit": self.skew_safety_limit,
            "inject_skew_amplify": self.inject_skew_amplify,
            "inject_budget_shortfall": self.inject_budget_shortfall,
            "inject_digest_task": self.inject_digest_task,
        }

    @property
    def config_digest(self) -> str:
        return sha256_obj(self.semantic_payload())

    @property
    def budget_digest(self) -> str:
        return sha256_obj(
            {
                "budget_version": self.budget_version,
                "memory_budget_bytes": self.memory_budget_bytes,
                "parallelism": self.parallelism,
                "effective_memory_budget_bytes": self.effective_memory_budget(),
            }
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["config_digest"] = self.config_digest
        payload["budget_digest"] = self.budget_digest
        return payload

    @classmethod
    def from_json_dict(cls, value: Mapping[str, Any]) -> "RunConfig":
        accepted = {
            "parallelism",
            "memory_budget_bytes",
            "budget_version",
            "split_threshold",
            "split_parts",
            "row_group_size",
            "ledger_batch_size",
            "split_strategy",
            "skew_safety_limit",
            "inject_skew_amplify",
            "inject_budget_shortfall",
            "inject_digest_task",
        }
        payload = {key: value[key] for key in accepted if key in value}
        config = cls(**payload)
        config.validate()
        return config


def query_fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256_obj(dict(payload))
