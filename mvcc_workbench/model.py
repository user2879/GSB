from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_json, digest_json


@dataclass(frozen=True)
class Transaction:
    txn_id: str
    start_seq: int
    outcome: str
    commit_seq: int | None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "txn_id": self.txn_id,
            "start_seq": self.start_seq,
            "outcome": self.outcome,
        }
        if self.commit_seq is not None:
            result["commit_seq"] = self.commit_seq
        return result


@dataclass(frozen=True)
class RecordVersion:
    row_key: str
    version_no: int
    txn_id: str
    value_summary: dict[str, Any]
    value_digest: str
    prev_version: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_key": self.row_key,
            "version_no": self.version_no,
            "txn_id": self.txn_id,
            "value_summary": self.value_summary,
            "value_digest": self.value_digest,
            "prev_version": self.prev_version,
        }


@dataclass(frozen=True)
class ReadPoint:
    read_point_id: str
    snapshot_seq: int
    reader_txn_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "read_point_id": self.read_point_id,
            "snapshot_seq": self.snapshot_seq,
            "reader_txn_id": self.reader_txn_id,
        }


@dataclass(frozen=True)
class PredicateCondition:
    column: str
    op: str
    value: int | float

    def as_dict(self) -> dict[str, Any]:
        return {"column": self.column, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class Predicate:
    mode: str
    conditions: tuple[PredicateCondition, ...]

    @property
    def column_count(self) -> int:
        return len(self.conditions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "conditions": [item.as_dict() for item in self.conditions],
        }


@dataclass(frozen=True)
class Query:
    query_id: str
    read_point_id: str
    row_key: str
    predicate: Predicate
    priority: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "predicate": self.predicate.as_dict(),
            "priority": self.priority,
        }


@dataclass(frozen=True)
class Budget:
    global_cost_budget: int
    round_cost_budget: int
    global_concurrency: int
    per_read_point_concurrency: int
    task_timeout_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_cost_budget": self.global_cost_budget,
            "round_cost_budget": self.round_cost_budget,
            "global_concurrency": self.global_concurrency,
            "per_read_point_concurrency": self.per_read_point_concurrency,
            "task_timeout_ms": self.task_timeout_ms,
        }


@dataclass(frozen=True)
class VisibilityPolicy:
    own_uncommitted_visible: bool
    own_write_ignores_snapshot: bool
    declaration: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "own_uncommitted_visible": self.own_uncommitted_visible,
            "own_write_ignores_snapshot": self.own_write_ignores_snapshot,
            "declaration": self.declaration,
        }


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    dataset_id: str
    txn_set_version: int
    chain_version: int
    budget_version: int
    read_point_set_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "txn_set_version": self.txn_set_version,
            "chain_version": self.chain_version,
            "budget_version": self.budget_version,
            "read_point_set_version": self.read_point_set_version,
        }


@dataclass(frozen=True)
class Dataset:
    manifest: Manifest
    transactions: tuple[Transaction, ...]
    records: tuple[RecordVersion, ...]
    read_points: tuple[ReadPoint, ...]
    queries: tuple[Query, ...]
    budget: Budget
    visibility_policy: VisibilityPolicy
    chains: dict[str, tuple[RecordVersion, ...]] = field(compare=False)
    txns_by_id: dict[str, Transaction] = field(compare=False)
    read_points_by_id: dict[str, ReadPoint] = field(compare=False)
    queries_by_id: dict[str, Query] = field(compare=False)

    def as_declaration_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "transactions": [item.as_dict() for item in self.transactions],
            "records": [item.as_dict() for item in self.records],
            "read_points": [item.as_dict() for item in self.read_points],
            "queries": [item.as_dict() for item in self.queries],
            "budget": self.budget.as_dict(),
            "visibility_policy": self.visibility_policy.as_dict(),
        }

    def txn_digest(self) -> str:
        return digest_json([item.as_dict() for item in self.transactions])

    def chain_digest(self) -> str:
        return digest_json([item.as_dict() for item in self.records])

    def read_point_digest(self) -> str:
        return digest_json([item.as_dict() for item in self.read_points])

    def query_digest(self) -> str:
        return digest_json([item.as_dict() for item in self.queries])

    def budget_digest(self) -> str:
        return digest_json(
            {
                "budget": self.budget.as_dict(),
                "visibility_policy": self.visibility_policy.as_dict(),
            }
        )

    def metadata_digest(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "txn_digest": self.txn_digest(),
            "chain_digest": self.chain_digest(),
            "read_point_digest": self.read_point_digest(),
            "query_digest": self.query_digest(),
            "budget_digest": self.budget_digest(),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_declaration_dict())

    def chain(self, row_key: str) -> tuple[RecordVersion, ...]:
        return self.chains[row_key]

    def query_refs_for_txn(self, txn_id: str) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for query in self.queries:
            if any(record.txn_id == txn_id for record in self.chains[query.row_key]):
                result.add((query.read_point_id, query.row_key))
        return result

    def sorted_queries(self, *, priority_first: bool = False) -> tuple[Query, ...]:
        if priority_first:
            return tuple(
                sorted(
                    self.queries,
                    key=lambda item: (
                        item.priority,
                        item.read_point_id,
                        item.query_id,
                    ),
                )
            )
        return tuple(
            sorted(
                self.queries,
                key=lambda item: (
                    item.read_point_id,
                    item.row_key,
                    item.query_id,
                ),
            )
        )


def stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
