"""Immutable domain model and deterministic indexes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import sha256_digest, version_label

OUTCOME_COMMITTED = "committed"
OUTCOME_ROLLEDBACK = "rolledback"
OUTCOME_PENDING = "pending"
EXTERNAL_OUTCOMES = frozenset({OUTCOME_COMMITTED, OUTCOME_ROLLEDBACK})


@dataclass(frozen=True)
class Transaction:
    id: str
    start_seq: int
    outcome: str
    commit_seq: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_seq": self.start_seq,
            "outcome": self.outcome,
            "commit_seq": self.commit_seq,
        }


@dataclass(frozen=True)
class VersionRecord:
    row_key: str
    tx_id: str
    version: int
    value_digest: str
    prev_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_key": self.row_key,
            "tx_id": self.tx_id,
            "version": self.version,
            "value_digest": self.value_digest,
            "prev_version": self.prev_version,
        }


@dataclass(frozen=True)
class ReadPoint:
    id: str
    snapshot_seq: int
    tx_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "snapshot_seq": self.snapshot_seq,
            "tx_id": self.tx_id,
        }


@dataclass(frozen=True)
class Query:
    id: str
    read_point_id: str
    row_key: str
    predicate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "read_point_id": self.read_point_id,
            "row_key": self.row_key,
            "predicate": self.predicate,
        }


@dataclass(frozen=True)
class Budget:
    global_concurrency: int
    per_read_point_concurrency: int

    def to_dict(self) -> dict[str, int]:
        return {
            "global_concurrency": self.global_concurrency,
            "per_read_point_concurrency": self.per_read_point_concurrency,
        }


@dataclass(frozen=True)
class Spec:
    name: str
    format_version: int
    transactions: tuple[Transaction, ...]
    records: tuple[VersionRecord, ...]
    read_points: tuple[ReadPoint, ...]
    queries: tuple[Query, ...]
    budget: Budget
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": "mvcc-diagnostic-spec",
            "format_version": self.format_version,
            "name": self.name,
            "transactions": [item.to_dict() for item in sorted(self.transactions, key=lambda x: x.id)],
            "records": [
                item.to_dict()
                for item in sorted(self.records, key=lambda x: (x.row_key, x.version))
            ],
            "read_points": [
                item.to_dict() for item in sorted(self.read_points, key=lambda x: x.id)
            ],
            "queries": [item.to_dict() for item in sorted(self.queries, key=lambda x: x.id)],
            "budget": self.budget.to_dict(),
        }
        if self.notes is not None:
            result["notes"] = self.notes
        return result


@dataclass
class SpecIndex:
    spec: Spec
    tx_by_id: dict[str, Transaction] = field(init=False)
    read_point_by_id: dict[str, ReadPoint] = field(init=False)
    query_by_id: dict[str, Query] = field(init=False)
    records_by_row_desc: dict[str, tuple[VersionRecord, ...]] = field(init=False)
    records_by_row_asc: dict[str, tuple[VersionRecord, ...]] = field(init=False)
    version_by_row_version: dict[tuple[str, int], VersionRecord] = field(init=False)
    queries_by_row: dict[str, tuple[Query, ...]] = field(init=False)
    query_pairs: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self) -> None:
        self.tx_by_id = {item.id: item for item in self.spec.transactions}
        self.read_point_by_id = {item.id: item for item in self.spec.read_points}
        self.query_by_id = {item.id: item for item in self.spec.queries}

        grouped: dict[str, list[VersionRecord]] = {}
        for record in self.spec.records:
            grouped.setdefault(record.row_key, []).append(record)
        self.records_by_row_asc = {
            key: tuple(sorted(items, key=lambda x: x.version))
            for key, items in sorted(grouped.items())
        }
        self.records_by_row_desc = {
            key: tuple(reversed(items)) for key, items in self.records_by_row_asc.items()
        }
        self.version_by_row_version = {
            (record.row_key, record.version): record for record in self.spec.records
        }

        by_row: dict[str, list[Query]] = {}
        for query in self.spec.queries:
            by_row.setdefault(query.row_key, []).append(query)
        self.queries_by_row = {
            key: tuple(sorted(items, key=lambda x: x.id))
            for key, items in sorted(by_row.items())
        }
        self.query_pairs = tuple(
            sorted({(query.read_point_id, query.row_key) for query in self.spec.queries})
        )

    @property
    def row_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.records_by_row_desc))

    @property
    def sorted_transactions(self) -> tuple[Transaction, ...]:
        return tuple(sorted(self.spec.transactions, key=lambda x: x.id))

    @property
    def sorted_records(self) -> tuple[VersionRecord, ...]:
        return tuple(sorted(self.spec.records, key=lambda x: (x.row_key, x.version)))

    @property
    def sorted_read_points(self) -> tuple[ReadPoint, ...]:
        return tuple(sorted(self.spec.read_points, key=lambda x: x.id))

    @property
    def sorted_queries(self) -> tuple[Query, ...]:
        return tuple(sorted(self.spec.queries, key=lambda x: x.id))

    def chain(self, row_key: str) -> tuple[VersionRecord, ...]:
        return self.records_by_row_desc.get(row_key, ())


def build_index(spec: Spec) -> SpecIndex:
    return SpecIndex(spec)


def transaction_set_version(spec: Spec) -> str:
    payload = {
        "kind": "transaction-set",
        "format_version": spec.format_version,
        "items": [tx.to_dict() for tx in sorted(spec.transactions, key=lambda x: x.id)],
    }
    return version_label("txset", payload)


def chain_digest(spec: Spec) -> str:
    payload = {
        "kind": "version-chains",
        "format_version": spec.format_version,
        "items": [
            record.to_dict()
            for record in sorted(spec.records, key=lambda x: (x.row_key, x.version))
        ],
    }
    return sha256_digest(payload)


def read_point_set_version(spec: Spec) -> str:
    payload = {
        "kind": "read-point-set",
        "format_version": spec.format_version,
        "read_points": [
            item.to_dict() for item in sorted(spec.read_points, key=lambda x: x.id)
        ],
        "queries": [item.to_dict() for item in sorted(spec.queries, key=lambda x: x.id)],
    }
    return version_label("rpset", payload)


def budget_version(spec: Spec) -> str:
    return version_label("budget", {"format_version": spec.format_version, **spec.budget.to_dict()})


def spec_digest(spec: Spec) -> str:
    return sha256_digest(
        {
            "kind": "mvcc-spec",
            "name": spec.name,
            "format_version": spec.format_version,
            "transactions": transaction_set_version(spec),
            "chains": chain_digest(spec),
            "read_points": read_point_set_version(spec),
            "budget": budget_version(spec),
        }
    )


def conclusion_digest(results: Iterable[Any]) -> str:
    items: list[Any] = []
    for result in results:
        if hasattr(result, "to_dict"):
            items.append(result.to_dict())
        elif dataclasses.is_dataclass(result):
            items.append(dataclasses.asdict(result))
        else:
            items.append(result)
    return sha256_digest(
        {
            "kind": "visibility-conclusions",
            "format_version": 1,
            "items": sorted(items, key=lambda x: str(x.get("query_id", ""))),
        }
    )
