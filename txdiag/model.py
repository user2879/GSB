"""Normalized immutable specification and runtime models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ISOLATION_LEVELS = (
    "read_uncommitted",
    "read_committed",
    "repeatable_read",
    "serializable",
)

ISOLATION_ALIASES = {
    "read_uncommitted": "read_uncommitted",
    "read uncommitted": "read_uncommitted",
    "读未提交": "read_uncommitted",
    "read_committed": "read_committed",
    "read committed": "read_committed",
    "读已提交": "read_committed",
    "repeatable_read": "repeatable_read",
    "repeatable read": "repeatable_read",
    "可重复读": "repeatable_read",
    "serializable": "serializable",
    "串行化": "serializable",
}


@dataclass(frozen=True)
class Operation:
    seq: int
    kind: str
    row: str
    value: Any = None
    has_value: bool = False


@dataclass(frozen=True)
class TransactionSpec:
    txid: str
    start_seq: int
    isolation: str
    operations: tuple[Operation, ...]

    @property
    def priority(self) -> tuple[int, str]:
        return (self.start_seq, self.txid)


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    kind: str = "reference"


@dataclass(frozen=True)
class InitialVersion:
    value: Any
    version_no: int
    status: str


@dataclass(frozen=True)
class Spec:
    schema_version: int
    row_keys: tuple[str, ...]
    dependency_edges: tuple[DependencyEdge, ...]
    transactions: tuple[TransactionSpec, ...]
    initial_versions: Mapping[str, InitialVersion]
    spec_sha256: str
    source_path: str
    operations_path: str | None = None
    normalized: Mapping[str, Any] = field(default_factory=dict)

    @property
    def row_index(self) -> dict[str, int]:
        return {row: idx for idx, row in enumerate(self.row_keys)}

    @property
    def tx_by_id(self) -> dict[str, TransactionSpec]:
        return {tx.txid: tx for tx in self.transactions}


@dataclass(frozen=True)
class RangeThresholds:
    tx_count: int = 8
    row_keys: int = 12
    dependency_edges: int = 10
    total: int = 24


@dataclass(frozen=True)
class RuntimeConfig:
    max_running: int = 4
    quotas: Mapping[str, int] = field(default_factory=dict)
    wait_timeout_ms: int = 2000
    wait_budget_ms: int = 10000
    lock_path: str = "auto"
    thresholds: RangeThresholds = field(default_factory=RangeThresholds)
    inject: tuple[str, ...] = ()
    logical_tick_ms: int = 10

    def quota_for(self, isolation: str) -> int:
        return int(self.quotas.get(isolation, self.max_running))

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_running": self.max_running,
            "quotas": dict(sorted(self.quotas.items())),
            "wait_timeout_ms": self.wait_timeout_ms,
            "wait_budget_ms": self.wait_budget_ms,
            "lock_path": self.lock_path,
            "thresholds": {
                "tx_count": self.thresholds.tx_count,
                "row_keys": self.thresholds.row_keys,
                "dependency_edges": self.thresholds.dependency_edges,
                "total": self.thresholds.total,
            },
            "inject": list(self.inject),
            "logical_tick_ms": self.logical_tick_ms,
        }


def choose_lock_path(spec: Spec, config: RuntimeConfig) -> tuple[str, dict[str, Any]]:
    if config.lock_path in ("row", "range"):
        return config.lock_path, {"reason": "explicit", "score": None}
    t = config.thresholds
    tx_count = len(spec.transactions)
    rows = len(spec.row_keys)
    edges = len(spec.dependency_edges)
    total = tx_count + rows + edges
    reasons: list[str] = []
    if tx_count >= t.tx_count:
        reasons.append(f"transactions>={t.tx_count}")
    if rows >= t.row_keys:
        reasons.append(f"row_keys>={t.row_keys}")
    if edges >= t.dependency_edges:
        reasons.append(f"dependency_edges>={t.dependency_edges}")
    if total >= t.total:
        reasons.append(f"total>={t.total}")
    selected = "range" if reasons else "row"
    return selected, {
        "reason": "threshold" if reasons else "below_threshold",
        "signals": reasons,
        "score": {"transactions": tx_count, "row_keys": rows,
                  "dependency_edges": edges, "total": total},
        "thresholds": config.as_dict()["thresholds"],
    }

