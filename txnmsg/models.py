"""Domain models and deterministic reporting objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_bytes, digest_value


FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
PREPARED = "prepared"
COMMITTING = "committing"
COMMITTED = "committed"
ABORTED = "aborted"
STATES = (PREPARED, COMMITTING, COMMITTED, ABORTED)
TERMINAL_STATES = frozenset((COMMITTED, ABORTED))
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    PREPARED: frozenset((COMMITTING, ABORTED)),
    COMMITTING: frozenset((COMMITTED, ABORTED)),
    COMMITTED: frozenset(),
    ABORTED: frozenset(),
}


@dataclass(frozen=True)
class Partition:
    partition_id: str
    partition_version: int
    max_position: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "partition_version": self.partition_version,
            "max_position": self.max_position,
        }


@dataclass(frozen=True)
class Message:
    position: int
    payload: Any
    message_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "payload": self.payload,
            "message_digest": self.message_digest,
        }


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    txn_seq: int
    partition_id: str
    local_txn_version: int
    messages: tuple[Message, ...]
    commit_time_ms: int

    def as_spec_dict(self, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "transaction_id": self.transaction_id,
            "txn_seq": self.txn_seq,
            "partition_id": self.partition_id,
            "local_txn_version": self.local_txn_version,
            "messages": [message.as_dict() for message in self.messages],
            "commit_time_ms": self.commit_time_ms,
        }
        if include_digest:
            result["spec_digest"] = self.spec_digest
        return result

    @property
    def spec_digest(self) -> str:
        return digest_value(self.as_spec_dict(include_digest=False))

    @property
    def spec_json(self) -> str:
        return canonical_bytes(self.as_spec_dict(include_digest=False)).decode("utf-8")

    def message_at(self, position: int) -> Message | None:
        for message in self.messages:
            if message.position == position:
                return message
        return None


@dataclass(frozen=True)
class PrefixReport:
    partition_id: str
    partition_version: int
    max_position: int
    prefix: int
    truncated_position: int | None
    truncated_reason: str | None
    truncated_transaction_id: str | None = None
    unknown_intervals: tuple[tuple[int, int], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "partition_version": self.partition_version,
            "max_position": self.max_position,
            "prefix": self.prefix,
            "truncated_position": self.truncated_position,
            "truncated_reason": self.truncated_reason,
            "truncated_transaction_id": self.truncated_transaction_id,
            "unknown_intervals": [[start, end] for start, end in self.unknown_intervals],
        }


@dataclass(frozen=True)
class UnknownTransaction:
    transaction_id: str
    partition_id: str
    intervals: tuple[tuple[int, int], ...]
    reason: str = "committing_after_forced_termination"
    requires_rerun: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "partition_id": self.partition_id,
            "intervals": [[start, end] for start, end in self.intervals],
            "reason": self.reason,
            "requires_rerun": self.requires_rerun,
        }


@dataclass(frozen=True)
class TransitionResult:
    transaction_id: str
    event_seq: int
    previous_state: str | None
    state: str
    prefix: PrefixReport

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "event_seq": self.event_seq,
            "previous_state": self.previous_state,
            "state": self.state,
            "prefix": self.prefix.as_dict(),
        }


@dataclass(frozen=True)
class RecoveryReport:
    recovered: bool
    ledger_format_version: int
    event_cursor: int
    persisted_prefix_repaired: bool
    unknown_transactions: tuple[UnknownTransaction, ...]
    partitions: tuple[PrefixReport, ...]
    note: str = "exit report at the forced-termination instant is unavailable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "recovered": self.recovered,
            "ledger_format_version": self.ledger_format_version,
            "event_cursor": self.event_cursor,
            "persisted_prefix_repaired": self.persisted_prefix_repaired,
            "unknown_transactions": [item.as_dict() for item in self.unknown_transactions],
            "partitions": [item.as_dict() for item in self.partitions],
            "exit_report_available": False,
            "note": self.note,
        }


@dataclass(frozen=True)
class LedgerConclusion:
    ledger_format_version: int
    partition_set_version: int
    transaction_set_version: int
    event_cursor: int
    ledger_digest: str
    partitions: tuple[PrefixReport, ...]
    unknown_transactions: tuple[UnknownTransaction, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ledger_format_version": self.ledger_format_version,
            "partition_set_version": self.partition_set_version,
            "transaction_set_version": self.transaction_set_version,
            "event_cursor": self.event_cursor,
            "ledger_digest": self.ledger_digest,
            "partitions": [item.as_dict() for item in self.partitions],
            "unknown_transactions": [item.as_dict() for item in self.unknown_transactions],
        }


@dataclass(frozen=True)
class ReplayReport:
    status: str
    event_cursor: int
    events_replayed: int
    unknown_transactions: tuple[UnknownTransaction, ...]
    partitions: tuple[PrefixReport, ...]
    transaction_states: tuple[tuple[str, str | None], ...]
    baseline: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "event_cursor": self.event_cursor,
            "events_replayed": self.events_replayed,
            "unknown_transactions": [item.as_dict() for item in self.unknown_transactions],
            "partitions": [item.as_dict() for item in self.partitions],
            "transaction_states": [
                {"transaction_id": transaction_id, "state": state}
                for transaction_id, state in self.transaction_states
            ],
            "baseline": self.baseline,
        }


@dataclass(frozen=True)
class RestoreDiff:
    partition_added: tuple[str, ...]
    partition_removed: tuple[str, ...]
    transaction_added: tuple[str, ...]
    transaction_removed: tuple[str, ...]
    version_changes: tuple[dict[str, Any], ...]
    transaction_state_changes: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "partition_added": list(self.partition_added),
            "partition_removed": list(self.partition_removed),
            "transaction_added": list(self.transaction_added),
            "transaction_removed": list(self.transaction_removed),
            "version_changes": list(self.version_changes),
            "transaction_state_changes": list(self.transaction_state_changes),
        }


@dataclass(frozen=True)
class RestoreResult:
    restored: bool
    replaced: bool
    snapshot_path: str
    backup_path: str | None
    snapshot_ledger_digest: str
    prior_ledger_digest: str
    diff: RestoreDiff
    event_cursor: int
    partitions: tuple[PrefixReport, ...]
    format_version_changed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "restored": self.restored,
            "replaced": self.replaced,
            "snapshot_path": self.snapshot_path,
            "backup_path": self.backup_path,
            "snapshot_ledger_digest": self.snapshot_ledger_digest,
            "prior_ledger_digest": self.prior_ledger_digest,
            "diff": self.diff.as_dict(),
            "event_cursor": self.event_cursor,
            "partitions": [item.as_dict() for item in self.partitions],
            "format_version_changed": self.format_version_changed,
        }


def merge_intervals(positions: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(positions))
    if not ordered:
        return ()
    intervals: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for position in ordered[1:]:
        if position == previous + 1:
            previous = position
            continue
        intervals.append((start, previous))
        start = previous = position
    intervals.append((start, previous))
    return tuple(intervals)
