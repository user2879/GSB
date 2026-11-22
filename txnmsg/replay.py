"""Pure event replay and deterministic consumable-prefix derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import message_digest
from .models import (
    ABORTED,
    ALLOWED_TRANSITIONS,
    COMMITTED,
    COMMITTING,
    PREPARED,
    Partition,
    PrefixReport,
    Transaction,
    UnknownTransaction,
    merge_intervals,
)


@dataclass
class ReplayMismatch(Exception):
    event_seq: int | None
    transaction_id: str | None
    field: str
    expected: Any
    actual: Any

    def __post_init__(self) -> None:
        super().__init__(
            f"replay mismatch at event {self.event_seq}, "
            f"transaction {self.transaction_id}, field {self.field}"
        )


@dataclass(frozen=True)
class ReplayOutcome:
    event_cursor: int
    events_replayed: int
    states: Mapping[str, str | None]
    final_event_seq: Mapping[str, int | None]
    visibility: tuple[dict[str, Any], ...]
    partitions: tuple[PrefixReport, ...]
    unknown_transactions: tuple[UnknownTransaction, ...]

    def transaction_state_rows(self) -> tuple[tuple[str, str | None], ...]:
        return tuple(sorted(self.states.items(), key=lambda item: item[0]))


def _normalise_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_seq": raw.get("event_seq"),
        "transaction_id": raw.get("transaction_id"),
        "state": raw.get("state"),
        "local_txn_version": raw.get("local_txn_version"),
        "partition_id": raw.get("partition_id"),
    }


def _state_reason(state: str | None) -> tuple[str, str | None]:
    if state == COMMITTED:
        return "committed", None
    if state == ABORTED:
        return "aborted", None
    if state == COMMITTING:
        return "unknown", None
    if state == PREPARED:
        return "prepared", None
    return "pending", None


def derive_prefix_reports(
    partitions: Iterable[Partition],
    transactions: Iterable[Transaction],
    states: Mapping[str, str | None],
) -> tuple[PrefixReport, ...]:
    transaction_map = {transaction.transaction_id: transaction for transaction in transactions}
    by_partition: dict[str, dict[int, str]] = {}
    for transaction in transaction_map.values():
        partition_positions = by_partition.setdefault(transaction.partition_id, {})
        for message in transaction.messages:
            partition_positions[message.position] = transaction.transaction_id

    unknown_by_partition: dict[str, list[int]] = {}
    for transaction in transaction_map.values():
        if states.get(transaction.transaction_id) == COMMITTING:
            unknown_by_partition.setdefault(transaction.partition_id, []).extend(
                message.position for message in transaction.messages
            )

    reports: list[PrefixReport] = []
    for partition in sorted(partitions, key=lambda item: item.partition_id):
        positions = by_partition.get(partition.partition_id, {})
        expected = 0
        prefix = -1
        truncated_position: int | None = None
        truncated_reason: str | None = None
        truncated_transaction_id: str | None = None
        for position in sorted(positions):
            if position > partition.max_position:
                break
            if position > expected:
                truncated_position = expected
                truncated_reason = "missing"
                break
            transaction_id = positions[position]
            state = states.get(transaction_id)
            if state == COMMITTED:
                prefix = position
                expected = position + 1
                continue
            reason, _ = _state_reason(state)
            truncated_position = position
            truncated_reason = reason
            truncated_transaction_id = transaction_id
            break
        if truncated_position is None and expected <= partition.max_position:
            truncated_position = expected
            truncated_reason = "missing"
        reports.append(
            PrefixReport(
                partition_id=partition.partition_id,
                partition_version=partition.partition_version,
                max_position=partition.max_position,
                prefix=prefix,
                truncated_position=truncated_position,
                truncated_reason=truncated_reason,
                truncated_transaction_id=truncated_transaction_id,
                unknown_intervals=merge_intervals(
                    unknown_by_partition.get(partition.partition_id, [])
                ),
            )
        )
    return tuple(reports)


def unknown_transactions(
    transactions: Iterable[Transaction],
    states: Mapping[str, str | None],
) -> tuple[UnknownTransaction, ...]:
    result: list[UnknownTransaction] = []
    for transaction in transactions:
        if states.get(transaction.transaction_id) != COMMITTING:
            continue
        result.append(
            UnknownTransaction(
                transaction_id=transaction.transaction_id,
                partition_id=transaction.partition_id,
                intervals=merge_intervals(message.position for message in transaction.messages),
            )
        )
    return tuple(sorted(result, key=lambda item: (item.partition_id, item.transaction_id)))


def replay_events(
    partitions: Sequence[Partition],
    transactions: Sequence[Transaction],
    events: Sequence[Mapping[str, Any]],
    *,
    visibility: Sequence[Mapping[str, Any]] | None = None,
    expected_cursor: int | None = None,
) -> ReplayOutcome:
    """Replay events from sequence 1 and independently derive all conclusions.

    The input event order is irrelevant: rows are sorted by ``event_seq`` and
    duplicate or missing sequence numbers are rejected.
    """

    partition_map = {partition.partition_id: partition for partition in partitions}
    transaction_map = {transaction.transaction_id: transaction for transaction in transactions}
    transient_states: dict[str, str | None] = {
        transaction_id: None for transaction_id in transaction_map
    }
    final_event_seq: dict[str, int | None] = {
        transaction_id: None for transaction_id in transaction_map
    }
    expected_visibility: list[dict[str, Any]] = []

    normalised = [_normalise_event(event) for event in events]
    normalised.sort(key=lambda event: (-1 if event["event_seq"] is None else event["event_seq"]))
    seen_sequences: set[int] = set()
    expected_sequence = 1
    for event in normalised:
        event_seq = event["event_seq"]
        if not isinstance(event_seq, int) or isinstance(event_seq, bool):
            raise ReplayMismatch(None, event.get("transaction_id"), "event_seq", expected_sequence, event_seq)
        if event_seq in seen_sequences:
            raise ReplayMismatch(event_seq, event.get("transaction_id"), "event_seq", "unique", event_seq)
        seen_sequences.add(event_seq)
        if event_seq != expected_sequence:
            raise ReplayMismatch(
                event_seq,
                event.get("transaction_id"),
                "event_seq",
                expected_sequence,
                event_seq,
            )
        expected_sequence += 1

        transaction_id = event["transaction_id"]
        transaction = transaction_map.get(transaction_id) if isinstance(transaction_id, str) else None
        if transaction is None:
            raise ReplayMismatch(
                event_seq,
                transaction_id if isinstance(transaction_id, str) else None,
                "transaction_id",
                "known transaction",
                transaction_id,
            )
        expected_partition = transaction.partition_id
        if event["partition_id"] != expected_partition:
            raise ReplayMismatch(
                event_seq,
                transaction.transaction_id,
                "partition_id",
                expected_partition,
                event["partition_id"],
            )
        if event["local_txn_version"] != transaction.local_txn_version:
            raise ReplayMismatch(
                event_seq,
                transaction.transaction_id,
                "local_txn_version",
                transaction.local_txn_version,
                event["local_txn_version"],
            )
        state = event["state"]
        previous_state = transient_states[transaction.transaction_id]
        if state not in ALLOWED_TRANSITIONS:
            raise ReplayMismatch(event_seq, transaction.transaction_id, "state", "known state", state)
        if previous_state is None:
            if state != PREPARED:
                raise ReplayMismatch(
                    event_seq,
                    transaction.transaction_id,
                    "state",
                    PREPARED,
                    state,
                )
        elif state not in ALLOWED_TRANSITIONS[previous_state]:
            raise ReplayMismatch(
                event_seq,
                transaction.transaction_id,
                "state",
                sorted(ALLOWED_TRANSITIONS[previous_state]),
                state,
            )
        transient_states[transaction.transaction_id] = state
        final_event_seq[transaction.transaction_id] = event_seq

        if state == COMMITTED:
            for message in transaction.messages:
                expected_visibility.append(
                    {
                        "transaction_id": transaction.transaction_id,
                        "partition_id": transaction.partition_id,
                        "position": message.position,
                        "message_digest": message_digest(
                            transaction.partition_id, message.position, message.payload
                        ),
                        "effective_event_seq": event_seq,
                    }
                )

    cursor = expected_sequence - 1
    if expected_cursor is not None and cursor != expected_cursor:
        raise ReplayMismatch(cursor + 1, None, "event_cursor", expected_cursor, cursor)

    if visibility is not None:
        actual_rows = [
            {
                "transaction_id": row.get("transaction_id"),
                "partition_id": row.get("partition_id"),
                "position": row.get("position"),
                "message_digest": row.get("message_digest"),
                "effective_event_seq": row.get("effective_event_seq"),
            }
            for row in visibility
        ]
        actual_rows.sort(
            key=lambda row: (
                row["effective_event_seq"] if isinstance(row["effective_event_seq"], int) else -1,
                str(row["transaction_id"]),
                row["position"] if isinstance(row["position"], int) else -1,
            )
        )
        expected_sorted = sorted(
            expected_visibility,
            key=lambda row: (row["effective_event_seq"], row["transaction_id"], row["position"]),
        )
        for index in range(max(len(actual_rows), len(expected_sorted))):
            expected = expected_sorted[index] if index < len(expected_sorted) else None
            actual = actual_rows[index] if index < len(actual_rows) else None
            if expected == actual:
                continue
            event_seq = None
            transaction_id = None
            field = f"visibility[{index}]"
            if expected is not None:
                event_seq = expected["effective_event_seq"]
                transaction_id = expected["transaction_id"]
                for key in expected:
                    if actual is None or actual.get(key) != expected[key]:
                        field = f"visibility[{index}].{key}"
                        break
            elif actual is not None:
                event_seq = actual.get("effective_event_seq")
                transaction_id = actual.get("transaction_id")
                field = f"visibility[{index}].extra"
            raise ReplayMismatch(event_seq, transaction_id, field, expected, actual)

    prefixes = derive_prefix_reports(partitions, transactions, transient_states)
    unknown = unknown_transactions(transactions, transient_states)
    return ReplayOutcome(
        event_cursor=cursor,
        events_replayed=len(normalised),
        states=transient_states,
        final_event_seq=final_event_seq,
        visibility=tuple(expected_visibility),
        partitions=prefixes,
        unknown_transactions=unknown,
    )


def states_from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    states: dict[str, str] = {}
    for event in sorted(events, key=lambda row: int(row.get("event_seq", 0))):
        transaction_id = event.get("transaction_id")
        state = event.get("state")
        if isinstance(transaction_id, str) and isinstance(state, str):
            states[transaction_id] = state
    return states
