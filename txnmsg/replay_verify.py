"""Post-restore replay comparison with first-difference reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import BusinessError, ReplayError
from .models import ReplayReport
from .observability import StructuredLogger
from .replay import ReplayMismatch, replay_events
from .snapshots import ParsedSnapshot, load_snapshot
from .store import Ledger


def _raise_replay(
    logger: StructuredLogger,
    message: str,
    *,
    event_seq: int | None,
    transaction_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    partition_id: str | None = None,
) -> None:
    error = ReplayError(
        message,
        transaction_id=transaction_id,
        partition_id=partition_id,
        field=field,
        event_seq=event_seq,
        record_seq=event_seq,
        details={"expected": expected, "actual": actual},
    )
    logger.write_error(error)
    raise error


def replay_verify(
    ledger: Ledger,
    snapshot_path: str | Path | None = None,
    *,
    logger: StructuredLogger | None = None,
) -> ReplayReport:
    """Replay the durable event stream and compare all stored conclusions.

    When a snapshot baseline is supplied, its already validated conclusion is
    compared item by item.  Transactions whose latest state is ``committing``
    are listed as unknown and are excluded from final-state mismatch failures.
    """

    active_logger = logger or StructuredLogger(ledger.path.parent / "logs" / "business.jsonl")
    partitions = ledger.partitions()
    transactions = ledger.transactions()
    events = ledger.events()
    visibility = ledger.visibility()
    cursor = ledger._event_cursor()
    try:
        outcome = replay_events(
            partitions,
            transactions,
            events,
            visibility=visibility,
            expected_cursor=cursor,
        )
    except ReplayMismatch as mismatch:
        _raise_replay(
            active_logger,
            "回放事件流不一致",
            event_seq=mismatch.event_seq,
            transaction_id=mismatch.transaction_id,
            field=mismatch.field,
            expected=mismatch.expected,
            actual=mismatch.actual,
        )

    unknown_ids = {item.transaction_id for item in outcome.unknown_transactions}

    # Compare the replay-derived final state with the materialised transaction
    # rows.  Unknown transactions are reported separately.
    state_rows = ledger.connection.execute(
        "SELECT transaction_id, current_state, last_event_seq FROM transactions ORDER BY transaction_id"
    ).fetchall()
    for row in state_rows:
        transaction_id = row["transaction_id"]
        if transaction_id in unknown_ids:
            continue
        expected_state = outcome.states.get(transaction_id)
        if row["current_state"] != expected_state:
            _raise_replay(
                active_logger,
                "事务最终状态与回放结论不一致",
                event_seq=outcome.final_event_seq.get(transaction_id),
                transaction_id=transaction_id,
                field=f"transactions[{transaction_id}].current_state",
                expected=expected_state,
                actual=row["current_state"],
            )
        expected_seq = outcome.final_event_seq.get(transaction_id)
        if row["last_event_seq"] != expected_seq:
            _raise_replay(
                active_logger,
                "事务最后事件序号与回放结论不一致",
                event_seq=expected_seq,
                transaction_id=transaction_id,
                field=f"transactions[{transaction_id}].last_event_seq",
                expected=expected_seq,
                actual=row["last_event_seq"],
            )

    baseline: ParsedSnapshot | None = None
    if snapshot_path is not None:
        try:
            baseline = load_snapshot(snapshot_path)
        except BusinessError as error:
            active_logger.write_error(error)
            raise

        if baseline.event_cursor != outcome.event_cursor:
            _raise_replay(
                active_logger,
                "事件游标与快照基线不一致",
                event_seq=min(baseline.event_cursor, outcome.event_cursor) + 1,
                transaction_id=None,
                field="event_cursor",
                expected=baseline.event_cursor,
                actual=outcome.event_cursor,
            )

        baseline_by_seq = {event["event_seq"]: event for event in baseline.events}
        current_by_seq = {event["event_seq"]: event for event in events}
        for sequence in range(1, max(len(baseline.events), len(events)) + 1):
            expected = baseline_by_seq.get(sequence)
            actual = current_by_seq.get(sequence)
            if expected == actual:
                continue
            fields = ("event_seq", "transaction_id", "state", "local_txn_version", "partition_id")
            transaction_id = None
            partition_id = None
            field = "events"
            if expected is not None:
                transaction_id = expected.get("transaction_id")
                partition_id = expected.get("partition_id")
                for key in fields:
                    if actual is None or actual.get(key) != expected.get(key):
                        field = f"events[{sequence}].{key}"
                        break
            elif actual is not None:
                transaction_id = actual.get("transaction_id")
                partition_id = actual.get("partition_id")
                field = f"events[{sequence}].extra"
            _raise_replay(
                active_logger,
                "事件记录与快照基线不一致",
                event_seq=sequence,
                transaction_id=transaction_id,
                partition_id=partition_id,
                field=field,
                expected=expected,
                actual=actual,
            )

        baseline_states: dict[str, str | None] = {
            transaction.transaction_id: None for transaction in baseline.transactions
        }
        baseline_final_seq: dict[str, int | None] = {
            transaction.transaction_id: None for transaction in baseline.transactions
        }
        for event in baseline.events:
            baseline_states[event["transaction_id"]] = event["state"]
            baseline_final_seq[event["transaction_id"]] = event["event_seq"]
        baseline_unknown_ids = {item.transaction_id for item in baseline.unknown_transactions}
        ignored_unknown = unknown_ids | baseline_unknown_ids
        current_transactions = {
            transaction.transaction_id: transaction for transaction in transactions
        }
        baseline_transactions = {
            transaction.transaction_id: transaction for transaction in baseline.transactions
        }
        all_transaction_ids = set(current_transactions) | set(baseline_transactions)
        transaction_order = sorted(
            all_transaction_ids,
            key=lambda transaction_id: (
                (
                    current_transactions.get(transaction_id)
                    or baseline_transactions[transaction_id]
                ).txn_seq,
                transaction_id,
            ),
        )
        for transaction_id in transaction_order:
            if transaction_id in ignored_unknown:
                continue
            if transaction_id not in current_transactions:
                _raise_replay(
                    active_logger,
                    "快照基线事务在当前账本中缺失",
                    event_seq=baseline_final_seq.get(transaction_id),
                    transaction_id=transaction_id,
                    field=f"transaction_states[{transaction_id}].membership",
                    expected="present",
                    actual="missing",
                )
            if transaction_id not in baseline_transactions:
                _raise_replay(
                    active_logger,
                    "当前账本存在快照基线之外的事务",
                    event_seq=outcome.final_event_seq.get(transaction_id),
                    transaction_id=transaction_id,
                    field=f"transaction_states[{transaction_id}].membership",
                    expected="missing",
                    actual="present",
                )
            expected_state = baseline_states.get(transaction_id)
            actual_state = outcome.states.get(transaction_id)
            if expected_state != actual_state:
                _raise_replay(
                    active_logger,
                    "事务最终状态与快照基线不一致",
                    event_seq=outcome.final_event_seq.get(transaction_id),
                    transaction_id=transaction_id,
                    field=f"transaction_states[{transaction_id}].state",
                    expected=expected_state,
                    actual=actual_state,
                )

        baseline_prefixes = {report.partition_id: report for report in baseline.prefixes}
        actual_prefixes = {report.partition_id: report for report in outcome.partitions}
        for partition_id in sorted(set(baseline_prefixes) | set(actual_prefixes)):
            expected = baseline_prefixes.get(partition_id)
            actual = actual_prefixes.get(partition_id)
            if expected == actual:
                continue
            if expected is None:
                field = f"partitions[{partition_id}].extra"
                expected_value = None
                actual_value = actual.as_dict() if actual is not None else None
            elif actual is None:
                field = f"partitions[{partition_id}]"
                expected_value = expected.as_dict()
                actual_value = None
            else:
                if expected.prefix != actual.prefix:
                    field = f"partitions[{partition_id}].prefix"
                    expected_value = expected.prefix
                    actual_value = actual.prefix
                elif expected.truncated_position != actual.truncated_position:
                    field = f"partitions[{partition_id}].truncated_position"
                    expected_value = expected.truncated_position
                    actual_value = actual.truncated_position
                elif expected.truncated_reason != actual.truncated_reason:
                    field = f"partitions[{partition_id}].truncated_reason"
                    expected_value = expected.truncated_reason
                    actual_value = actual.truncated_reason
                elif expected.truncated_transaction_id != actual.truncated_transaction_id:
                    field = f"partitions[{partition_id}].truncated_transaction_id"
                    expected_value = expected.truncated_transaction_id
                    actual_value = actual.truncated_transaction_id
                else:
                    field = f"partitions[{partition_id}].unknown_intervals"
                    expected_value = [
                        [start, end] for start, end in expected.unknown_intervals
                    ]
                    actual_value = [
                        [start, end] for start, end in actual.unknown_intervals
                    ]
            _raise_replay(
                active_logger,
                "分区可消费前缀与快照基线不一致",
                event_seq=baseline.event_cursor,
                transaction_id=None,
                partition_id=partition_id,
                field=field,
                expected=expected_value,
                actual=actual_value,
            )

    report = ReplayReport(
        status="PASS",
        event_cursor=outcome.event_cursor,
        events_replayed=outcome.events_replayed,
        unknown_transactions=outcome.unknown_transactions,
        partitions=outcome.partitions,
        transaction_states=outcome.transaction_state_rows(),
        baseline=None if snapshot_path is None else str(Path(snapshot_path)),
    )
    return report
