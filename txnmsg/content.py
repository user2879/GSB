"""Deterministic ledger-content representation and digests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .canonical import digest_value
from .models import FORMAT_VERSION, Partition, Transaction


def normalise_event(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_seq": raw.get("event_seq"),
        "local_txn_version": raw.get("local_txn_version"),
        "partition_id": raw.get("partition_id"),
        "state": raw.get("state"),
        "transaction_id": raw.get("transaction_id"),
    }


def normalise_visibility(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "effective_event_seq": raw.get("effective_event_seq"),
        "message_digest": raw.get("message_digest"),
        "partition_id": raw.get("partition_id"),
        "position": raw.get("position"),
        "transaction_id": raw.get("transaction_id"),
    }


def ledger_content(
    partitions: Iterable[Partition],
    transactions: Iterable[Transaction],
    events: Iterable[Mapping[str, Any]],
    visibility: Iterable[Mapping[str, Any]],
    *,
    partition_set_version: int,
    transaction_set_version: int,
    ledger_format_version: int = FORMAT_VERSION,
) -> dict[str, Any]:
    partition_rows = sorted(
        (partition.as_dict() for partition in partitions),
        key=lambda item: item["partition_id"],
    )
    transaction_rows = sorted(
        (
            transaction.as_spec_dict(include_digest=True)
            for transaction in transactions
        ),
        key=lambda item: (item["txn_seq"], item["transaction_id"]),
    )
    event_rows = sorted(
        (normalise_event(event) for event in events),
        key=lambda item: item["event_seq"] if isinstance(item["event_seq"], int) else -1,
    )
    visibility_rows = sorted(
        (normalise_visibility(row) for row in visibility),
        key=lambda item: (
            item["effective_event_seq"] if isinstance(item["effective_event_seq"], int) else -1,
            str(item["partition_id"]),
            item["position"] if isinstance(item["position"], int) else -1,
            str(item["transaction_id"]),
        ),
    )
    return {
        "ledger_format_version": ledger_format_version,
        "partition_set_version": partition_set_version,
        "transaction_set_version": transaction_set_version,
        "partitions": partition_rows,
        "transactions": transaction_rows,
        "events": event_rows,
        "visibility": visibility_rows,
    }


def ledger_digest(
    partitions: Iterable[Partition],
    transactions: Iterable[Transaction],
    events: Iterable[Mapping[str, Any]],
    visibility: Iterable[Mapping[str, Any]],
    *,
    partition_set_version: int,
    transaction_set_version: int,
    ledger_format_version: int = FORMAT_VERSION,
) -> str:
    return digest_value(
        ledger_content(
            partitions,
            transactions,
            events,
            visibility,
            partition_set_version=partition_set_version,
            transaction_set_version=transaction_set_version,
            ledger_format_version=ledger_format_version,
        )
    )


def digest_ledger_content(content: Mapping[str, Any]) -> str:
    return digest_value(content)


def snapshot_content(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "content_digest"}


def snapshot_digest(payload: Mapping[str, Any]) -> str:
    return digest_value(snapshot_content(payload))


def sequences_contiguous(sequence_numbers: Sequence[int]) -> bool:
    return list(sequence_numbers) == list(range(1, len(sequence_numbers) + 1))
