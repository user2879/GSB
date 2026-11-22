"""Deterministic, dependency-free demonstration data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def partitions_payload() -> dict[str, Any]:
    return {
        "format_version": 1,
        "partitions": [
            {"partition_id": "orders", "partition_version": 3, "max_position": 20},
            {"partition_id": "payments", "partition_version": 1, "max_position": 10},
        ],
    }


def transactions_payload() -> dict[str, Any]:
    return {
        "format_version": 1,
        "transactions": [
            {
                "transaction_id": "tx-order-001",
                "txn_seq": 1,
                "partition_id": "orders",
                "local_txn_version": 1,
                "commit_time_ms": 1700000000001,
                "messages": [
                    {"position": 0, "payload": {"order_id": "A-100", "amount": 10}},
                    {"position": 1, "payload": {"order_id": "A-101", "amount": 11}},
                ],
            },
            {
                "transaction_id": "tx-order-002",
                "txn_seq": 2,
                "partition_id": "orders",
                "local_txn_version": 2,
                "commit_time_ms": 1700000000002,
                "messages": [
                    {"position": 2, "payload": {"order_id": "A-102", "amount": 12}},
                    {"position": 3, "payload": {"order_id": "A-103", "amount": 13}},
                ],
            },
            {
                "transaction_id": "tx-pay-001",
                "txn_seq": 3,
                "partition_id": "payments",
                "local_txn_version": 1,
                "commit_time_ms": 1700000000003,
                "messages": [
                    {"position": 0, "payload": {"payment_id": "P-100", "amount": 10}}
                ],
            },
            {
                "transaction_id": "tx-order-003",
                "txn_seq": 4,
                "partition_id": "orders",
                "local_txn_version": 3,
                "commit_time_ms": 1700000000004,
                "messages": [
                    {"position": 4, "payload": {"order_id": "A-104", "amount": 14}},
                    {"position": 5, "payload": {"order_id": "A-105", "amount": 15}},
                ],
            },
            {
                "transaction_id": "tx-order-004",
                "txn_seq": 5,
                "partition_id": "orders",
                "local_txn_version": 1,
                "commit_time_ms": 1700000000005,
                "messages": [
                    {"position": 6, "payload": {"order_id": "A-106", "amount": 16}},
                    {"position": 7, "payload": {"order_id": "A-107", "amount": 17}},
                ],
            },
            {
                "transaction_id": "tx-pay-002",
                "txn_seq": 6,
                "partition_id": "payments",
                "local_txn_version": 2,
                "commit_time_ms": 1700000000006,
                "messages": [
                    {"position": 1, "payload": {"payment_id": "P-101", "amount": 11}},
                    {"position": 2, "payload": {"payment_id": "P-102", "amount": 12}},
                ],
            },
        ],
    }


def increment_transactions_payload() -> dict[str, Any]:
    return {
        "format_version": 1,
        "transactions": [
            {
                "transaction_id": "tx-pay-003",
                "txn_seq": 7,
                "partition_id": "payments",
                "local_txn_version": 3,
                "commit_time_ms": 1700000000007,
                "messages": [
                    {"position": 3, "payload": {"payment_id": "P-103", "amount": 13}},
                    {"position": 4, "payload": {"payment_id": "P-104", "amount": 14}},
                ],
            }
        ],
    }


def duplicate_position_payload() -> dict[str, Any]:
    return {
        "format_version": 1,
        "transactions": [
            {
                "transaction_id": "tx-dup-001",
                "txn_seq": 101,
                "partition_id": "orders",
                "local_txn_version": 1,
                "commit_time_ms": 1700000000101,
                "messages": [{"position": 8, "payload": {"order_id": "D-1"}}],
            },
            {
                "transaction_id": "tx-dup-002",
                "txn_seq": 102,
                "partition_id": "orders",
                "local_txn_version": 1,
                "commit_time_ms": 1700000000102,
                "messages": [{"position": 8, "payload": {"order_id": "D-2"}}],
            },
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_samples(workdir: str | os.PathLike[str]) -> dict[str, str]:
    directory = Path(workdir) / "samples"
    partitions_path = directory / "partitions.json"
    transactions_path = directory / "transactions.json"
    increment_path = directory / "transactions-increment.json"
    _write_json(partitions_path, partitions_payload())
    _write_json(transactions_path, transactions_payload())
    _write_json(increment_path, increment_transactions_payload())
    return {
        "partitions": str(partitions_path),
        "transactions": str(transactions_path),
        "increment": str(increment_path),
    }
