from __future__ import annotations

from pathlib import Path

from txnmsg.store import Ledger
from txnmsg.validation import parse_partitions, parse_transactions


def partitions_payload():
    return {
        "partitions": [
            {"partition_id": "orders", "partition_version": 2, "max_position": 10},
            {"partition_id": "payments", "partition_version": 1, "max_position": 6},
        ]
    }


def transactions_payload():
    return {
        "transactions": [
            {
                "transaction_id": "t1",
                "txn_seq": 1,
                "partition_id": "orders",
                "local_txn_version": 1,
                "commit_time_ms": 1,
                "messages": [
                    {"position": 0, "payload": {"id": "a", "n": 1}},
                    {"position": 1, "payload": {"id": "b", "n": 2}},
                ],
            },
            {
                "transaction_id": "t2",
                "txn_seq": 2,
                "partition_id": "orders",
                "local_txn_version": 2,
                "commit_time_ms": 2,
                "messages": [
                    {"position": 2, "payload": {"id": "c", "n": 3}},
                    {"position": 3, "payload": {"id": "d", "n": 4}},
                ],
            },
            {
                "transaction_id": "t3",
                "txn_seq": 3,
                "partition_id": "payments",
                "local_txn_version": 1,
                "commit_time_ms": 3,
                "messages": [
                    {"position": 0, "payload": {"id": "p", "n": 5}},
                ],
            },
        ]
    }


def load_domain():
    partitions = parse_partitions(partitions_payload())
    transactions = parse_transactions(transactions_payload(), partitions)
    return partitions, transactions


def make_ledger(path: Path) -> Ledger:
    partitions, transactions = load_domain()
    ledger = Ledger(path)
    ledger.initialise_partitions(partitions)
    ledger.ingest(transactions)
    return ledger


def extra_transaction(transaction_id: str = "t4", txn_seq: int = 4):
    partitions, _ = load_domain()
    return parse_transactions(
        {
            "transactions": [
                {
                    "transaction_id": transaction_id,
                    "txn_seq": txn_seq,
                    "partition_id": "orders",
                    "local_txn_version": 9,
                    "commit_time_ms": 100,
                    "messages": [
                        {"position": 4, "payload": {"id": "extra", "n": 9}}
                    ],
                }
            ]
        },
        partitions,
    )
