from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from txnmsg.errors import OrderError, SpecError
from txnmsg.observability import StructuredLogger
from txnmsg.validation import (
    load_partitions,
    load_transactions,
    parse_partitions,
    parse_transactions,
)

from tests.helpers import load_domain, make_ledger


class ValidationTests(unittest.TestCase):
    def test_valid_domain(self):
        partitions, transactions = load_domain()
        self.assertEqual([item.partition_id for item in partitions], ["orders", "payments"])
        self.assertEqual([item.transaction_id for item in transactions], ["t1", "t2", "t3"])
        self.assertEqual(transactions[0].messages[1].position, 1)

    def test_partition_types_reject_bool_and_negative(self):
        with self.assertRaises(SpecError) as raised:
            parse_partitions(
                {
                    "partitions": [
                        {"partition_id": "p", "partition_version": True, "max_position": 1}
                    ]
                }
            )
        self.assertEqual(raised.exception.code, "E_SPEC")
        self.assertEqual(raised.exception.field, "partitions[0].partition_version")

        with self.assertRaises(SpecError) as raised:
            parse_partitions(
                {
                    "partitions": [
                        {"partition_id": "p", "partition_version": 0, "max_position": -1}
                    ]
                }
            )
        self.assertEqual(raised.exception.field, "partitions[0].max_position")

    def test_first_invalid_transaction_and_field(self):
        partitions, _ = load_domain()
        payload = {
            "transactions": [
                {
                    "transaction_id": "ok-1",
                    "txn_seq": 1,
                    "partition_id": "orders",
                    "local_txn_version": 1,
                    "commit_time_ms": 1,
                    "messages": [{"position": 0, "payload": {"ok": True}}],
                },
                {
                    "transaction_id": "bad-local-version",
                    "txn_seq": 2,
                    "partition_id": "orders",
                    "local_txn_version": 0,
                    "commit_time_ms": 2,
                    "messages": [{"position": 1, "payload": {"ok": False}}],
                },
                {
                    "transaction_id": "later-bad",
                    "txn_seq": 3,
                    "partition_id": "orders",
                    "local_txn_version": 0,
                    "commit_time_ms": 3,
                    "messages": [{"position": 2, "payload": {"ok": False}}],
                },
            ]
        }
        with self.assertRaises(SpecError) as raised:
            parse_transactions(payload, partitions)
        self.assertEqual(raised.exception.transaction_id, "bad-local-version")
        self.assertEqual(raised.exception.field, "transactions[1].local_txn_version")

    def test_position_strictness_and_upper_bound_are_spec_errors(self):
        partitions, _ = load_domain()
        with self.assertRaises(SpecError) as raised:
            parse_transactions(
                {
                    "transactions": [
                        {
                            "transaction_id": "bad-pos",
                            "txn_seq": 1,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 1,
                            "messages": [
                                {"position": 1, "payload": 1},
                                {"position": 1, "payload": 2},
                            ],
                        }
                    ]
                },
                partitions,
            )
        self.assertEqual(raised.exception.field, "transactions[0].messages[1].position")

        with self.assertRaises(SpecError) as raised:
            parse_transactions(
                {
                    "transactions": [
                        {
                            "transaction_id": "too-high",
                            "txn_seq": 1,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 1,
                            "messages": [{"position": 11, "payload": 1}],
                        }
                    ]
                },
                partitions,
            )
        self.assertEqual(raised.exception.field, "transactions[0].messages[0].position")

    def test_duplicate_partition_position_is_order_error(self):
        partitions, _ = load_domain()
        with self.assertRaises(OrderError) as raised:
            parse_transactions(
                {
                    "transactions": [
                        {
                            "transaction_id": "dup-a",
                            "txn_seq": 1,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 1,
                            "messages": [{"position": 0, "payload": 1}],
                        },
                        {
                            "transaction_id": "dup-b",
                            "txn_seq": 2,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 2,
                            "messages": [{"position": 0, "payload": 2}],
                        },
                    ]
                },
                partitions,
            )
        self.assertEqual(raised.exception.code, "E_ORDER")
        self.assertEqual(raised.exception.transaction_id, "dup-b")
        self.assertEqual(raised.exception.partition_id, "orders")
        self.assertEqual(raised.exception.details["first_transaction_id"], "dup-a")

    def test_txn_seq_must_increase(self):
        partitions, _ = load_domain()
        with self.assertRaises(SpecError) as raised:
            parse_transactions(
                {
                    "transactions": [
                        {
                            "transaction_id": "seq-2",
                            "txn_seq": 2,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 1,
                            "messages": [{"position": 0, "payload": 1}],
                        },
                        {
                            "transaction_id": "seq-1",
                            "txn_seq": 1,
                            "partition_id": "orders",
                            "local_txn_version": 1,
                            "commit_time_ms": 2,
                            "messages": [{"position": 1, "payload": 2}],
                        },
                    ]
                },
                partitions,
            )
        self.assertEqual(raised.exception.transaction_id, "seq-1")
        self.assertEqual(raised.exception.field, "transactions[1].txn_seq")

    def test_malformed_json_logs_structured_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partitions_path = root / "partitions.json"
            partitions_path.write_text(json.dumps({"partitions": [{"partition_id": "p", "partition_version": 1, "max_position": 1}]}), encoding="utf-8")
            bad_path = root / "bad.json"
            bad_path.write_text('{"transactions": [', encoding="utf-8")
            logger = StructuredLogger(root / "logs" / "business.jsonl")
            with self.assertRaises(SpecError) as raised:
                partitions = load_partitions(partitions_path, logger)
                load_transactions(bad_path, partitions, logger)
            self.assertEqual(raised.exception.code, "E_SPEC")
            self.assertEqual(raised.exception.field, "file")
            record = json.loads((root / "logs" / "business.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["code"], "E_SPEC")
            self.assertIn("timestamp", record)
            self.assertIn("event_seq", record)
            self.assertIn("partition_id", record)

    def test_cross_batch_position_conflict_is_order_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.sqlite"
            ledger = make_ledger(path)
            try:
                partitions, _ = load_domain()
                duplicate = parse_transactions(
                    {
                        "transactions": [
                            {
                                "transaction_id": "late",
                                "txn_seq": 4,
                                "partition_id": "orders",
                                "local_txn_version": 1,
                                "commit_time_ms": 4,
                                "messages": [{"position": 3, "payload": "late"}],
                            }
                        ]
                    },
                    partitions,
                )
                with self.assertRaises(OrderError) as raised:
                    ledger.ingest(duplicate)
                self.assertEqual(raised.exception.code, "E_ORDER")
                self.assertEqual(raised.exception.transaction_id, "late")
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
