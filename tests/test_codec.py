from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from mwlab.canonical import canonical_lines
from mwlab.config import load_declarations
from mwlab.errors import LabError
from mwlab.records import (
    duplicate_idempotency_details,
    load_normalized_records,
    model_digest,
)
from mwlab.semantics import analyze

from .helpers import SAMPLES, ROOT


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.declarations = load_declarations(SAMPLES / "session.json")
        cls.records = load_normalized_records(SAMPLES / "records.jsonl", cls.declarations)

    def test_mixed_formats_and_provenance(self) -> None:
        self.assertEqual(len(self.records), 34)
        counts = {version: 0 for version in ("v1", "v2", "v3")}
        for record in self.records:
            counts[record.source_format] += 1
            self.assertGreater(record.line_no, 0)
            self.assertIn(record.source_format, ("v1", "v2", "v3"))
            self.assertEqual(record.source_group, record.consumer_group)
        self.assertEqual(counts, {"v1": 9, "v2": 9, "v3": 16})

    def test_unknown_version_is_e_format(self) -> None:
        with self.assertRaises(LabError) as raised:
            load_normalized_records(SAMPLES / "records_bad_version.jsonl", self.declarations)
        error = raised.exception
        self.assertEqual(error.code, "E_FORMAT")
        self.assertEqual(error.details["line"], 1)
        self.assertEqual(error.details["field"], "format_version")

    def test_reversed_range_is_e_format(self) -> None:
        with self.assertRaises(LabError) as raised:
            load_normalized_records(SAMPLES / "records_bad_range.jsonl", self.declarations)
        error = raised.exception
        self.assertEqual(error.code, "E_FORMAT")
        self.assertEqual(error.details["line"], 1)
        self.assertEqual(error.details["field"], "record.delivery_sequence_range")

    def test_negative_retry_and_missing_field_are_located(self) -> None:
        line = {
            "format_version": "v2",
            "consumer_group": "billing",
            "record": {
                "message_id": "orders:1",
                "topic": "orders",
                "acknowledged": True,
                "delivery_time_ms": 1700000200001,
                "retry_count": -1,
                "consumer_group": "billing",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps(line) + "\n", encoding="utf-8")
            with self.assertRaises(LabError) as raised:
                load_normalized_records(path, self.declarations)
            self.assertEqual(raised.exception.details["field"], "record.retry_count")
            self.assertEqual(raised.exception.details["line"], 1)
            del line["record"]["acknowledged"]
            path.write_text(json.dumps(line) + "\n", encoding="utf-8")
            with self.assertRaises(LabError) as raised_missing:
                load_normalized_records(path, self.declarations)
            self.assertEqual(raised_missing.exception.details["field"], "record.acknowledged")
            self.assertEqual(raised_missing.exception.details["line"], 1)

    def test_group_declared_version_mismatch_is_e_format(self) -> None:
        line = {
            "format_version": "v3",
            "consumer_group": "billing",
            "record": {
                "message_id": "orders:1",
                "topic": "orders",
                "acknowledged": True,
                "delivery_time_ms": 1700000200001,
                "retry_count": 0,
                "consumer_group": "billing",
                "idempotency_key": "x",
                "delivery_sequence_range": {"start": 0, "end": 0},
                "semantic_verdict": "exactly_once",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.jsonl"
            path.write_text(json.dumps(line) + "\n", encoding="utf-8")
            with self.assertRaises(LabError) as raised:
                load_normalized_records(path, self.declarations)
            self.assertEqual(raised.exception.details["field"], "format_version")
            self.assertEqual(raised.exception.details["expected"], "v2")

    def test_cross_group_idempotency_duplicate(self) -> None:
        details = duplicate_idempotency_details(self.records)
        self.assertIsNotNone(details)
        self.assertEqual(details["idempotency_key"], "idem-payments-shared")
        self.assertEqual(details["consumer_groups"], ["analytics", "warehouse"])

    def test_line_order_does_not_change_model_or_conclusions(self) -> None:
        baseline = analyze(self.records, self.declarations)
        source_lines = (SAMPLES / "records.jsonl").read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            for seed in (3, 17, 91):
                shuffled = source_lines[:]
                random.Random(seed).shuffle(shuffled)
                path = Path(directory) / f"shuffled-{seed}.jsonl"
                path.write_text("\n".join(shuffled) + "\n", encoding="utf-8")
                records = load_normalized_records(path, self.declarations)
                analysis = analyze(records, self.declarations)
                self.assertEqual(model_digest(records), model_digest(self.records))
                self.assertEqual(
                    canonical_lines(analysis["conclusions"]),
                    canonical_lines(baseline["conclusions"]),
                )
                self.assertEqual(
                    canonical_lines(analysis["conflicts"]),
                    canonical_lines(baseline["conflicts"]),
                )


if __name__ == "__main__":
    unittest.main()

