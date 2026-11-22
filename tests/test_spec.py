from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from txdiag.errors import SpecError
from txdiag.spec import load_spec

from .helpers import sample


class SpecValidationTests(unittest.TestCase):
    def test_samples_load_and_normalize(self) -> None:
        spec = load_spec(sample("all_isolation"))
        self.assertEqual(len(spec.transactions), 5)
        self.assertEqual(len(spec.row_keys), 4)
        self.assertEqual(len(spec.dependency_edges), 3)
        self.assertEqual(spec.transactions[0].txid, "RC1")
        self.assertEqual(spec.transactions[0].operations[0].seq, 1)
        self.assertEqual(len(spec.spec_sha256), 64)

    def _load_inline(self, doc: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            load_spec(path)

    def test_duplicate_txid_is_first_field_error(self) -> None:
        doc = {
            "schema_version": 1,
            "row_keys": ["A"],
            "transactions": [
                {"id": "T1", "start_seq": 1, "isolation": "read_committed",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
                {"id": "T1", "start_seq": 2, "isolation": "read_committed",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
            ],
        }
        with self.assertRaises(SpecError) as ctx:
            self._load_inline(doc)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.txid, "T1")
        self.assertEqual(ctx.exception.field, "id")

    def test_non_contiguous_operation_sequence_rejected(self) -> None:
        doc = {
            "schema_version": 1,
            "row_keys": ["A"],
            "transactions": [{
                "id": "T1", "start_seq": 1, "isolation": "read_committed",
                "operations": [
                    {"seq": 1, "kind": "read", "row": "A"},
                    {"seq": 3, "kind": "read", "row": "A"},
                ],
            }],
        }
        with self.assertRaises(SpecError) as ctx:
            self._load_inline(doc)
        self.assertEqual(ctx.exception.txid, "T1")
        self.assertEqual(ctx.exception.field, "operations[2].seq")

    def test_unknown_row_and_duplicate_start(self) -> None:
        unknown_row = {
            "schema_version": 1,
            "row_keys": ["A"],
            "transactions": [{
                "id": "T1", "start_seq": 1, "isolation": "read_committed",
                "operations": [{"seq": 1, "kind": "read", "row": "X"}],
            }],
        }
        with self.assertRaises(SpecError) as ctx:
            self._load_inline(unknown_row)
        self.assertEqual((ctx.exception.txid, ctx.exception.field),
                         ("T1", "operations.row"))

        duplicate_start = {
            "schema_version": 1,
            "row_keys": ["A"],
            "transactions": [
                {"id": "T1", "start_seq": 1, "isolation": "read_committed",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
                {"id": "T2", "start_seq": 1, "isolation": "read_committed",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
            ],
        }
        with self.assertRaises(SpecError) as ctx:
            self._load_inline(duplicate_start)
        self.assertEqual((ctx.exception.txid, ctx.exception.field), ("T2", "start_seq"))

    def test_chinese_isolations_accepted(self) -> None:
        doc = {
            "schema_version": 1,
            "row_keys": ["A"],
            "transactions": [
                {"id": "T1", "start_seq": 1, "isolation": "读未提交",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
                {"id": "T2", "start_seq": 2, "isolation": "串行化",
                 "operations": [{"seq": 1, "kind": "read", "row": "A"}]},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            spec = load_spec(path)
        self.assertEqual([tx.isolation for tx in spec.transactions],
                         ["read_uncommitted", "serializable"])


if __name__ == "__main__":
    unittest.main()

