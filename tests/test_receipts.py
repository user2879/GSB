from __future__ import annotations

import copy
import json
import unittest

from speaker_diag.errors import DiagnosticError
from speaker_diag.receipts import (
    context_for,
    normalize_receipt,
    normalize_receipt_file,
)
from speaker_diag.samples import sample_paths
from speaker_diag.spec import load_workload


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = sample_paths("samples")
        cls.spec = load_workload(cls.paths["v1"])
        cls.context = context_for(cls.spec, "B001")
        cls.raw = json.loads(cls.paths["r3"].read_text(encoding="utf-8"))

    def test_three_formats_normalize_to_same_attribution_digest(self) -> None:
        normalized = [
            normalize_receipt_file(self.paths[label], context=self.context) for label in ("r1", "r2", "r3")
        ]
        self.assertEqual(len({item.attribution_digest for item in normalized}), 1)
        self.assertEqual(normalized[0].filled_fields, tuple(sorted(normalized[0].filled_fields)))
        self.assertTrue(normalized[0].data["fill_basis"])
        self.assertFalse(normalized[2].filled_fields)

    def test_same_receipt_is_byte_identical_across_parses(self) -> None:
        first = normalize_receipt_file(self.paths["r2"], context=self.context)
        second = normalize_receipt_file(self.paths["r2"], context=self.context)
        self.assertEqual(first.canonical, second.canonical)
        self.assertEqual(first.sha256, second.sha256)

    def test_negative_elapsed_is_e_field(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt["elapsed_ms"] = -1
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "elapsed_ms")

    def test_duplicate_rank_is_e_field(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt["windows"][1]["rank"] = receipt["windows"][0]["rank"]
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "windows[].rank")

    def test_rank_out_of_range_is_e_field(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt["windows"][0]["rank"] = len(receipt["windows"]) + 1
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "windows[0].rank")

    def test_duplicate_reference_must_exist(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt["windows"][0]["duplicate_of"] = "W9999"
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_REF")
        self.assertEqual(caught.exception.field, "windows[0].duplicate_of")

    def test_context_mismatch_is_e_field(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt["worker_id"] = "W99"
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "worker_id")

    def test_r1_without_a_format_marker_is_inferred(self) -> None:
        receipt = copy.deepcopy(self.raw)
        receipt = {
            "batch_id": receipt["batch_id"],
            "windows": [
                {"window_id": window["window_id"], "attribution": window["attribution"]}
                for window in receipt["windows"]
            ],
        }
        normalized = normalize_receipt(receipt, context=self.context)
        self.assertEqual(normalized.source_format, "r1")
        self.assertIn("worker_id", normalized.filled_fields)

    def test_r2_missing_required_field_is_e_field(self) -> None:
        receipt = copy.deepcopy(self.raw)
        del receipt["param_version"]
        receipt["receipt_format"] = "r2"
        with self.assertRaises(DiagnosticError) as caught:
            normalize_receipt(receipt, context=self.context)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "param_version")


if __name__ == "__main__":
    unittest.main()
