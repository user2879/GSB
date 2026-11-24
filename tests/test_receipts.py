from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quantcalib.errors import FailureList
from quantcalib.loading import load_receipts
from quantcalib.receipts import build_receipt_ledger, convert_receipt_file
from quantcalib.samplegen import generate_samples


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.samples = generate_samples(self.root / "samples")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _corpus(self):
        failures = FailureList()
        corpus = load_receipts(
            [self.samples["q1"], self.samples["q2"], self.samples["q3"]],
            {"embedding.proj", "encoder.block", "retrieval.head"},
            failures,
        )
        self.assertFalse(failures.items)
        return corpus

    def test_missing_fields_and_election(self) -> None:
        corpus = self._corpus()
        q1 = next(item for item in corpus.receipts if item.source_version == "q1")
        q2 = next(item for item in corpus.receipts if item.source_version == "q2")
        q3 = next(item for item in corpus.receipts if item.source_version == "q3")
        self.assertIn("parameter_version", q1.missing_fields)
        self.assertIn("group_scale_digest", q2.missing_fields)
        self.assertEqual(q3.missing_fields, ())
        ledger = build_receipt_ledger(corpus)
        embedding = next(item for item in ledger["layers"] if item["layer"] == "embedding.proj")
        self.assertTrue(embedding["authoritative_receipt_id"].startswith("q3.jsonl"))
        self.assertEqual(embedding["decision_basis"], "q3_supersede_then_elapsed_ms")
        retrieval = next(item for item in ledger["layers"] if item["layer"] == "retrieval.head")
        self.assertEqual(retrieval["decision_basis"], "parameter_version_desc_then_filename_asc")
        q2_entry = next(item for item in ledger["entries"] if item["receipt_id"].startswith("q2.jsonl"))
        self.assertIsNotNone(q2_entry["overridden_by"])

    def test_orphan_receipt(self) -> None:
        failures = FailureList()
        load_receipts(
            [self.samples["orphan_receipts"]],
            {"embedding.proj", "encoder.block", "retrieval.head"},
            failures,
        )
        self.assertTrue(failures.has_code("E_RECEIPT_ORPHAN"))

    def test_q3_q1_roundtrip_restores_bypass_fields(self) -> None:
        base = self.root / "roundtrip"
        failures = FailureList()
        downgrade = convert_receipt_file(
            input_path=self.samples["q3"],
            target_version="q1",
            output_path=base / "q1.jsonl",
            sidecar_path=base / "sidecar.json",
            failures=failures,
        )
        self.assertTrue(downgrade["q3_bypass_fields_preserved"])
        first_line = (base / "q1.jsonl").read_text(encoding="utf-8").splitlines()[0]
        self.assertNotIn("group_scale_digest", first_line)
        restore = convert_receipt_file(
            input_path=base / "q1.jsonl",
            target_version="q3",
            output_path=base / "q3.jsonl",
            sidecar_path=base / "sidecar.json",
            failures=failures,
        )
        self.assertTrue(restore["restored_from_sidecar"])
        restored = (base / "q3.jsonl").read_text(encoding="utf-8")
        self.assertIn("sha256:embedding-group-scales-v3", restored)
        self.assertIn('"supersedes":["q2.jsonl"]', restored)

    def test_q3_q2_roundtrip(self) -> None:
        base = self.root / "roundtrip-q2"
        failures = FailureList()
        convert_receipt_file(
            input_path=self.samples["q3"],
            target_version="q2",
            output_path=base / "q2.jsonl",
            sidecar_path=base / "sidecar.json",
            failures=failures,
        )
        convert_receipt_file(
            input_path=base / "q2.jsonl",
            target_version="q3",
            output_path=base / "q3.jsonl",
            sidecar_path=base / "sidecar.json",
            failures=failures,
        )
        self.assertIn("group_scale_digest", (base / "q3.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
