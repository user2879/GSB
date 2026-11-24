from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from colpage_recovery.codec import decode_page_bytes
from colpage_recovery.demo import determinism_probe
from colpage_recovery.engine import ReplayEngine
from colpage_recovery.loader import load_bundle


class DeterminismTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = load_bundle("samples")

    def test_digest_map_identical_across_batch_and_slots(self) -> None:
        reference = ReplayEngine(self.bundle).replay_all(batch_size=2)
        expected = reference.digest_map()
        for batch_size, slots, quota in ((1, 1, 1), (3, 4, 2), (4, 2, 1), (8, 8, 4)):
            budget = replace(
                self.bundle.budget,
                ledger_batch_entries=batch_size,
                global_slots=slots,
                per_tx_quota=quota,
                work_delay_ms=0,
            )
            bundle = replace(self.bundle, budget=budget)
            result = ReplayEngine(bundle).replay_all()
            self.assertEqual(expected, result.digest_map())
            self.assertEqual(reference.final_content, result.final_content)
            reference_bytes = {record.lsn: record.content_bytes for record in reference.records}
            actual_bytes = {record.lsn: record.content_bytes for record in result.records}
            self.assertEqual(reference_bytes, actual_bytes)
            for page_id, data in result.final_content.items():
                decoded = decode_page_bytes(data)
                self.assertEqual(result.final_versions[page_id], decoded.version)

    def test_determinism_probe_report(self) -> None:
        report = determinism_probe("samples")
        self.assertEqual("matched", report["status"])
        self.assertEqual(4, len(report["variants"]))


if __name__ == "__main__":
    unittest.main()
