from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stemscope.report import run_acceptance


class AcceptanceTests(unittest.TestCase):
    def test_full_acceptance_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = run_acceptance("sample_data", root / "artifacts")
            self.assertTrue(report["acceptance"]["passed"], report["acceptance"])
            self.assertTrue(report["full_parity"]["final_interactive_vs_full"]["canonical_equal"])
            self.assertTrue(report["full_parity"]["interactive_vs_verification"]["canonical_equal"])
            self.assertTrue(report["concurrency"]["serial_vs_concurrent"]["canonical_equal"])
            self.assertEqual(report["snapshot"]["resume"]["status"], "valid")
            self.assertTrue(report["snapshot"]["stale_version"]["rejected"])
            self.assertTrue(report["snapshot"]["stale_config"]["rejected"])
            self.assertTrue(all(item["passed"] for item in report["injections"]))
            self.assertTrue(Path(report["artifacts"]["mix_final_wav"]).exists())
            self.assertTrue(Path(report["artifacts"]["dashboard_html"]).exists())
            loaded = json.loads(Path(report["artifacts"]["report_json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["report_fingerprint"], report["report_fingerprint"])


if __name__ == "__main__":
    unittest.main()
