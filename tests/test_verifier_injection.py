from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pagecache.cli import main as cli_main
from pagecache.verifier import CHECK_ORDER, verify_artifacts
from tests.helpers import build_snapshot_run, make_samples, read_json, write_json


class VerifierAndInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data, cls.meta = make_samples(cls.root)
        cls.run_dir = cls.root / "verify-run"
        cls.artifacts = build_snapshot_run(data=cls.data, meta=cls.meta, run_dir=cls.run_dir, position=5)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def _verify(self, output_path: Path) -> dict:
        return verify_artifacts(
            snapshot_path=self.artifacts["snapshot_path"],
            output_path=output_path,
            manifest_path=self.artifacts["manifest_path"],
            pages_dir=self.artifacts["pages_dir"],
            overflow_dir=self.artifacts["overflow_dir"],
        )

    def test_verifier_passes_and_uses_stable_check_order(self) -> None:
        report = self._verify(self.artifacts["output_path"])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual([check["name"] for check in report["checks"]], list(CHECK_ORDER))
        self.assertEqual(report["summary"]["checks_failed"], 0)

    def test_verifier_detects_bad_mean_and_error_bound(self) -> None:
        output = read_json(self.artifacts["output_path"])
        output["groups"][0]["mean_scaled"] += 1
        output["groups"][0]["sum_abs_error_upper_bound"]["numerator"] += 1
        broken_output = self.root / "broken-output.json"
        write_json(broken_output, output)
        report = self._verify(broken_output)
        self.assertEqual(report["status"], "FAIL")
        statuses = {check["name"]: check["status"] for check in report["checks"]}
        self.assertEqual(statuses["mean_derivation"], "FAIL")
        self.assertEqual(statuses["sum_error_bound"], "FAIL")
        self.assertEqual([check["name"] for check in report["checks"]], list(CHECK_ORDER))

    def test_cli_injections_report_three_required_codes_and_structured_log(self) -> None:
        log_path = self.root / "injection.log"
        manifest = self.artifacts["manifest_path"]
        pages = self.artifacts["pages_dir"]
        overflow = self.artifacts["overflow_dir"]
        work = self.root / "injections"
        commands = [
            ["inject", "--kind", "manifest_tail_truncate", "--manifest", str(manifest), "--pages-dir", str(pages), "--overflow-dir", str(overflow), "--work-dir", str(work / "manifest"), "--log", str(log_path)],
            ["inject", "--kind", "segment_checksum_mismatch", "--manifest", str(manifest), "--pages-dir", str(pages), "--overflow-dir", str(overflow), "--work-dir", str(work / "segment"), "--log", str(log_path)],
            ["inject", "--kind", "memory_budget_zero", "--manifest", str(manifest), "--pages-dir", str(pages), "--overflow-dir", str(overflow), "--work-dir", str(work / "budget"), "--log", str(log_path)],
        ]
        results = []
        for command in commands:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = cli_main(command)
            self.assertEqual(exit_code, 0)
            results.append(json.loads(buffer.getvalue()))
        self.assertEqual([result["actual_code"] for result in results], ["E_FORMAT", "E_INTEGRITY", "E_BUDGET"])
        log_rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["code"] for row in log_rows], ["E_FORMAT", "E_INTEGRITY", "E_BUDGET"])
        self.assertTrue(all(row["event"] == "injection" for row in log_rows))


if __name__ == "__main__":
    unittest.main()
