import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from receipt_ledger.engine import LedgerEngine
from receipt_ledger.errors import E_VIEW_STALE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def comparable(report: dict) -> dict:
    """剔除墙钟相关字段，保留全部账本字段做逐字段比较。"""

    stripped = json.loads(json.dumps(report, sort_keys=True))
    stripped.pop("view_flags", None)
    stripped.get("snapshot_manifest", {}).pop("global_version", None)
    if "elapsed_ms" in stripped:
        stripped.pop("elapsed_ms")
    return stripped


class EngineEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workers = os.path.join(ROOT, "samples", "workers.json")
        cls.samples = os.path.join(ROOT, "samples")
        cls.commands = os.path.join(ROOT, "samples", "commands.json")
        cls.faulty_workers = os.path.join(ROOT, "samples_faulty", "workers.json")
        cls.faulty_samples = os.path.join(ROOT, "samples_faulty")
        cls.faulty_commands = os.path.join(ROOT, "samples_faulty", "commands.json")
        cls.inject = os.path.join(ROOT, "samples_faulty", "injection.json")

    def fresh_dir(self, name: str) -> str:
        path = os.path.join(tempfile.gettempdir(), "receipt-ledger-test", name)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        return path

    def test_full_rebuild_matches_incremental_rerun_and_concurrency(self):
        serial_dir = self.fresh_dir("serial")
        parallel_dir = self.fresh_dir("parallel")
        rerun_dir = self.fresh_dir("rerun")

        serial = LedgerEngine(
            self.workers, self.samples, self.commands, serial_dir, max_workers=1
        ).run_authoritative()
        parallel = LedgerEngine(
            self.workers, self.samples, self.commands, parallel_dir, max_workers=8
        ).run_authoritative()

        first = LedgerEngine(
            self.workers, self.samples, self.commands, rerun_dir, max_workers=1
        ).run_authoritative()
        second = LedgerEngine(
            self.workers, self.samples, self.commands, rerun_dir, max_workers=4
        ).run_authoritative()

        self.assertEqual(
            comparable(serial)["ledger"], comparable(parallel)["ledger"]
        )
        self.assertEqual(
            comparable(serial)["resources"], comparable(parallel)["resources"]
        )
        self.assertEqual(
            comparable(serial)["failures"], comparable(parallel)["failures"]
        )
        self.assertEqual(
            comparable(serial)["todo_queue"], comparable(parallel)["todo_queue"]
        )
        for field in (
            "ledger",
            "resources",
            "failures",
            "todo_queue",
            "attribution_summary",
            "totals",
            "conversion_log",
        ):
            self.assertEqual(comparable(first)[field], comparable(second)[field], field)
        self.assertEqual(
            len(first["reuse_log"]),
            len(second["reuse_log"]),
        )
        self.assertTrue(
            all(item["action"] in {"reuse", "computed"} or item["stage"] == "invalidation"
                for item in second["reuse_log"])
        )

    def test_snapshot_manifest_tracks_versions_and_reuses_unchanged_stage(self):
        out_dir = self.fresh_dir("snapshot")
        engine = LedgerEngine(self.workers, self.samples, self.commands, out_dir)
        engine.run_authoritative()
        manifest_path = os.path.join(out_dir, "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(set(manifest["stages"]), {"parse", "attribution", "resources", "summary"})
        self.assertEqual(manifest["command_cursor"], 8)
        self.assertTrue(all(item["version"] == 1 for item in manifest["stages"].values()))

        engine_two = LedgerEngine(self.workers, self.samples, self.commands, out_dir)
        report_two = engine_two.run_authoritative()
        reuse_stages = {
            item["stage"]
            for item in report_two["reuse_log"]
            if item["action"] == "reuse" and item["stage"] in {"parse", "attribution", "resources", "summary"}
        }
        self.assertEqual(
            {"parse", "attribution", "resources", "summary"},
            reuse_stages,
        )

    def test_fast_view_matches_authoritative_field_by_field(self):
        out_dir = self.fresh_dir("fast")
        clock = FakeClock()
        engine = LedgerEngine(
            self.workers, self.samples, self.commands, out_dir, clock=clock
        )
        authoritative = engine.run_authoritative()

        fast = LedgerEngine(
            self.workers, self.samples, self.commands,
            os.path.join(out_dir, "fast"), clock=clock,
        ).fast_view(budget_ms=10_000)
        self.assertEqual(fast["mode"], "fast")
        self.assertEqual(fast["view_flags"], [])
        self.assertEqual(
            fast["attribution_summary"], authoritative["attribution_summary"]
        )
        self.assertEqual(
            fast["totals"]["invalid_entry_count"],
            authoritative["totals"]["invalid_entry_count"],
        )

    def test_fast_view_stale_returns_last_completed_conclusion(self):
        out_dir = self.fresh_dir("stale")
        clock = FakeClock()
        engine = LedgerEngine(
            self.workers, self.samples, self.commands, out_dir, clock=clock
        )
        engine.run_authoritative()
        clock.now += 100
        stale = engine.fast_view(budget_ms=1)
        self.assertIn(E_VIEW_STALE, stale["view_flags"])
        self.assertIn("progress", stale)

    def test_injection_isolation_and_hit_ids(self):
        out_dir = self.fresh_dir("inject")
        report = LedgerEngine(
            self.faulty_workers,
            self.faulty_samples,
            self.faulty_commands,
            out_dir,
            injection_path=self.inject,
        ).run_authoritative()
        hit_ids = set(report["injection"]["hit_ids"])
        self.assertEqual(
            hit_ids, {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
        )
        codes = {item["code"] for item in report["failures"]}
        self.assertIn("E_NON_FINITE_DURATION", codes)
        self.assertIn("E_RECEIPT_TRUNCATED", codes)
        self.assertIn("E_DUPLICATE_REQUEST_ID", codes)
        for failure in report["failures"]:
            if failure["injection_id"]:
                self.assertTrue(failure["injection_id"].startswith("INJ-"))

    def test_cli_writes_report_and_manifest(self):
        out_dir = self.fresh_dir("cli")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "receipt_ledger",
                "--config",
                self.workers,
                "--samples",
                self.samples,
                "--commands",
                self.commands,
                "--output",
                out_dir,
                "--workers",
                "3",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("report.json", result.stdout)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "report.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "snapshot_manifest.json")))


if __name__ == "__main__":
    unittest.main()
