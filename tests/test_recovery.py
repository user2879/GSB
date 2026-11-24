from __future__ import annotations

import sys
import unittest
import copy
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from colpage_recovery.demo import run_end_to_end_demo, snapshot_injection_report
from colpage_recovery.engine import ReplayEngine
from colpage_recovery.errors import ReplayMismatchError, SnapshotMismatchError
from colpage_recovery.loader import build_bundle, load_json_file
from colpage_recovery.service import DiagnosticService
from colpage_recovery.viewmodel import build_state_view_model
SAMPLES = ROOT / "samples"


class RecoveryTest(unittest.TestCase):
    def test_clean_replay_recover_and_snapshot_tamper(self) -> None:
        state_dir = ROOT / "state" / "test-clean-recovery"
        with DiagnosticService(state_dir=state_dir, sample_dir=SAMPLES) as service:
            service.initialize(reset=True)
            run = service.replay_and_commit()
            self.assertEqual(8, run["cursor_lsn"])
            report = service.recover_and_verify()
            self.assertEqual("verified", report["status"])
            self.assertEqual([], report["unknown_batches"])

            generation = service.generation
            ledger = service.require_ledger()
            ledger.tamper_content_digest(generation)
            with self.assertRaises(SnapshotMismatchError) as ctx:
                service.verify_ledger_content_files(generation)
            self.assertEqual("E_SNAPSHOT", ctx.exception.code)
            self.assertEqual("content_digest", ctx.exception.field)
            self.assertIsNotNone(ctx.exception.page_id)

    def test_spec_change_increments_generation_and_marks_stale(self) -> None:
        state_dir = ROOT / "state" / "test-spec-evolution"
        with DiagnosticService(state_dir=state_dir, sample_dir=SAMPLES) as service:
            first_generation = service.initialize(reset=True)
            service.replay_and_commit()
            original_budget = service.bundle.budget
            service.bundle = replace(
                service.bundle,
                budget=replace(original_budget, global_slots=original_budget.global_slots + 2),
            )
            second_generation = service.initialize(reset=False)
            self.assertEqual(first_generation + 1, second_generation)
            stale_rows = service.ledger.query_dicts(
                "SELECT generation,stale FROM spec_generation ORDER BY generation"
            )
            self.assertEqual([(1, 1), (2, 0)], [(row["generation"], row["stale"]) for row in stale_rows])

        model = build_state_view_model(state_dir)
        self.assertEqual(2, model["current_generation"])
        self.assertGreaterEqual(model["stale_counts"]["generations"], 1)

    def test_snapshot_injection_report_code(self) -> None:
        report = snapshot_injection_report(
            SAMPLES, ROOT / "state" / "test-snapshot-injection"
        )
        self.assertFalse(report["ok"])
        self.assertEqual("E_SNAPSHOT", report["error"]["code"])

    def test_replay_mismatch_locates_first_page_and_field(self) -> None:
        state_dir = ROOT / "state" / "test-replay-mismatch"
        with DiagnosticService(state_dir=state_dir, sample_dir=SAMPLES) as service:
            service.initialize(reset=True)
            service.replay_and_commit()
            reference = service.engine.replay_all()
            bad = service.engine.replay_all(batch_size=4)
            bad.records[0].content_digest = "f" * 64
            bad.batches[0].records[0].content_digest = "f" * 64
            with self.assertRaises(ReplayMismatchError) as ctx:
                service._compare_replay_to_reference_and_ledger(
                    bad,
                    reference,
                    check_ledger=False,
                    phase="unit-forced-mismatch",
                )
            self.assertEqual("E_REPLAY", ctx.exception.code)
            self.assertEqual("content_digest", ctx.exception.field)
            self.assertEqual(1, ctx.exception.lsn)
            self.assertEqual("P-A", ctx.exception.page_id)

    def test_latest_uncommitted_physical_version_excluded_from_current_view(self) -> None:
        pages_payload = load_json_file(SAMPLES / "pages.json")
        wal_payload = load_json_file(SAMPLES / "wal.json")
        budget_payload = load_json_file(SAMPLES / "budget.json")
        wal_payload = copy.deepcopy(wal_payload)
        wal_payload["entries"] = [
            entry for entry in wal_payload["entries"] if entry["lsn"] <= 6
        ]
        bundle = build_bundle(pages_payload, wal_payload, budget_payload)
        state_dir = ROOT / "state" / "test-latest-invisible-view"
        with DiagnosticService(state_dir=state_dir, bundle=bundle) as service:
            service.initialize(reset=True)
            service.replay_and_commit()
            report = service.recover_and_verify()
            self.assertEqual("verified", report["status"])
            physical = {
                row["page_id"]: row
                for row in service.ledger.latest_page_versions(service.generation)
            }
            current_view = service.ledger.current_view(service.generation)
            visible = {row["page_id"]: row for row in current_view["rows"]}
            self.assertEqual(3, physical["P-A"]["version"])
            self.assertEqual(0, physical["P-A"]["visible"])
            self.assertEqual(2, visible["P-A"]["version"])
            self.assertEqual(1, visible["P-A"]["visible"])
            self.assertEqual(2, visible["P-B"]["version"])

    def test_force_kill_subprocess_recovery_and_visibility(self) -> None:
        report = run_end_to_end_demo(
            SAMPLES,
            ROOT / "state" / "test-e2e-demo",
            report_path=None,
        )
        self.assertTrue(report["force_kill"]["nonzero_exit_as_expected"])
        self.assertNotEqual(0, report["force_kill"]["returncode"])
        if sys.platform != "win32":
            self.assertEqual(137, report["force_kill"]["returncode"])
        self.assertEqual(1, len(report["recovery"]["unknown_batches"]))
        self.assertEqual("verified", report["recovery"]["status"])
        self.assertTrue(report["spec_evolution"]["monotonic_increment"])
        invisible = report["state_view"]["invisible_versions"]
        self.assertEqual(
            {("P-A", 3, 5), ("P-B", 3, 6)},
            {(item["page_id"], item["version"], item["lsn"]) for item in invisible},
        )
        self.assertEqual(
            ["T1", "T2", "T4"],
            report["recovery"]["visibility"]["committed_txids"],
        )
        phases = [step["phase"] for step in report["recovery"]["verification_steps"]]
        self.assertIn("committed_checkpoint_overlap", phases)
        self.assertIn("full_recovery_interval", phases)


if __name__ == "__main__":
    unittest.main()
