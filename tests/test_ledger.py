from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from affinity_scheduler.errors import E_SNAPSHOT, SnapshotError
from affinity_scheduler.ledger import (
    begin_session,
    commit_pod_result,
    entry_from_result,
    load_entries_readonly,
    recover_ledger,
    snapshot_content_digest,
    sync_snapshot,
)

from tests.helpers import fake_result


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "ledger.sqlite3"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _entries(self, results: list[dict]) -> dict[str, dict]:
        return {
            result["pod_id"]: entry_from_result(
                result,
                format_versions=["1", "2"],
                status="committed",
            )
            for result in results
        }

    def test_snapshot_replacement_lists_added_deleted_replaced(self) -> None:
        first = self._entries(
            [fake_result("pod-0001", "node-a"), fake_result("pod-0002", "node-b")]
        )
        report = sync_snapshot(
            self.db,
            first,
            format_versions=["1", "2"],
            parameter_version=1,
            scale_factor=1000,
        )
        self.assertFalse(report.mismatch)
        second = self._entries(
            [fake_result("pod-0001", "node-c", total_score=60), fake_result("pod-0003", "node-d")]
        )
        report = sync_snapshot(
            self.db,
            second,
            format_versions=["3"],
            parameter_version=2,
            scale_factor=1000,
        )
        self.assertTrue(report.mismatch)
        self.assertEqual(
            report.content_digest,
            snapshot_content_digest(
                second,
                format_versions=["3"],
                parameter_version=2,
                scale_factor=1000,
            ),
        )
        self.assertEqual(report.added, ("pod-0003",))
        self.assertEqual(report.deleted, ("pod-0002",))
        self.assertEqual(report.replaced, ("pod-0001",))
        self.assertEqual(set(load_entries_readonly(self.db)), {"pod-0001", "pod-0003"})

    def test_injected_digest_mismatch_reports_e_snapshot_after_replace(self) -> None:
        entries = self._entries([fake_result("pod-0001", "node-a")])
        sync_snapshot(
            self.db,
            entries,
            format_versions=["3"],
            parameter_version=1,
            scale_factor=1000,
        )
        with self.assertRaises(SnapshotError) as caught:
            sync_snapshot(
                self.db,
                entries,
                format_versions=["3"],
                parameter_version=1,
                scale_factor=1000,
                inject_digest_mismatch=True,
            )
        self.assertEqual(caught.exception.code, E_SNAPSHOT)
        self.assertEqual(set(load_entries_readonly(self.db)), {"pod-0001"})

    def test_forced_termination_recovery_only_trusts_committed_rows(self) -> None:
        session_id = begin_session(
            self.db,
            ["pod-0001", "pod-0002", "pod-0003"],
            format_versions=["3"],
            parameter_version=1,
            scale_factor=1000,
        )
        commit_pod_result(
            self.db,
            session_id,
            fake_result("pod-0001", "node-a"),
            format_versions=["3"],
            cursor=1,
        )
        recovery = recover_ledger(self.db)
        self.assertEqual(recovery["committed"], ["pod-0001"])
        self.assertEqual(recovery["unknown"], ["pod-0002", "pod-0003"])
        self.assertTrue(recovery["requires_rerun"])
        self.assertFalse(recovery["exit_report_available"])
        self.assertEqual(recovery["recovery_basis"], "committed_only")


if __name__ == "__main__":
    unittest.main()
