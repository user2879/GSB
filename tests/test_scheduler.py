from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from batch_audit.audit import run_audit
from batch_audit.engine import RunOptions, Scheduler, WorkspacePaths
from batch_audit.snapshot import read_snapshot, write_snapshot

from .helpers import assignment_map, load_json, make_workspace


class SchedulerTests(unittest.TestCase):
    def test_slot_processing_order_does_not_change_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ws_a, _ = make_workspace(root / "a")
            ws_b, _ = make_workspace(root / "b")
            report_a = Scheduler(
                WorkspacePaths.under(ws_a),
                RunOptions(supported_slots=(0, 1, 2, 3, 4)),
            ).run()
            report_b = Scheduler(
                WorkspacePaths.under(ws_b),
                RunOptions(supported_slots=(0, 1, 2, 3, 4), slot_order=(4, 3, 2, 1, 0)),
            ).run()
            self.assertEqual(assignment_map(report_a), assignment_map(report_b))
            self.assertEqual(report_a["empty_slots"], [4])
            self.assertEqual(report_a["status"], "complete")

    def test_feed_file_order_does_not_change_grouping_or_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ws_a, _ = make_workspace(root / "a")
            ws_b, _ = make_workspace(root / "b")
            feed = load_json(ws_a / "feed.json")
            feed["entries"] = list(reversed(feed["entries"]))
            (ws_b / "feed.json").write_text(
                json.dumps(feed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report_a = Scheduler(WorkspacePaths.under(ws_a)).run()
            report_b = Scheduler(WorkspacePaths.under(ws_b)).run()
            self.assertEqual(assignment_map(report_a), assignment_map(report_b))
            self.assertEqual(
                [slot["participant_sample_ids"] for slot in report_a["slots"]],
                [slot["participant_sample_ids"] for slot in report_b["slots"]],
            )

    def test_retransmission_is_overwritten_not_reassigned(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            report = Scheduler(WorkspacePaths.under(ws)).run()
            overwritten = report["overwritten_entries"]
            self.assertEqual(len(overwritten), 2)
            self.assertTrue(all(item["kept_arrival_seq"] < item["arrival_seq"] for item in overwritten))
            all_ids = [member["sample_id"] for slot in report["slots"] for member in slot["members"]]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            for slot in report["slots"]:
                for member in slot["members"]:
                    self.assertEqual(member["batch_output_digest"], slot["output_digest"])

    def test_replay_is_idempotent_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            first = Scheduler(paths).run()
            second = Scheduler(paths).run()
            self.assertEqual(first["generation"], second["generation"])
            self.assertEqual(assignment_map(first), assignment_map(second))
            audit = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self.assertEqual(audit["code"], "AUDIT_OK")
            self.assertEqual(audit["audit_scope"], "full")

    def test_file_reordering_after_commit_does_not_stale_or_break_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            first = Scheduler(paths).run()
            feed = load_json(paths.feed)
            feed["entries"] = list(reversed(feed["entries"]))
            paths.feed.write_text(
                json.dumps(feed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            second = Scheduler(paths).run()
            self.assertEqual(first["generation"], second["generation"])
            self.assertEqual(assignment_map(first), assignment_map(second))
            self.assertEqual(run_audit(ledger_path=paths.ledger, feed_path=paths.feed)["code"], "AUDIT_OK")

    def test_numbering_version_change_invalidates_previous_ids_as_a_whole(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            first = Scheduler(paths).run()
            second = Scheduler(paths, RunOptions(numbering_version="BA-V2")).run()
            self.assertEqual(first["generation"], 1)
            self.assertEqual(second["generation"], 2)
            self.assertEqual(second["version_evolution"]["affected_samples"], 10)
            self.assertEqual(second["version_evolution"]["affected_slots"], [0, 1, 2, 3, 4])
            old_ids = {value[2] for value in assignment_map(first).values()}
            current_ids = {value[2] for value in assignment_map(second).values()}
            self.assertTrue(current_ids.isdisjoint(old_ids))
            self.assertTrue(all(identifier.startswith("BA-V2:") for identifier in current_ids))
            audit = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self.assertEqual(audit["generation"], 2)
            self.assertEqual(audit["code"], "AUDIT_OK")

    def test_snapshot_version_mismatch_replaces_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            first = Scheduler(paths).run()
            snapshot = read_snapshot(paths.snapshot)
            assert snapshot is not None
            write_snapshot(
                paths.snapshot,
                type(snapshot)(
                    generation=99,
                    fingerprint=snapshot.fingerprint,
                    status=snapshot.status,
                    completed_slots=snapshot.completed_slots,
                    empty_slots=snapshot.empty_slots,
                    unknown_slots=snapshot.unknown_slots,
                ),
            )
            second = Scheduler(paths).run()
            self.assertGreater(second["generation"], first["generation"])
            self.assertEqual(second["generation"], 100)
            self.assertEqual(second["version_evolution"]["replacement_reason"], "snapshot_version_mismatch")
            self.assertEqual(second["version_evolution"]["affected_samples"], 10)

    def test_model_asset_evolution_creates_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ws, _ = make_workspace(Path(temp))
            paths = WorkspacePaths.under(ws)
            first = Scheduler(paths).run()
            make_workspace(Path(temp), seed=20261121)
            second = Scheduler(paths).run()
            self.assertEqual(second["generation"], first["generation"] + 1)
            self.assertEqual(second["version_evolution"]["replacement_reason"], "fingerprint_changed")
            self.assertNotEqual(
                first["fingerprints"]["param_digest"],
                second["fingerprints"]["param_digest"],
            )
            self.assertEqual(run_audit(ledger_path=paths.ledger, feed_path=paths.feed)["code"], "AUDIT_OK")


if __name__ == "__main__":
    unittest.main()
