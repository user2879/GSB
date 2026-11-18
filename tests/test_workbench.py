from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from emergency_drill.errors import (
    E_ABORT,
    E_CYCLE,
    E_PLAN,
    E_REPLAY,
    E_SNAPSHOT,
    E_STAGE,
    BusinessError,
    PlanValidationError,
    SimulatedCrash,
)
from emergency_drill.plan import load_plan
from emergency_drill.workbench import Workbench


ROOT = Path(__file__).resolve().parents[1]


class WorkbenchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="drill-tests-")
        self.root = Path(self.temp.name)
        self.db_path = self.root / "session.db"
        self.snapshots_dir = self.root / "snapshots"
        self.log_path = self.root / "logs" / "drill.jsonl"
        self.wb = Workbench(self.db_path, self.snapshots_dir, self.log_path)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def stage_map(self, summary: dict) -> dict[int, dict]:
        return {stage["seq"]: stage for stage in summary["stages"]}

    def test_plan_validation_is_strong_and_atomic(self) -> None:
        valid = load_plan(ROOT / "samples" / "plan_valid.json")
        self.assertEqual(valid.version, 1)
        self.assertEqual([stage.seq for stage in valid.stages], [1, 2, 3, 4, 5])

        with self.assertRaises(PlanValidationError) as corrupted:
            load_plan(ROOT / "samples" / "plan_corrupt.json")
        self.assertEqual(corrupted.exception.code, E_PLAN)
        self.assertEqual(corrupted.exception.stage_seq, 2)
        self.assertEqual(
            corrupted.exception.field_path, "stages[1].estimated_seconds"
        )

        with self.assertRaises(PlanValidationError) as cyclic:
            load_plan(ROOT / "samples" / "plan_cycle.json")
        self.assertEqual(cyclic.exception.code, E_CYCLE)
        self.assertEqual(cyclic.exception.stage_seq, 3)

        gap = valid.to_dict()
        gap["stages"] = gap["stages"][:1] + gap["stages"][2:]
        with self.assertRaises(PlanValidationError) as gap_error:
            from emergency_drill.plan import validate_plan_object

            validate_plan_object(gap)
        self.assertEqual(gap_error.exception.code, E_PLAN)

        bad_event = valid.to_dict()
        bad_event["events"][0]["offset_ms"] = -1
        with self.assertRaises(PlanValidationError) as event_error:
            from emergency_drill.plan import validate_plan_object

            validate_plan_object(bad_event)
        self.assertEqual(event_error.exception.code, E_PLAN)
        self.assertEqual(event_error.exception.stage_seq, 1)
        self.assertEqual(event_error.exception.field_path, "events[0].offset_ms")

        self.wb.create_session("atomic", ROOT / "samples" / "plan_valid.json")
        before = self.wb.repository.get_session("atomic")
        with self.assertRaises(PlanValidationError):
            self.wb.publish_plan("atomic", ROOT / "samples" / "plan_corrupt.json")
        after = self.wb.repository.get_session("atomic")
        self.assertEqual(before.plan_version, after.plan_version)
        self.assertEqual(before.generation, after.generation)
        self.assertEqual(before.plan_digest, after.plan_digest)

    def test_normal_run_ledger_events_snapshot_restore_and_replay(self) -> None:
        self.wb.create_session("normal", ROOT / "samples" / "plan_valid.json")
        summary = self.wb.run_session("normal")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["logical_clock_ms"], 12000)
        self.assertEqual([stage["status"] for stage in summary["stages"]], ["succeeded"] * 5)
        self.assertEqual([event["status"] for event in summary["events"]], ["triggered"] * 3)
        self.assertEqual(
            [stage["migration_count"] for stage in summary["stages"]], [2, 2, 2, 2, 2]
        )

        entries = self.wb.repository.ledger_rows("normal", 1)
        seqs = [entry.seq for entry in entries]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), len(set(seqs)))
        transitions = [entry for entry in entries if entry.kind == "stage_transition"]
        self.assertEqual(len(transitions), 10)
        for index, entry in enumerate(transitions):
            self.assertEqual(entry.logical_ms >= 0, True)
            self.assertEqual(entry.plan_version, 1)
            if index:
                self.assertGreater(entry.seq, transitions[index - 1].seq)

        snapshot = self.wb.save_snapshot("normal")
        restored = self.wb.restore_snapshot("normal", snapshot["file_path"])
        self.assertTrue(restored["whole_replacement"])
        self.assertFalse(restored["incremental_merge"])
        replay = self.wb.replay_snapshot("normal", snapshot["file_path"])
        self.assertTrue(replay.reproducible)
        self.assertEqual(replay.compared_stage_count, 5)
        self.assertTrue(all(item["equal"] for item in replay.comparisons))

        log_lines = self.log_path.read_text(encoding="utf-8").splitlines()
        self.assertTrue(log_lines)
        records = [json.loads(line) for line in log_lines]
        self.assertTrue(all(record["clock_source"] == "logical" for record in records))

    def test_failure_skip_and_continue_policies(self) -> None:
        self.wb.create_session("skip", ROOT / "samples" / "plan_valid.json")
        self.wb.inject_event(
            "skip",
            {
                "event_id": "fail-stage-3",
                "type": "switchover_failed",
                "offset_ms": 6000,
                "stage_seq": 3,
            },
        )
        summary = self.wb.run_session("skip")
        stages = self.stage_map(summary)
        self.assertEqual(stages[3]["status"], "failed")
        self.assertEqual(stages[4]["status"], "skipped")
        self.assertEqual(stages[5]["status"], "skipped")
        self.assertEqual(stages[4]["skip_from_stage_seq"], 3)
        self.assertEqual(stages[4]["business_code"], E_STAGE)
        self.assertIn("上游阶段 3 失败", stages[4]["skip_reason"])
        events = {event["event_id"]: event for event in summary["events"]}
        self.assertEqual(events["evt-validation-timeout-4"]["status"], "cancelled")
        ledger = self.wb.repository.ledger_rows("skip", 1)
        skip_entries = [
            item
            for item in ledger
            if item.kind == "stage_transition" and item.to_status == "skipped"
        ]
        self.assertEqual([item.stage_seq for item in skip_entries], [4, 5])
        self.assertTrue(all(item.code == E_STAGE for item in skip_entries))

        plan_raw = json.loads(
            (ROOT / "samples" / "plan_valid.json").read_text(encoding="utf-8")
        )
        plan_raw["stages"][2]["failure_policy"] = "continue_downstream"
        self.wb.create_session("continue", plan_raw)
        self.wb.inject_event(
            "continue",
            {
                "event_id": "fail-stage-3-continue",
                "type": "switchover_failed",
                "offset_ms": 6000,
                "stage_seq": 3,
            },
        )
        continued = self.wb.run_session("continue")
        continued_stages = self.stage_map(continued)
        self.assertEqual(continued_stages[3]["status"], "failed")
        self.assertEqual(continued_stages[4]["status"], "succeeded")
        self.assertEqual(continued_stages[5]["status"], "succeeded")

    def test_abort_skips_remaining_and_records_code(self) -> None:
        self.wb.create_session("abort", ROOT / "samples" / "plan_valid.json")
        summary = self.wb.run_session("abort", abort_after_stage=2)
        self.assertEqual(summary["status"], "aborted")
        stages = self.stage_map(summary)
        self.assertEqual(stages[1]["status"], "succeeded")
        self.assertEqual(stages[2]["status"], "succeeded")
        self.assertEqual(stages[3]["status"], "skipped")
        self.assertEqual(stages[3]["business_code"], E_ABORT)
        self.assertEqual(stages[5]["status"], "skipped")

    def test_plan_refresh_replaces_generation_and_marks_old_stale(self) -> None:
        self.wb.create_session("refresh", ROOT / "samples" / "plan_valid.json")
        self.wb.run_session("refresh")
        report = self.wb.publish_plan(
            "refresh", ROOT / "samples" / "plan_valid_v2.json", auto_bump=True
        )
        self.assertEqual(report["old_plan_version"], 1)
        self.assertEqual(report["new_plan_version"], 2)
        self.assertEqual(report["generation"], 2)
        self.assertEqual(report["diff"]["added_stage_seqs"], [6])
        self.assertEqual(report["diff"]["removed_stage_seqs"], [])
        changed = {item["seq"]: item["fields"] for item in report["diff"]["changed_stages"]}
        self.assertIn("name", changed[1])
        self.assertIn("estimated_seconds", changed[2])

        model = self.wb.render_model("refresh")
        self.assertEqual(model["plan_version"], 2)
        self.assertEqual(len(model["stages"]), 6)
        self.assertTrue(all(stage["status"] == "pending" for stage in model["stages"]))
        self.assertEqual(model["generation"], 2)
        entries = self.wb.repository.ledger_rows("refresh", 2)
        self.assertEqual([entry.kind for entry in entries], ["plan_replaced"])

        with sqlite3.connect(self.db_path) as connection:
            old_state_count = connection.execute(
                "SELECT COUNT(*) FROM phase_states WHERE session_id='refresh' AND generation=1 AND stale=1"
            ).fetchone()[0]
            old_event_count = connection.execute(
                "SELECT COUNT(*) FROM events WHERE session_id='refresh' AND generation=1 AND stale=1"
            ).fetchone()[0]
            old_plan_stale = connection.execute(
                "SELECT stale FROM plan_versions WHERE session_id='refresh' AND version=1"
            ).fetchone()[0]
        self.assertEqual(old_state_count, 5)
        self.assertEqual(old_event_count, 3)
        self.assertEqual(old_plan_stale, 1)

        with self.assertRaises(PlanValidationError):
            self.wb.publish_plan(
                "refresh", ROOT / "samples" / "plan_valid.json", auto_bump=False
            )
        bumped = self.wb.publish_plan(
            "refresh", ROOT / "samples" / "plan_valid_v2.json", auto_bump=True
        )
        self.assertEqual(bumped["new_plan_version"], 3)

        removed = json.loads(
            (ROOT / "samples" / "plan_valid_v2.json").read_text(encoding="utf-8")
        )
        removed["stages"] = [stage for stage in removed["stages"] if stage["seq"] != 6]
        removed_report = self.wb.publish_plan("refresh", removed, auto_bump=True)
        self.assertEqual(removed_report["new_plan_version"], 4)
        self.assertEqual(removed_report["diff"]["removed_stage_seqs"], [6])
        self.assertEqual(removed_report["diff"]["added_stage_seqs"], [])

    def test_snapshot_restore_is_whole_replacement_and_old_is_read_only(self) -> None:
        self.wb.create_session("restore", ROOT / "samples" / "plan_valid.json")
        self.wb.run_session("restore")
        old_snapshot = self.wb.save_snapshot("restore")
        self.wb.publish_plan("restore", ROOT / "samples" / "plan_valid_v2.json")
        new_snapshot = self.wb.save_snapshot("restore")
        listed = {item["snapshot_id"]: item for item in self.wb.list_snapshots("restore")}
        self.assertTrue(listed[old_snapshot["snapshot_id"]]["read_only"])
        self.assertFalse(listed[new_snapshot["snapshot_id"]]["read_only"])

        restored = self.wb.restore_snapshot("restore", old_snapshot["file_path"])
        self.assertTrue(restored["whole_replacement"])
        self.assertEqual(restored["restored_plan_version"], 1)
        self.assertFalse(restored["plan_version_matches"])
        self.assertFalse(restored["plan_digest_matches"])
        model = self.wb.render_model("restore")
        self.assertEqual(model["plan_version"], 1)
        self.assertEqual([stage["status"] for stage in model["stages"]], ["succeeded"] * 5)
        listed_after = self.wb.list_snapshots("restore")
        self.assertTrue(all(item["read_only"] for item in listed_after))
        self.assertTrue(Path(old_snapshot["file_path"]).exists())
        replay = self.wb.replay_snapshot("restore", old_snapshot["file_path"])
        self.assertTrue(replay.reproducible)

    def test_snapshot_digest_mismatch_is_e_snapshot(self) -> None:
        self.wb.create_session("digest", ROOT / "samples" / "plan_valid.json")
        self.wb.run_session("digest")
        snapshot = self.wb.save_snapshot("digest")
        payload = json.loads(Path(snapshot["file_path"]).read_text(encoding="utf-8"))
        payload["phase_states"][0]["status"] = "failed"
        tampered = self.root / "tampered.json"
        tampered.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(BusinessError) as raised:
            self.wb.load_snapshot(tampered)
        self.assertEqual(raised.exception.code, E_SNAPSHOT)
        self.assertEqual(raised.exception.field_path, "content_digest")
        self.assertEqual(raised.exception.stage_seq, 1)

    def test_forced_crash_recovery_unknown_tail_resume_and_replay(self) -> None:
        self.wb.create_session("crash", ROOT / "samples" / "plan_valid.json")
        with self.assertRaises(SimulatedCrash) as crash:
            self.wb.run_session("crash", crash_after_stage=2)
        self.assertEqual(crash.exception.stage_seq, 3)
        report = self.wb.recover("crash")
        self.assertEqual(report.recovered_stage_count, 2)
        self.assertEqual(report.recovered_stage_seqs, [1, 2])
        self.assertEqual(report.unknown_stage_seqs, [3])
        self.assertEqual(report.pending_stage_seqs, [4, 5])
        self.assertEqual(len(report.incomplete_tail_seqs), 1)
        self.assertFalse(report.exit_report_available)
        self.assertIn("不完整尾记录", report.tail_action)
        model = self.wb.render_model("crash")
        self.assertEqual(model["unknown_stage_seqs"], [3])
        self.assertTrue(model["stages"][2]["unknown"])

        resumed = self.wb.resume("crash")
        self.assertEqual(resumed["recovery"]["unknown_stage_seqs"], [3])
        self.assertEqual(resumed["run"]["status"], "completed")
        self.assertTrue(
            all(stage["status"] == "succeeded" for stage in resumed["run"]["stages"])
        )
        snapshot = self.wb.save_snapshot("crash")
        restored = self.wb.restore_snapshot("crash", snapshot["file_path"])
        self.assertTrue(restored["whole_replacement"])
        replay = self.wb.replay_snapshot("crash", snapshot["file_path"])
        self.assertTrue(replay.reproducible)

    def test_replay_reports_first_inconsistent_stage_and_field(self) -> None:
        self.wb.create_session("replay", ROOT / "samples" / "plan_valid.json")
        self.wb.run_session("replay")
        snapshot = self.wb.save_snapshot("replay")
        self.wb.restore_snapshot("replay", snapshot["file_path"])
        self.wb.inject_event(
            "replay",
            {
                "event_id": "late-extra",
                "type": "switchover_failed",
                "offset_ms": 6100,
                "stage_seq": 3,
            },
        )
        with self.assertRaises(BusinessError) as event_mismatch:
            self.wb.replay_snapshot("replay", snapshot["file_path"])
        self.assertEqual(event_mismatch.exception.code, E_REPLAY)
        self.assertEqual(event_mismatch.exception.stage_seq, 3)
        self.assertEqual(event_mismatch.exception.field_path, "events.length")

        self.wb.restore_snapshot("replay", snapshot["file_path"])
        self.wb.publish_plan("replay", ROOT / "samples" / "plan_valid_v2.json")
        with self.assertRaises(BusinessError) as plan_mismatch:
            self.wb.replay_snapshot("replay", snapshot["file_path"])
        self.assertEqual(plan_mismatch.exception.code, E_REPLAY)
        self.assertEqual(plan_mismatch.exception.stage_seq, 1)
        self.assertEqual(plan_mismatch.exception.field_path, "plan_version")

    def test_render_order_is_stable(self) -> None:
        self.wb.create_session("render", ROOT / "samples" / "plan_valid.json")
        model = self.wb.render_model("render")
        self.assertEqual([stage["seq"] for stage in model["stages"]], [1, 2, 3, 4, 5])
        self.assertEqual(
            [event["injection_seq"] for event in model["events"]], [1, 2, 3]
        )
        self.assertEqual(model["clock_source"], "logical")
        self.assertEqual(model["plan_version"], 1)


if __name__ == "__main__":
    unittest.main()
