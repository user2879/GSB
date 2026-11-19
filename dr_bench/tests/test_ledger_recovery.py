import json
import tempfile
import unittest
from pathlib import Path

from dr_bench.engine import PATH_INCREMENTAL, run_full, run_incremental
from dr_bench.injection import (
    FAULT_BAD_RATIO,
    FAULT_CORRUPT_SCENE,
    FAULT_MISSING_SNAPSHOT_VERSION,
    StructuredLogger,
    exercise_fault,
    strip_snapshot_version,
)
from dr_bench.errors import InjectError, SceneError, SnapshotError
from dr_bench.ledger import Ledger, read_snapshot, write_snapshot, scene_digest
from dr_bench.loader import parse_scene
from dr_bench.recovery import replay_from_ledger, run_session_with_crash
from dr_bench.samplegen import build_demo_scene


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.scene = parse_scene(build_demo_scene())

    def tearDown(self):
        self.temp.cleanup()

    def test_persist_and_replay_matches(self):
        db = self.root / "session.sqlite3"
        ledger = Ledger(db)
        recovery = ledger.initialize_session(self.scene, PATH_INCREMENTAL)
        self.assertIsNone(recovery.last_committed_round)
        conclusions = run_incremental(self.scene)
        for conclusion in conclusions:
            ledger.begin_round(conclusion["round_index"], PATH_INCREMENTAL)
            ledger.commit_round(conclusion["round_index"], PATH_INCREMENTAL, conclusion)
        ledger.close()

        second = Ledger(db)
        report = replay_from_ledger(self.scene, second, PATH_INCREMENTAL)
        second.close()
        self.assertEqual(report.verified_rounds, list(self.scene.rounds))
        self.assertEqual(report.rerun_rounds, [])

    def test_crash_pending_round_is_rerun(self):
        db = self.root / "crash.sqlite3"
        ledger = Ledger(db)
        crash = run_session_with_crash(
            self.scene, ledger, PATH_INCREMENTAL, crash_round=3
        )
        ledger.close()
        self.assertEqual(crash["pending_round"], 3)
        self.assertNotIn(3, crash["committed_rounds"])

        recovered = Ledger(db)
        state = recovered.initialize_session(self.scene, PATH_INCREMENTAL)
        self.assertEqual(state.pending_rounds, [3])
        report = replay_from_ledger(self.scene, recovered, PATH_INCREMENTAL)
        recovered.close()
        self.assertIn(3, report.rerun_rounds)
        self.assertEqual(sorted(report.verified_rounds + report.rerun_rounds), list(self.scene.rounds))

    def test_digest_mismatch_triggers_wholesale_replacement(self):
        db = self.root / "replace.sqlite3"
        ledger = Ledger(db)
        ledger.initialize_session(self.scene, PATH_INCREMENTAL)
        ledger.close()

        changed = build_demo_scene()
        changed["scene_version"] = 2
        changed["services"][0]["cooldown_ms"] = 9999
        new_scene = parse_scene(changed)
        second = Ledger(db)
        state = second.initialize_session(new_scene, PATH_INCREMENTAL)
        second.close()
        self.assertTrue(state.replaced)
        self.assertEqual(state.recovered_rounds, [])

    def test_snapshot_round_trip_and_missing_version(self):
        db = self.root / "snap.sqlite3"
        snapshot_dir = self.root / "snapshots"
        ledger = Ledger(db)
        ledger.initialize_session(self.scene, PATH_INCREMENTAL)
        target = write_snapshot(snapshot_dir, self.scene, ledger)
        ledger.close()
        payload = read_snapshot(snapshot_dir, self.scene.scene_id)
        self.assertIn("snapshot_format_version", payload)

        strip_snapshot_version(target)
        with self.assertRaises(SnapshotError) as ctx:
            read_snapshot(snapshot_dir, self.scene.scene_id)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_injection_faults_log_structured_lines(self):
        log_path = self.root / "events.log.jsonl"
        logger = StructuredLogger(log_path)
        with self.assertRaises(SceneError):
            exercise_fault(
                FAULT_CORRUPT_SCENE,
                {"scene_dict": build_demo_scene()},
                logger,
            )
        with self.assertRaises(InjectError):
            exercise_fault(FAULT_BAD_RATIO, {"ratio": 1.5}, logger)

        db = self.root / "fault.sqlite3"
        ledger = Ledger(db)
        ledger.initialize_session(self.scene, PATH_INCREMENTAL)
        snapshot_file = write_snapshot(self.root / "snaps", self.scene, ledger)
        ledger.close()
        strip_snapshot_version(snapshot_file)
        with self.assertRaises(SnapshotError):
            exercise_fault(
                FAULT_MISSING_SNAPSHOT_VERSION,
                {"snapshot_dir": self.root / "snaps", "scene_id": self.scene.scene_id},
                logger,
            )
        lines = [json.loads(line) for line in log_path.read_text().splitlines()]
        self.assertEqual([line["code"] for line in lines], ["E_SCENE", "E_INJECT", "E_SNAPSHOT"])


class PlannerTests(unittest.TestCase):
    def test_planner_is_deterministic_and_stable(self):
        from dr_bench.planner import select_path

        scene = parse_scene(build_demo_scene())
        first = select_path(scene.scale(), 0.25)
        second = select_path(scene.scale(), 0.25)
        self.assertEqual(first["selected_path"], second["selected_path"])
        self.assertEqual(first["ranking"], second["ranking"])
        for item in first["ranking"]:
            self.assertIn("estimated_total", item)


if __name__ == "__main__":
    unittest.main()
