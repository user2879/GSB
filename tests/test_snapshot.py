import json
import os
import tempfile
import unittest

from emergency_drill.engine import simulate
from emergency_drill.errors import ReplayError, SnapshotError
from emergency_drill.loader import load_bundle
from emergency_drill.repository import Ledger
from emergency_drill.sample_data import generate_samples
from emergency_drill.session import DrillSession, tamper_snapshot_digest


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        generate_samples(self.tmp.name)
        self.plan, self.events = load_bundle(
            os.path.join(self.tmp.name, "plan_valid.json"),
            os.path.join(self.tmp.name, "events_valid.json"),
        )
        self.ledger_path = os.path.join(self.tmp.name, "drill.sqlite3")
        self.ledger = Ledger(self.ledger_path)
        self.ledger.save_plan(self.plan, self.events)
        self.ledger.append_simulation(self.plan, simulate(self.plan, self.events))
        self.session = DrillSession(self.ledger)
        self.snapshot_path = os.path.join(self.tmp.name, "snapshot.json")

    def tearDown(self):
        self.ledger.close()
        self.tmp.cleanup()

    def test_snapshot_roundtrip_replay_matches(self):
        self.session.create_snapshot(1, self.snapshot_path)
        report = self.session.replay_snapshot(1, self.snapshot_path)
        self.assertTrue(report["matched"])
        self.assertEqual(len(report["checked_stage_ids"]), 5)

    def test_tampered_digest_raises_e_snapshot(self):
        self.session.create_snapshot(1, self.snapshot_path)
        bad_path = os.path.join(self.tmp.name, "bad.json")
        tamper_snapshot_digest(self.snapshot_path, bad_path)
        with self.assertRaises(SnapshotError) as caught:
            self.session.open_snapshot(1, bad_path)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")

    def test_version_mismatch_requests_full_replace_with_stable_diff(self):
        self.session.create_snapshot(1, self.snapshot_path)
        plan_v2 = self.plan.with_version(self.plan.stages, 2)
        self.ledger.save_plan(plan_v2, self.events)
        self.ledger.append_simulation(plan_v2, simulate(plan_v2, self.events))
        _snapshot, diff = self.session.open_snapshot(2, self.snapshot_path)
        self.assertTrue(diff["needs_full_replace"])
        self.assertEqual(diff["removed_stage_ids"], [])
        self.assertEqual(diff["snapshot_version"], 1)

    def test_replay_difference_locates_first_stage_and_field(self):
        self.session.create_snapshot(1, self.snapshot_path)
        with open(self.snapshot_path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        snapshot["state_machine"]["detect"]["state"] = "failed"
        # 保持摘要不符优先被拦截；直接篡改内部状态但重算摘要用于重放差异路径。
        from emergency_drill.models import digest_of

        body = {key: value for key, value in snapshot.items() if key != "content_digest"}
        snapshot["content_digest"] = digest_of(body)
        with open(self.snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle)
        with self.assertRaises(ReplayError) as caught:
            self.session.replay_snapshot(1, self.snapshot_path)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.context["stage_id"], "detect")
        self.assertEqual(caught.exception.context["field"], "state")


if __name__ == "__main__":
    unittest.main()
