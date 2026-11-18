import json
import tempfile
import unittest
from pathlib import Path

from qos.changeset import ChangeSet
from qos.errors import ReplayError, SnapshotError
from qos.models import parse_rules
from qos.ruleset import load_rules
from qos.snapshot import Snapshot, load_snapshot, save_snapshot
from qos.workbench import Workbench

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "ledger.db")
        self.log = str(Path(self.tmp.name) / "qos.jsonl")
        self.snap_path = str(Path(self.tmp.name) / "snapshot.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _wb(self):
        return Workbench(self.db, self.log)

    def test_snapshot_roundtrip_and_replay(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        baseline = list(wb.rules)
        cs = ChangeSet.from_dict(json.loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        wb.apply_change(cs)
        wb.simulate()
        wb.create_snapshot(self.snap_path)
        result = wb.replay(self.snap_path, baseline_rules=baseline)
        self.assertTrue(result["ok"])
        self.assertIn("plan_hash", result)
        wb.close()

    def test_missing_version_raises_e_snap(self):
        with self.assertRaises(SnapshotError) as ctx:
            load_snapshot(SAMPLES / "snapshot_missing_versions.json")
        self.assertEqual(ctx.exception.code, "E_SNAP")
        self.assertIn("plan_version", ctx.exception.details["missing_fields"])

    def test_version_mismatch_triggers_full_replace_with_change_list(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        wb.create_snapshot(self.snap_path)
        # 当前会话再做一次修改，造成版本差异
        cs = ChangeSet.from_dict(json.loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        wb.apply_change(cs)
        result = wb.restore_snapshot(self.snap_path)
        diff = result["diff"]
        self.assertFalse(diff["version_match"])
        self.assertEqual(diff["restore_mode"], "full-replace")
        self.assertIn("R1_BASE", diff["changes"]["modified"])
        wb.close()

    def test_replay_detects_tampered_snapshot_e_replay(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        baseline = list(wb.rules)
        cs = ChangeSet.from_dict(json.loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        wb.apply_change(cs)
        wb.create_snapshot(self.snap_path)
        wb.close()

        data = json.loads(Path(self.snap_path).read_text(encoding="utf-8"))
        data["plan_hash"] = "0" * 64  # 篡改计划摘要
        Path(self.snap_path).write_text(json.dumps(data), encoding="utf-8")

        wb2 = self._wb()
        wb2.load(SAMPLES / "rules.json")
        with self.assertRaises(ReplayError) as ctx:
            wb2.replay(self.snap_path, baseline_rules=baseline)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("basis", ctx.exception.details)
        wb2.close()

    def test_replay_detects_invalidation_sequence_drift(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        baseline = list(wb.rules)
        cs = ChangeSet.from_dict(json.loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        wb.apply_change(cs)
        wb.create_snapshot(self.snap_path)
        wb.close()

        data = json.loads(Path(self.snap_path).read_text(encoding="utf-8"))
        data["invalidation_sequence"] = ["R1_BASE"]  # 截断传播集合
        Path(self.snap_path).write_text(json.dumps(data), encoding="utf-8")

        wb2 = self._wb()
        wb2.load(SAMPLES / "rules.json")
        with self.assertRaises(ReplayError) as ctx:
            wb2.replay(self.snap_path, baseline_rules=baseline)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.details["field"],
                         "invalidation_sequence")
        wb2.close()


if __name__ == "__main__":
    unittest.main()
