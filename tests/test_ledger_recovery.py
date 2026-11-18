import tempfile
import unittest
from pathlib import Path

from qos.changeset import ChangeSet
from qos.workbench import Workbench

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class LedgerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "ledger.db")
        self.log = str(Path(self.tmp.name) / "qos.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def _wb(self):
        return Workbench(self.db, self.log)

    def test_committed_plan_is_recoverable(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        expected_version = wb.plan.plan_version
        wb.close()

        wb2 = self._wb()
        report = wb2.recover()
        self.assertEqual(report["recovered_plan_version"], expected_version)
        self.assertEqual(report["unknown_rules"], [])
        wb2.close()

    def test_forced_termination_marks_rules_unknown(self):
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        op_id = wb.simulate_crash_during_compile()
        wb.close()  # OPEN 操作故意未提交

        wb2 = self._wb()
        report = wb2.recover()
        self.assertIn(op_id, [op["op_id"] for op in report["open_operations"]])
        self.assertTrue(report["unknown_rules"])
        self.assertIn("R1_BASE", report["unknown_rules"])
        self.assertIsNotNone(report["ledger_version"])
        # 恢复报告不包含“退出报告”承诺
        self.assertIn("note", report)
        resolved = wb2.resolve_unknown()
        self.assertGreaterEqual(resolved, 1)
        wb2.close()

    def test_invalidation_and_observation_persisted(self):
        import json
        import sqlite3
        wb = self._wb()
        wb.load(SAMPLES / "rules.json")
        cs = ChangeSet.from_dict(json.loads(
            (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
        wb.apply_change(cs)
        wb.simulate()
        wb.close()

        conn = sqlite3.connect(self.db)
        inv = conn.execute("SELECT COUNT(*) AS c FROM invalidations").fetchone()
        obs = conn.execute("SELECT COUNT(*) AS c FROM observations").fetchone()
        plans = conn.execute(
            "SELECT plan_version FROM plans ORDER BY plan_version").fetchall()
        conn.close()
        self.assertEqual(inv[0], 1)
        self.assertEqual(obs[0], 1)
        self.assertEqual([row[0] for row in plans], [1, 2])


if __name__ == "__main__":
    unittest.main()
