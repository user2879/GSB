import os
import tempfile
import unittest

from emergency_drill.engine import simulate
from emergency_drill.loader import load_bundle
from emergency_drill.repository import Ledger
from emergency_drill.sample_data import generate_samples


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        generate_samples(self.tmp.name)
        self.plan, self.events = load_bundle(
            os.path.join(self.tmp.name, "plan_valid.json"),
            os.path.join(self.tmp.name, "events_valid.json"),
        )
        self.ledger_path = os.path.join(self.tmp.name, "work", "drill.sqlite3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_simulation_persisted_with_digest_and_skip_traces(self):
        ledger = Ledger(self.ledger_path)
        ledger.save_plan(self.plan, self.events)
        result = simulate(self.plan, self.events)
        ledger.append_simulation(self.plan, result)
        self.assertTrue(ledger.ledger_digest())
        transitions = ledger.transitions(self.plan.version)
        self.assertTrue(any(item["to_state"] == "skipped" for item in transitions))
        trace_rows = ledger.conn.execute("SELECT * FROM skip_traces").fetchall()
        self.assertTrue(trace_rows)
        progress = ledger.progress(self.plan.version)
        self.assertIn(progress["state"], ("succeeded", "skipped"))
        self.assertEqual(progress["cursor_stage"], "recover")
        ledger.close()

        reopened = Ledger(self.ledger_path)
        self.assertEqual(len(reopened.active_outcomes(self.plan.version)), len(self.plan.stages))
        reopened.close()

    def test_stale_marking_excludes_old_version_from_current_view(self):
        ledger = Ledger(self.ledger_path)
        ledger.save_plan(self.plan, self.events)
        first = simulate(self.plan, self.events)
        ledger.append_simulation(self.plan, first)

        plan_v2 = self.plan.with_version(self.plan.stages, 2)
        ledger.save_plan(plan_v2, self.events)
        second = simulate(plan_v2, self.events)
        ledger.append_simulation(plan_v2, second, mark_stale_versions={1})
        stale_rows = ledger.conn.execute(
            "SELECT COUNT(*) AS n FROM outcomes WHERE stale=1"
        ).fetchone()["n"]
        self.assertEqual(stale_rows, len(self.plan.stages))
        self.assertEqual(ledger.stats(2)["total"], len(self.plan.stages))
        ledger.close()


if __name__ == "__main__":
    unittest.main()
