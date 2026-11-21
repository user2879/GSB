import os
import tempfile
import unittest

from emergency_drill.engine import simulate
from emergency_drill.loader import load_bundle
from emergency_drill.repository import Ledger
from emergency_drill.sample_data import generate_samples
from emergency_drill.viewmodel import DiagnosticViewModel, render_text


class ViewModelTests(unittest.TestCase):
    def test_text_render_is_stable_and_read_only(self):
        tmp = tempfile.TemporaryDirectory()
        generate_samples(tmp.name)
        plan, events = load_bundle(
            os.path.join(tmp.name, "plan_valid.json"),
            os.path.join(tmp.name, "events_valid.json"),
        )
        ledger_path = os.path.join(tmp.name, "drill.sqlite3")
        ledger = Ledger(ledger_path)
        ledger.save_plan(plan, events)
        ledger.append_simulation(plan, simulate(plan, events))
        before = ledger.ledger_digest()
        view = DiagnosticViewModel(ledger, 1, ["recover"], ["snapshot.json"]).build()
        text = render_text(view)
        after = ledger.ledger_digest()
        self.assertEqual(before, after)
        self.assertIn("本轮失效集合: recover", text)
        self.assertIn("注入事件时间轴", text)
        self.assertEqual([row["stage_id"] for row in view["stage_rows"]],
                         plan.stage_ids_topo())
        ledger.close()
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
