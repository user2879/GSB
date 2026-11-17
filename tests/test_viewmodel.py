import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from chaoskit.spec import load_spec
from chaoskit.timeline import build_timeline
from chaoskit.ledger import Ledger
from chaoskit.viewmodel import build_view_model, render_text_report

ROOT = Path(__file__).resolve().parent.parent


class ViewModelTest(unittest.TestCase):
    def test_view_model_and_text_report(self):
        spec = load_spec(str(ROOT / "examples" / "sample_experiment.json"))
        entries = build_timeline(spec)
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.sqlite3")
            ledger.initialize_experiment(spec, entries)
            ledger.mark_running(0)
            ledger.mark_unknown_in_flight()
            ledger.close()

            conn = sqlite3.connect(str(Path(tmp) / "ledger.sqlite3"))
            conn.row_factory = sqlite3.Row
            try:
                view = build_view_model(conn)
            finally:
                conn.close()

            self.assertEqual(view["ledger_format_version"], 1)
            self.assertEqual(view["seed"], spec.seed)
            self.assertEqual(view["unknown_indexes"], [0])
            self.assertEqual([row.index for row in view["timeline_rows"]],
                             list(range(len(entries))))
            report = render_text_report(view)
            self.assertIn("Timeline (stable order)", report)
            self.assertIn("Whole-snapshot replacements", report)


if __name__ == "__main__":
    unittest.main()
