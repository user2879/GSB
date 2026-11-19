from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from h2mux.declaration import load_declaration, load_script
from h2mux.demo import run_demo
from h2mux.session import H2Session
from h2mux.util import digest_value
from h2mux.viewmodel import build_viewmodel


ROOT = Path(__file__).resolve().parents[1]


class ViewModelAndDemoTests(unittest.TestCase):
    def test_refresh_is_read_only_and_stable(self) -> None:
        declaration = load_declaration(ROOT / "samples" / "session.json")
        script = load_script(ROOT / "samples" / "frames.json", declaration)
        session = H2Session(declaration, script).run()
        before = digest_value(session.ledger.to_dict())
        first = build_viewmodel(session)
        second = build_viewmodel(session)
        after = digest_value(session.ledger.to_dict())
        self.assertEqual(before, after)
        self.assertEqual(first, second)
        self.assertEqual(first["current_strategy"], "weighted_round_robin")
        self.assertTrue(first["frame_timeline"])
        self.assertTrue(first["state_transitions"])
        self.assertTrue(first["priority_rows"])
        self.assertTrue(first["empty_waits"])
        self.assertIsNotNone(first["frame_sequence_digest"])

    def test_demo_three_scenarios_pass(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            summary = run_demo(Path(directory))
            self.assertTrue(summary["all_checks_passed"])
            self.assertEqual(len(summary["scenario_order"]), 3)
            self.assertEqual(summary["scenario_3"]["replay"]["status"], "MATCH")
            self.assertEqual(summary["tail_truncation"]["attribution"], "script_tail_truncated")


if __name__ == "__main__":
    unittest.main()
