from __future__ import annotations

import unittest

from txdiag.canonical import canonical_text
from txdiag.engine import execute
from txdiag.gui_model import build_gui_model
from txdiag.model import RuntimeConfig
from txdiag.spec import load_spec

from .helpers import sample


class GuiModelTests(unittest.TestCase):
    def test_model_is_byte_stable_and_sorted(self) -> None:
        spec = load_spec(sample("deadlock"))
        cfg = RuntimeConfig(max_running=4, wait_timeout_ms=1000, wait_budget_ms=5000)
        result = execute(spec, cfg, "row")
        first = build_gui_model(spec, result)
        second = build_gui_model(spec, execute(spec, cfg, "row"))
        self.assertEqual(canonical_text(first), canonical_text(second))
        self.assertEqual([tx["txid"] for tx in first["transactions"]], ["TX1", "TX2", "TX3"])
        self.assertEqual([row["row"] for row in first["row_nodes"]],
                         ["row:A", "row:B", "row:C"])
        self.assertEqual(first["deadlocks"][0]["victim"], "TX2")
        event_keys = [(event["round"], event["event_seq"]) for event in first["events"]]
        self.assertEqual(event_keys, sorted(event_keys))


if __name__ == "__main__":
    unittest.main()

