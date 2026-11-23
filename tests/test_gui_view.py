import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs.config import SessionConfig
from ccobs.gui import build_view_model
from ccobs.ledger import KIND_ACK, KIND_DUP_ACK, Event, Ledger

CFG = SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                    scale_factor=3, session_budget=64, batch_size=4,
                    window_batches=2, dup_ack_limit=3)


def make_events(kinds):
    return [Event(event_seq=1000 + i, kind=k, rtt_ms=10,
                  clock_ms=(i + 1) * 10, arrival_order=i)
            for i, k in enumerate(kinds)]


class GuiViewTest(unittest.TestCase):
    def test_view_model_stable_order(self):
        kinds = [KIND_ACK] * 12 + [KIND_DUP_ACK] * 4
        ledger = Ledger(CFG, 1)
        for event in make_events(kinds):
            ledger.ingest_event(event)
        model_a = build_view_model(ledger, sealed_stale={0},
                                   snapshots=["s1"])
        model_b = build_view_model(ledger, sealed_stale={0},
                                   snapshots=["s1"])
        self.assertEqual(model_a, model_b)
        self.assertEqual([w["index"] for w in model_a["windows"]],
                         sorted(w["index"] for w in model_a["windows"]))
        for window in model_a["windows"]:
            names = [i["name"] for i in window["items"]]
            self.assertEqual(names, sorted(names))
        self.assertEqual(model_a["stale_windows"], [0])
        # 观测项展示字段齐全: 量化误差上界与受影响标记
        item = model_a["windows"][0]["items"][0]
        for key in ("error_bound", "affected", "halted", "value"):
            self.assertIn(key, item)


if __name__ == "__main__":
    unittest.main()
