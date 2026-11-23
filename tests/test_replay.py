import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs.config import SessionConfig
from ccobs.errors import E_REPLAY
from ccobs.ledger import KIND_ACK, KIND_DUP_ACK, Event, Ledger
from ccobs.replay import replay_verify
from ccobs.snapshot import SnapshotStore

CFG = SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                    scale_factor=3, session_budget=64, batch_size=4,
                    window_batches=2, dup_ack_limit=10)


def make_events(kinds):
    events = []
    clock = 0
    for i, kind in enumerate(kinds):
        clock += 10 if kind == KIND_ACK else 2
        events.append(Event(event_seq=1000 + i, kind=kind,
                            rtt_ms=10 if kind == KIND_ACK else 0,
                            clock_ms=clock, arrival_order=i))
    return events


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "snap.db")
        self.kinds = [KIND_ACK] * 6 + [KIND_DUP_ACK] * 2 + [KIND_ACK] * 8
        self.events = make_events(self.kinds)
        ledger = Ledger(CFG, 1)
        for event in self.events:
            ledger.ingest_event(event)
        store = SnapshotStore(self.db)
        store.save_session(ledger, self.events, True, script_digest="s" * 64)
        store.close()

    def test_replay_ok(self):
        store = SnapshotStore(self.db)
        result = replay_verify(CFG, 1, store.load_events(), store.load_trace())
        self.assertTrue(result.ok, str(result))
        self.assertGreater(result.compared, 0)
        store.close()

    def test_replay_mismatch_locates_first_event_and_item(self):
        store = SnapshotStore(self.db)
        # 篡改轨迹中某一项的值
        store.conn.execute(
            "UPDATE trace SET value = value + 1 WHERE rowid = "
            "(SELECT rowid FROM trace ORDER BY rowid LIMIT 1 OFFSET 42)")
        store.conn.commit()
        # 绕过整体摘要直接比对轨迹(摘要校验由 E_SNAPSHOT 负责)
        trace = store.load_trace()
        result = replay_verify(CFG, 1, store.load_events(), trace)
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, E_REPLAY)
        self.assertIn("event_seq=", str(result.error))
        self.assertIn("item=", str(result.error))
        store.close()


if __name__ == "__main__":
    unittest.main()
