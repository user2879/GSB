"""端到端: 127.0.0.1 回环对端 + 账本推进 + 快照恢复 + 重放校验。"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs.config import PeerScript, ScriptEntry, SessionConfig
from ccobs.errors import E_SCRIPT
from ccobs.session import build_ledger, run_session
from ccobs.snapshot import SnapshotStore
from ccobs.replay import replay_verify


def make_config(port):
    return SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                         scale_factor=3, session_budget=64, batch_size=4,
                         window_batches=2, dup_ack_limit=3, peer_port=port)


def make_script(actions):
    entries = []
    for i, action in enumerate(actions):
        delay = {"ack": 10, "delay": 25, "dup_ack": 2,
                 "reorder": 5, "loss": 0, "gap": 5}[action]
        kw = {"skip": 2} if action == "gap" else {}
        entries.append(ScriptEntry(index=i, event_seq=1000 + i,
                                   action=action, delay_ms=delay, **kw))
    import hashlib, json
    digest = hashlib.sha256(json.dumps(
        [e.__dict__ for e in entries], sort_keys=True).encode()).hexdigest()
    return PeerScript(entries=tuple(entries), digest=digest)


ACTIONS = (["ack", "ack", "delay", "ack",
            "dup_ack", "dup_ack", "dup_ack", "ack",
            "loss", "ack", "reorder", "ack",
            "ack", "ack", "ack", "ack"])


class IntegrationTest(unittest.TestCase):
    def test_loopback_session_deterministic(self):
        config = make_config(0)
        script = make_script(ACTIONS)
        events_a = run_session(config, script)
        events_b = run_session(config, script)
        self.assertEqual([e.record() for e in events_a],
                         [e.record() for e in events_b])
        ledger_a = build_ledger(config, 1, events_a)
        ledger_b = build_ledger(config, 1, events_b)
        self.assertEqual([b.digest for b in ledger_a.batches],
                         [b.digest for b in ledger_b.batches])
        for i in range(len(ledger_a.sealed_windows)):
            self.assertEqual(ledger_a.sealed_bytes(i),
                             ledger_b.sealed_bytes(i))
        self.assertEqual(len(ledger_a.batches), 4)
        self.assertEqual(len(ledger_a.sealed_windows), 2)

    def test_reorder_reply_arrives_out_of_order(self):
        config = make_config(0)
        script = make_script(ACTIONS)
        events = run_session(config, script)
        reorder = next(e for e in events if e.event_seq == 1010)
        after = next(e for e in events if e.event_seq == 1011)
        self.assertGreater(reorder.arrival_order, after.arrival_order)

    def test_gap_injection_raises_e_script(self):
        actions = list(ACTIONS)
        actions[10] = "gap"
        config = make_config(0)
        script = make_script(actions)
        events = run_session(config, script)
        ledger = build_ledger(config, 1, events)
        gaps = [a for a in ledger.anomalies if a.code == E_SCRIPT]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].details["gap_start"], 1010)
        self.assertEqual(gaps[0].details["missing"], 2)

    def test_full_cycle_snapshot_recover_replay(self):
        tmp = tempfile.mkdtemp()
        db = os.path.join(tmp, "e2e.db")
        config = make_config(0)
        script = make_script(ACTIONS)
        events = run_session(config, script)
        ledger = build_ledger(config, 1, events)
        store = SnapshotStore(db)
        store.save_session(ledger, events, True, script_digest=script.digest)
        store.close()

        store = SnapshotStore(db)
        result = store.recover(config, script)
        self.assertTrue(result.exit_report_available)
        replay = replay_verify(config, 1, store.load_events(),
                               store.load_trace())
        self.assertTrue(replay.ok, str(replay))
        store.close()


if __name__ == "__main__":
    unittest.main()
