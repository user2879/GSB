import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs.config import SessionConfig
from ccobs.errors import E_SNAPSHOT, BizError
from ccobs.ledger import KIND_ACK, Event, Ledger
from ccobs.snapshot import SnapshotStore, diff_windows

CFG = SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                    scale_factor=3, session_budget=64, batch_size=4,
                    window_batches=2, dup_ack_limit=3)


class FakeScript:
    def __init__(self, n, digest="s" * 64):
        self.entries = list(range(n))
        self.digest = digest


def make_events(n, start=1000):
    return [Event(event_seq=start + i, kind=KIND_ACK, rtt_ms=10,
                  clock_ms=(i + 1) * 10, arrival_order=i)
            for i in range(n)]


def run_ledger(events, version=1, config=CFG):
    ledger = Ledger(config, version)
    for event in events:
        ledger.ingest_event(event)
    return ledger


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "snap.db")

    def test_roundtrip_recover(self):
        store = SnapshotStore(self.db)
        events = make_events(16)
        ledger = run_ledger(events)
        store.save_session(ledger, events, True, script_digest="s" * 64)
        store.close()

        store = SnapshotStore(self.db)
        result = store.recover(CFG, FakeScript(16))
        self.assertTrue(result.exit_report_available)
        self.assertEqual(result.recovered_events, 16)
        self.assertEqual([b.record() for b in result.ledger.batches],
                         [b.record() for b in ledger.batches])
        store.close()

    def test_tamper_detected_as_e_snapshot(self):
        store = SnapshotStore(self.db)
        events = make_events(16)
        ledger = run_ledger(events)
        store.save_session(ledger, events, True, script_digest="s" * 64)
        # 直接篡改数据库中的观测值
        store.conn.execute("UPDATE items SET value = value + 1")
        store.conn.commit()
        with self.assertRaises(BizError) as ctx:
            store.recover(CFG, FakeScript(16))
        self.assertEqual(ctx.exception.code, E_SNAPSHOT)
        # 整体替换: 数据已清空
        self.assertEqual(store.load_meta(), {})
        store.close()

    def test_forced_kill_recovery_marks_unknown_batches(self):
        store = SnapshotStore(self.db)
        events = make_events(16)
        # 模拟强制终止: 只有前 2 个完整批次落盘, 会话未完成
        partial = run_ledger(events[:8])
        store.save_session(partial, events[:8], False, script_digest="s" * 64)
        store.close()

        store = SnapshotStore(self.db)
        result = store.recover(CFG, FakeScript(16))
        self.assertEqual(result.recovered_events, 8)
        self.assertEqual(result.unknown_batches, [2, 3])
        self.assertFalse(result.exit_report_available)
        store.close()

    def test_version_bump_and_stale_replacement(self):
        store = SnapshotStore(self.db)
        events = make_events(16)
        ledger_v1 = run_ledger(events, version=1)
        store.save_session(ledger_v1, events, True, script_digest="a" * 64)
        old_windows = store.load_windows()

        # 脚本演进: 摘要变化 -> 版本单调递增
        version, changed, diff = store.negotiate_version(
            CFG, FakeScript(16, digest="b" * 64))
        self.assertTrue(changed)
        self.assertEqual(version, 2)
        self.assertIn("script_digest", diff)

        ledger_v2 = run_ledger(events, version=2)
        report = diff_windows(old_windows, ledger_v2.all_windows())
        self.assertEqual([e["window"] for e in report["version_changed"]],
                         [0, 1])
        self.assertEqual(report["added"], [])
        self.assertEqual(report["removed"], [])
        store.close()

    def test_same_content_keeps_version(self):
        store = SnapshotStore(self.db)
        events = make_events(16)
        ledger = run_ledger(events)
        store.save_session(ledger, events, True, script_digest="a" * 64)
        version, changed, _ = store.negotiate_version(
            CFG, FakeScript(16, digest="a" * 64))
        self.assertFalse(changed)
        self.assertEqual(version, 1)
        store.close()


if __name__ == "__main__":
    unittest.main()
