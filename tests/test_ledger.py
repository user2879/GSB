import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ccobs import fixedpoint as fp
from ccobs.config import SessionConfig
from ccobs.errors import E_CWND, E_SCRIPT
from ccobs.ledger import (KIND_ACK, KIND_DUP_ACK, KIND_TIMEOUT, Event,
                          Ledger)

CFG = SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                    scale_factor=3, session_budget=64, batch_size=4,
                    window_batches=2, dup_ack_limit=3)


def make_events(kinds, start=1000):
    events = []
    clock = 0
    for i, kind in enumerate(kinds):
        clock += 10 if kind == KIND_ACK else (200 if kind == KIND_TIMEOUT else 2)
        events.append(Event(event_seq=start + i, kind=kind,
                            rtt_ms=10 if kind == KIND_ACK else 0,
                            clock_ms=clock, arrival_order=i))
    return events


def run_ledger(kinds, config=CFG):
    ledger = Ledger(config, 1)
    for event in make_events(kinds, config.initial_seq):
        ledger.ingest_event(event)
    return ledger


class LedgerTest(unittest.TestCase):
    def test_batch_boundaries_deterministic_across_runs(self):
        kinds = [KIND_ACK] * 16
        first = run_ledger(kinds)
        second = run_ledger(kinds)
        self.assertEqual([b.record() for b in first.batches],
                         [b.record() for b in second.batches])
        for i in range(len(first.sealed_windows)):
            self.assertEqual(first.sealed_bytes(i), second.sealed_bytes(i))

    def test_batch_and_window_layout(self):
        ledger = run_ledger([KIND_ACK] * 16)
        self.assertEqual(len(ledger.batches), 4)
        self.assertEqual(ledger.batches[0].event_lo, 1000)
        self.assertEqual(ledger.batches[0].event_hi, 1003)
        self.assertEqual(len(ledger.sealed_windows), 2)
        self.assertEqual(ledger.sealed_windows[0].batch_lo, 0)
        self.assertEqual(ledger.sealed_windows[0].batch_hi, 1)
        self.assertIsNone(ledger.verify_sealed())

    def test_sealed_windows_byte_identical_and_verifiable(self):
        ledger = run_ledger([KIND_ACK] * 16)
        snapshot = ledger.sealed_bytes(0)
        # 继续推进不得改变已封存窗口
        ledger2 = run_ledger([KIND_ACK] * 16)
        self.assertEqual(snapshot, ledger2.sealed_bytes(0))
        # 篡改封存结论必须被逐项校验发现
        ledger2.sealed_windows[0].items[fp.ITEM_CWND].value += 1
        self.assertEqual(ledger2.verify_sealed(), 0)

    def test_only_affected_subset_recomputed(self):
        # 第一批全 ack, 第二批全 dup_ack: SRTT 不依赖 dup_ack, 不应重算
        kinds = [KIND_ACK] * 4 + [KIND_DUP_ACK] * 4
        cfg = SessionConfig(initial_seq=1000, cwnd_max=65535, mss=1460,
                            scale_factor=3, session_budget=64, batch_size=4,
                            window_batches=2, dup_ack_limit=10)
        ledger = run_ledger(kinds, config=cfg)
        window = ledger.sealed_windows[0]
        srtt = window.items[fp.ITEM_SRTT]
        cwnd = window.items[fp.ITEM_CWND]
        self.assertEqual(srtt.recomputes, 1)       # 仅首批重算过一次
        self.assertGreaterEqual(cwnd.recomputes, 2)  # CWND 受 dup_ack 影响
        self.assertEqual(srtt.batch_hi, 0)           # 依赖区间不含第二批

    def test_gap_raises_e_script_and_halts_items(self):
        events = make_events([KIND_ACK] * 16)
        # 制造缺口: 1008 缺失, 直接跳到 1009
        gapped = [e for e in events if e.event_seq != 1008]
        ledger = Ledger(CFG, 1)
        for event in gapped:
            ledger.ingest_event(event)
        codes = [a.code for a in ledger.anomalies]
        self.assertIn(E_SCRIPT, codes)
        gap = next(a for a in ledger.anomalies if a.code == E_SCRIPT)
        self.assertEqual(gap.details["gap_start"], 1008)
        self.assertEqual(gap.details["missing"], 1)
        # 缺口前已封存窗口结论不受影响
        self.assertEqual(len(ledger.sealed_windows), 1)
        self.assertIsNone(ledger.verify_sealed())
        # 缺口后观测项停止推进
        self.assertEqual(ledger.halted_items, set(fp.ALL_ITEMS))

    def test_dup_ack_cluster_raises_e_cwnd(self):
        kinds = [KIND_ACK] * 4 + [KIND_DUP_ACK] * 4  # 窗口内 4 次 > 上限 3
        ledger = run_ledger(kinds)
        cwnd_err = next(a for a in ledger.anomalies if a.code == E_CWND)
        self.assertEqual(cwnd_err.details["over"], 1)
        self.assertEqual(cwnd_err.details["limit"], 3)
        window = ledger.sealed_windows[0]
        # CWND 在超限窗口内停更(依赖区间不含第二批), SRTT 结论不受影响
        self.assertEqual(window.items[fp.ITEM_CWND].batch_hi, 0)
        self.assertIn(fp.ITEM_SRTT, window.items)
        srtt = window.items[fp.ITEM_SRTT]
        self.assertFalse(srtt.halted)

    def test_dup_ack_within_limit_no_error(self):
        kinds = [KIND_ACK] * 4 + [KIND_DUP_ACK] * 3 + [KIND_ACK]
        ledger = run_ledger(kinds)
        self.assertNotIn(E_CWND, [a.code for a in ledger.anomalies])


if __name__ == "__main__":
    unittest.main()
