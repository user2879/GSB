import random
import unittest

from netdiag.constants import (CAUSE_OOO, CAUSE_PENDING, CAUSE_RETRANSMIT,
                               CAUSE_TRUE_LOSS, FLAG_ACK, FLAG_RETX)
from netdiag.eventlog import parse_event_log
from netdiag.gaps import analyze_direction
from netdiag.merge import merge_sources
from netdiag.errors import MergeConflictError
from netdiag.wire import decode_buffer, encode_frame

FLOW = dict(src_ip="10.0.0.1", dst_ip="10.0.0.2", src_port=40001,
            dst_port=50001, protocol=6)
DIRECTION = "10.0.0.1:40001->10.0.0.2:50001"


def f(seq, ts, flags=FLAG_ACK):
    return encode_frame(timestamp_us=ts, seq=seq, flags=flags, **FLOW)


def capture(seq_flags_ts):
    return decode_buffer(b"".join(f(seq, ts, fl) for seq, fl, ts in seq_flags_ts))


class GapTests(unittest.TestCase):
    def sample(self):
        # 1..4 正常；跳 11 打开 5..10；7/8 RETX 回填；9/10 迟到回填
        plan = [(1, FLAG_ACK, 0), (2, FLAG_ACK, 10), (3, FLAG_ACK, 20),
                (4, FLAG_ACK, 30), (11, FLAG_ACK, 40),
                (7, FLAG_ACK | FLAG_RETX, 50),
                (8, FLAG_ACK | FLAG_RETX, 60),
                (9, FLAG_ACK, 70), (10, FLAG_ACK, 80),
                (12, FLAG_ACK, 90), (13, FLAG_ACK, 100)]
        return capture(plan)

    def test_three_causes_classified(self):
        cap = self.sample()
        log = parse_event_log({
            "format": "loss-event-list", "version": "v3",
            "events": [{"event_id": 1, "flow": cap.frames[0].flow_key,
                        "link_id": "L1", "window_ms": [0, 1],
                        "loss_count": 4, "retransmitted": True,
                        "ooo_depth": 4, "direction": DIRECTION}]})
        gaps, _ = analyze_direction(cap.frames, log.events)
        labels = {(g.start_seq, g.end_seq): g.cause for g in gaps}
        self.assertEqual(labels[(5, 6)], CAUSE_TRUE_LOSS)
        self.assertEqual(labels[(7, 8)], CAUSE_RETRANSMIT)
        self.assertEqual(labels[(9, 10)], CAUSE_OOO)

    def test_order_independence(self):
        cap = self.sample()
        log = parse_event_log({
            "format": "loss-event-list", "version": "v3",
            "events": [{"event_id": 1, "flow": cap.frames[0].flow_key,
                        "link_id": "L1", "window_ms": [0, 1],
                        "loss_count": 4, "retransmitted": True,
                        "ooo_depth": 4, "direction": DIRECTION}]})
        baseline = [(g.start_seq, g.end_seq, g.cause)
                    for g in analyze_direction(cap.frames, log.events)[0]]
        rng = random.Random(1234)
        for _ in range(10):
            shuffled = cap.frames[:]
            rng.shuffle(shuffled)
            result = [(g.start_seq, g.end_seq, g.cause)
                      for g in analyze_direction(shuffled, log.events)[0]]
            self.assertEqual(result, baseline)

    def test_conflicting_fill_marks_pending(self):
        plan = [(1, FLAG_ACK, 0), (5, FLAG_ACK, 10),
                (2, FLAG_ACK | FLAG_RETX, 20),
                (2, FLAG_ACK, 30)]
        cap = capture(plan)
        log = parse_event_log({
            "format": "loss-event-list", "version": "v3",
            "events": [{"event_id": 1, "flow": cap.frames[0].flow_key,
                        "link_id": "L1", "window_ms": [0, 1],
                        "loss_count": 1, "retransmitted": False,
                        "ooo_depth": 3, "direction": DIRECTION}]})
        gaps, _ = analyze_direction(cap.frames, log.events)
        self.assertTrue(any(g.cause == CAUSE_PENDING for g in gaps))


class MergeTests(unittest.TestCase):
    def test_arbitration_keeps_override(self):
        cap = self_gap_capture()
        log = parse_event_log({
            "format": "loss-event-list", "version": "v2",
            "events": [{"event_id": 9, "flow": cap.frames[0].flow_key,
                        "link_id": "L1", "window_ms": [0, 100],
                        "loss_count": 7, "direction": DIRECTION}],
            "coverage": {cap.frames[0].flow_key: {DIRECTION: [1, 5]}}})
        result = merge_sources(cap, [log])
        obs = result.model.flows[cap.frames[0].flow_key]
        self.assertEqual(obs.arbitration_count, 1)
        override = obs.overrides[0]
        self.assertEqual(override.winner, "wire")
        self.assertEqual(override.original_loss_count, 7)
        self.assertEqual(override.adopted_loss_count, 2)

    def test_coverage_conflict_raises_e_merge(self):
        cap = self_gap_capture()
        log = parse_event_log({
            "format": "loss-event-list", "version": "v2",
            "events": [{"event_id": 9, "flow": cap.frames[0].flow_key,
                        "link_id": "L1", "window_ms": [0, 100],
                        "loss_count": 1, "direction": DIRECTION}],
            "coverage": {cap.frames[0].flow_key: {DIRECTION: [50, 60]}}})
        with self.assertRaises(MergeConflictError) as ctx:
            merge_sources(cap, [log])
        self.assertEqual(ctx.exception.code, "E_MERGE")
        self.assertIn("wire_interval", ctx.exception.context)


def self_gap_capture():
    plan = [(1, FLAG_ACK, 0), (2, FLAG_ACK, 10), (5, FLAG_ACK, 20)]
    return capture(plan)


if __name__ == "__main__":
    unittest.main()
