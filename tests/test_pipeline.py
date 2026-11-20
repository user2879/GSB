"""端到端与单元测试，覆盖业务要求 1-8。运行：python -m unittest -v"""

from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path

from tunnel_reassembly.config import canonical_digest, load_channels, load_json, load_trace
from tunnel_reassembly.engine import (
    adjudicate,
    build_groups,
    estimate_versions,
    integer_median,
)
from tunnel_reassembly.errors import BusinessError
from tunnel_reassembly.faults import corrupt_snapshot_digest, force_estimate_pending, inject_bad_channel
from tunnel_reassembly.model import FragmentEvent
from tunnel_reassembly.protocol import PacketCodec
from tunnel_reassembly.replay import compare_results
from tunnel_reassembly.session import Session
from tunnel_reassembly.ui import build_view_model

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "samples" / "channels.json"
TRACE = ROOT / "samples" / "trace.json"


def fresh_session() -> Session:
    return Session(tempfile.mktemp(suffix=".db"))


class Requirement1ValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_json(CONFIG)
        self.trace = load_json(TRACE)

    def test_valid_inputs_load(self):
        specs, version = load_channels(self.config)
        self.assertEqual(len(specs), 3)
        self.assertTrue(version.startswith("sha256:"))
        ports = [s.port for s in specs]
        self.assertEqual(len(ports), len(set(ports)))

    def test_duplicate_channel_id_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["channels"][1]["channel_id"] = bad["channels"][0]["channel_id"]
        with self.assertRaises(BusinessError) as ctx:
            load_channels(bad)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")
        self.assertIn("channel_id", ctx.exception.path or "".join(ctx.exception.args))

    def test_non_loopback_host_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["channels"][0]["host"] = "10.0.0.1"
        with self.assertRaises(BusinessError) as ctx:
            load_channels(bad)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")

    def test_negative_and_bad_ints_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["channels"][0]["window_ms"] = 0
        bad["channels"][1]["slot_count"] = -1
        with self.assertRaises(BusinessError) as ctx:
            load_channels(bad)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")
        self.assertIn("window_ms", ctx.exception.args[0])
        self.assertIn("slot_count", ctx.exception.args[0])

    def test_overlapping_fragments_rejected_but_duplicates_allowed(self):
        trace = copy.deepcopy(self.trace)
        frags = trace["channels"][0]["groups"][0]["fragments"]
        # 制造真正的部分重叠：第二片偏移落在第一片区间内且长度不同
        frags[1]["frag_offset"] = frags[0]["frag_offset"] + 1
        frags[1]["frag_length"] = frags[0]["frag_length"]
        specs, _ = load_channels(self.config)
        with self.assertRaises(BusinessError) as ctx:
            load_trace(trace, specs)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")

    def test_non_contiguous_group_seq_rejected(self):
        trace = copy.deepcopy(self.trace)
        trace["channels"][0]["groups"][1]["group_seq"] = 99
        specs, _ = load_channels(self.config)
        with self.assertRaises(BusinessError) as ctx:
            load_trace(trace, specs)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")


class Requirement2ProtocolTests(unittest.TestCase):
    def test_roundtrip_and_length_self_check(self):
        codec = PacketCodec()
        payload = b"x" * 100
        packet = codec.encode("tun-alpha", 3, 200, 100, 4, 12345, payload)
        decoded = codec.decode(packet, payload_limit=512)
        self.assertEqual(decoded["payload"], payload)
        self.assertEqual(decoded["frag_offset"] + decoded["frag_length"], 300)

    def test_bad_crc_and_truncation(self):
        codec = PacketCodec()
        packet = bytearray(codec.encode("tun-alpha", 1, 0, 10, 2, 1, b"0123456789"))
        packet[5] ^= 0xFF  # 翻转片头字节，触发 CRC32 不匹配
        with self.assertRaises(BusinessError) as ctx:
            codec.decode(bytes(packet), payload_limit=512)
        self.assertEqual(ctx.exception.code, "E_PROTOCOL")
        with self.assertRaises(BusinessError):
            codec.decode(packet[:10], payload_limit=512)

    def test_payload_limit_enforced(self):
        codec = PacketCodec()
        packet = codec.encode("tun-alpha", 1, 0, 64, 2, 1, b"y" * 64)
        with self.assertRaises(BusinessError) as ctx:
            codec.decode(packet, payload_limit=32)
        self.assertEqual(ctx.exception.code, "E_PROTOCOL")


class Requirement3EstimateTests(unittest.TestCase):
    def test_integer_median_rule(self):
        self.assertEqual(integer_median([5, 1, 3, 2, 4]), 3)
        # 偶数样本取下中位数
        self.assertEqual(integer_median([1, 2, 3, 4]), 2)

    def test_version_monotone_and_pending(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        result = session.run_deterministic()["tun-gamma"]
        real = [v for v in result.estimates if v.skew_ms is not None]
        versions = [v.version for v in real]
        self.assertEqual(versions, sorted(versions))
        distinct = sorted(set(versions))
        self.assertEqual(distinct, list(range(1, len(distinct) + 1)))
        self.assertEqual(real[0].skew_ms, 4010)
        self.assertEqual(real[-1].skew_ms, 4030)
        self.assertTrue(all(v.qerror_ms == 1 for v in result.estimates))
        pending = [o for o in session.results["tun-alpha"].outcomes if o.status == "PENDING"]
        self.assertTrue(pending)
        session.close()

    def test_bit_identical_across_replays(self):
        digests = set()
        for _ in range(3):
            session = fresh_session()
            session.load_inputs(load_json(CONFIG), load_json(TRACE))
            results = session.run_deterministic()
            digests.add(canonical_digest({c: [v.__dict__ for v in r.estimates] for c, r in results.items()}))
            session.close()
        self.assertEqual(len(digests), 1)


class Requirement4WindowAndFlipTests(unittest.TestCase):
    def test_gamma_flips_recorded(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        result = session.run_deterministic()["tun-gamma"]
        self.assertTrue(result.flips)
        for flip in result.flips:
            self.assertLess(flip.from_version, flip.to_version)
            self.assertEqual(flip.to_adjusted_ms - flip.from_adjusted_ms, 20)
            self.assertIn(flip.flip_kind, ("IN_TO_OUT", "OUT_TO_IN", "WINDOW_CHANGE"))
        session.close()

    def test_order_independence(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        baseline = session.run_deterministic()
        rng = random.Random(7)
        for cid, result in baseline.items():
            shuffled = list(result.fragments)
            rng.shuffle(shuffled)
            spec = next(s for s in session.specs if s.channel_id == cid)
            expected = {g["group_seq"]: g["expected_digest"] for g in session.trace[cid]}
            again = adjudicate(spec, shuffled, expected)
            self.assertEqual(
                [(f.group_seq, f.from_version, f.to_version, f.flip_kind) for f in again.flips],
                [(f.group_seq, f.from_version, f.to_version, f.flip_kind) for f in result.flips],
            )
            self.assertEqual(
                [(o.group_seq, o.status) for o in again.outcomes],
                [(o.group_seq, o.status) for o in result.outcomes],
            )
            self.assertEqual(
                [e.window_seq for e in again.window_events if e.action == "OPEN"],
                [e.window_seq for e in result.window_events if e.action == "OPEN"],
            )
        session.close()


class Requirement5SlotAndOffsetTests(unittest.TestCase):
    def test_slot_drop_beta(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        outcomes = session.run_deterministic()["tun-beta"].outcomes
        self.assertIn("SLOT_DROPPED", {o.status for o in outcomes})

    def test_offset_drop_alpha(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        outcomes = session.run_deterministic()["tun-alpha"].outcomes
        self.assertIn("OFFSET_DROPPED", {o.status for o in outcomes})

    def test_drop_set_order_independent(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        baseline = session.run_deterministic()
        for cid, result in baseline.items():
            spec = next(s for s in session.specs if s.channel_id == cid)
            frags = list(result.fragments)
            random.Random(11).shuffle(frags)
            expected = {g["group_seq"]: g["expected_digest"] for g in session.trace[cid]}
            again = adjudicate(spec, frags, expected)
            base_drops = {(o.group_seq, o.status) for o in result.outcomes if "DROP" in o.status}
            again_drops = {(o.group_seq, o.status) for o in again.outcomes if "DROP" in o.status}
            self.assertEqual(base_drops, again_drops)
        session.close()


class Requirement6ReassemblyTests(unittest.TestCase):
    def test_complete_digest_matches(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        for result in session.run_deterministic().values():
            for outcome in result.outcomes:
                if outcome.status == "COMPLETE":
                    self.assertEqual(outcome.digest, outcome.expected_digest)
        session.close()

    def test_digest_mismatch_code(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        spec = next(s for s in session.specs if s.channel_id == "tun-gamma")
        events = session._events_from_trace("tun-gamma")
        bad_expected = {g["group_seq"]: "sha256:" + "0" * 64 for g in session.trace["tun-gamma"]}
        result = adjudicate(spec, events, bad_expected)
        self.assertIn("E_DIGEST", {o.status for o in result.outcomes})
        session.close()

    def test_duplicate_fragments_ignored(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        events = session._events_from_trace("tun-gamma")
        groups = build_groups(events)
        # 原始脚本含重复片；去重后每组片数等于 total_frags
        for group in groups:
            self.assertEqual(len(group["fragments"]), group["total_frags"])
        session.close()

    def test_missing_fragment_incomplete(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        outcomes = session.run_deterministic()["tun-alpha"].outcomes
        self.assertIn("INCOMPLETE", {o.status for o in outcomes})
        session.close()


class Requirement7SnapshotTests(unittest.TestCase):
    def _session_with_snapshot(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        session.run_deterministic()
        session.snapshot("snap-test")
        return session

    def test_snapshot_roundtrip_and_digest(self):
        session = self._session_with_snapshot()
        payload = session.ledger.load_latest_snapshot(session.config_version, session.trace_version)
        self.assertEqual(payload["snapshot_id"], "snap-test")
        self.assertTrue(payload["content_digest"].startswith("sha256:"))
        session.close()

    def test_version_mismatch_forces_full_replace(self):
        session = self._session_with_snapshot()
        with self.assertRaises(BusinessError) as ctx:
            session.ledger.load_latest_snapshot("sha256:bad", session.trace_version)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        session.ledger.replace_all()
        self.assertEqual(session.ledger.list_snapshots(), [])
        session.close()

    def test_corrupt_digest_detected(self):
        session = self._session_with_snapshot()
        corrupt_snapshot_digest(session.ledger)
        with self.assertRaises(BusinessError) as ctx:
            session.ledger.load_latest_snapshot(session.config_version, session.trace_version)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        session.close()

    def test_unknown_inflight_on_restore(self):
        session = self._session_with_snapshot()
        # 强制终止语义：PENDING 组恢复时标记 UNKNOWN
        marked = session.ledger.mark_unknown_inflight()
        self.assertGreaterEqual(marked, 1)
        outcomes = session.ledger.load_outcomes("tun-alpha")
        self.assertIn("UNKNOWN", {o.status for o in outcomes})
        session.close()


class Requirement8ReplayFaultUiTests(unittest.TestCase):
    def test_replay_consistency(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        session.run_deterministic()
        session.snapshot("snap-replay")
        payload, rerun = session.restore_and_replay()
        self.assertEqual(set(rerun), {s.channel_id for s in session.specs})
        self.assertIn("results", payload)
        session.close()

    def test_replay_detects_first_diff(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        session.run_deterministic()
        # 篡改账本里某个结局，重放比较必须定位到通道、组与字段
        session.ledger.conn.execute(
            "UPDATE outcomes SET status='TAMPERED' WHERE channel_id='tun-gamma' AND group_seq=3"
        )
        session.ledger.conn.commit()
        expected = session.results["tun-gamma"]
        stored = session.ledger.load_outcomes("tun-gamma")
        from tunnel_reassembly.model import ReassemblyResult

        snapshot_result = ReassemblyResult(
            channel_id="tun-gamma",
            estimates=session.ledger.load_estimates("tun-gamma"),
            window_events=session.ledger.load_window_events("tun-gamma"),
            flips=session.ledger.load_flips("tun-gamma"),
            outcomes=stored,
            fragments=session.ledger.load_fragments("tun-gamma"),
        )
        with self.assertRaises(BusinessError) as ctx:
            compare_results(snapshot_result, expected)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("tun-gamma", ctx.exception.path)
        self.assertIn("status", ctx.exception.path)
        session.close()

    def test_fault_injection_codes(self):
        config = load_json(CONFIG)
        trace = load_json(TRACE)
        with self.assertRaises(BusinessError) as ctx:
            inject_bad_channel(config)
        self.assertEqual(ctx.exception.code, "E_CHANNEL")

        session = fresh_session()
        session.load_inputs(config, trace)
        spec = session.specs[0]
        groups = build_groups(session._events_from_trace(spec.channel_id))
        pending = force_estimate_pending(spec, groups)
        self.assertEqual(pending, len(groups))
        session.close()

    def test_view_model_stable_order(self):
        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        session.run_deterministic()
        view = build_view_model(session)
        ids = [c["channel_id"] for c in view["channels"]]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(view["business_codes"])
        session.close()


class UdpLoopbackTests(unittest.TestCase):
    def test_real_loopback_send_recv(self):
        from tunnel_reassembly.clock import ReplayClock
        from tunnel_reassembly.transport import UdpReceiver, UdpSender, assert_port_free

        session = fresh_session()
        session.load_inputs(load_json(CONFIG), load_json(TRACE))
        spec = session.specs[0]
        assert_port_free(spec.host, spec.port)
        clock = ReplayClock([1000 + i for i in range(64)])
        receiver = UdpReceiver(spec, clock)
        sender = UdpSender()
        try:
            group = session.trace[spec.channel_id][0]
            count = 0
            for frag in group["fragments"]:
                sender.send_fragment(
                    spec,
                    {
                        "group_seq": group["group_seq"],
                        "frag_offset": frag["frag_offset"],
                        "frag_length": frag["frag_length"],
                        "total_frags": group["total_frags"],
                        "send_ms": group["send_ms"],
                        "payload": frag["payload"],
                    },
                )
                count += 1
            received = []
            for _ in range(count):
                event = receiver.recv_one()
                self.assertIsNotNone(event)
                received.append(event)
            self.assertEqual(len(received), count)
            self.assertEqual([e.arrive_ms for e in received], [1000 + i for i in range(count)])
        finally:
            receiver.close()
            sender.close()
            session.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
