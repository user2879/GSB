import json
import tempfile
import unittest
from pathlib import Path

from fanout_ledger.config import parse_scene
from fanout_ledger.engine import run_session
from fanout_ledger.errors import ReplayError, SnapshotError
from fanout_ledger.ledger import Ledger
from fanout_ledger.metrics import (SCALE_KBPS, SCALE_RATIO, aggregate,
                                   canonical_judgment, judge_slo,
                                   nearest_rank_p95)
from fanout_ledger.mock_service import start_all, stop_all
from fanout_ledger.recovery import recover, resume_session, verify_replay
from fanout_ledger.snapshot import load_snapshot, verify_compatibility


def scene_payload(port_base=48730, deliveries=1):
    return {
        "scene_version": "metric-v1",
        "nodes": [
            {"node_id": "a", "host": "127.0.0.1", "port": port_base,
             "process_ms": 25, "failure_rate": 4000, "slow_rate": 0,
             "slow_extra_ms": 0, "seed": 0},
            {"node_id": "b", "host": "127.0.0.1", "port": port_base + 1,
             "process_ms": 25, "failure_rate": 0, "slow_rate": 0,
             "slow_extra_ms": 0, "seed": 1},
        ],
        "task": {"task_id": "t", "payload_bytes": 4096, "max_attempts": 3,
                 "attempt_timeout_ms": 200, "backoff_ms": [5, 10],
                 "deliveries_per_node": deliveries},
        "scheduling": {"global_concurrency": 2, "node_concurrency": 1,
                       "session_budget_ms": 10000},
        "slo": {"success_rate_percent": 90, "p95_latency_ms": 500},
    }


class MetricsTests(unittest.TestCase):
    def test_nearest_rank_p95(self):
        self.assertEqual(nearest_rank_p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 10)
        self.assertEqual(nearest_rank_p95([5, 1, 3, 2, 4]), 5)

    def test_fixed_point_scales_and_truncation(self):
        scene = parse_scene(scene_payload())
        # 1 success out of 3 -> floor(1/3 * 10000) = 3333
        from fanout_ledger.ledger import AttemptRecord
        # three independent deliveries; exactly one succeeds -> 1/3 fixed point
        records = [
            AttemptRecord("a", i, 0, 0, 0, 0, 10, 10,
                          "success" if i == 0 else "failure",
                          10, 0, 200 if i == 0 else 500, "")
            for i in range(3)
        ]
        metrics = aggregate(records, scene)
        self.assertEqual(metrics["global"]["success_rate_scale"], SCALE_RATIO)
        self.assertEqual(metrics["global"]["success_rate_scaled"], 3333)
        self.assertEqual(metrics["global"]["throughput_scale"], SCALE_KBPS)
        self.assertIn("truncation", metrics["global"]["quantization"]["success_rate"])

    def test_judgment_is_byte_stable(self):
        scene = parse_scene(scene_payload())
        from fanout_ledger.ledger import AttemptRecord
        records = [AttemptRecord("a", 0, 0, 0, 0, 0, 10, 10, "failure",
                                 10, 0, 500, "")]
        metrics = aggregate(records, scene)
        j1 = canonical_judgment(judge_slo(metrics, scene))
        j2 = canonical_judgment(judge_slo(metrics, scene))
        self.assertEqual(j1, j2)
        decoded = json.loads(j1)
        self.assertFalse(decoded["passed"])
        self.assertTrue(any(r["type"] == "success_rate" for r in decoded["reasons"]))

    def test_p95_breach_reason_uses_integer_ms(self):
        payload = scene_payload()
        payload["slo"]["p95_latency_ms"] = 1
        scene = parse_scene(payload)
        from fanout_ledger.ledger import AttemptRecord
        records = [AttemptRecord("a", 0, 0, 0, 0, 0, 10, 10, "success",
                                 10, 0, 200, "")]
        judgment = judge_slo(aggregate(records, scene), scene)
        reason_types = {r["type"] for r in judgment["reasons"]}
        self.assertIn("p95_latency", reason_types)


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, port_base):
        scene = parse_scene(scene_payload(port_base=port_base, deliveries=2))
        handles = start_all(scene.nodes)
        try:
            result = run_session(scene, self.run_dir, session_id="sr-1")
        finally:
            stop_all(handles)
        return scene, result

    def test_snapshot_digest_roundtrip_and_atomic_dir(self):
        scene, result = self._run(48731)
        snapshot = load_snapshot(result.final_snapshot)
        self.assertEqual(snapshot.body["scene_version"], scene.scene_version)
        self.assertEqual(snapshot.body["payload_digest"], scene.payload_digest)
        # no temp files left behind after atomic replacement
        leftovers = list((self.run_dir / "snapshots").glob(".snap-*.tmp"))
        self.assertEqual(leftovers, [])

    def test_tampered_snapshot_raises_e_snapshot(self):
        _, result = self._run(48732)
        envelope = json.loads(result.final_snapshot.read_text(encoding="utf-8"))
        envelope["digest"] = "0" * 64
        result.final_snapshot.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaises(SnapshotError) as ctx:
            load_snapshot(result.final_snapshot)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_replay_matches_recorded_outcomes(self):
        scene, result = self._run(48733)
        snapshot = load_snapshot(result.final_snapshot)
        handles = start_all(scene.nodes)
        try:
            outcome = verify_replay(scene, snapshot)
        finally:
            stop_all(handles)
        self.assertGreater(outcome["matched_attempts"], 0)
        self.assertEqual(outcome["replay_success_rate_scaled"],
                         snapshot.body["metrics"]["global"]["success_rate_scaled"])
        self.assertLessEqual(outcome["p95_jitter_ms"],
                                     outcome["p95_tolerance_ms"])

    def test_non_reproducible_mock_config_rejected(self):
        scene, result = self._run(48734)
        snapshot = load_snapshot(result.final_snapshot)
        changed = parse_scene(scene_payload(port_base=48734, deliveries=2))
        object.__setattr__(changed.nodes[0], "failure_rate", 9999)
        mock_config = [
            {"node_id": n.node_id, "failure_rate": n.failure_rate,
             "slow_rate": n.slow_rate, "process_ms": n.process_ms,
             "slow_extra_ms": n.slow_extra_ms, "seed": n.seed}
            for n in changed.nodes
        ]
        with self.assertRaises(SnapshotError) as ctx:
            verify_compatibility(snapshot, changed.scene_version,
                                 changed.payload_digest, mock_config)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        items = [p["item"] for p in ctx.exception.details["problems"]]
        self.assertTrue(any("mock_config" in item for item in items))

    def test_replay_divergence_locates_first_attempt(self):
        scene, result = self._run(48735)
        snapshot = load_snapshot(result.final_snapshot)
        # corrupt one recorded outcome in the trajectory while keeping digest
        # verification bypassed: build an in-memory snapshot object directly
        body = json.loads(json.dumps(snapshot.body))
        body["trajectory"][0]["outcome"] = (
            "failure" if body["trajectory"][0]["outcome"] == "success" else "success")
        from fanout_ledger.snapshot import Snapshot
        tampered = Snapshot(body=body, digest="ignored")
        handles = start_all(scene.nodes)
        try:
            with self.assertRaises(ReplayError) as ctx:
                verify_replay(scene, tampered)
        finally:
            stop_all(handles)
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.details["field"], "outcome")
        self.assertIn("node_id", ctx.exception.details)
        self.assertIn("attempt_seq", ctx.exception.details)

    def test_recovery_marks_unknown_and_reruns_missing(self):
        scene = parse_scene(scene_payload(port_base=48736, deliveries=2))
        handles = start_all(scene.nodes)
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "sr-2")
        try:
            from fanout_ledger.scheduler import FanoutScheduler
            scheduler = FanoutScheduler(
                scene, ledger,
                snapshot_dir=str(self.run_dir / "snapshots"),
                only_jobs=[("a", 0), ("b", 0)])  # simulate partial committed state
            scheduler.run()
            snapshot = scheduler.snapshot("running")
            missing = recover(scene, ledger, snapshot)
        finally:
            ledger.close()
            stop_all(handles)
        self.assertEqual(sorted(missing), [("a", 1), ("b", 1)])
        # reopen and resume the missing half
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "sr-2")
        handles = start_all(scene.nodes)
        try:
            resume = resume_session(scene, ledger, snapshot,
                                    self.run_dir / "snapshots")
            events = ledger.fetch_recovery_events()
            deliveries = ledger.fetch_deliveries()
        finally:
            ledger.close()
            stop_all(handles)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(deliveries), 4)
        self.assertEqual(resume["rerun_jobs"], sorted(missing))


if __name__ == "__main__":
    unittest.main()
