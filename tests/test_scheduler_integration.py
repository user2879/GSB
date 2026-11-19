import tempfile
import threading
import unittest
from pathlib import Path

from fanout_ledger.config import parse_scene
from fanout_ledger.engine import run_session
from fanout_ledger.errors import PortError
from fanout_ledger.ledger import Ledger
from fanout_ledger.mock_service import start_all, stop_all
from fanout_ledger.scheduler import FanoutScheduler, RuntimeCounters


def make_scene(overrides=None, port_base=48722):
    payload = {
        "scene_version": "int-v1",
        "nodes": [
            {"node_id": f"n{i}", "host": "127.0.0.1", "port": port_base + i,
             "process_ms": 15, "failure_rate": 0, "slow_rate": 0,
             "slow_extra_ms": 0, "seed": 1}
            for i in range(4)
        ],
        "task": {"task_id": "t", "payload_bytes": 2048, "max_attempts": 3,
                 "attempt_timeout_ms": 300, "backoff_ms": [5, 10, 15],
                 "deliveries_per_node": 2},
        "scheduling": {"global_concurrency": 2, "node_concurrency": 1,
                       "session_budget_ms": 10000},
        "slo": {"success_rate_percent": 100, "p95_latency_ms": 1000},
    }
    if overrides:
        overrides(payload)
    return parse_scene(payload)


class SchedulerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_fanout_all_success_and_ledger_rows(self):
        scene = make_scene(port_base=48723)
        handles = start_all(scene.nodes)
        try:
            result = run_session(scene, self.run_dir, session_id="s1")
        finally:
            stop_all(handles)
        # 4 nodes x 2 deliveries, each first-attempt success
        self.assertEqual(result.metrics["global"]["samples"], 8)
        self.assertEqual(result.metrics["global"]["success_count"], 8)
        self.assertTrue(result.judgment["passed"])
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s1")
        rows = ledger.fetch_attempts()
        ledger.close()
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row.outcome, "success")
            self.assertGreaterEqual(row.ended_ms, row.started_ms)
            self.assertEqual(row.budget_consumed_ms, row.elapsed_ms)

    def test_dual_concurrency_budget_never_exceeded(self):
        scene = make_scene(port_base=48724)
        # make every request take a measurable amount of time
        for node in scene.nodes:
            object.__setattr__(node, "process_ms", 60)
        handles = start_all(scene.nodes)
        samples: list[tuple[int, int]] = []
        lock = threading.Lock()

        def cb(counters: RuntimeCounters):
            with lock:
                samples.append((counters.running, dict(counters.node_running)))

        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s2")
        try:
            scheduler = FanoutScheduler(scene, ledger, progress_cb=cb)
            scheduler.run()
        finally:
            ledger.close()
            stop_all(handles)
        self.assertTrue(samples, "progress samples expected")
        for running, per_node in samples:
            self.assertLessEqual(running, scene.scheduling.global_concurrency)
            for value in per_node.values():
                self.assertLessEqual(value, scene.scheduling.node_concurrency)

    def test_queue_wait_recorded_with_monotonic_clock(self):
        scene = make_scene(port_base=48725)
        handles = start_all(scene.nodes)
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s3")
        try:
            FanoutScheduler(scene, ledger).run()
            rows = ledger.fetch_attempts()
        finally:
            ledger.close()
            stop_all(handles)
        self.assertTrue(rows)
        for row in rows:
            self.assertGreaterEqual(row.wait_end_ms, row.wait_start_ms)
            self.assertGreaterEqual(row.started_ms, row.wait_end_ms)

    def test_real_socket_timeout_and_retry_then_success(self):
        scene = make_scene(port_base=48726)
        # seed 0 + slow_rate 4000: attempt 0 bucket 0 -> slow; delay exceeds timeout
        slow_node = scene.nodes[0]
        object.__setattr__(slow_node, "slow_rate", 4000)
        object.__setattr__(slow_node, "slow_extra_ms", 300)
        object.__setattr__(slow_node, "seed", 0)
        object.__setattr__(scene.task, "attempt_timeout_ms", 100)
        handles = start_all(scene.nodes)
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s4")
        try:
            FanoutScheduler(scene, ledger).run()
            rows = [r for r in ledger.fetch_attempts() if r.node_id == "n0"]
        finally:
            ledger.close()
            stop_all(handles)
        outcomes = [r.outcome for r in rows]
        self.assertEqual(outcomes[0], "timeout")
        self.assertIn("success", outcomes)
        # timeout must be a real socket timeout note, not a synthesized sleep flag
        self.assertIn("socket timeout", rows[0].note)
        for row in rows:
            self.assertLessEqual(row.elapsed_ms, 100 + 150)

    def test_retry_exhaustion_marks_e_retry_with_waits(self):
        scene = make_scene(port_base=48727)
        bad_node = scene.nodes[0]
        object.__setattr__(bad_node, "failure_rate", 7000)
        object.__setattr__(bad_node, "seed", 2)  # buckets all in failure band
        handles = start_all(scene.nodes)
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s5")
        try:
            FanoutScheduler(scene, ledger).run()
            markers = ledger.fetch_retry_exhaustions()
            deliveries = ledger.fetch_deliveries()
        finally:
            ledger.close()
            stop_all(handles)
        n0_markers = [m for m in markers if m["node_id"] == "n0"]
        self.assertEqual(len(n0_markers), scene.task.deliveries_per_node)
        for marker in n0_markers:
            self.assertEqual(marker["attempts_used"], scene.task.max_attempts)
            self.assertEqual(marker["last_outcome"], "failure")
            # backoff [5,10,15] but only two waits happen across three attempts
            self.assertEqual(marker["total_backoff_ms"], 15)
        n0_deliveries = [d for d in deliveries if d["node_id"] == "n0"]
        self.assertTrue(all(d["final_outcome"] != "success" for d in n0_deliveries))

    def test_unreachable_is_real_connection_error(self):
        scene = make_scene(port_base=48728)
        # start services for all nodes except n0 -> genuine connection refused
        handles = start_all(scene.nodes[1:])
        ledger = Ledger(self.run_dir / "ledger.sqlite3", "s6")
        try:
            FanoutScheduler(scene, ledger).run()
            rows = [r for r in ledger.fetch_attempts() if r.node_id == "n0"]
        finally:
            ledger.close()
            stop_all(handles)
        self.assertTrue(rows)
        self.assertTrue(all(r.outcome == "unreachable" for r in rows))
        # real connect-phase socket error: refused on most stacks, connect
        # timeout on loopback stacks that drop unreachable SYNs
        self.assertTrue(
            "refused" in rows[0].note or "unreachable" in rows[0].note,
            rows[0].note,
        )

    def test_occupied_port_raises_e_port(self):
        import socket
        scene = make_scene(port_base=48729)
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", scene.nodes[0].port))
        holder.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                start_all(scene.nodes)
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            holder.close()


if __name__ == "__main__":
    unittest.main()
