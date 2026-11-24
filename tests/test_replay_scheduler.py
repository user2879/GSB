import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slowquery_diag.errors import (  # noqa: E402
    AckError,
    PortError,
    ReplayTimeoutError,
)
from slowquery_diag.replay_server import (  # noqa: E402
    DeterministicReplayService,
    ReplayClient,
)
from slowquery_diag.scheduler import (  # noqa: E402
    LoopbackServiceProcess,
    ReplayJob,
    ReplayScheduler,
)


def _catalog(path, entries):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"catalog_format": 1, "replay_catalog": entries}, handle)


def _entry(cid, source, stall_ms=0):
    return {
        "candidate_id": cid, "source_id": source,
        "est_latency_us": 1234, "est_rows_scanned": 56,
        "stall_ms": stall_ms,
    }


class ReplayServiceTests(unittest.TestCase):
    def test_subprocess_lifecycle_and_ack(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "catalog.json")
            _catalog(catalog_path, [_entry("c1", "s1"), _entry("c2", "s2")])
            process = LoopbackServiceProcess(catalog_path)
            port = process.start()
            try:
                client = ReplayClient(port=port, timeout_ms=2000)
                hello = client.hello()
                self.assertEqual(hello["type"], "hello_ack")
                ack = client.replay("c1", "s1", "SELECT 1")
                self.assertEqual(ack["est_latency_us"], 1234)
                self.assertEqual(ack["est_rows_scanned"], 56)
            finally:
                process.stop()

    def test_port_occupied_raises_e_port(self):
        service = DeterministicReplayService({})
        port = service.start(0)
        try:
            clash = DeterministicReplayService({})
            with self.assertRaises(PortError) as ctx:
                clash.start(port)
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            service.stop()

    def test_real_socket_timeout_raises_e_timeout(self):
        catalog = {
            "stall@src": {
                "candidate_id": "stall", "source_id": "src",
                "est_latency_us": 1, "est_rows_scanned": 1,
                "stall_ms": 800,
            }
        }
        service = DeterministicReplayService(catalog)
        port = service.start(0)
        try:
            with self.assertRaises(ReplayTimeoutError) as ctx:
                ReplayClient(port=port, timeout_ms=200).replay(
                    "stall", "src", "SELECT 1"
                )
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            service.stop()

    def test_missing_ack_field_raises_e_ack(self):
        from slowquery_diag import replay_server as rs
        original = DeterministicReplayService._handle_replay

        def broken(self, conn, request):
            rs._send_frame(conn, {"type": "ack", "candidate_id": "x"})

        DeterministicReplayService._handle_replay = broken
        service = DeterministicReplayService({"x@src": _entry("x", "src")})
        port = service.start(0)
        try:
            with self.assertRaises(AckError) as ctx:
                ReplayClient(port=port).replay("x", "src", "SELECT 1")
            self.assertEqual(ctx.exception.code, "E_ACK")
        finally:
            DeterministicReplayService._handle_replay = original
            service.stop()


class SchedulerQuotaTests(unittest.TestCase):
    def test_global_and_per_source_limits_never_exceeded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "catalog.json")
            sources = ["s1", "s2", "s3"]
            entries = [
                _entry(f"c{i:02d}", sources[i % 3]) for i in range(12)
            ]
            _catalog(catalog_path, entries)
            process = LoopbackServiceProcess(catalog_path)
            port = process.start()
            try:
                scheduler = ReplayScheduler(
                    port=port,
                    global_concurrency=2,
                    per_source_quota={"s1": 1, "s2": 1, "s3": 1},
                    socket_timeout_ms=2000,
                )
                jobs = [
                    ReplayJob(e["candidate_id"], e["source_id"], "sql")
                    for e in entries
                ]
                results = scheduler.run_all(jobs)
                self.assertTrue(all(r.ok for r in results))
                self.assertLessEqual(
                    scheduler.stats.peak_global_inflight, 2
                )
                self.assertTrue(all(
                    peak == 1
                    for peak in scheduler.stats.peak_per_source.values()
                ))
                self.assertGreaterEqual(scheduler.stats.queued_wait_count, 1)
                # Service-side view independently confirms the global cap.
                server_stats = ReplayClient(port=port).stats()
                self.assertLessEqual(server_stats["peak_inflight"], 2)
                self.assertEqual(server_stats["total_requests"], 12)
            finally:
                process.stop()

    def test_wait_timestamps_are_monotonic_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "catalog.json")
            entries = [_entry(f"c{i}", "only") for i in range(4)]
            _catalog(catalog_path, entries)
            process = LoopbackServiceProcess(catalog_path)
            port = process.start()
            try:
                scheduler = ReplayScheduler(
                    port=port, global_concurrency=2,
                    per_source_quota={"only": 1},
                    socket_timeout_ms=2000,
                )
                jobs = [ReplayJob(e["candidate_id"], "only", "sql")
                        for e in entries]
                results = scheduler.run_all(jobs)
                for result in results:
                    job = result.job
                    self.assertGreater(job.finished_at_ms, 0)
                    self.assertLessEqual(job.wait_start_ms, job.wait_end_ms)
                    self.assertLessEqual(job.wait_end_ms, job.started_at_ms)
            finally:
                process.stop()


if __name__ == "__main__":
    unittest.main()
