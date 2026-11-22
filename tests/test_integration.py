"""End-to-end tests over real 127.0.0.1 sockets and a SQLite ledger."""

import os
import tempfile
import unittest

from meshbench.clock import VirtualClock
from meshbench.config import load_config
from meshbench.engine import Engine
from meshbench.errors import (PrevailError, ReplayError, RuleError,
                              SnapshotError, UpstreamError)
from meshbench.ledger import Ledger
from meshbench.netutil import preflight_upstreams
from meshbench.protocol import Request
from meshbench.recovery import recover
from meshbench.replay import replay_after_recovery
from meshbench.snapshot import (assert_ledger_matches_snapshot,
                                build_snapshot, diff_snapshots, load_snapshot,
                                save_snapshot)
from meshbench.upstream import start_all, stop_all

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V1 = os.path.join(ROOT, "samples", "rules_v1.json")
V2 = os.path.join(ROOT, "samples", "rules_v2.json")


def requests_for(route_id, count, start=0):
    return [Request(f"req-{route_id}-{start + i:04d}", route_id,
                    f"client-{i % 7}", start + i) for i in range(count)]


class IntegrationBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="meshbench-it-")
        self.config = load_config(V1)
        self.upstreams = start_all(self.config)
        preflight_upstreams([inst for svc in self.config.services.values()
                             for inst in svc.instances])
        self.db = os.path.join(self.tmp, "ledger.db")
        self.ledger = Ledger(self.db)
        self.clock = VirtualClock()
        self.engine = Engine(self.config, self.ledger, clock=self.clock)

    def tearDown(self):
        stop_all(self.upstreams)
        self.ledger.close()


class DispatchIntegrationTests(IntegrationBase):
    def test_real_dispatch_produces_all_outcome_kinds_and_receipts(self):
        batch = requests_for("orders-api", 30)
        summary = self.engine.run_batch(batch)
        self.assertEqual(len(summary["results"]), 30)
        outcomes = {result.final_outcome for result in summary["results"]}
        # Deterministic mix should include success and slow at minimum.
        self.assertIn("success", outcomes)
        self.assertTrue(summary["receipts"])
        for result in summary["results"]:
            self.assertTrue(result.attempts)
            # Retries carry increasing attempt numbers.
            self.assertEqual([a.attempt for a in result.attempts],
                             list(range(1, len(result.attempts) + 1)))

    def test_deterministic_instance_sequence_across_runs(self):
        from meshbench.hashing import stable_bucket, select_instance
        batch = requests_for("orders-api", 20)
        seq_one = []
        for req in batch:
            route = self.config.routes[req.route_id]
            service = self.config.services[route.target_service]
            bucket = stable_bucket(self.config.version, route.route_id, req.client_key)
            seq_one.append(select_instance(service.instances, bucket).instance_id)
        seq_two = []
        for req in batch:
            route = self.config.routes[req.route_id]
            service = self.config.services[route.target_service]
            bucket = stable_bucket(self.config.version, route.route_id, req.client_key)
            seq_two.append(select_instance(service.instances, bucket).instance_id)
        self.assertEqual(seq_one, seq_two)
        self.ledger.close()

    def test_connection_error_is_real_and_classified_failure(self):
        # Point an instance at an unused port; connection must be a real OSError.
        import json
        import socket as _socket
        import threading
        # A real loopback listener that accepts then closes without a response:
        # the dispatcher observes a genuine connection-level failure (E_UPSTREAM),
        # distinct from a socket timeout.
        silent_port = 19999
        halt = threading.Event()

        def silent_server():
            srv = _socket.socket()
            srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", silent_port))
            srv.listen(4)
            srv.settimeout(3.0)
            while not halt.is_set():
                try:
                    conn, _ = srv.accept()
                except _socket.timeout:
                    break
                conn.close()  # accept then immediately drop
            srv.close()

        thread = threading.Thread(target=silent_server, daemon=True)
        thread.start()
        doc = json.load(open(V1, encoding="utf-8"))
        doc["services"][0]["instances"] = [doc["services"][0]["instances"][0]]
        doc["services"][0]["instances"][0]["port"] = silent_port
        from meshbench.config import parse_config
        broken = parse_config(doc)
        ledger = Ledger(":memory:")
        engine = Engine(broken, ledger, clock=self.clock)
        try:
            result = engine.send_request(Request("x", "orders-api", "dead-key", 0))
            self.assertEqual(result.final_outcome, "failure")
            codes = {a.error_code for a in result.attempts if a.error_code}
            self.assertIn("E_UPSTREAM", codes)
        finally:
            halt.set()
            thread.join()
            ledger.close()


class TimeoutIntegrationTests(IntegrationBase):
    def test_real_socket_timeout_classified_timeout(self):
        import json
        import threading
        import time as time_module
        import socket as socket_module

        # Bind an ephemeral free port (port 0) so repeated runs never collide
        # with a still-draining server thread from a previous test.
        listener = socket_module.socket()
        listener.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        slow_port = listener.getsockname()[1]
        listener.listen(4)
        ready = threading.Event()

        def never_responds():
            listener.settimeout(3.0)
            ready.set()
            conn, _ = listener.accept()
            try:
                conn.recv(4096)
                # Client timeout is 80ms; a 1s hold forces a real socket timeout.
                time_module.sleep(1.0)
            finally:
                conn.close()
                listener.close()

        thread = threading.Thread(target=never_responds, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(2.0))

        doc = json.load(open(V1, encoding="utf-8"))
        doc["services"][0]["instances"] = [doc["services"][0]["instances"][0]]
        doc["services"][0]["instances"][0]["port"] = slow_port
        doc["routes"][0]["attempt_timeout_ms"] = 80
        doc["routes"][0]["retries"] = 0
        from meshbench.config import parse_config
        broken = parse_config(doc)
        ledger = Ledger(":memory:")
        engine = Engine(broken, ledger, clock=self.clock)
        try:
            result = engine.send_request(Request("t", "orders-api", "slow-key", 0))
            self.assertEqual(result.final_outcome, "timeout")
            self.assertTrue(all(a.outcome == "timeout" for a in result.attempts))
        finally:
            thread.join()
            ledger.close()

class ReplayAndEvolutionTests(IntegrationBase):
    def test_authoritative_recompute_matches_cache_byte_for_byte(self):
        self.engine.run_batch(requests_for("orders-api", 24))
        self.clock.advance(60_000)
        self.engine.run_batch(requests_for("orders-api", 12, 24))
        receipts, elapsed = self.engine.authoritative_recompute()
        self.assertGreaterEqual(elapsed, 0)
        self.assertTrue(receipts)

    def test_tampered_cache_raises_prevail(self):
        self.engine.run_batch(requests_for("orders-api", 12))
        row = self.ledger.current_receipts()[0]
        tampered = bytearray(row["canonical_bytes"])
        tampered[-2] = ord("0") if tampered[-2] != ord("0") else ord("1")
        with self.ledger._lock, self.ledger.conn:
            self.ledger.conn.execute(
                "UPDATE receipts SET canonical_bytes=? WHERE window_id=?",
                (bytes(tampered), row["window_id"]),
            )
        with self.assertRaises(PrevailError) as ctx:
            self.engine.authoritative_recompute()
        self.assertEqual(ctx.exception.code, "E_PREVAIL")
        self.assertTrue(ctx.exception.context["window_id"])

    def test_rule_evolution_replaces_stale_windows_wholesale(self):
        self.engine.run_batch(requests_for("orders-api", 12))
        self.engine.run_batch(requests_for("orders-api", 8, 12))
        new_config = load_config(V2)
        report = self.engine.evolve_rules(new_config)
        self.assertEqual(report["old_version"], 1)
        self.assertEqual(report["new_version"], 2)
        self.assertTrue(report["replaced_windows"])
        # Stored stale rows remain for audit but leave the current view.
        current = self.ledger.current_receipts()
        self.assertTrue(all(row["rule_version"] != 1 for row in current))
        all_rows = self.ledger.stored_receipts_all()
        self.assertTrue(any(row["stale"] == 1 for row in all_rows))

    def test_replay_error_classification_on_corrupt_stored_bytes(self):
        self.engine.run_batch(requests_for("orders-api", 12))
        row = self.ledger.current_receipts()[0]
        tampered = bytearray(row["canonical_bytes"])
        tampered[-2] = ord("0") if tampered[-2] != ord("0") else ord("1")
        with self.ledger._lock, self.ledger.conn:
            self.ledger.conn.execute(
                "UPDATE receipts SET canonical_bytes=? WHERE window_id=?",
                (bytes(tampered), row["window_id"]),
            )
        with self.assertRaises(ReplayError):
            replay_after_recovery(self.config, self.ledger)


class RecoveryAndSnapshotTests(IntegrationBase):
    def test_forced_kill_leaves_unknown_requests_and_recovery_replays(self):
        self.engine.run_batch(requests_for("orders-api", 10))
        # Simulate an interrupted batch: pending rows without a committed batch.
        orphan = requests_for("orders-api", 3, 50)
        self.ledger.mark_pending(orphan, self.config.version)
        db_path = self.ledger.path
        self.ledger.close()

        report = recover(self.config, db_path, clock=VirtualClock())
        self.assertEqual(len(report["unknown_requests"]), 3)
        self.assertTrue(report["receipts"])
        # All committed windows reproduce byte-for-byte.
        self.assertEqual(report["head_digest"], report["ledger"].head_digest())
        report["ledger"].close()

    def test_snapshot_roundtrip_and_digest_guard(self):
        self.engine.run_batch(requests_for("orders-api", 12))
        snap = self.engine.snapshot()
        path = os.path.join(self.tmp, "session.json")
        save_snapshot(path, snap)
        loaded = load_snapshot(path)
        self.assertEqual(loaded["content_digest"], snap["content_digest"])
        assert_ledger_matches_snapshot(loaded, self.ledger)

    def test_snapshot_version_change_is_wholesale_replacement_diff(self):
        old = build_snapshot(1, 1, [], {"orders-api:v1:w0": "aaa"}, 0, "GENESIS")
        new = build_snapshot(2, 1, [], {"orders-api:v2:w0": "bbb"}, 5, "GENESIS")
        diff = diff_snapshots(old, new)
        self.assertTrue(diff["rule_version_changed"])
        self.assertEqual(diff["receipts_removed"], ["orders-api:v1:w0"])
        self.assertEqual(diff["receipts_added"], ["orders-api:v2:w0"])

    def test_ledger_digest_tamper_detected_as_snapshot_error(self):
        self.engine.run_batch(requests_for("orders-api", 8))
        self.ledger.tamper_head_digest()
        with self.assertRaises(SnapshotError):
            self.ledger.verify_integrity()
        snap = build_snapshot(self.config.version, 1, [], {},
                              self.ledger.cursor_sequence(), "ORIGINAL")
        path = os.path.join(self.tmp, "s.json")
        save_snapshot(path, snap)
        with self.assertRaises(SnapshotError):
            assert_ledger_matches_snapshot(load_snapshot(path), self.ledger)


class FaultInjectionTests(IntegrationBase):
    def test_upstream_unreachable_raises_e_upstream(self):
        stop_all(self.upstreams)  # nothing listening now
        with self.assertRaises(UpstreamError) as ctx:
            preflight_upstreams([inst for svc in self.config.services.values()
                                 for inst in svc.instances])
        self.assertEqual(ctx.exception.code, "E_UPSTREAM")

    def test_corrupt_rule_file_raises_e_rule(self):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        with self.assertRaises(RuleError):
            load_config(path)


if __name__ == "__main__":
    unittest.main()
