from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from edge_shard_diag.demo import free_loopback_ports, write_demo_spec
from edge_shard_diag.errors import Codes, DiagnosticError
from edge_shard_diag.ledger import Ledger
from edge_shard_diag.nodepool import REPO_ROOT, NodePool
from edge_shard_diag.scheduler import Scheduler
from edge_shard_diag.spec import load_spec_bundle
from edge_shard_diag.tcpio import renew_lease_request, request_response
from edge_shard_diag.invalidation import compare_semantic_views


class SchedulerTcpTests(unittest.TestCase):
    def test_real_tcp_scheduler_completes_and_queues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            spec = write_demo_spec(root / "spec", allocate_ports=True)
            bundle = load_spec_bundle(spec)
            ledger = Ledger(root / "ledger.sqlite")
            pool = NodePool(bundle, root / "logs")
            pool.start()
            try:
                report = Scheduler(bundle, ledger, pool).run()
                self.assertEqual(report["committed"], ["shard-0001", "shard-0002", "shard-0003", "shard-0004"])
                self.assertGreaterEqual(len(report["wait_records"]), 1)
                self.assertTrue(all(record["wait_duration_ms"] >= 0 for record in report["wait_records"]))
                usage = ledger.active_usage(1)
                self.assertEqual(usage["resource_units"], 0)
                self.assertEqual(usage["active_leases"], 0)

                grant = ledger.grant_lease(
                    lease_id="lease-test-renew",
                    shard_id="shard-0001",
                    node_id="node-a",
                    capacity_version=1,
                    resource_units=1,
                    ttl_ms=5_000,
                    budget_units=3,
                    concurrency_quota=2,
                    lease_limit=2,
                )
                node = bundle.nodes_by_id["node-a"]
                request_response(
                    node.host,
                    node.port,
                    {"type": "lease_grant", "lease_id": grant["lease_id"], "lease_version": 1},
                    handshake_node_id="node-a",
                )
                renewed = Scheduler(bundle, ledger, pool).renew_lease(grant["lease_id"])
                self.assertEqual(renewed["lease_version"], 2)
                with self.assertRaises(DiagnosticError) as ctx:
                    renew_lease_request(node.host, node.port, node_id="node-a", lease_id=grant["lease_id"], expected_version=1)
                self.assertEqual(ctx.exception.code, Codes.LEASE)
            finally:
                pool.stop()
                ledger.close()

    def test_connection_refused_is_e_port_and_silent_socket_is_e_timeout(self) -> None:
        port = free_loopback_ports(1)[0]
        with self.assertRaises(DiagnosticError) as ctx:
            request_response("127.0.0.1", port, {"type": "handshake"}, timeout=0.2)
        self.assertEqual(ctx.exception.code, Codes.PORT)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            with self.assertRaises(DiagnosticError) as ctx:
                request_response("127.0.0.1", listener.getsockname()[1], {"type": "handshake"}, timeout=0.2)
            self.assertEqual(ctx.exception.code, Codes.TIMEOUT)
        finally:
            listener.close()

    def test_node_bind_failure_reports_e_port(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            result = subprocess.run(
                [sys.executable, "-m", "edge_shard_diag.node_worker", "--node-id", "busy", "--port", str(port)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO_ROOT)},
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(Codes.PORT, result.stdout + result.stderr)
        finally:
            listener.close()


class CapacityLedgerTests(unittest.TestCase):
    def test_snapshot_version_mismatch_forces_new_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            spec = write_demo_spec(root / "spec", allocate_ports=True)
            bundle = load_spec_bundle(spec)
            ledger = Ledger(root / "ledger.sqlite")
            first, replaced, _ = ledger.open_or_replace_snapshot(bundle, 1, {}, {})
            self.assertTrue(replaced)
            manifest = spec / "manifest.json"
            payload = json.loads(manifest.read_text())
            payload["node_set_version"] = 99
            manifest.write_text(json.dumps(payload))
            changed = load_spec_bundle(spec)
            second, replaced, reasons = ledger.open_or_replace_snapshot(changed, 1, {}, {})
            self.assertTrue(replaced)
            self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
            self.assertTrue(any("node_set_version" in reason or "content_digest" in reason for reason in reasons))
            self.assertEqual(len(ledger.list_snapshots()), 2)
            ledger.close()

    def test_replay_compare_uses_e_replay(self) -> None:
        expected = {"s1": {"summary": "same", "cursor": 1}}
        actual = {"s1": {"summary": "different", "cursor": 1}}
        with self.assertRaises(DiagnosticError) as ctx:
            compare_semantic_views(actual, expected, code=Codes.REPLAY, order=["s1"])
        self.assertEqual(ctx.exception.code, Codes.REPLAY)
        self.assertEqual(ctx.exception.field, "summary")

    def test_expiry_and_revocation_never_double_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            spec = write_demo_spec(root / "spec", allocate_ports=True)
            cap_path = spec / "capacity_versions.json"
            payload = json.loads(cap_path.read_text())
            payload["versions"][0]["budget_units"] = 6
            cap_path.write_text(json.dumps(payload))
            bundle = load_spec_bundle(spec)
            ledger = Ledger(root / "ledger.sqlite")
            cap = bundle.capacity_by_version[1]
            ledger.initialize_capacity_slots(cap)
            common = dict(capacity_version=1, budget_units=6, concurrency_quota=2, lease_limit=2)
            ledger.grant_lease(lease_id="long", shard_id="shard-0001", node_id="node-a", resource_units=2, ttl_ms=10_000, **common)
            ledger.grant_lease(lease_id="short", shard_id="shard-0003", node_id="node-c", resource_units=2, ttl_ms=30, **common)
            time.sleep(0.07)
            expired = ledger.expire_due_leases()
            self.assertEqual([row["lease_id"] for row in expired], ["short"])
            revoked = ledger.revoke_capacity_leases(2)
            self.assertEqual(revoked["revoked_lease_ids"], ["long"])
            self.assertEqual(ledger.expire_due_leases(), [])
            held_units = ledger.conn.execute("SELECT COUNT(*) c FROM resource_owners WHERE status='HELD'").fetchone()["c"]
            held_slots = ledger.conn.execute("SELECT COUNT(*) c FROM lease_slots WHERE status='HELD'").fetchone()["c"]
            self.assertEqual((held_units, held_slots), (0, 0))
            statuses = {row["lease_id"]: row["status"] for row in ledger.conn.execute("SELECT lease_id,status FROM leases")}
            self.assertEqual(statuses, {"long": "REVOKED", "short": "EXPIRED"})
            ledger.close()


if __name__ == "__main__":
    unittest.main()
