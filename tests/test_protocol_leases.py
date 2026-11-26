from __future__ import annotations

import socket
import time
import unittest

from sharddiag.cluster import occupy_port_for_test
from sharddiag.errors import DiagnosticError, LeaseError, PortError, ProtocolError, TimeoutDiagnosticError
from sharddiag.leases import LeaseManager
from sharddiag.node import NodeServer
from sharddiag.protocol import JsonLineClient, validate_compute_receipt


class ProtocolErrorTests(unittest.TestCase):
    def test_read_timeout_uses_real_socket_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        client = JsonLineClient("127.0.0.1", port, timeout_ms=80)
        started = time.monotonic()
        with self.assertRaises(TimeoutDiagnosticError) as ctx:
            client.handshake(timeout_ms=80)
        elapsed = time.monotonic() - started
        self.assertEqual("E_TIMEOUT", ctx.exception.code)
        self.assertGreaterEqual(elapsed, 0.07)
        client.close()
        listener.close()

    def test_missing_receipt_field_is_e_proto(self) -> None:
        with self.assertRaises(ProtocolError):
            validate_compute_receipt({}, expected_node_id="n", expected_shard_id="s")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def fake_server() -> None:
            conn, _ = listener.accept()
            with conn:
                conn.recv(65536)
                conn.sendall(b'{"ok": true, "type": "hello_ack"}\n')

        import threading

        thread = threading.Thread(target=fake_server, daemon=True)
        thread.start()
        client = JsonLineClient("127.0.0.1", port, expected_node_id="node-x", timeout_ms=500)
        with self.assertRaises(ProtocolError) as ctx:
            client.handshake()
        self.assertEqual("E_PROTO", ctx.exception.code)
        client.close()
        listener.close()

    def test_port_occupation_is_e_port(self) -> None:
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        server = NodeServer(
            node_id="occupied",
            host="127.0.0.1",
            port=port,
            seed=1,
            fixed_point_decimals=0,
        )
        try:
            with self.assertRaises(PortError) as ctx:
                server.start()
            self.assertEqual("E_PORT", ctx.exception.code)
        finally:
            occupied.close()
            server.close()


class LeaseManagerTests(unittest.TestCase):
    def test_node_limit_version_and_expiry(self) -> None:
        manager = LeaseManager()
        first = manager.grant(
            shard_id="s1",
            node_id="n1",
            capacity_version=1,
            resource_units=2,
            lease_limit=1,
            ttl_ms=30,
        )
        with self.assertRaises(LeaseError):
            manager.grant(
                shard_id="s2",
                node_id="n1",
                capacity_version=1,
                resource_units=1,
                lease_limit=1,
                ttl_ms=30,
            )

        unchanged_expiry = first["expires_monotonic_ms"]
        with self.assertRaises(LeaseError) as ctx:
            manager.renew_local(first["lease_id"], 99, max_renewals=3, ttl_ms=1000)
        self.assertEqual("E_LEASE", ctx.exception.code)
        self.assertEqual(1, ctx.exception.details["basis_generation"])
        self.assertEqual(unchanged_expiry, manager.get(first["lease_id"])["expires_monotonic_ms"])

        time.sleep(0.05)
        expired = manager.reconcile_expired()
        self.assertEqual([first["lease_id"]], [item["lease_id"] for item in expired])
        self.assertEqual(0, manager.used_units())

    def test_renewal_count_limit(self) -> None:
        manager = LeaseManager()
        lease = manager.grant(
            shard_id="s", node_id="n", capacity_version=1, resource_units=1, lease_limit=2, ttl_ms=1000
        )
        with self.assertRaises(LeaseError):
            manager.renew_local(lease["lease_id"], 1, max_renewals=0, ttl_ms=1000)

    def test_expiry_and_generation_recovery_are_mutually_exclusive(self) -> None:
        manager = LeaseManager()
        expired_lease = manager.grant(
            shard_id="exp", node_id="n", capacity_version=1, resource_units=3, lease_limit=10, ttl_ms=20
        )
        generation_lease = manager.grant(
            shard_id="gen", node_id="n", capacity_version=1, resource_units=4, lease_limit=10, ttl_ms=10_000
        )
        time.sleep(0.04)
        expired = manager.reconcile_expired()
        revoked = manager.revoke_for_capacity(2)
        second_reconcile = manager.reconcile_expired()
        self.assertEqual([expired_lease["lease_id"]], [item["lease_id"] for item in expired])
        self.assertEqual([generation_lease["lease_id"]], [item["lease_id"] for item in revoked])
        self.assertEqual([], second_reconcile)
        self.assertEqual(0, manager.used_units())
        manager.assert_single_holder()


class NodeLeaseGenerationTests(unittest.TestCase):
    def test_node_optimistic_renew_reports_generation_after_advance(self) -> None:
        server = NodeServer(
            node_id="gen-node",
            host="127.0.0.1",
            port=0,
            seed=7,
            fixed_point_decimals=1,
            lease_limit=4,
            max_renewals=3,
        )
        server.start()
        port = server._socket.getsockname()[1]
        try:
            with JsonLineClient("127.0.0.1", port, expected_node_id="gen-node", timeout_ms=1000) as client:
                client.handshake()
                granted = client.lease_grant(
                    {
                        "lease_id": "lease-gen-s1-gen1",
                        "shard_id": "s1",
                        "capacity_version": 1,
                        "granted_monotonic_ms": 1,
                        "expires_monotonic_ms": 10_000_000,
                        "ttl_ms": 1000,
                    }
                )
                self.assertEqual(1, granted["lease_version"])
                advanced = client.capacity_advance(
                    {"to_version": 2, "parent_version": 1, "concurrency_quota": 2, "lease_limit": 4}
                )
                self.assertEqual([granted["lease_id"]], advanced["revoked_lease_ids"])
                with self.assertRaises(DiagnosticError) as ctx:
                    client.lease_renew(
                        {"lease_id": granted["lease_id"], "expected_version": 1, "max_renewals": 3}
                    )
                self.assertEqual("E_LEASE", ctx.exception.code)
                self.assertEqual(2, ctx.exception.details["basis_generation"])
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
