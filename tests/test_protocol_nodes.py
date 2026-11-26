import json
import socket
import threading
import time
import unittest

from shardmesh.cluster import NodeProcess
from shardmesh.errors import ShardMeshError
from shardmesh.leases import LeaseManager
from shardmesh.protocol import assert_port_available, handshake, rpc
from shardmesh.samples import aggregate_batch, generate_batch
from shardmesh.spec import NodeSpec


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def node_spec(port: int, **overrides) -> NodeSpec:
    values = dict(
        id="n-test",
        host="127.0.0.1",
        port=port,
        concurrency=2,
        lease_limit=2,
        seed=42,
        capacity_version=1,
    )
    values.update(overrides)
    return NodeSpec(**values)


class ProtocolErrorTests(unittest.TestCase):
    def test_port_occupied_reports_e_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = free_port()
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        try:
            with self.assertRaises(ShardMeshError) as ctx:
                assert_port_available("127.0.0.1", port)
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            listener.close()

    def test_handshake_timeout_reports_e_timeout(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_silently():
            try:
                conn, _ = server.accept()
                time.sleep(1.0)
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=accept_silently, daemon=True)
        thread.start()
        try:
            with self.assertRaises(ShardMeshError) as ctx:
                handshake("127.0.0.1", port, "n-timeout", timeout_ms=120)
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            server.close()
            thread.join(timeout=2)

    def test_missing_response_field_reports_e_field(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def respond_without_node_id():
            conn, _ = server.accept()
            conn.makefile("rb").readline()
            conn.sendall(b'{"ok":true,"type":"hello_ack","protocol_version":1}\n')
            conn.close()

        thread = threading.Thread(target=respond_without_node_id, daemon=True)
        thread.start()
        try:
            with self.assertRaises(ShardMeshError) as ctx:
                handshake("127.0.0.1", port, "n-field", timeout_ms=500)
            self.assertEqual(ctx.exception.code, "E_FIELD")
        finally:
            server.close()
            thread.join(timeout=2)


class RealNodeSubprocessTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.process = NodeProcess(node_spec(self.port), scale=100, log_dir="artifacts/test-logs", request_timeout_ms=2000)
        self.process.start()

    def tearDown(self):
        self.process.stop()

    def test_aggregate_json_line_response(self):
        batch = generate_batch(seed=7, rows=64)
        response = self.process.aggregate(
            {"shard_id": "s-test", "capacity_version": 1, "batch": batch}
        )
        self.assertEqual(response["node_id"], "n-test")
        self.assertEqual(response["shard_id"], "s-test")
        self.assertEqual(response["capacity_version"], 1)
        self.assertIn("elapsed_ms", response)
        expected = aggregate_batch(batch, 100)
        self.assertEqual(response["summary"]["checksum"], expected["checksum"])

    def test_lease_optimistic_version_and_renew_limit(self):
        lease = {
            "lease_id": "lease-s-test-g1",
            "shard_id": "s-test",
            "capacity_version": 1,
        }
        opened = self.process.lease_open(lease, ttl_ms=1000)
        self.assertEqual(opened["version"], 1)
        renewed = self.process.lease_renew(lease["lease_id"], 1, ttl_ms=1000, max_renewals=1)
        self.assertEqual(renewed["version"], 2)
        self.assertEqual(renewed["renew_count"], 1)

        with self.assertRaises(ShardMeshError) as ctx:
            self.process.lease_renew(lease["lease_id"], 1, ttl_ms=1000, max_renewals=5)
        self.assertEqual(ctx.exception.code, "E_LEASE")

        with self.assertRaises(ShardMeshError) as limit:
            self.process.lease_renew(lease["lease_id"], 2, ttl_ms=1000, max_renewals=1)
        self.assertEqual(limit.exception.code, "E_RENEW")


class LeaseExpiryTests(unittest.TestCase):
    def test_real_monotonic_expiry_releases_capacity(self):
        released = []
        manager = LeaseManager(on_expire=lambda lease: released.append(lease.shard_id))
        manager.start_reaper()
        try:
            manager.issue(
                lease_id="lease-exp",
                shard_id="s-exp",
                node_id="n1",
                ttl_ms=60,
                capacity_version=1,
            )
            self.assertEqual(manager.active_count("n1"), 1)
            deadline = time.monotonic() + 1.0
            while manager.active_count("n1") and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.active_count("n1"), 0)
            self.assertIn("s-exp", released)
        finally:
            manager.stop_reaper()


if __name__ == "__main__":
    unittest.main()
