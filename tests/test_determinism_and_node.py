"""Requirement 4: deterministic content and real loopback node subprocesses."""
import socket
import time
import unittest

from shardmeta.content import generate_shard_bytes, compute_checksum
from shardmeta.errors import PortError, ReplyError, TimeoutError
from shardmeta.models import NodeSpec
from shardmeta.node import NodeClient, NodeOrchestrator, NodeServer


class DeterminismTest(unittest.TestCase):
    def test_bytes_and_digest_order_and_batch_independent(self):
        kwargs = dict(node_seed=42, object_seed=7, key="obj", shard_index=2, size=64)
        first = generate_shard_bytes(**kwargs)
        # Different call order / interleaving must not change the bytes.
        _ = generate_shard_bytes(node_seed=99, object_seed=1, key="x",
                                 shard_index=0, size=32)
        second = generate_shard_bytes(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(
            compute_checksum("n1", "obj", 2, first).shard_digest,
            compute_checksum("n1", "obj", 2, second).shard_digest,
        )

    def test_digest_binds_node_key_index_content(self):
        data = generate_shard_bytes(1, 2, "k", 0, 32)
        base = compute_checksum("n1", "k", 0, data).shard_digest
        self.assertNotEqual(
            base, compute_checksum("n2", "k", 0, data).shard_digest)
        self.assertNotEqual(
            base, compute_checksum("n1", "other", 0, data).shard_digest)
        self.assertNotEqual(
            base, compute_checksum("n1", "k", 1, data).shard_digest)


class NodeTransportTest(unittest.TestCase):
    def setUp(self):
        self.spec = NodeSpec(
            node_id="test-node", host="127.0.0.1", port=43055,
            capacity_slots=4, lease_limit=4, seed=11, meta_version=1,
        )

    def test_real_subprocess_upload_and_grant(self):
        with NodeOrchestrator({"test-node": self.spec}) as orchestrator:
            client = orchestrator.client("test-node")
            pong = client.ping()
            self.assertTrue(pong["ok"])
            grant = client.call(
                {"type": "grant", "object_key": "k", "shard_index": 0,
                 "ttl_ms": 500}, expect_fields=("lease",))
            self.assertIn("lease_id", grant["lease"])
            upload = client.call(
                {"type": "upload", "object_key": "k", "shard_index": 0,
                 "object_seed": 7, "size": 16, "merge_id": "M-x"},
                expect_fields=("checksum", "merge_id"),
            )
            self.assertEqual(len(upload["checksum"]["shard_digest"]), 16)
            self.assertEqual(upload["meta_version"], 1)

    def test_port_in_use_maps_to_E_PORT(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", self.spec.port))
        holder.listen(1)
        try:
            server = NodeServer(self.spec)
            with self.assertRaises(PortError) as ctx:
                server.start()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            holder.close()

    def test_connection_refused_maps_to_E_PORT(self):
        unused = NodeSpec(
            node_id="ghost", host="127.0.0.1", port=43099,
            capacity_slots=1, lease_limit=1, seed=1, meta_version=1,
        )
        client = NodeClient(unused, timeout_ms=500)
        with self.assertRaises(PortError):
            client.ping()

    def test_missing_reply_field_maps_to_E_REPLY(self):
        # A raw server that replies without required fields.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 43077))
        server_sock.listen(1)
        spec = NodeSpec(node_id="bad", host="127.0.0.1", port=43077,
                        capacity_slots=1, lease_limit=1, seed=1, meta_version=1)
        client = NodeClient(spec, timeout_ms=1000)

        def serve_once():
            conn, _ = server_sock.accept()
            conn.recv(4096)
            conn.sendall(b'{"hello":"world"}\n')
            conn.close()

        import threading
        thread = threading.Thread(target=serve_once)
        thread.start()
        with self.assertRaises(ReplyError) as ctx:
            client.ping()
        self.assertEqual(ctx.exception.code, "E_REPLY")
        thread.join()
        server_sock.close()

    def test_timeout_maps_to_E_TIMEOUT(self):
        # Accept but never reply.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 43088))
        server_sock.listen(1)
        spec = NodeSpec(node_id="slow", host="127.0.0.1", port=43088,
                        capacity_slots=1, lease_limit=1, seed=1, meta_version=1)
        client = NodeClient(spec, timeout_ms=200)

        def hold():
            conn, _ = server_sock.accept()
            time.sleep(0.6)
            conn.close()

        import threading
        thread = threading.Thread(target=hold)
        thread.start()
        with self.assertRaises(TimeoutError) as ctx:
            client.ping()
        self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        thread.join()
        server_sock.close()


if __name__ == "__main__":
    unittest.main()
