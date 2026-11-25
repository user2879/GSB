from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from lease_arbitrator.content import build_shard
from lease_arbitrator.errors import FieldError, HandshakeError, LeaseConflictError, PortError
from lease_arbitrator.process import NodeCluster
from lease_arbitrator.protocol import request_json_line, require_receipt_fields, wait_for_node
from lease_arbitrator.specs import NodeSpec


def free_tcp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


class ProtocolErrorTests(unittest.TestCase):
    def test_e_port_when_port_occupied(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with tempfile.TemporaryDirectory() as temp:
            cluster = NodeCluster(
                [NodeSpec("busy", "127.0.0.1", port, 1, 1, 1)],
                run_dir=Path(temp),
                handshake_timeout_ms=200,
            )
            with self.assertRaises(PortError) as ctx:
                cluster.start_node("busy")
            self.assertEqual(ctx.exception.code, "E_PORT")
        listener.close()

    def test_real_handshake_timeout(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with self.assertRaises(HandshakeError) as ctx:
                wait_for_node(
                    host="127.0.0.1",
                    port=port,
                    node_id="slow",
                    process=None,
                    handshake_timeout_ms=250,
                    request_timeout_ms=80,
                )
            self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
        finally:
            listener.close()

    def test_missing_receipt_field_is_e_field(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_empty() -> None:
            conn, _ = server.accept()
            conn.recv(4096)
            conn.sendall(b"{}\n")
            conn.close()

        thread = threading.Thread(target=accept_empty, daemon=True)
        thread.start()
        response = request_json_line("127.0.0.1", port, {"action": "write"}, timeout_ms=1000)
        with self.assertRaises(FieldError) as ctx:
            require_receipt_fields(response, ("lease_id", "digest"), node_id="fake")
        self.assertEqual(ctx.exception.code, "E_FIELD")
        thread.join(timeout=1)
        server.close()


class NodeSubprocessTests(unittest.TestCase):
    def test_write_digest_and_optimistic_renew(self) -> None:
        port = free_tcp_port()
        node = NodeSpec("node-test", "127.0.0.1", port, 3, 2, 4242)
        with tempfile.TemporaryDirectory() as temp:
            cluster = NodeCluster([node], run_dir=Path(temp), shard_bytes=48, handshake_timeout_ms=4000)
            try:
                handshake = cluster.start_node("node-test")
                self.assertEqual(handshake["node_id"], "node-test")
                payload = {
                    "action": "write",
                    "artifact_key": "unit-key",
                    "shard_index": 0,
                    "lease_ms": 60000,
                    "request_seq": 77,
                    "quota_version": 1,
                }
                receipt = cluster.call("node-test", payload)
                require_receipt_fields(receipt, ("lease_id", "digest", "quota_version", "elapsed_ms"))
                expected = build_shard(node_id=node.node_id, node_seed=node.seed, artifact_key="unit-key", shard_index=0, byte_count=48)
                self.assertEqual(receipt["digest"], expected["digest"])
                renewed = cluster.call(
                    "node-test",
                    {
                        "action": "renew",
                        "lease_id": receipt["lease_id"],
                        "lease_version": 1,
                        "lease_ms": 60000,
                        "request_seq": 78,
                        "quota_version": 1,
                    },
                )
                self.assertEqual(renewed["lease_version"], 2)
                with self.assertRaises(LeaseConflictError) as ctx:
                    cluster.call(
                        "node-test",
                        {
                            "action": "renew",
                            "lease_id": receipt["lease_id"],
                            "lease_version": 1,
                            "lease_ms": 60000,
                            "request_seq": 79,
                            "quota_version": 1,
                        },
                    )
                self.assertEqual(ctx.exception.code, "E_LEASE")
            finally:
                cluster.stop_all()


if __name__ == "__main__":
    unittest.main()
