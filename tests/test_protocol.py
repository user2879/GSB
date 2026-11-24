from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path

from quota_ledger.client import NodeClient, NodeProcess
from quota_ledger.errors import BusinessError
from quota_ledger.spec import NodeSpec

from .helpers import free_port_base


def node_spec(port: int) -> NodeSpec:
    return NodeSpec(
        id="node-test",
        host="127.0.0.1",
        port=port,
        concurrency=2,
        lease_limit=2,
        seed=99,
        capacity_version=1,
    )


class _FakeServer:
    def __init__(self, port: int, mode: str):
        self.port = port
        self.mode = mode
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.ready.wait(1.0)

    def _run(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen(1)
        self.ready.set()
        conn, _ = listener.accept()
        try:
            conn.recv(65536)
            if self.mode == "hang":
                time.sleep(0.5)
            elif self.mode == "missing":
                conn.sendall(b'{"ok":true,"node_id":"node-test"}\n')
        finally:
            conn.close()
            listener.close()


class ProtocolTests(unittest.TestCase):
    def test_real_socket_timeout(self) -> None:
        base = free_port_base(1)
        server = _FakeServer(base, "hang")
        server.start()
        client = NodeClient(node_spec(base), timeout_s=0.12)
        started = time.monotonic()
        with self.assertRaises(BusinessError) as ctx:
            client.ping()
        elapsed = time.monotonic() - started
        self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 1.0)
        server.thread.join(timeout=1.0)

    def test_reply_missing_required_field(self) -> None:
        base = free_port_base(1)
        server = _FakeServer(base, "missing")
        server.start()
        client = NodeClient(node_spec(base), timeout_s=1.0)
        with self.assertRaises(BusinessError) as ctx:
            client.ping()
        self.assertEqual(ctx.exception.code, "E_REPLY")
        self.assertEqual(ctx.exception.field, "ready")
        server.thread.join(timeout=1.0)

    def test_occupied_port_reports_e_port(self) -> None:
        base = free_port_base(1)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", base))
        listener.listen(1)
        process = NodeProcess(node_spec(base), 1000, handshake_timeout_s=1.0)
        try:
            with self.assertRaises(BusinessError) as ctx:
                process.start()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            listener.close()
            process.stop()

    def test_node_process_handshake_timeout_is_real_socket_timeout(self) -> None:
        base = free_port_base(1)
        hanging = Path(__file__).with_name("hanging_node.py")
        process = NodeProcess(
            node_spec(base),
            1000,
            handshake_timeout_s=0.45,
            command_override=[sys.executable, str(hanging), str(base)],
        )
        started = time.monotonic()
        try:
            with self.assertRaises(BusinessError) as ctx:
                process.start()
            elapsed = time.monotonic() - started
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
            self.assertGreaterEqual(elapsed, 0.4)
            self.assertLess(elapsed, 2.5)
        finally:
            process.stop()

    def test_node_process_optimistic_lease_version(self) -> None:
        base = free_port_base(1)
        process = NodeProcess(node_spec(base), 2000, handshake_timeout_s=3.0)
        process.start()
        try:
            ping = process.client.ping()
            self.assertTrue(ping["ready"])
            first = process.client.acquire_lease(lease_id="l1", shard_id="s1", ttl_ms=2000)
            self.assertEqual(first["version"], 1)
            renewed = process.client.renew_lease(lease_id="l1", expected_version=1, ttl_ms=2000)
            self.assertEqual(renewed["version"], 2)
            before = renewed["expires_ms"]
            with self.assertRaises(BusinessError) as ctx:
                process.client.renew_lease(lease_id="l1", expected_version=1, ttl_ms=5000)
            self.assertEqual(ctx.exception.code, "E_LEASE")
            process.client.release_lease(lease_id="l1")
            self.assertGreaterEqual(before, renewed["expires_ms"])
        finally:
            process.stop()


if __name__ == "__main__":
    unittest.main()
