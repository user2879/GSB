from __future__ import annotations

import socket
import threading
import time
import unittest
from pathlib import Path

from leasemesh.cluster import NodeCluster, free_port
from leasemesh.errors import MeshError
from leasemesh.protocol import request_json_line
from leasemesh.scheduler import BoundedScheduler
from leasemesh.spec import load_declaration


ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class SchedulerTests(unittest.TestCase):
    def test_global_limit_is_real(self) -> None:
        scheduler = BoundedScheduler(global_limit=2, per_node_limit=5)
        release = threading.Event()
        observed_max = 0
        lock = threading.Lock()

        def job(_job):
            nonlocal observed_max
            with lock:
                observed_max = max(observed_max, scheduler.active_global)
            self.assertLessEqual(scheduler.active_global, 2)
            release.wait(2.0)
            return True

        futures = [scheduler.submit({"request_id": f"g{i}", "node_id": "node-a"}, job) for i in range(4)]
        self.assertTrue(wait_for(lambda: scheduler.snapshot()["active_global"] == 2))
        snapshot = scheduler.snapshot()
        self.assertEqual(2, snapshot["max_active_global"])
        release.set()
        self.assertTrue(all(future.result()["ok"] for future in futures))
        scheduler.shutdown()
        self.assertLessEqual(observed_max, 2)

    def test_per_node_limit_is_real(self) -> None:
        scheduler = BoundedScheduler(global_limit=4, per_node_limit=1)
        release = threading.Event()

        def job(_job):
            release.wait(2.0)
            return True

        futures = [
            scheduler.submit({"request_id": "a1", "node_id": "node-a"}, job),
            scheduler.submit({"request_id": "a2", "node_id": "node-a"}, job),
            scheduler.submit({"request_id": "a3", "node_id": "node-a"}, job),
            scheduler.submit({"request_id": "b1", "node_id": "node-b"}, job),
        ]
        self.assertTrue(wait_for(lambda: scheduler.snapshot()["active_by_node"].get("node-a") == 1))
        time.sleep(0.05)
        snapshot = scheduler.snapshot()
        self.assertEqual(1, snapshot["active_by_node"]["node-a"])
        self.assertLessEqual(snapshot["active_global"], 4)
        self.assertGreaterEqual(snapshot["max_active_by_node"]["node-a"], 1)
        release.set()
        self.assertTrue(all(future.result()["ok"] for future in futures))
        scheduler.shutdown()


class ProtocolErrorTests(unittest.TestCase):
    def test_handshake_timeout(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def accept_silent() -> None:
            try:
                conn, _ = server.accept()
                time.sleep(0.5)
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=accept_silent, daemon=True)
        thread.start()
        try:
            with self.assertRaises(MeshError) as caught:
                request_json_line(
                    "127.0.0.1",
                    port,
                    {"op": "ping"},
                    handshake_timeout_ms=80,
                    timeout_ms=500,
                )
            self.assertEqual("E_TIMEOUT", caught.exception.code)
        finally:
            server.close()

    def test_missing_receipt_field(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def stub() -> None:
            conn, _ = server.accept()
            conn.settimeout(2)
            conn.recv(65536)
            conn.sendall(b'{"ok":true,"type":"hello_ack","node_id":"node-a"}\n')
            conn.recv(65536)
            conn.sendall(b'{"ok":true,"node_id":"node-a"}\n')
            conn.close()

        thread = threading.Thread(target=stub, daemon=True)
        thread.start()
        try:
            with self.assertRaises(MeshError) as caught:
                request_json_line(
                    "127.0.0.1",
                    port,
                    {"op": "grant", "lease_id": "lease-x"},
                    node_id="node-a",
                    handshake_timeout_ms=500,
                    timeout_ms=1000,
                )
            self.assertEqual("E_FIELD", caught.exception.code)
            self.assertEqual("lease_id", caught.exception.field)
        finally:
            server.close()

    def test_read_timeout_after_handshake(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def stub() -> None:
            conn, _ = server.accept()
            conn.settimeout(2)
            conn.recv(65536)
            conn.sendall(b'{"ok":true,"type":"hello_ack","node_id":"node-a"}\n')
            time.sleep(0.5)
            conn.close()

        thread = threading.Thread(target=stub, daemon=True)
        thread.start()
        try:
            with self.assertRaises(MeshError) as caught:
                request_json_line(
                    "127.0.0.1",
                    port,
                    {"op": "ping"},
                    node_id="node-a",
                    handshake_timeout_ms=500,
                    timeout_ms=80,
                )
            self.assertEqual("E_TIMEOUT", caught.exception.code)
            self.assertEqual("response", caught.exception.field)
        finally:
            server.close()

    def test_port_occupancy_reports_e_port(self) -> None:
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", free_port()))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        spec = load_declaration(ROOT / "samples/declarations.json")
        spec["nodes"] = spec["nodes"][:1]
        spec["nodes"][0]["port"] = port
        try:
            with self.assertRaises(MeshError) as caught:
                NodeCluster(spec, ROOT / "work" / "test-port-occupancy").start()
            self.assertEqual("E_PORT", caught.exception.code)
        finally:
            occupied.close()


if __name__ == "__main__":
    unittest.main()
