import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from shard_diag.errors import FieldError, PortError, ProtocolTimeout
from shard_diag.node_manager import NodeManager
from shard_diag.protocol import ensure_port_available, find_free_port, request_json, require_fields
from shard_diag.spec import NodeSpec, RuntimeConfig, Spec


def one_node_spec(port=None, **runtime):
    node = NodeSpec("n1", "127.0.0.1", port or find_free_port(), 2, 2, 4242, 1)
    config = RuntimeConfig(io_timeout_ms=1000, ready_timeout_ms=5000, **runtime)
    return Spec(1, (node,), (), 10, config)


class FakeLineServer:
    def __init__(self, response_lines):
        self.response_lines = response_lines
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        try:
            conn, _ = self.sock.accept()
            with conn:
                conn.settimeout(2)
                try:
                    conn.recv(65536)
                except OSError:
                    pass
                for line in self.response_lines:
                    conn.sendall((json.dumps(line) + "\n").encode())
                if not self.response_lines:
                    threading.Event().wait(1.0)
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class ProtocolTests(unittest.TestCase):
    def test_port_occupied_code(self):
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with self.assertRaises(PortError) as ctx:
                ensure_port_available(port, "busy-node")
            self.assertEqual(ctx.exception.code, "E_PORT")
            self.assertEqual(ctx.exception.object_id, "busy-node")
        finally:
            listener.close()

    def test_timeout_code(self):
        server = FakeLineServer([])
        try:
            with self.assertRaises(ProtocolTimeout) as ctx:
                request_json(server.port, {"type": "ping"}, timeout_ms=60)
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            server.close()

    def test_missing_handshake_field_code(self):
        server = FakeLineServer([{"ok": True, "type": "hello_ack"}])
        try:
            with self.assertRaises(FieldError) as ctx:
                request_json(server.port, {"type": "ping"}, timeout_ms=500)
            self.assertEqual(ctx.exception.code, "E_FIELD")
            self.assertEqual(ctx.exception.field, "node_id")
        finally:
            server.close()

    def test_missing_result_field_code(self):
        ack = {"ok": True, "type": "hello_ack", "node_id": "fake", "protocol_version": 1}
        server = FakeLineServer([ack, {"ok": True}])
        try:
            response = request_json(server.port, {"type": "aggregate_batch"}, timeout_ms=500)
            with self.assertRaises(FieldError) as ctx:
                require_fields(response, ["partial"], "s1")
            self.assertEqual(ctx.exception.code, "E_FIELD")
        finally:
            server.close()

    def test_real_subprocess_aggregate_force_kill_releases_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = one_node_spec()
            manager = NodeManager(spec, Path(tmp) / "runtime")
            manager.start_all()
            try:
                response = manager.call("n1", {
                    "type": "aggregate_batch",
                    "shard_id": "s1",
                    "quota_version": 1,
                    "batch_index": 0,
                    "batch_count": 1,
                    "rows": [[0, 3], [1, -2], [0, 7]],
                })
                self.assertTrue(response["ok"])
                self.assertEqual(response["partial"]["participated_rows"], 3)

                errors = []

                def slow_call():
                    try:
                        manager.call("n1", {
                            "type": "aggregate_batch", "shard_id": "slow",
                            "quota_version": 1, "batch_index": 0, "batch_count": 1,
                            "rows": [[0, 1]], "processing_delay_ms": 800,
                        }, timeout_ms=2000)
                    except Exception as exc:
                        errors.append(exc)

                worker = threading.Thread(target=slow_call, daemon=True)
                worker.start()
                manager.force_kill_node("n1")
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                ensure_port_available(spec.nodes[0].port, "n1")
                self.assertNotIn("n1", manager.processes)
                manager.start_node("n1")
                restarted = manager.call("n1", {"type": "ping"})
                self.assertEqual(restarted["type"], "pong")
            finally:
                manager.stop_all(force=True)


if __name__ == "__main__":
    unittest.main()
