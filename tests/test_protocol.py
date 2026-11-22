from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from meshwork.errors import MeshError
from meshwork.model import RequestSpec
from meshwork.protocol import check_port_free, send_attempt, send_ping
from tests.helpers import free_ports


class FakeServer:
    def __init__(self, response_factory, delay: float = 0.0) -> None:
        self.response_factory = response_factory
        self.delay = delay
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = int(self.sock.getsockname()[1])
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            try:
                conn.recv(65536)
                if self.delay:
                    time.sleep(self.delay)
                payload = self.response_factory()
                if payload is not None:
                    conn.sendall(payload)
            except OSError:
                pass

    def close(self) -> None:
        self.sock.close()
        self.thread.join(timeout=1)


class ProtocolTests(unittest.TestCase):
    def test_missing_reply_field_is_e_reply(self) -> None:
        response = {
            "protocol": 1,
            "request_id": "r1",
            "attempt_no": 1,
            # instance_id is deliberately missing
            "status": "ok",
            "latency_class": "fast",
            "delay_ms": 1,
        }
        server = FakeServer(
            lambda: json.dumps(response).encode("utf-8") + b"\n"
        )
        try:
            result = send_attempt(
                "127.0.0.1",
                server.port,
                "inst",
                "svc",
                RequestSpec("r1", "svc", {}, 1),
                1,
                1,
                1,
                200,
            )
        finally:
            server.close()
        self.assertEqual(result.error_code, "E_REPLY")
        self.assertEqual(result.outcome, "reply_error")

    def test_attempt_timeout_is_a_real_socket_timeout(self) -> None:
        server = FakeServer(lambda: None, delay=0.3)
        started = time.monotonic()
        try:
            result = send_attempt(
                "127.0.0.1",
                server.port,
                "inst",
                "svc",
                RequestSpec("r1", "svc", {}, 1),
                1,
                1,
                1,
                80,
            )
            elapsed = time.monotonic() - started
        finally:
            server.close()
        self.assertEqual(result.error_code, "E_TIMEOUT")
        self.assertEqual(result.outcome, "timeout")
        self.assertGreaterEqual(elapsed, 0.06)
        self.assertLess(elapsed, 0.28)

    def test_unreachable_handshake_is_e_host(self) -> None:
        port = free_ports(1)[0]
        with self.assertRaises(MeshError) as ctx:
            send_ping("127.0.0.1", port, "inst", 100)
        self.assertEqual(ctx.exception.code, "E_HOST")

    def test_incomplete_handshake_reply_is_e_reply(self) -> None:
        server = FakeServer(
            lambda: json.dumps({"protocol": 1, "type": "pong"}).encode("utf-8") + b"\n"
        )
        try:
            with self.assertRaises(MeshError) as ctx:
                send_ping("127.0.0.1", server.port, "inst", 500)
        finally:
            server.close()
        self.assertEqual(ctx.exception.code, "E_REPLY")

    def test_occupied_port_is_e_port(self) -> None:
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = int(holder.getsockname()[1])
        try:
            with self.assertRaises(MeshError) as ctx:
                check_port_free("127.0.0.1", port)
        finally:
            holder.close()
        self.assertEqual(ctx.exception.code, "E_PORT")
        self.assertEqual(ctx.exception.location, f"127.0.0.1:{port}")


if __name__ == "__main__":
    unittest.main()
