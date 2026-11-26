from __future__ import annotations

import socket
import threading
import unittest

from gsb.errors import PortError, ProtocolTimeout
from gsb.protocol import call_jsonline, check_port_available


class ProtocolErrorTests(unittest.TestCase):
    def test_occupied_port_reports_e_port(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            with self.assertRaises(PortError) as ctx:
                check_port_available("127.0.0.1", port)
            self.assertEqual(ctx.exception.code, "E_PORT")
            self.assertEqual(int(ctx.exception.position), port)
        finally:
            server.close()

    def test_silent_peer_reports_e_timeout(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        accepted = threading.Event()

        def accept_silently() -> None:
            try:
                conn, _ = server.accept()
                accepted.set()
                # Keep the socket open but never send a JSON-line reply.
                threading.Event().wait(2.0)
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=accept_silently, daemon=True)
        thread.start()
        try:
            with self.assertRaises(ProtocolTimeout) as ctx:
                call_jsonline(
                    "127.0.0.1",
                    port,
                    {"type": "hello"},
                    timeout_ms=80,
                )
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
            self.assertTrue(accepted.wait(1.0))
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
