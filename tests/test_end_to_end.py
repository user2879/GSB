"""End-to-end tests using real loopback arbiter subprocesses."""
import json
import socket
import tempfile
import unittest
from pathlib import Path

from distlock.errors import HandshakeError, PortError, ReplyError
from distlock.models import Domain, ScaleFactor
from distlock.orchestrator import ArbiterPool
from distlock.protocol import (
    HANDSHAKE_TIMEOUT_MS, ArbiterClient, check_port_free, encode,
)

DOMAINS = {
    "d1": Domain("d1", 1, 500, 100, 2, 1),
    "d2": Domain("d2", 2, 500, 100, 2, 1),
}


class ProtocolErrorTests(unittest.TestCase):
    def test_e_port_when_port_occupied(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        port = holder.getsockname()[1]
        try:
            with self.assertRaises(PortError) as ctx:
                check_port_free("127.0.0.1", port)
            self.assertEqual(ctx.exception.code, "E_PORT")
            self.assertEqual(ctx.exception.context["port"], port)
        finally:
            holder.close()

    def test_e_handshake_on_wrong_server_reply(self):
        # A raw server that never answers HELLO -> bounded handshake timeout.
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.bind(("127.0.0.1", 0))
        raw.listen(1)
        port = raw.getsockname()[1]

        def silent_server():
            conn, _ = raw.accept()
            conn.recv(4096)  # read HELLO but never reply
            conn.close()
            raw.close()

        import threading
        t = threading.Thread(target=silent_server, daemon=True)
        t.start()
        with self.assertRaises(HandshakeError) as ctx:
            ArbiterClient("127.0.0.1", port, "d1", 1, 1,
                          handshake_timeout_ms=400)
        self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
        t.join(timeout=2)

    def test_e_reply_on_missing_receipt_field(self):
        raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw.bind(("127.0.0.1", 0))
        raw.listen(1)
        port = raw.getsockname()[1]

        def bad_server():
            conn, _ = raw.accept()
            conn.recv(4096)
            conn.sendall(encode({"type": "HELLO_ACK", "domain_id": "d1"}))
            conn.recv(4096)
            # RECEIPT without the required remaining_ms field.
            conn.sendall(encode({"type": "RECEIPT", "seq": 1,
                                 "domain_id": "d1", "holder_id": "h1",
                                 "verdict": "GRANTED"}))
            conn.close()
            raw.close()

        import threading
        t = threading.Thread(target=bad_server, daemon=True)
        t.start()
        client = ArbiterClient("127.0.0.1", port, "d1", 1, 1,
                               handshake_timeout_ms=2000)
        with self.assertRaises(ReplyError) as ctx:
            client.request(1, "h1", "ACQUIRE", 5)
        self.assertEqual(ctx.exception.code, "E_REPLY")
        self.assertEqual(ctx.exception.context["field"], "remaining_ms")
        client.close()
        t.join(timeout=2)


class SubprocessRestartTests(unittest.TestCase):
    def test_restart_discards_in_memory_ownership(self):
        workdir = Path(tempfile.mkdtemp())
        pool = ArbiterPool(DOMAINS, "127.0.0.1", 0, 10000, 1,
                           runtime_dir=workdir)
        # Assign free ephemeral ports explicitly.
        ports = []
        for proc in pool.processes.values():
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            ports.append((proc, s.getsockname()[1]))
            s.close()
        for proc, port in ports:
            proc.port = port
        pool.start_all()
        try:
            c1 = pool.client("d1")
            self.assertEqual(c1.request(1, "h1", "ACQUIRE", 5)["verdict"],
                             "GRANTED")
            self.assertEqual(c1.status()["active_leases"][0]["holder_id"], "h1")
            # Force kill + restart: fresh generation, empty hold table.
            pool.restart_domain("d1")
            c2 = pool.client("d1")
            self.assertEqual(c2.status()["active_leases"], [])
            self.assertEqual(pool.generations()["d1"], 2)
            # Old lease gone; h1 can acquire again cleanly after replay.
            self.assertEqual(c2.request(2, "h1", "ACQUIRE", 5)["verdict"],
                             "GRANTED")
        finally:
            pool.shutdown()

    def test_status_is_json_serializable_and_read_only(self):
        workdir = Path(tempfile.mkdtemp())
        pool = ArbiterPool({"d1": DOMAINS["d1"]}, "127.0.0.1", 0, 10000, 1,
                           runtime_dir=workdir)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        for proc in pool.processes.values():
            proc.port = port
        pool.start_all()
        try:
            client = pool.client("d1")
            client.request(1, "h1", "ACQUIRE", 5)
            status = client.status()
            json.dumps(status)  # must cross the JSON-lines wire cleanly
            self.assertEqual(status["watermark"]["ratio_fixed"], 10000)
        finally:
            pool.shutdown()


class NumpyWatermarkTests(unittest.TestCase):
    def test_vector_floor_matches_scalar(self):
        from distlock.models import vector_watermark
        import numpy as np
        active = [0, 1, 2, 3]
        capacity = [3, 3, 3, 3]
        got = vector_watermark(active, capacity, 10000)
        expected = np.array([0, 3333, 6666, 10000], dtype=np.int64)
        self.assertTrue(np.array_equal(got, expected))


if __name__ == "__main__":
    unittest.main()
