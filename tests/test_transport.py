import socket
import threading
import time
import unittest

from rlcb.codes import E_PORT, E_REPLY, E_TIMEOUT, PortError, ReplyError
from rlcb.transport import ServiceClient


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.client = ServiceClient(connect_timeout_ms=500,
                                    read_timeout_ms=1000,
                                    handshake_timeout_ms=3000)
        self.processes = []

    def tearDown(self):
        for sp in self.processes:
            sp.close()

    def _spawn(self, port):
        sp = self.client.spawn_and_connect(
            service_id=f"svc{port}", host="127.0.0.1", port=port,
            config_version=1, seed=20261126, ttl_ms=250)
        self.processes.append(sp)
        return sp

    def test_real_loopback_execute_roundtrip(self):
        sp = self._spawn(33011)
        reply = self.client.request(sp, {
            "kind": "execute", "batch_id": "B1",
            "requests": [{"request_id": "r1", "arrival_seq": 0,
                          "failed": 1}],
        })
        self.assertEqual(reply["service_id"], sp.service_id)
        self.assertIn("summary", reply)
        self.assertIn("elapsed_ms", reply)
        self.assertEqual(reply["config_version"], 1)

    def test_deterministic_summary_bytes(self):
        sp = self._spawn(33012)
        msg = {"kind": "execute", "batch_id": "B1", "requests": [
            {"request_id": "r1", "arrival_seq": 0, "failed": 0}]}
        import json
        r1 = self.client.request(sp, msg)
        r2 = self.client.request(sp, msg)
        self.assertEqual(json.dumps(r1["summary"], sort_keys=True),
                         json.dumps(r2["summary"], sort_keys=True))

    def test_port_in_use_raises_e_port(self):
        holder = socket.socket()
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 33020))
        holder.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                self.client.spawn_and_connect(
                    service_id="svcBlocked", host="127.0.0.1", port=33020,
                    config_version=1, seed=1, ttl_ms=10)
            self.assertEqual(ctx.exception.code, E_PORT)
        finally:
            holder.close()

    def test_read_timeout_e_timeout(self):
        # A peer that accepts and holds the connection but never replies to a
        # request must surface a real socket read timeout as E_TIMEOUT.
        holder = socket.socket()
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 33031))
        holder.listen(5)
        holder.settimeout(5)
        established = threading.Event()

        def hold_connections():
            try:
                conn, _ = holder.accept()
                established.set()
                time.sleep(3)  # never reply
                conn.close()
            except OSError:
                pass

        thread = threading.Thread(target=hold_connections, daemon=True)
        thread.start()
        client = ServiceClient(connect_timeout_ms=500,
                               read_timeout_ms=150,
                               handshake_timeout_ms=800)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(("127.0.0.1", 33031))
            established.wait(2)
            from rlcb.transport import ServiceProcess
            sp = ServiceProcess("svcSilent", "127.0.0.1", 33031,
                                process=None, sock=sock)
            with self.assertRaises(Exception) as ctx:
                client.request(sp, {"kind": "hello", "batch_id": ""})
            self.assertEqual(ctx.exception.code, E_TIMEOUT)
            sock.close()
        finally:
            holder.close()
            thread.join(timeout=2)

    def test_occupied_port_raises_e_port_on_connect(self):
        holder = socket.socket()
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 33032))
        holder.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                self.client.spawn_and_connect(
                    service_id="svcBlocked", host="127.0.0.1", port=33032,
                    config_version=1, seed=1, ttl_ms=10)
            self.assertEqual(ctx.exception.code, E_PORT)
        finally:
            holder.close()

    def test_missing_reply_field_e_reply(self):
        sp = self._spawn(33013)
        # A malformed kind yields an error envelope lacking required fields.
        with self.assertRaises(ReplyError) as ctx:
            self.client.request(sp, {"kind": "explode"})
        self.assertEqual(ctx.exception.code, E_REPLY)


if __name__ == "__main__":
    unittest.main()
