import socket
import subprocess
import sys
import unittest

from arbplatform.client import ServiceSupervisor
from arbplatform.errors import PortError, ReplyError, TimeoutError
from arbplatform.protocol import send_line


def obs(env, seq, key="k", value=1):
    return {"obs_id": f"{env}:i:{seq}:{key}", "env_id": env,
            "instance_id": "i", "seq": seq, "gen_time_ms": seq,
            "key": key, "value": value}


class TcpServiceTest(unittest.TestCase):
    def setUp(self):
        self.supervisor = ServiceSupervisor(
            ready_timeout_s=5.0, io_timeout_s=2.0)
        self.env_id = "env-test"
        self.port = 49201

    def tearDown(self):
        self.supervisor.stop_all()

    def test_real_subprocess_batch_and_arbitration(self):
        self.supervisor.start(self.env_id, "127.0.0.1", self.port)
        ack = self.supervisor.request(
            self.env_id,
            {"type": "batch", "observations": [obs(self.env_id, 1),
                                               obs(self.env_id, 2)]})
        self.assertEqual(ack["type"], "batch_ack")
        reply = self.supervisor.request(
            self.env_id, {"type": "arbitrate", "keys": ["k"]})
        for field_name in ("env_id", "seq_low", "seq_high",
                           "authority_digest", "elapsed_ms"):
            self.assertIn(field_name, reply)
        self.assertEqual(reply["env_id"], self.env_id)
        self.assertEqual((reply["seq_low"], reply["seq_high"]), (1, 2))
        self.assertIsInstance(reply["elapsed_ms"], int)

    def test_port_occupied_raises_e_port(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", self.port + 1))
        holder.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                self.supervisor.start("env-busy", "127.0.0.1", self.port + 1)
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            holder.close()

    def test_handshake_timeout_raises_e_timeout(self):
        # Inject a real subprocess that stays alive (port probe passes) but
        # never prints READY. The supervisor's monotonic readiness deadline
        # raises E_TIMEOUT without any artificial network delay.
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        slow = ServiceSupervisor(
            ready_timeout_s=0.4, io_timeout_s=0.4,
            process_factory=lambda *a, **k: proc)
        try:
            with self.assertRaises(TimeoutError) as ctx:
                slow.start("env-slow", "127.0.0.1", self.port + 2)
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            proc.kill()
            proc.wait()
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            slow.stop_all()
    def test_missing_reply_field_raises_e_reply(self):
        self.supervisor.start(self.env_id, "127.0.0.1", self.port)
        handle = self.supervisor.handles[self.env_id]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect((handle.host, handle.port))
        # Malformed service reply lacking all required fields.
        send_line(sock, {"type": "batch", "observations": []})
        # Service itself replies correctly; instead validate the guard directly.
        sock.close()
        with self.assertRaises(ReplyError) as ctx:
            from arbplatform.protocol import validate_reply
            validate_reply({"type": "arbitration_reply", "env_id": self.env_id})
        self.assertEqual(ctx.exception.code, "E_REPLY")
        self.assertIn("seq_low", ctx.exception.field)

    def test_io_timeout_raises_e_timeout(self):
        # A real listening socket that accepts but never replies; the client's
        # own socket timeout surfaces as E_TIMEOUT (no artificial delay).
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", self.port + 3))
        holder.listen(1)
        fast = ServiceSupervisor(ready_timeout_s=2.0, io_timeout_s=0.3)
        from arbplatform.client import ServiceHandle
        fast.handles["env-silent"] = ServiceHandle(
            "env-silent", "127.0.0.1", self.port + 3, process=None)
        try:
            with self.assertRaises(TimeoutError) as ctx:
                fast.request("env-silent",
                             {"type": "arbitrate", "keys": ["k"]})
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
        finally:
            holder.close()
            fast.stop_all()


if __name__ == "__main__":
    unittest.main()
