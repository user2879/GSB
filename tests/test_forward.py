"""Real loopback forwarding tests (upstream child processes)."""

import socket
import subprocess
import sys
import time
import unittest

from workbench import E_RECEIPT, E_TIMEOUT, E_UPSTREAM, OK
from workbench.forward import forward_request

PORT_OK = 24201
PORT_DEAD = 24299


class ForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = subprocess.Popen(
            [sys.executable, "-m", "workbench.upstream",
             "--port", str(PORT_OK), "--service", "svc-t"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                conn = socket.create_connection(("127.0.0.1", PORT_OK),
                                                timeout=0.2)
                conn.close()
                break
            except OSError:
                time.sleep(0.02)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=5)

    def test_roundtrip_ok(self):
        result = forward_request("127.0.0.1", PORT_OK, 5, "svc-t", "c1")
        self.assertEqual(result.code, OK)
        self.assertIn("seq=5", result.summary)
        self.assertIn("svc=svc-t", result.summary)

    def test_upstream_unreachable(self):
        result = forward_request("127.0.0.1", PORT_DEAD, 1, "svc-t", "c1",
                                 timeout_s=0.5)
        self.assertEqual(result.code, E_UPSTREAM)

    def test_real_socket_timeout(self):
        start = time.monotonic()
        result = forward_request("127.0.0.1", PORT_OK, 2, "svc-t",
                                 "__timeout__", timeout_s=1.0)
        elapsed = time.monotonic() - start
        self.assertEqual(result.code, E_TIMEOUT)
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 5.0)

    def test_missing_receipt_fields(self):
        result = forward_request("127.0.0.1", PORT_OK, 3, "svc-t",
                                 "__badreceipt__", timeout_s=1.0)
        self.assertEqual(result.code, E_RECEIPT)

    def test_receipt_service_mismatch(self):
        result = forward_request("127.0.0.1", PORT_OK, 4, "svc-t",
                                 "__wrongsvc__", timeout_s=1.0)
        self.assertEqual(result.code, E_RECEIPT)


if __name__ == "__main__":
    unittest.main()
