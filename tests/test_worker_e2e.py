import socket
import unittest

from batch_receipt_ledger.errors import PortError, TimeoutError_
from batch_receipt_ledger.processes import ManagedWorker
from batch_receipt_ledger.protocol import WorkerClient
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class WorkerE2ETest(unittest.TestCase):
    def test_real_loopback_inference_and_handshake(self):
        raw = make_spec(workers=1, deliveries=3)
        spec = parse_spec(raw)
        with ManagedWorker(spec.workers[0], raw) as worker:
            with worker.client(2000, 2000) as client:
                pong = client.request({"type": "ping"})
                self.assertEqual(pong["type"], "pong")
                ack = client.request({
                    "type": "batch",
                    "batch_id": "b-test",
                    "param_version": 1,
                    "items": [
                        {"request_id": "req-001", "value": 1001},
                        {"request_id": "req-002", "value": 1002},
                    ],
                })
        self.assertEqual(ack["type"], "batch_ack")
        self.assertEqual(len(ack["results"]), 2)
        self.assertGreaterEqual(ack["elapsed_ms"], 0)
        self.assertTrue(all(len(item["output_digest"]) == 32
                            for item in ack["results"]))

    def test_port_conflict_reports_e_port(self):
        raw = make_spec(workers=1)
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind(("127.0.0.1", raw["workers"][0]["port"]))
        blocker.listen(1)
        try:
            spec = parse_spec(raw)
            with self.assertRaises(PortError) as ctx:
                ManagedWorker(spec.workers[0], raw,
                              ready_timeout_ms=3000).start()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            blocker.close()

    def test_connect_timeout_vs_dead_port(self):
        raw = make_spec(workers=1)
        spec = parse_spec(raw)
        with self.assertRaises(PortError):
            with WorkerClient(spec.workers[0].port, 200, 1000):
                pass

    def test_read_timeout_e_timeout(self):
        raw = make_spec(workers=1)
        spec = parse_spec(raw)
        with ManagedWorker(spec.workers[0], raw):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("127.0.0.1", spec.workers[0].port))
            sock.recv(1024)
            from batch_receipt_ledger.protocol import recv_line
            with self.assertRaises(TimeoutError_) as ctx:
                recv_line(sock, 50, location="test")
            self.assertEqual(ctx.exception.code, "E_TIMEOUT")
            sock.close()


if __name__ == "__main__":
    unittest.main()
