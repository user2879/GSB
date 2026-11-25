from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from speaker_diag.engine import make_batch_payloads
from speaker_diag.errors import DiagnosticError
from speaker_diag.orchestration import WorkerProcess, _connect_loopback
from speaker_diag.protocol import LineReader, perform_handshake
from speaker_diag.receipts import context_for, normalize_receipt
from speaker_diag.spec import dump_workload
from speaker_diag.synthetic import synthesize

from tests.helpers import tiny_spec


class LoopbackTests(unittest.TestCase):
    def test_real_subprocess_and_loopback_socket_batch(self) -> None:
        spec = tiny_spec()
        bundle = synthesize(spec)
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "spec.json"
            spec_path.write_text(dump_workload(spec), encoding="utf-8")
            worker = spec.workers[0]
            process = WorkerProcess(spec, worker, spec_path).start()
            try:
                payload = make_batch_payloads(spec, bundle)[0]
                receipt = process.request_batch(payload)
                normalized = normalize_receipt(receipt, context=context_for(spec, "B001"))
                self.assertEqual(normalized.batch_id, "B001")
                self.assertEqual(normalized.data["worker_id"], "W01")
                self.assertEqual(normalized.data["participant_window_ids"], ["W0001", "W0002"])
                self.assertIsInstance(receipt["elapsed_ms"], int)
                self.assertGreaterEqual(receipt["elapsed_ms"], 0)
            finally:
                process.stop()

    def test_occupied_port_is_e_port(self) -> None:
        spec = tiny_spec()
        worker = spec.workers[0]
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", worker.port))
        listener.listen(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "spec.json"
                path.write_text(dump_workload(spec), encoding="utf-8")
                with self.assertRaises(DiagnosticError) as caught:
                    WorkerProcess(spec, worker, path).start()
            self.assertEqual(caught.exception.code, "E_PORT")
            self.assertEqual(caught.exception.field, "workers[0].port")
        finally:
            listener.close()

    def test_handshake_timeout_is_real_socket_wait(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        stop = threading.Event()

        def server() -> None:
            try:
                connection, _ = listener.accept()
            except OSError:
                return
            try:
                stop.wait(1.0)
            finally:
                connection.close()
                listener.close()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        start = time.monotonic()
        try:
            with self.assertRaises(DiagnosticError) as caught:
                perform_handshake(
                    sock,
                    LineReader(sock),
                    expected_worker_id="W01",
                    expected_param_version=1,
                    expected_param_digest="0" * 64,
                    timeout_ms=70,
                )
            elapsed = time.monotonic() - start
            self.assertEqual(caught.exception.code, "E_HANDSHAKE")
            self.assertEqual(caught.exception.details["timeout_source"], "socket.recv")
            self.assertGreaterEqual(elapsed, 0.05)
        finally:
            sock.close()
            stop.set()
            thread.join(timeout=1.0)

    def test_loopback_client_never_self_connects(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        accepted: list[int] = []

        def server() -> None:
            connection, peer = listener.accept()
            accepted.append(int(peer[1]))
            connection.close()
            listener.close()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        client = _connect_loopback("127.0.0.1", port, 0.5)
        client_port = int(client.getsockname()[1])
        try:
            self.assertNotEqual(client_port, port)
            self.assertEqual(int(client.getpeername()[1]), port)
        finally:
            client.close()
            thread.join(timeout=1.0)
        self.assertEqual(accepted, [client_port])


if __name__ == "__main__":
    unittest.main()
