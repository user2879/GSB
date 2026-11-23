from __future__ import annotations

import socket
import sys
import unittest

from gateway_diag.backends import BackendProcess, validate_receipt
from gateway_diag.errors import HandshakeError, PortError, ReceiptError
from gateway_diag.model import parse_spec_obj
from gateway_diag.parameters import generate_parameters, infer_request

from tests.helpers import base_spec, reserve_ports


class BackendProcessTests(unittest.TestCase):
    def test_real_loopback_worker_handshake_receipt_and_restart(self) -> None:
        ports = reserve_ports(2)
        spec = parse_spec_obj(base_spec(ports))
        backend = spec.backends[0]
        parameters = generate_parameters(backend)
        process = BackendProcess(backend, parameters, startup_timeout=5, receipt_timeout=5)
        process.start()
        try:
            receipt = process.infer(
                "batch-001",
                [(1, (0.1, -0.2, 0.3, 0.4)), (2, (0.4, 0.3, -0.2, 0.1))],
            )
            self.assertEqual(receipt.backend_id, "stable")
            self.assertEqual(receipt.parameter_digest, parameters.parameter_digest)
            self.assertEqual(set(receipt.request_ids), {1, 2})
            self.assertEqual(len(receipt.output_digest), 64)
            expected_first = infer_request(parameters, 1, (0.1, -0.2, 0.3, 0.4))[1]
            self.assertEqual(receipt.output_digests[1], expected_first)
        finally:
            process.close()

        restarted = BackendProcess(backend, parameters, startup_timeout=5, receipt_timeout=5)
        restarted.start()
        try:
            receipt = restarted.infer("batch-002", [(3, (0.2, 0.3, 0.4, 0.5))])
            self.assertEqual(receipt.request_ids, (3,))
        finally:
            restarted.close()

    def test_occupied_port_reports_e_port_with_backend_and_port(self) -> None:
        port = reserve_ports(1)[0]
        occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupant.bind(("127.0.0.1", port))
        occupant.listen(1)
        try:
            spec = parse_spec_obj(base_spec([port, reserve_ports(1)[0]]))
            backend = spec.backends[0]
            process = BackendProcess(
                backend,
                generate_parameters(backend),
                startup_timeout=3,
                receipt_timeout=3,
            )
            with self.assertRaises(PortError) as caught:
                process.start()
            self.assertEqual(caught.exception.code, "E_PORT")
            self.assertEqual(caught.exception.backend_id, "stable")
            self.assertEqual(caught.exception.port, port)
        finally:
            occupant.close()

    def test_missing_ready_line_reports_e_handshake(self) -> None:
        port = reserve_ports(1)[0]
        spec = parse_spec_obj(base_spec([port, reserve_ports(1)[0]]))
        backend = spec.backends[0]
        process = BackendProcess(
            backend,
            generate_parameters(backend),
            startup_timeout=0.2,
            receipt_timeout=1,
            worker_command=[sys.executable, "-c", "import time; time.sleep(1)"],
        )
        with self.assertRaises(HandshakeError) as caught:
            process.start()
        self.assertEqual(caught.exception.code, "E_HANDSHAKE")
        self.assertEqual(caught.exception.backend_id, "stable")
        self.assertEqual(caught.exception.port, port)

    def test_receipt_missing_field_reports_e_receipt(self) -> None:
        spec = parse_spec_obj(base_spec(reserve_ports(2)))
        backend = spec.backends[0]
        parameters = generate_parameters(backend)
        raw = {
            "type": "receipt",
            "backend_id": backend.id,
            "parameter_digest": parameters.parameter_digest,
            "batch_id": "batch-x",
            "request_ids": [1],
            "output_digests": {"1": "a" * 64},
            "elapsed_ms": 1,
            "worker_pid": 123,
        }
        with self.assertRaises(ReceiptError) as caught:
            validate_receipt(
                raw,
                backend,
                "batch-x",
                [1],
                parameters.parameter_digest,
            )
        self.assertEqual(caught.exception.code, "E_RECEIPT")
        self.assertEqual(caught.exception.field, "output_digest")
        self.assertEqual(caught.exception.backend_id, "stable")
        self.assertEqual(caught.exception.port, backend.port)


if __name__ == "__main__":
    unittest.main()
