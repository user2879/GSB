from __future__ import annotations

import socket
import subprocess
import sys
import threading
import unittest
from decimal import Decimal
from types import MappingProxyType

from config_arbiter.client import ServiceProcess
from config_arbiter.errors import ProtocolError
from config_arbiter.spec import EnvironmentSpec, SliceSpec

from .util import free_port, session_spec


def environment(port: int, env_id: str = "env-a") -> EnvironmentSpec:
    return EnvironmentSpec(
        id=env_id,
        host="127.0.0.1",
        port=port,
        weight_text="1",
        weight=Decimal(1),
        weight_units=1000,
        weight_error_units=Decimal(0),
        slices=(SliceSpec(version=1, values=MappingProxyType({"x": "1.2345"})),),
    )


class OneShotServer:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(1)
        self.port = int(self.socket.getsockname()[1])
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            connection, _ = self.socket.accept()
            with connection:
                self.handler(connection)
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass


class ServiceProtocolTests(unittest.TestCase):
    def test_real_subprocess_receipt_is_deterministic(self) -> None:
        port = free_port()
        process = ServiceProcess(environment(port), seed=19)
        try:
            process.start(1500)
            session = session_spec()
            first = process.send_slice(process.environment.current_slice, session)
            second = process.send_slice(process.environment.current_slice, session)
            self.assertEqual(first.service_id, "env-a")
            self.assertEqual(first.slice_version, 1)
            self.assertEqual(first.aggregate_summary["x"], 1234)
            self.assertEqual(first.sample_digest, second.sample_digest)
            self.assertEqual(first.digest, second.digest)
            self.assertGreaterEqual(first.elapsed_ms_monotonic, 0)
        finally:
            process.stop()

    def test_port_conflict_maps_to_e_port(self) -> None:
        port = free_port()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(1)
            process = ServiceProcess(environment(port), seed=5)
            with self.assertRaises(ProtocolError) as caught:
                process.start(1500)
            self.assertEqual(caught.exception.code, "E_PORT")
        finally:
            listener.close()

    def test_socket_handshake_timeout_maps_to_e_timeout(self) -> None:
        release = threading.Event()

        def handler(connection: socket.socket) -> None:
            connection.recv(4096)
            release.wait(2.0)

        server = OneShotServer(handler)
        dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        process = ServiceProcess(environment(server.port), seed=1)
        process.process = dummy
        try:
            with self.assertRaises(ProtocolError) as caught:
                process.send_slice(process.environment.current_slice, session_spec(handshake_timeout_ms=120))
            self.assertEqual(caught.exception.code, "E_TIMEOUT")
        finally:
            release.set()
            server.close()
            process.stop()

    def test_missing_receipt_field_maps_to_e_field(self) -> None:
        def handler(connection: socket.socket) -> None:
            connection.recv(4096)
            connection.sendall(b'{"service_id":"env-a"}\n')

        server = OneShotServer(handler)
        dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        process = ServiceProcess(environment(server.port), seed=1)
        process.process = dummy
        try:
            with self.assertRaises(ProtocolError) as caught:
                process.send_slice(process.environment.current_slice, session_spec())
            self.assertEqual(caught.exception.code, "E_FIELD")
            self.assertEqual(caught.exception.field, "response.slice_version")
        finally:
            server.close()
            process.stop()

    def test_connection_refused_maps_to_e_port(self) -> None:
        port = free_port()
        dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
        process = ServiceProcess(environment(port), seed=1)
        process.process = dummy
        try:
            with self.assertRaises(ProtocolError) as caught:
                process.send_slice(process.environment.current_slice, session_spec())
            self.assertEqual(caught.exception.code, "E_PORT")
        finally:
            process.stop()


if __name__ == "__main__":
    unittest.main()
