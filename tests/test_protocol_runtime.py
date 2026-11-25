from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from shard_diag.errors import DiagnosticError, ErrorCode
from shard_diag.protocol import WorkerClient
from shard_diag.runtime import DatabaseRuntime
from shard_diag.spec import DatabaseSpec

from conftest import free_port


class FakeServer:
    def __init__(self, handler):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = int(self.sock.getsockname()[1])
        self.handler = handler
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error: Exception | None = None

    def _run(self):
        try:
            connection, _ = self.sock.accept()
            with connection:
                connection.settimeout(2)
                self.handler(connection)
        except Exception as error:
            self.error = error
        finally:
            self.sock.close()

    def start(self):
        self.thread.start()
        return self

    def join(self):
        self.thread.join(timeout=3)
        if self.error:
            raise self.error


def _read_json(connection):
    data = b""
    while not data.endswith(b"\n"):
        chunk = connection.recv(1)
        if not chunk:
            raise EOFError
        data += chunk
    return json.loads(data)


def _send_json(connection, payload):
    connection.sendall((json.dumps(payload) + "\n").encode())


def _fake_database(port):
    return DatabaseSpec("fake", "127.0.0.1", port, 1, 1, 0, 2)


def test_real_worker_batch_receipt_and_restart(running_pool, spec_bundle):
    client = running_pool.client("db-a")
    probes = [
        {"event_seq": 1, "route_key": 210, "table": "orders", "route_version": 2, "sample_count": 2},
        {"event_seq": 2, "route_key": 45, "table": "audit", "route_version": 2, "sample_count": 2},
    ]
    results = client.probe_batch(probes, batch_id="protocol-test")
    assert len(results) == 2
    assert results[0]["db_id"] == "db-a"
    assert results[0]["hit_shards"] == sorted(results[0]["hit_shards"])
    assert results[0]["route_version"] == 2
    assert results[0]["elapsed_ms"] >= 0
    assert results[0]["sample_keys"][0] == 210

    runtime = running_pool.runtime("db-a")
    runtime.kill()
    runtime.start()
    result = runtime.client.probe_batch(
        [{"event_seq": 1, "route_key": 210, "table": "orders", "route_version": 2, "sample_count": 1}],
        batch_id="after-restart",
    )[0]
    assert result["db_id"] == "db-a"
    assert "orders:s1" in result["hit_shards"]


def test_port_in_use_reports_E_PORT(running_pool, spec_bundle):
    command = [
        sys.executable,
        "-u",
        "-m",
        "shard_diag.worker",
        "--spec",
        str(spec_bundle["current_path"]),
        "--db-id",
        "db-a",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        check=False,
    )
    assert completed.returncode != 0
    payloads = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    fatal = [payload for payload in payloads if payload.get("type") == "fatal"]
    assert fatal and fatal[0]["code"] == ErrorCode.PORT.value


def test_runtime_does_not_accept_an_unrelated_worker_on_the_same_port(spec_bundle):
    first = DatabaseRuntime(spec_bundle["current_path"], spec_bundle["spec"].database("db-a"))
    second = DatabaseRuntime(spec_bundle["current_path"], spec_bundle["spec"].database("db-a"))
    try:
        first.start()
        with pytest.raises(DiagnosticError) as caught:
            second.start()
        assert caught.value.code == ErrorCode.PORT.value
        assert "pid" in caught.value.message
        assert second.running is False
        assert first.running is True
    finally:
        second.stop(force=True)
        first.stop(force=True)


def test_handshake_timeout_is_real_socket_timeout():
    def handler(connection):
        time.sleep(0.4)

    server = FakeServer(handler).start()
    client = WorkerClient(_fake_database(server.port), timeout_seconds=0.1)
    with pytest.raises(DiagnosticError) as caught:
        client.ping()
    assert caught.value.code == ErrorCode.TIMEOUT.value
    server.join()


def test_missing_probe_receipt_field_is_E_PROTO():
    def handler(connection):
        hello = _read_json(connection)
        _send_json(connection, {"type": "hello_ack", "db_id": "fake", "protocol": 1})
        request = _read_json(connection)
        assert request["type"] == "probe_batch"
        _send_json(
            connection,
            {
                "ok": True,
                "type": "probe_batch_result",
                "results": [
                    {
                        "db_id": "fake",
                        "event_seq": 1,
                        "hit_shards": ["orders:s1"],
                        "route_version": 2,
                        # elapsed_ms is intentionally absent
                    }
                ],
            },
        )

    server = FakeServer(handler).start()
    client = WorkerClient(_fake_database(server.port), timeout_seconds=0.5)
    with pytest.raises(DiagnosticError) as caught:
        client.probe_batch(
            [{"event_seq": 1, "route_key": 1, "table": "orders", "route_version": 2}],
        )
    assert caught.value.code == ErrorCode.PROTO.value
    assert caught.value.field == "elapsed_ms"
    server.join()


def test_wrong_route_version_in_receipt_is_E_PROTO():
    def handler(connection):
        _read_json(connection)
        _send_json(connection, {"type": "hello_ack", "db_id": "fake", "protocol": 1})
        _read_json(connection)
        _send_json(
            connection,
            {
                "ok": True,
                "results": [
                    {
                        "db_id": "fake",
                        "event_seq": 1,
                        "hit_shards": ["orders:s1"],
                        "route_version": 1,
                        "elapsed_ms": 0.1,
                    }
                ],
            },
        )

    server = FakeServer(handler).start()
    client = WorkerClient(_fake_database(server.port), timeout_seconds=0.5)
    with pytest.raises(DiagnosticError) as caught:
        client.probe_batch([{"event_seq": 1, "route_key": 1, "table": "orders", "route_version": 2}])
    assert caught.value.code == ErrorCode.PROTO.value
    assert caught.value.field == "route_version"
    server.join()
