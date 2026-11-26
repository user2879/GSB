from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from workbench.errors import FieldError, PortError, TimeoutError as WorkbenchTimeout
from workbench.orchestrator import NodeOrchestrator, allocate_free_port
from workbench.protocol import call_node, handshake
from workbench.testkit import make_test_spec, write_test_spec


def _silent_server(port_holder: list[int]) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port_holder.append(server.getsockname()[1])
    conn, _ = server.accept()
    time.sleep(1.0)
    conn.close()
    server.close()


def test_handshake_timeout_uses_real_socket_timeout():
    holder: list[int] = []
    thread = threading.Thread(target=_silent_server, args=(holder,), daemon=True)
    thread.start()
    for _ in range(50):
        if holder:
            break
        time.sleep(0.01)
    with pytest.raises(WorkbenchTimeout) as exc:
        handshake("127.0.0.1", holder[0], "n1", 120)
    assert exc.value.code == "E_TIMEOUT"
    thread.join(timeout=2)


def _field_server(port_holder: list[int]) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port_holder.append(server.getsockname()[1])
    conn, _ = server.accept()
    reader = conn.makefile("rb")
    reader.readline()
    conn.sendall(json.dumps({"type": "hello_ack", "node_id": "n1"}).encode() + b"\n")
    reader.readline()
    conn.sendall(json.dumps({"ok": True, "request_id": "r1", "node_id": "n1"}).encode() + b"\n")
    time.sleep(0.1)
    conn.close()
    server.close()


def test_missing_receipt_field_is_e_field():
    holder: list[int] = []
    thread = threading.Thread(target=_field_server, args=(holder,), daemon=True)
    thread.start()
    for _ in range(50):
        if holder:
            break
        time.sleep(0.01)
    node = {"node_id": "n1", "host": "127.0.0.1", "port": holder[0]}
    with pytest.raises(FieldError) as exc:
        call_node(node, {"type": "aggregate_batch", "request_id": "r1", "shard_id": "s1"}, timeout_ms=1000)
    assert exc.value.code == "E_FIELD"
    assert exc.value.field.startswith("shard_id") or exc.value.field == "shard_id"
    thread.join(timeout=2)


def test_occupied_port_is_e_port(tmp_path):
    raw = make_test_spec(node_count=1)
    port = int(raw["nodes"][0]["port"])
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupier.bind(("127.0.0.1", port))
    occupier.listen(1)
    spec_path = tmp_path / "occupied.json"
    spec = write_test_spec(spec_path, raw)
    orchestrator = NodeOrchestrator(spec, tmp_path / "work")
    try:
        with pytest.raises(PortError) as exc:
            orchestrator.start_node("n1")
        assert exc.value.code == "E_PORT"
    finally:
        occupier.close()
        orchestrator.stop_all()
