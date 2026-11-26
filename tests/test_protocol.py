from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from hopledger.errors import FieldError, PortInUseError, TimeoutError as HopTimeout
from hopledger.proc import ProcessManager
from hopledger.protocol import PROTOCOL_NAME, call_jsonl, encode_line, validate_receipt
from hopledger.sample import make_sample_configs


def test_handshake_timeout_comes_from_real_socket():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    accepted = threading.Event()
    release = threading.Event()

    def silent_accept() -> None:
        conn, _ = server.accept()
        accepted.set()
        release.wait(2.0)
        conn.close()

    thread = threading.Thread(target=silent_accept, daemon=True)
    thread.start()
    start = time.monotonic()
    with pytest.raises(HopTimeout) as exc_info:
        from hopledger.protocol import handshake
        handshake(port, timeout_seconds=0.2)
    elapsed = time.monotonic() - start
    release.set()
    thread.join(timeout=1)
    server.close()
    assert exc_info.value.code == "E_TIMEOUT"
    assert exc_info.value.details["field"] == "handshake"
    assert elapsed >= 0.18


def test_missing_receipt_field_is_efield():
    receipt = {
        "schema": "hopledger.receipt/1",
        "type": "hop_receipt",
        "hop_id": "fwd1",
        "node_id": "fwd1",
        "request_id": "req-1",
        "config_version": 1,
        "scale_factor": 1,
        # monotonic_ms intentionally missing.
    }
    with pytest.raises(FieldError) as exc_info:
        validate_receipt(receipt)
    assert exc_info.value.code == "E_FIELD"
    assert exc_info.value.field == "monotonic_ms"


def test_occupied_port_is_eport(tmp_path: Path):
    bundle = make_sample_configs(tmp_path / "config")
    port = bundle.chain.hop("fwd1").port
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", port))
    occupied.listen(1)
    manager = ProcessManager(bundle, tmp_path / "logs")
    try:
        with pytest.raises(PortInUseError) as exc_info:
            manager.start_node("fwd1")
        assert exc_info.value.code == "E_PORT"
        assert exc_info.value.details["port"] == port
    finally:
        manager.stop_all()
        occupied.close()


def test_missing_receipt_field_over_socket_is_efield():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def malformed_server() -> None:
        conn, _ = server.accept()
        conn.sendall(encode_line({
            "type": "hello_ack",
            "protocol": PROTOCOL_NAME,
        }))
        conn.recv(4096)
        conn.sendall(encode_line({
            "type": "hop_receipt",
            "ok": True,
            "receipt": {
                "schema": "hopledger.receipt/1",
                "type": "hop_receipt",
                "hop_id": "fwd1",
                "node_id": "fwd1",
                "request_id": "req-1",
                "config_version": 1,
                "scale_factor": 1,
            },
        }))
        conn.close()

    thread = threading.Thread(target=malformed_server, daemon=True)
    thread.start()
    try:
        with pytest.raises(FieldError) as exc_info:
            call_jsonl(port=port, message={"request_id": "req-1", "node_id": "fwd1"},
                       timeout_seconds=1.0)
        assert exc_info.value.code == "E_FIELD"
        assert exc_info.value.field == "monotonic_ms"
    finally:
        thread.join(timeout=1)
        server.close()
