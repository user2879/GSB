from __future__ import annotations

import socket
import threading
import time

import pytest

from lease_arbiter.errors import TransportError
from lease_arbiter.spec import NodeSpec
from lease_arbiter.transport import JsonLineNodeClient, LiveNodeCluster

from .conftest import write_local_spec


def test_real_node_process_protocol_and_stale_label(tmp_path):
    spec = write_local_spec(tmp_path)
    with LiveNodeCluster(spec):
        client = JsonLineNodeClient(spec.connect_timeout_ms, spec.read_timeout_ms)
        alpha = spec.node_by_id["alpha"]
        granted = client.request(
            alpha,
            {"op": "grant", "lease_id": "a-cache-1", "lease_type": "cache", "expected_label": 0, "request_id": "g1"},
        )
        assert granted["ok"] is True
        assert granted["label"] == 1
        renewed = client.request(
            alpha,
            {"op": "renew", "lease_id": "a-cache-1", "expected_label": 1, "request_id": "r1"},
        )
        assert renewed["ok"] is True
        assert renewed["label"] == 2
        stale = client.request(
            alpha,
            {"op": "renew", "lease_id": "a-cache-1", "expected_label": 1, "request_id": "r2"},
        )
        assert stale["ok"] is False
        assert stale["code"] == "E_STALE"
        assert stale["current_label"] == 2
        snapshot = client.request(alpha, {"op": "snapshot", "request_id": "s1"})
        record = next(item for item in snapshot["leases"] if item["lease_id"] == "a-cache-1")
        assert record["label"] == 2
        assert record["state"] == "renewed"
        released = client.request(
            alpha,
            {"op": "release", "lease_id": "a-cache-1", "expected_label": 2, "request_id": "x1"},
        )
        assert released["ok"] is True
        assert released["label"] == 3


def test_occupied_loopback_port_reports_e_port(tmp_path):
    spec = write_local_spec(tmp_path)
    first = LiveNodeCluster(spec)
    first.start()
    second = LiveNodeCluster(spec)
    try:
        with pytest.raises(TransportError) as caught:
            second.start()
        assert caught.value.code == "E_PORT"
    finally:
        second.stop()
        first.stop()


def test_socket_read_timeout_reports_e_timeout():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    def slow_server() -> None:
        try:
            conn, _ = listener.accept()
            with conn:
                conn.recv(4096)
                time.sleep(0.5)
        finally:
            listener.close()

    thread = threading.Thread(target=slow_server, daemon=True)
    thread.start()
    client = JsonLineNodeClient(connect_timeout_ms=200, read_timeout_ms=80)
    node = NodeSpec("slow", "127.0.0.1", port, 1)
    try:
        with pytest.raises(TransportError) as caught:
            client.request(node, {"op": "grant", "lease_id": "x", "lease_type": "cache", "expected_label": 0})
        assert caught.value.code == "E_TIMEOUT"
    finally:
        thread.join(timeout=1.0)
