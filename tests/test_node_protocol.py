from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from cachelease.client import NodeClient, NodePool
from cachelease.content import generate_content
from cachelease.errors import HandshakeError, ProtocolError, ReplyError
from cachelease.node import mono_ms
from cachelease.spec import NodeSpec, load_config

from .helpers import free_port, one_node_config


def test_real_node_reserve_commit_expiry_and_failed_request_no_leak(tmp_path: Path) -> None:
    raw, config_path = one_node_config(tmp_path, capacity_slots=1, lease_limit=2, grace_ms=60)
    config = load_config(config_path)
    with NodePool(config, tmp_path / "nodes") as pool:
        client = pool.get("node-a")
        reserved = client.reserve("core", 5_000, 1, 60, 1)
        status = client.status()
        assert status["watermark"]["used_slots"] == 1
        assert status["watermark"]["active_leases"] == 0
        assert status["watermark"]["lease_remaining"] == 1
        committed = client.commit(reserved["reservation_id"])
        expected = generate_content("node-a", "core", raw["nodes"][0]["seed"], 2, 32)
        assert committed["digest"] == expected.digest
        assert client.status()["watermark"]["used_slots"] == 1
        assert client.status()["watermark"]["active_leases"] == 1

        with pytest.raises(ProtocolError) as exc:
            client.reserve("api", 5_000, 2, 60, 1)
        assert exc.value.code == "E_CAPACITY"
        after_failure = client.status()
        assert after_failure["watermark"]["used_slots"] == 1
        assert after_failure["watermark"]["active_leases"] == 1

        client.release(committed["lease_id"])
        expired = client.reserve("core", 5_000, 3, 50, 1)
        time.sleep(0.12)
        status = client.status()
        assert status["watermark"]["used_slots"] == 0
        assert status["watermark"]["active_leases"] == 0
        assert status["reservations"] == []
        with pytest.raises(ProtocolError) as exc:
            client.commit(expired["reservation_id"])
        assert exc.value.code == "E_EXPIRED"
        assert client.status()["watermark"]["used_slots"] == 0


def test_quota_renew_and_revoke_dispositions(tmp_path: Path) -> None:
    raw, config_path = one_node_config(tmp_path, capacity_slots=4, lease_limit=4)
    config = load_config(config_path)
    with NodePool(config, tmp_path / "nodes") as pool:
        client = pool.get("node-a")
        first = client.reserve("core", 5_000, 1, 1_000, 1)
        first_lease = client.commit(first["reservation_id"])
        second = client.reserve("api", 5_000, 2, 1_000, 1)
        second_lease = client.commit(second["reservation_id"])

        renewed = client.set_quota(2, 4, 4, "renew")
        statuses = {item["status"] for item in renewed["lease_dispositions"]}
        assert statuses == {"renewed"}
        status = client.status()
        assert status["quota_version"] == 2
        assert len(status["leases"]) == 2
        old_ids = {first_lease["lease_id"], second_lease["lease_id"]}
        assert old_ids.isdisjoint({item["lease_id"] for item in status["leases"]})

        revoked = client.set_quota(3, 4, 4, "revoke")
        assert all(item["status"] == "revoked" for item in revoked["lease_dispositions"])
        status = client.status()
        assert status["leases"] == []
        assert status["watermark"]["used_slots"] == 0

        pending = client.reserve("core", 5_000, 9, 1_000, 3)
        assert client.status()["reservations"]
        replaced = client.set_quota(4, 4, 4, "renew")
        assert replaced["reservation_dispositions"][0]["reservation_id"] == pending["reservation_id"]
        assert replaced["reservation_dispositions"][0]["status"] == "rolled_back"
        assert client.status()["reservations"] == []
        with pytest.raises(ProtocolError) as exc:
            client.commit(pending["reservation_id"])
        assert exc.value.code == "E_EXPIRED"


def test_handshake_missing_reply_fields_is_e_reply(tmp_path: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)
    port = int(listener.getsockname()[1])

    def serve_once() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                json.dumps({"ok": True, "protocol_version": 1, "node_id": "node-a"}).encode("utf-8") + b"\n"
            )
        listener.close()

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    spec = NodeSpec("node-a", "127.0.0.1", port, 1, 1, 1, 1, 1)
    client = NodeClient(spec, timeout_ms=500)
    with pytest.raises(ReplyError) as exc:
        client.handshake(mono_ms() + 1_000)
    assert exc.value.code == "E_REPLY"
    thread.join(timeout=1.0)


def test_handshake_timeout_uses_real_socket_timeout() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    port = int(listener.getsockname()[1])
    stop = threading.Event()

    def accept_and_stall() -> None:
        listener.settimeout(0.05)
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            threading.Thread(target=_stall_connection, args=(conn, stop), daemon=True).start()
        listener.close()

    thread = threading.Thread(target=accept_and_stall, daemon=True)
    thread.start()
    spec = NodeSpec("node-a", "127.0.0.1", port, 1, 1, 1, 1, 1)
    client = NodeClient(spec, timeout_ms=100)
    started = mono_ms()
    with pytest.raises(HandshakeError):
        client.handshake(mono_ms() + 280)
    elapsed = mono_ms() - started
    stop.set()
    thread.join(timeout=1.0)
    assert elapsed >= 90


def _stall_connection(conn: socket.socket, stop: threading.Event) -> None:
    with conn:
        conn.settimeout(0.05)
        while not stop.is_set():
            try:
                conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
