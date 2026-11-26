import json
import socket
import threading

import pytest

from cacheflow.cluster import NodeHandle
from cacheflow.errors import FieldError, PortInUseError, ProtocolTimeout, StaleGenerationError
from cacheflow.protocol import JsonLineClient, encode_line
from cacheflow.spec import NodeSpec


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_raw_server(response: bytes | None = None, *, silent: bool = False):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        try:
            conn, _ = server.accept()
            with conn:
                if silent:
                    threading.Event().wait(1.0)
                elif response is not None:
                    conn.sendall(response)
                    threading.Event().wait(0.2)
        except OSError:
            pass
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, thread


def test_handshake_timeout_uses_real_socket_timeout():
    port, thread = start_raw_server(silent=True)
    client = JsonLineClient("127.0.0.1", port, node_id="node-x", timeout_seconds=0.2)
    with pytest.raises(ProtocolTimeout) as exc:
        client.connect()
        client.handshake()
    assert exc.value.code == "E_TIMEOUT"
    client.close()
    thread.join(timeout=2)


def test_missing_handshake_field_reports_e_field():
    response = encode_line({"type": "hello_ack", "protocol": 1})
    port, thread = start_raw_server(response)
    client = JsonLineClient("127.0.0.1", port, node_id="node-x", timeout_seconds=1.0)
    with pytest.raises(FieldError) as exc:
        client.connect()
        client.handshake()
    assert exc.value.code == "E_FIELD"
    assert exc.value.field == "node_id"
    client.close()
    thread.join(timeout=2)


def test_port_busy_reports_e_port():
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupier.bind(("127.0.0.1", 0))
    occupier.listen(1)
    port = occupier.getsockname()[1]
    spec = NodeSpec("node-busy", "127.0.0.1", port, 2, 1, "seed")
    handle = NodeHandle(spec, startup_timeout=2.0)
    try:
        with pytest.raises(PortInUseError) as exc:
            handle.start()
        assert exc.value.code == "E_PORT"
        assert exc.value.field == "port"
    finally:
        handle.kill()
        occupier.close()


def test_real_node_generation_tag_and_e_stale():
    port = free_port()
    spec = NodeSpec("node-gen", "127.0.0.1", port, 3, 2, "seed-gen")
    handle = NodeHandle(spec, startup_timeout=5.0).start()
    try:
        client = handle.client(timeout_seconds=3.0)
        grant = client.call("grant", lease_id="lease-1", artifact_key="art", ttl_ms=60000)
        assert grant["generation"] == 1
        renewed = client.call("renew", lease_id="lease-1", generation=1)
        assert renewed["generation"] == 2 and renewed["state"] == "renewed"
        with pytest.raises(StaleGenerationError) as exc:
            client.call("renew", lease_id="lease-1", generation=1)
        assert exc.value.code == "E_STALE"
        assert exc.value.current_generation == 2
        released = client.call("release", lease_id="lease-1", generation=2)
        assert released["generation"] == 3 and released["state"] == "released"
        client.close()

        status_client = handle.client(timeout_seconds=3.0)
        status = status_client.call("status")
        lease = next(row for row in status["leases"] if row["lease_id"] == "lease-1")
        assert lease["generation"] == 3 and lease["state"] == "released"
        status_client.close()
    finally:
        assert handle.terminate() in {0, -15}
