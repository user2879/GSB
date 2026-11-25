import socket
import threading
import time

import pytest

from lease_middleware.cluster import NodeCluster
from lease_middleware.errors import LeaseError, PortError, ReplyError, TimeoutError
from lease_middleware.protocol import LeaseNodeClient


@pytest.fixture()
def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_real_subprocess_tag_matching(tmp_path, free_port):
    node = {
        "node_id": "unit-node",
        "host": "127.0.0.1",
        "port": free_port,
        "lease_limit": 2,
        "in_flight_limit": 1,
        "node_generation": 1,
    }
    cluster = NodeCluster([node], log_dir=tmp_path / "logs", handshake_timeout=1.0)
    cluster.start_all()
    try:
        client = cluster.clients["unit-node"]
        granted = client.grant("lease-x", "equity-x", 1000)
        assert granted["current_tag"] == 1

        with pytest.raises(LeaseError) as exc:
            client.renew("lease-x", 99)
        assert exc.value.details["current_tag"] == 1

        renewed = client.renew("lease-x", 1)
        assert renewed["current_tag"] == 2
        with pytest.raises(LeaseError) as exc:
            client.release("lease-x", 1)
        assert exc.value.details["current_tag"] == 2
        released = client.release("lease-x", 2)
        assert released["current_tag"] == 3
    finally:
        cluster.stop_all()


def test_port_occupation_reports_e_port(tmp_path, free_port):
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", free_port))
    holder.listen(1)
    node = {
        "node_id": "busy-node",
        "host": "127.0.0.1",
        "port": free_port,
        "lease_limit": 1,
        "in_flight_limit": 1,
    }
    cluster = NodeCluster([node], log_dir=tmp_path / "logs")
    try:
        with pytest.raises(PortError) as exc:
            cluster.start_all()
        assert exc.value.code == "E_PORT"
    finally:
        cluster.stop_all()
        holder.close()


def _raw_reply_server(port: int, mode: str) -> None:
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)

    def serve():
        conn, _ = server.accept()
        with conn:
            conn.recv(4096)
            if mode == "timeout":
                time.sleep(1)
            else:
                conn.sendall(b'{"ok":true}\n')

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()


def test_handshake_timeout_and_missing_reply_field(free_port):
    _raw_reply_server(free_port, "timeout")
    client = LeaseNodeClient("127.0.0.1", free_port, timeout=0.2)
    with pytest.raises(TimeoutError) as exc:
        client.handshake()
    assert exc.value.code == "E_TIMEOUT"

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    reply_port = sock.getsockname()[1]
    sock.close()
    _raw_reply_server(reply_port, "reply")
    client = LeaseNodeClient("127.0.0.1", reply_port, timeout=1)
    with pytest.raises(ReplyError) as exc:
        client.handshake()
    assert exc.value.code == "E_REPLY"
    assert "reply.node_id" in exc.value.field
