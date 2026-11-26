from __future__ import annotations

import socket
import threading
import time

import pytest

from waterlease.errors import LeaseConflictError, PortInUseError, RpcTimeoutError
from waterlease.leases import LeaseManager
from waterlease.node_worker import NodeContext, NodeRpcServer
from waterlease.process_manager import ProcessManager
from waterlease.protocol import rpc
from waterlease.spec import NodeDecl
from tests.support import free_port


def make_node(port: int | None = None) -> NodeDecl:
    return NodeDecl(
        node_id="node-test",
        host="127.0.0.1",
        port=port or free_port(),
        seed=4242,
        concurrency_quota=2,
        max_leases=2,
        fixed_point_scale=100,
        processing_delay_ms=0,
    )


def test_port_binding_conflict_reports_e_port() -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = free_port()
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        context = NodeContext("node-test", "127.0.0.1", port, 1)
        with pytest.raises(PortInUseError) as exc:
            NodeRpcServer(context)
        assert exc.value.code == "E_PORT"
    finally:
        blocker.close()


def test_read_timeout_reports_e_timeout() -> None:
    port = free_port()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)

    def accept_and_hold() -> None:
        try:
            conn, _ = server.accept()
            time.sleep(1.0)
            conn.close()
        except OSError:
            pass

    thread = threading.Thread(target=accept_and_hold, daemon=True)
    thread.start()
    try:
        with pytest.raises(RpcTimeoutError) as exc:
            rpc("127.0.0.1", port, {"type": "ping"}, timeout_ms=80)
        assert exc.value.code == "E_TIMEOUT"
    finally:
        server.close()


def test_lease_optimistic_version_and_real_expiry() -> None:
    node = make_node()
    with ProcessManager([node], startup_timeout_s=8) as processes:
        leases = LeaseManager(processes, ttl_ms=120)
        lease = leases.issue(
            shard_id="shard-a",
            node_id=node.node_id,
            capacity_version="cap-v1",
        )
        original_expiry = lease.expires_at_ns
        with pytest.raises(LeaseConflictError) as exc:
            leases.renew(lease.lease_id, expected_version=999)
        assert exc.value.code == "E_LEASE"
        assert lease.version == 1
        assert lease.expires_at_ns == original_expiry

        renewed = leases.renew(lease.lease_id, expected_version=1)
        assert renewed.version == 2
        assert renewed.renew_count == 1

        short = leases.issue(
            shard_id="shard-expire",
            node_id=node.node_id,
            capacity_version="cap-v1",
        )
        short.ttl_ms = 40
        short.expires_at_ns = time.monotonic_ns() + 40_000_000
        time.sleep(0.08)
        expired = leases.expire_due()
        assert any(item.lease_id == short.lease_id for item in expired)
        with pytest.raises(LeaseConflictError):
            leases.renew(short.lease_id, expected_version=short.version)
