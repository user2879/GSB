from __future__ import annotations

import socket

from lease_arbitrator.cluster import NodeCluster


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def test_node_generation_cas_and_e_stale(tmp_path) -> None:
    node = {
        "node_id": "n1",
        "host": "127.0.0.1",
        "port": free_port(),
        "quota": 4,
        "max_active": 2,
        "seed": 123,
        "quota_version": 1,
    }
    with NodeCluster([node], timeout=1.0, start_timeout=5.0, log_dir=tmp_path / "logs") as cluster:
        applied = cluster.call(
            "n1",
            {"type": "apply", "lease_id": "L1", "units": 2, "generation": 1},
        )
        assert applied["ok"] is True
        assert applied["generation_after"] == 1

        renewed = cluster.call(
            "n1",
            {"type": "renew", "lease_id": "L1", "generation": 1},
        )
        assert renewed["ok"] is True
        assert renewed["generation_before"] == 1
        assert renewed["generation_after"] == 2

        stale = cluster.call(
            "n1",
            {"type": "renew", "lease_id": "L1", "generation": 1},
        )
        assert stale["ok"] is False
        assert stale["error"] == "E_STALE"
        assert stale["current_generation"] == 2

        valid_after_stale = cluster.call(
            "n1",
            {"type": "renew", "lease_id": "L1", "generation": 2},
        )
        assert valid_after_stale["ok"] is True
        assert valid_after_stale["generation_after"] == 3

        released = cluster.call(
            "n1",
            {"type": "release", "lease_id": "L1", "generation": 3},
        )
        assert released["ok"] is True
        assert released["generation_after"] == 4


def test_node_rejects_forward_generation_on_new_lease(tmp_path) -> None:
    node = {
        "node_id": "n1",
        "host": "127.0.0.1",
        "port": free_port(),
        "quota": 4,
        "max_active": 2,
        "seed": 124,
        "quota_version": 1,
    }
    with NodeCluster([node], timeout=1.0, log_dir=tmp_path / "logs") as cluster:
        response = cluster.call(
            "n1",
            {"type": "apply", "lease_id": "L2", "units": 1, "generation": 3},
        )
        assert response["ok"] is False
        assert response["error"] == "E_GEN"
