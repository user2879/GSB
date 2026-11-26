from __future__ import annotations

import threading

from reco_shard_diag.client import JsonLineClient
from reco_shard_diag.models import NodeSpec
from reco_shard_diag.node import NodeRuntime, NodeTcpServer


def test_real_tcp_json_lines_lease_expiry_and_duplicate_delivery():
    now = [1000]
    spec = NodeSpec("node-test", "127.0.0.1", 0, 3, 3)
    runtime = NodeRuntime(spec=spec, clock_ms=lambda: now[0])
    server = NodeTcpServer(("127.0.0.1", 0), runtime)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = JsonLineClient("127.0.0.1", port)
    try:
        assert client.request({"operation": "ping"})["ok"]
        first = client.request({"operation": "acquire", "shard_id": "s", "ttl_ms": 5})
        second = client.request({"operation": "acquire", "shard_id": "s", "ttl_ms": 30})
        assert first["lease_generation"] == 1
        assert second["lease_generation"] == 2
        aggregate = {
            "operation": "aggregate",
            "shard_id": "s",
            "batch_id": "b",
            "quota_version": 1,
            "width": 16,
            "dimensions": 2,
            "samples": [
                {"group": "g0", "values": [1, 2]},
                {"group": "g0", "values": [3, 4]},
            ],
        }
        response1 = client.request(
            {
                **aggregate,
                "request_id": "r1",
                "lease_id": first["lease_id"],
                "expected_generation": first["lease_generation"],
            }
        )
        response2 = client.request(
            {
                **aggregate,
                "request_id": "r2",
                "lease_id": second["lease_id"],
                "expected_generation": second["lease_generation"],
            }
        )
        assert response1["ok"]
        assert response2["ok"]
        assert response2["duplicate_delivery"] is True
        assert response2["supersedes_request_id"] == "r1"
        now[0] += 50
        sweep = client.request({"operation": "sweep"})
        expired_ids = {entry["lease_id"] for entry in sweep["expired"]}
        assert first["lease_id"] in expired_ids
        assert second["lease_id"] in expired_ids
        renew = client.request(
            {
                "operation": "renew",
                "lease_id": first["lease_id"],
                "expected_generation": first["lease_generation"],
                "ttl_ms": 10,
            }
        )
        assert renew["code"] == "E_LEASE_EXPIRED"
        assert renew["lease_generation"] == runtime.lease_generation
        reused = client.request({"operation": "acquire", "shard_id": "s2", "ttl_ms": 10})
        assert reused["ok"]
        assert sweep["active_work_units"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
