from __future__ import annotations

import socket
import threading
import time

from leaseaudit.driver import MockSourceClient
from leaseaudit.engine import Lease
from leaseaudit.mock_service import DeterministicHTTPServer
from leaseaudit.scenario import validate_scenario


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _source(port, mode, processing_ms=0):
    scenario = validate_scenario(
        {
            "cluster_capacity": 2,
            "lease_duration_ms": 1000,
            "renew_interval_ms": 100,
            "reclaim_threshold": 1,
            "load_duration_ms": 1000,
            "sources": [
                {
                    "source_id": "source",
                    "address": "127.0.0.1",
                    "port": port,
                    "lease_share": 1,
                    "reserve_quota": 0,
                    "mode": mode,
                    "processing_ms": processing_ms,
                }
            ],
        }
    )
    return scenario.sources[0]


def test_success_and_fixed_delay_mock_sources():
    for mode, processing_ms in (("success", 0), ("fixed_delay", 20)):
        port = _free_port()
        server = DeterministicHTTPServer(("127.0.0.1", port), "source", mode, processing_ms)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            spec = _source(port, mode, processing_ms)
            client = MockSourceClient(spec)
            lease = Lease("lease-1", "source", 1, 1000, 9000)
            result = client.request(lease, 2000)
            assert result.outcome == "success"
            assert result.status_code == 200
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_refuse_mode_is_real_loopback_connection_refusal():
    port = _free_port()
    spec = _source(port, "refuse")
    client = MockSourceClient(spec)
    lease = Lease("lease-1", "source", 1, 1000, 9000)
    result = client.request(lease, 2000)
    assert result.outcome == "connection_refused"


def test_client_refuses_expired_lease_before_network_io():
    spec = _source(_free_port(), "success")
    client = MockSourceClient(spec)
    lease = Lease("lease-1", "source", 1, 1000, 2000)
    result = client.request(lease, 2000)
    assert result.outcome == "client_rejected_expired_lease"
    assert client.request_counter == 0
