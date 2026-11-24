from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from tunnel_interop.spec import load_spec, load_spec_mapping


def free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def make_spec_data(
    *,
    route_count: int = 2,
    fragment_size: int = 256,
    max_retries: int = 0,
    timeout_ms: int = 150,
    sample_bytes: int = 650,
    global_limit: int = 3,
    per_tunnel_quota: int = 2,
    ack_timeout_ms: int = 250,
) -> dict[str, Any]:
    ports = [free_udp_port() for _ in range(route_count)]
    while len(set(ports)) != len(ports):
        ports = [free_udp_port() for _ in range(route_count)]
    endpoints = []
    routes = []
    for index in range(route_count):
        endpoint_id = f"ep-{index}"
        tunnel_id = f"tun-{index}"
        endpoints.append(
            {
                "id": endpoint_id,
                "host": "127.0.0.1",
                "port": ports[index],
                "mtu": 1400,
                "encapsulation_overhead": 80,
                "replay_window": 16,
                "seed": 1000 + index,
            }
        )
        routes.append(
            {
                "prefix": f"10.{50 + index}.0.0/16",
                "next_hop": endpoint_id,
                "priority": 100 + index,
                "tunnel_id": tunnel_id,
            }
        )
    return {
        "versions": {
            "declaration": 1,
            "endpoint_set": 1,
            "route_table": 1,
            "fragment_policy": 1,
        },
        "endpoints": endpoints,
        "routes": routes,
        "fragment_policy": {
            "fragment_size": fragment_size,
            "max_retries": max_retries,
            "reassembly_timeout_ms": timeout_ms,
        },
        "runtime": {
            "global_inflight_limit": global_limit,
            "per_tunnel_inflight_quota": per_tunnel_quota,
            "ack_timeout_ms": ack_timeout_ms,
            "fragment_format": "f3",
            "max_message_bytes": 8 * 1024 * 1024,
            "receiver_idle_timeout_ms": 3000,
        },
        "sample": {"bytes_per_message": sample_bytes},
    }


@pytest.fixture
def spec_data() -> dict[str, Any]:
    return make_spec_data()


@pytest.fixture
def spec(spec_data):
    return load_spec_mapping(spec_data, source_path="<test>")


def write_spec(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def load_written_spec(path: Path, data: dict[str, Any]):
    write_spec(path, data)
    return load_spec(path)


def clone(data: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(data)
