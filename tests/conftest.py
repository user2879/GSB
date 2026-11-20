from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def free_ports():
    ports = [_free_port() for _ in range(4)]
    return ports


@pytest.fixture
def node_specs(free_ports):
    return [
        {"id": "node-a", "host": "127.0.0.1", "port": free_ports[0], "faults": {}},
        {"id": "node-b", "host": "127.0.0.1", "port": free_ports[1], "faults": {}},
        {"id": "node-c", "host": "127.0.0.1", "port": free_ports[2], "faults": {}},
    ]


@pytest.fixture
def v3_manifest():
    return {
        "format_version": 3,
        "entries": [
            {"key": "a", "value": 10, "node": "node-a", "config_version": 3,
             "strategy": {"mode": "targeted", "retries": 2, "timeout_ms": 1000}},
            {"key": "b", "value": 20, "node": "node-a", "config_version": 3,
             "strategy": {"mode": "targeted", "retries": 2, "timeout_ms": 1000}},
            {"key": "c", "value": 2, "node": "node-b", "config_version": 3,
             "strategy": {"mode": "targeted", "retries": 1, "timeout_ms": 1000}},
            {"key": "d", "value": 6, "node": "node-b", "config_version": 3,
             "strategy": {"mode": "targeted", "retries": 1, "timeout_ms": 1000}},
            {"key": "shared", "value": 1, "node": "node-c", "config_version": 3,
             "strategy": {"mode": "broadcast", "retries": 2, "timeout_ms": 1000}},
        ],
        "derivations": [
            {"name": "ab-sum", "inputs": ["a", "b"], "op": "sum", "target_node": "node-a"},
            {"name": "cd-mean", "inputs": ["c", "d"], "op": "mean", "target_node": "node-b"},
        ],
    }


@pytest.fixture
def write_manifest(tmp_path):
    def _write(payload: dict, name: str = "manifest.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def write_nodes(tmp_path):
    def _write(nodes: list[dict], name: str = "nodes.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")
        return path

    return _write
