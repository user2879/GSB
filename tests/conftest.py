from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from shard_diag.runtime import DatabasePool
from shard_diag.session import DiagnosticSession, session_identifier
from shard_diag.spec import Spec, load_spec, validate_spec


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_spec_values(route_version: int = 2, ports: tuple[int, int] | None = None) -> dict[str, Any]:
    if ports is None:
        ports = (free_port(), free_port())
    return {
        "format_version": 1,
        "global_lease_limit": 4,
        "databases": [
            {
                "id": "db-a",
                "host": "127.0.0.1",
                "port": ports[0],
                "lease_limit": 3,
                "concurrency": 2,
                "seed": 101,
                "route_version": route_version,
            },
            {
                "id": "db-b",
                "host": "127.0.0.1",
                "port": ports[1],
                "lease_limit": 2,
                "concurrency": 1,
                "seed": 202,
                "route_version": route_version,
            },
        ],
        "tables": [
            {
                "name": "orders",
                "key_column": "order_id",
                "route_version": 1,
                "boundaries": [0, 100, 200, 400, 800],
            },
            {
                "name": "orders",
                "key_column": "order_id",
                "route_version": 2,
                "boundaries": [0, 80, 220, 450, 800],
            },
            {
                "name": "orders",
                "key_column": "order_id",
                "route_version": 3,
                "boundaries": [0, 70, 200, 460, 850],
            },
            {
                "name": "audit",
                "key_column": "audit_id",
                "route_version": 1,
                "boundaries": [0, 50, 150, 300],
            },
            {
                "name": "audit",
                "key_column": "audit_id",
                "route_version": 2,
                "boundaries": [0, 40, 180, 300],
            },
            {
                "name": "audit",
                "key_column": "audit_id",
                "route_version": 3,
                "boundaries": [0, 35, 170, 320],
            },
        ],
        "events": [
            {"seq": 1, "kind": "write", "database": "db-a", "table": "orders", "route_key": 210},
            {"seq": 2, "kind": "query", "database": "db-a", "table": "audit", "route_key": 45},
            {"seq": 3, "kind": "write", "database": "db-b", "table": "orders", "route_key": 90},
            {"seq": 4, "kind": "query", "database": "db-b", "table": "orders", "route_key": 210},
            {"seq": 5, "kind": "write", "database": "db-a", "table": "orders", "route_key": 700},
            {"seq": 6, "kind": "query", "database": "db-b", "table": "audit", "route_key": 250},
        ],
    }


def write_spec(path: Path, values: dict[str, Any]) -> Path:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def spec_bundle(tmp_path: Path) -> dict[str, Any]:
    ports = (free_port(), free_port())
    current_values = make_spec_values(2, ports)
    evolved_values = make_spec_values(3, ports)
    edited_values = make_spec_values(2, ports)
    edited_values["events"] = list(edited_values["events"])
    edited_values["events"][1] = {
        "seq": 2,
        "kind": "write",
        "database": "db-a",
        "table": "audit",
        "route_key": 45,
    }
    current_path = write_spec(tmp_path / "current.json", current_values)
    evolved_path = write_spec(tmp_path / "evolved.json", evolved_values)
    edited_path = write_spec(tmp_path / "edited.json", edited_values)
    return {
        "ports": ports,
        "current_values": current_values,
        "evolved_values": evolved_values,
        "edited_values": edited_values,
        "current_path": current_path,
        "evolved_path": evolved_path,
        "edited_path": edited_path,
        "spec": load_spec(current_path),
        "evolved_spec": load_spec(evolved_path),
        "edited_spec": load_spec(edited_path),
    }


@pytest.fixture
def running_pool(spec_bundle: dict[str, Any]):
    spec: Spec = spec_bundle["spec"]
    pool = DatabasePool(spec, spec_bundle["current_path"])
    try:
        pool.start_all(["db-b", "db-a"])
        yield pool
    finally:
        pool.stop_all(force=True)

