from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from lease_arbiter.spec import Spec, load_spec


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_spec() -> Spec:
    return load_spec(
        ROOT / "config" / "nodes.json",
        ROOT / "config" / "leases.json",
        ROOT / "samples" / "events.jsonl",
    )


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def write_local_spec(
    tmp_path: Path,
    *,
    events_text: str | None = None,
    nodes_override: dict[str, Any] | None = None,
    leases_override: dict[str, Any] | None = None,
) -> Spec:
    nodes_payload: dict[str, Any] = {
        "schema_version": 1,
        "declaration_version": 1,
        "global_concurrency": 3,
        "connect_timeout_ms": 500,
        "read_timeout_ms": 500,
        "worker_threads": 4,
        "nodes": [
            {"node_id": "alpha", "host": "127.0.0.1", "port": free_port(), "max_leases": 2},
            {"node_id": "beta", "host": "127.0.0.1", "port": free_port(), "max_leases": 2},
        ],
    }
    leases_payload: dict[str, Any] = {
        "schema_version": 1,
        "declaration_version": 1,
        "leases": [
            {"lease_id": "a-cache-1", "node_id": "alpha", "lease_type": "cache", "ttl_ms": 1200, "renew_interval_ms": 200, "quota_version": 1},
            {"lease_id": "a-cache-2", "node_id": "alpha", "lease_type": "cache", "ttl_ms": 800, "renew_interval_ms": 200, "quota_version": 1},
            {"lease_id": "b-session-1", "node_id": "beta", "lease_type": "session", "ttl_ms": 700, "renew_interval_ms": 200, "quota_version": 1},
            {"lease_id": "b-cache-1", "node_id": "beta", "lease_type": "cache", "ttl_ms": 1000, "renew_interval_ms": 200, "quota_version": 1},
        ],
    }
    if nodes_override:
        nodes_payload.update(nodes_override)
    if leases_override:
        leases_payload.update(leases_override)
    events_text = events_text or (ROOT / "samples" / "events.jsonl").read_text(encoding="utf-8")
    nodes_path = tmp_path / "nodes.json"
    leases_path = tmp_path / "leases.json"
    events_path = tmp_path / "events.jsonl"
    nodes_path.write_text(json.dumps(nodes_payload, indent=2) + "\n", encoding="utf-8")
    leases_path.write_text(json.dumps(leases_payload, indent=2) + "\n", encoding="utf-8")
    events_path.write_text(events_text, encoding="utf-8")
    return load_spec(nodes_path, leases_path, events_path)


def clone_declaration(
    tmp_path: Path,
    spec: Spec,
    *,
    declaration_version: int,
    nodes: list[dict[str, Any]] | None = None,
    leases: list[dict[str, Any]] | None = None,
) -> Spec:
    nodes_payload = {
        "schema_version": 1,
        "declaration_version": declaration_version,
        "global_concurrency": spec.global_concurrency,
        "connect_timeout_ms": spec.connect_timeout_ms,
        "read_timeout_ms": spec.read_timeout_ms,
        "worker_threads": spec.worker_threads,
        "nodes": nodes or [node.to_dict() for node in spec.nodes],
    }
    leases_payload = {
        "schema_version": 1,
        "declaration_version": declaration_version,
        "leases": leases or [lease.to_dict() for lease in spec.leases],
    }
    nodes_path = tmp_path / f"nodes-{declaration_version}.json"
    leases_path = tmp_path / f"leases-{declaration_version}.json"
    events_path = tmp_path / f"events-{declaration_version}.jsonl"
    nodes_path.write_text(json.dumps(nodes_payload, indent=2) + "\n", encoding="utf-8")
    leases_path.write_text(json.dumps(leases_payload, indent=2) + "\n", encoding="utf-8")
    events_path.write_text(
        "\n".join(json.dumps(event.to_dict(), separators=(",", ":")) for event in spec.events) + "\n",
        encoding="utf-8",
    )
    return load_spec(nodes_path, leases_path, events_path)
