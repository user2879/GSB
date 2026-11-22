from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from driftwatch.config import parse_criteria
from driftwatch.models import METRICS, AgentSpec, AggregationSpec, Criteria


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def make_agent(
    agent_id: str,
    port: int,
    *,
    seed: int = 11,
    version: str = "agg-v1",
    kind: str = "median",
    window: int = 5,
    hosts: tuple[str, ...] = ("host-a",),
    metrics: tuple[str, ...] = METRICS,
    profiles: dict[str, Any] | None = None,
    per_agent_concurrency: int = 1,
) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id,
        host="127.0.0.1",
        port=port,
        seed=seed,
        aggregation=AggregationSpec(kind=kind, window=window, version=version),
        hosts=hosts,
        metrics=metrics,
        per_agent_concurrency=per_agent_concurrency,
        profiles=profiles or {},
    )


def make_criteria(**overrides: Any) -> Criteria:
    values = {
        "global_concurrency": 2,
        "per_agent_concurrency": 1,
        "min_sources": 2,
        "scale_exp": 3,
        "socket_timeout_ms": 1000,
        "handshake_timeout_ms": 3000,
        "max_retries": 1,
        "retry_backoff_ms": 5,
        "boundary_rule": "tolerance_floor",
    }
    values.update(overrides)
    return parse_criteria(values)


def write_json(path: Path, value: Any) -> Path:
    import json

    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def snapshot_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshots": [
            {
                "seq": 1,
                "parent_seq": None,
                "hosts": {
                    "host-a": {
                        "expected": {"cpu_quota": 100000, "memory_pages": 64000, "processes": 500},
                        "tolerance_bps": {"cpu_quota": 500, "memory_pages": 500, "processes": 500},
                    },
                    "host-b": {
                        "expected": {"cpu_quota": 80000, "memory_pages": 52000, "processes": 420},
                        "tolerance_bps": 600,
                    },
                },
            },
            {
                "seq": 2,
                "parent_seq": 1,
                "hosts": {
                    "host-a": {
                        "expected": {"cpu_quota": 98000, "memory_pages": 63000, "processes": 495},
                        "tolerance_bps": 300,
                    },
                    "host-b": {
                        "expected": {"cpu_quota": 78000, "memory_pages": 51000, "processes": 410},
                        "tolerance_bps": 400,
                    },
                },
            },
        ],
    }
