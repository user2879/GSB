from __future__ import annotations

import json
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config_ledger.simulator import (
    Scenario,
    SimulatorHTTPServer,
    reserve_free_port,
    start_server_in_thread,
)
from config_ledger.spec import Spec, parse_spec_data


def make_spec_data(
    ports: dict[str, int],
    keys: dict[str, dict[str, Any]],
    *,
    global_concurrency: int = 3,
    per_node_quota: int = 1,
    timeout_ms: int = 500,
    latency_budget_ms: int = 1000,
    window: str = "test-window",
    parameter_version: int = 1,
) -> dict[str, Any]:
    return {
        "name": "unit-test",
        "parameter_version": parameter_version,
        "global_concurrency": global_concurrency,
        "observation_timeout_ms": timeout_ms,
        "ui_latency_budget_ms": latency_budget_ms,
        "observation_window": window,
        "arbitration_rule": "majority_then_latest",
        "keys": keys,
        "nodes": [
            {
                "id": node_id,
                "host": "127.0.0.1",
                "port": ports[node_id],
                "quota": per_node_quota,
            }
            for node_id in sorted(ports)
        ],
    }


def parse_test_spec(data: dict[str, Any]) -> Spec:
    return parse_spec_data(data, source="<test>")


def write_json(path: Path, value: Any) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@contextmanager
def running_servers(
    scenarios: dict[str, dict[str, Any]],
    *,
    invalid_json_nodes: set[str] | None = None,
) -> Iterator[dict[str, int]]:
    invalid_json_nodes = invalid_json_nodes or set()
    scenario = Scenario(scenarios, source="<test-scenario>")
    servers: list[SimulatorHTTPServer] = []
    threads: list[threading.Thread] = []
    ports: dict[str, int] = {}
    try:
        for node_id in sorted(scenarios):
            port = reserve_free_port()
            server, thread = start_server_in_thread(
                scenario,
                node_id=node_id,
                port=port,
                invalid_json=node_id in invalid_json_nodes,
            )
            ports[node_id] = port
            servers.append(server)
            threads.append(thread)
        yield ports
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


@contextmanager
def temp_workspace() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="config-ledger-test-") as directory:
        yield Path(directory)


def log_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
