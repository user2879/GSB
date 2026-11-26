from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any

from waterlease.spec import load_spec


ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def write_test_spec(tmp_path: Path, *, delay_ms: int = 5, budget: int = 6) -> tuple[Path, Any]:
    base = load_spec(ROOT / "spec")
    raw = copy.deepcopy(base.raw)
    raw["session"]["global_resource_budget"] = budget
    raw["session"]["lease_ttl_ms"] = 2000
    for index, node in enumerate(raw["nodes"]):
        node["port"] = free_port()
        node["processing_delay_ms"] = delay_ms
    directory = tmp_path / "spec"
    directory.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("session.json", "session"),
        ("nodes.json", "nodes"),
        ("shards.json", "shards"),
        ("capacity_versions.json", "capacity_versions"),
        ("events.json", "events"),
    ):
        value = raw[key]
        if key in {"nodes", "shards", "capacity_versions", "events"}:
            value = {key: value}
        (directory / name).write_text(json.dumps(value), encoding="utf-8")
    return directory, load_spec(directory)
