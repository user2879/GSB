from __future__ import annotations

import copy
import json
import socket
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sample_spec_dict(*, free_ports: bool = True, small_rows: bool = True) -> dict[str, Any]:
    raw = json.loads((ROOT / "config" / "spec.json").read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    if free_ports:
        for node in raw["nodes"]:
            node["port"] = free_port()
    if small_rows:
        for shard in raw["shards"]:
            shard["sample_rows"] = min(int(shard["sample_rows"]), 4096)
    raw["lease_ttl_ms"] = 1_000
    return raw


def write_spec(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path
