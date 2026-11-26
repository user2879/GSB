from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from .example import build_example_spec
from .orchestrator import allocate_free_port
from .spec import Spec, validate_spec


def make_test_spec(
    *,
    node_count: int = 3,
    ports: list[int] | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if ports is None:
        ports = [allocate_free_port() for _ in range(node_count)]
    if node_count == 3:
        raw = build_example_spec(ports[:3])
    else:
        raw = build_example_spec([allocate_free_port() for _ in range(3)])
        raw["nodes"] = raw["nodes"][:node_count]
        allowed = {f"s{i}" for i in range(1, 7)}
        if node_count == 1:
            for shard in raw["shards"]:
                shard["node_id"] = "n1"
        elif node_count == 2:
            for shard in raw["shards"]:
                if shard["node_id"] == "n3":
                    shard["node_id"] = "n2"
    raw = copy.deepcopy(raw)
    if mutate:
        mutate(raw)
    return raw


def write_test_spec(path: str | Path, raw: dict[str, Any] | None = None) -> Spec:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = make_test_spec()
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return validate_spec(raw, source=str(path))
