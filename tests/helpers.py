from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from cdn_edge.netproto import port_is_available


def free_port_base(host: str = "127.0.0.1", need: int = 4, start: int = 20000) -> int:
    for base in range(start, 30000 - need, 7):
        if all(port_is_available(host, base + offset) for offset in range(need)):
            return base
    raise RuntimeError("could not find a free high-port window")


def sample_raw() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "samples" / "sample_spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


def remap_ports(raw: dict[str, Any], base: int) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    result["session"]["source_port"] = base
    for index, node in enumerate(result["nodes"]):
        node["port"] = base + index + 1
    return result
