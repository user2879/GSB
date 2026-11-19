"""关系名映射表：v1 短码 <-> v2 命名关系。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # 兼容 {"v1_to_v2": {...}} 或直接的 {短码: 命名关系}。
    if "v1_to_v2" in data:
        v1_to_v2 = dict(data["v1_to_v2"])
    else:
        v1_to_v2 = dict(data)
    v2_to_v1 = {named: code for code, named in v1_to_v2.items()}
    return {
        "v1_to_v2": v1_to_v2,
        "v2_to_v1": v2_to_v1,
        "named_relations": set(v1_to_v2.values()),
    }
