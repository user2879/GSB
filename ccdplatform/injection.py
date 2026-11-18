"""命令行异常注入：仅改变输入文件或会话参数，不改业务代码。

三类注入
--------
* range  : 把指定配置项 value 改写为超出定点物理范围的值 -> E_RANGE
* cycle  : 在清单中新增一条成环派生引用 -> E_CYCLE
* grid   : 以缺失 grid 元数据的快照打开会话 -> E_GRID
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from .fixedpoint import GridSpec


def inject_range(manifest_path: str | Path, target_key: str, exponent: int,
                 int_bits: int = 63) -> Dict[str, Any]:
    grid = GridSpec(exponent=exponent, int_bits=int_bits)
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    overflow_value = (grid.qmax + 1) / grid.scale * 10  # 必然越界
    hit = False
    for version in data["versions"]:
        for item in version["items"]:
            if item["key"] == target_key:
                item["value"] = overflow_value
                hit = True
    if not hit:
        raise KeyError(f"注入目标键不存在: {target_key}")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"injection": "range", "key": target_key, "value": overflow_value}


def inject_cycle(manifest_path: str | Path, key_a: str, key_b: str) -> Dict[str, Any]:
    """让 key_a 引用 key_b、key_b 引用 key_a，构造确定性的二元环。"""
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    latest = data["versions"][-1]
    for item in latest["items"]:
        if item["key"] == key_a:
            item["derivation"] = {"refs": [[key_b, 1.0]]}
        if item["key"] == key_b:
            item["derivation"] = {"refs": [[key_a, 1.0]]}
    present = {it["key"] for it in latest["items"]}
    if not {key_a, key_b} <= present:
        raise KeyError(f"成环注入键缺失: {key_a}, {key_b}")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"injection": "cycle", "edges": [[key_a, key_b], [key_b, key_a]]}


def strip_grid_from_snapshot(db_path: str | Path) -> Dict[str, Any]:
    """删除快照 meta 中的 grid 行，模拟精度格度缺失。"""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM meta WHERE key='grid'")
        conn.commit()
    finally:
        conn.close()
    return {"injection": "grid", "db": str(db_path)}
