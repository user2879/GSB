"""三类异常注入：仅修改输入文件或运行参数，不修改源码。

- inject_capacity_overflow : 改写热层容量上限 -> E_CAPACITY
- inject_truncated_object  : 截断对象文件并破坏摘要 -> E_INTEGRITY
- inject_version_conflict  : 复制账本并改写格式版本 -> E_VERSION
注入均作用于工作区副本或在账本可恢复前提下进行，并写结构化日志。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .applog import StructuredLogger
from .workspace import TIERS_FILE


def inject_capacity_overflow(workspace: str | Path,
                             logger: StructuredLogger) -> Path:
    """把热层容量改写为不足以容纳任一待迁对象，触发 E_CAPACITY。"""
    tiers_path = Path(workspace) / TIERS_FILE
    doc = json.loads(tiers_path.read_text(encoding="utf-8"))
    objects_path = Path(workspace) / "objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8"))["objects"]
    min_size = min(o["size_bytes"] for o in objects)
    for tier in doc["tiers"]:
        if tier["role"] == "cold":
            # 冷层连最小对象都接纳不了，首个迁出动作即被拒绝
            tier["capacity_bytes"] = max(0, min_size - 1)
    tiers_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    logger.error("fault_injected", code="E_CAPACITY",
                 fault="capacity_overflow",
                 cold_capacity=min_size - 1, min_object_size=min_size)
    return tiers_path


def inject_truncated_object(workspace: str | Path,
                            logger: StructuredLogger,
                            object_key: str | None = None) -> Path:
    """截断热层某个对象文件，使摘要校验失败，触发 E_INTEGRITY。"""
    objects_path = Path(workspace) / "objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8"))["objects"]
    target = next(o for o in objects
                  if object_key is None or o["key"] == object_key)
    hot_dir = Path(workspace) / "tiers" / "hot"
    victim = hot_dir / f"{target['content_digest']}.bin"
    if not victim.is_file():
        raise FileNotFoundError(f"待截断对象文件不存在: {victim}")
    original = victim.read_bytes()
    cut = max(1, len(original) // 2)
    victim.write_bytes(original[:cut])
    logger.error("fault_injected", code="E_INTEGRITY", fault="truncated_object",
                 key=target["key"], path=str(victim),
                 original_bytes=len(original), truncated_bytes=cut,
                 expected_digest=target["content_digest"])
    return victim


def inject_version_conflict(workspace: str | Path,
                            logger: StructuredLogger,
                            claimed_version: int = 999) -> Path:
    """复制账本并把副本的格式版本改成不支持的值，触发 E_VERSION。"""
    source = Path(workspace) / "ledger.db"
    target = Path(workspace) / "ledger.conflict.db"
    import sqlite3
    # WAL 模式下先把 WAL 合并进主库，避免复制后看不到最新页
    checker = sqlite3.connect(str(source))
    try:
        checker.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checker.commit()
    finally:
        checker.close()
    shutil.copy2(source, target)
    conn = sqlite3.connect(str(target))
    try:
        conn.execute("UPDATE meta SET value=? WHERE key='format_version'",
                     (str(claimed_version),))
        conn.commit()
    finally:
        conn.close()
    logger.error("fault_injected", code="E_VERSION", fault="version_conflict",
                 ledger=str(target), claimed_format_version=claimed_version)
    return target
