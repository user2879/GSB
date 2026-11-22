"""异常注入：派生成环、快照截断、内容版本摘要篡改。"""

import os
import sqlite3


def cycle_derivation_override():
    """返回一个使派生图成环的 derived_from 覆盖（/index.html 派生自下游 /page/home）。"""
    return ["/page/home"]


def truncate_snapshot(snapshot_path, keep_ratio=0.6):
    size = os.path.getsize(snapshot_path)
    keep = max(1, int(size * keep_ratio))
    with open(snapshot_path, "r+b") as fh:
        fh.truncate(keep)
    return {"snapshot": snapshot_path, "original_size": size, "kept": keep}


def tamper_snapshot_digest(snapshot_path):
    """篡改快照内记录的内容摘要（内容集合不变 -> 摘要不符）。"""
    conn = sqlite3.connect(snapshot_path)
    conn.execute(
        "UPDATE meta SET value='tampered-digest' WHERE key='content_digest'"
    )
    conn.commit()
    conn.close()
    return {"snapshot": snapshot_path, "field": "meta.content_digest"}
