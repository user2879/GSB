"""确定性工具：内容摘要、保持时长、JSONL 结构化日志。"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

_log_lock = threading.Lock()


def canonical_json(obj) -> str:
    """跨平台、跨字典插入顺序稳定的 JSON 文本。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_of(obj) -> str:
    """对任意可 JSON 化对象计算稳定 SHA-256 摘要。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def rolling_digest(previous: str, obj) -> str:
    """链式摘要：把上一摘要并入当前块，防止重排与篡改。"""
    payload = previous + "|" + canonical_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hold_duration_ms(demand: int, priority: int, rid: str) -> int:
    """请求被授予后的保持时长（毫秒），仅由请求字段确定性推导。

    样本声明中没有显式完成时刻，使用确定性函数生成：
    base + 按需求线性增长 + 优先级修正 + 请求标识哈希抖动。
    这样相同输入在任何执行路径、任何释放顺序下完成时刻一致。
    """
    rid_hint = int(hashlib.sha256(rid.encode("utf-8")).hexdigest()[:8], 16) % 37
    return 40 + demand * 12 + (6 - min(max(priority, 0), 9)) * 5 + rid_hint


class StructuredLogger:
    """追加式 JSONL 结构化日志，字段稳定、可追溯到具体标识。"""

    def __init__(self, path: str):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    def log(self, level: str, event: str, **fields):
        entry = {
            "ts_ms": int(time.time() * 1000),
            "level": level,
            "event": event,
        }
        entry.update(fields)
        line = canonical_json(entry)
        if self.path:
            with _log_lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        return line

    def info(self, event, **fields):
        return self.log("INFO", event, **fields)

    def warn(self, event, **fields):
        return self.log("WARN", event, **fields)

    def error(self, event, **fields):
        return self.log("ERROR", event, **fields)
