"""结构化错误码与结构化日志。

错误码约定：
  E_SPEC      规格声明非法（加载期强校验拒绝）
  E_INTEGRITY 块内容/摘要完整性校验失败
  E_OVERSUB   节点配额之和超出全局预算（超订）
  E_SNAPSHOT  快照/账本版本或内容摘要与磁盘状态不一致
  E_REPLAY    恢复后重放与账本记录不一致
"""

from __future__ import annotations

import json
import os
import threading
import time

E_SPEC = "E_SPEC"
E_INTEGRITY = "E_INTEGRITY"
E_OVERSUB = "E_OVERSUB"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"


class WorkbenchError(Exception):
    """携带结构化上下文的业务错误。"""

    def __init__(self, code: str, message: str, **context):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.context = context

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "context": self.context}


class StructuredLog:
    """JSONL 结构化日志，线程安全，写入工作目录 logs/events.jsonl。"""

    def __init__(self, workdir: str):
        self._dir = os.path.join(workdir, "logs")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, "events.jsonl")
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    def emit(self, code: str, message: str, **context) -> dict:
        event = {
            "ts_ms": int(time.time() * 1000),
            "code": code,
            "message": message,
            "context": context,
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return event

    def emit_error(self, err: WorkbenchError) -> dict:
        return self.emit(err.code, err.message, **err.context)

    def read_all(self) -> list:
        if not os.path.exists(self._path):
            return []
        with open(self._path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
