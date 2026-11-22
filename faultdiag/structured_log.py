"""结构化 JSON 行日志, 写入工作目录 logs/ 下, 供诊断与审计。"""

from __future__ import annotations

import json
from pathlib import Path

from .util import monotonic_ms


class StructuredLog:
    def __init__(self, workdir):
        logdir = Path(workdir) / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        self._path = logdir / "faultdiag.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, level: str, msg: str, code: str | None = None, details: dict | None = None) -> None:
        entry = {"ts_mono_ms": monotonic_ms(), "level": level, "msg": msg}
        if code:
            entry["code"] = code
        if details:
            entry["details"] = details
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def info(self, msg: str, **kw) -> None:
        self.emit("info", msg, **kw)

    def error(self, msg: str, **kw) -> None:
        self.emit("error", msg, **kw)
