"""通用工具: 摘要、规范化 JSON、原子写、落盘、结构化日志。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """键排序、无多余空白的规范化 JSON, 用于确定性摘要与快照比较。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_of_json(obj: Any) -> str:
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def fsync_file(fh: Any) -> None:
    fh.flush()
    os.fsync(fh.fileno())


def fsync_dir(path: Path) -> None:
    """目录落盘, 保证新建/改名在崩溃后可见(Windows 上可能不支持, 安全降级)。"""
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(str(path), flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        # Windows 旧实现或文件系统不支持目录 fsync 时忽略
        pass


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """临时文件 + 改名的原子替换, 改名后对目录落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fsync_file(fh)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def append_fsynced(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line if line.endswith("\n") else line + "\n")
        fsync_file(fh)


class StructuredLogger:
    """按行写 JSON 的结构化日志; 业务码事件单独可枚举。"""

    def __init__(self, path: Path, clock: Any = None) -> None:
        self.path = path
        self._clock = clock
        self.events: list[dict[str, Any]] = []

    def _ts(self) -> int:
        return self._clock.now_ms() if self._clock is not None else 0

    def log(
        self,
        event: str,
        *,
        code: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_ms": self._ts(),
            "event": event,
            "code": code,
            "detail": dict(detail or {}),
        }
        self.events.append(record)
        append_fsynced(self.path, canonical_json(record))
        return record

    def log_error(self, event: str, err: BaseException) -> dict[str, Any]:
        if hasattr(err, "to_dict"):
            payload = err.to_dict()  # type: ignore[attr-defined]
            code = payload.pop("code", getattr(err, "code", "E_INTERNAL"))
            message = payload.pop("message", str(err))
            return self.log(event, code=code, detail={"message": message, **payload})
        return self.log(event, code="E_INTERNAL", detail={"message": str(err)})

    def codes(self) -> list[str]:
        return [e["code"] for e in self.events if e.get("code")]


def stable_sorted(items: Iterable[Any], key: Any) -> list[Any]:
    """显式稳定排序, 保证渲染/报告顺序与遍历顺序无关。"""
    return sorted(items, key=key)
