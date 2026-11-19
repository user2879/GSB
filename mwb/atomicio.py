"""确定性序列化、内容摘要与原子文件写出工具。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any

from .errors import E_IO, E_PARSE, WorkbenchError


def canonical_json(obj: Any) -> str:
    """规范化 JSON：键排序、无空白、非 ASCII 不转义，保证摘要稳定。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest_of(obj: Any) -> str:
    """对任意可 JSON 化对象计算 SHA-256 摘要（十六进制字符串）。"""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def atomic_write_json(path: str, payload: Any) -> None:
    """先写同目录临时文件，再 ``os.replace`` 整体替换，保证原子可见。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except OSError as exc:
        raise WorkbenchError(E_IO, f"原子写出失败: {path}", {"reason": str(exc)}) from exc


def read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise WorkbenchError(E_IO, f"文件不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkbenchError(E_PARSE, f"JSON 解析失败: {path}", {"at": exc.pos}) from exc
    except OSError as exc:
        raise WorkbenchError(E_IO, f"读取失败: {path}", {"reason": str(exc)}) from exc


def cleanup_temp_files(directory: str) -> None:
    """清理上次强制终止可能残留的同目录临时文件。"""
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        if name.startswith(".tmp-") and name.endswith(".json"):
            try:
                os.unlink(os.path.join(directory, name))
            except OSError:
                pass
