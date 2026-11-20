"""确定性 JSON 读写工具。

统一编码规则（sort_keys、不转义非 ASCII、固定分隔符、不输出 NaN），
保证同一逻辑数据在任意机器上序列化结果逐字节一致，便于逐字段比对。
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def dumps(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical(data: Any) -> str:
    """规范化 JSON 文本，用于逐字段一致性比对。"""
    return dumps(data)


def atomic_write_text(path: str, text: str) -> None:
    """先写同目录临时文件再原子替换，保证不会出现半写文件。"""

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str, data: Any) -> None:
    atomic_write_text(path, dumps(data))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
