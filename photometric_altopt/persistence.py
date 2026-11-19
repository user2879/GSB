"""原子持久化与内容哈希原语。

写入约定：所有产物先写到同目录下唯一临时文件，``flush + fsync`` 后用
``os.replace`` 原子替换，最后写 ``*.committed`` 标记文件（同样原子）。
读取方只承认带提交标记的产物，写入中途崩溃（强制终止）时只剩临时文件，
重启后按垃圾清理，不构成任何已提交状态。

全部使用 pathlib，兼容 Windows 与 POSIX。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """对数组的字节表示做哈希；同 dtype/同形状/同值 => 同哈希。"""
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    """确定性 JSON：sort_keys + 固定分隔符 + 无空白随机性。"""
    text = json.dumps(payload, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)
    atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def atomic_save_npy(path: Path, array: np.ndarray) -> str:
    """原子写 NumPy ``.npy``，返回数组内容哈希（基于原始字节，而非容器）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, np.ascontiguousarray(array), allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return sha256_array(array)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def commit_marker(path: Path) -> Path:
    return Path(str(path) + ".committed")


def atomic_commit(path: Path, marker_payload: dict[str, Any] | None = None) -> None:
    """为已原子写入的主文件补提交标记。"""
    payload = marker_payload if marker_payload is not None else {"committed": True}
    atomic_write_json(commit_marker(path), payload)


def is_committed(path: Path) -> bool:
    return Path(path).exists() and commit_marker(path).exists()


def require_committed(path: Path) -> Path:
    path = Path(path)
    if not is_committed(path):
        raise FileNotFoundError(f"产物未提交或不存在: {path}")
    return path


def cleanup_temp_files(directory: Path) -> list[str]:
    """清理崩溃残留的 ``.tmp`` 与无主提交标记，返回被清理的文件名。"""
    directory = Path(directory)
    removed: list[str] = []
    if not directory.exists():
        return removed
    for item in directory.rglob("*"):
        if item.is_file() and item.name.endswith(".tmp"):
            item.unlink()
            removed.append(str(item))
    for item in directory.rglob("*.committed"):
        main = Path(str(item)[: -len(".committed")])
        if not main.exists():
            item.unlink()
            removed.append(str(item))
    return removed


def directory_manifest(directory: Path) -> dict[str, str]:
    """目录内已提交文件的确定性清单：相对路径 -> 文件内容哈希。"""
    directory = Path(directory)
    manifest: dict[str, str] = {}
    if not directory.exists():
        return manifest
    for item in sorted(directory.rglob("*")):
        if not item.is_file() or item.name.endswith(".committed") or item.name.endswith(".tmp"):
            continue
        if not commit_marker(item).exists():
            continue
        rel = item.relative_to(directory).as_posix()
        manifest[rel] = sha256_file(item)
    return dict(sorted(manifest.items()))
