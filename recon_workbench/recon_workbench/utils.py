"""确定性工具：规范化序列化、内容指纹、原子写入、定点量化。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
from typing import Any, Iterable, Tuple

import numpy as np


def canonical_bytes(obj: Any) -> bytes:
    """对 JSON 兼容对象做规范化（键排序、无空白）序列化。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def content_hash(obj: Any) -> str:
    """任意 JSON 兼容对象的内容指纹（sha256 前 16 位）。"""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()[:16]


def file_hash(path: str) -> str:
    """文件字节内容指纹，分块读取避免大文件全量驻留。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def atomic_write_bytes(path: str, data: bytes) -> None:
    """同目录临时文件 + rename 的原子写。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path: str, obj: Any) -> None:
    atomic_write_bytes(path, canonical_bytes(obj))


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_replace_dir(src_dir: str, dst_dir: str) -> None:
    """整体目录替换：先在目标父目录生成临时目录，rename 原子换入。

    ``src_dir`` 与 ``dst_dir`` 必须位于同一文件系统（同一父目录）。
    """
    parent = os.path.dirname(os.path.abspath(dst_dir))
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(dst_dir.rstrip(os.sep))
    tmp_dir = tempfile.mkdtemp(prefix=f".{base}.stage-", dir=parent)
    os.rmdir(tmp_dir)
    shutil.move(src_dir, tmp_dir)
    if os.path.lexists(dst_dir):
        backup = dst_dir + ".old"
        if os.path.lexists(backup):
            shutil.rmtree(backup)
        os.replace(dst_dir, backup)
        os.replace(tmp_dir, dst_dir)
        shutil.rmtree(backup)
    else:
        os.replace(tmp_dir, dst_dir)


def fixed_point_quantize(values: np.ndarray, fractional_bits: int) -> np.ndarray:
    """定点量化：round 到 2^-frac 网格。

    使用 round-half-to-even（NumPy 默认），float32/float64 行为各自独立，
    保证同输入同精度档位的逐位可复现。
    """
    scale = float(1 << fractional_bits)
    arr = np.asarray(values)
    return np.round(arr * scale) / scale


def depth_quantize(depth: np.ndarray, depth_levels: int) -> np.ndarray:
    """将正深度值离散到 ``depth_levels`` 个均匀档位（线性量化到整数档号）。"""
    arr = np.asarray(depth, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, -1, dtype=np.int64)
    dmin = float(finite.min())
    dmax = float(finite.max())
    if dmax <= dmin:
        return np.where(np.isfinite(arr), 0, -1).astype(np.int64)
    step = (dmax - dmin) / depth_levels
    safe = np.where(np.isfinite(arr), arr, dmin)
    levels = np.floor((safe - dmin) / step)
    levels = np.clip(levels, 0, depth_levels - 1).astype(np.int64)
    return np.where(np.isfinite(arr), levels, -1).astype(np.int64)


def voxel_key(point: np.ndarray, voxel_size: float) -> Tuple[int, int, int]:
    """三维点的体素键，floor 到体素网格，跨精度档位可比较。"""
    coords = np.asarray(point, dtype=np.float64) / float(voxel_size)
    if not np.isfinite(coords).all():
        raise ValueError("体素键输入含非有限坐标")
    return tuple(int(np.floor(v)) for v in coords)  # type: ignore[return-value]


def stable_sort_points(points: np.ndarray) -> np.ndarray:
    """按坐标字典序返回稳定排序索引（x, y, z 升序）。"""
    order = np.lexsort(
        (points[:, 2], points[:, 1], points[:, 0])
    )
    return np.asarray(order, dtype=np.int64)


def finite_float(obj: Any) -> Any:
    """递归校验结构内所有浮点均为有限数，非有限抛出 AssertionError。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            finite_float(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            finite_float(value)
    elif isinstance(obj, float):
        assert np.isfinite(obj), "遇到非有限浮点"
    return obj


def iter_pairs(n_views: int) -> Iterable[Tuple[int, int]]:
    """按视图序号稳定枚举相邻视图对 (i, i+1)，顺序即处理顺序。"""
    for i in range(n_views - 1):
        yield i, i + 1
