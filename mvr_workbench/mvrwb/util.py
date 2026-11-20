"""确定性工具：定点量化、规范化 JSON、内容指纹、原子提交与耗时统计。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# 定点小数量化：round-half-away-from-zero，跨平台 / 跨线程确定性
# ---------------------------------------------------------------------------


def quantize_scalar(value: float, frac_bits: int) -> int:
    if not math.isfinite(value):
        raise ValueError(f"非有限值无法定点量化: {value!r}")
    scale = 1 << frac_bits
    scaled = value * scale
    # 正负对称的半远离零舍入
    return int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))


def quantize_float(value: float, frac_bits: int) -> float:
    """量化后还原为浮点，保证指纹输入与线程调度顺序无关。"""
    return quantize_scalar(value, frac_bits) / float(1 << frac_bits)


def ensure_finite(obj: Any, label: str) -> None:
    """递归校验数值均为有限值，命中非有限值抛出 ValueError。"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            ensure_finite(value, f"{label}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            ensure_finite(value, f"{label}[{index}]")
    elif isinstance(obj, (np.floating, float)):
        if not math.isfinite(float(obj)):
            raise ValueError(f"{label} 出现非有限特征值: {obj!r}")
    elif isinstance(obj, np.ndarray):
        if not np.isfinite(obj).all():
            raise ValueError(f"{label} 数组出现非有限特征值")


# ---------------------------------------------------------------------------
# 规范化 JSON 与内容指纹
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """确定性 JSON：键排序、固定分隔符、不转义非 ASCII、紧凑输出。"""
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_fingerprint(obj: Any) -> str:
    if isinstance(obj, bytes):
        return hashlib.sha256(obj).hexdigest()
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def fingerprint_many(*parts: Any) -> str:
    """由多个上游指纹 / 参数对象派生一个组合指纹。"""
    joined = [p if isinstance(p, str) else content_fingerprint(p) for p in parts]
    return hashlib.sha256("\u001f".join(joined).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 文件 / 目录原子操作
# ---------------------------------------------------------------------------


def write_json_atomic(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as handle:
        handle.write(canonical_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def read_json(path: str) -> Any:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def commit_directory(stage_dir: str, target_dir: str) -> None:
    """阶段产物先写 stage_dir 临时目录，再整体替换 target_dir。

    os.replace 对目录的替换在同一文件系统上是原子的 POSIX rename 语义。
    """
    parent = os.path.dirname(os.path.abspath(target_dir))
    os.makedirs(parent, exist_ok=True)
    backup_dir = f"{target_dir}.old.{os.getpid()}"
    if os.path.exists(target_dir):
        os.replace(target_dir, backup_dir)
    os.replace(stage_dir, target_dir)
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)


# ---------------------------------------------------------------------------
# 稳定排序键
# ---------------------------------------------------------------------------


def view_pair_id(view_a: int, view_b: int) -> str:
    lo, hi = sorted((int(view_a), int(view_b)))
    return f"vp-{lo:04d}-{hi:04d}"


def pair_sort_key(pair_id: str) -> tuple[int, int]:
    _, left, right = pair_id.split("-")
    return int(left), int(right)


# ---------------------------------------------------------------------------
# 阶段耗时
# ---------------------------------------------------------------------------


@contextmanager
def timed_stage(bucket: dict[str, float], name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        bucket[name] = bucket.get(name, 0.0) + (time.perf_counter() - start) * 1000.0
