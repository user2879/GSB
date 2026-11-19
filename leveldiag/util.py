"""确定性工具：规范化 JSON、内容哈希、浮点比较口径。

所有工件均通过 :func:`canonical_dumps` 序列化，保证：
  * dict 按键排序，不依赖字典遍历顺序；
  * 统一 LF 换行、无尾随空白；
  * 浮点使用 Python repr（双精度最短往返表示），跨平台逐字节一致；
  * numpy 标量/数组在写入前转换为 Python 原生类型。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def to_native(obj: Any) -> Any:
    """递归把 numpy 类型转换为 Python 原生类型，保证 JSON 可序列化。"""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def canonical_dumps(obj: Any) -> bytes:
    """规范化序列化：sort_keys + LF + 双精度最短往返浮点。"""
    native = to_native(obj)
    return json.dumps(
        native,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(obj: Any) -> str:
    """对任意可 JSON 化对象计算 sha256 内容哈希（十六进制）。"""
    return hashlib.sha256(canonical_dumps(obj)).hexdigest()


def file_hash(path: str) -> str:
    """对已落盘文件计算 sha256（逐字节）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def ulp(x: float) -> float:
    """返回双精度浮点数 x 的一个 ULP 量级（次正规按最小值处理）。"""
    if not math.isfinite(x):
        raise ValueError("ulp 仅支持有限浮点")
    if x == 0.0:
        return 5e-324
    return math.ulp(abs(x))


# 判定口径（统一约定，账本与复算器共用同一实现）：
#   总比较带 band = 声明比较容差 tol + 传播误差上界 error_bound。
#   gt 要求 observed > threshold：
#       observed - threshold >  band           -> PASS（确定通过）
#       observed - threshold < -band           -> FAIL（确定失败）
#       其余（阈值落在观测值的误差区间内）       -> UNCERTAIN
#   lt 对称。
#   ge/le（非严格）：仅在明确违约时 FAIL，阈值落入误差区间一律 UNCERTAIN，
#       即边界相等（observed == threshold）也标记 UNCERTAIN，禁止静默四舍五入。
#   eq/interval 为精确离散量（计数类），不携带误差带，按数学关系判定。


def classify(
    direction: str,
    observed: float,
    threshold: float,
    tol: float,
    error_bound: float,
) -> str:
    """按统一口径返回 PASS / FAIL / UNCERTAIN。

    总比较带 = 声明比较容差 tol + 传播误差上界 error_bound。
    带为 0 时（离散结构量）退化为纯数学比较，边界相等按方向的数学含义确定。
    """
    band = float(tol) + float(error_bound)
    d = float(observed) - float(threshold)
    if band == 0.0:
        if direction == "gt":
            return "PASS" if d > 0 else "FAIL"
        if direction == "ge":
            return "PASS" if d >= 0 else "FAIL"
        if direction == "lt":
            return "PASS" if d < 0 else "FAIL"
        if direction == "le":
            return "PASS" if d <= 0 else "FAIL"
        if direction in ("eq", "interval"):
            return "PASS" if d == 0.0 else "FAIL"
        raise ValueError(f"未知比较方向: {direction}")
    if direction == "gt":  # 要求 observed > threshold
        if d > band:
            return "PASS"
        if d <= -band:
            return "FAIL"
        return "UNCERTAIN"
    if direction == "ge":
        if d < -band:
            return "FAIL"
        if d > band:
            return "PASS"
        return "UNCERTAIN"
    if direction == "lt":  # 要求 observed < threshold
        if d < -band:
            return "PASS"
        if d >= band:
            return "FAIL"
        return "UNCERTAIN"
    if direction == "le":
        if d > band:
            return "FAIL"
        if d < -band:
            return "PASS"
        return "UNCERTAIN"
    if direction in ("eq", "interval"):
        # 区间/相等判定由调用方自行处理，这里只做对称带判定
        if abs(d) <= band:
            return "PASS"
        return "FAIL"
    raise ValueError(f"未知比较方向: {direction}")


def interval_status(
    direction: str,
    observed: float,
    threshold: float,
    tol: float,
    error_bound: float,
) -> dict:
    """生成判定的完整误差区间记录。"""
    band = float(tol) + float(error_bound)
    return {
        "direction": direction,
        "observed": float(observed),
        "threshold": float(threshold),
        "comparison_tolerance": float(tol),
        "propagated_error_bound": float(error_bound),
        "total_band": band,
        "plausible_low": float(observed) - band,
        "plausible_high": float(observed) + band,
        "status": classify(direction, observed, threshold, tol, error_bound),
        "on_exact_boundary": float(observed) == float(threshold),
    }
