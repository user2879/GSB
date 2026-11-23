"""稳定哈希、定点换算与确定性 JSON I/O。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import ROUND_HALF_UP, Decimal


def canonical_json(obj) -> str:
    """跨进程稳定的规范化 JSON 文本（排序键、固定分隔符、不转义非 ASCII）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(obj) -> str:
    """对任意可 JSON 化对象计算内容指纹。"""
    return sha256_text(canonical_json(obj))[:16]


def combine_fingerprint(*parts: str) -> str:
    return sha256_text("|".join(parts))[:16]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---- 定点小数（配置 fixed_point_bits 位宽） ----
def float_to_fixed(value: float, bits: int) -> int:
    """四舍五入（ROUND_HALF_UP）为 2^bits 定标整数。"""
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be fixed-point quantized")
    quant = Decimal(1).scaleb(-bits)
    return int(Decimal(repr(float(value))).quantize(quant, rounding=ROUND_HALF_UP) * (1 << bits))


def fixed_to_float(fixed: int, bits: int) -> float:
    return int(fixed) / float(1 << bits)


def round_half_up(value: float) -> int:
    """确定性四舍五入，避免 Python banker's rounding 的跨语言歧义。"""
    return int(Decimal(repr(float(value))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def snap_to_grid(value: float, grid: float, fixed_bits: int) -> float:
    """把 value 吸附到 grid 网格，且结果按 2^fixed_bits 定点量化后仍与网格对齐。"""
    if grid <= 0:
        raise ValueError("grid must be positive")
    scale = 1 << fixed_bits
    index = round_half_up(value / grid)
    fixed = round_half_up(grid * index * scale)
    return fixed / scale
