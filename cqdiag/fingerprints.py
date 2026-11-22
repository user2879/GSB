"""内容指纹与阶段缓存键派生 (业务要求 6)."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def _hash_obj(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def column_content_fp(col_values: np.ndarray, col_missing: np.ndarray) -> str:
    """列内容指纹: 覆盖逐元素值与缺失标记."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(col_values, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(col_missing, dtype=bool).tobytes())
    return h.hexdigest()


def calib_param_fp(lo_permille: int, hi_permille: int) -> str:
    return _hash_obj({"lo": lo_permille, "hi": hi_permille})


def calib_key(content_fp: str, param_fp: str) -> str:
    return _hash_obj({"stage": "calib", "content": content_fp, "params": param_fp})


def scale_key(calib_k: str, scheme: str, bits: int) -> str:
    return _hash_obj({"stage": "scale", "up": calib_k, "scheme": scheme, "bits": bits})


def quant_key(scale_k: str, content_fp: str, frac_bits: int, saturation: str) -> str:
    return _hash_obj(
        {"stage": "quant", "up": scale_k, "content": content_fp,
         "frac_bits": frac_bits, "saturation": saturation}
    )


def dequant_key(quant_k: str, content_fp: str, abs_tol: float, rel_tol: float) -> str:
    return _hash_obj(
        {"stage": "dequant", "up": quant_k, "content": content_fp,
         "abs_tol": abs_tol, "rel_tol": rel_tol}
    )


def stratify_key(tensor: str, labels: tuple[int, ...], dequant_keys: list[str],
                 group_oor_limit: float) -> str:
    return _hash_obj(
        {"stage": "stratify", "tensor": tensor, "labels": list(labels),
         "dequant_keys": dequant_keys, "group_oor_limit": group_oor_limit}
    )


def config_fp(config) -> str:
    """配置哈希, 用于清单版本变化识别."""
    return _hash_obj(
        {
            "tensors": [[t.name, list(t.shape), t.column_axis] for t in config.tensors],
            "lo": config.quantile_lo_permille,
            "hi": config.quantile_hi_permille,
            "scheme": config.scheme,
            "candidates": list(config.quant_bits_candidates),
            "quant_bits": config.quant_bits,
            "frac_bits": config.frac_bits,
            "saturation": config.saturation,
            "abs_tol": config.abs_tol,
            "rel_tol": config.rel_tol,
            "groups": {k: list(v) for k, v in config.groups.items()},
            "group_oor_limit": config.group_oor_limit,
        }
    )
