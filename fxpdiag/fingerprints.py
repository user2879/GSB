"""内容指纹、参数指纹与分阶段缓存键派生。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def short_digest(payload: Any, length: int = 16) -> str:
    return _digest(payload)[:length]


def row_activation_fp(layer: str, row_index: int, row: list[float]) -> str:
    return short_digest({"kind": "activation", "layer": layer, "row": row_index, "v": row})


def param_fp(gamma: float, beta: float, scheme: str, rounding_mode: str) -> str:
    return short_digest(
        {
            "kind": "param",
            "gamma": gamma,
            "beta": beta,
            "scheme": scheme,
            "rounding": rounding_mode,
        }
    )


def config_fp(frac_bits: int, tier: str, rounding_mode: str, version: str) -> str:
    return short_digest(
        {
            "kind": "config",
            "frac_bits": frac_bits,
            "tier": tier,
            "rounding": rounding_mode,
            "version": version,
        }
    )


def cache_key(
    stage: str,
    upstream_fps: Iterable[str],
    parameter_fingerprint: str,
    quant_key: str,
    exchange_version: str,
) -> str:
    """阶段缓存键 = 上游内容指纹 + 参数指纹 + 量化档位 + 交换版本号。"""

    payload = {
        "stage": stage,
        "upstream": sorted(upstream_fps),
        "param": parameter_fingerprint,
        "quant": quant_key,
        "version": exchange_version,
    }
    return f"{stage}:{short_digest(payload, 24)}"


def conversion_cache_key(
    layer: str,
    row_index: int,
    weight_fp: str,
    exchange_version: str,
    sidecar_fp: str | None,
) -> str:
    payload = {
        "stage": "version_conversion",
        "layer": layer,
        "row": row_index,
        "weight": weight_fp,
        "version": exchange_version,
        "sidecar": sidecar_fp or "none",
    }
    return f"version_conversion:{short_digest(payload, 24)}"


def config_hash(config_dict: dict) -> str:
    return short_digest({"kind": "op_config", "cfg": config_dict}, 24)
