"""确定性内容指纹与缓存键派生。

全部采用规范化 JSON（sort_keys、无多余空白、无随机盐）+ SHA-256，
保证增量运行与全量重建在相同内容下得到逐字符一致的指纹。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_fingerprint(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def derive_cache_key(prefix: str, *upstream_fingerprints: str) -> str:
    """由若干上游内容指纹派生产物缓存键。"""
    payload = {"prefix": prefix, "upstream": list(upstream_fingerprints)}
    return prefix + ":" + sha256_text(canonical_json(payload))[:16]


def sensor_coeff_fingerprint(sensor_entry: dict) -> str:
    """传感器级定标系数指纹：增益/偏移/位宽/饱和上限/参考时刻。"""
    coeff = {
        "sensor_id": sensor_entry["sensor_id"],
        "bands": sensor_entry["bands"],
        "bit_width": sensor_entry["bit_width"],
        "saturation_limit": sensor_entry["saturation_limit"],
        "reference_time_ms": sensor_entry["reference_time_ms"],
    }
    return content_fingerprint(coeff)


def calibration_table_fingerprint(table: dict) -> str:
    return content_fingerprint({sid: sensor_coeff_fingerprint(entry)
                                for sid, entry in sorted(table.items())})


def regression_config_fingerprint(config: dict) -> str:
    """回归参数指纹（仅参与求解/对齐的参数，不含预算/并发等调度参数）。"""
    keys = ("reference_sensor", "source_whitelist", "feature_orders",
            "initial_weights", "regularization", "fixed_point_bits",
            "fixed_point_frac_bits", "quantization_levels", "near_singular_rcond")
    return content_fingerprint({k: config[k] for k in keys if k in config})
