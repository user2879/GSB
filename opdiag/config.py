"""执行配置枚举校验与规范化。"""

from __future__ import annotations

from .errors import E_ENUM_INVALID
from .fixedpoint import ALLOWED_BIT_WIDTHS, ALLOWED_QUANT_MODES

ALLOWED_PRECISION = ("fp32", "fp64")
DEFAULTS = {
    "precision": "fp32",
    "bit_width": 16,
    "quant_mode": "uniform",
    "tolerance": 1e-5,
    "max_workers": 2,
    "element_budget": 256,
    "snapshot_interval": 3,
    "enable_fusion": True,
    "output_dir": "output",
}


def normalize_config(raw: dict) -> tuple[dict, list[dict]]:
    """填入默认值并做枚举校验，返回 (config, enum_failures)。"""
    cfg = dict(DEFAULTS)
    cfg.update(raw or {})
    failures: list[dict] = []

    if cfg["precision"] not in ALLOWED_PRECISION:
        failures.append({
            "code": E_ENUM_INVALID,
            "message": f"precision must be one of {ALLOWED_PRECISION}",
            "field": "precision", "value": cfg["precision"],
        })
    if cfg["bit_width"] not in ALLOWED_BIT_WIDTHS:
        failures.append({
            "code": E_ENUM_INVALID,
            "message": f"bit_width must be one of {ALLOWED_BIT_WIDTHS}",
            "field": "bit_width", "value": cfg["bit_width"],
        })
    if cfg["quant_mode"] not in ALLOWED_QUANT_MODES:
        failures.append({
            "code": E_ENUM_INVALID,
            "message": f"quant_mode must be one of {ALLOWED_QUANT_MODES}",
            "field": "quant_mode", "value": cfg["quant_mode"],
        })
    for key in ("max_workers", "element_budget", "snapshot_interval"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            failures.append({
                "code": E_ENUM_INVALID,
                "message": f"{key} must be positive int",
                "field": key, "value": cfg[key],
            })
    if not isinstance(cfg["tolerance"], (int, float)) or cfg["tolerance"] <= 0:
        failures.append({
            "code": E_ENUM_INVALID,
            "message": "tolerance must be positive number",
            "field": "tolerance", "value": cfg["tolerance"],
        })
    return cfg, failures
