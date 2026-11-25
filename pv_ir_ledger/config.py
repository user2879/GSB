from __future__ import annotations

import json
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from . import COLORMAP_VERSION, GRADE_TABLE_VERSION
from .errors import ErrorCode, make_issue
from .hashing import fingerprint


ALLOWED_COLORMAPS = ("IRON_RED_V1", "GRAY_V1")
ALLOWED_QUANT_LEVELS = (8, 16, 32, 64, 128, 256)
MAX_CONCURRENCY_LIMIT = 8

DEFAULT_CONFIG: dict[str, Any] = {
    "schema": "pv-ir-config-v1",
    "fixed_width": 320,
    "fixed_height": 240,
    "pixel_mode": "RGB",
    "colormap_name": "IRON_RED_V1",
    "colormap_version": COLORMAP_VERSION,
    "grade_table_version": GRADE_TABLE_VERSION,
    "grid_rows": 4,
    "grid_cols": 6,
    "percentile_low_permille": 300,
    "percentile_high_permille": 980,
    "min_coverage_permille": 900,
    "min_hotspot_area_permille": 20,
    "bbox_padding_px": 2,
    "grade_thresholds_permille": [240, 400, 560, 720],
    "fixed_decimals": 3,
    "quant_levels": 128,
    "quant_scheme": "UNIFORM_FLOOR_V1",
    "heat_bin_count": 32,
    "sample_budget": 10**12,
    "max_concurrency": 4,
    "max_concurrency_limit": MAX_CONCURRENCY_LIMIT,
}


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_json(path: str | Path) -> tuple[Any, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_float=str), issues
    except FileNotFoundError:
        issues.append(make_issue(ErrorCode.CONFIG_INVALID, "JSON file not found", path=str(path)))
    except json.JSONDecodeError as exc:
        issues.append(
            make_issue(
                ErrorCode.CONFIG_INVALID,
                "JSON parse failed",
                path=str(path),
                line=exc.lineno,
                column=exc.colno,
            )
        )
    return None, issues


def validate_config(raw: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return None, [make_issue(ErrorCode.CONFIG_INVALID, "configuration must be a JSON object")]

    cfg = deepcopy(DEFAULT_CONFIG)
    for key, value in raw.items():
        cfg[key] = value

    def reject(message: str, **context: Any) -> None:
        issues.append(make_issue(ErrorCode.CONFIG_INVALID, message, **context))

    if cfg.get("schema") != "pv-ir-config-v1":
        reject("unsupported configuration schema", field="schema", value=cfg.get("schema"))

    for field in ("fixed_width", "fixed_height"):
        if not is_int(cfg.get(field)) or cfg[field] <= 0:
            reject("fixed image dimension must be a positive integer", field=field, value=cfg.get(field))
    if cfg.get("pixel_mode") != "RGB":
        reject("pixel_mode must be RGB", field="pixel_mode", value=cfg.get("pixel_mode"))

    if cfg.get("colormap_name") not in ALLOWED_COLORMAPS:
        reject(
            "colormap_name is outside the allowed table",
            field="colormap_name",
            value=cfg.get("colormap_name"),
            allowed=list(ALLOWED_COLORMAPS),
        )

    for field in ("grid_rows", "grid_cols"):
        if not is_int(cfg.get(field)) or not (1 <= cfg[field] <= 64):
            reject("grid dimension must be an integer in [1,64]", field=field, value=cfg.get(field))

    low = cfg.get("percentile_low_permille")
    high = cfg.get("percentile_high_permille")
    if not is_int(low) or not (0 <= low <= 1000):
        reject("percentile must be an integer in [0,1000]", field="percentile_low_permille", value=low)
    if not is_int(high) or not (0 <= high <= 1000):
        reject("percentile must be an integer in [0,1000]", field="percentile_high_permille", value=high)
    if is_int(low) and is_int(high) and not low < high:
        reject("percentile lower bound must be smaller than upper bound", low=low, high=high)

    for field in ("min_coverage_permille", "min_hotspot_area_permille"):
        value = cfg.get(field)
        if not is_int(value) or not (0 <= value <= 1000):
            reject("ratio permille must be an integer in [0,1000]", field=field, value=value)

    if not is_int(cfg.get("bbox_padding_px")) or not (0 <= cfg["bbox_padding_px"] <= 32):
        reject("bbox_padding_px must be an integer in [0,32]", value=cfg.get("bbox_padding_px"))

    thresholds = cfg.get("grade_thresholds_permille")
    if not isinstance(thresholds, list) or len(thresholds) != 4:
        reject("grade_thresholds_permille must contain four thresholds", value=thresholds)
    else:
        normalized_thresholds: list[int] = []
        for index, value in enumerate(thresholds):
            if not is_int(value) or not (0 <= value <= 1000):
                reject("grade threshold must be an integer in [0,1000]", index=index, value=value)
            else:
                normalized_thresholds.append(value)
        if len(normalized_thresholds) == 4:
            if any(normalized_thresholds[i] >= normalized_thresholds[i + 1] for i in range(3)):
                reject("grade thresholds must be strictly increasing", value=normalized_thresholds)

    decimals = cfg.get("fixed_decimals")
    if not is_int(decimals) or not (0 <= decimals <= 4):
        reject("fixed_decimals must be an integer in [0,4]", value=decimals)

    levels = cfg.get("quant_levels")
    if not is_int(levels) or levels not in ALLOWED_QUANT_LEVELS:
        reject("quant_levels must be one of the allowed levels", value=levels, allowed=list(ALLOWED_QUANT_LEVELS))

    if cfg.get("quant_scheme") != "UNIFORM_FLOOR_V1":
        reject("unsupported quantization scheme", value=cfg.get("quant_scheme"))

    if not is_int(cfg.get("heat_bin_count")) or not (2 <= cfg["heat_bin_count"] <= 256):
        reject("heat_bin_count must be an integer in [2,256]", value=cfg.get("heat_bin_count"))

    budget = cfg.get("sample_budget")
    if not is_int(budget) or budget < 0:
        reject("sample_budget must be a non-negative integer", value=budget)

    concurrency = cfg.get("max_concurrency")
    limit = cfg.get("max_concurrency_limit", MAX_CONCURRENCY_LIMIT)
    if not is_int(limit) or limit != MAX_CONCURRENCY_LIMIT:
        cfg["max_concurrency_limit"] = MAX_CONCURRENCY_LIMIT
        limit = MAX_CONCURRENCY_LIMIT
    if not is_int(concurrency) or not (1 <= concurrency <= limit):
        reject("max_concurrency must be between 1 and the hard upper limit", value=concurrency, upper=limit)

    if issues:
        return None, issues
    cfg["config_hash"] = config_hash(cfg)
    return cfg, []


def load_config(path: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    raw, issues = load_json(path)
    if issues:
        return None, issues
    return validate_config(raw)


def config_hash(config: dict[str, Any]) -> str:
    material = {key: value for key, value in config.items() if key != "config_hash"}
    return fingerprint(material)


def parameter_fingerprint(config: dict[str, Any], fields: list[str], *versions: str) -> str:
    material: dict[str, Any] = {"versions": list(versions)}
    for field in fields:
        material[field] = config.get(field)
    return fingerprint(material)


def quant_scheme_name(config: dict[str, Any]) -> str:
    return f"PV_Q{config['fixed_decimals']}_L{config['quant_levels']}_{config['quant_scheme']}"


def fixed_scale(decimals: int) -> int:
    return 10**decimals


def round_half_up(value: Decimal | str | int, digits: int = 0) -> int | Decimal:
    quant = Decimal(1).scaleb(-digits)
    result = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    if digits == 0:
        return int(result)
    return result


def permille_to_fixed(permille: int, decimals: int) -> int:
    scale = fixed_scale(decimals)
    return int(
        (Decimal(int(permille)) * Decimal(scale) / Decimal(1000)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )


def fixed_to_permille(value_fixed: int, decimals: int) -> int:
    scale = fixed_scale(decimals)
    return int(
        (Decimal(int(value_fixed)) * Decimal(1000) / Decimal(scale)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )
