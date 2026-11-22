"""配置加载、校验与规范化哈希。"""
import hashlib
import json
import os

from .errors import EGWError, E_CONFIG_INVALID

VALID_ROUNDING = ("TRUNCATE", "FLOOR", "HALF_UP", "HALF_EVEN")
VALID_AGG_MODES = ("SUM", "MEAN")
DEFAULT_AGG_MODES = ["SUM", "MEAN"]


def _load_raw(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise EGWError(E_CONFIG_INVALID, "配置根节点必须是对象")
    return data


def _require_positive_int(data, key):
    if key not in data:
        raise EGWError(E_CONFIG_INVALID, f"缺少必填配置项 {key}", {"field": key})
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EGWError(
            E_CONFIG_INVALID,
            f"配置项 {key} 必须为正整数，实际为 {value!r}",
            {"field": key},
        )
    return value


def load_config(path):
    data = _load_raw(path)
    cfg = {}
    try:
        cfg["dim"] = _require_positive_int(data, "dim")
        cfg["quant_level"] = _require_positive_int(data, "quant_level")
        cfg["fractional_bits"] = _require_positive_int(data, "fractional_bits")
        cfg["concurrency"] = _require_positive_int(data, "concurrency")
        cfg["point_budget"] = _require_positive_int(data, "point_budget")
    except EGWError:
        raise

    headroom = data.get("accumulator_headroom_bits", 4)
    if isinstance(headroom, bool) or not isinstance(headroom, int) or headroom <= 0:
        raise EGWError(
            E_CONFIG_INVALID,
            "accumulator_headroom_bits 必须为正整数",
            {"field": "accumulator_headroom_bits"},
        )
    cfg["accumulator_headroom_bits"] = headroom

    rounding = data.get("rounding", "HALF_UP")
    if rounding not in VALID_ROUNDING:
        raise EGWError(
            E_CONFIG_INVALID,
            f"未知舍入规则 {rounding!r}，允许值 {VALID_ROUNDING}",
            {"field": "rounding"},
        )
    cfg["rounding"] = rounding

    agg_modes = data.get("agg_modes", list(DEFAULT_AGG_MODES))
    if not isinstance(agg_modes, list) or not agg_modes:
        raise EGWError(E_CONFIG_INVALID, "agg_modes 必须是非空数组", {"field": "agg_modes"})
    for mode in agg_modes:
        if mode not in VALID_AGG_MODES:
            raise EGWError(
                E_CONFIG_INVALID,
                f"聚合模式枚举含非法值 {mode!r}",
                {"field": "agg_modes"},
            )
    cfg["agg_modes"] = agg_modes

    tolerance = data.get("tolerance", 1e-3)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise EGWError(E_CONFIG_INVALID, "tolerance 必须为正数", {"field": "tolerance"})
    cfg["tolerance"] = float(tolerance)

    top_k = data.get("top_k", 3)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise EGWError(E_CONFIG_INVALID, "top_k 必须为正整数", {"field": "top_k"})
    cfg["top_k"] = top_k

    alt = data.get("alternate_gear")
    if alt is not None:
        if not isinstance(alt, dict):
            raise EGWError(E_CONFIG_INVALID, "alternate_gear 必须是对象", {"field": "alternate_gear"})
        alt_level = alt.get("quant_level")
        alt_bits = alt.get("fractional_bits")
        if (
            isinstance(alt_level, bool) or not isinstance(alt_level, int) or alt_level <= 0
            or isinstance(alt_bits, bool) or not isinstance(alt_bits, int) or alt_bits <= 0
        ):
            raise EGWError(
                E_CONFIG_INVALID,
                "alternate_gear 的 quant_level/fractional_bits 必须为正整数",
                {"field": "alternate_gear"},
            )
        cfg["alternate_gear"] = {"quant_level": alt_level, "fractional_bits": alt_bits}

    cfg.setdefault("name", data.get("name", "embedding_gather_workbench"))
    return cfg


def canonical_bytes(cfg):
    """规范化配置字节（剔除易变字段，排序键），用于配置哈希与续跑判定。"""
    stable = {k: v for k, v in cfg.items() if k != "name"}
    return json.dumps(stable, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def config_hash(cfg):
    return hashlib.sha256(canonical_bytes(cfg)).hexdigest()


def gear_key(quant_level, fractional_bits):
    return f"L{quant_level}Q{fractional_bits}"
