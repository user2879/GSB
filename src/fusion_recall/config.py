from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from .errors import E_INPUT_INVALID, ConfigValidationError
from .hashing import canonical_hash, composite_key


ALLOWED_FRAC_BITS = (8, 16, 32)
ALLOWED_QUANT_TIERS = ("floor", "nearest", "ceil")


def is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def rational_pair(value: Any) -> tuple[int, int]:
    """Return an exact positive numerator/denominator pair."""

    if isinstance(value, bool):
        raise ConfigValidationError("boolean is not a valid weight")
    if isinstance(value, int):
        if value <= 0:
            raise ConfigValidationError("weight must be positive")
        return value, 1
    if isinstance(value, float):
        value = str(value)
    if isinstance(value, str) and "/" in value:
        left, right = value.split("/", 1)
        try:
            numerator, denominator = int(left.strip()), int(right.strip())
        except ValueError as exc:
            raise ConfigValidationError(f"invalid rational weight: {value!r}") from exc
        if numerator <= 0 or denominator <= 0:
            raise ConfigValidationError("weight must be a finite positive rational")
        return numerator, denominator
    if not isinstance(value, str):
        raise ConfigValidationError("weight must be a decimal string or integer")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise ConfigValidationError(f"invalid decimal weight: {value!r}") from exc
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise ConfigValidationError("weight must be a finite positive decimal")
    ratio = decimal_value.as_integer_ratio()
    numerator, denominator = ratio
    if denominator < 0:
        numerator = -numerator
        denominator = -denominator
    return numerator, denominator


def normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigValidationError("configuration must be an object")

    views_raw = raw.get("views")
    if not isinstance(views_raw, list) or not views_raw:
        raise ConfigValidationError("views must be a non-empty list")
    views: list[dict[str, str]] = []
    seen_view_names: set[str] = set()
    seen_languages: set[str] = set()
    for item in views_raw:
        if not isinstance(item, dict):
            raise ConfigValidationError("each view must be an object")
        name = item.get("name")
        language = item.get("language")
        if not isinstance(name, str) or not name:
            raise ConfigValidationError("view.name must be a non-empty string")
        if not isinstance(language, str) or not language:
            raise ConfigValidationError("view.language must be a non-empty string")
        if name in seen_view_names:
            raise ConfigValidationError(f"duplicate view name: {name}")
        seen_view_names.add(name)
        seen_languages.add(language)
        views.append({"name": name, "language": language})

    allowed_languages = raw.get("allowed_languages", sorted(seen_languages))
    if not isinstance(allowed_languages, list) or not all(
        isinstance(item, str) and item for item in allowed_languages
    ):
        raise ConfigValidationError("allowed_languages must be a list of strings")
    allowed_language_set = set(allowed_languages)
    if not seen_languages.issubset(allowed_language_set):
        raise ConfigValidationError("view languages must be in allowed_languages")

    field_weights_raw = raw.get("field_weights")
    if not isinstance(field_weights_raw, dict) or not field_weights_raw:
        raise ConfigValidationError("field_weights must be a non-empty object")
    field_weights: dict[str, int] = {}
    for field, weight in field_weights_raw.items():
        if not isinstance(field, str) or not field:
            raise ConfigValidationError("field names must be non-empty strings")
        if not is_positive_int(weight):
            raise ConfigValidationError(f"field weight for {field} must be a positive integer")
        field_weights[field] = weight

    field_order_raw = raw.get("field_order", list(field_weights))
    if not isinstance(field_order_raw, list) or not field_order_raw:
        raise ConfigValidationError("field_order must be a non-empty list")
    field_order = [item for item in field_order_raw if isinstance(item, str)]
    if len(field_order) != len(field_order_raw) or set(field_order) != set(field_weights):
        raise ConfigValidationError("field_order must exactly match field_weights")

    top_k = raw.get("top_k", 3)
    if not is_positive_int(top_k):
        raise ConfigValidationError("top_k must be a positive integer")

    saturation_k = raw.get("saturation_k", 2)
    if not is_positive_int(saturation_k):
        raise ConfigValidationError("saturation_k must be a positive integer")

    language_weights_raw = raw.get("language_weights", {view["name"]: 1 for view in views})
    if not isinstance(language_weights_raw, dict):
        raise ConfigValidationError("language_weights must be an object")
    language_weights: dict[str, str] = {}
    language_weight_rational: dict[str, list[int]] = {}
    for view in views:
        name = view["name"]
        value = language_weights_raw.get(name, 1)
        numerator, denominator = rational_pair(value)
        language_weights[name] = f"{numerator}/{denominator}"
        language_weight_rational[name] = [numerator, denominator]
    extra_weights = set(language_weights_raw) - {view["name"] for view in views}
    if extra_weights:
        raise ConfigValidationError(f"unknown language weight views: {sorted(extra_weights)}")

    fixed_raw = raw.get("fixed_point", {})
    if not isinstance(fixed_raw, dict):
        raise ConfigValidationError("fixed_point must be an object")
    frac_bits = fixed_raw.get("frac_bits", 16)
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise ConfigValidationError(
            f"frac_bits must be one of {ALLOWED_FRAC_BITS}, got {frac_bits!r}"
        )
    quant_tier = fixed_raw.get("quant_tier", "nearest")
    if quant_tier not in ALLOWED_QUANT_TIERS:
        raise ConfigValidationError(
            f"quant_tier must be one of {ALLOWED_QUANT_TIERS}, got {quant_tier!r}"
        )
    reference_extra_bits = fixed_raw.get("reference_extra_bits", 8)
    if not is_positive_int(reference_extra_bits) or reference_extra_bits > 32:
        raise ConfigValidationError("reference_extra_bits must be an integer in [1, 32]")

    budget_raw = raw.get("budget", {})
    if not isinstance(budget_raw, dict):
        raise ConfigValidationError("budget must be an object")
    batch_points = budget_raw.get("batch_points", 10_000)
    query_quota_points = budget_raw.get("query_quota_points", 10_000)
    if not is_positive_int(batch_points):
        raise ConfigValidationError("budget.batch_points must be a positive integer")
    if not is_positive_int(query_quota_points):
        raise ConfigValidationError("budget.query_quota_points must be a positive integer")

    normalized = {
        "views": views,
        "view_names": [view["name"] for view in views],
        "allowed_languages": sorted(allowed_language_set),
        "field_weights": field_weights,
        "field_order": field_order,
        "top_k": top_k,
        "saturation_k": saturation_k,
        "language_weights": language_weights,
        "language_weight_rational": language_weight_rational,
        "fixed_point": {
            "frac_bits": frac_bits,
            "quant_tier": quant_tier,
            "reference_extra_bits": reference_extra_bits,
        },
        "budget": {
            "batch_points": batch_points,
            "query_quota_points": query_quota_points,
        },
    }
    normalized["config_fingerprint"] = config_fingerprint(normalized)
    normalized["language_weights_fingerprint"] = language_weights_fingerprint(normalized)
    normalized["fixed_point_fingerprint"] = fixed_point_fingerprint(normalized)
    normalized["full_config_fingerprint"] = full_config_fingerprint(normalized)
    return normalized


def _scoring_material(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "views": config["views"],
        "field_weights": config["field_weights"],
        "field_order": config["field_order"],
        "top_k": config["top_k"],
        "saturation_k": config["saturation_k"],
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    return composite_key(
        "cfg",
        canonical_hash(_scoring_material(config)),
    )


def language_weights_fingerprint(config: dict[str, Any]) -> str:
    return composite_key("langw", canonical_hash(config["language_weight_rational"]))


def fixed_point_fingerprint(config: dict[str, Any]) -> str:
    return composite_key("fixed", canonical_hash(config["fixed_point"]))


def full_config_fingerprint(config: dict[str, Any]) -> str:
    material = deepcopy(config)
    for key in (
        "config_fingerprint",
        "language_weights_fingerprint",
        "fixed_point_fingerprint",
        "full_config_fingerprint",
    ):
        material.pop(key, None)
    return composite_key("fullcfg", canonical_hash(material))


def quantize_div(numerator: int, denominator: int, tier: str) -> int:
    """Quantize a non-negative rational quotient to an integer."""

    if denominator <= 0:
        raise ZeroDivisionError("quantization denominator must be positive")
    if tier == "floor":
        return numerator // denominator
    if tier == "ceil":
        return -((-numerator) // denominator) if numerator < 0 else (
            (numerator + denominator - 1) // denominator
        )
    if tier == "nearest":
        if numerator >= 0:
            return (2 * numerator + denominator) // (2 * denominator)
        return -((-2 * numerator + denominator) // (2 * denominator))
    raise ConfigValidationError(f"unsupported quantization tier: {tier}")


def scale_for_bits(frac_bits: int) -> int:
    return 1 << frac_bits


def validate_config_safe(raw: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        return normalize_config(raw), None
    except ConfigValidationError as exc:
        return None, {
            "code": E_INPUT_INVALID,
            "source": "config",
            "line": None,
            "item_id": None,
            "message": str(exc),
        }
