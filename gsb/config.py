from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .canonical import fingerprint
from .errors import InputValidationError
from .fixedpoint import quantize_float, validate_fixed_point_config


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputValidationError(f"{field} must be an object", field=field)
    return value


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"{field} must be an integer", field=field)
    if minimum is not None and value < minimum:
        raise InputValidationError(f"{field} must be >= {minimum}", field=field)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import json

        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputValidationError(f"config file not found: {path}", field="config") from exc
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"config JSON is invalid: {exc.msg}",
            field=f"config:{path}:line-{exc.lineno}",
        ) from exc
    return validate_config(config)


def validate_config(raw: Any) -> dict[str, Any]:
    config = _require_object(raw, "config")

    version = config.get("config_version")
    if not isinstance(version, str) or not version.strip():
        raise InputValidationError(
            "config.config_version must be a nonempty string",
            field="config_version",
        )

    domains_raw = config.get("domains")
    if not isinstance(domains_raw, list) or not domains_raw:
        raise InputValidationError("domains must be a nonempty list", field="domains")

    width, tier = validate_fixed_point_config(config)

    domains: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    weight_sum = 0.0
    target_sum = 0.0
    weight_scale = 10**width
    target_scale_total = 0

    for index, domain in enumerate(domains_raw):
        base = f"domains[{index}]"
        domain = _require_object(domain, base)
        name = domain.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InputValidationError(f"{base}.name must be a nonempty string", field=f"{base}.name")
        if name in seen_names:
            raise InputValidationError(f"duplicate domain {name!r}", field=f"{base}.name")
        seen_names.add(name)

        weight = domain.get("sampling_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise InputValidationError(
                f"{base}.sampling_weight must be a nonnegative finite number",
                field=f"domains.{name}.sampling_weight",
            )
        weight_float = float(weight)
        if not math.isfinite(weight_float) or weight_float < 0:
            raise InputValidationError(
                f"{base}.sampling_weight must be a nonnegative finite number",
                field=f"domains.{name}.sampling_weight",
            )
        weight_fp = quantize_float(weight_float, width, tier)
        if weight_fp < 0:
            raise InputValidationError(
                "sampling_weight quantizes to a negative fixed-point value",
                field=f"domains.{name}.sampling_weight",
            )
        weight_sum += weight_float

        target = domain.get("target_ratio")
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise InputValidationError(
                f"{base}.target_ratio must be a finite number in [0, 1]",
                field=f"domains.{name}.target_ratio",
            )
        target_float = float(target)
        if not math.isfinite(target_float) or not 0.0 <= target_float <= 1.0:
            raise InputValidationError(
                f"{base}.target_ratio must be a finite number in [0, 1]",
                field=f"domains.{name}.target_ratio",
            )
        target_fp = quantize_float(target_float, width, tier)
        target_scale_total += target_fp
        target_sum += target_float

        quota = domain.get("quota")
        if quota is not None:
            _require_int(quota, f"domains.{name}.quota", minimum=1)

        domains.append(
            {
                "name": name,
                "sampling_weight": weight_float,
                "sampling_weight_fp": weight_fp,
                "target_ratio": target_float,
                "target_ratio_fp": target_fp,
                "quota": quota,
            }
        )

    if not weight_sum > 0:
        raise InputValidationError(
            "sum of domain sampling_weight values must be positive",
            field="domains.sampling_weight",
        )
    if target_scale_total != weight_scale:
        raise InputValidationError(
            "quantized target_ratio values must sum exactly to 1.0 at fixed-point scale",
            field="domains.target_ratio",
            details={
                "quantized_sum": target_scale_total,
                "expected": weight_scale,
                "decimal_width": width,
            },
        )

    bitmap_width = _require_int(
        config.get("missing_bitmap_width"), "missing_bitmap_width", minimum=1
    )

    rules_raw = config.get("missing_rules")
    if not isinstance(rules_raw, list):
        raise InputValidationError("missing_rules must be a list", field="missing_rules")
    rules: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for index, rule in enumerate(rules_raw):
        rule_field = f"missing_rules[{index}]"
        rule = _require_object(rule, rule_field)
        code = rule.get("code")
        if not isinstance(code, str) or not code.strip():
            raise InputValidationError(f"{rule_field}.code must be a nonempty string", field=f"{rule_field}.code")
        if code in seen_codes:
            raise InputValidationError(f"duplicate missing rule code {code!r}", field=f"missing_rules.{code}")
        seen_codes.add(code)
        position = _require_int(rule.get("position"), f"missing_rules.{code}.position", minimum=0)
        if position >= bitmap_width:
            raise InputValidationError(
                f"missing_rules.{code}.position must be < missing_bitmap_width",
                field=f"missing_rules.{code}.position",
            )
        config_field = rule.get("config_field")
        if config_field is None:
            config_field = f"missing_rules[{index}].position"
        elif not isinstance(config_field, str) or not config_field:
            raise InputValidationError(
                f"missing_rules.{code}.config_field must be a nonempty string",
                field=f"missing_rules.{code}.config_field",
            )
        rules.append(
            {
                "code": code,
                "position": position,
                "config_field": config_field,
            }
        )

    cleaning = _require_object(config.get("cleaning"), "cleaning")
    min_tokens = _require_int(
        cleaning.get("min_total_token_count"),
        "cleaning.min_total_token_count",
        minimum=0,
    )

    dedup = _require_object(config.get("dedup"), "dedup")
    threshold = _require_int(
        dedup.get("similarity_threshold_permille"),
        "dedup.similarity_threshold_permille",
        minimum=0,
    )
    if threshold > 1000:
        raise InputValidationError(
            "dedup.similarity_threshold_permille must be between 0 and 1000",
            field="dedup.similarity_threshold_permille",
        )
    shard_size = _require_int(dedup.get("shard_size"), "dedup.shard_size", minimum=1)

    score_weights = dedup.get("score_weights_permille")
    if not isinstance(score_weights, dict):
        raise InputValidationError(
            "dedup.score_weights_permille must be an object",
            field="dedup.score_weights_permille",
        )
    score_keys = ("token_volume", "completeness", "domain_weight")
    score_values = []
    for key in score_keys:
        value = score_weights.get(key)
        _require_int(value, f"dedup.score_weights_permille.{key}", minimum=0)
        score_values.append(value)
    if sum(score_values) != 1000:
        raise InputValidationError(
            "dedup.score_weights_permille must sum to 1000",
            field="dedup.score_weights_permille",
            details={"sum": sum(score_values)},
        )
    token_cap = _require_int(dedup.get("score_token_cap"), "dedup.score_token_cap", minimum=1)

    sampling = _require_object(config.get("sampling"), "sampling")
    sample_count = _require_int(sampling.get("sample_count"), "sampling.sample_count", minimum=1)
    seed = _require_int(sampling.get("seed"), "sampling.seed", minimum=0)

    concurrency = _require_object(config.get("concurrency"), "concurrency")
    max_workers = _require_int(concurrency.get("max_workers"), "concurrency.max_workers", minimum=1)
    cleaning_workers = _require_int(
        concurrency.get("cleaning_workers"), "concurrency.cleaning_workers", minimum=1
    )
    dedup_workers = _require_int(concurrency.get("dedup_workers"), "concurrency.dedup_workers", minimum=1)
    if cleaning_workers > max_workers or dedup_workers > max_workers:
        raise InputValidationError(
            "stage worker counts must not exceed concurrency.max_workers",
            field="concurrency",
            details={
                "max_workers": max_workers,
                "cleaning_workers": cleaning_workers,
                "dedup_workers": dedup_workers,
            },
        )

    budgets = _require_object(config.get("budgets"), "budgets")
    global_units = _require_int(budgets.get("global_units"), "budgets.global_units", minimum=0)
    domain_quota = _require_int(budgets.get("domain_quota"), "budgets.domain_quota", minimum=1)
    comparison_limit = _require_int(
        budgets.get("shard_comparison_limit"),
        "budgets.shard_comparison_limit",
        minimum=1,
    )

    normalized = {
        "config_version": version,
        "domains": domains,
        "domain_names": [item["name"] for item in domains],
        "domain_map": {item["name"]: item for item in domains},
        "weight_total_fp": sum(item["sampling_weight_fp"] for item in domains),
        "missing_bitmap_width": bitmap_width,
        "missing_rules": rules,
        "cleaning": {"min_total_token_count": min_tokens},
        "dedup": {
            "similarity_threshold_permille": threshold,
            "shard_size": shard_size,
            "score_weights_permille": {key: value for key, value in zip(score_keys, score_values)},
            "score_token_cap": token_cap,
        },
        "sampling": {"sample_count": sample_count, "seed": seed},
        "fixed_point": {"decimal_width": width, "quantization_tier": tier},
        "concurrency": {
            "max_workers": max_workers,
            "cleaning_workers": cleaning_workers,
            "dedup_workers": dedup_workers,
        },
        "budgets": {
            "global_units": global_units,
            "domain_quota": domain_quota,
            "shard_comparison_limit": comparison_limit,
        },
    }
    normalized["config_hash"] = fingerprint(_hashable_config(normalized))
    return normalized


def _hashable_config(normalized: dict[str, Any]) -> dict[str, Any]:
    """Project normalized config to JSON-safe content for fingerprinting."""

    return {
        key: value
        for key, value in normalized.items()
        if key not in {"domain_map", "config_hash"}
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    return config.get("config_hash") or fingerprint(_hashable_config(config))


def scoped_config_fingerprint(config: dict[str, Any], scope: dict[str, Any]) -> str:
    base = config_fingerprint(config)
    return fingerprint({"global_config_hash": base, "scope": scope})
