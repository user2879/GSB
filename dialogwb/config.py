from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .util import deep_merge, fingerprint, round_weight


class ConfigError(ValueError):
    pass


def normalize_config(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError("config must be a JSON object")

    compression_window = _positive_int(
        value.get("compression_window", 3), "compression_window"
    )
    max_summary_length = _positive_int(
        value.get("max_summary_length", 80), "max_summary_length"
    )
    max_candidates = _positive_int(
        value.get("max_candidates_per_round", 4), "max_candidates_per_round"
    )
    snapshot_interval = _positive_int(
        value.get("snapshot_interval", 1), "snapshot_interval"
    )
    concurrency = _positive_int(value.get("concurrency", 1), "concurrency")

    threshold = value.get("dedupe_similarity_threshold", 0.6)
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ConfigError("dedupe_similarity_threshold must be between 0 and 1")

    point_weights = value.get("point_weights", {})
    if not isinstance(point_weights, dict):
        raise ConfigError("point_weights must be an object")
    base = _non_negative_float(point_weights.get("base", 1.0), "point_weights.base")
    decay = _non_negative_float(
        point_weights.get("position_decay", 0.85), "point_weights.position_decay"
    )
    if decay > 1.0:
        raise ConfigError("point_weights.position_decay must be <= 1")
    sentence_bonus = _non_negative_float(
        point_weights.get("sentence_bonus", 0.0), "point_weights.sentence_bonus"
    )
    term_weights_value = point_weights.get("term_weights", {})
    if not isinstance(term_weights_value, dict):
        raise ConfigError("point_weights.term_weights must be an object")
    term_weights: dict[str, float] = {}
    for key in sorted(term_weights_value):
        if not isinstance(key, str) or not key:
            raise ConfigError("term weight keys must be non-empty strings")
        term_weights[key] = _non_negative_float(
            term_weights_value[key], f"point_weights.term_weights.{key}"
        )

    raw_templates = value.get("template_weights", [])
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ConfigError("template_weights must be a non-empty array")
    templates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in raw_templates:
        if not isinstance(item, dict):
            raise ConfigError("template_weights entries must be objects")
        template_id = item.get("id")
        template_format = item.get("format")
        if not isinstance(template_id, str) or not template_id:
            raise ConfigError("template id must be a non-empty string")
        if template_id in seen_ids:
            raise ConfigError(f"duplicate template id: {template_id}")
        if not isinstance(template_format, str) or "{point}" not in template_format:
            raise ConfigError(f"template {template_id} must contain {{point}}")
        seen_ids.add(template_id)
        templates.append(
            {
                "id": template_id,
                "format": template_format,
                "weight": _non_negative_float(
                    item.get("weight", 1.0), f"template_weights.{template_id}.weight"
                ),
            }
        )
    if sum(item["weight"] for item in templates) <= 0:
        raise ConfigError("template weights must have a positive sum")
    templates.sort(key=lambda item: item["id"])

    return {
        "compression_window": compression_window,
        "max_summary_length": max_summary_length,
        "max_candidates_per_round": max_candidates,
        "dedupe_similarity_threshold": round_weight(threshold),
        "snapshot_interval": snapshot_interval,
        "concurrency": concurrency,
        "point_weights": {
            "base": round_weight(base),
            "position_decay": round_weight(decay),
            "sentence_bonus": round_weight(sentence_bonus),
            "term_weights": {
                key: round_weight(term_weights[key]) for key in sorted(term_weights)
            },
        },
        "template_weights": [
            {
                "id": item["id"],
                "format": item["format"],
                "weight": round_weight(item["weight"]),
            }
            for item in templates
        ],
    }


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load config: {exc}") from exc
    return normalize_config(value)


def merge_config(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ConfigError("config patch must be an object")
    merged = deep_merge(base, patch)
    if "point_weights" in patch and isinstance(patch["point_weights"], dict):
        base_terms = base.get("point_weights", {}).get("term_weights", {})
        patch_terms = patch["point_weights"].get("term_weights", {})
        if isinstance(base_terms, dict) and isinstance(patch_terms, dict):
            merged["point_weights"]["term_weights"] = {
                **base_terms,
                **patch_terms,
            }
    return normalize_config(merged)


def extraction_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "compression_window": config["compression_window"],
        "max_candidates_per_round": config["max_candidates_per_round"],
        "point_weights": config["point_weights"],
    }


def merge_config_view(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "dedupe_similarity_threshold": config["dedupe_similarity_threshold"],
    }


def rewrite_config_view(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_summary_length": config["max_summary_length"],
        "template_weights": config["template_weights"],
    }


def config_hash(config: dict[str, Any]) -> str:
    return fingerprint(normalize_config(config))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if float(value) < 0:
        raise ConfigError(f"{name} must be non-negative")
    return float(value)

