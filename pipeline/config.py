from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .errors import ConfigError


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}")
    if maximum is not None:
        _require(result <= maximum, f"{name} must be <= {maximum}")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    if minimum is not None:
        _require(value >= minimum, f"{name} must be >= {minimum}")
    return value


def validate_config(config: dict) -> dict:
    _require(isinstance(config, dict), "configuration root must be an object")
    _require(isinstance(config.get("pipeline_version"), str) and config["pipeline_version"].strip(), "pipeline_version must be non-empty")

    normalization = config.get("normalization")
    _require(isinstance(normalization, dict), "normalization must be an object")
    for key in ("casefold", "nfkc", "collapse_whitespace"):
        _require(isinstance(normalization.get(key), bool), f"normalization.{key} must be boolean")
    pattern = normalization.get("token_pattern")
    _require(isinstance(pattern, str) and pattern, "normalization.token_pattern must be non-empty")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ConfigError(f"normalization.token_pattern is invalid: {exc}") from exc

    allowed = config.get("allowed_languages")
    _require(isinstance(allowed, list) and allowed and all(isinstance(item, str) and item for item in allowed), "allowed_languages must be a non-empty string list")
    _require(len(set(allowed)) == len(allowed), "allowed_languages must be unique")
    index_languages = config.get("index_languages")
    _require(isinstance(index_languages, list) and index_languages and set(index_languages).issubset(allowed), "index_languages must be a non-empty subset of allowed_languages")
    _require(len(index_languages) == 2 and set(index_languages) == {"zh", "en"}, "index_languages must be exactly [zh, en] for the bilingual bridge")

    lid = config.get("language_detection")
    _require(isinstance(lid, dict), "language_detection must be an object")
    _require(isinstance(lid.get("enabled"), bool), "language_detection.enabled must be boolean")
    _number(lid.get("mixed_ratio_threshold"), "language_detection.mixed_ratio_threshold", minimum=0.0, maximum=1.0)

    translation = config.get("translation")
    _require(isinstance(translation, dict), "translation must be an object")
    _number(translation.get("bridge_weight"), "translation.bridge_weight", minimum=0.0, maximum=1.0)

    bm25 = config.get("bm25")
    _require(isinstance(bm25, dict), "bm25 must be an object")
    _number(bm25.get("k1"), "bm25.k1", minimum=1e-12)
    _number(bm25.get("b"), "bm25.b", minimum=0.0, maximum=1.0)
    _integer(bm25.get("title_repeat"), "bm25.title_repeat", minimum=0)

    _integer(config.get("recall_limit"), "recall_limit", minimum=1)

    fusion = config.get("fusion")
    _require(isinstance(fusion, dict), "fusion must be an object")
    mono_weight = _number(fusion.get("mono_weight"), "fusion.mono_weight", minimum=0.0)
    cross_weight = _number(fusion.get("cross_weight"), "fusion.cross_weight", minimum=0.0)
    _require(mono_weight + cross_weight > 0.0, "fusion weights must not both be zero")
    _number(fusion.get("temperature"), "fusion.temperature", minimum=1e-12)

    fixed = config.get("fixed_point")
    _require(isinstance(fixed, dict), "fixed_point must be an object")
    bits = _integer(fixed.get("bits"), "fixed_point.bits", minimum=8)
    _require(bits <= 64, "fixed_point.bits must be <= 64")
    _integer(fixed.get("scale"), "fixed_point.scale", minimum=1)

    max_grade = _integer(config.get("max_relevance_grade"), "max_relevance_grade", minimum=1)
    _require(max_grade <= 255, "max_relevance_grade must be <= 255")
    thresholds = config.get("relevance_thresholds")
    _require(isinstance(thresholds, list) and thresholds, "relevance_thresholds must be non-empty")
    previous_score = -math.inf
    grades = []
    for index, item in enumerate(thresholds):
        _require(isinstance(item, dict), f"relevance_thresholds[{index}] must be an object")
        score = _number(item.get("min_score"), f"relevance_thresholds[{index}].min_score", minimum=0.0, maximum=1.0)
        grade = _integer(item.get("grade"), f"relevance_thresholds[{index}].grade", minimum=0)
        _require(grade <= max_grade, f"relevance_thresholds[{index}].grade exceeds max_relevance_grade")
        _require(score >= previous_score, "relevance_thresholds must be sorted by min_score")
        previous_score = score
        grades.append(grade)
    _require(len(set(grades)) == len(grades), "relevance_threshold grades must be unique")
    _require(0 in grades, "relevance_thresholds must contain grade 0")
    _require(max_grade in grades, "relevance_thresholds must contain max_relevance_grade")

    evaluation = config.get("evaluation")
    _require(isinstance(evaluation, dict), "evaluation must be an object")
    metrics = evaluation.get("metrics")
    _require(isinstance(metrics, list) and metrics, "evaluation.metrics must be non-empty")
    supported = {"precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k", "grade_mae", "grade_accuracy"}
    _require(set(metrics).issubset(supported), f"evaluation.metrics must be subset of {sorted(supported)}")
    _require(len(set(metrics)) == len(metrics), "evaluation.metrics must be unique")
    _integer(evaluation.get("k"), "evaluation.k", minimum=1)

    threads = config.get("threads")
    _require(isinstance(threads, dict), "threads must be an object")
    _integer(threads.get("max_workers"), "threads.max_workers", minimum=1)

    budget = config.get("budget")
    _require(isinstance(budget, dict), "budget must be an object")
    _integer(budget.get("global_retrieval_points"), "budget.global_retrieval_points", minimum=0)
    _integer(budget.get("per_query_quota"), "budget.per_query_quota", minimum=0)

    snapshot = config.get("snapshot")
    _require(isinstance(snapshot, dict), "snapshot must be an object")
    _integer(snapshot.get("batch_size"), "snapshot.batch_size", minimum=1)
    return config


def load_config(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {path}: {exc}") from exc
    validate_config(config)
    return config, sha256_json(config)


def with_overrides(config: dict, *, max_workers: int | None = None, global_budget: int | None = None, per_query_quota: int | None = None) -> tuple[dict, str]:
    result = copy.deepcopy(config)
    if max_workers is not None:
        result["threads"]["max_workers"] = int(max_workers)
    if global_budget is not None:
        result["budget"]["global_retrieval_points"] = int(global_budget)
    if per_query_quota is not None:
        result["budget"]["per_query_quota"] = int(per_query_quota)
    validate_config(result)
    return result, sha256_json(result)

