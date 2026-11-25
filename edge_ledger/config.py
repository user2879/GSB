from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import digest
from .errors import Failure
from .models import ALLOWED_DECIMAL_PLACES, QUANTIZATION_STEPS


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "dedupe": {
        "threshold_permille": 720,
        "subword_size": 3,
    },
    "quality": {
        "weights_permille": {
            "completeness": 400,
            "template_coverage": 300,
            "answer_length": 300,
        },
        "answer_soft_limit_chars": 24,
        "answer_decay_per_char": 4_000,
        "answer_floor": 100_000,
    },
    "fixed_point": {
        "decimal_places": 4,
        "quantization": "Q1000",
    },
    "scheduler": {
        "max_workers": 4,
        "render_point_budget": 10_000,
        "source_quota_points": 10_000,
        "points_per_render": 1,
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    dedupe_threshold_permille: int
    subword_size: int
    weights_permille: dict[str, int]
    answer_soft_limit_chars: int
    answer_decay_per_char: int
    answer_floor: int
    decimal_places: int
    quantization: str
    quantum_step: int
    max_workers: int
    render_point_budget: int
    source_quota_points: int
    points_per_render: int
    normalized: dict[str, Any]

    @property
    def config_hash(self) -> str:
        return digest(self.normalized)

    @property
    def quality_fingerprint(self) -> str:
        return digest(
            {
                "weights_permille": self.weights_permille,
                "answer_soft_limit_chars": self.answer_soft_limit_chars,
                "answer_decay_per_char": self.answer_decay_per_char,
                "answer_floor": self.answer_floor,
                "decimal_places": self.decimal_places,
                "quantization": self.quantization,
            }
        )

    @property
    def dedupe_fingerprint(self) -> str:
        return digest(
            {
                "threshold_permille": self.dedupe_threshold_permille,
                "subword_size": self.subword_size,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.normalized)


def _merge_defaults(raw: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)

    def merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, Mapping) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(merged, raw)
    return merged


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_config(raw: Mapping[str, Any] | None) -> tuple[PipelineConfig, list[Failure]]:
    failures: list[Failure] = []
    source = raw or {}
    merged = _merge_defaults(source)

    allowed_top = {"schema_version", "dedupe", "quality", "fixed_point", "scheduler"}
    for key in sorted(merged):
        if key not in allowed_top:
            failures.append(
                Failure(
                    code="E_CONFIG_INVALID",
                    stage="load_config",
                    message=f"unknown configuration key: {key}",
                    target_kind="config",
                    target_id=key,
                )
            )

    def fail(path: str, message: str) -> None:
        failures.append(
            Failure(
                code="E_CONFIG_INVALID",
                stage="load_config",
                message=f"{path}: {message}",
                target_kind="config",
                target_id=path,
            )
        )

    schema_version = merged.get("schema_version")
    if not _is_int(schema_version) or schema_version != 1:
        fail("schema_version", "must be integer 1")

    dedupe = merged.get("dedupe")
    if not isinstance(dedupe, dict):
        fail("dedupe", "must be a mapping")
        dedupe = copy.deepcopy(DEFAULT_CONFIG["dedupe"])
    threshold = dedupe.get("threshold_permille")
    if not _is_int(threshold) or not 0 <= threshold <= 1000:
        fail("dedupe.threshold_permille", "must be an integer from 0 to 1000")
        threshold = DEFAULT_CONFIG["dedupe"]["threshold_permille"]
    subword_size = dedupe.get("subword_size")
    if not _is_int(subword_size) or not 1 <= subword_size <= 8:
        fail("dedupe.subword_size", "must be an integer from 1 to 8")
        subword_size = DEFAULT_CONFIG["dedupe"]["subword_size"]

    quality = merged.get("quality")
    if not isinstance(quality, dict):
        fail("quality", "must be a mapping")
        quality = copy.deepcopy(DEFAULT_CONFIG["quality"])
    weights = quality.get("weights_permille")
    if not isinstance(weights, dict):
        fail("quality.weights_permille", "must be a mapping")
        weights = copy.deepcopy(DEFAULT_CONFIG["quality"]["weights_permille"])
    expected_weight_keys = {"completeness", "template_coverage", "answer_length"}
    for key in sorted(weights):
        if key not in expected_weight_keys:
            fail(f"quality.weights_permille.{key}", "unknown weight")
    for key in sorted(expected_weight_keys):
        value = weights.get(key)
        if not _is_int(value) or value < 0:
            fail(f"quality.weights_permille.{key}", "must be a non-negative integer")
    weights_valid = all(
        _is_int(weights.get(key)) and weights.get(key) >= 0
        for key in expected_weight_keys
    )
    if weights_valid:
        if sum(int(weights[key]) for key in expected_weight_keys) != 1000:
            fail("quality.weights_permille", "weights must sum to 1000")
            weights = copy.deepcopy(DEFAULT_CONFIG["quality"]["weights_permille"])
    else:
        weights = copy.deepcopy(DEFAULT_CONFIG["quality"]["weights_permille"])

    answer_soft_limit = quality.get("answer_soft_limit_chars")
    if not _is_int(answer_soft_limit) or answer_soft_limit < 0:
        fail("quality.answer_soft_limit_chars", "must be a non-negative integer")
        answer_soft_limit = DEFAULT_CONFIG["quality"]["answer_soft_limit_chars"]
    answer_decay = quality.get("answer_decay_per_char")
    if not _is_int(answer_decay) or answer_decay < 0:
        fail("quality.answer_decay_per_char", "must be a non-negative integer")
        answer_decay = DEFAULT_CONFIG["quality"]["answer_decay_per_char"]
    answer_floor = quality.get("answer_floor")
    if not _is_int(answer_floor) or not 0 <= answer_floor <= 1_000_000:
        fail("quality.answer_floor", "must be an integer from 0 to 1000000")
        answer_floor = DEFAULT_CONFIG["quality"]["answer_floor"]

    fixed_point = merged.get("fixed_point")
    if not isinstance(fixed_point, dict):
        fail("fixed_point", "must be a mapping")
        fixed_point = copy.deepcopy(DEFAULT_CONFIG["fixed_point"])
    decimal_places = fixed_point.get("decimal_places")
    if not _is_int(decimal_places) or decimal_places not in ALLOWED_DECIMAL_PLACES:
        fail("fixed_point.decimal_places", f"must be one of {list(ALLOWED_DECIMAL_PLACES)}")
        decimal_places = DEFAULT_CONFIG["fixed_point"]["decimal_places"]
    quantization = fixed_point.get("quantization")
    if quantization not in QUANTIZATION_STEPS:
        fail("fixed_point.quantization", f"must be one of {sorted(QUANTIZATION_STEPS)}")
        quantization = DEFAULT_CONFIG["fixed_point"]["quantization"]

    scheduler = merged.get("scheduler")
    if not isinstance(scheduler, dict):
        fail("scheduler", "must be a mapping")
        scheduler = copy.deepcopy(DEFAULT_CONFIG["scheduler"])
    max_workers = scheduler.get("max_workers")
    if not _is_int(max_workers) or not 1 <= max_workers <= 64:
        fail("scheduler.max_workers", "must be an integer from 1 to 64")
        max_workers = DEFAULT_CONFIG["scheduler"]["max_workers"]
    render_budget = scheduler.get("render_point_budget")
    if not _is_int(render_budget) or render_budget < 0:
        fail("scheduler.render_point_budget", "must be a non-negative integer")
        render_budget = DEFAULT_CONFIG["scheduler"]["render_point_budget"]
    source_quota = scheduler.get("source_quota_points")
    if not _is_int(source_quota) or source_quota < 0:
        fail("scheduler.source_quota_points", "must be a non-negative integer")
        source_quota = DEFAULT_CONFIG["scheduler"]["source_quota_points"]
    points_per_render = scheduler.get("points_per_render")
    if not _is_int(points_per_render) or points_per_render < 1:
        fail("scheduler.points_per_render", "must be a positive integer")
        points_per_render = DEFAULT_CONFIG["scheduler"]["points_per_render"]

    normalized = {
        "schema_version": 1,
        "dedupe": {
            "threshold_permille": threshold,
            "subword_size": subword_size,
        },
        "quality": {
            "weights_permille": {
                "completeness": weights["completeness"],
                "template_coverage": weights["template_coverage"],
                "answer_length": weights["answer_length"],
            },
            "answer_soft_limit_chars": answer_soft_limit,
            "answer_decay_per_char": answer_decay,
            "answer_floor": answer_floor,
        },
        "fixed_point": {
            "decimal_places": decimal_places,
            "quantization": quantization,
        },
        "scheduler": {
            "max_workers": max_workers,
            "render_point_budget": render_budget,
            "source_quota_points": source_quota,
            "points_per_render": points_per_render,
        },
    }
    config = PipelineConfig(
        schema_version=1,
        dedupe_threshold_permille=threshold,
        subword_size=subword_size,
        weights_permille={
            "completeness": weights["completeness"],
            "template_coverage": weights["template_coverage"],
            "answer_length": weights["answer_length"],
        },
        answer_soft_limit_chars=answer_soft_limit,
        answer_decay_per_char=answer_decay,
        answer_floor=answer_floor,
        decimal_places=decimal_places,
        quantization=quantization,
        quantum_step=QUANTIZATION_STEPS[quantization],
        max_workers=max_workers,
        render_point_budget=render_budget,
        source_quota_points=source_quota,
        points_per_render=points_per_render,
        normalized=normalized,
    )
    return config, sorted(failures, key=Failure.sort_key)
