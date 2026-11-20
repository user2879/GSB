"""Strict experiment, scale-plan and injection configuration handling."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import SCHEMA_VERSION
from .errors import ConfigValidationError, InjectionValidationError, PlanValidationError
from .util import stable_hash


TOP_LEVEL_KEYS = {
    "schema_version",
    "resolution",
    "labels",
    "label_rates",
    "features",
    "classifiers",
    "threshold",
    "precision",
    "precision_levels",
    "concurrency",
    "max_concurrency",
    "snapshot_interval",
    "strategy_candidates",
    "evaluation",
}
FEATURE_KEYS = {
    "supported",
    "color_bins",
    "color_bins_candidates",
    "hog_bins",
    "hog_bins_candidates",
    "hog_cells",
    "expected_dimensions",
}
CLASSIFIER_KEYS = {
    "supported",
    "logistic_regression",
    "nearest_centroid",
    "knn",
}
LR_KEYS = {"learning_rate", "epochs", "l2"}
CENTROID_KEYS = {"metric"}
KNN_KEYS = {"k", "k_candidates", "tie_break", "vote"}
STRATEGY_KEYS = {
    "id",
    "feature_method",
    "color_bins",
    "hog_bins",
    "classifier",
    "knn_k",
    "lr_epochs",
    "learning_rate",
    "l2",
    "threshold_delta",
    "base_score",
    "scale_bonus",
    "imbalance_bonus",
    "label_bonus",
}
EVALUATION_KEYS = {
    "probability_decimals",
    "metric_decimals",
    "boundary_policy",
    "cost_weight",
}
PRECISION_KEYS = {"quantization_levels", "rounding_decimals"}
EXPECTED_KEYS = {"feature_method", "color_bins", "hog_bins", "dimension"}
PLAN_TOP_KEYS = {"schema_version", "tiers"}
PLAN_TIER_KEYS = {"id", "max_samples", "max_labels", "description"}
INJECTION_TOP_KEYS = {"schema_version", "items"}
INJECTION_ITEM_KEYS = {"id", "kind", "sample", "stage", "value", "enabled"}

SUPPORTED_FEATURES = {"color", "hog", "concat"}
SUPPORTED_CLASSIFIERS = {"logistic_regression", "nearest_centroid", "knn"}
BOUNDARY_POLICY = "equality_positive_labels_lexicographic"


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def feature_dimension(
    method: str, color_bins: int, hog_bins: int, hog_cells: tuple[int, int]
) -> int:
    color_dimension = color_bins * 3
    hog_dimension = hog_bins * hog_cells[0] * hog_cells[1]
    if method == "color":
        return color_dimension
    if method == "hog":
        return hog_dimension
    if method == "concat":
        return color_dimension + hog_dimension
    raise ValueError(f"unknown feature method: {method}")


def _unknown_keys(mapping: dict[str, Any], allowed: set[str], where: str) -> list[str]:
    return [
        f"illegal configuration key {where}.{key}"
        for key in sorted(mapping)
        if key not in allowed
    ]


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path
    config_hash: str
    labels: list[str]
    label_rates: dict[str, float]
    resolution: tuple[int, int]
    features: dict[str, Any]
    classifiers: dict[str, Any]
    threshold: float
    precision: str
    precision_levels: dict[str, dict[str, int]]
    concurrency: int
    max_concurrency: int
    snapshot_interval: int
    strategies: list[dict[str, Any]]
    evaluation: dict[str, Any]
    expected_dimensions: dict[tuple[str, int, int], int]

    def precision_info(self, level: str | None = None) -> dict[str, int]:
        selected = level or self.precision
        return dict(self.precision_levels[selected])

    def feature_params(
        self,
        strategy: dict[str, Any] | None = None,
        *,
        precision: str | None = None,
        feature_method: str | None = None,
    ) -> dict[str, Any]:
        features = self.features
        method = feature_method or (strategy or {}).get("feature_method", "color")
        color_bins = int((strategy or {}).get("color_bins", features["color_bins"]))
        hog_bins = int((strategy or {}).get("hog_bins", features["hog_bins"]))
        if feature_method and feature_method != (strategy or {}).get("feature_method"):
            color_bins = int(features["color_bins"])
            hog_bins = int(features["hog_bins"])
        hog_cells = tuple(int(item) for item in features["hog_cells"])
        dimension = feature_dimension(method, color_bins, hog_bins, hog_cells)
        precision_value = self.precision_info(precision)
        return {
            "feature_method": method,
            "color_bins": color_bins,
            "hog_bins": hog_bins,
            "hog_cells": list(hog_cells),
            "feature_dimension": dimension,
            "quantization_levels": precision_value["quantization_levels"],
            "rounding_decimals": precision_value["rounding_decimals"],
            "precision_level": precision or self.precision,
            "rounding_mode": "numpy.rint_then_round_half_even",
            "quantization_mode": "unit_interval_rint_levels",
        }

    def classifier_params(
        self,
        classifier: str,
        strategy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        strategy = strategy or {}
        if classifier == "logistic_regression":
            base = dict(self.classifiers["logistic_regression"])
            if "lr_epochs" in strategy:
                base["epochs"] = int(strategy["lr_epochs"])
            if "learning_rate" in strategy:
                base["learning_rate"] = float(strategy["learning_rate"])
            if "l2" in strategy:
                base["l2"] = float(strategy["l2"])
            return {
                "classifier": classifier,
                "learning_rate": float(base["learning_rate"]),
                "epochs": int(base["epochs"]),
                "l2": float(base["l2"]),
            }
        if classifier == "nearest_centroid":
            base = dict(self.classifiers["nearest_centroid"])
            return {"classifier": classifier, "metric": str(base["metric"])}
        if classifier == "knn":
            base = dict(self.classifiers["knn"])
            k = int(strategy.get("knn_k", base["k"]))
            return {
                "classifier": classifier,
                "k": k,
                "tie_break": str(base["tie_break"]),
                "vote": str(base["vote"]),
            }
        raise ValueError(f"unknown classifier: {classifier}")


def _validate_precision(
    raw: dict[str, Any], errors: list[str]
) -> tuple[str, dict[str, dict[str, int]]]:
    levels = raw.get("precision_levels")
    normalized: dict[str, dict[str, int]] = {}
    if not isinstance(levels, dict) or not levels:
        errors.append("precision_levels must be a non-empty mapping")
        levels = {}
    for name, item in levels.items():
        if not isinstance(name, str) or not name:
            errors.append("precision level names must be non-empty strings")
            continue
        if not isinstance(item, dict):
            errors.append(f"precision_levels.{name} must be a mapping")
            continue
        errors.extend(_unknown_keys(item, PRECISION_KEYS, f"precision_levels.{name}"))
        quantization = item.get("quantization_levels")
        decimals = item.get("rounding_decimals")
        if not is_int(quantization) or quantization <= 0:
            errors.append(f"precision_levels.{name}.quantization_levels must be a positive integer")
        if not is_int(decimals) or decimals < 0 or decimals > 15:
            errors.append(f"precision_levels.{name}.rounding_decimals must be an integer in 0..15")
        if is_int(quantization) and is_int(decimals):
            normalized[name] = {
                "quantization_levels": quantization,
                "rounding_decimals": decimals,
            }
    selected = raw.get("precision")
    if not isinstance(selected, str) or selected not in normalized:
        errors.append("precision must name an entry in precision_levels")
        selected = next(iter(normalized), "standard")
    return selected, normalized


def _validate_features(
    raw: dict[str, Any], errors: list[str]
) -> tuple[dict[str, Any], dict[tuple[str, int, int], int]]:
    features = raw.get("features")
    if not isinstance(features, dict):
        errors.append("features must be a mapping")
        return {}, {}
    errors.extend(_unknown_keys(features, FEATURE_KEYS, "features"))
    supported = features.get("supported")
    if not isinstance(supported, list) or not supported:
        errors.append("features.supported must be a non-empty list")
        supported = []
    else:
        for method in supported:
            if method not in SUPPORTED_FEATURES:
                errors.append(f"unsupported feature method: {method}")
        if len(set(supported)) != len(supported):
            errors.append("features.supported must not contain duplicates")
    color_bins = features.get("color_bins")
    hog_bins = features.get("hog_bins")
    hog_cells = features.get("hog_cells")
    color_candidates = features.get("color_bins_candidates")
    hog_candidates = features.get("hog_bins_candidates")
    if not is_int(color_bins) or color_bins <= 0:
        errors.append("features.color_bins must be a positive integer")
    if not is_int(hog_bins) or hog_bins <= 0:
        errors.append("features.hog_bins must be a positive integer")
    if (
        not isinstance(hog_cells, list)
        or len(hog_cells) != 2
        or not all(is_int(item) and item > 0 for item in hog_cells)
    ):
        errors.append("features.hog_cells must be two positive integers")
        hog_cells = [2, 2]
    for name, values in (
        ("color_bins_candidates", color_candidates),
        ("hog_bins_candidates", hog_candidates),
    ):
        if not isinstance(values, list) or not values:
            errors.append(f"features.{name} must be a non-empty list")
        elif not all(is_int(item) and item > 0 for item in values):
            errors.append(f"features.{name} must contain positive integers")
        elif len(set(values)) != len(values):
            errors.append(f"features.{name} must not contain duplicates")
    expected_raw = features.get("expected_dimensions")
    expected: dict[tuple[str, int, int], int] = {}
    if not isinstance(expected_raw, list) or not expected_raw:
        errors.append("features.expected_dimensions must be a non-empty list")
        expected_raw = []
    for index, item in enumerate(expected_raw):
        where = f"features.expected_dimensions[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be a mapping")
            continue
        errors.extend(_unknown_keys(item, EXPECTED_KEYS, where))
        method = item.get("feature_method")
        item_color = item.get("color_bins")
        item_hog = item.get("hog_bins")
        declared = item.get("dimension")
        if method not in SUPPORTED_FEATURES:
            errors.append(f"{where}.feature_method is invalid")
            continue
        if not is_int(item_color) or item_color <= 0 or not is_int(item_hog) or item_hog <= 0:
            errors.append(f"{where} bin values must be positive integers")
            continue
        if not is_int(declared) or declared <= 0:
            errors.append(f"{where}.dimension must be a positive integer")
            continue
        key = (str(method), int(item_color), int(item_hog))
        if key in expected and expected[key] != declared:
            errors.append(f"duplicate expected dimension for {method} color={item_color} hog={item_hog}")
        expected[key] = int(declared)
        computed = feature_dimension(
            str(method), int(item_color), int(item_hog), (int(hog_cells[0]), int(hog_cells[1]))
        )
        if computed != declared:
            errors.append(
                "feature dimension mismatch for "
                f"{method} color={item_color} hog={item_hog}: declared {declared}, computed {computed}"
            )
    if is_int(color_bins) and is_int(hog_bins) and isinstance(hog_cells, list):
        for method in supported:
            key = (str(method), int(color_bins), int(hog_bins))
            if key not in expected and method in SUPPORTED_FEATURES:
                errors.append(
                    "missing expected dimension for base feature "
                    f"{method} color={color_bins} hog={hog_bins}"
                )
    normalized = dict(features)
    normalized["supported"] = [str(item) for item in supported]
    normalized["hog_cells"] = [int(hog_cells[0]), int(hog_cells[1])]
    return normalized, expected


def _validate_classifiers(
    raw: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    classifiers = raw.get("classifiers")
    if not isinstance(classifiers, dict):
        errors.append("classifiers must be a mapping")
        return {}
    errors.extend(_unknown_keys(classifiers, CLASSIFIER_KEYS, "classifiers"))
    supported = classifiers.get("supported")
    if not isinstance(supported, list) or not supported:
        errors.append("classifiers.supported must be a non-empty list")
        supported = []
    else:
        for name in supported:
            if name not in SUPPORTED_CLASSIFIERS:
                errors.append(f"unsupported classifier: {name}")
        if len(set(supported)) != len(supported):
            errors.append("classifiers.supported must not contain duplicates")
    logistic = classifiers.get("logistic_regression")
    if not isinstance(logistic, dict):
        errors.append("classifiers.logistic_regression must be a mapping")
        logistic = {}
    else:
        errors.extend(_unknown_keys(logistic, LR_KEYS, "classifiers.logistic_regression"))
        for key, minimum, strict in (
            ("learning_rate", 0.0, True),
            ("epochs", 0, True),
            ("l2", 0.0, False),
        ):
            value = logistic.get(key)
            ok_type = is_int(value) if key == "epochs" else isinstance(value, (int, float)) and not isinstance(value, bool)
            if not ok_type or not math.isfinite(float(value)) or float(value) < minimum or (strict and float(value) == minimum):
                comparator = "positive" if strict else "non-negative"
                errors.append(f"classifiers.logistic_regression.{key} must be {comparator}")
    centroid = classifiers.get("nearest_centroid")
    if not isinstance(centroid, dict):
        errors.append("classifiers.nearest_centroid must be a mapping")
        centroid = {}
    else:
        errors.extend(_unknown_keys(centroid, CENTROID_KEYS, "classifiers.nearest_centroid"))
        if centroid.get("metric") != "euclidean":
            errors.append("classifiers.nearest_centroid.metric must be euclidean")
    knn = classifiers.get("knn")
    if not isinstance(knn, dict):
        errors.append("classifiers.knn must be a mapping")
        knn = {}
    else:
        errors.extend(_unknown_keys(knn, KNN_KEYS, "classifiers.knn"))
        candidates = knn.get("k_candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(is_int(item) and item > 0 for item in candidates)
        ):
            errors.append("classifiers.knn.k_candidates must contain positive integers")
            candidates = []
        elif len(set(candidates)) != len(candidates):
            errors.append("classifiers.knn.k_candidates must not contain duplicates")
        if not is_int(knn.get("k")) or int(knn.get("k", 0)) not in candidates:
            errors.append("classifiers.knn.k must be present in k_candidates")
        if knn.get("tie_break") != "distance_then_sample_name":
            errors.append("classifiers.knn.tie_break must be distance_then_sample_name")
        if knn.get("vote") != "laplace_positive_fraction":
            errors.append("classifiers.knn.vote must be laplace_positive_fraction")
    return {
        "supported": [str(item) for item in supported],
        "logistic_regression": dict(logistic),
        "nearest_centroid": dict(centroid),
        "knn": dict(knn),
    }


def _validate_strategies(
    raw: dict[str, Any],
    errors: list[str],
    features: dict[str, Any],
    classifiers: dict[str, Any],
    expected: dict[tuple[str, int, int], int],
) -> list[dict[str, Any]]:
    strategies = raw.get("strategy_candidates")
    if not isinstance(strategies, list) or not strategies:
        errors.append("strategy_candidates must be a non-empty list")
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    supported_features = set(features.get("supported", []))
    supported_classifiers = set(classifiers.get("supported", []))
    hog_cells = tuple(int(item) for item in features.get("hog_cells", [2, 2]))
    knn_candidates = set(classifiers.get("knn", {}).get("k_candidates", []))
    for index, item in enumerate(strategies):
        where = f"strategy_candidates[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be a mapping")
            continue
        errors.extend(_unknown_keys(item, STRATEGY_KEYS, where))
        strategy_id = item.get("id")
        if not isinstance(strategy_id, str) or not strategy_id:
            errors.append(f"{where}.id must be a non-empty string")
            continue
        if strategy_id in seen:
            errors.append(f"duplicate strategy id: {strategy_id}")
            continue
        seen.add(strategy_id)
        method = item.get("feature_method")
        classifier = item.get("classifier")
        if method not in supported_features:
            errors.append(f"{where}.feature_method must be in features.supported")
            continue
        if classifier not in supported_classifiers:
            errors.append(f"{where}.classifier must be in classifiers.supported")
            continue
        color_bins = item.get("color_bins", features.get("color_bins"))
        hog_bins = item.get("hog_bins", features.get("hog_bins"))
        if not is_int(color_bins) or color_bins <= 0 or not is_int(hog_bins) or hog_bins <= 0:
            errors.append(f"{where} bins must be positive integers")
            continue
        key = (str(method), int(color_bins), int(hog_bins))
        computed = feature_dimension(str(method), int(color_bins), int(hog_bins), hog_cells)
        if key not in expected:
            errors.append(
                f"missing expected dimension for strategy {strategy_id}: "
                f"{method} color={color_bins} hog={hog_bins}"
            )
        elif expected[key] != computed:
            errors.append(
                f"feature dimension mismatch for strategy {strategy_id}: "
                f"declared {expected[key]}, computed {computed}"
            )
        if classifier == "knn":
            k = item.get("knn_k", classifiers.get("knn", {}).get("k"))
            if not is_int(k) or int(k) not in knn_candidates:
                errors.append(f"{where}.knn_k must be present in classifiers.knn.k_candidates")
        delta = item.get("threshold_delta", 0.0)
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not math.isfinite(float(delta)):
            errors.append(f"{where}.threshold_delta must be finite")
        for numeric_key in ("base_score", "imbalance_bonus", "label_bonus"):
            if numeric_key in item and (
                not isinstance(item[numeric_key], (int, float))
                or isinstance(item[numeric_key], bool)
                or not math.isfinite(float(item[numeric_key]))
            ):
                errors.append(f"{where}.{numeric_key} must be finite")
        scale_bonus = item.get("scale_bonus", {})
        if not isinstance(scale_bonus, dict):
            errors.append(f"{where}.scale_bonus must be a mapping")
        else:
            for scale_name, scale_value in scale_bonus.items():
                if scale_name not in {"small", "medium", "large"}:
                    errors.append(f"illegal scale bonus bucket in {where}: {scale_name}")
                if not isinstance(scale_value, (int, float)) or isinstance(scale_value, bool):
                    errors.append(f"{where}.scale_bonus.{scale_name} must be finite")
        normalized.append(
            {
                **item,
                "id": strategy_id,
                "color_bins": int(color_bins),
                "hog_bins": int(hog_bins),
                "threshold_delta": float(delta),
                "base_score": float(item.get("base_score", 1.0)),
                "imbalance_bonus": float(item.get("imbalance_bonus", 0.0)),
                "label_bonus": float(item.get("label_bonus", 0.0)),
                "scale_bonus": {
                    "small": float(item.get("scale_bonus", {}).get("small", 0.0)),
                    "medium": float(item.get("scale_bonus", {}).get("medium", 0.0)),
                    "large": float(item.get("scale_bonus", {}).get("large", 0.0)),
                },
            }
        )
    return normalized


def parse_config(raw: Any, path: Path | str = "<memory>") -> ExperimentConfig:
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise ConfigValidationError(["configuration root must be a mapping"])
    errors.extend(_unknown_keys(raw, TOP_LEVEL_KEYS, "root"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    resolution = raw.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(is_int(item) and item > 0 for item in resolution)
    ):
        errors.append("resolution must contain two positive integers")
        resolution = [1, 1]
    labels_raw = raw.get("labels")
    labels: list[str] = []
    if not isinstance(labels_raw, list) or not labels_raw:
        errors.append("labels must be a non-empty list (E_SAMPLE_INVALID would be impossible)")
    elif not all(isinstance(item, str) and item for item in labels_raw):
        errors.append("labels must contain non-empty strings")
    else:
        labels = sorted(labels_raw)
        if len(set(labels)) != len(labels):
            errors.append("labels must not contain duplicates")
    label_rates_raw = raw.get("label_rates", {})
    label_rates: dict[str, float] = {}
    if not isinstance(label_rates_raw, dict):
        errors.append("label_rates must be a mapping")
        label_rates_raw = {}
    for label, rate in label_rates_raw.items():
        if label not in labels:
            errors.append(f"label_rates contains unknown label: {label}")
            continue
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not math.isfinite(float(rate))
            or float(rate) < 0.0
            or float(rate) > 1.0
        ):
            errors.append(f"label_rates.{label} must be in [0, 1]")
        else:
            label_rates[str(label)] = float(rate)
    features, expected = _validate_features(raw, errors)
    classifiers = _validate_classifiers(raw, errors)
    threshold = raw.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or float(threshold) < 0.0
        or float(threshold) > 1.0
    ):
        errors.append("threshold must be a finite number in [0, 1]")
        threshold = 0.5
    precision, precision_levels = _validate_precision(raw, errors)
    concurrency = raw.get("concurrency")
    max_concurrency = raw.get("max_concurrency")
    if not is_int(max_concurrency) or max_concurrency <= 0:
        errors.append("max_concurrency must be a positive integer")
        max_concurrency = 1
    if not is_int(concurrency) or concurrency <= 0:
        errors.append("concurrency must be a positive integer")
        concurrency = 1
    elif is_int(max_concurrency) and concurrency > max_concurrency:
        errors.append("concurrency must not exceed max_concurrency")
    snapshot_interval = raw.get("snapshot_interval")
    if not is_int(snapshot_interval) or snapshot_interval <= 0:
        errors.append("snapshot_interval must be a positive integer")
        snapshot_interval = 1
    evaluation_raw = raw.get("evaluation", {})
    if not isinstance(evaluation_raw, dict):
        errors.append("evaluation must be a mapping")
        evaluation_raw = {}
    errors.extend(_unknown_keys(evaluation_raw, EVALUATION_KEYS, "evaluation"))
    probability_decimals = evaluation_raw.get("probability_decimals", 10)
    metric_decimals = evaluation_raw.get("metric_decimals", 8)
    cost_weight = evaluation_raw.get("cost_weight", 0.035)
    boundary_policy = evaluation_raw.get("boundary_policy", BOUNDARY_POLICY)
    if not is_int(probability_decimals) or not 0 <= probability_decimals <= 15:
        errors.append("evaluation.probability_decimals must be an integer in 0..15")
        probability_decimals = 10
    if not is_int(metric_decimals) or not 0 <= metric_decimals <= 15:
        errors.append("evaluation.metric_decimals must be an integer in 0..15")
        metric_decimals = 8
    if not isinstance(cost_weight, (int, float)) or isinstance(cost_weight, bool) or float(cost_weight) < 0:
        errors.append("evaluation.cost_weight must be non-negative")
        cost_weight = 0.035
    if boundary_policy != BOUNDARY_POLICY:
        errors.append(f"evaluation.boundary_policy must be {BOUNDARY_POLICY}")
    evaluation = {
        "probability_decimals": int(probability_decimals),
        "metric_decimals": int(metric_decimals),
        "boundary_policy": BOUNDARY_POLICY,
        "cost_weight": float(cost_weight),
    }
    strategies = _validate_strategies(raw, errors, features, classifiers, expected)
    if errors:
        raise ConfigValidationError(errors)
    return ExperimentConfig(
        raw=copy.deepcopy(raw),
        path=Path(path),
        config_hash=stable_hash(raw),
        labels=labels,
        label_rates=label_rates,
        resolution=(int(resolution[0]), int(resolution[1])),
        features=features,
        classifiers=classifiers,
        threshold=float(threshold),
        precision=precision,
        precision_levels=precision_levels,
        concurrency=int(concurrency),
        max_concurrency=int(max_concurrency),
        snapshot_interval=int(snapshot_interval),
        strategies=strategies,
        evaluation=evaluation,
        expected_dimensions=expected,
    )


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigValidationError([f"cannot read YAML {path}: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise ConfigValidationError([f"configuration root in {path} must be a mapping"])
    return raw


def load_config(path: Path, threshold_override: Any | None = None) -> ExperimentConfig:
    raw = load_yaml_mapping(path)
    if threshold_override is not None:
        raw = copy.deepcopy(raw)
        raw["threshold"] = threshold_override
    return parse_config(raw, path)


def load_plan(path: Path, labels: list[str] | None = None) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanValidationError([f"cannot read scale plan {path}: {exc}"]) from exc
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise PlanValidationError(["scale plan root must be an object"])
    errors.extend(_unknown_keys(raw, PLAN_TOP_KEYS, "root"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"scale plan schema_version must be {SCHEMA_VERSION}")
    tiers = raw.get("tiers")
    normalized_tiers: list[dict[str, Any]] = []
    if not isinstance(tiers, list) or not tiers:
        errors.append("scale plan tiers must be a non-empty list")
        tiers = []
    seen_ids: set[str] = set()
    previous_samples = 0
    previous_labels = 0
    for index, tier in enumerate(tiers):
        where = f"tiers[{index}]"
        if not isinstance(tier, dict):
            errors.append(f"{where} must be an object")
            continue
        errors.extend(_unknown_keys(tier, PLAN_TIER_KEYS, where))
        tier_id = tier.get("id")
        max_samples = tier.get("max_samples")
        max_labels = tier.get("max_labels")
        if not isinstance(tier_id, str) or not tier_id:
            errors.append(f"{where}.id must be a non-empty string")
            continue
        if tier_id in seen_ids:
            errors.append(f"duplicate tier id: {tier_id}")
        seen_ids.add(tier_id)
        if not is_int(max_samples) or max_samples <= 0:
            errors.append(f"{where}.max_samples must be a positive integer")
            continue
        if not is_int(max_labels) or max_labels <= 0:
            errors.append(f"{where}.max_labels must be a positive integer")
            continue
        if max_samples < previous_samples:
            errors.append(
                f"scale tiers are not monotonic: {tier_id}.max_samples {max_samples} < {previous_samples}"
            )
        if max_labels < previous_labels:
            errors.append(
                f"scale tiers are not monotonic: {tier_id}.max_labels {max_labels} < {previous_labels}"
            )
        if labels is not None and max_labels > len(labels):
            errors.append(f"{tier_id}.max_labels exceeds configured label count {len(labels)}")
        previous_samples = int(max_samples)
        previous_labels = int(max_labels)
        normalized_tiers.append(
            {
                "id": tier_id,
                "max_samples": int(max_samples),
                "max_labels": int(max_labels),
                "description": str(tier.get("description", "")),
            }
        )
    if errors:
        raise PlanValidationError(errors)
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "tiers": normalized_tiers,
        "plan_hash": stable_hash(raw),
    }
    return normalized


@dataclass
class InjectionItem:
    id: str
    kind: str
    sample: str | None
    stage: str | None
    value: Any
    enabled: bool


@dataclass
class InjectionSet:
    items: list[InjectionItem]
    path: Path

    def by_kind(self, kind: str, sample: str | None = None) -> InjectionItem | None:
        for item in self.items:
            if item.kind != kind:
                continue
            if item.sample is None or item.sample == sample:
                return item
        return None

    def threshold_value(self) -> float | None:
        item = self.by_kind("threshold_out_of_range")
        if item is None:
            return None
        return float(item.value)


VALID_INJECTION_KINDS = {
    "truncate_png": "load",
    "nonfinite_feature": "feature",
    "unknown_label": "load",
    "threshold_out_of_range": "config",
}


def load_injections(path: Path | None, selected_ids: list[str] | None = None) -> InjectionSet:
    if path is None:
        return InjectionSet(items=[], path=Path("<none>"))
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InjectionValidationError([f"cannot read injection config {path}: {exc}"]) from exc
    errors: list[str] = []
    if not isinstance(raw, dict):
        raise InjectionValidationError(["injection configuration root must be a mapping"])
    errors.extend(_unknown_keys(raw, INJECTION_TOP_KEYS, "root"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"injection schema_version must be {SCHEMA_VERSION}")
    raw_items = raw.get("items", [])
    if not isinstance(raw_items, list):
        errors.append("injection items must be a list")
        raw_items = []
    items: list[InjectionItem] = []
    seen_ids: set[str] = set()
    selected = set(selected_ids or [])
    for index, item in enumerate(raw_items):
        where = f"items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be a mapping")
            continue
        errors.extend(_unknown_keys(item, INJECTION_ITEM_KEYS, where))
        item_id = item.get("id")
        kind = item.get("kind")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{where}.id must be a non-empty string")
            continue
        if item_id in seen_ids:
            errors.append(f"duplicate injection id: {item_id}")
        seen_ids.add(item_id)
        if kind not in VALID_INJECTION_KINDS:
            errors.append(f"unknown injection kind: {kind}")
            continue
        sample = item.get("sample")
        if VALID_INJECTION_KINDS[str(kind)] == "config":
            if "value" not in item:
                errors.append(f"{where}.value is required for {kind}")
        elif not isinstance(sample, str) or not sample:
            errors.append(f"{where}.sample is required for {kind}")
        default_stage = VALID_INJECTION_KINDS[str(kind)]
        stage = item.get("stage", default_stage)
        if stage != default_stage:
            errors.append(f"{where}.stage must be {default_stage} for {kind}")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            errors.append(f"{where}.enabled must be boolean")
            enabled = True
        if selected:
            active = item_id in selected
        else:
            active = bool(enabled)
        if active:
            items.append(
                InjectionItem(
                    id=item_id,
                    kind=str(kind),
                    sample=str(sample) if sample is not None else None,
                    stage=str(stage),
                    value=copy.deepcopy(item.get("value")),
                    enabled=bool(enabled),
                )
            )
    if selected:
        missing = sorted(selected - seen_ids)
        if missing:
            errors.append("unknown injection ids requested: " + ", ".join(missing))
    if errors:
        raise InjectionValidationError(errors)
    return InjectionSet(items=items, path=path)


def config_with_threshold(raw: dict[str, Any], value: Any) -> dict[str, Any]:
    changed = copy.deepcopy(raw)
    changed["threshold"] = value
    return changed
