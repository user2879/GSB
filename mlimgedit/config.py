"""Configuration loading and strict validation.

The configuration describes the label vocabulary, per-class positive ratios,
feature extraction, classifier hyper-parameters, decision thresholds,
fixed-point quantization, thread-pool concurrency and the per-batch feature
point budget. Every field is validated against an explicit rule set.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_QUANT_LEVELS, ALLOWED_ROUNDING

FEATURE_MODES = ("color_hist", "hog", "color_hog")
CLASSIFIER_NAMES = ("logreg", "centroid", "knn")
MAX_THREAD_LIMIT = 16

DEFAULT_LOGREG = {"learning_rate": 0.1, "epochs": 25, "l2": 0.0}
DEFAULT_CENTROID = {"distance": "euclidean"}
DEFAULT_KNN = {"k": 3, "vote": "distance", "distance": "euclidean"}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Config:
    labels: tuple[str, ...]
    positive_ratios: dict[str, float]
    feature_mode: str
    bins: int
    hog_cell: int
    classifiers: dict[str, dict[str, Any]]
    threshold: float
    label_thresholds: dict[str, float]
    quant_level: str
    frac_bits: int
    rounding: str
    threads: int
    feature_point_budget: int
    snapshot_every: int
    image_size: tuple[int, int]
    seed: int

    def with_updates(self, updates: dict[str, Any]) -> "Config":
        """Return a new Config with validated partial updates applied.

        Nested mapping sections (feature/runtime/...) are deep-merged so a
        partial update such as ``{"feature": {"mode": "hog"}}`` preserves the
        sibling fields instead of replacing the whole section.
        """
        data = self.to_dict()
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                merged = dict(data[key])
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict) and isinstance(merged.get(sub_key), dict):
                        merged[sub_key] = {**merged[sub_key], **sub_value}
                    else:
                        merged[sub_key] = sub_value
                data[key] = merged
            else:
                data[key] = value
        return load_config_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "positive_ratios": dict(self.positive_ratios),
            "feature": {
                "mode": self.feature_mode,
                "bins": self.bins,
                "hog_cell": self.hog_cell,
            },
            "classifiers": {name: dict(params) for name, params in self.classifiers.items()},
            "threshold": self.threshold,
            "label_thresholds": dict(self.label_thresholds),
            "quantization": {
                "quant_level": self.quant_level,
                "frac_bits": self.frac_bits,
                "rounding": self.rounding,
            },
            "runtime": {
                "threads": self.threads,
                "feature_point_budget": self.feature_point_budget,
                "snapshot_every": self.snapshot_every,
            },
            "image_size": list(self.image_size),
            "seed": self.seed,
        }

    def effective_threshold(self, label: str) -> float:
        return float(self.label_thresholds.get(label, self.threshold))


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return load_config_dict(data)


def load_config_dict(data: dict[str, Any]) -> Config:
    labels = data.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ConfigError("labels must be a non-empty list")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ConfigError("labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise ConfigError("labels must not contain duplicates")
    labels_tuple = tuple(labels)

    ratios_raw = data.get("positive_ratios", {})
    if not isinstance(ratios_raw, dict):
        raise ConfigError("positive_ratios must be an object")
    ratios: dict[str, float] = {}
    for label in labels_tuple:
        ratio = ratios_raw.get(label, 0.5)
        ratio = _finite_number(ratio, f"positive_ratios.{label}")
        if not 0.0 <= ratio <= 1.0:
            raise ConfigError(f"positive_ratios.{label} must be within [0, 1]")
        ratios[label] = ratio

    feature = data.get("feature", {})
    mode = feature.get("mode", "color_hog")
    if mode not in FEATURE_MODES:
        raise ConfigError(f"feature.mode must be one of {FEATURE_MODES}")
    bins = feature.get("bins", 8)
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 64:
        raise ConfigError("feature.bins must be an integer in [2, 64]")
    hog_cell = feature.get("hog_cell", 8)
    if not isinstance(hog_cell, int) or isinstance(hog_cell, bool) or hog_cell < 4:
        raise ConfigError("feature.hog_cell must be an integer >= 4")

    classifiers_raw = data.get("classifiers", {})
    if not isinstance(classifiers_raw, dict):
        raise ConfigError("classifiers must be an object")
    classifiers: dict[str, dict[str, Any]] = {}
    for name in CLASSIFIER_NAMES:
        params = dict(classifiers_raw.get(name, {}))
        classifiers[name] = _validate_classifier(name, params)

    threshold = _finite_number(data.get("threshold", 0.5), "threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError("threshold must be within [0, 1]")

    label_thresholds_raw = data.get("label_thresholds", {})
    if not isinstance(label_thresholds_raw, dict):
        raise ConfigError("label_thresholds must be an object")
    label_thresholds: dict[str, float] = {}
    for label, value in label_thresholds_raw.items():
        if label not in labels_tuple:
            raise ConfigError(f"label_thresholds references unknown label {label!r}")
        value = _finite_number(value, f"label_thresholds.{label}")
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"label_thresholds.{label} must be within [0, 1]")
        label_thresholds[label] = value

    quant = data.get("quantization", {})
    quant_level = quant.get("quant_level", "Q1.15")
    frac_bits = quant.get("frac_bits", 15)
    rounding = quant.get("rounding", "HALF_UP")
    if quant_level not in ALLOWED_QUANT_LEVELS:
        raise ConfigError(f"quant_level must be one of {ALLOWED_QUANT_LEVELS}")
    if frac_bits not in ALLOWED_FRAC_BITS:
        raise ConfigError(f"frac_bits must be one of {ALLOWED_FRAC_BITS}")
    if rounding not in ALLOWED_ROUNDING:
        raise ConfigError(f"rounding must be one of {ALLOWED_ROUNDING}")
    _validate_quant_pair(quant_level, frac_bits)

    runtime = data.get("runtime", {})
    threads = runtime.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ConfigError("runtime.threads must be a positive integer")
    threads = min(threads, MAX_THREAD_LIMIT)
    budget = runtime.get("feature_point_budget", 10_000)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ConfigError("runtime.feature_point_budget must be a positive integer")
    snapshot_every = runtime.get("snapshot_every", 1)
    if not isinstance(snapshot_every, int) or isinstance(snapshot_every, bool) or snapshot_every < 1:
        raise ConfigError("runtime.snapshot_every must be a positive integer")

    image_size_raw = data.get("image_size", [160, 120])
    if (
        not isinstance(image_size_raw, list)
        or len(image_size_raw) != 2
        or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in image_size_raw)
    ):
        raise ConfigError("image_size must be [width, height] positive integers")

    seed = data.get("seed", 20261123)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError("seed must be an integer")

    return Config(
        labels=labels_tuple,
        positive_ratios=ratios,
        feature_mode=mode,
        bins=bins,
        hog_cell=hog_cell,
        classifiers=classifiers,
        threshold=threshold,
        label_thresholds=label_thresholds,
        quant_level=quant_level,
        frac_bits=frac_bits,
        rounding=rounding,
        threads=threads,
        feature_point_budget=budget,
        snapshot_every=snapshot_every,
        image_size=(int(image_size_raw[0]), int(image_size_raw[1])),
        seed=seed,
    )


def _validate_quant_pair(quant_level: str, frac_bits: int) -> None:
    if quant_level == "Q0.8" and frac_bits != 8:
        raise ConfigError("Q0.8 requires frac_bits=8")
    if quant_level == "Q1.15" and frac_bits != 15:
        raise ConfigError("Q1.15 requires frac_bits=15")
    if quant_level == "Q2.30" and frac_bits != 30:
        raise ConfigError("Q2.30 requires frac_bits=30")


def _validate_classifier(name: str, params: dict[str, Any]) -> dict[str, Any]:
    if name == "logreg":
        merged = {**DEFAULT_LOGREG, **params}
        lr = _finite_number(merged["learning_rate"], "logreg.learning_rate")
        if lr <= 0:
            raise ConfigError("logreg.learning_rate must be positive")
        epochs = merged["epochs"]
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ConfigError("logreg.epochs must be a positive integer")
        l2 = _finite_number(merged["l2"], "logreg.l2")
        if l2 < 0:
            raise ConfigError("logreg.l2 must be non-negative")
        return {"learning_rate": lr, "epochs": epochs, "l2": l2}
    if name == "centroid":
        merged = {**DEFAULT_CENTROID, **params}
        if merged["distance"] != "euclidean":
            raise ConfigError("centroid.distance supports only 'euclidean'")
        return {"distance": "euclidean"}
    if name == "knn":
        merged = {**DEFAULT_KNN, **params}
        if merged["distance"] != "euclidean":
            raise ConfigError("knn.distance supports only 'euclidean'")
        if merged["vote"] not in ("uniform", "distance"):
            raise ConfigError("knn.vote must be 'uniform' or 'distance'")
        k = merged["k"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ConfigError("knn.k must be a positive integer")
        return {"k": k, "vote": merged["vote"], "distance": "euclidean"}
    raise ConfigError(f"unknown classifier {name!r}")
