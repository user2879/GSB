"""YAML experiment configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from .errors import ConfigError
from .util import content_fingerprint

REQUIRED_TOP_LEVEL = ("experiment", "prediction", "series")
REQUIRED_PREDICTION = (
    "lag_orders",
    "diff_order",
    "seasonal_period",
    "seasonal_terms",
    "rolling_window",
    "forecast_steps",
    "feature_rules",
    "precision_tier",
    "concurrency",
    "snapshot_interval",
)
VALID_PRECISION = ("low", "standard", "high")
VALID_RULES = ("lag", "rolling_mean")

# Deterministic gradient-descent hyper-parameters per precision tier.
TIER_PARAMS = {
    "low": {"epochs": 20, "learning_rate": 0.01, "batch_size": 4},
    "standard": {"epochs": 60, "learning_rate": 0.005, "batch_size": 8},
    "high": {"epochs": 120, "learning_rate": 0.002, "batch_size": 16},
}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: dict
    prediction: dict
    series_specs: tuple[dict, ...]
    raw: dict = field(repr=False)

    @property
    def config_hash(self) -> str:
        """Fingerprint over every value that can change a computed artifact."""
        prediction_artifacts = {
            key: value
            for key, value in self.prediction.items()
            if key not in ("concurrency", "snapshot_interval")
        }
        relevant = {
            "experiment": self.experiment,
            "prediction": prediction_artifacts,
            "series": [
                {
                    "series_id": s["series_id"],
                    "seed": s["seed"],
                    "length": s["length"],
                    "trend": s["trend"],
                    "seasonal": s["seasonal"],
                    "noise": s["noise"],
                    "anomalies": s.get("anomalies", []),
                }
                for s in self.series_specs
            ],
        }
        return content_fingerprint(relevant)

    @property
    def block_size(self) -> int:
        return int(self.experiment.get("block_size", 16))

    @property
    def tier_params(self) -> dict:
        return TIER_PARAMS[self.prediction["precision_tier"]]

    def prediction_fingerprint(self) -> str:
        return content_fingerprint(
            {
                key: value
                for key, value in self.prediction.items()
                if key not in ("concurrency", "snapshot_interval")
            }
        )


def _require_mapping(node: Any, name: str) -> dict:
    if not isinstance(node, dict):
        raise ConfigError(f"section '{name}' must be a mapping")
    return node


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"parameter '{name}' must be a positive integer, got {value!r}")
    return value


def load_config(path: str) -> ExperimentConfig:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML syntax: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("top-level YAML document must be a mapping")

    for key in REQUIRED_TOP_LEVEL:
        if key not in raw:
            raise ConfigError(f"missing required section '{key}'")
    experiment = _require_mapping(raw["experiment"], "experiment")
    prediction = _require_mapping(raw["prediction"], "prediction")

    for key in REQUIRED_PREDICTION:
        if key not in prediction:
            raise ConfigError(f"missing required prediction parameter '{key}'")

    lags = prediction["lag_orders"]
    if not isinstance(lags, list) or not lags:
        raise ConfigError("'lag_orders' must be a non-empty list of positive integers")
    for lag in lags:
        _require_positive_int(lag, "lag_orders[]")
    if len(set(lags)) != len(lags):
        raise ConfigError("'lag_orders' must not contain duplicates")

    _require_positive_int(prediction["seasonal_period"], "seasonal_period")
    _require_positive_int(prediction["rolling_window"], "rolling_window")
    _require_positive_int(prediction["forecast_steps"], "forecast_steps")
    _require_positive_int(prediction["concurrency"], "concurrency")
    _require_positive_int(prediction["snapshot_interval"], "snapshot_interval")

    diff_order = prediction["diff_order"]
    if isinstance(diff_order, bool) or not isinstance(diff_order, int) or diff_order < 0:
        raise ConfigError("'diff_order' must be a non-negative integer")

    seasonal_terms = prediction["seasonal_terms"]
    if (
        isinstance(seasonal_terms, bool)
        or not isinstance(seasonal_terms, int)
        or seasonal_terms < 1
    ):
        raise ConfigError("'seasonal_terms' must be a positive integer")

    rules = prediction["feature_rules"]
    if not isinstance(rules, list) or not rules:
        raise ConfigError("'feature_rules' must be a non-empty list")
    for rule in rules:
        if rule not in VALID_RULES:
            raise ConfigError(
                f"unknown feature rule {rule!r}; allowed: {', '.join(VALID_RULES)}"
            )

    tier = prediction["precision_tier"]
    if tier not in VALID_PRECISION:
        raise ConfigError(
            f"precision_tier must be one of {', '.join(VALID_PRECISION)}, got {tier!r}"
        )

    series_specs = raw["series"]
    if not isinstance(series_specs, list) or not series_specs:
        raise ConfigError("'series' must be a non-empty list")
    seen: set[str] = set()
    for spec in series_specs:
        if not isinstance(spec, dict):
            raise ConfigError("every series entry must be a mapping")
        sid = spec.get("series_id")
        if not isinstance(sid, str) or not sid:
            raise ConfigError("every series requires a non-empty 'series_id'")
        if sid in seen:
            raise ConfigError(f"duplicate series_id '{sid}' in configuration")
        seen.add(sid)
        for key in ("seed", "length", "trend", "seasonal", "noise"):
            if key not in spec:
                raise ConfigError(f"series '{sid}' missing parameter '{key}'")
        _require_positive_int(spec["seed"], f"series.{sid}.seed")
        _require_positive_int(spec["length"], f"series.{sid}.length")
        for key in ("trend", "seasonal", "noise"):
            if not isinstance(spec[key], dict):
                raise ConfigError(f"series '{sid}'.{key} must be a mapping")
        anomalies = spec.get("anomalies", [])
        if not isinstance(anomalies, list):
            raise ConfigError(f"series '{sid}'.anomalies must be a list")
        for item in anomalies:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("index"), int)
                or isinstance(item.get("index"), bool)
                or not isinstance(item.get("delta"), (int, float))
            ):
                raise ConfigError(
                    f"series '{sid}' has an invalid anomaly entry "
                    "(need integer 'index' and numeric 'delta')"
                )

    return ExperimentConfig(
        experiment=experiment,
        prediction=prediction,
        series_specs=tuple(series_specs),
        raw=raw,
    )
