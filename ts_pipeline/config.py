"""Prediction configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import fingerprint, to_builtin
from .errors import E_CONFIG_INVALID, Failure
from .yamlutil import load_yaml_file


SUPPORTED_MODELS = ("autoregressive", "moving_average", "naive_persistence")
ALLOWED_DECIMAL_PLACES = (0, 1, 2, 3, 4, 5, 6)
ALLOWED_QUANTIZATION_LEVELS = (1, 2, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class BacktestConfig:
    window_length: int
    step: int
    forecast_horizon: int
    error_tolerance: float


@dataclass(frozen=True)
class ModelConfig:
    enabled: tuple[str, ...]
    moving_average_window: int
    feature_lag_order: int
    autoregressive_regularization: float

    def parameter_fingerprint(self, model_name: str) -> str:
        payload: dict[str, Any] = {
            "schema": "model-params-v1",
            "model": model_name,
        }
        if model_name == "moving_average":
            payload["window"] = self.moving_average_window
        elif model_name == "autoregressive":
            payload["lags"] = self.feature_lag_order
            payload["regularization"] = self.autoregressive_regularization
        return fingerprint(payload)


@dataclass(frozen=True)
class QuantizationConfig:
    fixed_decimal_places: int
    quantization_level: int

    @property
    def step(self) -> float:
        return self.quantization_level / float(10**self.fixed_decimal_places)

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "schema": "quantization-v1",
                "decimal_places": self.fixed_decimal_places,
                "level": self.quantization_level,
            }
        )


@dataclass(frozen=True)
class SchedulerConfig:
    concurrency: int
    batch_point_budget: int
    per_series_point_quota: int | None


@dataclass(frozen=True)
class SnapshotConfig:
    commit_interval_windows: int
    version: int


@dataclass(frozen=True)
class Config:
    source_path: Path
    samples_path: Path
    injection_manifest_path: Path
    backtest: BacktestConfig
    models: ModelConfig
    quantization: QuantizationConfig
    scheduler: SchedulerConfig
    snapshot: SnapshotConfig
    raw: dict[str, Any]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_path": str(self.samples_path),
            "injection_manifest_path": str(self.injection_manifest_path),
            "backtest": to_builtin(self.backtest.__dict__),
            "models": {
                "enabled": list(self.models.enabled),
                "moving_average_window": self.models.moving_average_window,
                "feature_lag_order": self.models.feature_lag_order,
                "autoregressive_regularization": self.models.autoregressive_regularization,
            },
            "quantization": {
                "fixed_decimal_places": self.quantization.fixed_decimal_places,
                "quantization_level": self.quantization.quantization_level,
                "step": self.quantization.step,
            },
            "scheduler": {
                "concurrency": self.scheduler.concurrency,
                "batch_point_budget": self.scheduler.batch_point_budget,
                "per_series_point_quota": self.scheduler.per_series_point_quota,
            },
            "snapshot": {
                "commit_interval_windows": self.snapshot.commit_interval_windows,
                "version": self.snapshot.version,
            },
            "config_hash": self.config_hash,
        }


@dataclass
class LoadedConfig:
    config: Config | None
    failures: list[Failure]
    fatal: bool


def _as_nonnegative_int(value: Any, name: str, failures: list[Failure], *, positive: bool = False) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be an integer"))
        return None
    if positive and value <= 0:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be positive"))
        return None
    if not positive and value < 0:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be non-negative"))
        return None
    return value


def _as_finite_float(value: Any, name: str, failures: list[Failure], *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be numeric"))
        return None
    result = float(value)
    if not math.isfinite(result):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be finite"))
        return None
    if nonnegative and result < 0:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be non-negative"))
        return None
    return result


def _mapping(value: Any, name: str, failures: list[Failure]) -> dict[str, Any]:
    if not isinstance(value, dict):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be a mapping"))
        return {}
    return value


def _resolve_path(value: Any, base_dir: Path, name: str, failures: list[Failure]) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"{name} must be a non-empty path"))
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config(
    path: Path,
    *,
    samples_override: Path | None = None,
    injection_override: Path | None = None,
) -> LoadedConfig:
    failures: list[Failure] = []
    path = path.resolve()
    try:
        raw = load_yaml_file(path)
    except Exception as exc:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", f"cannot parse config: {exc}"))
        return LoadedConfig(None, failures, True)
    if not isinstance(raw, dict):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", "config root must be a mapping"))
        return LoadedConfig(None, failures, True)

    schema_version = raw.get("schema_version", 1)
    if schema_version != 1:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", "unsupported schema_version"))
        return LoadedConfig(None, failures, True)

    backtest_raw = _mapping(raw.get("backtest"), "backtest", failures)
    models_raw = _mapping(raw.get("models"), "models", failures)
    quantization_raw = _mapping(raw.get("quantization"), "quantization", failures)
    scheduler_raw = _mapping(raw.get("scheduler"), "scheduler", failures)
    snapshot_raw = _mapping(raw.get("snapshot"), "snapshot", failures)

    window_length = _as_nonnegative_int(backtest_raw.get("window_length"), "backtest.window_length", failures, positive=True)
    step = _as_nonnegative_int(backtest_raw.get("step"), "backtest.step", failures, positive=True)
    forecast_horizon = _as_nonnegative_int(
        backtest_raw.get("forecast_horizon"), "backtest.forecast_horizon", failures
    )
    tolerance = _as_finite_float(
        backtest_raw.get("error_tolerance"), "backtest.error_tolerance", failures, nonnegative=True
    )
    if window_length is not None and step is not None and step > window_length:
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", "backtest.step must not exceed window_length"))

    enabled_raw = models_raw.get("enabled")
    enabled: tuple[str, ...] = ()
    if not isinstance(enabled_raw, list) or not all(isinstance(item, str) for item in enabled_raw):
        failures.append(Failure(E_CONFIG_INVALID, "load_validation", "models.enabled must be a list of names"))
    else:
        enabled = tuple(sorted(set(enabled_raw)))
        unknown = set(enabled) - set(SUPPORTED_MODELS)
        if unknown:
            failures.append(
                Failure(E_CONFIG_INVALID, "load_validation", f"unknown models: {sorted(unknown)}")
            )
        if set(enabled) != set(SUPPORTED_MODELS):
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "load_validation",
                    "models.enabled must contain naive_persistence, moving_average and autoregressive",
                )
            )

    ma_window = _as_nonnegative_int(
        models_raw.get("moving_average_window"), "models.moving_average_window", failures, positive=True
    )
    lag_order = _as_nonnegative_int(
        models_raw.get("feature_lag_order"), "models.feature_lag_order", failures, positive=True
    )
    regularization = _as_finite_float(
        models_raw.get("autoregressive_regularization"),
        "models.autoregressive_regularization",
        failures,
        nonnegative=True,
    )

    decimal_places = quantization_raw.get("fixed_decimal_places")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, int) or decimal_places not in ALLOWED_DECIMAL_PLACES:
        failures.append(
            Failure(
                E_CONFIG_INVALID,
                "load_validation",
                f"quantization.fixed_decimal_places must be one of {list(ALLOWED_DECIMAL_PLACES)}",
            )
        )
        decimal_places = None
    quantization_level = quantization_raw.get("quantization_level")
    if (
        isinstance(quantization_level, bool)
        or not isinstance(quantization_level, int)
        or quantization_level not in ALLOWED_QUANTIZATION_LEVELS
    ):
        failures.append(
            Failure(
                E_CONFIG_INVALID,
                "load_validation",
                f"quantization.quantization_level must be one of {list(ALLOWED_QUANTIZATION_LEVELS)}",
            )
        )
        quantization_level = None

    concurrency = _as_nonnegative_int(
        scheduler_raw.get("concurrency"), "scheduler.concurrency", failures, positive=True
    )
    batch_budget = _as_nonnegative_int(
        scheduler_raw.get("batch_point_budget"), "scheduler.batch_point_budget", failures
    )
    quota = scheduler_raw.get("per_series_point_quota")
    if quota is not None:
        quota = _as_nonnegative_int(quota, "scheduler.per_series_point_quota", failures, positive=True)

    commit_interval = _as_nonnegative_int(
        snapshot_raw.get("commit_interval_windows"), "snapshot.commit_interval_windows", failures, positive=True
    )
    snapshot_version = _as_nonnegative_int(
        snapshot_raw.get("version", 1), "snapshot.version", failures, positive=True
    )

    base_dir = path.parent
    samples_path = samples_override.resolve() if samples_override else _resolve_path(
        raw.get("samples"), base_dir, "samples", failures
    )
    injection_path = injection_override.resolve() if injection_override else _resolve_path(
        raw.get("injection_manifest"), base_dir, "injection_manifest", failures
    )

    required_values = (
        window_length,
        step,
        forecast_horizon,
        tolerance,
        ma_window,
        lag_order,
        regularization,
        decimal_places,
        quantization_level,
        concurrency,
        batch_budget,
        commit_interval,
        snapshot_version,
        samples_path,
        injection_path,
    )
    if any(item is None for item in required_values):
        return LoadedConfig(None, failures, True)

    config = Config(
        source_path=path,
        samples_path=samples_path,  # type: ignore[arg-type]
        injection_manifest_path=injection_path,  # type: ignore[arg-type]
        backtest=BacktestConfig(window_length, step, forecast_horizon, tolerance),  # type: ignore[arg-type]
        models=ModelConfig(enabled, ma_window, lag_order, regularization),  # type: ignore[arg-type]
        quantization=QuantizationConfig(decimal_places, quantization_level),  # type: ignore[arg-type]
        scheduler=SchedulerConfig(concurrency, batch_budget, quota),  # type: ignore[arg-type]
        snapshot=SnapshotConfig(commit_interval, snapshot_version),  # type: ignore[arg-type]
        raw=raw,
        config_hash=fingerprint(
            {
                "schema": "pipeline-config-v1",
                "raw": raw,
                "samples_path": str(samples_path),
                "injection_manifest_path": str(injection_path),
            }
        ),
    )
    return LoadedConfig(config, failures, False)
