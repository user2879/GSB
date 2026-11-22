"""Experiment configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, E_MIX_RANGE, E_SCHEME_UNKNOWN, fail
from .util import content_hash, to_jsonable


SCHEME_ALIASES = {
    "l2": "l2",
    "ridge": "l2",
    "l1": "l1",
    "lasso": "l1",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "enet": "elastic_net",
}
EXCHANGE_VERSIONS = {"v1", "v2"}
HARD_CONCURRENCY_LIMIT = 16


@dataclass(frozen=True)
class ModelConfig:
    name: str
    scheme: str
    penalty: float
    mix: float | None
    features: tuple[str, ...] | None
    epochs_override: int | None = None

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scheme": self.scheme,
            "penalty": self.penalty,
            "mix": self.mix,
            "features": list(self.features) if self.features is not None else None,
            "epochs_override": self.epochs_override,
        }


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 60
    learning_rate: float = 0.05
    batch_size: int = 16
    tolerance: float = 1e-10
    l1_zero_subgradient: float = 0.0
    snapshot_every_batches: int = 4

    def canonical(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "tolerance": self.tolerance,
            "l1_zero_subgradient": self.l1_zero_subgradient,
            "snapshot_every_batches": self.snapshot_every_batches,
        }


@dataclass(frozen=True)
class BudgetConfig:
    global_epochs: int = 1000
    per_model_epochs: int = 120

    def canonical(self) -> dict[str, Any]:
        return {
            "global_epochs": self.global_epochs,
            "per_model_epochs": self.per_model_epochs,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    max_concurrency: int = 4
    concurrency_limit: int = HARD_CONCURRENCY_LIMIT

    def canonical(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.max_concurrency,
            "concurrency_limit": self.concurrency_limit,
        }


@dataclass(frozen=True)
class SnapshotConfig:
    strict_resume: bool = False
    expected_version: int | None = None
    keep_generations: int = 3

    def canonical(self) -> dict[str, Any]:
        return {
            "strict_resume": self.strict_resume,
            "expected_version": self.expected_version,
            "keep_generations": self.keep_generations,
        }


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    exchange_version: str
    data: dict[str, Any]
    models: list[ModelConfig]
    training: TrainingConfig
    budget: BudgetConfig
    runtime: RuntimeConfig
    snapshot: SnapshotConfig
    faults: dict[str, dict[str, Any]]
    conversion_tolerance: float
    source_path: str | None = None
    normalized: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        return content_hash("cfg", self.normalized)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, f"configuration file not found: {path}")
    except yaml.YAMLError as exc:
        fail(E_CONFIG_INVALID, f"invalid YAML configuration: {exc}")
    if not isinstance(loaded, dict):
        fail(E_CONFIG_INVALID, "configuration root must be a mapping")
    return loaded


def _int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(E_CONFIG_INVALID, f"{field_name} must be an integer", value=value)
    if minimum is not None and value < minimum:
        fail(E_CONFIG_INVALID, f"{field_name} must be >= {minimum}", value=value)
    return value


def _float(value: Any, field_name: str, *, finite: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(E_CONFIG_INVALID, f"{field_name} must be numeric", value=value)
    result = float(value)
    if finite and (result != result or result in (float("inf"), float("-inf"))):
        fail(E_CONFIG_INVALID, f"{field_name} must be finite", value=value)
    return result


def normalize_scheme(value: Any) -> str:
    key = str(value).strip().lower() if value is not None else ""
    if key not in SCHEME_ALIASES:
        fail(E_SCHEME_UNKNOWN, f"unknown regularization scheme: {value!r}", scheme=value)
    return SCHEME_ALIASES[key]


def _resolve_data(value: Any, base_dir: Path) -> dict[str, Any]:
    if isinstance(value, str):
        data_path = (base_dir / value).resolve()
        return _load_yaml(data_path)
    if isinstance(value, dict):
        return value
    fail(E_CONFIG_INVALID, "data must be an inline mapping or a YAML file path")


def _normalize_model(raw: Any, index: int) -> ModelConfig:
    if not isinstance(raw, dict):
        fail(E_CONFIG_INVALID, "each model entry must be a mapping", index=index)
    name = str(raw.get("name", "")).strip()
    if not name:
        fail(E_CONFIG_INVALID, "model name is required", index=index)
    scheme = normalize_scheme(raw.get("scheme"))

    penalty_raw = raw.get("penalty", raw.get("lambda", 0.01))
    penalty = _float(penalty_raw, f"models[{name}].penalty")
    if penalty < 0.0:
        fail(E_CONFIG_INVALID, "penalty must be non-negative", model=name, value=penalty)

    mix_present = "mix" in raw or "l1_ratio" in raw
    mix_raw = raw.get("mix", raw.get("l1_ratio", 0.5 if scheme == "elastic_net" else None))
    mix: float | None = None
    if mix_raw is not None:
        mix = _float(mix_raw, f"models[{name}].mix", finite=False)
        if not math.isfinite(mix) or mix < 0.0 or mix > 1.0:
            fail(E_MIX_RANGE, "mix must be in the closed interval [0, 1]", model=name, value=mix)
    elif mix_present:
        fail(E_MIX_RANGE, "mix must not be null", model=name)

    features_raw = raw.get("features", raw.get("feature_whitelist"))
    features: tuple[str, ...] | None
    if features_raw is None:
        features = None
    elif isinstance(features_raw, str):
        features = (features_raw,)
    elif isinstance(features_raw, (list, tuple)):
        names = [str(item).strip() for item in features_raw]
        if any(not item for item in names):
            fail(E_CONFIG_INVALID, "feature whitelist contains an empty name", model=name)
        features = tuple(sorted(set(names)))
    else:
        fail(E_CONFIG_INVALID, "features must be a string or list of strings", model=name)

    epochs_override = raw.get("epochs")
    if epochs_override is not None:
        epochs_override = _int(epochs_override, f"models[{name}].epochs", minimum=1)

    return ModelConfig(
        name=name,
        scheme=scheme,
        penalty=penalty,
        mix=mix,
        features=features,
        epochs_override=epochs_override,
    )


def _normalize_faults(raw: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if raw is None:
        return result
    if isinstance(raw, dict):
        items = [{"model": key, **value} for key, value in raw.items() if isinstance(value, dict)]
    elif isinstance(raw, list):
        items = raw
    else:
        fail(E_CONFIG_INVALID, "faults must be a list or mapping")

    valid_stages = {"standardize", "gradient", "train", "conversion"}
    for item in items:
        if not isinstance(item, dict) or not item.get("model"):
            fail(E_CONFIG_INVALID, "fault entry requires a model")
        model = str(item["model"])
        stage = str(item.get("stage", "gradient"))
        if stage not in valid_stages:
            fail(E_CONFIG_INVALID, f"unknown fault stage: {stage}", model=model)
        epoch = _int(item.get("epoch", 1), f"faults[{model}].epoch", minimum=1)
        exception_type = str(item.get("exception", "RuntimeError"))
        result[model] = {
            "model": model,
            "stage": stage,
            "epoch": epoch,
            "exception": exception_type,
            "message": str(item.get("message", f"injected {exception_type} in {stage}")),
        }
    return result


def load_config(path: str | Path, *, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = _load_yaml(source)
    if isinstance(raw.get("experiment"), dict):
        experiment = raw["experiment"]
        merged: dict[str, Any] = {**experiment, **{k: v for k, v in raw.items() if k != "experiment"}}
        raw = merged
    if overrides:
        raw = {**raw, **overrides}

    name = str(raw.get("name", source.stem)).strip() or source.stem
    seed = _int(raw.get("seed", 20261123), "seed")
    version = str(raw.get("exchange_version", "v2")).lower()
    if version not in EXCHANGE_VERSIONS:
        fail(E_CONFIG_INVALID, "exchange_version must be v1 or v2", value=version)

    data = _resolve_data(raw.get("data"), source.parent)
    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        fail(E_CONFIG_INVALID, "models must be a non-empty list")
    models = [_normalize_model(item, index) for index, item in enumerate(models_raw)]
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        fail(E_CONFIG_INVALID, "model names must be unique")

    training_raw = raw.get("training") or {}
    if not isinstance(training_raw, dict):
        fail(E_CONFIG_INVALID, "training must be a mapping")
    training = TrainingConfig(
        epochs=_int(training_raw.get("epochs", 60), "training.epochs", minimum=1),
        learning_rate=_float(training_raw.get("learning_rate", 0.05), "training.learning_rate"),
        batch_size=_int(training_raw.get("batch_size", 16), "training.batch_size", minimum=1),
        tolerance=_float(training_raw.get("tolerance", 1e-10), "training.tolerance"),
        l1_zero_subgradient=_float(
            training_raw.get("l1_zero_subgradient", 0.0), "training.l1_zero_subgradient"
        ),
        snapshot_every_batches=_int(
            training_raw.get("snapshot_every_batches", 4), "training.snapshot_every_batches", minimum=1
        ),
    )
    if training.learning_rate <= 0.0:
        fail(E_CONFIG_INVALID, "training.learning_rate must be positive")
    if training.tolerance < 0.0:
        fail(E_CONFIG_INVALID, "training.tolerance must be non-negative")
    if training.l1_zero_subgradient not in (-1.0, 0.0, 1.0):
        fail(E_CONFIG_INVALID, "training.l1_zero_subgradient must be -1, 0, or 1")

    budget_raw = raw.get("budget") or {}
    if not isinstance(budget_raw, dict):
        fail(E_CONFIG_INVALID, "budget must be a mapping")
    budget = BudgetConfig(
        global_epochs=_int(budget_raw.get("global_epochs", 1000), "budget.global_epochs", minimum=0),
        per_model_epochs=_int(
            budget_raw.get("per_model_epochs", 120), "budget.per_model_epochs", minimum=1
        ),
    )

    runtime_raw = raw.get("runtime") or {}
    if not isinstance(runtime_raw, dict):
        fail(E_CONFIG_INVALID, "runtime must be a mapping")
    concurrency_limit = _int(
        runtime_raw.get("concurrency_limit", HARD_CONCURRENCY_LIMIT),
        "runtime.concurrency_limit",
        minimum=1,
    )
    if concurrency_limit > HARD_CONCURRENCY_LIMIT:
        fail(
            E_CONFIG_INVALID,
            f"runtime.concurrency_limit cannot exceed {HARD_CONCURRENCY_LIMIT}",
            value=concurrency_limit,
        )
    max_concurrency = _int(
        runtime_raw.get("max_concurrency", min(4, concurrency_limit)),
        "runtime.max_concurrency",
        minimum=1,
    )
    if max_concurrency > concurrency_limit:
        fail(
            E_CONFIG_INVALID,
            "runtime.max_concurrency exceeds the configured upper limit",
            max_concurrency=max_concurrency,
            limit=concurrency_limit,
        )
    runtime = RuntimeConfig(max_concurrency=max_concurrency, concurrency_limit=concurrency_limit)

    snapshot_raw = raw.get("snapshot") or {}
    if not isinstance(snapshot_raw, dict):
        fail(E_CONFIG_INVALID, "snapshot must be a mapping")
    expected_version = snapshot_raw.get("expected_version")
    if expected_version is not None:
        expected_version = _int(expected_version, "snapshot.expected_version", minimum=0)
    snapshot = SnapshotConfig(
        strict_resume=bool(snapshot_raw.get("strict_resume", False)),
        expected_version=expected_version,
        keep_generations=_int(snapshot_raw.get("keep_generations", 3), "snapshot.keep_generations", minimum=1),
    )

    tolerance = _float(raw.get("conversion_tolerance", 0.001), "conversion_tolerance")
    if tolerance < 0.0:
        fail(E_CONFIG_INVALID, "conversion_tolerance must be non-negative")

    faults = _normalize_faults(raw.get("faults"))
    unknown_fault_models = sorted(set(faults) - set(names))
    if unknown_fault_models:
        fail(E_CONFIG_INVALID, "fault references an unknown model", models=unknown_fault_models)

    normalized = to_jsonable(
        {
            "name": name,
            "seed": seed,
            "exchange_version": version,
            "data": data,
            "models": sorted((model.canonical() for model in models), key=lambda item: item["name"]),
            "training": training.canonical(),
            "budget": budget.canonical(),
            "faults": [faults[key] for key in sorted(faults)],
            "conversion_tolerance": tolerance,
        }
    )
    # State/output paths and CLI-only switches are intentionally excluded from the hash.
    return ExperimentConfig(
        name=name,
        seed=seed,
        exchange_version=version,
        data=data,
        models=models,
        training=training,
        budget=budget,
        runtime=runtime,
        snapshot=snapshot,
        faults=faults,
        conversion_tolerance=tolerance,
        source_path=str(source),
        normalized=normalized,
    )
