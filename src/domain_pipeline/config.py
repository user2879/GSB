from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover - exercised only on NumPy-only installations
    from . import yaml_compat as _yaml

from .canonical import canonical_json_text, fingerprint


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "data": {
        "id_field": "id",
        "label_field": "label",
        "split_field": "split",
        "numeric_columns": ["age", "income"],
        "categorical_columns": ["region", "channel"],
        "categorical_domains": {
            "region": ["east", "west", "north", "south"],
            "channel": ["web", "store", "partner"],
        },
    },
    "features": {"bins": 4},
    "training": {
        "selected_candidate": 0,
        "candidates": [
            {
                "id": "balanced",
                "initial_weight_scale": 0.05,
                "initial_bias": 0.0,
                "l2": 0.001,
            },
            {
                "id": "regularized",
                "initial_weight_scale": 0.02,
                "initial_bias": 0.0,
                "l2": 0.01,
            },
        ],
        "batch_size": 4,
        "learning_rate": 0.1,
        "epochs": 4,
    },
    "quantization": {"fractional_bits": 12, "scale_level": 4},
    "concurrency": {
        "max_workers": 4,
        "max_workers_limit": 8,
        "batch_budget": 64,
    },
    "snapshot": {"commit_interval_batches": 1},
}


def _failure(path: str, code: str, message: str) -> dict[str, Any]:
    return {
        "scope": "config",
        "path": path,
        "codes": [code],
        "message": message,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def deep_merge_defaults(value: Any, defaults: Any) -> Any:
    if isinstance(value, dict) and isinstance(defaults, dict):
        merged = copy.deepcopy(value)
        for key, default_value in defaults.items():
            if key in merged:
                merged[key] = deep_merge_defaults(merged[key], default_value)
            else:
                merged[key] = copy.deepcopy(default_value)
        return merged
    return copy.deepcopy(value)


def load_config_yaml(path_text: str) -> dict[str, Any]:
    with open(path_text, "r", encoding="utf-8") as handle:
        loaded = _yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("pipeline configuration must be a YAML mapping")
    return loaded


def dump_config_yaml(config: dict[str, Any]) -> str:
    return _yaml.safe_dump(
        config,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    data: dict[str, Any]
    features: dict[str, Any]
    training: dict[str, Any]
    quantization: dict[str, Any]
    concurrency: dict[str, Any]
    snapshot: dict[str, Any]
    failures: list[dict[str, Any]]
    config_hash: str

    def selected_candidate(self) -> dict[str, Any]:
        selected = self.training["selected_candidate"]
        candidates = self.training["candidates"]
        if isinstance(selected, int) and 0 <= selected < len(candidates):
            return candidates[selected]
        for candidate in candidates:
            if candidate.get("id") == selected:
                return candidate
        return candidates[0]

    def training_params(self) -> dict[str, Any]:
        candidate = self.selected_candidate()
        return {
            "candidate": copy.deepcopy(candidate),
            "batch_size": self.training["batch_size"],
            "learning_rate": self.training["learning_rate"],
            "epochs": self.training["epochs"],
        }

    def feature_params(self) -> dict[str, Any]:
        return {
            "bins": self.features["bins"],
            "numeric_columns": list(self.data["numeric_columns"]),
            "categorical_columns": list(self.data["categorical_columns"]),
            "categorical_domains": copy.deepcopy(self.data["categorical_domains"]),
            "encoding": "one_hot_bins_categories_missing_v1",
        }

    def quantization_params(self) -> dict[str, Any]:
        return {
            "fractional_bits": self.quantization["fractional_bits"],
            "scale_level": self.quantization["scale_level"],
            "rounding": "half_up",
        }


def validate_config(raw_input: dict[str, Any]) -> PipelineConfig:
    raw = deep_merge_defaults(raw_input, DEFAULT_CONFIG)
    failures: list[dict[str, Any]] = []

    data = raw.get("data")
    if not isinstance(data, dict):
        failures.append(_failure("data", "E_CONFIG_TYPE", "data must be a mapping"))
        data = copy.deepcopy(DEFAULT_CONFIG["data"])
        raw["data"] = data

    numeric_columns = data.get("numeric_columns")
    if not isinstance(numeric_columns, list) or not numeric_columns:
        failures.append(
            _failure(
                "data.numeric_columns",
                "E_CONFIG_COLUMNS",
                "numeric_columns must be a non-empty list",
            )
        )
        numeric_columns = copy.deepcopy(DEFAULT_CONFIG["data"]["numeric_columns"])
        data["numeric_columns"] = numeric_columns
    if any(not isinstance(column, str) or not column for column in numeric_columns):
        failures.append(
            _failure(
                "data.numeric_columns",
                "E_CONFIG_COLUMNS",
                "numeric column names must be non-empty strings",
            )
        )
        numeric_columns = [str(column) for column in numeric_columns if str(column)]

    categorical_columns = data.get("categorical_columns")
    if not isinstance(categorical_columns, list):
        failures.append(
            _failure(
                "data.categorical_columns",
                "E_CONFIG_COLUMNS",
                "categorical_columns must be a list",
            )
        )
        categorical_columns = copy.deepcopy(
            DEFAULT_CONFIG["data"]["categorical_columns"]
        )
        data["categorical_columns"] = categorical_columns
    if any(not isinstance(column, str) or not column for column in categorical_columns):
        failures.append(
            _failure(
                "data.categorical_columns",
                "E_CONFIG_COLUMNS",
                "categorical column names must be non-empty strings",
            )
        )
        categorical_columns = [
            str(column) for column in categorical_columns if str(column)
        ]

    if len(set(numeric_columns)) != len(numeric_columns):
        failures.append(
            _failure(
                "data.numeric_columns",
                "E_CONFIG_COLUMNS",
                "numeric_columns contains duplicates",
            )
        )
    if len(set(categorical_columns)) != len(categorical_columns):
        failures.append(
            _failure(
                "data.categorical_columns",
                "E_CONFIG_COLUMNS",
                "categorical_columns contains duplicates",
            )
        )
    if set(numeric_columns) & set(categorical_columns):
        failures.append(
            _failure(
                "data",
                "E_CONFIG_COLUMNS",
                "numeric and categorical columns must be disjoint",
            )
        )

    domains = data.get("categorical_domains")
    if not isinstance(domains, dict):
        failures.append(
            _failure(
                "data.categorical_domains",
                "E_CONFIG_DOMAIN",
                "categorical_domains must be a mapping",
            )
        )
        domains = copy.deepcopy(DEFAULT_CONFIG["data"]["categorical_domains"])
        data["categorical_domains"] = domains
    for column in categorical_columns:
        domain = domains.get(column)
        if not isinstance(domain, list) or not domain:
            failures.append(
                _failure(
                    f"data.categorical_domains.{column}",
                    "E_CONFIG_DOMAIN",
                    "each categorical column requires a non-empty declared domain",
                )
            )
            domains[column] = [f"__default_{column}"]
            continue
        if len({canonical_json_text(value) for value in domain}) != len(domain):
            failures.append(
                _failure(
                    f"data.categorical_domains.{column}",
                    "E_CONFIG_DOMAIN",
                    "declared category values must be unique",
                )
            )

    for field in ("id_field", "label_field", "split_field"):
        value = data.get(field)
        if not isinstance(value, str) or not value:
            failures.append(
                _failure(
                    f"data.{field}",
                    "E_CONFIG_FIELD",
                    f"{field} must be a non-empty string",
                )
            )
            data[field] = DEFAULT_CONFIG["data"][field]

    features = raw.get("features")
    if not isinstance(features, dict):
        failures.append(
            _failure("features", "E_CONFIG_TYPE", "features must be a mapping")
        )
        features = copy.deepcopy(DEFAULT_CONFIG["features"])
        raw["features"] = features
    bins = features.get("bins")
    if not _is_int(bins) or bins <= 0:
        failures.append(
            _failure("features.bins", "E_CONFIG_BINS", "bins must be a positive integer")
        )
        features["bins"] = DEFAULT_CONFIG["features"]["bins"]

    training = raw.get("training")
    if not isinstance(training, dict):
        failures.append(
            _failure("training", "E_CONFIG_TYPE", "training must be a mapping")
        )
        training = copy.deepcopy(DEFAULT_CONFIG["training"])
        raw["training"] = training

    candidates = training.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        failures.append(
            _failure(
                "training.candidates",
                "E_CONFIG_CANDIDATE",
                "candidates must be a non-empty list",
            )
        )
        candidates = copy.deepcopy(DEFAULT_CONFIG["training"]["candidates"])
        training["candidates"] = candidates
    candidate_ids: list[str] = []
    normalized_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            failures.append(
                _failure(
                    f"training.candidates.{index}",
                    "E_CONFIG_CANDIDATE",
                    "candidate must be a mapping",
                )
            )
            candidate = copy.deepcopy(DEFAULT_CONFIG["training"]["candidates"][0])
        candidate = deep_merge_defaults(
            candidate, DEFAULT_CONFIG["training"]["candidates"][0]
        )
        candidate_id = candidate.get("id", f"candidate-{index}")
        if not isinstance(candidate_id, str) or not candidate_id:
            failures.append(
                _failure(
                    f"training.candidates.{index}.id",
                    "E_CONFIG_CANDIDATE",
                    "candidate id must be a non-empty string",
                )
            )
            candidate_id = f"candidate-{index}"
            candidate["id"] = candidate_id
        candidate_ids.append(candidate_id)
        scale = candidate.get("initial_weight_scale")
        if not _is_finite_number(scale) or float(scale) < 0:
            failures.append(
                _failure(
                    f"training.candidates.{index}.initial_weight_scale",
                    "E_CONFIG_CANDIDATE",
                    "initial_weight_scale must be a finite non-negative number",
                )
            )
            candidate["initial_weight_scale"] = 0.0
        bias = candidate.get("initial_bias")
        if not _is_finite_number(bias):
            failures.append(
                _failure(
                    f"training.candidates.{index}.initial_bias",
                    "E_CONFIG_CANDIDATE",
                    "initial_bias must be finite",
                )
            )
            candidate["initial_bias"] = 0.0
        l2 = candidate.get("l2")
        if not _is_finite_number(l2) or float(l2) < 0:
            failures.append(
                _failure(
                    f"training.candidates.{index}.l2",
                    "E_CONFIG_CANDIDATE",
                    "l2 must be a finite non-negative number",
                )
            )
            candidate["l2"] = 0.0
        normalized_candidates.append(candidate)
    if len(set(candidate_ids)) != len(candidate_ids):
        failures.append(
            _failure(
                "training.candidates",
                "E_CONFIG_CANDIDATE",
                "candidate ids must be unique",
            )
        )
    training["candidates"] = normalized_candidates

    selected = training.get("selected_candidate", 0)
    valid_selection = False
    if _is_int(selected):
        valid_selection = 0 <= selected < len(normalized_candidates)
    elif isinstance(selected, str):
        valid_selection = selected in candidate_ids
    if not valid_selection:
        failures.append(
            _failure(
                "training.selected_candidate",
                "E_CONFIG_CANDIDATE",
                "selected_candidate must identify one candidate",
            )
        )
        training["selected_candidate"] = 0

    for field in ("batch_size", "epochs"):
        value = training.get(field)
        if not _is_int(value) or value <= 0:
            failures.append(
                _failure(
                    f"training.{field}",
                    "E_CONFIG_TRAINING",
                    f"{field} must be a positive integer",
                )
            )
            training[field] = DEFAULT_CONFIG["training"][field]
    learning_rate = training.get("learning_rate")
    if not _is_finite_number(learning_rate) or float(learning_rate) <= 0:
        failures.append(
            _failure(
                "training.learning_rate",
                "E_CONFIG_TRAINING",
                "learning_rate must be a finite positive number",
            )
        )
        training["learning_rate"] = DEFAULT_CONFIG["training"]["learning_rate"]

    quantization = raw.get("quantization")
    if not isinstance(quantization, dict):
        failures.append(
            _failure(
                "quantization", "E_CONFIG_TYPE", "quantization must be a mapping"
            )
        )
        quantization = copy.deepcopy(DEFAULT_CONFIG["quantization"])
        raw["quantization"] = quantization
    fractional_bits = quantization.get("fractional_bits")
    if not _is_int(fractional_bits) or not 1 <= fractional_bits <= 30:
        failures.append(
            _failure(
                "quantization.fractional_bits",
                "E_CONFIG_QUANTIZATION",
                "fractional_bits must be an integer in [1, 30]",
            )
        )
        quantization["fractional_bits"] = DEFAULT_CONFIG["quantization"][
            "fractional_bits"
        ]
    scale_level = quantization.get("scale_level")
    if not _is_int(scale_level) or scale_level <= 0:
        failures.append(
            _failure(
                "quantization.scale_level",
                "E_CONFIG_QUANTIZATION",
                "scale_level must be a positive integer",
            )
        )
        quantization["scale_level"] = DEFAULT_CONFIG["quantization"]["scale_level"]
    elif scale_level > (1 << quantization["fractional_bits"]):
        failures.append(
            _failure(
                "quantization.scale_level",
                "E_CONFIG_QUANTIZATION",
                "scale_level must not exceed the fixed-point denominator",
            )
        )
        quantization["scale_level"] = 1

    concurrency = raw.get("concurrency")
    if not isinstance(concurrency, dict):
        failures.append(
            _failure(
                "concurrency", "E_CONFIG_TYPE", "concurrency must be a mapping"
            )
        )
        concurrency = copy.deepcopy(DEFAULT_CONFIG["concurrency"])
        raw["concurrency"] = concurrency
    limit = concurrency.get("max_workers_limit")
    if not _is_int(limit) or limit <= 0:
        failures.append(
            _failure(
                "concurrency.max_workers_limit",
                "E_CONFIG_CONCURRENCY",
                "max_workers_limit must be a positive integer",
            )
        )
        limit = DEFAULT_CONFIG["concurrency"]["max_workers_limit"]
        concurrency["max_workers_limit"] = limit
    workers = concurrency.get("max_workers")
    if not _is_int(workers) or workers <= 0:
        failures.append(
            _failure(
                "concurrency.max_workers",
                "E_CONFIG_CONCURRENCY",
                "max_workers must be a positive integer",
            )
        )
        workers = 1
        concurrency["max_workers"] = workers
    if workers > limit:
        failures.append(
            _failure(
                "concurrency.max_workers",
                "E_CONFIG_CONCURRENCY_LIMIT",
                "max_workers exceeds max_workers_limit and is clamped",
            )
        )
        concurrency["max_workers"] = limit
    budget = concurrency.get("batch_budget")
    if not _is_int(budget) or budget <= 0:
        failures.append(
            _failure(
                "concurrency.batch_budget",
                "E_CONFIG_BUDGET",
                "batch_budget must be a positive integer",
            )
        )
        concurrency["batch_budget"] = 1

    snapshot_config = raw.get("snapshot")
    if not isinstance(snapshot_config, dict):
        failures.append(
            _failure("snapshot", "E_CONFIG_TYPE", "snapshot must be a mapping")
        )
        snapshot_config = copy.deepcopy(DEFAULT_CONFIG["snapshot"])
        raw["snapshot"] = snapshot_config
    interval = snapshot_config.get("commit_interval_batches")
    if not _is_int(interval) or interval <= 0:
        failures.append(
            _failure(
                "snapshot.commit_interval_batches",
                "E_CONFIG_SNAPSHOT",
                "commit_interval_batches must be a positive integer",
            )
        )
        snapshot_config["commit_interval_batches"] = 1

    config_hash = fingerprint("pipeline-config:v1", raw)
    return PipelineConfig(
        raw=raw,
        data=data,
        features=features,
        training=training,
        quantization=quantization,
        concurrency=concurrency,
        snapshot=snapshot_config,
        failures=sorted(failures, key=lambda item: (item["path"], item["codes"])),
        config_hash=config_hash,
    )
