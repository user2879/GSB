"""Experiment configuration loading, validation and candidate normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .data import DataDescription, load_data_description, normalize_kind, valid_seed
from .errors import (
    E_CONFIG_INVALID,
    E_DBSCAN_RADIUS_INVALID,
    E_HYPERPARAM_OUT_OF_RANGE,
    BusinessError,
    Failure,
)
from .util import canonical_json, sha256_value


METHOD_ALIASES = {
    "kmeans": "kmeans",
    "k-means": "kmeans",
    "dbscan": "dbscan",
    "gaussian_mixture": "gaussian_mixture",
    "gaussian-mixture": "gaussian_mixture",
    "gmm": "gaussian_mixture",
}
DEFAULT_METHODS = ["kmeans", "dbscan", "gaussian_mixture"]
DEFAULT_METRICS = ["silhouette", "calinski_harabasz", "log_likelihood"]
DEFAULT_WEIGHTS = {
    "silhouette": 0.4,
    "calinski_harabasz": 0.3,
    "log_likelihood": 0.3,
}
COVARIANCE_TYPES = {"full", "tied", "diag", "spherical"}


@dataclass(frozen=True)
class Candidate:
    method: str
    params: dict[str, Any]
    params_key: str
    sort_key: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "params_key": self.params_key,
            "sort_key": self.sort_key,
        }


@dataclass
class ExperimentConfig:
    path: Path
    raw: dict[str, Any]
    description: DataDescription
    description_path: Path
    config_hash: str
    name: str
    seed: int
    seed_source: str
    target: dict[str, Any]
    preprocessing: dict[str, Any]
    enabled_methods: list[str]
    candidates: list[Candidate]
    evaluation: dict[str, Any]
    execution: dict[str, Any]
    failures: list[Failure] = field(default_factory=list)

    def candidate_by_key(self) -> dict[str, Candidate]:
        return {candidate.sort_key: candidate for candidate in self.candidates}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BusinessError(
            E_CONFIG_INVALID,
            f"experiment config does not exist: {path}",
            stage="config",
            path=str(path),
            fatal=True,
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"cannot parse experiment config: {exc}",
            stage="config",
            path=str(path),
            fatal=True,
        ) from exc
    if not isinstance(raw, dict):
        raise BusinessError(
            E_CONFIG_INVALID,
            "experiment config root must be a mapping",
            stage="config",
            path=str(path),
            fatal=True,
        )
    return raw


def _config_failure(message: str, path: str, candidate_key: str | None = None) -> Failure:
    return Failure(
        code=E_CONFIG_INVALID,
        message=message,
        stage="config",
        candidate_key=candidate_key,
        path=path,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_candidate_params(
    method: str,
    params: dict[str, Any],
    n_samples: int,
    *,
    error_code: str = E_HYPERPARAM_OUT_OF_RANGE,
) -> dict[str, Any]:
    """Validate and normalize one candidate's hyperparameters."""
    normalized = dict(params)
    if method == "kmeans":
        n_clusters = normalized.get("n_clusters", 0)
        if not _positive_int(n_clusters) or int(n_clusters) > n_samples:
            raise BusinessError(
                error_code,
                f"KMeans n_clusters must be in [1, {n_samples}]; got {n_clusters!r}",
                stage="clustering",
                path="n_clusters",
                extra={"parameter": "n_clusters", "value": n_clusters},
            )
        max_iter = normalized.get("max_iter", 100)
        if not _positive_int(max_iter):
            raise BusinessError(
                error_code,
                f"KMeans max_iter must be a positive integer; got {max_iter!r}",
                stage="clustering",
                path="max_iter",
                extra={"parameter": "max_iter", "value": max_iter},
            )
        tol = normalized.get("tol", 0.0001)
        if not _is_number(tol) or float(tol) <= 0:
            raise BusinessError(
                error_code,
                f"KMeans tol must be positive and finite; got {tol!r}",
                stage="clustering",
                path="tol",
                extra={"parameter": "tol", "value": tol},
            )
        return {
            "n_clusters": int(n_clusters),
            "max_iter": int(max_iter),
            "tol": float(tol),
        }

    if method == "dbscan":
        eps = normalized.get("eps", 0.0)
        if not _is_number(eps):
            raise BusinessError(
                error_code,
                f"DBSCAN eps must be finite; got {eps!r}",
                stage="clustering",
                path="eps",
                extra={"parameter": "eps", "value": eps},
            )
        if float(eps) < 0:
            raise BusinessError(
                E_DBSCAN_RADIUS_INVALID,
                f"DBSCAN eps must not be negative; got {eps!r}",
                stage="clustering",
                path="eps",
                extra={"parameter": "eps", "value": eps},
            )
        if float(eps) == 0:
            raise BusinessError(
                error_code,
                "DBSCAN eps must be positive; got 0",
                stage="clustering",
                path="eps",
                extra={"parameter": "eps", "value": eps},
            )
        min_samples = normalized.get("min_samples", 0)
        if not _positive_int(min_samples):
            raise BusinessError(
                error_code,
                f"DBSCAN min_samples must be a positive integer; got {min_samples!r}",
                stage="clustering",
                path="min_samples",
                extra={"parameter": "min_samples", "value": min_samples},
            )
        return {"eps": float(eps), "min_samples": int(min_samples)}

    if method == "gaussian_mixture":
        n_components = normalized.get("n_components", 0)
        if (
            not _positive_int(n_components)
            or int(n_components) > n_samples
        ):
            raise BusinessError(
                error_code,
                f"GMM n_components must be in [1, {n_samples}]; got {n_components!r}",
                stage="clustering",
                path="n_components",
                extra={"parameter": "n_components", "value": n_components},
            )
        covariance_type = normalized.get("covariance_type", "full")
        if covariance_type not in COVARIANCE_TYPES:
            raise BusinessError(
                error_code,
                f"GMM covariance_type must be one of {sorted(COVARIANCE_TYPES)}; got {covariance_type!r}",
                stage="clustering",
                path="covariance_type",
                extra={"parameter": "covariance_type", "value": covariance_type},
            )
        max_iter = normalized.get("max_iter", 100)
        if not _positive_int(max_iter):
            raise BusinessError(
                error_code,
                f"GMM max_iter must be a positive integer; got {max_iter!r}",
                stage="clustering",
                path="max_iter",
                extra={"parameter": "max_iter", "value": max_iter},
            )
        tol = normalized.get("tol", 0.0001)
        if not _is_number(tol) or float(tol) <= 0:
            raise BusinessError(
                error_code,
                f"GMM tol must be positive and finite; got {tol!r}",
                stage="clustering",
                path="tol",
                extra={"parameter": "tol", "value": tol},
            )
        return {
            "n_components": int(n_components),
            "covariance_type": str(covariance_type),
            "max_iter": int(max_iter),
            "tol": float(tol),
        }

    raise BusinessError(
        error_code,
        f"unsupported clustering method: {method!r}",
        stage="clustering",
        path="method",
    )


def candidate_from_params(method: str, params: dict[str, Any], n_samples: int) -> Candidate:
    canonical_method = METHOD_ALIASES.get(str(method).strip().lower())
    if canonical_method is None:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"unknown clustering method: {method!r}",
            stage="config",
            path="clustering.methods",
        )
    normalized = validate_candidate_params(canonical_method, params, n_samples, error_code=E_CONFIG_INVALID)
    params_key = canonical_json(normalized)
    sort_key = f"{canonical_method}:{sha256_value(params_key)[:16]}"
    return Candidate(
        method=canonical_method,
        params=normalized,
        params_key=params_key,
        sort_key=sort_key,
    )


def _parse_methods(raw_clustering: dict[str, Any], failures: list[Failure]) -> list[str]:
    methods_raw = raw_clustering.get("methods")
    if methods_raw is None:
        failures.append(_config_failure("clustering.methods is missing; using all built-in methods", "clustering.methods"))
        return list(DEFAULT_METHODS)
    if not isinstance(methods_raw, list) or not methods_raw:
        failures.append(_config_failure("clustering.methods must be a non-empty list", "clustering.methods"))
        return list(DEFAULT_METHODS)
    methods: list[str] = []
    for index, value in enumerate(methods_raw):
        canonical = METHOD_ALIASES.get(str(value).strip().lower())
        if canonical is None:
            failures.append(
                _config_failure(f"unknown clustering method: {value!r}", f"clustering.methods[{index}]")
            )
            continue
        if canonical not in methods:
            methods.append(canonical)
    return methods or list(DEFAULT_METHODS)


def _raw_candidates(raw_clustering: dict[str, Any], failures: list[Failure]) -> list[tuple[str, dict[str, Any], str]]:
    candidates_raw = raw_clustering.get("candidates")
    entries: list[tuple[str, dict[str, Any], str]] = []
    if isinstance(candidates_raw, dict):
        for method_raw, values in candidates_raw.items():
            method = METHOD_ALIASES.get(str(method_raw).strip().lower())
            if method is None:
                failures.append(
                    _config_failure(f"unknown candidate method: {method_raw!r}", f"clustering.candidates.{method_raw}")
                )
                continue
            if not isinstance(values, list):
                failures.append(
                    _config_failure(
                        f"candidates for {method_raw} must be a list",
                        f"clustering.candidates.{method_raw}",
                    )
                )
                continue
            for index, params in enumerate(values):
                if not isinstance(params, dict):
                    failures.append(
                        _config_failure(
                            f"candidate for {method_raw} must be a mapping",
                            f"clustering.candidates.{method_raw}[{index}]",
                        )
                    )
                    continue
                entries.append((method, params, f"clustering.candidates.{method_raw}[{index}]"))
    elif isinstance(candidates_raw, list):
        for index, item in enumerate(candidates_raw):
            if not isinstance(item, dict):
                failures.append(_config_failure("candidate must be a mapping", f"clustering.candidates[{index}]"))
                continue
            method_raw = item.get("method")
            method = METHOD_ALIASES.get(str(method_raw).strip().lower()) if method_raw is not None else None
            params = item.get("params", {key: value for key, value in item.items() if key != "method"})
            if method is None:
                failures.append(
                    _config_failure(f"candidate method is invalid: {method_raw!r}", f"clustering.candidates[{index}]")
                )
                continue
            if not isinstance(params, dict):
                failures.append(
                    _config_failure("candidate params must be a mapping", f"clustering.candidates[{index}].params")
                )
                continue
            entries.append((method, params, f"clustering.candidates[{index}]"))
    else:
        failures.append(_config_failure("clustering.candidates must be a mapping or list", "clustering.candidates"))
    return entries


def _parse_clustering(
    raw: dict[str, Any],
    description: DataDescription,
    failures: list[Failure],
) -> tuple[list[str], list[Candidate]]:
    clustering = raw.get("clustering")
    if not isinstance(clustering, dict):
        failures.append(_config_failure("missing required clustering structure", "clustering"))
        clustering = {}
    methods = _parse_methods(clustering, failures)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for method, params, location in _raw_candidates(clustering, failures):
        if method not in methods:
            failures.append(
                _config_failure(
                    f"candidate method {method} is not enabled by clustering.methods",
                    location,
                )
            )
            continue
        try:
            candidate = candidate_from_params(method, params, description.rows)
        except BusinessError as exc:
            failures.append(
                _config_failure(exc.failure.message, location, candidate_key=f"{method}:raw")
            )
            continue
        if candidate.sort_key in seen:
            failures.append(
                _config_failure(
                    f"duplicate candidate parameters for {method}: {candidate.params_key}",
                    location,
                    candidate_key=candidate.sort_key,
                )
            )
            continue
        seen.add(candidate.sort_key)
        candidates.append(candidate)
    for method in methods:
        if not any(candidate.method == method for candidate in candidates):
            failures.append(
                _config_failure(f"no valid candidates supplied for enabled method {method}", "clustering.candidates")
            )
    candidates.sort(key=lambda candidate: candidate.sort_key)
    return methods, candidates


def _parse_preprocessing(
    raw: dict[str, Any],
    description: DataDescription,
    failures: list[Failure],
) -> dict[str, Any]:
    raw_preprocessing = raw.get("preprocessing")
    if not isinstance(raw_preprocessing, dict):
        failures.append(_config_failure("missing required preprocessing structure", "preprocessing"))
        raw_preprocessing = {}

    method_raw = raw_preprocessing.get("method", "standardize")
    aliases = {
        "standardize": "standardize",
        "standardization": "standardize",
        "standard_scaler": "standardize",
        "zscore": "standardize",
        "quantile": "quantile",
        "quantile_transform": "quantile",
        "quantile_transformer": "quantile",
    }
    method = aliases.get(str(method_raw).strip().lower())
    if method is None:
        failures.append(
            _config_failure(f"unsupported preprocessing method: {method_raw!r}", "preprocessing.method")
        )
        method = "standardize"

    fit_fraction_raw = raw_preprocessing.get("fit_fraction", 0.8)
    if not _is_number(fit_fraction_raw) or not 0 < float(fit_fraction_raw) <= 1:
        failures.append(
            _config_failure("preprocessing.fit_fraction must be in (0, 1]", "preprocessing.fit_fraction")
        )
        fit_fraction = 0.8
    else:
        fit_fraction = float(fit_fraction_raw)

    expected_types_raw = raw_preprocessing.get("expected_types", {})
    if not isinstance(expected_types_raw, dict):
        failures.append(
            _config_failure("preprocessing.expected_types must be a mapping", "preprocessing.expected_types")
        )
        expected_types_raw = {}

    include_raw = raw_preprocessing.get("include_columns")
    if include_raw is None:
        include_raw = raw_preprocessing.get("features")
    if include_raw is None and isinstance(raw_preprocessing.get("columns"), dict):
        include_raw = list(raw_preprocessing["columns"].keys())
        for name, declared in raw_preprocessing["columns"].items():
            expected_types_raw.setdefault(name, declared)
    if include_raw is None:
        include_raw = [column.name for column in description.columns]
    if not isinstance(include_raw, list) or not all(isinstance(name, str) for name in include_raw):
        failures.append(
            _config_failure("preprocessing.include_columns must be a string list", "preprocessing.include_columns")
        )
        include_raw = [column.name for column in description.columns]

    description_map = description.column_map
    valid_names: list[str] = []
    for index, name in enumerate(include_raw):
        if name not in description_map:
            failures.append(
                _config_failure(f"unknown column: {name}", f"preprocessing.include_columns[{index}]")
            )
            continue
        if name not in valid_names:
            valid_names.append(name)

    for index, (name, expected) in enumerate(expected_types_raw.items()):
        if not isinstance(name, str):
            failures.append(
                _config_failure("expected_types keys must be column names", f"preprocessing.expected_types[{index}]")
            )
            continue
        if name not in description_map:
            failures.append(
                _config_failure(f"unknown column in expected_types: {name}", f"preprocessing.expected_types.{name}")
            )
            continue
        expected_kind = normalize_kind(expected)
        actual_kind = description_map[name].kind
        if expected_kind is None or expected_kind != actual_kind:
            failures.append(
                _config_failure(
                    f"type mismatch for {name}: expected {expected!r}, description has {actual_kind}",
                    f"preprocessing.expected_types.{name}",
                )
            )
            valid_names = [value for value in valid_names if value != name]

    drop_raw = raw_preprocessing.get("drop_columns", [])
    if drop_raw is None:
        drop_raw = []
    if not isinstance(drop_raw, list):
        failures.append(_config_failure("preprocessing.drop_columns must be a list", "preprocessing.drop_columns"))
        drop_raw = []
    drop_names: set[str] = set()
    for index, name in enumerate(drop_raw):
        if not isinstance(name, str):
            failures.append(
                _config_failure("drop_columns entries must be strings", f"preprocessing.drop_columns[{index}]")
            )
        elif name not in description_map:
            failures.append(
                _config_failure(f"unknown column in drop_columns: {name}", f"preprocessing.drop_columns[{index}]")
            )
        else:
            drop_names.add(name)
    valid_names = [name for name in valid_names if name not in drop_names]
    if not valid_names:
        raise BusinessError(
            E_CONFIG_INVALID,
            "no usable feature columns remain after validation",
            stage="config",
            fatal=True,
        )

    quantile_raw = raw_preprocessing.get("quantile", {})
    if not isinstance(quantile_raw, dict):
        failures.append(_config_failure("preprocessing.quantile must be a mapping", "preprocessing.quantile"))
        quantile_raw = {}
    n_quantiles = quantile_raw.get("n_quantiles", 32)
    if not isinstance(n_quantiles, int) or isinstance(n_quantiles, bool) or n_quantiles < 2:
        failures.append(
            _config_failure("quantile.n_quantiles must be an integer >= 2", "preprocessing.quantile.n_quantiles")
        )
        n_quantiles = 32
    output_distribution = quantile_raw.get("output_distribution", "uniform")
    if output_distribution not in {"uniform", "normal"}:
        failures.append(
            _config_failure(
                "quantile.output_distribution must be uniform or normal",
                "preprocessing.quantile.output_distribution",
            )
        )
        output_distribution = "uniform"

    return {
        "method": method,
        "fit_fraction": fit_fraction,
        "include_columns": valid_names,
        "drop_columns": sorted(drop_names),
        "quantile": {
            "n_quantiles": int(n_quantiles),
            "output_distribution": output_distribution,
        },
        "column_descriptions": {
            name: {
                "name": name,
                "kind": description_map[name].kind,
                "declared_kind": description_map[name].declared_kind,
                "options": description_map[name].options,
            }
            for name in valid_names
        },
    }


def _parse_evaluation(raw: dict[str, Any], failures: list[Failure]) -> dict[str, Any]:
    evaluation_raw = raw.get("evaluation")
    if not isinstance(evaluation_raw, dict):
        failures.append(_config_failure("missing required evaluation structure", "evaluation"))
        evaluation_raw = {}
    metrics_raw = evaluation_raw.get("metrics", DEFAULT_METRICS)
    if not isinstance(metrics_raw, list) or not metrics_raw:
        failures.append(_config_failure("evaluation.metrics must be a non-empty list", "evaluation.metrics"))
        metrics_raw = DEFAULT_METRICS
    metrics: list[str] = []
    for index, metric in enumerate(metrics_raw):
        if metric not in DEFAULT_METRICS:
            failures.append(
                _config_failure(f"unsupported metric: {metric!r}", f"evaluation.metrics[{index}]")
            )
        elif metric not in metrics:
            metrics.append(metric)
    for metric in DEFAULT_METRICS:
        if metric not in metrics:
            metrics.append(metric)

    primary = evaluation_raw.get("primary", "composite")
    if primary != "composite" and primary not in DEFAULT_METRICS:
        failures.append(
            _config_failure(
                f"evaluation.primary must be composite or one of {DEFAULT_METRICS}",
                "evaluation.primary",
            )
        )
        primary = "composite"
    weights_raw = evaluation_raw.get("weights", DEFAULT_WEIGHTS)
    if not isinstance(weights_raw, dict):
        failures.append(_config_failure("evaluation.weights must be a mapping", "evaluation.weights"))
        weights_raw = DEFAULT_WEIGHTS
    weights: dict[str, float] = {}
    for metric in DEFAULT_METRICS:
        value = weights_raw.get(metric, DEFAULT_WEIGHTS[metric])
        if not _is_number(value) or float(value) < 0:
            failures.append(
                _config_failure(
                    f"evaluation weight for {metric} must be a non-negative finite number",
                    f"evaluation.weights.{metric}",
                )
            )
            value = DEFAULT_WEIGHTS[metric]
        weights[metric] = float(value)
    if sum(weights.values()) <= 0:
        failures.append(_config_failure("evaluation weights cannot all be zero", "evaluation.weights"))
        weights = dict(DEFAULT_WEIGHTS)
    return {
        "metrics": metrics,
        "primary": primary,
        "weights": weights,
        "silhouette_epsilon": 1e-12,
    }


def _parse_execution(raw: dict[str, Any], failures: list[Failure]) -> dict[str, Any]:
    execution_raw = raw.get("execution")
    if not isinstance(execution_raw, dict):
        failures.append(_config_failure("missing required execution structure", "execution"))
        execution_raw = {}

    concurrency = execution_raw.get("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0:
        failures.append(
            _config_failure("execution.concurrency must be a positive integer", "execution.concurrency")
        )
        concurrency = 1
    commit_interval = execution_raw.get("snapshot_commit_interval", 1)
    if not isinstance(commit_interval, int) or isinstance(commit_interval, bool) or commit_interval <= 0:
        failures.append(
            _config_failure(
                "execution.snapshot_commit_interval must be a positive integer",
                "execution.snapshot_commit_interval",
            )
        )
        commit_interval = 1
    latency_budget = execution_raw.get("interaction_latency_budget_ms", 1500)
    if not _is_number(latency_budget) or float(latency_budget) <= 0:
        failures.append(
            _config_failure(
                "execution.interaction_latency_budget_ms must be positive",
                "execution.interaction_latency_budget_ms",
            )
        )
        latency_budget = 1500.0
    snapshot_version = execution_raw.get("snapshot_version", 1)
    if not isinstance(snapshot_version, int) or isinstance(snapshot_version, bool) or snapshot_version <= 0:
        failures.append(
            _config_failure("execution.snapshot_version must be a positive integer", "execution.snapshot_version")
        )
        snapshot_version = 1
    return {
        "concurrency": int(concurrency),
        "snapshot_commit_interval": int(commit_interval),
        "interaction_latency_budget_ms": float(latency_budget),
        "snapshot_version": int(snapshot_version),
    }


def load_experiment_config(
    config_path: Path,
    data_description_override: Path | None = None,
) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    raw = _load_yaml(config_path)
    failures: list[Failure] = []

    description_raw = raw.get("data")
    if not isinstance(description_raw, dict):
        failures.append(_config_failure("missing required data structure", "data"))
        description_raw = {}
    description_value = data_description_override
    if description_value is None:
        configured = description_raw.get("description")
        if isinstance(configured, str) and configured.strip():
            description_value = Path(configured)
        else:
            failures.append(
                _config_failure("data.description is missing", "data.description")
            )
            description_value = Path("data_description.yaml")
    if not description_value.is_absolute():
        description_value = (config_path.parent / description_value).resolve()
    description = load_data_description(description_value)
    failures.extend(description.failures)

    target_raw = raw.get("target")
    if not isinstance(target_raw, dict):
        failures.append(_config_failure("missing required target structure", "target"))
        target = {}
    else:
        target = target_raw
        task = target.get("task")
        if task not in {"clustering", "cluster", "unsupervised_clustering"}:
            failures.append(
                _config_failure("target.task must identify a clustering task", "target.task")
            )

    experiment_raw = raw.get("experiment")
    if not isinstance(experiment_raw, dict):
        failures.append(_config_failure("missing required experiment structure", "experiment"))
        experiment_raw = {}
    name = experiment_raw.get("name", config_path.stem)
    if not isinstance(name, str) or not name.strip():
        failures.append(_config_failure("experiment.name must be a non-empty string", "experiment.name"))
        name = config_path.stem

    config_seed = experiment_raw.get("seed")
    if config_seed is None:
        seed = description.seed
        seed_source = "data.description"
    elif valid_seed(config_seed):
        seed = int(config_seed)
        seed_source = "experiment.seed"
    else:
        failures.append(
            _config_failure(
                "experiment.seed must be a non-negative integer below 2**63",
                "experiment.seed",
            )
        )
        seed = description.seed
        seed_source = "data.description.fallback"

    preprocessing = _parse_preprocessing(raw, description, failures)
    methods, candidates = _parse_clustering(raw, description, failures)
    evaluation = _parse_evaluation(raw, failures)
    execution = _parse_execution(raw, failures)
    return ExperimentConfig(
        path=config_path,
        raw=raw,
        description=description,
        description_path=description.path,
        config_hash=sha256_value(raw),
        name=name.strip(),
        seed=seed,
        seed_source=seed_source,
        target=target,
        preprocessing=preprocessing,
        enabled_methods=methods,
        candidates=candidates,
        evaluation=evaluation,
        execution=execution,
        failures=failures,
    )
