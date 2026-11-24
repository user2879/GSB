from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import (
    E_CLUSTER_COUNT_RANGE,
    E_CLUSTER_COUNT_ZERO,
    E_COLUMN_DESCRIPTION_MISSING,
    E_COLUMN_UNKNOWN,
    E_CONFIG_INVALID,
    E_CONCURRENCY_NON_POSITIVE,
    E_EPS_NON_POSITIVE,
    E_GLOBAL_BUDGET_NON_POSITIVE,
    E_LATENCY_BUDGET_NON_POSITIVE,
    E_TYPE_MISMATCH,
    Issue,
)
from .util import as_float, as_int, deep_copy, merge_dict, read_yaml_mapping, stable_hash


SUPPORTED_METHODS = {"kmeans", "dbscan", "gmm"}
SUPPORTED_PREPROCESSING = {"standardize", "quantile"}
SUPPORTED_COVARIANCE = {"full", "diag", "spherical", "tied"}


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "seed": 1,
    "preprocessing": {
        "method": "standardize",
        "train_fraction": 1.0,
        "missing_fill": "median",
        "quantile_tie_policy": "average_rank",
    },
    "evaluation": {
        "metrics": ["silhouette", "calinski_harabasz", "density_log_likelihood"],
        "density_regularization": 1e-6,
    },
    "methods": {
        "kmeans": {"max_iter": 25, "tolerance": 1e-8},
        "dbscan": {"metric": "euclidean"},
        "gmm": {
            "max_iter": 25,
            "tolerance": 1e-6,
            "covariance_regularization": 1e-5,
            "init": "farthest",
        },
    },
    "scheduler": {
        "global_iteration_budget": 300,
        "per_candidate_iteration_quota": 40,
        "concurrency": 4,
        "interaction_latency_budget_ms": 250,
        "snapshot_commit_interval": 4,
    },
    "candidates": [],
    "injections": [],
}


def load_config(path: str | Path) -> tuple[dict[str, Any], list[Issue]]:
    raw, error = read_yaml_mapping(path)
    if error is not None or raw is None:
        issue = Issue(
            E_CONFIG_INVALID,
            error or "configuration must be a YAML mapping",
            context={"path": str(path), "scope": "config"},
        )
        fallback = deep_copy(DEFAULT_CONFIG)
        return fallback, [issue]
    return normalize_config(raw)


def normalize_config(raw: dict[str, Any]) -> tuple[dict[str, Any], list[Issue]]:
    issues: list[Issue] = []
    config = merge_dict(DEFAULT_CONFIG, raw if isinstance(raw, dict) else {})

    version = as_int(config.get("schema_version"))
    if version != 1:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "schema_version must be 1",
                context={"scope": "config", "field": "schema_version"},
            )
        )
    config["schema_version"] = 1

    seed = as_int(config.get("seed"))
    if seed is None:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "seed must be an integer",
                context={"scope": "config", "field": "seed"},
            )
        )
        config["seed"] = 1
    else:
        config["seed"] = seed

    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        issues.append(
            Issue(E_CONFIG_INVALID, "preprocessing must be a mapping", context={"scope": "config"})
        )
        preprocessing = deep_copy(DEFAULT_CONFIG["preprocessing"])
    method = preprocessing.get("method", "standardize")
    if method not in SUPPORTED_PREPROCESSING:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                f"unsupported preprocessing method: {method!r}",
                context={"scope": "preprocessing", "field": "method"},
            )
        )
        method = "standardize"
    train_fraction = as_float(preprocessing.get("train_fraction", 1.0))
    if train_fraction is None or not (0.0 < train_fraction <= 1.0):
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "preprocessing.train_fraction must be in (0, 1]",
                context={"scope": "preprocessing", "field": "train_fraction"},
            )
        )
        train_fraction = 1.0
    fill = preprocessing.get("missing_fill", "median")
    if fill != "median":
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "only deterministic median missing-fill is supported",
                context={"scope": "preprocessing", "field": "missing_fill"},
            )
        )
        fill = "median"
    tie_policy = preprocessing.get("quantile_tie_policy", "average_rank")
    if tie_policy not in {"average_rank", "lower_rank", "upper_rank"}:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "unknown quantile tie policy",
                context={"scope": "preprocessing", "field": "quantile_tie_policy"},
            )
        )
        tie_policy = "average_rank"
    preprocessing = {
        "method": method,
        "train_fraction": train_fraction,
        "missing_fill": fill,
        "quantile_tie_policy": tie_policy,
    }
    config["preprocessing"] = preprocessing

    methods = config.get("methods")
    if not isinstance(methods, dict):
        issues.append(
            Issue(E_CONFIG_INVALID, "methods must be a mapping", context={"scope": "config"})
        )
        methods = {}
    normalized_methods = deep_copy(DEFAULT_CONFIG["methods"])
    for name in SUPPORTED_METHODS:
        supplied = methods.get(name, {})
        if not isinstance(supplied, dict):
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"methods.{name} must be a mapping",
                    context={"scope": "methods", "field": name},
                )
            )
            supplied = {}
        normalized_methods[name] = merge_dict(normalized_methods.get(name, {}), supplied)
    for name, settings in normalized_methods.items():
        max_iter = as_int(settings.get("max_iter", 25))
        if max_iter is None or max_iter <= 0:
            issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    f"methods.{name}.max_iter must be a positive integer",
                    context={"scope": "methods", "field": f"{name}.max_iter"},
                )
            )
            max_iter = 25
        settings["max_iter"] = max_iter
        if "tolerance" in settings:
            tolerance = as_float(settings["tolerance"])
            if tolerance is None or tolerance < 0:
                issues.append(
                    Issue(
                        E_CONFIG_INVALID,
                        f"methods.{name}.tolerance must be finite and non-negative",
                        context={"scope": "methods", "field": f"{name}.tolerance"},
                    )
                )
                tolerance = 1e-6
            settings["tolerance"] = tolerance
    gmm_settings = normalized_methods["gmm"]
    dbscan_metric = normalized_methods["dbscan"].get("metric", "euclidean")
    if dbscan_metric != "euclidean":
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "only deterministic euclidean DBSCAN distance is supported",
                context={"scope": "methods", "field": "dbscan.metric"},
            )
        )
        dbscan_metric = "euclidean"
    normalized_methods["dbscan"]["metric"] = dbscan_metric
    gmm_init = gmm_settings.get("init", "farthest")
    if gmm_init != "farthest":
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "only deterministic farthest GMM initialization is supported",
                context={"scope": "methods", "field": "gmm.init"},
            )
        )
        gmm_init = "farthest"
    gmm_settings["init"] = gmm_init
    regularization = as_float(gmm_settings.get("covariance_regularization", 1e-5))
    if regularization is None or regularization <= 0:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "gmm covariance_regularization must be positive",
                context={"scope": "methods", "field": "gmm.covariance_regularization"},
            )
        )
        regularization = 1e-5
    gmm_settings["covariance_regularization"] = regularization
    config["methods"] = normalized_methods

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict):
        issues.append(
            Issue(E_CONFIG_INVALID, "evaluation must be a mapping", context={"scope": "config"})
        )
        evaluation = deep_copy(DEFAULT_CONFIG["evaluation"])
    metrics = evaluation.get("metrics", DEFAULT_CONFIG["evaluation"]["metrics"])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        issues.append(
            Issue(
                E_TYPE_MISMATCH,
                "evaluation.metrics must be a list of strings",
                context={"scope": "evaluation", "field": "metrics"},
            )
        )
        metrics = deep_copy(DEFAULT_CONFIG["evaluation"]["metrics"])
    supported_metrics = {"silhouette", "calinski_harabasz", "density_log_likelihood"}
    unknown_metrics = [metric for metric in metrics if metric not in supported_metrics]
    if unknown_metrics:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "unsupported evaluation metrics were removed",
                context={"scope": "evaluation", "field": "metrics", "metrics": unknown_metrics},
            )
        )
        metrics = [metric for metric in metrics if metric in supported_metrics]
        if not metrics:
            metrics = deep_copy(DEFAULT_CONFIG["evaluation"]["metrics"])
    density_reg = as_float(evaluation.get("density_regularization", 1e-6))
    if density_reg is None or density_reg <= 0:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "evaluation.density_regularization must be positive",
                context={"scope": "evaluation", "field": "density_regularization"},
            )
        )
        density_reg = 1e-6
    config["evaluation"] = {
        "metrics": list(dict.fromkeys(metrics)),
        "density_regularization": density_reg,
    }

    scheduler = config.get("scheduler")
    if not isinstance(scheduler, dict):
        issues.append(
            Issue(E_CONFIG_INVALID, "scheduler must be a mapping", context={"scope": "config"})
        )
        scheduler = deep_copy(DEFAULT_CONFIG["scheduler"])
    scheduler = merge_dict(DEFAULT_CONFIG["scheduler"], scheduler)

    budget = as_int(scheduler.get("global_iteration_budget"))
    if budget is None or budget <= 0:
        issues.append(
            Issue(
                E_GLOBAL_BUDGET_NON_POSITIVE,
                "global_iteration_budget must be a positive integer",
                context={"scope": "scheduler", "field": "global_iteration_budget"},
            )
        )
        budget = max(0, budget or 0)
    scheduler["global_iteration_budget"] = budget

    quota = as_int(scheduler.get("per_candidate_iteration_quota"))
    if quota is None or quota <= 0:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "per_candidate_iteration_quota must be a positive integer",
                context={"scope": "scheduler", "field": "per_candidate_iteration_quota"},
            )
        )
        quota = 1
    scheduler["per_candidate_iteration_quota"] = quota

    concurrency = as_int(scheduler.get("concurrency"))
    if concurrency is None or concurrency <= 0:
        issues.append(
            Issue(
                E_CONCURRENCY_NON_POSITIVE,
                "concurrency must be a positive integer",
                context={"scope": "scheduler", "field": "concurrency"},
            )
        )
        concurrency = 1
    scheduler["concurrency"] = concurrency

    latency = as_float(scheduler.get("interaction_latency_budget_ms"))
    if latency is None or latency <= 0:
        issues.append(
            Issue(
                E_LATENCY_BUDGET_NON_POSITIVE,
                "interaction_latency_budget_ms must be positive",
                context={"scope": "scheduler", "field": "interaction_latency_budget_ms"},
            )
        )
        latency = 0.0
    scheduler["interaction_latency_budget_ms"] = latency

    interval = as_int(scheduler.get("snapshot_commit_interval", 4))
    if interval is None or interval <= 0:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "snapshot_commit_interval must be a positive integer",
                context={"scope": "scheduler", "field": "snapshot_commit_interval"},
            )
        )
        interval = 1
    scheduler["snapshot_commit_interval"] = interval
    config["scheduler"] = scheduler

    candidates = config.get("candidates", [])
    if not isinstance(candidates, list):
        issues.append(
            Issue(E_TYPE_MISMATCH, "candidates must be a list", context={"scope": "config"})
        )
        candidates = []
    config["candidates"] = deep_copy(candidates)

    injections = config.get("injections", [])
    if not isinstance(injections, list):
        issues.append(
            Issue(E_TYPE_MISMATCH, "injections must be a list", context={"scope": "config"})
        )
        injections = []
    config["injections"] = deep_copy(injections)

    return config, issues


def runtime_settings(config: dict[str, Any], method: str, params: dict[str, Any]) -> dict[str, Any]:
    defaults = deep_copy(config["methods"].get(method, {}))
    settings = merge_dict(defaults, {k: v for k, v in params.items() if k in defaults})
    if method in {"kmeans", "gmm"}:
        maximum = as_int(params.get("max_iter", settings.get("max_iter", 25)))
        if maximum is None or maximum <= 0:
            maximum = 25
        settings["max_iter"] = maximum
        tolerance = as_float(params.get("tolerance", settings.get("tolerance", 1e-6)))
        if tolerance is None or tolerance < 0:
            tolerance = 1e-6
        settings["tolerance"] = tolerance
    return settings


def validate_candidate(
    candidate: Any,
    *,
    row_count: int,
    available_columns: set[str],
    declared_columns: set[str],
) -> tuple[dict[str, Any] | None, list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(candidate, dict):
        return None, [
            Issue(E_TYPE_MISMATCH, "candidate must be a mapping", context={"scope": "candidate"})
        ]
    identifier = candidate.get("id")
    if not isinstance(identifier, str) or not identifier.strip():
        issues.append(
            Issue(E_CONFIG_INVALID, "candidate.id must be a non-empty string", context={"scope": "candidate"})
        )
        identifier = f"candidate-{stable_hash(candidate)[:8]}"
    method = candidate.get("method")
    if method not in SUPPORTED_METHODS:
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                f"unsupported method: {method!r}",
                context={"scope": "candidate", "candidate": identifier},
            )
        )
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        issues.append(
            Issue(
                E_TYPE_MISMATCH,
                "candidate.params must be a mapping",
                context={"scope": "candidate", "candidate": identifier},
            )
        )
        params = {}
    features = candidate.get("feature_columns")
    if features is None:
        features = sorted(available_columns)
    if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
        issues.append(
            Issue(
                E_TYPE_MISMATCH,
                "feature_columns must be a non-empty list of strings",
                context={"scope": "candidate", "candidate": identifier},
            )
        )
        features = sorted(available_columns)
    features = list(dict.fromkeys(features))
    for column in features:
        if column not in declared_columns:
            issues.append(
                Issue(
                    E_COLUMN_UNKNOWN if column not in available_columns else E_COLUMN_DESCRIPTION_MISSING,
                    f"candidate references unavailable column {column!r}",
                    context={"scope": "candidate", "candidate": identifier, "column": column},
                )
            )
        elif column not in available_columns:
            issues.append(
                Issue(
                    E_COLUMN_DESCRIPTION_MISSING,
                    f"column description is missing for {column!r}",
                    context={"scope": "candidate", "candidate": identifier, "column": column},
                )
            )

    train_fraction = as_float(candidate.get("train_fraction", 1.0))
    if train_fraction is None or not (0 < train_fraction <= 1):
        issues.append(
            Issue(
                E_CONFIG_INVALID,
                "candidate.train_fraction must be in (0, 1]",
                context={"scope": "candidate", "candidate": identifier},
            )
        )
        train_fraction = 1.0

    if method == "kmeans":
        n_clusters = params.get("n_clusters")
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int,)):
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    "kmeans.n_clusters must be an integer",
                    context={"scope": "candidate", "candidate": identifier, "parameter": "n_clusters"},
                )
            )
        elif n_clusters == 0:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_ZERO,
                    "kmeans n_clusters must not be zero",
                    context={"scope": "candidate", "candidate": identifier, "n_clusters": n_clusters},
                )
            )
        elif n_clusters <= 0:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_RANGE,
                    "kmeans n_clusters must be positive",
                    context={"scope": "candidate", "candidate": identifier, "n_clusters": n_clusters},
                )
            )
        elif n_clusters > row_count:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_RANGE,
                    "kmeans n_clusters exceeds sample count",
                    context={"scope": "candidate", "candidate": identifier, "n_clusters": n_clusters},
                )
            )
    elif method == "dbscan":
        eps = params.get("eps")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    "dbscan.eps must be numeric",
                    context={"scope": "candidate", "candidate": identifier, "parameter": "eps"},
                )
            )
        elif float(eps) <= 0:
            issues.append(
                Issue(
                    E_EPS_NON_POSITIVE,
                    "dbscan.eps must be positive",
                    context={"scope": "candidate", "candidate": identifier, "eps": eps},
                )
            )
        min_samples = params.get("min_samples")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples <= 0:
            issues.append(
                Issue(
                    E_CONFIG_INVALID if min_samples is not None else E_TYPE_MISMATCH,
                    "dbscan.min_samples must be a positive integer",
                    context={"scope": "candidate", "candidate": identifier, "parameter": "min_samples"},
                )
            )
    elif method == "gmm":
        n_components = params.get("n_components")
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    "gmm.n_components must be an integer",
                    context={"scope": "candidate", "candidate": identifier, "parameter": "n_components"},
                )
            )
        elif n_components == 0:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_ZERO,
                    "gmm n_components must not be zero",
                    context={"scope": "candidate", "candidate": identifier, "n_components": n_components},
                )
            )
        elif n_components <= 0:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_RANGE,
                    "gmm n_components must be positive",
                    context={"scope": "candidate", "candidate": identifier, "n_components": n_components},
                )
            )
        elif n_components > row_count:
            issues.append(
                Issue(
                    E_CLUSTER_COUNT_RANGE,
                    "gmm n_components exceeds sample count",
                    context={"scope": "candidate", "candidate": identifier, "n_components": n_components},
                )
            )
        covariance = params.get("covariance_type", "diag")
        if covariance not in SUPPORTED_COVARIANCE:
            issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    f"unsupported covariance_type: {covariance!r}",
                    context={"scope": "candidate", "candidate": identifier},
                )
            )
        if "covariance_regularization" in params:
            regularization = as_float(params.get("covariance_regularization"))
            if regularization is None or regularization <= 0:
                issues.append(
                    Issue(
                        E_CONFIG_INVALID,
                        "gmm.covariance_regularization must be a positive finite number",
                        context={
                            "scope": "candidate",
                            "candidate": identifier,
                            "parameter": "covariance_regularization",
                        },
                    )
                )

    if method in {"kmeans", "gmm"}:
        if "max_iter" in params:
            maximum = params.get("max_iter")
            if isinstance(maximum, bool) or not isinstance(maximum, int):
                issues.append(
                    Issue(
                        E_TYPE_MISMATCH,
                        f"{method}.max_iter must be an integer",
                        context={"scope": "candidate", "candidate": identifier, "parameter": "max_iter"},
                    )
                )
            elif maximum <= 0:
                issues.append(
                    Issue(
                        E_CONFIG_INVALID,
                        f"{method}.max_iter must be positive",
                        context={"scope": "candidate", "candidate": identifier, "parameter": "max_iter"},
                    )
                )
        if "tolerance" in params:
            tolerance = as_float(params.get("tolerance"))
            if tolerance is None or tolerance < 0:
                issues.append(
                    Issue(
                        E_TYPE_MISMATCH,
                        f"{method}.tolerance must be a finite non-negative number",
                        context={"scope": "candidate", "candidate": identifier, "parameter": "tolerance"},
                    )
                )

    normalized = {
        "id": identifier,
        "method": method,
        "params": deep_copy(params),
        "feature_columns": features,
        "train_fraction": train_fraction,
    }
    if issues:
        return normalized, issues
    return normalized, []
