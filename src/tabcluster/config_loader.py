"""Configuration & data-description loading with non-aborting validation.

Every invalid item produces a business code in the failure log; all other
valid items keep being parsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import (FailureLog, E_CONFIG_INVALID, E_CLUSTER_ZERO,
                     E_BUDGET_INVALID)
from .fingerprint import fingerprint

SUPPORTED_PREPROCESS = ("standard", "quantile")
SUPPORTED_METHODS = ("kmeans", "dbscan", "gmm")
SUPPORTED_METRICS = ("silhouette", "calinski_harabasz", "log_likelihood")
SUPPORTED_COVARIANCE = ("full", "diag", "spherical", "tied")
COLUMN_TYPES = ("numeric", "categorical", "high_cardinality", "redundant")

DEFAULT_GLOBAL_BUDGET = 1_000_000
DEFAULT_CANDIDATE_QUOTA = 10_000
DEFAULT_CONCURRENCY = 1
DEFAULT_SNAPSHOT_INTERVAL = 1


@dataclass
class ColumnSpec:
    name: str
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class DataDescription:
    seed: int
    base_rows: int
    scales: dict
    train_fraction: float
    columns: list  # list[ColumnSpec]
    raw: dict

    @property
    def desc_fingerprint(self) -> str:
        return fingerprint("data-description", self.raw)


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    sample_scale: str
    preprocess_method: str
    preprocess_params: dict
    methods: dict            # method -> list[param dict] candidate grids
    metrics: list
    global_budget: int
    candidate_quota: int
    concurrency: int
    snapshot_interval: int
    gmm_reg_covar: float
    gmm_tol: float
    kmeans_max_iter: int
    gmm_max_iter: int
    raw: dict

    @property
    def config_hash(self) -> str:
        return fingerprint("experiment-config", self.raw)


def _load_yaml(path: Path, failures: FailureLog, kind: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        failures.add(E_CONFIG_INVALID, f"{kind} file not found: {path}",
                     stage="load", path=str(path))
    except yaml.YAMLError as exc:
        failures.add(E_CONFIG_INVALID, f"{kind} YAML parse error: {exc}",
                     stage="load", path=str(path))
    return None


def load_data_description(path, failures: FailureLog) -> DataDescription | None:
    raw = _load_yaml(Path(path), failures, "data description")
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "data description root must be a mapping",
                     stage="load")
        return None

    seed = raw.get("seed", 0)
    base_rows = raw.get("base_rows", 0)
    scales = raw.get("scales", {"base": 1})
    train_fraction = raw.get("train_fraction", 0.8)
    columns_raw = raw.get("columns")

    if not isinstance(base_rows, int) or base_rows <= 0:
        failures.add(E_CONFIG_INVALID, "base_rows must be a positive integer",
                     stage="load", value=base_rows)
        base_rows = 0
    if not isinstance(columns_raw, list) or not columns_raw:
        failures.add(E_CONFIG_INVALID,
                     "target structure 'columns' missing or empty",
                     stage="load")
        columns_raw = []

    columns: list[ColumnSpec] = []
    seen: set[str] = set()
    known_names = [c.get("name") for c in columns_raw if isinstance(c, dict)]
    for idx, col in enumerate(columns_raw):
        if not isinstance(col, dict):
            failures.add(E_CONFIG_INVALID, f"column #{idx} is not a mapping",
                         stage="load")
            continue
        name = col.get("name")
        ctype = col.get("type")
        if not name or not isinstance(name, str):
            failures.add(E_CONFIG_INVALID, f"column #{idx} missing name",
                         stage="load")
            continue
        if name in seen:
            failures.add(E_CONFIG_INVALID, f"duplicate column name '{name}'",
                         stage="load", column=name)
            continue
        if ctype not in COLUMN_TYPES:
            failures.add(E_CONFIG_INVALID,
                         f"column '{name}' has unknown type '{ctype}'",
                         stage="load", column=name)
            continue
        params = {k: v for k, v in col.items() if k not in ("name", "type")}
        # type-conformity checks
        if ctype == "numeric":
            dist = params.get("distribution", "normal")
            if dist not in ("normal", "uniform", "int"):
                failures.add(E_CONFIG_INVALID,
                             f"numeric column '{name}' bad distribution '{dist}'",
                             stage="load", column=name)
                continue
        elif ctype == "categorical":
            cats = params.get("categories")
            if not isinstance(cats, list) or not cats:
                failures.add(E_CONFIG_INVALID,
                             f"categorical column '{name}' missing categories",
                             stage="load", column=name)
                continue
        elif ctype == "high_cardinality":
            card = params.get("cardinality")
            if not isinstance(card, int) or card <= 0:
                failures.add(E_CONFIG_INVALID,
                             f"high_cardinality column '{name}' bad cardinality",
                             stage="load", column=name)
                continue
        elif ctype == "redundant":
            src = params.get("source")
            if src not in known_names:
                failures.add(E_CONFIG_INVALID,
                             f"redundant column '{name}' references unknown "
                             f"column '{src}'", stage="load", column=name)
                continue
        seen.add(name)
        columns.append(ColumnSpec(name=name, type=ctype, params=params))

    return DataDescription(seed=int(seed), base_rows=base_rows,
                           scales=scales if isinstance(scales, dict) else {"base": 1},
                           train_fraction=float(train_fraction),
                           columns=columns, raw=raw)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def load_experiment_config(path, failures: FailureLog) -> ExperimentConfig | None:
    raw = _load_yaml(Path(path), failures, "experiment config")
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "experiment config root must be a mapping",
                     stage="load")
        return None

    name = str(raw.get("experiment_name", "experiment"))
    seed = int(raw.get("seed", 0))
    sample_scale = str(raw.get("sample_scale", "base"))

    prep = raw.get("preprocess", {})
    if not isinstance(prep, dict):
        failures.add(E_CONFIG_INVALID, "'preprocess' must be a mapping",
                     stage="load")
        prep = {}
    prep_method = prep.get("method", "standard")
    if prep_method not in SUPPORTED_PREPROCESS:
        failures.add(E_CONFIG_INVALID,
                     f"unknown preprocess method '{prep_method}', "
                     f"falling back to 'standard'", stage="load")
        prep_method = "standard"
    prep_params = {k: v for k, v in prep.items() if k != "method"}

    # --- methods / candidate grids ---------------------------------------
    methods_raw = raw.get("methods")
    methods: dict[str, list[dict]] = {}
    if not isinstance(methods_raw, dict) or not methods_raw:
        failures.add(E_CONFIG_INVALID,
                     "target structure 'methods' missing or empty", stage="load")
        methods_raw = {}
    for method, spec in methods_raw.items():
        if method not in SUPPORTED_METHODS:
            failures.add(E_CONFIG_INVALID, f"unknown method '{method}'",
                         stage="load", method=method)
            continue
        if not isinstance(spec, dict):
            failures.add(E_CONFIG_INVALID,
                         f"method '{method}' spec must be a mapping",
                         stage="load", method=method)
            continue
        grids: list[dict] = []
        if method == "kmeans":
            for k in spec.get("n_clusters", []):
                if not _positive_int(k):
                    code = E_CLUSTER_ZERO if k == 0 else E_CONFIG_INVALID
                    failures.add(code, f"kmeans n_clusters={k} invalid; "
                                       f"candidate dropped",
                                 stage="load", method=method, value=k)
                    continue
                grids.append({"n_clusters": int(k)})
        elif method == "dbscan":
            for eps in spec.get("eps", []):
                for ms in spec.get("min_samples", []):
                    bad = False
                    if not (isinstance(eps, (int, float)) and eps > 0):
                        failures.add(E_CONFIG_INVALID,
                                     f"dbscan eps={eps} invalid; candidate dropped",
                                     stage="load", method=method, value=eps)
                        bad = True
                    if not _positive_int(ms):
                        failures.add(E_CONFIG_INVALID,
                                     f"dbscan min_samples={ms} invalid; "
                                     f"candidate dropped",
                                     stage="load", method=method, value=ms)
                        bad = True
                    if not bad:
                        grids.append({"eps": float(eps), "min_samples": int(ms)})
        elif method == "gmm":
            for k in spec.get("n_components", []):
                for ct in spec.get("covariance_type", ["full"]):
                    bad = False
                    if not _positive_int(k):
                        code = E_CLUSTER_ZERO if k == 0 else E_CONFIG_INVALID
                        failures.add(code, f"gmm n_components={k} invalid; "
                                           f"candidate dropped",
                                     stage="load", method=method, value=k)
                        bad = True
                    if ct not in SUPPORTED_COVARIANCE:
                        failures.add(E_CONFIG_INVALID,
                                     f"gmm covariance_type='{ct}' invalid; "
                                     f"candidate dropped",
                                     stage="load", method=method, value=ct)
                        bad = True
                    if not bad:
                        grids.append({"n_components": int(k),
                                      "covariance_type": str(ct)})
        if grids:
            methods[method] = grids

    # --- metrics ----------------------------------------------------------
    metrics_raw = raw.get("metrics", list(SUPPORTED_METRICS))
    metrics: list[str] = []
    if not isinstance(metrics_raw, list):
        failures.add(E_CONFIG_INVALID, "'metrics' must be a list", stage="load")
        metrics_raw = list(SUPPORTED_METRICS)
    for m in metrics_raw:
        if m not in SUPPORTED_METRICS:
            failures.add(E_CONFIG_INVALID, f"unknown metric '{m}'", stage="load",
                         metric=m)
            continue
        metrics.append(str(m))
    if not metrics:
        metrics = list(SUPPORTED_METRICS)

    # --- budget / quota / concurrency ------------------------------------
    budget_raw = raw.get("budget", {})
    if not isinstance(budget_raw, dict):
        failures.add(E_CONFIG_INVALID, "'budget' must be a mapping", stage="load")
        budget_raw = {}
    global_budget = budget_raw.get("global_iterations", DEFAULT_GLOBAL_BUDGET)
    if not _positive_int(global_budget):
        failures.add(E_BUDGET_INVALID,
                     f"global iteration budget {global_budget!r} not positive; "
                     f"falling back to {DEFAULT_GLOBAL_BUDGET}", stage="load")
        global_budget = DEFAULT_GLOBAL_BUDGET
    candidate_quota = budget_raw.get("candidate_quota", DEFAULT_CANDIDATE_QUOTA)
    if not _positive_int(candidate_quota):
        failures.add(E_BUDGET_INVALID,
                     f"candidate quota {candidate_quota!r} not positive; "
                     f"falling back to {DEFAULT_CANDIDATE_QUOTA}", stage="load")
        candidate_quota = DEFAULT_CANDIDATE_QUOTA

    concurrency = raw.get("concurrency", DEFAULT_CONCURRENCY)
    if not _positive_int(concurrency):
        failures.add(E_CONFIG_INVALID,
                     f"concurrency {concurrency!r} not positive; "
                     f"falling back to {DEFAULT_CONCURRENCY}", stage="load")
        concurrency = DEFAULT_CONCURRENCY

    snap = raw.get("snapshot", {})
    if not isinstance(snap, dict):
        snap = {}
    snapshot_interval = snap.get("interval", DEFAULT_SNAPSHOT_INTERVAL)
    if not _positive_int(snapshot_interval):
        failures.add(E_CONFIG_INVALID,
                     f"snapshot interval {snapshot_interval!r} not positive; "
                     f"falling back to {DEFAULT_SNAPSHOT_INTERVAL}", stage="load")
        snapshot_interval = DEFAULT_SNAPSHOT_INTERVAL

    gmm_cfg = raw.get("gmm", {}) if isinstance(raw.get("gmm", {}), dict) else {}
    km_cfg = raw.get("kmeans", {}) if isinstance(raw.get("kmeans", {}), dict) else {}
    gmm_reg_covar = float(gmm_cfg.get("reg_covar", 1e-6))
    gmm_tol = float(gmm_cfg.get("tol", 1e-6))
    gmm_max_iter = int(gmm_cfg.get("max_iter", 100))
    kmeans_max_iter = int(km_cfg.get("max_iter", 100))

    return ExperimentConfig(
        name=name, seed=seed, sample_scale=sample_scale,
        preprocess_method=prep_method, preprocess_params=prep_params,
        methods=methods, metrics=metrics,
        global_budget=int(global_budget), candidate_quota=int(candidate_quota),
        concurrency=int(concurrency), snapshot_interval=int(snapshot_interval),
        gmm_reg_covar=gmm_reg_covar, gmm_tol=gmm_tol,
        kmeans_max_iter=kmeans_max_iter, gmm_max_iter=gmm_max_iter,
        raw=raw,
    )


def load_injections(path, failures: FailureLog) -> tuple[list, str]:
    """Returns (injection list, injection manifest hash)."""
    if path is None:
        return [], fingerprint("injections", [])
    raw = _load_yaml(Path(path), failures, "injection manifest")
    if raw is None:
        return [], fingerprint("injections", [])
    items = raw.get("injections", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        failures.add(E_CONFIG_INVALID, "injection manifest must be a list",
                     stage="load")
        items = []
    valid = []
    for idx, inj in enumerate(items):
        if not isinstance(inj, dict) or "type" not in inj:
            failures.add(E_CONFIG_INVALID,
                         f"injection #{idx} missing 'type'", stage="load")
            continue
        if inj["type"] not in ("replace_sample", "candidate_fault"):
            failures.add(E_CONFIG_INVALID,
                         f"unknown injection type '{inj['type']}'", stage="load")
            continue
        valid.append(inj)
    return valid, fingerprint("injections", valid)
