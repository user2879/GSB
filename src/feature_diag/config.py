"""Experiment YAML loading, scenario validation, and configuration fingerprints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .data_spec import CATEGORY_KINDS, DataSpec, load_data_spec
from .errors import E_CONFIG_INVALID, ConfigLoadError, Failure
from .fingerprint import canonical_json, fingerprint, sha256_text

PRECISION_TIERS = ("float32", "float64")
REDUCTION_PATHS = ("pca", "random_projection", "variance_filter")
DEFAULT_TOLERANCES = {
    "cross_encoding": 1e-6,
    "standardization": 1e-5,
    "projection": 1e-4,
    "final_metric": 1e-5,
}
STAGE_ORDER = ("cross_encoding", "standardization", "projection", "final_metric")


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass
class ScenarioConfig:
    scenario_id: str
    sample_sizes: tuple[int, ...]
    cross_orders: tuple[int, ...]
    candidate_columns: tuple[str, ...]
    cardinality_budget: int
    overflow_policy: str
    precision_tiers: tuple[str, ...]
    reductions: dict[str, dict[str, Any]]
    decision: dict[str, Any]
    target_column: str
    tolerances: dict[str, float]
    max_workers: int
    seed: int
    data_spec_name: str
    fingerprint: str = ""

    def resolved(self) -> dict[str, Any]:
        return {
            "id": self.scenario_id,
            "sample_sizes": list(self.sample_sizes),
            "cross_orders": list(self.cross_orders),
            "candidate_columns": list(self.candidate_columns),
            "cardinality_budget": self.cardinality_budget,
            "overflow_policy": self.overflow_policy,
            "precision_tiers": list(self.precision_tiers),
            "reductions": self.reductions,
            "decision": self.decision,
            "target_column": self.target_column,
            "tolerances": self.tolerances,
            "seed": self.seed,
            "data_spec": self.data_spec_name,
        }


@dataclass
class ExperimentConfig:
    source_path: str
    version: int
    schema_version: int
    seed: int
    data_spec: DataSpec
    tolerances: dict[str, float]
    boundary: dict[str, float]
    cache_dir: Path
    snapshot_dir: Path
    max_workers: int
    scenarios: list[ScenarioConfig]
    failures: list[Failure]
    config_hash: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def precision_fingerprints(self) -> dict[str, str]:
        payload = {
            "engine_schema": self.schema_version,
            "tiers": list(PRECISION_TIERS),
            "stage_order": list(STAGE_ORDER),
            "tolerances": self.tolerances,
        }
        return {
            tier: fingerprint({"precision": tier, **payload})
            for tier in PRECISION_TIERS
        }


def _resolve_path(base: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigLoadError(f"{field_name} must be a path string", field=field_name)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"cannot read config {path}: {exc}", field="config") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"invalid experiment YAML: {exc}", field="config") from exc
    if not isinstance(raw, dict):
        raise ConfigLoadError("experiment config root must be a mapping", field="config")
    return raw


def _canonical_reduction(
    raw_reductions: Any,
    scenario_id: str,
    failures: list[Failure],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_reductions, dict):
        failures.append(_failure(scenario_id, "reductions", "invalid_reductions", "reductions must be a mapping"))
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in REDUCTION_PATHS:
        raw_path = raw_reductions.get(path, {})
        if not isinstance(raw_path, dict):
            failures.append(_failure(scenario_id, f"reductions.{path}", "invalid_reduction", "reduction config must be a mapping"))
            continue
        config = dict(raw_path)
        components = config.get("components", 4)
        if not _is_positive_int(components):
            failures.append(
                _failure(
                    scenario_id,
                    f"reductions.{path}.components",
                    "invalid_reduction",
                    "components must be a positive integer",
                )
            )
            continue
        candidates = config.get("candidates")
        if candidates is None:
            candidates = list(range(1, int(components) + 1))
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(not _is_positive_int(item) for item in candidates)
        ):
            failures.append(
                _failure(
                    scenario_id,
                    f"reductions.{path}.candidates",
                    "invalid_reduction",
                    "candidates must be a non-empty list of positive integers",
                )
            )
            continue
        config["components"] = int(components)
        config["candidates"] = sorted({int(item) for item in candidates})
        if path == "variance_filter":
            minimum = config.get("min_variance", 0.0)
            if not _is_number(minimum) or float(minimum) < 0.0:
                failures.append(
                    _failure(
                        scenario_id,
                        "reductions.variance_filter.min_variance",
                        "invalid_reduction",
                        "min_variance must be a non-negative number",
                    )
                )
                continue
            config["min_variance"] = float(minimum)
        result[path] = config
    return result


def _failure(scenario_id: str, field_name: str, reason: str, message: str, details: dict[str, Any] | None = None) -> Failure:
    return Failure(
        code=E_CONFIG_INVALID,
        scope=scenario_id,
        field=field_name,
        reason=reason,
        message=message,
        details=details or {},
    )


def _load_scenario(
    raw: Any,
    *,
    index: int,
    data_spec: DataSpec,
    root_seed: int,
    root_tolerances: dict[str, float],
    root_workers: int,
) -> tuple[ScenarioConfig | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(raw, dict):
        return None, [_failure(f"scenario[{index}]", "scenarios", "invalid_scenario", "scenario must be a mapping")]
    scenario_id = raw.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        scenario_id = f"scenario[{index}]"
        failures.append(_failure(scenario_id, "id", "invalid_scenario", "scenario id must be a non-empty string"))

    raw_sizes = raw.get("sample_sizes", [raw.get("sample_size")])
    if not isinstance(raw_sizes, list):
        raw_sizes = [raw_sizes]
    sample_sizes: list[int] = []
    for value in raw_sizes:
        if not _is_positive_int(value):
            failures.append(
                _failure(scenario_id, "sample_sizes", "invalid_sample_size", "sample sizes must be positive integers", {"value": value})
            )
        else:
            sample_sizes.append(int(value))
    sample_sizes = sorted(set(sample_sizes))
    if not sample_sizes:
        failures.append(_failure(scenario_id, "sample_sizes", "invalid_sample_size", "at least one sample size is required"))

    raw_orders = raw.get("cross_orders", [])
    cross_orders: list[int] = []
    if not isinstance(raw_orders, list) or not raw_orders:
        failures.append(_failure(scenario_id, "cross_orders", "invalid_cross_order", "cross_orders must be a non-empty list"))
    else:
        for value in raw_orders:
            if not _is_positive_int(value):
                failures.append(
                    _failure(
                        scenario_id,
                        "cross_orders",
                        "non_positive_cross_order",
                        "cross order must be a positive integer",
                        {"value": value},
                    )
                )
            else:
                cross_orders.append(int(value))
    cross_orders = sorted(set(cross_orders))

    raw_candidates = raw.get("candidate_columns")
    candidate_columns: list[str] = []
    if not isinstance(raw_candidates, list) or not raw_candidates:
        failures.append(
            _failure(scenario_id, "candidate_columns", "unknown_column", "candidate_columns must be a non-empty list")
        )
    else:
        for name in raw_candidates:
            if not isinstance(name, str) or name not in data_spec.column_map:
                failures.append(
                    _failure(
                        scenario_id,
                        "candidate_columns",
                        "unknown_column",
                        f"unknown candidate column: {name!r}",
                        {"column": name},
                    )
                )
                continue
            column = data_spec.column_map[name]
            if column.kind not in CATEGORY_KINDS:
                failures.append(
                    _failure(
                        scenario_id,
                        "candidate_columns",
                        "type_mismatch",
                        f"candidate column {name!r} has type {column.kind!r}, expected a categorical type",
                        {"column": name, "actual_type": column.kind},
                    )
                )
                continue
            candidate_columns.append(name)
    candidate_columns = sorted(set(candidate_columns))

    budget = raw.get("cardinality_budget", 0)
    if not _is_positive_int(budget):
        failures.append(
            _failure(
                scenario_id,
                "cardinality_budget",
                "non_positive_cardinality_budget",
                "cardinality budget must be a positive integer",
                {"value": budget},
            )
        )
        budget = 0

    policy = raw.get("overflow_policy", "hash_bucket")
    if policy not in {"hash_bucket", "skip"}:
        failures.append(
            _failure(
                scenario_id,
                "overflow_policy",
                "invalid_overflow_policy",
                "overflow_policy must be hash_bucket or skip",
                {"value": policy},
            )
        )

    raw_precisions = raw.get("precision_tiers", list(PRECISION_TIERS))
    valid_precisions = True
    if not isinstance(raw_precisions, list) or not raw_precisions:
        valid_precisions = False
        failures.append(
            _failure(scenario_id, "precision_tiers", "invalid_precision_tier", "precision_tiers must be a non-empty list")
        )
    else:
        for tier in raw_precisions:
            if tier not in PRECISION_TIERS:
                valid_precisions = False
                failures.append(
                    _failure(
                        scenario_id,
                        "precision_tiers",
                        "invalid_precision_tier",
                        f"unsupported precision tier: {tier!r}",
                        {"value": tier, "allowed": list(PRECISION_TIERS)},
                    )
                )
        if valid_precisions and set(raw_precisions) != set(PRECISION_TIERS):
            valid_precisions = False
            failures.append(
                _failure(
                    scenario_id,
                    "precision_tiers",
                    "invalid_precision_tier",
                    "both float32 and float64 tiers are required",
                    {"value": raw_precisions},
                )
            )
    precision_tiers = tuple(PRECISION_TIERS) if valid_precisions else tuple()

    target_column = raw.get("target_column", data_spec.target_column)
    if not isinstance(target_column, str) or target_column not in data_spec.column_map:
        failures.append(
            _failure(
                scenario_id,
                "target_column",
                "target_missing",
                f"target column is missing: {target_column!r}",
                {"column": target_column},
            )
        )
        target_column = ""

    reductions = _canonical_reduction(raw.get("reductions", {}), scenario_id, failures)
    if reductions and set(reductions) != set(REDUCTION_PATHS):
        failures.append(
            _failure(
                scenario_id,
                "reductions",
                "invalid_reduction",
                "pca, random_projection, and variance_filter reductions are all required",
            )
        )

    tolerances = dict(root_tolerances)
    raw_tolerances = raw.get("tolerances", {})
    if raw_tolerances is not None:
        if not isinstance(raw_tolerances, dict):
            failures.append(_failure(scenario_id, "tolerances", "invalid_tolerance", "tolerances must be a mapping"))
        else:
            for key, value in raw_tolerances.items():
                if key not in STAGE_ORDER or not _is_number(value) or float(value) < 0.0:
                    failures.append(
                        _failure(
                            scenario_id,
                            f"tolerances.{key}",
                            "invalid_tolerance",
                            "tolerance keys must name a stage and use non-negative numbers",
                        )
                    )
                else:
                    tolerances[key] = float(value)

    decision = dict(raw.get("decision", {}))
    metric = decision.get("metric", "first_coordinate")
    if metric not in {"first_coordinate", "l2_norm", "mean_absolute"}:
        failures.append(_failure(scenario_id, "decision.metric", "invalid_decision", "unsupported downstream metric"))
    threshold_mode = decision.get("threshold_mode", "fixed")
    if threshold_mode not in {"fixed", "reference_median"}:
        failures.append(
            _failure(scenario_id, "decision.threshold_mode", "invalid_decision", "threshold_mode must be fixed or reference_median")
        )
    threshold = decision.get("threshold", 0.0)
    if not _is_number(threshold):
        failures.append(_failure(scenario_id, "decision.threshold", "invalid_decision", "threshold must be numeric"))
    else:
        decision["threshold"] = float(threshold)
    decision.setdefault("metric", metric)
    decision.setdefault("threshold_mode", threshold_mode)

    scenario_workers = raw.get("max_workers", root_workers)
    if not _is_positive_int(scenario_workers):
        failures.append(_failure(scenario_id, "max_workers", "invalid_max_workers", "max_workers must be a positive integer"))
        scenario_workers = 1
    effective_workers = min(int(scenario_workers), int(root_workers))

    seed = raw.get("seed", root_seed)
    if isinstance(seed, bool) or not isinstance(seed, int):
        failures.append(_failure(scenario_id, "seed", "invalid_seed", "seed must be an integer"))
        seed = root_seed

    if failures:
        return None, failures

    scenario = ScenarioConfig(
        scenario_id=scenario_id,
        sample_sizes=tuple(sample_sizes),
        cross_orders=tuple(cross_orders),
        candidate_columns=tuple(candidate_columns),
        cardinality_budget=int(budget),
        overflow_policy=policy,
        precision_tiers=precision_tiers,
        reductions=reductions,
        decision=decision,
        target_column=target_column,
        tolerances=tolerances,
        max_workers=effective_workers,
        seed=int(seed),
        data_spec_name=data_spec.name,
    )
    scenario.fingerprint = fingerprint({"scenario": scenario.resolved(), "data_spec": data_spec.content_hash})
    return scenario, []


def load_experiment(config_path: str | Path) -> ExperimentConfig:
    source = Path(config_path).resolve()
    base = source.parent
    raw = _load_yaml(source)

    version = raw.get("version", 1)
    if not _is_positive_int(version):
        raise ConfigLoadError("version must be a positive integer", field="version")
    engine = raw.get("engine", {})
    if engine is None:
        engine = {}
    if not isinstance(engine, dict):
        raise ConfigLoadError("engine must be a mapping", field="engine")
    schema_version = engine.get("schema_version", 1)
    if not _is_positive_int(schema_version):
        raise ConfigLoadError("engine.schema_version must be a positive integer", field="engine.schema_version")

    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigLoadError("seed must be an integer", field="seed")

    data_spec_path = _resolve_path(base, raw.get("data_spec"), "data_spec")
    data_spec = load_data_spec(data_spec_path)

    tolerances = dict(DEFAULT_TOLERANCES)
    raw_tolerances = raw.get("tolerances", {})
    if not isinstance(raw_tolerances, dict):
        raise ConfigLoadError("tolerances must be a mapping", field="tolerances")
    for key, value in raw_tolerances.items():
        if key not in STAGE_ORDER or not _is_number(value) or float(value) < 0.0:
            raise ConfigLoadError(f"invalid tolerance: {key}", field="tolerances")
        tolerances[key] = float(value)

    boundary = {"neighborhood": 0.0, "tie_tolerance": 1e-9}
    raw_boundary = raw.get("boundary", {})
    if not isinstance(raw_boundary, dict):
        raise ConfigLoadError("boundary must be a mapping", field="boundary")
    for key in boundary:
        if key in raw_boundary:
            value = raw_boundary[key]
            if not _is_number(value) or float(value) < 0.0:
                raise ConfigLoadError(f"boundary.{key} must be a non-negative number", field=f"boundary.{key}")
            boundary[key] = float(value)

    root_workers = raw.get("max_workers", 1)
    if not _is_positive_int(root_workers):
        raise ConfigLoadError("max_workers must be a positive integer", field="max_workers")

    cache_dir = _resolve_path(base, raw.get("cache_dir", "cache"), "cache_dir")
    snapshot_dir = _resolve_path(base, raw.get("snapshot_dir", "snapshot"), "snapshot_dir")

    scenarios: list[ScenarioConfig] = []
    failures: list[Failure] = []
    raw_scenarios = raw.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ConfigLoadError("scenarios must be a non-empty list", field="scenarios")
    seen_ids: set[str] = set()
    for index, raw_scenario in enumerate(raw_scenarios):
        scenario, scenario_failures = _load_scenario(
            raw_scenario,
            index=index,
            data_spec=data_spec,
            root_seed=int(seed),
            root_tolerances=tolerances,
            root_workers=int(root_workers),
        )
        if scenario is not None:
            if scenario.scenario_id in seen_ids:
                failures.append(
                    _failure(
                        scenario.scenario_id,
                        "id",
                        "invalid_scenario",
                        f"duplicate scenario id: {scenario.scenario_id}",
                    )
                )
            else:
                seen_ids.add(scenario.scenario_id)
                scenarios.append(scenario)
        failures.extend(scenario_failures)
    failures.sort(key=lambda item: (item.scope, item.field, item.reason, canonical_json(item.details)))
    scenarios.sort(key=lambda item: item.scenario_id)

    analysis_payload = {
        "version": int(version),
        "engine_schema": int(schema_version),
        "seed": int(seed),
        "data_spec_hash": data_spec.content_hash,
        "tolerances": tolerances,
        "boundary": boundary,
        "scenarios": [scenario.resolved() for scenario in scenarios],
        "failures": [failure.as_dict() for failure in failures],
    }
    config_hash = sha256_text(canonical_json(analysis_payload))

    return ExperimentConfig(
        source_path=str(source),
        version=int(version),
        schema_version=int(schema_version),
        seed=int(seed),
        data_spec=data_spec,
        tolerances=tolerances,
        boundary=boundary,
        cache_dir=cache_dir,
        snapshot_dir=snapshot_dir,
        max_workers=int(root_workers),
        scenarios=scenarios,
        failures=failures,
        config_hash=config_hash,
        raw=raw,
    )
