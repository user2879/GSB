"""Configuration loading and fail-soft validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import mini_yaml
from .util import fingerprint

ENGINE_REVISION = "feature-pipeline-v1.1.0"

BUSINESS_CODES: dict[str, str] = {
    "E_CONFIG": "configuration document is missing or malformed",
    "E_UNKNOWN_COLUMN": "a referenced column is not present in the column specification",
    "E_TYPE_MISMATCH": "a column or reference has an incompatible type",
    "E_CROSS_ORDER": "crossing order must be positive and feasible",
    "E_CARDINALITY_BUDGET": "cardinality budget and hash-bucket settings must be positive",
    "E_QUANTIZATION_PROFILE": "quantization tier or fixed-point profile is invalid",
    "E_PRECISION_TIER": "both single and double precision tiers are required",
    "E_REDUCTION_DIM": "reduction dimensions must be positive integers",
    "E_SAMPLE": "sample data or sample manifest is missing, stale, or corrupt",
    "E_DATA": "sample or replacement column data is malformed",
    "E_TARGET": "the configured target column is unavailable",
}


def failure(code: str, item: str, message: str, severity: str = "error") -> dict[str, Any]:
    if code not in BUSINESS_CODES:
        raise ValueError(f"unknown business code: {code}")
    return {
        "code": code,
        "item": str(item),
        "message": str(message),
        "severity": severity,
        "description": BUSINESS_CODES[code],
    }


def normalize_failures(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {
        (item["code"], item["item"], item["message"], item["severity"]): item
        for item in items
    }
    ordered = sorted(
        unique.values(),
        key=lambda item: (item["code"], item["item"], item["message"], item["severity"]),
    )
    return [{**item, "seq": index + 1} for index, item in enumerate(ordered)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _load_columns_spec(path: Path, failures: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = mini_yaml.load(path)
    except Exception as exc:  # validation deliberately aggregates parse failures
        failures.append(failure("E_CONFIG", "columns", f"cannot parse {path.name}: {exc}"))
        return {"seed": 0, "rows": 0, "columns": []}
    if not isinstance(payload, dict):
        failures.append(failure("E_CONFIG", "columns", "column specification root must be a mapping"))
        return {"seed": 0, "rows": 0, "columns": []}

    seed = payload.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        failures.append(failure("E_CONFIG", "columns.seed", "seed must be a non-negative integer"))
        seed = 0
    rows = payload.get("rows")
    if not _is_positive_int(rows):
        failures.append(failure("E_CONFIG", "columns.rows", "rows must be a positive integer"))
        rows = 0
    columns = payload.get("columns")
    if not isinstance(columns, list):
        failures.append(failure("E_CONFIG", "columns.columns", "columns must be a list"))
        columns = []
    return {"seed": seed, "rows": rows, "columns": columns, "raw": payload}


def _validate_descriptor(
    descriptor: Any,
    index: int,
    names_seen: set[str],
    failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(descriptor, dict):
        failures.append(failure("E_TYPE_MISMATCH", f"columns[{index}]", "column descriptor must be a mapping"))
        return None
    name = descriptor.get("name")
    if not isinstance(name, str) or not name:
        failures.append(failure("E_TYPE_MISMATCH", f"columns[{index}].name", "column name must be a non-empty string"))
        return None
    if name in names_seen:
        failures.append(failure("E_TYPE_MISMATCH", name, "duplicate column name"))
        return None
    names_seen.add(name)
    column_type = descriptor.get("type")
    if column_type not in {"category", "numeric"}:
        failures.append(
            failure("E_TYPE_MISMATCH", name, f"unsupported column type {column_type!r}; expected category or numeric")
        )
        return None
    missing_rate = descriptor.get("missing_rate", 0.0)
    if not _is_number(missing_rate) or not 0.0 <= float(missing_rate) < 1.0:
        failures.append(failure("E_TYPE_MISMATCH", name, "missing_rate must be a number in [0, 1)"))
        return None
    if column_type == "category":
        values = descriptor.get("values")
        if not isinstance(values, list) or not values:
            failures.append(failure("E_TYPE_MISMATCH", name, "category columns require a non-empty values list"))
            return None
        if any(value is None or isinstance(value, (dict, list)) for value in values):
            failures.append(failure("E_TYPE_MISMATCH", name, "category values must be scalar non-null values"))
            return None
        if len({str(value) for value in values}) != len(values):
            failures.append(failure("E_TYPE_MISMATCH", name, "category values must be unique"))
            return None
    else:
        minimum = descriptor.get("min", 0.0)
        maximum = descriptor.get("max", 1.0)
        if not _is_number(minimum) or not _is_number(maximum) or float(minimum) > float(maximum):
            failures.append(failure("E_TYPE_MISMATCH", name, "numeric min/max must be finite and ordered"))
            return None
        descriptor = {**descriptor, "min": float(minimum), "max": float(maximum)}
        if float(minimum) == float(maximum):
            descriptor = {**descriptor, "max": float(maximum) + 1.0}
    return {**descriptor, "missing_rate": float(missing_rate)}


@dataclass
class PipelineConfig:
    source_path: Path
    root: Path
    raw: dict[str, Any]
    columns_raw: dict[str, Any]
    descriptors: list[dict[str, Any]]
    seed: int
    rows: int
    config_hash: str
    generation_config_hash: str
    failures: list[dict[str, Any]]
    paths: dict[str, Path]

    @property
    def column_by_name(self) -> dict[str, dict[str, Any]]:
        return {descriptor["name"]: descriptor for descriptor in self.descriptors}

    @property
    def valid_whitelist(self) -> list[str]:
        whitelist = self.raw["crossing"].get("whitelist", [])
        columns = self.column_by_name
        return sorted(
            {
                name
                for name in whitelist
                if isinstance(name, str)
                and name in columns
                and columns[name].get("type") == "category"
            }
        )

    @property
    def max_workers(self) -> int:
        return int(self.raw["runtime"]["max_workers"])

    @property
    def temp_dir(self) -> Path:
        return self.paths["temp_dir"]

    def precision_fingerprint(self, tier: str) -> str:
        precision = self.raw["precision"]
        quant = precision["quantization"]
        profile_name = str(quant["tier"])
        profile = quant["profiles"][profile_name]
        return fingerprint(
            {
                "engine_revision": ENGINE_REVISION,
                "tier": tier,
                "dtype": "float32" if tier == "single" else "float64",
                "quantization_tier": profile_name,
                "quantization_enabled": bool(profile.get("enabled", False)),
                "fractional_bits": int(profile.get("fractional_bits", 0)),
            }
        )

    def quantization_profile(self) -> dict[str, Any]:
        quantization = self.raw["precision"]["quantization"]
        return quantization["profiles"][str(quantization["tier"])]

    def stage_tolerance(self) -> float:
        return float(self.raw["precision"]["stage_tolerance"])


def _default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "seed": 0,
        "data_config": "configs/columns.yaml",
        "paths": {
            "sample": "data/sample.json",
            "sample_manifest": "data/sample_manifest.json",
            "overrides": "data/overrides",
            "cache": "cache",
            "report": "reports/diagnostic_report.json",
        },
        "crossing": {
            "order": 2,
            "whitelist": [],
            "cardinality_budget": 16,
            "overflow_strategy": "hash_bucket",
            "hash_buckets": 8,
            "hash_seed": 4177,
        },
        "standardization": {"method": "zscore", "epsilon": 1e-12},
        "reduction": {
            "pca": {"dimensions": 4},
            "random_projection": {"dimensions": 4, "seed_salt": "rp-v1"},
            "variance": {"dimensions": 4},
        },
        "precision": {
            "tiers": ["single", "double"],
            "stage_tolerance": 1e-5,
            "quantization": {
                "tier": "lossless",
                "profiles": {
                    "lossless": {"enabled": False, "fractional_bits": 0},
                    "fine": {"enabled": True, "fractional_bits": 12},
                    "coarse": {"enabled": True, "fractional_bits": 4},
                },
            },
        },
        "metrics": {"decision_threshold": 0.0, "target_column": "target"},
        "runtime": {"max_workers": 4, "temp_dir": ".tmp"},
    }


def _deep_merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_path(root: Path, value: Any, label: str, failures: list[dict[str, Any]]) -> Path:
    if not isinstance(value, str) or not value:
        failures.append(failure("E_CONFIG", label, "path must be a non-empty string"))
        return root / f".invalid-{label.replace('.', '-')}"
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_crossing(
    raw: dict[str, Any],
    descriptors: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    crossing = raw["crossing"]
    order = crossing.get("order")
    if not _is_positive_int(order):
        failures.append(failure("E_CROSS_ORDER", "crossing.order", "crossing order must be a positive integer"))
    budget = crossing.get("cardinality_budget")
    if not _is_positive_int(budget):
        failures.append(
            failure("E_CARDINALITY_BUDGET", "crossing.cardinality_budget", "cardinality budget must be a positive integer")
        )
    hash_buckets = crossing.get("hash_buckets")
    if not _is_positive_int(hash_buckets):
        failures.append(
            failure("E_CARDINALITY_BUDGET", "crossing.hash_buckets", "hash bucket count must be a positive integer")
        )
    strategy = crossing.get("overflow_strategy")
    if strategy not in {"hash_bucket", "skip"}:
        failures.append(
            failure("E_TYPE_MISMATCH", "crossing.overflow_strategy", "overflow_strategy must be hash_bucket or skip")
        )
    hash_seed = crossing.get("hash_seed")
    if not isinstance(hash_seed, int) or isinstance(hash_seed, bool) or hash_seed < 0:
        failures.append(failure("E_TYPE_MISMATCH", "crossing.hash_seed", "hash_seed must be a non-negative integer"))
    whitelist = crossing.get("whitelist")
    if not isinstance(whitelist, list) or any(not isinstance(name, str) for name in whitelist):
        failures.append(failure("E_TYPE_MISMATCH", "crossing.whitelist", "whitelist must be a list of column names"))
        whitelist = []
    column_map = {descriptor["name"]: descriptor for descriptor in descriptors}
    valid: list[str] = []
    for name in whitelist:
        descriptor = column_map.get(name)
        if descriptor is None:
            failures.append(failure("E_UNKNOWN_COLUMN", name, "whitelist column is not defined in columns.yaml"))
        elif descriptor.get("type") != "category":
            failures.append(failure("E_TYPE_MISMATCH", name, "crossing whitelist may contain only category columns"))
        else:
            valid.append(name)
    valid = sorted(set(valid))
    if _is_positive_int(order) and order > len(valid):
        failures.append(
            failure(
                "E_CROSS_ORDER",
                "crossing.order",
                f"crossing order {order} exceeds the {len(valid)} usable category columns",
            )
        )


def _validate_reduction(raw: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    reduction = raw["reduction"]
    for path_name in ("pca", "random_projection", "variance"):
        settings = reduction.get(path_name)
        if not isinstance(settings, dict):
            failures.append(failure("E_CONFIG", f"reduction.{path_name}", "path settings must be a mapping"))
            continue
        dimensions = settings.get("dimensions")
        if not _is_positive_int(dimensions):
            failures.append(
                failure("E_REDUCTION_DIM", f"reduction.{path_name}.dimensions", "dimensions must be a positive integer")
            )
    seed_salt = reduction.get("random_projection", {}).get("seed_salt", "")
    if not isinstance(seed_salt, str):
        failures.append(failure("E_CONFIG", "reduction.random_projection.seed_salt", "seed_salt must be a string"))


def _validate_standardization(raw: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    settings = raw["standardization"]
    if settings.get("method") != "zscore":
        failures.append(failure("E_CONFIG", "standardization.method", "only zscore is supported"))
    epsilon = settings.get("epsilon")
    if not _is_number(epsilon) or float(epsilon) < 0:
        failures.append(
            failure("E_CONFIG", "standardization.epsilon", "epsilon must be a non-negative number")
        )


def _validate_precision(raw: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    precision = raw["precision"]
    tiers = precision.get("tiers")
    if isinstance(tiers, list) and len(tiers) == 2 and set(tiers) == {"single", "double"}:
        precision["tiers"] = ["single", "double"]
    else:
        failures.append(
            failure("E_PRECISION_TIER", "precision.tiers", "tiers must be exactly [single, double]")
        )
        precision["tiers"] = ["single", "double"]
    tolerance = precision.get("stage_tolerance")
    if not _is_number(tolerance) or float(tolerance) < 0:
        failures.append(
            failure("E_CONFIG", "precision.stage_tolerance", "stage_tolerance must be a non-negative number")
        )
        precision["stage_tolerance"] = 1e-5
    quantization = precision.get("quantization")
    if not isinstance(quantization, dict):
        failures.append(failure("E_QUANTIZATION_PROFILE", "precision.quantization", "quantization must be a mapping"))
        return
    profiles = quantization.get("profiles")
    tier_name = quantization.get("tier")
    if not isinstance(profiles, dict) or not profiles:
        failures.append(
            failure("E_QUANTIZATION_PROFILE", "precision.quantization.profiles", "profiles must be a non-empty mapping")
        )
        return
    if tier_name not in profiles:
        failures.append(
            failure("E_QUANTIZATION_PROFILE", "precision.quantization.tier", f"unknown quantization tier {tier_name!r}")
        )
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            failures.append(
                failure("E_QUANTIZATION_PROFILE", f"precision.quantization.profiles.{name}", "profile must be a mapping")
            )
            continue
        enabled = profile.get("enabled", False)
        bits = profile.get("fractional_bits", 0)
        if not isinstance(enabled, bool) or not isinstance(bits, int) or isinstance(bits, bool) or bits < 0:
            failures.append(
                failure(
                    "E_QUANTIZATION_PROFILE",
                    f"precision.quantization.profiles.{name}",
                    "enabled must be boolean and fractional_bits a non-negative integer",
                )
            )
        elif enabled and bits > 23:
            failures.append(
                failure(
                    "E_QUANTIZATION_PROFILE",
                    f"precision.quantization.profiles.{name}.fractional_bits",
                    "single precision supports at most 23 fractional fixed-point bits",
                )
            )


def load_pipeline_config(config_path: str | Path, *, root: str | Path | None = None) -> PipelineConfig:
    source_path = Path(config_path).resolve()
    workspace_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    failures: list[dict[str, Any]] = []
    try:
        supplied = mini_yaml.load(source_path)
    except Exception as exc:
        failures.append(failure("E_CONFIG", "pipeline", str(exc)))
        supplied = {}
    if not isinstance(supplied, dict):
        failures.append(failure("E_CONFIG", "pipeline", "configuration root must be a mapping"))
        supplied = {}
    defaults = _default_config()
    raw = _deep_merge(defaults, supplied)
    for section in (
        "paths",
        "crossing",
        "standardization",
        "reduction",
        "precision",
        "metrics",
        "runtime",
    ):
        if not isinstance(raw.get(section), dict):
            failures.append(failure("E_CONFIG", section, f"{section} must be a mapping"))
            raw[section] = defaults[section]
    for section, keys in {
        "precision": ("quantization",),
        "runtime": (),
        "crossing": (),
    }.items():
        for key in keys:
            if not isinstance(raw[section].get(key), dict):
                failures.append(failure("E_CONFIG", f"{section}.{key}", f"{section}.{key} must be a mapping"))
                raw[section][key] = defaults[section][key]

    seed = raw.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        failures.append(failure("E_CONFIG", "seed", "seed must be a non-negative integer"))
        raw["seed"] = 0

    paths: dict[str, Path] = {}
    for key in ("sample", "sample_manifest", "overrides", "cache", "report"):
        paths[key] = _resolve_path(workspace_root, raw["paths"].get(key), f"paths.{key}", failures)
    paths["temp_dir"] = _resolve_path(workspace_root, raw["runtime"].get("temp_dir"), "runtime.temp_dir", failures)

    data_config_path = _resolve_path(workspace_root, raw.get("data_config"), "data_config", failures)
    columns_raw = _load_columns_spec(data_config_path, failures)
    descriptor_failures: list[dict[str, Any]] = []
    names_seen: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    for index, descriptor in enumerate(columns_raw.get("columns", [])):
        validated = _validate_descriptor(descriptor, index, names_seen, descriptor_failures)
        if validated is not None:
            descriptors.append(validated)
    failures.extend(descriptor_failures)

    if not descriptors:
        failures.append(failure("E_TYPE_MISMATCH", "columns", "no valid column descriptors were loaded"))

    _validate_crossing(raw, descriptors, failures)
    _validate_standardization(raw, failures)
    _validate_reduction(raw, failures)
    _validate_precision(raw, failures)

    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        failures.append(failure("E_CONFIG", "metrics", "metrics must be a mapping"))
        raw["metrics"] = {"decision_threshold": 0.0, "target_column": "target"}
    elif not isinstance(metrics.get("target_column"), str) or not metrics.get("target_column"):
        failures.append(failure("E_CONFIG", "metrics.target_column", "target_column must be a non-empty string"))
    if not _is_number(metrics.get("decision_threshold")):
        failures.append(
            failure("E_CONFIG", "metrics.decision_threshold", "decision_threshold must be a finite number")
        )
        metrics["decision_threshold"] = 0.0

    runtime = raw.get("runtime")
    if not isinstance(runtime, dict) or not _is_positive_int(runtime.get("max_workers")):
        failures.append(failure("E_CONFIG", "runtime.max_workers", "max_workers must be a positive integer"))
        if not isinstance(raw.get("runtime"), dict):
            raw["runtime"] = dict(defaults["runtime"])
        raw["runtime"]["max_workers"] = 1

    rows = columns_raw.get("rows", 0)
    columns_seed = columns_raw.get("seed", 0)
    generation_payload = {
        "seed": columns_seed,
        "rows": rows,
        "descriptors": descriptors,
    }
    generation_config_hash = fingerprint(generation_payload)
    business_payload = {
        "engine_revision": ENGINE_REVISION,
        "seed": raw["seed"],
        "generation_config_hash": generation_config_hash,
        "crossing": raw["crossing"],
        "standardization": raw["standardization"],
        "reduction": raw["reduction"],
        "precision": raw["precision"],
        "metrics": raw["metrics"],
    }
    config_hash = fingerprint(business_payload)

    return PipelineConfig(
        source_path=source_path,
        root=workspace_root,
        raw=raw,
        columns_raw=columns_raw,
        descriptors=descriptors,
        seed=int(columns_seed),
        rows=int(rows),
        config_hash=config_hash,
        generation_config_hash=generation_config_hash,
        failures=normalize_failures(failures),
        paths=paths,
    )


def config_for_public_report(config: PipelineConfig) -> dict[str, Any]:
    """Return path-independent business configuration for reports/fingerprints."""

    return {
        "source": config.source_path.name,
        "seed": config.seed,
        "rows": config.rows,
        "columns": config.descriptors,
        "crossing": config.raw["crossing"],
        "standardization": config.raw["standardization"],
        "reduction": config.raw["reduction"],
        "precision": config.raw["precision"],
        "metrics": config.raw["metrics"],
        "runtime": {"max_workers": config.max_workers},
    }
