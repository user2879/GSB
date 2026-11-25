"""YAML loading and whole-schema validation.

Validation collects every issue in a stable order instead of aborting at the
first bad column or strategy field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import (
    ENCODING_WIDTHS,
    ERROR_BUDGET_INVALID,
    ERROR_DUPLICATE_FEATURE,
    ERROR_DUPLICATE_PAIR,
    ERROR_ORDER_LIMIT_INVALID,
    ERROR_PAIR_INVALID,
    ERROR_POLICY_INVALID,
    ERROR_TYPE_MISMATCH,
    ERROR_UNKNOWN_COLUMN,
    ERROR_WIDTH_INVALID,
    EXPANSION_MODES,
    FIXED_POINT_WIDTHS,
    OVER_BUDGET_ACTIONS,
    QUANTIZATION_LEVELS,
)
from .hashing import canonical_json, sha256_hex


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    cardinality: int | None = None
    categories: tuple[str, ...] = ()
    distribution: str = "uniform"
    low: float = 0.0
    high: float = 100.0
    mean: float = 50.0
    std: float = 15.0
    zipf_a: float = 1.25
    missing_rate: float = 0.0


@dataclass(frozen=True)
class Strategy:
    candidate_pairs: tuple[tuple[str, str], ...]
    order_limit: int
    cardinality_budget: int
    point_budget: int
    encoding_width: int
    fixed_point_width: int
    quantization_levels: int
    over_budget_action: str = "truncate"
    truncate_buckets: int | None = None
    max_workers: int = 4
    snapshot_commit_interval: int = 1
    expansion_mode: str = "universe"
    global_version: int = 1
    dictionary_preview_limit: int = 20
    enable_flip_analysis: bool = True

    @property
    def width_gear_name(self) -> str:
        return (
            f"e{self.encoding_width}"
            f"_f{self.fixed_point_width}"
            f"_q{self.quantization_levels}"
        )


@dataclass
class LoadedBundle:
    features_path: Path | None
    strategy_path: Path | None
    raw_features: dict[str, Any]
    raw_strategy: dict[str, Any]
    columns: tuple[ColumnSpec, ...]
    strategy: Strategy | None
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue["severity"] == "error" for issue in self.issues)

    def issues_by_code(self, code: str) -> list[dict[str, Any]]:
        return [issue for issue in self.issues if issue["code"] == code]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _issue(
    code: str,
    path: str,
    message: str,
    *,
    ref: Any = None,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "path": path,
        "message": message,
        "ref": ref,
    }


def _read_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML root must be a mapping: {target}")
    return dict(data)


def load_bundle(
    features_path: str | Path,
    strategy_path: str | Path | None = None,
) -> LoadedBundle:
    features = _read_yaml(features_path)
    strategy: dict[str, Any] = {}
    if strategy_path is not None:
        strategy = _read_yaml(strategy_path)
    else:
        strategy = dict(features.get("strategy") or {})
    return load_bundle_from_dicts(
        features,
        strategy,
        features_path=Path(features_path),
        strategy_path=Path(strategy_path) if strategy_path else None,
    )


def load_bundle_from_dicts(
    features: Mapping[str, Any],
    strategy: Mapping[str, Any] | None = None,
    *,
    features_path: Path | None = None,
    strategy_path: Path | None = None,
) -> LoadedBundle:
    raw_features = dict(features)
    raw_strategy = dict(strategy or raw_features.get("strategy") or {})
    issues: list[dict[str, Any]] = []
    columns = _parse_columns(raw_features, issues)
    resolved_strategy = _parse_strategy(raw_strategy, columns, issues)
    return LoadedBundle(
        features_path=features_path,
        strategy_path=strategy_path,
        raw_features=raw_features,
        raw_strategy=raw_strategy,
        columns=tuple(columns),
        strategy=resolved_strategy,
        issues=issues,
    )


def _parse_columns(
    raw_features: dict[str, Any], issues: list[dict[str, Any]]
) -> list[ColumnSpec]:
    raw_columns = raw_features.get("columns", [])
    if not isinstance(raw_columns, list):
        issues.append(
            _issue(
                "E_FEATURE_TABLE_INVALID",
                "columns",
                "columns must be a list of feature declarations",
            )
        )
        return []

    seen_names: set[str] = set()
    columns: list[ColumnSpec] = []
    for index, raw_column in enumerate(raw_columns):
        path = f"columns[{index}]"
        if not isinstance(raw_column, Mapping):
            issues.append(_issue("E_COLUMN_INVALID", path, "column must be a mapping"))
            continue
        name = raw_column.get("name")
        if not isinstance(name, str) or not name.strip():
            issues.append(_issue("E_COLUMN_INVALID", f"{path}.name", "name is required"))
            continue
        if name in seen_names:
            issues.append(
                _issue(
                    ERROR_DUPLICATE_FEATURE,
                    f"{path}.name",
                    f"feature name {name!r} is not unique",
                    ref=name,
                )
            )
            continue
        seen_names.add(name)

        dtype = raw_column.get("type")
        if dtype not in ("categorical", "numeric"):
            issues.append(
                _issue(
                    "E_COLUMN_INVALID",
                    f"{path}.type",
                    "type must be categorical or numeric",
                    ref=dtype,
                )
            )
            continue

        missing_rate = raw_column.get("missing_rate", 0.0)
        if not isinstance(missing_rate, (int, float)) or isinstance(
            missing_rate, bool
        ) or not 0.0 <= float(missing_rate) < 1.0:
            issues.append(
                _issue(
                    "E_COLUMN_INVALID",
                    f"{path}.missing_rate",
                    "missing_rate must be in [0, 1)",
                    ref=missing_rate,
                )
            )
            missing_rate = 0.0

        if dtype == "categorical":
            categories_raw = raw_column.get("categories")
            categories: tuple[str, ...] = ()
            cardinality = raw_column.get("cardinality")
            if categories_raw is not None:
                if not isinstance(categories_raw, list) or not categories_raw:
                    issues.append(
                        _issue(
                            "E_COLUMN_INVALID",
                            f"{path}.categories",
                            "categories must be a non-empty list",
                        )
                    )
                    continue
                categories = tuple(str(value) for value in categories_raw)
                cardinality = len(categories)
            else:
                if not _is_int(cardinality) or cardinality <= 0:
                    issues.append(
                        _issue(
                            "E_COLUMN_INVALID",
                            f"{path}.cardinality",
                            "categorical cardinality must be a positive integer",
                            ref=cardinality,
                        )
                    )
                    continue
            columns.append(
                ColumnSpec(
                    name=name,
                    dtype=dtype,
                    cardinality=int(cardinality),
                    categories=tuple(sorted(set(categories))) if categories else (),
                    zipf_a=float(raw_column.get("zipf_a", 1.25)),
                    distribution=str(raw_column.get("distribution", "uniform")),
                    missing_rate=float(missing_rate),
                )
            )
        else:
            low = raw_column.get("low", 0.0)
            high = raw_column.get("high", 100.0)
            if not isinstance(low, (int, float)) or not isinstance(
                high, (int, float)
            ) or float(high) <= float(low):
                issues.append(
                    _issue(
                        "E_COLUMN_INVALID",
                        path,
                        "numeric low/high must be numbers with high > low",
                    )
                )
                continue
            columns.append(
                ColumnSpec(
                    name=name,
                    dtype=dtype,
                    distribution=str(raw_column.get("distribution", "uniform")),
                    low=float(low),
                    high=float(high),
                    mean=float(raw_column.get("mean", (low + high) / 2.0)),
                    std=float(raw_column.get("std", (high - low) / 6.0)),
                    missing_rate=float(missing_rate),
                )
            )
    return columns


def _parse_strategy(
    raw: dict[str, Any],
    columns: list[ColumnSpec],
    issues: list[dict[str, Any]],
) -> Strategy | None:
    values: dict[str, Any] = {}
    scalar_ok = True

    order_limit = raw.get("order_limit", 2)
    if not _is_int(order_limit) or order_limit <= 0:
        issues.append(
            _issue(
                ERROR_ORDER_LIMIT_INVALID,
                "strategy.order_limit",
                "order_limit must be a positive integer",
                ref=order_limit,
            )
        )
        scalar_ok = False
        order_limit = 2
    values["order_limit"] = int(order_limit)

    cardinality_budget = raw.get("cardinality_budget", 64)
    point_budget = raw.get("point_budget", 1_000_000)
    for key, value in (
        ("cardinality_budget", cardinality_budget),
        ("point_budget", point_budget),
    ):
        if not _is_int(value) or value <= 0:
            issues.append(
                _issue(
                    ERROR_BUDGET_INVALID,
                    f"strategy.{key}",
                    f"{key} must be a positive integer",
                    ref=value,
                )
            )
            scalar_ok = False
            if key == "cardinality_budget":
                cardinality_budget = 64
            else:
                point_budget = 1_000_000
    values["cardinality_budget"] = int(cardinality_budget)
    values["point_budget"] = int(point_budget)

    encoding_width = raw.get("encoding_width", 16)
    fixed_point_width = raw.get("fixed_point_width", 16)
    quantization_levels = raw.get("quantization_levels", 16)
    if encoding_width not in ENCODING_WIDTHS:
        issues.append(
            _issue(
                ERROR_WIDTH_INVALID,
                "strategy.encoding_width",
                f"encoding_width must be one of {list(ENCODING_WIDTHS)}",
                ref=encoding_width,
            )
        )
        scalar_ok = False
        encoding_width = 16
    if fixed_point_width not in FIXED_POINT_WIDTHS:
        issues.append(
            _issue(
                ERROR_WIDTH_INVALID,
                "strategy.fixed_point_width",
                f"fixed_point_width must be one of {list(FIXED_POINT_WIDTHS)}",
                ref=fixed_point_width,
            )
        )
        scalar_ok = False
        fixed_point_width = 16
    if quantization_levels not in QUANTIZATION_LEVELS:
        issues.append(
            _issue(
                ERROR_WIDTH_INVALID,
                "strategy.quantization_levels",
                f"quantization_levels must be one of {list(QUANTIZATION_LEVELS)}",
                ref=quantization_levels,
            )
        )
        scalar_ok = False
        quantization_levels = 16
    values.update(
        encoding_width=int(encoding_width),
        fixed_point_width=int(fixed_point_width),
        quantization_levels=int(quantization_levels),
    )

    action = raw.get("over_budget_action", "truncate")
    if action not in OVER_BUDGET_ACTIONS:
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.over_budget_action",
                f"over_budget_action must be one of {list(OVER_BUDGET_ACTIONS)}",
                ref=action,
            )
        )
        scalar_ok = False
        action = "truncate"
    values["over_budget_action"] = action

    truncate_buckets = raw.get("truncate_buckets")
    if truncate_buckets is not None and (
        not _is_int(truncate_buckets) or truncate_buckets <= 0
    ):
        issues.append(
            _issue(
                ERROR_BUDGET_INVALID,
                "strategy.truncate_buckets",
                "truncate_buckets must be a positive integer when provided",
                ref=truncate_buckets,
            )
        )
        scalar_ok = False
        truncate_buckets = None
    values["truncate_buckets"] = (
        int(truncate_buckets) if truncate_buckets is not None else None
    )

    max_workers = raw.get("max_workers", 4)
    if not _is_int(max_workers) or max_workers <= 0:
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.max_workers",
                "max_workers must be a positive integer",
                ref=max_workers,
            )
        )
        scalar_ok = False
        max_workers = 4
    values["max_workers"] = int(max_workers)

    snapshot_interval = raw.get("snapshot_commit_interval", 1)
    if not _is_int(snapshot_interval) or snapshot_interval <= 0:
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.snapshot_commit_interval",
                "snapshot_commit_interval must be a positive integer",
                ref=snapshot_interval,
            )
        )
        scalar_ok = False
        snapshot_interval = 1
    values["snapshot_commit_interval"] = int(snapshot_interval)

    expansion_mode = raw.get("expansion_mode", "universe")
    if expansion_mode not in EXPANSION_MODES:
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.expansion_mode",
                f"expansion_mode must be one of {list(EXPANSION_MODES)}",
                ref=expansion_mode,
            )
        )
        scalar_ok = False
        expansion_mode = "universe"
    values["expansion_mode"] = expansion_mode

    global_version = raw.get("global_version", 1)
    if not _is_int(global_version):
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.global_version",
                "global_version must be an integer",
                ref=global_version,
            )
        )
        scalar_ok = False
        global_version = 1
    values["global_version"] = int(global_version)

    preview_limit = raw.get("dictionary_preview_limit", 20)
    if not _is_int(preview_limit) or preview_limit < 0:
        issues.append(
            _issue(
                ERROR_POLICY_INVALID,
                "strategy.dictionary_preview_limit",
                "dictionary_preview_limit must be a non-negative integer",
                ref=preview_limit,
            )
        )
        preview_limit = 20
    values["dictionary_preview_limit"] = int(preview_limit)
    values["enable_flip_analysis"] = bool(raw.get("enable_flip_analysis", True))

    pairs = _parse_candidate_pairs(raw.get("candidate_pairs", []), columns, issues)
    if not scalar_ok:
        # The object is still returned with safe fallbacks so callers can show
        # a complete failure list and build diagnostic examples if desired.
        pass
    return Strategy(candidate_pairs=pairs, **values)


def _parse_candidate_pairs(
    raw_pairs: Any,
    columns: list[ColumnSpec],
    issues: list[dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    if raw_pairs is None:
        raw_pairs = []
    if not isinstance(raw_pairs, list):
        issues.append(
            _issue(
                ERROR_PAIR_INVALID,
                "strategy.candidate_pairs",
                "candidate_pairs must be a list of two-column pairs",
            )
        )
        return ()

    column_types = {column.name: column.dtype for column in columns}
    normalized: dict[tuple[str, str], int] = {}
    for index, raw_pair in enumerate(raw_pairs):
        path = f"strategy.candidate_pairs[{index}]"
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            issues.append(
                _issue(
                    ERROR_PAIR_INVALID,
                    path,
                    "candidate pair must contain exactly two column names",
                    ref=raw_pair,
                )
            )
            continue
        left, right = raw_pair
        if not isinstance(left, str) or not isinstance(right, str):
            issues.append(
                _issue(
                    ERROR_PAIR_INVALID,
                    path,
                    "candidate pair entries must be strings",
                    ref=raw_pair,
                )
            )
            continue

        endpoint_failed = False
        for endpoint_index, name in enumerate((left, right)):
            endpoint_path = f"{path}[{endpoint_index}]"
            if name not in column_types:
                issues.append(
                    _issue(
                        ERROR_UNKNOWN_COLUMN,
                        endpoint_path,
                        f"candidate pair references unknown column {name!r}",
                        ref=name,
                    )
                )
                endpoint_failed = True
            elif column_types[name] != "categorical":
                issues.append(
                    _issue(
                        ERROR_TYPE_MISMATCH,
                        endpoint_path,
                        f"cross column {name!r} must be categorical",
                        ref=name,
                    )
                )
                endpoint_failed = True
        if endpoint_failed:
            continue

        if left == right:
            issue = _issue(
                ERROR_DUPLICATE_PAIR,
                path,
                f"self-cross {left!r} is degenerate and was deduplicated",
                ref=[left, right],
            )
            issue["reason"] = "self_cross"
            issues.append(issue)
            continue

        pair = tuple(sorted((left, right)))
        if pair in normalized:
            issue = _issue(
                ERROR_DUPLICATE_PAIR,
                path,
                "duplicate candidate pair was deduplicated",
                ref=list(raw_pair),
            )
            issue["reason"] = "duplicate_pair"
            issue["first_index"] = normalized[pair]
            issues.append(issue)
            continue
        normalized[pair] = index

    return tuple(sorted(normalized.keys()))


def policy_fingerprint(strategy: Strategy) -> str:
    """Stable hash of policy fields that affect adjudication/snapshots."""

    payload = {
        "candidate_pairs": [list(pair) for pair in strategy.candidate_pairs],
        "order_limit": strategy.order_limit,
        "cardinality_budget": strategy.cardinality_budget,
        "point_budget": strategy.point_budget,
        "encoding_width": strategy.encoding_width,
        "fixed_point_width": strategy.fixed_point_width,
        "quantization_levels": strategy.quantization_levels,
        "over_budget_action": strategy.over_budget_action,
        "truncate_buckets": strategy.truncate_buckets,
        "max_workers": strategy.max_workers,
        "snapshot_commit_interval": strategy.snapshot_commit_interval,
        "expansion_mode": strategy.expansion_mode,
        "global_version": strategy.global_version,
    }
    return sha256_hex("policy-v1", canonical_json(payload))


def width_gear_fingerprint(
    encoding_width: int,
    fixed_point_width: int,
    quantization_levels: int,
) -> str:
    return sha256_hex(
        "width-gear-v1",
        encoding_width,
        fixed_point_width,
        quantization_levels,
    )


def strategy_fingerprint(strategy: Strategy) -> str:
    return policy_fingerprint(strategy)
