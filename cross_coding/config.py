from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import (
    ALLOWED_CARDINALITY_STRATEGIES,
    ALLOWED_ENCODING_WIDTH_BITS,
    ALLOWED_FIXED_POINT_BITS,
    ALLOWED_QUANTIZATION_LEVELS,
    CODE_CONFIG_INVALID,
    DETAIL_FIXED_POINT_NOT_ALLOWED,
    DETAIL_REFERENCE_NOT_CATEGORICAL,
    DETAIL_UNKNOWN_COLUMN,
    DETAIL_VALUE_OUT_OF_RANGE,
    DETAIL_WIDTH_NOT_ALLOWED,
)
from .hashing import fingerprint
from .table import build_table, canonical_feature_hash, canonical_policy_hash


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    if not isinstance(doc, dict):
        raise ValueError("YAML root must be a mapping")
    return doc


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []

    def add(
        self,
        path: str,
        message: str,
        *,
        detail_code: str | None = None,
        value: Any = None,
    ) -> None:
        self.errors.append(
            {
                "code": CODE_CONFIG_INVALID,
                "detail_code": detail_code or CODE_CONFIG_INVALID,
                "stage": "config",
                "path": path,
                "message": message,
                "value": value,
            }
        )

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _table_spec(feature_doc: dict[str, Any]) -> dict[str, Any]:
    return feature_doc.get("table", feature_doc)


def validate_feature_doc(doc: dict[str, Any], report: ValidationReport) -> None:
    table = _table_spec(doc)
    if not isinstance(table, dict):
        report.add("feature_table", "feature table must be a mapping")
        return

    if not _is_int(table.get("seed")):
        report.add("feature_table.seed", "seed must be an integer")
    if not _is_int(table.get("row_count")) or table.get("row_count", 0) <= 0:
        report.add("feature_table.row_count", "row_count must be a positive integer")

    columns = table.get("columns")
    if not isinstance(columns, list) or not columns:
        report.add("feature_table.columns", "columns must be a non-empty list")
        return

    seen: set[str] = set()
    for index, column in enumerate(columns):
        base = f"feature_table.columns[{index}]"
        if not isinstance(column, dict):
            report.add(base, "column specification must be a mapping")
            continue
        name = column.get("name")
        if not isinstance(name, str) or not name:
            report.add(f"{base}.name", "column name must be a non-empty string", value=name)
        elif name in seen:
            report.add(f"{base}.name", f"duplicate feature name: {name}", value=name)
        else:
            seen.add(name)

        column_type = column.get("type")
        if column_type not in ("categorical", "numeric"):
            report.add(f"{base}.type", "type must be categorical or numeric", value=column_type)
            continue

        if column_type == "categorical":
            categories = column.get("categories")
            if not isinstance(categories, list) or not categories or not all(
                isinstance(item, (str, int)) and not isinstance(item, bool)
                for item in categories
            ):
                report.add(f"{base}.categories", "categorical columns need non-empty categories")
            elif len(set(map(str, categories))) != len(categories):
                report.add(f"{base}.categories", "categories must be unique")

        if column_type == "numeric":
            low = column.get("min", 0)
            high = column.get("max", 100)
            if not isinstance(low, (int, float)) or isinstance(low, bool):
                report.add(f"{base}.min", "min must be numeric", value=low)
            if not isinstance(high, (int, float)) or isinstance(high, bool):
                report.add(f"{base}.max", "max must be numeric", value=high)
            elif isinstance(low, (int, float)) and low > high:
                report.add(f"{base}.max", "max must be greater than or equal to min")

        values = column.get("values")
        if values is not None:
            row_count = table.get("row_count")
            if _is_int(row_count) and len(values) != row_count:
                report.add(
                    f"{base}.values",
                    "values length must equal row_count",
                    value=len(values),
                )
            else:
                allowed = set(column.get("categories", [])) if column_type == "categorical" else None
                low = column.get("min")
                high = column.get("max")
                for value_index, value in enumerate(values):
                    if value is None:
                        continue
                    if column_type == "categorical":
                        if allowed is not None and value not in allowed:
                            report.add(
                                f"{base}.values[{value_index}]",
                                f"categorical value {value!r} is outside declared categories",
                                detail_code=DETAIL_VALUE_OUT_OF_RANGE,
                                value=value,
                            )
                            break
                    else:
                        if (
                            isinstance(low, (int, float))
                            and isinstance(high, (int, float))
                            and not (low <= value <= high)
                        ):
                            report.add(
                                f"{base}.values[{value_index}]",
                                f"numeric value {value!r} is outside [{low}, {high}]",
                                detail_code=DETAIL_VALUE_OUT_OF_RANGE,
                                value=value,
                            )
                            break

        bitmap = column.get("missing_bitmap")
        if bitmap is not None:
            row_count = table.get("row_count")
            if not isinstance(bitmap, list) or not all(isinstance(x, bool) for x in bitmap):
                report.add(f"{base}.missing_bitmap", "missing_bitmap must be a boolean list")
            elif _is_int(row_count) and len(bitmap) != row_count:
                report.add(f"{base}.missing_bitmap", "missing_bitmap length must equal row_count")


def _default_policy() -> dict[str, Any]:
    return {
        "candidate_pairs": [],
        "max_order": 2,
        "cardinality_budget": 16,
        "over_cardinality_strategy": "truncate_hash_buckets",
        "encoding_width_bits": 8,
        "fixed_point_bits": 16,
        "quantization_levels": 16,
        "encoding_point_budget": 10_000_000,
        "max_workers": 4,
        "snapshot_commit_interval": 1,
    }


def normalize_policy(doc: dict[str, Any], feature_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = _default_policy()
    policy.update(doc or {})
    doc = doc or {}
    aliases = {
        "max_cross_order": "max_order",
        "cross_order_limit": "max_order",
        "order_limit": "max_order",
        "encoding_bit_width": "encoding_width_bits",
        "code_width_bits": "encoding_width_bits",
        "fixed_point_bit_width": "fixed_point_bits",
        "fixed_point_width_bits": "fixed_point_bits",
        "decimal_bits": "fixed_point_bits",
        "quantization_bins": "quantization_levels",
        "quantization_level": "quantization_levels",
        "thread_pool_size": "max_workers",
        "concurrency": "max_workers",
        "worker_count": "max_workers",
        "point_budget": "encoding_point_budget",
    }
    for alias, target in aliases.items():
        if alias in doc and target not in doc:
            policy[target] = doc[alias]
        policy.pop(alias, None)
    if "candidate_column_pairs" in doc and "candidate_pairs" not in doc:
        policy["candidate_pairs"] = doc["candidate_column_pairs"]
    policy.pop("candidate_column_pairs", None)
    if "point_budget" in (doc or {}) and "encoding_point_budget" not in (doc or {}):
        policy["encoding_point_budget"] = doc["point_budget"]

    raw_pairs = policy.get("candidate_pairs") or []
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for pair in raw_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            left, right = str(pair[0]), str(pair[1])
            normalized = tuple(sorted((left, right)))
            if normalized[0] != normalized[1] and normalized not in seen:
                seen.add(normalized)
                pairs.append(normalized)
    policy["candidate_pairs"] = [list(pair) for pair in pairs]
    policy["allowed_encoding_width_bits"] = list(ALLOWED_ENCODING_WIDTH_BITS)
    policy["allowed_fixed_point_bits"] = list(ALLOWED_FIXED_POINT_BITS)
    policy["allowed_quantization_levels"] = list(ALLOWED_QUANTIZATION_LEVELS)
    if feature_doc is not None:
        policy["feature_config_fingerprint"] = canonical_feature_hash(feature_doc)
    policy["policy_fingerprint"] = canonical_policy_hash(
        {k: v for k, v in policy.items() if k != "policy_fingerprint"}
    )
    return policy


def validate_policy_doc(
    doc: dict[str, Any],
    feature_doc: dict[str, Any],
    report: ValidationReport,
) -> dict[str, Any] | None:
    policy = normalize_policy(doc, feature_doc)
    table = _table_spec(feature_doc)
    columns = {c.get("name"): c for c in table.get("columns", []) if isinstance(c, dict)}

    raw_pairs = doc.get("candidate_pairs", []) if isinstance(doc, dict) else []
    if not isinstance(raw_pairs, list):
        report.add("cross_policy.candidate_pairs", "candidate_pairs must be a list")
    else:
        for index, pair in enumerate(raw_pairs):
            path = f"cross_policy.candidate_pairs[{index}]"
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                report.add(path, "candidate pair must contain exactly two column names", value=pair)
                continue
            for position, name in enumerate(pair):
                if not isinstance(name, str) or not name:
                    report.add(f"{path}[{position}]", "column name must be a non-empty string")
                elif name not in columns:
                    report.add(
                        f"{path}[{position}]",
                        f"candidate pair references unknown column: {name}",
                        detail_code=DETAIL_UNKNOWN_COLUMN,
                        value=name,
                    )
                elif columns[name].get("type") != "categorical":
                    report.add(
                        f"{path}[{position}]",
                        f"candidate pair column {name!r} must be categorical",
                        detail_code=DETAIL_REFERENCE_NOT_CATEGORICAL,
                        value=name,
                    )

    positive_int_fields = ("max_order", "max_workers", "snapshot_commit_interval")
    for field in positive_int_fields:
        value = policy.get(field)
        if not _is_int(value) or value <= 0:
            report.add(f"cross_policy.{field}", f"{field} must be a positive integer", value=value)

    for field in ("cardinality_budget", "encoding_point_budget"):
        value = policy.get(field)
        if not _is_int(value) or value < 0:
            report.add(f"cross_policy.{field}", f"{field} must be a non-negative integer", value=value)

    width = policy.get("encoding_width_bits")
    if width not in ALLOWED_ENCODING_WIDTH_BITS:
        report.add(
            "cross_policy.encoding_width_bits",
            f"encoding_width_bits must be one of {list(ALLOWED_ENCODING_WIDTH_BITS)}",
            detail_code=DETAIL_WIDTH_NOT_ALLOWED,
            value=width,
        )

    fixed_bits = policy.get("fixed_point_bits")
    if fixed_bits not in ALLOWED_FIXED_POINT_BITS:
        report.add(
            "cross_policy.fixed_point_bits",
            f"fixed_point_bits must be one of {list(ALLOWED_FIXED_POINT_BITS)}",
            detail_code=DETAIL_FIXED_POINT_NOT_ALLOWED,
            value=fixed_bits,
        )

    levels = policy.get("quantization_levels")
    if levels not in ALLOWED_QUANTIZATION_LEVELS:
        report.add(
            "cross_policy.quantization_levels",
            f"quantization_levels must be one of {list(ALLOWED_QUANTIZATION_LEVELS)}",
            value=levels,
        )

    strategy = policy.get("over_cardinality_strategy")
    if strategy not in ALLOWED_CARDINALITY_STRATEGIES:
        report.add(
            "cross_policy.over_cardinality_strategy",
            f"strategy must be one of {list(ALLOWED_CARDINALITY_STRATEGIES)}",
            value=strategy,
        )

    enumeration_mode = policy.get("combination_enumeration", "all_subsets")
    if enumeration_mode not in ("all_subsets", "pair_seeded", "connected_subsets"):
        report.add(
            "cross_policy.combination_enumeration",
            "combination_enumeration must be all_subsets, pair_seeded, or connected_subsets",
            value=enumeration_mode,
        )

    return policy if report.ok else None


def load_bundle_from_docs(
    feature_doc: dict[str, Any],
    policy_doc: dict[str, Any],
    *,
    feature_path: str | None = None,
    policy_path: str | None = None,
) -> tuple[dict[str, Any] | None, ValidationReport]:
    report = ValidationReport()
    validate_feature_doc(feature_doc, report)
    policy = validate_policy_doc(policy_doc or {}, feature_doc, report)
    if not report.ok:
        return None, report
    assert policy is not None
    table = build_table(feature_doc)
    bundle = {
        "feature_doc": feature_doc,
        "policy_doc": policy_doc or {},
        "policy": policy,
        "table": table,
        "feature_path": feature_path,
        "policy_path": policy_path,
        "config_fingerprint": fingerprint(
            "bundle",
            {
                "feature": canonical_feature_hash(feature_doc),
                "policy": policy["policy_fingerprint"],
                "table": table["table_fingerprint"],
            },
        ),
    }
    return bundle, report


def load_bundle(
    feature_path: str | Path,
    policy_path: str | Path,
) -> tuple[dict[str, Any] | None, ValidationReport]:
    report = ValidationReport()
    feature_doc: dict[str, Any] | None = None
    policy_doc: dict[str, Any] | None = None
    try:
        feature_doc = load_yaml_file(feature_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        report.add(
            "feature_yaml",
            f"cannot load feature YAML: {exc}",
            detail_code="E_YAML_PARSE_ERROR",
            value=str(feature_path),
        )
    try:
        policy_doc = load_yaml_file(policy_path)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        report.add(
            "policy_yaml",
            f"cannot load policy YAML: {exc}",
            detail_code="E_YAML_PARSE_ERROR",
            value=str(policy_path),
        )
    if feature_doc is None or policy_doc is None:
        return None, report
    return load_bundle_from_docs(
        feature_doc,
        policy_doc,
        feature_path=str(feature_path),
        policy_path=str(policy_path),
    )


def invalid_bundle_report(errors: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in errors:
        counts[item["code"]] = counts.get(item["code"], 0) + 1
        detail = item.get("detail_code")
        if detail and detail != item["code"]:
            counts[detail] = counts.get(detail, 0) + 1
    result = {
        "ok": False,
        "status": "config_invalid",
        "business_code_counts": counts,
        "failures": errors,
        "failure_list": errors,
        "combinations": [],
        "pending_queue": [],
    }
    result.update(extra)
    return result
