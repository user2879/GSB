"""Configuration and feature metadata loading with strict validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .errors import ConfigInvalidError
from .models import GLOBAL_PRECISION_LABELS, SUPPORTED_VERSIONS, FeatureMeta, MergeConfig


REQUIRED_KEYS = (
    "feature_whitelist",
    "scaling_method",
    "missing_fill",
    "conflict_priority",
    "numeric_tolerance",
    "max_workers",
    "batch_size",
)
SCALING_METHODS = ("raw", "percentile", "zscore")
CONFLICT_MODES = ("source_version", "precision_label", "abs_value")
FILL_STRATEGIES = ("zero", "mean", "median", "constant")


def load_merge_config(path: str | Path) -> MergeConfig:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigInvalidError(f"merge config not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigInvalidError(f"merge config is not valid JSON: {exc}") from exc
    return parse_merge_config(data)


def parse_merge_config(data: dict) -> MergeConfig:
    if not isinstance(data, dict):
        raise ConfigInvalidError("merge config root must be a JSON object")

    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ConfigInvalidError(
            f"missing required config field(s): {', '.join(missing)}"
        )

    whitelist = data["feature_whitelist"]
    if not isinstance(whitelist, list) or not whitelist or not all(
        isinstance(item, str) and item for item in whitelist
    ):
        raise ConfigInvalidError(
            "feature_whitelist must be a non-empty list of feature names"
        )

    scaling_method = data["scaling_method"]
    if scaling_method not in SCALING_METHODS:
        raise ConfigInvalidError(
            f"scaling_method must be one of {SCALING_METHODS}, got {scaling_method!r}"
        )

    fill = data["missing_fill"]
    if not isinstance(fill, dict) or "strategy" not in fill:
        raise ConfigInvalidError("missing_fill must be an object containing strategy")
    if fill["strategy"] not in FILL_STRATEGIES:
        raise ConfigInvalidError(
            f"missing_fill.strategy must be one of {FILL_STRATEGIES}"
        )
    if fill["strategy"] == "constant" and not isinstance(
        fill.get("value"), (int, float)
    ):
        raise ConfigInvalidError(
            "missing_fill.value numeric constant is required for strategy=constant"
        )

    conflict_priority = data["conflict_priority"]
    if conflict_priority not in CONFLICT_MODES:
        raise ConfigInvalidError(
            f"conflict_priority must be one of {CONFLICT_MODES}"
        )

    tolerance = data["numeric_tolerance"]
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
        raise ConfigInvalidError("numeric_tolerance must be a non-negative number")

    max_workers = data["max_workers"]
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        raise ConfigInvalidError("max_workers must be a positive integer")

    batch_size = data["batch_size"]
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ConfigInvalidError("batch_size must be a positive integer")

    source_order = data.get("source_order", list(SUPPORTED_VERSIONS))
    if (
        not isinstance(source_order, list)
        or len(source_order) != len(set(source_order))
        or any(item not in SUPPORTED_VERSIONS for item in source_order)
        or set(source_order) != set(SUPPORTED_VERSIONS)
    ):
        raise ConfigInvalidError(
            f"source_order must be a permutation of {list(SUPPORTED_VERSIONS)}"
        )

    precision_rank = data.get("precision_rank", list(GLOBAL_PRECISION_LABELS))
    if (
        not isinstance(precision_rank, list)
        or not precision_rank
        or any(item not in GLOBAL_PRECISION_LABELS for item in precision_rank)
        or len(set(precision_rank)) != len(precision_rank)
    ):
        raise ConfigInvalidError(
            f"precision_rank must be unique labels from {list(GLOBAL_PRECISION_LABELS)}"
        )

    percentile_tie_rule = data.get("percentile_tie_rule", "average_rank")
    if percentile_tie_rule not in ("average_rank", "min_rank", "max_rank"):
        raise ConfigInvalidError(
            "percentile_tie_rule must be average_rank, min_rank or max_rank"
        )

    return MergeConfig(
        feature_whitelist=list(whitelist),
        scaling_method=scaling_method,
        missing_fill=dict(fill),
        conflict_priority=conflict_priority,
        source_order=list(source_order),
        precision_rank=list(precision_rank),
        numeric_tolerance=float(tolerance),
        max_workers=max_workers,
        batch_size=batch_size,
        percentile_tie_rule=percentile_tie_rule,
    )


def load_feature_metadata(path: str | Path) -> dict[str, FeatureMeta]:
    target = Path(path)
    rows: dict[str, FeatureMeta] = {}
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"feature", "semantic_type", "allowed_precisions"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ConfigInvalidError(
                f"metadata table must contain columns {sorted(required)}"
            )
        for line_no, row in enumerate(reader, start=2):
            meta = FeatureMeta.from_row(row)
            if not meta.feature:
                raise ConfigInvalidError(
                    f"metadata row {line_no} has empty feature name"
                )
            unknown = [
                label
                for label in meta.allowed_precisions
                if label not in GLOBAL_PRECISION_LABELS
            ]
            if unknown:
                raise ConfigInvalidError(
                    f"metadata row {line_no} allows unknown precision label(s): {unknown}"
                )
            rows[meta.feature] = meta
    if not rows:
        raise ConfigInvalidError("metadata table is empty")
    return rows
