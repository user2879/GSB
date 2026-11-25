from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

from .config import load_bundle_from_docs
from .pipeline import comparison_view, run_pipeline


def apply_edit(bundle: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    feature_doc = copy.deepcopy(bundle["feature_doc"])
    policy_doc = copy.deepcopy(bundle["policy_doc"])
    edit_type = edit["type"]

    if edit_type == "add_pair":
        pairs = policy_doc.setdefault("candidate_pairs", [])
        pair = list(edit["pair"])
        normalized = sorted(pair)
        if normalized not in [sorted(item) for item in pairs]:
            pairs.append(pair)
    elif edit_type == "remove_pair":
        target = sorted(edit["pair"])
        policy_doc["candidate_pairs"] = [
            pair for pair in policy_doc.get("candidate_pairs", []) if sorted(pair) != target
        ]
    elif edit_type == "set_max_order":
        policy_doc["max_order"] = int(edit["value"])
    elif edit_type == "set_cardinality_budget":
        policy_doc["cardinality_budget"] = int(edit["value"])
    elif edit_type == "set_point_budget":
        policy_doc["encoding_point_budget"] = int(edit["value"])
        policy_doc.pop("point_budget", None)
    elif edit_type == "set_encoding_width":
        policy_doc["encoding_width_bits"] = int(edit["value"])
    elif edit_type == "set_fixed_point_bits":
        policy_doc["fixed_point_bits"] = int(edit["value"])
    elif edit_type == "set_quantization_levels":
        policy_doc["quantization_levels"] = int(edit["value"])
    elif edit_type == "set_max_workers":
        policy_doc["max_workers"] = int(edit["value"])
    elif edit_type == "replace_column":
        _replace_column_values(feature_doc, bundle, edit)
    else:
        raise ValueError(f"unknown edit type: {edit_type}")

    new_bundle, errors = load_bundle_from_docs(
        feature_doc,
        policy_doc,
        feature_path=bundle.get("feature_path"),
        policy_path=bundle.get("policy_path"),
    )
    if not errors.ok:
        raise ValueError(f"edit produced invalid configuration: {[e['message'] for e in errors.errors]}")
    assert new_bundle is not None
    new_bundle["feature_doc"] = feature_doc
    new_bundle["policy_doc"] = policy_doc
    return new_bundle


def _replace_column_values(feature_doc: dict[str, Any], old_bundle: dict[str, Any], edit: dict[str, Any]) -> None:
    column_name = edit["column"]
    table_spec = feature_doc.get("table", feature_doc)
    column = next(column for column in table_spec["columns"] if column["name"] == column_name)
    row_count = int(table_spec["row_count"])

    if "values" in edit:
        values = list(edit["values"])
        if len(values) != row_count:
            raise ValueError("replacement values length must match row_count")
        column["values"] = values
        column.pop("data_seed", None)
        return

    seed = int(edit.get("replacement_seed", edit.get("seed", 91001)))
    rng = random.Random(seed)
    old_table = old_bundle["table"]["columns"][column_name]
    bitmap = list(old_table["missing_bitmap"])
    values: list[Any] = []
    if column["type"] == "categorical":
        categories = column["categories"]
        for missing in bitmap:
            values.append(None if missing else rng.choice(categories))
    else:
        low = float(column.get("min", 0))
        high = float(column.get("max", 100))
        integer_values = bool(column.get("integer", False))
        for missing in bitmap:
            if missing:
                values.append(None)
            elif integer_values:
                values.append(rng.randint(int(low), int(high)))
            else:
                values.append(round(rng.uniform(low, high), 6))
    column["values"] = values
    column["missing_bitmap"] = bitmap
    column["replacement_seed"] = seed


def run_incremental_sequence(
    initial_bundle: dict[str, Any],
    edits: list[dict[str, Any]],
    state_dir: str | Path,
) -> list[dict[str, Any]]:
    bundle = initial_bundle
    reports = []
    for cursor, edit in enumerate(edits, start=1):
        bundle = apply_edit(bundle, edit)
        reports.append(
            run_pipeline(bundle, str(state_dir), command_cursor=cursor, parallel=True)
        )
    return reports


def run_blank_full_at_each_step(
    initial_bundle: dict[str, Any],
    edits: list[dict[str, Any]],
    state_root: str | Path,
) -> list[dict[str, Any]]:
    """Rebuild from an empty store after every edit and return canonical views."""
    root = Path(state_root)
    bundle = initial_bundle
    views = []
    for cursor, edit in enumerate(edits, start=1):
        bundle = apply_edit(bundle, edit)
        blank_dir = root / f"blank-step-{cursor}"
        report = run_pipeline(bundle, str(blank_dir), command_cursor=cursor, parallel=False)
        views.append(comparison_view(report))
    return views


def assert_views_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    if left != right:
        import json

        raise AssertionError(
            "comparison views differ:\n"
            + json.dumps({"incremental": left, "blank_full": right}, ensure_ascii=False, indent=2)[:12000]
        )
