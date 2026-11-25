#!/usr/bin/env python3
"""Deterministic edit-sequence fuzz checks for incremental/full equality."""

from __future__ import annotations

import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from category_cross_diagnostic.comparison import deep_diff, normalize_report  # noqa: E402
from category_cross_diagnostic.config import load_bundle_from_dicts  # noqa: E402
from category_cross_diagnostic.engine import DiagnosticEngine  # noqa: E402
from category_cross_diagnostic.sample_data import generate_table  # noqa: E402


def base_dicts() -> tuple[dict[str, Any], dict[str, Any]]:
    return {
        "seed": 7711,
        "row_count": 96,
        "columns": [
            {"name": "a", "type": "categorical", "cardinality": 7},
            {"name": "b", "type": "categorical", "cardinality": 9},
            {"name": "c", "type": "categorical", "cardinality": 11},
            {"name": "d", "type": "categorical", "cardinality": 5},
            {"name": "x", "type": "numeric", "low": 0, "high": 10},
        ],
    }, {
        "expansion_mode": "clique",
        "candidate_pairs": [["a", "b"], ["a", "c"], ["b", "c"]],
        "order_limit": 3,
        "cardinality_budget": 40,
        "point_budget": 100000,
        "encoding_width": 12,
        "fixed_point_width": 16,
        "quantization_levels": 16,
        "over_budget_action": "truncate",
        "max_workers": 3,
        "snapshot_commit_interval": 2,
    }


def make_engine(
    features: dict[str, Any],
    strategy: dict[str, Any],
    state: Path,
    overrides: dict[str, list[Any]],
) -> DiagnosticEngine:
    bundle = load_bundle_from_dicts(features, strategy)
    if not bundle.ok:
        raise AssertionError(bundle.issues)
    table = generate_table(bundle)
    for name, values in overrides.items():
        table.replace_column(name, values)
    return DiagnosticEngine(bundle, table, state_dir=state)


def mutate(
    rng: random.Random,
    features: dict[str, Any],
    strategy: dict[str, Any],
    overrides: dict[str, list[Any]],
    edit_index: int,
) -> str:
    categorical = [c["name"] for c in features["columns"] if c["type"] == "categorical"]
    operation = rng.choice(
        [
            "add_pair",
            "remove_pair",
            "order",
            "cardinality_budget",
            "point_budget",
            "encoding_width",
            "width_gear",
            "replace_column",
        ]
    )
    if operation == "add_pair":
        left, right = rng.sample(categorical, 2)
        pair = sorted((left, right))
        if pair not in [sorted(p) for p in strategy["candidate_pairs"]]:
            strategy["candidate_pairs"].append(pair)
        return f"{edit_index}:add_pair:{pair}"
    if operation == "remove_pair" and strategy["candidate_pairs"]:
        removed = strategy["candidate_pairs"].pop(rng.randrange(len(strategy["candidate_pairs"])))
        return f"{edit_index}:remove_pair:{removed}"
    if operation == "order":
        strategy["order_limit"] = rng.randint(1, 4)
        return f"{edit_index}:order:{strategy['order_limit']}"
    if operation == "cardinality_budget":
        strategy["cardinality_budget"] = rng.choice([1, 4, 16, 40, 120, 500])
        return f"{edit_index}:cardinality_budget:{strategy['cardinality_budget']}"
    if operation == "point_budget":
        strategy["point_budget"] = rng.choice([96, 1000, 5000, 100000])
        return f"{edit_index}:point_budget:{strategy['point_budget']}"
    if operation == "encoding_width":
        strategy["encoding_width"] = rng.choice([8, 12, 16, 32])
        return f"{edit_index}:encoding_width:{strategy['encoding_width']}"
    if operation == "width_gear":
        width, levels = rng.choice([(8, 8), (16, 16), (32, 32)])
        strategy["fixed_point_width"] = width
        strategy["quantization_levels"] = levels
        return f"{edit_index}:width_gear:{width}/{levels}"

    name = rng.choice(categorical)
    cardinality = next(c["cardinality"] for c in features["columns"] if c["name"] == name)
    overrides[name] = [
        None if (edit_index * 17 + row) % 29 == 0 else f"{name}__v{(row + edit_index) % cardinality}"
        for row in range(features["row_count"])
    ]
    return f"{edit_index}:replace_column:{name}"


def run_case(case_seed: int, root: Path, edit_count: int) -> dict[str, Any]:
    rng = random.Random(case_seed)
    features, strategy = base_dicts()
    state = root / f"case-{case_seed}"
    overrides: dict[str, list[Any]] = {}
    make_engine(features, strategy, state, overrides).run()
    edit_log: list[str] = []
    for edit_index in range(edit_count):
        edit_log.append(mutate(rng, features, strategy, overrides, edit_index))
        incremental = make_engine(features, strategy, state, overrides).run()

    fresh = make_engine(
        features, strategy, root / f"fresh-{case_seed}", overrides
    ).run()
    diffs = deep_diff(normalize_report(incremental), normalize_report(fresh))
    return {
        "seed": case_seed,
        "edits": edit_log,
        "match": not diffs,
        "differences": diffs[:5],
        "combination_count": incremental["combination_count"],
        "pending_count": len(incremental["pending"]),
    }


def main() -> int:
    output = ROOT / "artifacts" / "fuzz_sequence_validation.json"
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as temporary:
        root = Path(temporary)
        cases = [run_case(9100 + offset, root, 8) for offset in range(12)]
    result = {
        "schema": "edit-sequence-fuzz/v1",
        "case_count": len(cases),
        "all_match": all(case["match"] for case in cases),
        "failed_cases": [case for case in cases if not case["match"]],
        "cases": cases,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_match": result["all_match"], "case_count": len(cases)}, indent=2))
    if result["all_match"]:
        print(f"wrote {output.relative_to(ROOT)}")
        return 0
    print(json.dumps(result["failed_cases"][:2], ensure_ascii=False, indent=2))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
