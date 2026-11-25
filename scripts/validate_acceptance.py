#!/usr/bin/env python3
"""Run local deterministic acceptance checks and write a JSON result."""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from category_cross_diagnostic.comparison import deep_diff, normalize_report  # noqa: E402
from category_cross_diagnostic.config import (  # noqa: E402
    load_bundle_from_dicts,
    width_gear_fingerprint,
)
from category_cross_diagnostic.engine import DiagnosticEngine  # noqa: E402
from category_cross_diagnostic.encoding import encode_combo  # noqa: E402
from category_cross_diagnostic.enumeration import enumerate_combinations  # noqa: E402
from category_cross_diagnostic.sample_data import generate_table  # noqa: E402


def load_example_dicts() -> tuple[dict[str, Any], dict[str, Any]]:
    features = yaml.safe_load((ROOT / "examples/feature_table.yaml").read_text())
    strategy = yaml.safe_load((ROOT / "examples/strategy.yaml").read_text())
    return features, strategy


def build_engine(
    features: dict[str, Any],
    strategy: dict[str, Any],
    state_dir: Path,
    *,
    serial: bool = False,
    replace_membership: bool = False,
    stop_event: Any = None,
) -> DiagnosticEngine:
    bundle = load_bundle_from_dicts(features, strategy)
    if not bundle.ok:
        raise AssertionError(bundle.issues)
    table = generate_table(bundle)
    if replace_membership:
        rng = np.random.default_rng(990280)
        categories = [f"membership__c{index:04d}" for index in range(6)]
        values = [categories[int(index)] for index in rng.integers(0, 6, size=table.row_count)]
        for index in range(0, table.row_count, 37):
            values[index] = None
        table.replace_column("membership", values)
    return DiagnosticEngine(
        bundle,
        table,
        state_dir=state_dir,
        serial=serial,
        stop_event=stop_event,
    )


def apply_edit(step: int, features: dict[str, Any], strategy: dict[str, Any]) -> tuple[str, bool]:
    replace_membership = False
    if step == 1:
        strategy["candidate_pairs"].append(["city", "browser"])
        label = "add_pair_city_browser"
    elif step == 2:
        strategy["candidate_pairs"] = [
            pair for pair in strategy["candidate_pairs"] if pair != ["device", "membership"]
        ]
        label = "remove_pair_device_membership"
    elif step == 3:
        strategy["order_limit"] = 2
        label = "lower_order_limit_to_2"
    elif step == 4:
        strategy["order_limit"] = 4
        label = "raise_order_limit_to_4"
    elif step == 5:
        strategy["cardinality_budget"] = 220
        label = "change_cardinality_budget_to_220"
    elif step == 6:
        strategy["point_budget"] = 260000
        label = "change_point_budget_to_260000"
    elif step == 7:
        strategy["encoding_width"] = 8
        label = "switch_encoding_width_to_8"
    elif step == 8:
        strategy["fixed_point_width"] = 8
        strategy["quantization_levels"] = 8
        label = "switch_fixed_point_gear_to_8_8"
    elif step == 9:
        strategy["encoding_width"] = 16
        strategy["fixed_point_width"] = 16
        strategy["quantization_levels"] = 16
        label = "restore_width_gear_16_16_16"
    elif step == 10:
        replace_membership = True
        label = "replace_single_column_membership"
    else:
        raise ValueError(step)
    return label, replace_membership


def run_edit_sequence(root: Path) -> dict[str, Any]:
    state_dir = root / "incremental_state"
    features, strategy = load_example_dicts()
    timeline: list[dict[str, Any]] = []
    last_report: dict[str, Any] | None = None
    replace_membership = False

    engine = build_engine(features, strategy, state_dir)
    initial = engine.run()
    timeline.append(_step_summary("initial", initial))

    for step in range(1, 11):
        label, replacement = apply_edit(step, features, strategy)
        replace_membership = replace_membership or replacement
        engine = build_engine(
            features,
            strategy,
            state_dir,
            replace_membership=replace_membership,
        )
        report = engine.run()
        last_report = report
        timeline.append(_step_summary(label, report))

    assert last_report is not None
    expected_affected = {
        item["combo_id"]
        for item in last_report["combinations"]
        if "membership" in item["columns"]
    }
    impact = _check_final_column_invalidation(last_report, expected_affected)
    return {
        "final_report": last_report,
        "final_features": features,
        "final_strategy": strategy,
        "timeline": timeline,
        "impact": impact,
    }


def run_twin_sequence(root: Path) -> dict[str, Any]:
    twin_root = root / "twin_state"
    features, strategy = load_example_dicts()
    replace_membership = False
    report = build_engine(features, strategy, twin_root).run()
    for step in range(1, 11):
        _label, replacement = apply_edit(step, features, strategy)
        replace_membership = replace_membership or replacement
        report = build_engine(
            features,
            strategy,
            twin_root,
            replace_membership=replace_membership,
        ).run()
    return report


def validate_errors() -> dict[str, Any]:
    features = {
        "seed": 7,
        "row_count": 20,
        "columns": [
            {"name": "a", "type": "categorical", "cardinality": 3},
            {"name": "b", "type": "categorical", "cardinality": 4},
            {"name": "num", "type": "numeric", "low": 0, "high": 1},
            {"name": "dup", "type": "categorical", "cardinality": 2},
            {"name": "dup", "type": "categorical", "cardinality": 2},
        ],
    }
    strategy = {
        "candidate_pairs": [
            ["ghost", "a"],
            ["num", "a"],
            ["a", "a"],
            ["a", "b"],
            ["b", "a"],
        ],
        "order_limit": 0,
        "cardinality_budget": -1,
        "point_budget": 0,
        "encoding_width": 7,
        "fixed_point_width": 3,
        "quantization_levels": 5,
        "max_workers": 2,
    }
    first = load_bundle_from_dicts(features, strategy)
    second = load_bundle_from_dicts(copy.deepcopy(features), copy.deepcopy(strategy))
    codes = {issue["code"] for issue in first.issues}
    required = {
        "E_UNKNOWN_COLUMN",
        "E_TYPE_MISMATCH",
        "E_ORDER_LIMIT_INVALID",
        "E_BUDGET_INVALID",
        "E_WIDTH_INVALID",
        "E_DUPLICATE_PAIR",
    }
    order_equal = [issue["code"] for issue in first.issues] == [
        issue["code"] for issue in second.issues
    ]
    return {
        "required_codes_present": sorted(required & codes),
        "missing_codes": sorted(required - codes),
        "all_required_present": required <= codes,
        "failure_order_deterministic": order_equal,
        "parsed_column_count": len(first.columns),
        "issue_count": len(first.issues),
    }


def validate_enumeration_and_hashes() -> dict[str, Any]:
    features, strategy = load_example_dicts()
    bundle = load_bundle_from_dicts(features, strategy)
    table = generate_table(bundle)
    combos = enumerate_combinations(table, bundle.strategy)
    ids = [combo.combo_id for combo in combos]
    expected_first_11 = [
        "cross__channel",
        "cross__city",
        "cross__device",
        "cross__membership",
        "cross__channel__city",
        "cross__channel__device",
        "cross__city__device",
        "cross__city__membership",
        "cross__device__membership",
        "cross__channel__city__device",
        "cross__city__device__membership",
    ]
    deterministic = enumerate_combinations(table, bundle.strategy)
    fingerprints_equal = [c.base_fingerprint for c in combos] == [
        c.base_fingerprint for c in deterministic
    ]

    triple = next(c for c in combos if c.combo_id == "cross__channel__city__device")
    wide = encode_combo(
        triple, table, encoding_width=16, fixed_point_width=16, quantization_levels=16
    )
    narrow = encode_combo(
        triple, table, encoding_width=8, fixed_point_width=16, quantization_levels=16
    )
    active_wide = triple.width_fingerprint(width_gear_fingerprint(16, 16, 16))
    active_narrow = triple.width_fingerprint(width_gear_fingerprint(8, 16, 16))
    return {
        "initial_order_matches": ids == expected_first_11,
        "actual_order": ids,
        "fingerprints_deterministic": fingerprints_equal,
        "width_switch": {
            "combo_fingerprint_changed": active_wide != active_narrow,
            "dictionary_digest_changed": wide.dictionary_digest != narrow.dictionary_digest,
            "code_sample_changed": wide.first_row_codes[:16] != narrow.first_row_codes[:16],
            "unique_counts": [narrow.unique_code_count, wide.unique_code_count],
            "collision_counts": [narrow.collision_count, wide.collision_count],
            "unique_count_changed": narrow.unique_code_count != wide.unique_code_count,
            "collision_count_changed": narrow.collision_count != wide.collision_count,
        },
    }


def validate_dual_budget(root: Path) -> dict[str, Any]:
    features, strategy = load_example_dicts()
    strategy["point_budget"] = 90000
    state = root / "dual_budget"
    report = build_engine(features, strategy, state).run()
    combo = next(
        item
        for item in report["combinations"]
        if item["combo_id"] == "cross__channel__city__device"
    )
    decision = combo["budget_decision"]
    pending_ids = [item["combo_id"] for item in report["pending"]]
    return {
        "joint_trigger_recorded": bool(decision["joint_constraint_triggered"]),
        "decision_order_constraints": [step["constraint"] for step in decision["decision_sequence"]],
        "dominant_constraint": decision["dominant_constraint"],
        "terminal_constraint": decision["terminal_constraint"],
        "boundary_combo": combo["combo_id"],
        "point_status": decision["point"]["status"],
        "pending_ids": pending_ids,
        "pending_order_correct": pending_ids == ["cross__city__device__membership"],
        "pending_code": report["pending"][0]["code"] if report["pending"] else None,
    }


def validate_snapshot_recovery(root: Path) -> dict[str, Any]:
    features, strategy = load_example_dicts()
    state = root / "recovery_state"
    event = None
    bundle = load_bundle_from_dicts(features, strategy)
    table = generate_table(bundle)
    engine = DiagnosticEngine(
        bundle, table, state_dir=state, serial=True
    )

    original_commit = engine._commit_snapshot  # noqa: SLF001

    def commit_and_stop(*args: Any, **kwargs: Any) -> Any:
        result = original_commit(*args, **kwargs)
        if kwargs.get("status") == "checkpoint":
            engine.request_stop()
        return result

    engine._commit_snapshot = commit_and_stop  # type: ignore[method-assign]  # noqa: SLF001
    interrupted = engine.run()
    completed_after_interrupt = interrupted["snapshot"]["latest_manifest"][
        "completed_combo_count"
    ]

    recovered = DiagnosticEngine(
        load_bundle_from_dicts(features, strategy),
        generate_table(load_bundle_from_dicts(features, strategy)),
        state_dir=state,
        serial=True,
    ).run()

    fresh_state = root / "recovery_fresh"
    fresh = DiagnosticEngine(
        load_bundle_from_dicts(features, strategy),
        generate_table(load_bundle_from_dicts(features, strategy)),
        state_dir=fresh_state,
        serial=True,
    ).run()
    diffs = deep_diff(
        normalize_report(recovered),
        normalize_report(fresh),
        path="recovered_vs_fresh",
    )
    return {
        "interrupted_status": interrupted["status"],
        "completed_after_interrupt": completed_after_interrupt,
        "recovery_decision": recovered["snapshot"]["recovery_decision"],
        "recovered_combo_count": recovered["adjudicated_count"],
        "fresh_combo_count": fresh["adjudicated_count"],
        "matches_fresh_after_recovery": not diffs,
        "first_differences": diffs[:5],
    }


def _check_final_column_invalidation(
    report: dict[str, Any], expected_affected: set[str]
) -> dict[str, Any]:
    recomputed = {
        entry["combo_id"]
        for entry in report["cache_ledger"]
        if entry["status"] == "recomputed"
    }
    reused = {
        entry["combo_id"]
        for entry in report["cache_ledger"]
        if entry["status"] == "reused"
    }
    all_combos = {item["combo_id"] for item in report["combinations"]}
    affected_stages = {
        (entry["combo_id"], entry["stage"])
        for entry in report["cache_ledger"]
        if entry["status"] == "recomputed"
    }
    expected_stages = {
        (combo_id, stage)
        for combo_id in expected_affected
        for stage in (
            "combination_enumeration",
            "encoding_dictionary_summary",
            "cardinality_diagnosis",
            "budget_decision",
        )
    }
    return {
        "expected_affected": sorted(expected_affected),
        "actual_recomputed_combo_ids": sorted(recomputed),
        "actual_reused_combo_ids": sorted(reused),
        "affected_set_exact": recomputed == expected_affected,
        "all_four_stages_recomputed": affected_stages == expected_stages,
        "unaffected_all_reused": reused == all_combos - expected_affected,
        "reused_entries": report["cache_summary"]["reused_count"],
        "recomputed_entries": report["cache_summary"]["recomputed_count"],
    }


def _step_summary(label: str, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "status": report["status"],
        "policy_config_hash": report["policy_config_hash"],
        "width_gear_fingerprint": report["width_gear_fingerprint"],
        "combination_count": report["combination_count"],
        "adjudicated_count": report["adjudicated_count"],
        "pending_count": len(report["pending"]),
        "reused_count": report["cache_summary"]["reused_count"],
        "recomputed_count": report["cache_summary"]["recomputed_count"],
        "flip_count": report["flip_attribution"]["summary"]["flip_count"],
    }


def main() -> int:
    output_path = ROOT / "artifacts" / "acceptance_validation.json"
    output_report_path = ROOT / "artifacts" / "acceptance_diagnostic_report.json"
    root = Path(tempfile.mkdtemp(prefix="category-cross-acceptance-", dir=ROOT / "artifacts"))
    try:
        sequence = run_edit_sequence(root)
        incremental_report = sequence["final_report"]

        fresh_parallel = build_engine(
            sequence["final_features"],
            sequence["final_strategy"],
            root / "fresh_parallel",
            replace_membership=True,
        ).run()
        fresh_serial = build_engine(
            sequence["final_features"],
            sequence["final_strategy"],
            root / "fresh_serial",
            serial=True,
            replace_membership=True,
        ).run()

        incremental_vs_full_diffs = deep_diff(
            normalize_report(incremental_report),
            normalize_report(fresh_parallel),
            path="incremental_vs_full",
        )
        serial_vs_parallel_diffs = deep_diff(
            normalize_report(fresh_serial, cache_status=True),
            normalize_report(fresh_parallel, cache_status=True),
            path="serial_vs_parallel",
        )

        twin_report = run_twin_sequence(root)
        ledger_match = normalize_report(
            incremental_report, cache_status=True, runtime=True
        )["cache_ledger"] == normalize_report(
            twin_report, cache_status=True, runtime=True
        )["cache_ledger"]

        result_errors = validate_errors()
        result_enumeration = validate_enumeration_and_hashes()
        result_dual = validate_dual_budget(root)
        result_recovery = validate_snapshot_recovery(root)
        result_consistency = {
            "incremental_vs_blank_full_field_match": not incremental_vs_full_diffs,
            "incremental_vs_blank_full_first_differences": incremental_vs_full_diffs[:10],
            "serial_vs_parallel_field_match": not serial_vs_parallel_diffs,
            "serial_vs_parallel_first_differences": serial_vs_parallel_diffs[:10],
            "twin_sequence_reuse_ledger_match": ledger_match,
            "cache_key_count": len(
                {entry["cache_key"] for entry in incremental_report["cache_ledger"]}
            ),
        }

        validation_ok = (
            result_errors["all_required_present"]
            and result_errors["failure_order_deterministic"]
        )
        enumeration_width_ok = all(
            value is True
            for key, value in result_enumeration.items()
            if key
            in (
                "initial_order_matches",
                "fingerprints_deterministic",
            )
        ) and all(
            result_enumeration["width_switch"][key] is True
            for key in (
                "combo_fingerprint_changed",
                "dictionary_digest_changed",
                "code_sample_changed",
                "unique_count_changed",
                "collision_count_changed",
            )
        )
        dual_ok = all(
            result_dual[key] is True
            for key in (
                "joint_trigger_recorded",
                "pending_order_correct",
            )
        ) and result_dual["pending_code"] == "E_POINT_BUDGET_EXHAUSTED"
        recovery_ok = result_recovery["matches_fresh_after_recovery"]
        consistency_ok = all(
            result_consistency[key] is True
            for key in (
                "incremental_vs_blank_full_field_match",
                "serial_vs_parallel_field_match",
                "twin_sequence_reuse_ledger_match",
            )
        )
        invalidation_ok = sequence["impact"]["affected_set_exact"] and sequence["impact"][
            "all_four_stages_recomputed"
        ]
        all_checks_passed = all(
            (
                validation_ok,
                enumeration_width_ok,
                dual_ok,
                recovery_ok,
                consistency_ok,
                invalidation_ok,
            )
        )

        result = {
            "schema": "acceptance-validation/v1",
            "temporary_root": str(root.relative_to(ROOT)),
            "all_checks_passed": all_checks_passed,
            "check_groups": {
                "validation_errors": validation_ok,
                "enumeration_and_width": enumeration_width_ok,
                "dual_budget": dual_ok,
                "snapshot_recovery": recovery_ok,
                "consistency": consistency_ok,
                "precise_invalidation": invalidation_ok,
            },
            "validation_errors": result_errors,
            "enumeration_and_width": result_enumeration,
            "dual_budget": result_dual,
            "snapshot_recovery": result_recovery,
            "incremental_sequence_timeline": sequence["timeline"],
            "final_column_invalidation": sequence["impact"],
            "consistency": result_consistency,
            "concurrency": fresh_parallel["concurrency"],
            "final_counts": {
                "combinations": incremental_report["combination_count"],
                "adjudicated": incremental_report["adjudicated_count"],
                "pending": len(incremental_report["pending"]),
                "issues": incremental_report["failure_count"],
                "flip_checks": incremental_report["flip_attribution"]["summary"][
                    "adjacent_pair_checks"
                ],
            },
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_report_path.write_text(
            json.dumps(incremental_report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["consistency"], ensure_ascii=False, indent=2))
        print(json.dumps(result["final_column_invalidation"], ensure_ascii=False, indent=2))
        print(json.dumps(result["check_groups"], ensure_ascii=False, indent=2))
        print(f"wrote {output_path.relative_to(ROOT)}")
        return 0 if all_checks_passed else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
