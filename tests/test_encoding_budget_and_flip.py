from __future__ import annotations

from cross_coding.config import load_bundle_from_docs
from cross_coding.pipeline import run_pipeline
from cross_coding.replay import apply_edit


def test_encoding_is_deterministic_and_retains_high_width_reference(bundle) -> None:
    first = run_pipeline(bundle)
    second = run_pipeline(bundle)
    key = "a|b|c"
    first_combo = next(item for item in first["combinations"] if item["canonical_key"] == key)
    second_combo = next(item for item in second["combinations"] if item["canonical_key"] == key)
    assert first_combo["encoding"]["codes"] == second_combo["encoding"]["codes"]
    assert first_combo["encoding"]["unique_count"] <= first_combo["encoding"]["width_capacity"]
    ref = first_combo["flip_attribution"]["reference_high_width"]
    assert ref["width_bits"] > first["width_profile"]["encoding_width_bits"]
    assert ref["unique_count"] >= first_combo["encoding"]["unique_count"]
    assert ref["collision_count"] <= first_combo["encoding"]["collision_count"]

    value_to_rows: dict[tuple, list[int]] = {}
    table = bundle["table"]
    for row in range(table["row_count"]):
        value = tuple(table["columns"][name]["values"][row] for name in ("a", "b", "c"))
        value_to_rows.setdefault(value, []).append(row)
    codes = first_combo["encoding"]["codes"]
    for rows in value_to_rows.values():
        assert len({codes[row] for row in rows}) == 1


def test_cardinality_truncation_and_skip() -> None:
    feature = {
        "seed": 99,
        "row_count": 40,
        "columns": [
            {"name": "a", "type": "categorical", "categories": [f"c{i}" for i in range(20)]},
            {"name": "b", "type": "categorical", "categories": [f"d{i}" for i in range(20)]},
        ],
    }
    policy = {
        "candidate_pairs": [["a", "b"]],
        "max_order": 2,
        "cardinality_budget": 3,
        "over_cardinality_strategy": "truncate_hash_buckets",
        "encoding_width_bits": 16,
        "fixed_point_bits": 16,
        "quantization_levels": 16,
        "encoding_point_budget": 10_000_000,
        "max_workers": 1,
    }
    bundle, errors = load_bundle_from_docs(feature, policy)
    assert errors.ok
    report = run_pipeline(bundle)
    pair = next(item for item in report["combinations"] if item["canonical_key"] == "a|b")
    ruling = pair["budget_decision"]
    assert ruling["over_budget"]
    assert ruling["action"] == "truncate_hash_buckets"
    assert ruling["bucket_count"] == 3
    assert ruling["final_unique_count"] <= 3

    skip_bundle = apply_edit(bundle, {"type": "set_cardinality_budget", "value": 0})
    skip_bundle["policy"]["over_cardinality_strategy"] = "skip"
    skip_report = run_pipeline(skip_bundle)
    skip_pair = next(item for item in skip_report["combinations"] if item["canonical_key"] == "a|b")
    assert skip_pair["budget_decision"]["action"] == "skip"
    assert skip_pair["budget_decision"]["verdict"] == "CARDINALITY_SKIPPED"


def test_point_budget_creates_stable_pending_and_priority(tmp_path, bundle) -> None:
    bundle["policy"]["encoding_point_budget"] = bundle["table"]["row_count"] * 8 * 2
    state = tmp_path / "point-state"
    first = run_pipeline(bundle, str(state))
    unit_cost = bundle["table"]["row_count"] * 8
    assert first["point_plan"]["unit_cost"] == unit_cost
    assert [item["rank"] for item in first["pending_queue"]] == [2, 3, 4, 5, 6, 7, 8]
    assert [item["code"] for item in first["pending_queue"]] == ["E_POINT_BUDGET_PENDING"] * 7

    larger = apply_edit(bundle, {"type": "set_point_budget", "value": 10_000_000})
    second = run_pipeline(larger, str(state))
    assert second["execution"]["previous_pending_priority"] == [
        item["canonical_key"] for item in first["pending_queue"]
    ]
    assert second["execution"]["scheduler_submission_order"][:7] == [
        item["canonical_key"] for item in first["pending_queue"]
    ]
    assert second["pending_queue"] == []


def test_width_and_quant_changes_produce_real_flips(bundle) -> None:
    width_report = run_pipeline(bundle)
    wide_bundle = apply_edit(bundle, {"type": "set_encoding_width", "value": 16})
    wide_report = run_pipeline(wide_bundle)
    for before, after in zip(width_report["combinations"], wide_report["combinations"]):
        assert before["combo_fingerprint"] != after["combo_fingerprint"]

    changed_unique = 0
    for before, after in zip(width_report["combinations"], wide_report["combinations"]):
        if before["encoding"]["unique_count"] != after["encoding"]["unique_count"]:
            changed_unique += 1
    assert changed_unique >= 1

    quant_bundle = apply_edit(bundle, {"type": "set_quantization_levels", "value": 32})
    quant_report = run_pipeline(quant_bundle)
    assert all(
        before["combo_fingerprint"] != after["combo_fingerprint"]
        for before, after in zip(width_report["combinations"], quant_report["combinations"])
    )

    fixed_bundle = apply_edit(bundle, {"type": "set_fixed_point_bits", "value": 24})
    fixed_report = run_pipeline(fixed_bundle)
    assert all(
        before["combo_fingerprint"] != after["combo_fingerprint"]
        for before, after in zip(width_report["combinations"], fixed_report["combinations"])
    )
    numeric_name = next(iter(width_report["numeric_quantization"]))
    assert (
        width_report["numeric_quantization"][numeric_name]["quantization_digest"]
        != fixed_report["numeric_quantization"][numeric_name]["quantization_digest"]
    )

    sample = next(
        item
        for item in width_report["combinations"]
        if item["flip_attribution"]["changed_scenario_count"] > 0
    )
    sources = sample["flip_attribution"]["source_counts"]
    assert "hash_salt_change" in sources
    scenario_names = {item["scenario"] for item in sample["flip_attribution"]["scenarios"]}
    assert {"width_up", "quant_up"} & scenario_names
