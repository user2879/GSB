from __future__ import annotations

import copy
from decimal import Decimal

from instruction_factory.cache import ContentCache
from instruction_factory.loader import load_inputs
from instruction_factory.plan import final_state
from instruction_factory.pipeline import run_pipeline
from instruction_factory.reporting import compare_reports
from instruction_factory.scoring import Quantizer

from .conftest import DATA_DIR


def test_repeated_pipeline_is_field_identical() -> None:
    inputs = load_inputs(DATA_DIR)
    state, failures = final_state(inputs)
    assert failures == []
    first = run_pipeline(state=state, inputs=inputs, cache=ContentCache(None)).report
    second = run_pipeline(state=state, inputs=inputs, cache=ContentCache(None)).report
    comparison = compare_reports(first, second)
    assert comparison["equal"], comparison["mismatches"]
    assert [item["candidate_id"] for item in first["candidates"]] == [item["candidate_id"] for item in second["candidates"]]
    assert [item["sample_seq"] for item in first["candidates"]] == [item["sample_seq"] for item in second["candidates"]]
    assert [item["source_name"] for item in first["candidates"]] == [item["source_name"] for item in second["candidates"]]


def test_quantization_width_and_step_change_integer_and_residual() -> None:
    raw = Decimal("85.347")
    width_two = Quantizer(places=2, step=Decimal("0.01")).quantize(raw)
    width_three = Quantizer(places=3, step=Decimal("0.01")).quantize(raw)
    coarse = Quantizer(places=3, step=Decimal("0.05")).quantize(Decimal("85.33"))
    precise = Quantizer(places=2, step=Decimal("0.01")).quantize(Decimal("85.33"))

    assert width_two.fixed == 8535
    assert width_three.fixed == 85350
    assert width_two.display == "85.35"
    assert width_three.display == "85.350"
    assert width_two.residual != width_three.residual or width_two.fixed != width_three.fixed
    assert coarse.fixed != precise.fixed
    assert coarse.display == "85.350"
    assert precise.display == "85.33"


def test_actual_pipeline_quantization_changes_candidate_scores() -> None:
    inputs = load_inputs(DATA_DIR)
    state, _ = final_state(inputs)
    baseline = run_pipeline(state=state, inputs=inputs, cache=ContentCache(None)).report

    changed_state = copy.deepcopy(state)
    changed_state.config["fixed_decimal_places"] = 2
    changed_state.config["quantization_step"] = 0.05
    changed = run_pipeline(state=changed_state, inputs=inputs, cache=ContentCache(None)).report

    baseline_candidate = baseline["candidates"][0]
    changed_candidate = changed["candidates"][0]
    assert baseline_candidate["candidate_id"] == changed_candidate["candidate_id"]
    assert baseline_candidate["total_fixed"] != changed_candidate["total_fixed"]
    assert baseline_candidate["total_display"].count(".") == 1
    assert len(baseline_candidate["total_display"].split(".")[1]) == 3
    assert len(changed_candidate["total_display"].split(".")[1]) == 2
    assert baseline_candidate["quantization_residual"] != changed_candidate["quantization_residual"]
    assert changed["quantization"]["fixed_decimal_places"] == 2


def test_boundary_samples_are_score_equal_and_id_sorted() -> None:
    inputs = load_inputs(DATA_DIR)
    state, _ = final_state(inputs)
    report = run_pipeline(state=state, inputs=inputs, cache=ContentCache(None)).report
    boundary = report["boundary_samples"]
    assert boundary
    assert [item["candidate_id"] for item in boundary] == sorted(item["candidate_id"] for item in boundary)
    assert all(item["total_fixed"] == item["threshold_fixed"] for item in boundary)
    assert all(
        candidate["boundary_decision"] == "keep_and_order_by_candidate_id"
        for candidate in report["candidates"]
        if candidate["boundary"]
    )

