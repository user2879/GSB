from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from workbench.alignment import align_score
from workbench.features import extract_text_features, extract_visual_features
from workbench.hashing import hash_json
from workbench.samples import load_samples
from workbench.scheduler import Task, run_task_batch

from .conftest import Workspace


def test_feature_extraction_is_deterministic(workspace: Workspace) -> None:
    config = workspace.load_config()
    loaded = load_samples(workspace.samples)
    sample = loaded.samples[0]
    visual_a = extract_visual_features(sample, config)
    visual_b = extract_visual_features(sample, config)
    text_a = extract_text_features(sample, config)
    text_b = extract_text_features(sample, config)
    assert hash_json(visual_a) == hash_json(visual_b)
    assert hash_json(text_a) == hash_json(text_b)
    assert all(block["rounding_residual"] for block in visual_a["blocks"])
    assert all(entry["rounding_residual"] for entry in text_a["entries"])


def test_quantization_level_changes_internal_integers_and_scores(
    workspace: Workspace,
) -> None:
    config_a = workspace.load_config().with_quantization_level(0)
    config_b = workspace.load_config().with_quantization_level(2)
    loaded = load_samples(workspace.samples)
    sample = loaded.samples[0]
    visual_a = extract_visual_features(sample, config_a)
    visual_b = extract_visual_features(sample, config_b)
    text_a = extract_text_features(sample, config_a)
    text_b = extract_text_features(sample, config_b)
    score_a = align_score(visual_a, text_a, config_a)
    score_b = align_score(visual_b, text_b, config_b)

    assert visual_a["fixed_decimal_bits"] != visual_b["fixed_decimal_bits"]
    assert visual_a["blocks"][0]["quantized"] != visual_b["blocks"][0]["quantized"]
    assert text_a["entries"][0]["quantized"] != text_b["entries"][0]["quantized"]
    assert (score_a["score_fixed"], score_a["fixed_scale"]) != (
        score_b["score_fixed"],
        score_b["fixed_scale"],
    )


def test_thread_pool_respects_limit_and_serial_result_order() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    def work(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value

    tasks = [
        Task(
            sample=f"sample-{index:02d}",
            stage="visual",
            row=index,
            col=0,
            tail="block",
            points=1,
            fn=lambda value=index: work(value),
        )
        for index in range(20)
    ]
    result = run_task_batch(tasks, worker_limit=3, point_budget=100)
    assert result.active_peak <= 3
    assert result.active_peak >= 2
    assert result.completed_task_count == 20
    ordered_keys = [task.stable_key for task in sorted(tasks, key=lambda task: task.sort_key)]
    assert sorted(result.results) == sorted(ordered_keys)


def test_point_budget_leaves_stable_pending_queue(workspace: Workspace) -> None:
    config = workspace.load_config()
    report = workspace.run(
        mode="full",
        output_name="budget-output",
        config=type(config).from_mapping({**config.raw, "feature_point_budget": 10}),
    )
    assert "E_POINT_BUDGET_PENDING" in report["business_codes"]
    assert report["scheduler"]["consumed_points"] <= 10
    assert report["pending_tasks"]
    assert report["retrieval_index"]["queries"] == []
    stage_order = {"visual": 0, "text": 1, "align": 2}
    pending = report["pending_tasks"]
    expected = sorted(
        pending,
        key=lambda item: (
            item["sample"],
            stage_order[item["stage"]],
            item["row"],
            item["col"],
            item["sort_key"],
        ),
    )
    assert pending == expected
    assert all(item["code"] == "E_POINT_BUDGET_PENDING" for item in report["pending_tasks"])


def test_default_config_exercises_exact_threshold_boundary(workspace: Workspace) -> None:
    report = workspace.run(mode="full")
    query = next(
        item for item in report["retrieval_index"]["queries"] if item["query"] == "sample-01"
    )
    boundary = next(item for item in query["results"] if item["sample"] == "sample-02")
    assert boundary["boundary"] is True
    assert boundary["score_fixed"] == query["threshold_fixed"]
    assert boundary["quantization_comparisons"]
    sensitive = report["boundary_diagnostics"]["sensitive"]
    assert sensitive
    assert any(item["first_exceeded_stage"] for item in sensitive)
    assert "E_BOUNDARY_SENSITIVE" in report["business_codes"]
