from __future__ import annotations

from pathlib import Path

from instruction_factory.constants import (
    E_GENERATION_BUDGET_EXHAUSTED,
    E_SCORE_BUDGET_EXHAUSTED,
    E_TASK_BUDGET_EXHAUSTED,
)
from instruction_factory.loader import load_inputs
from instruction_factory.runner import run_full

from .conftest import read_json, write_json


def _prepare(sample_dir: Path, mutate) -> Path:
    plan_path = sample_dir / "plan.json"
    plan = read_json(plan_path)
    plan["batches"] = []
    write_json(plan_path, plan)
    config_path = sample_dir / "config.json"
    config = read_json(config_path)
    config["global_score_budget"] = 1000
    config["per_batch_generation_points"] = 1000
    config["task_quotas"] = {"classify": 100, "rewrite": 100, "summarize": 100}
    mutate(config)
    write_json(config_path, config)
    return sample_dir


def test_task_quota_stops_only_that_task(sample_dir: Path, tmp_path: Path) -> None:
    def mutate(config: dict) -> None:
        config["task_quotas"]["classify"] = 2

    inputs = load_inputs(_prepare(sample_dir, mutate))
    report, code = run_full(inputs=inputs, output_dir=tmp_path / "out", reset=True)
    assert code == 0
    task_failures = [item for item in report["failures"] if item["code"] == E_TASK_BUDGET_EXHAUSTED]
    assert [item["candidate_id"] for item in task_failures] == [
        "cand-classify-000003",
        "cand-classify-000004",
        "cand-classify-000005",
    ]
    assert report["task_statistics"]["classify"]["scored_candidate_count"] == 2
    assert report["task_statistics"]["rewrite"]["scored_candidate_count"] == 5
    assert report["task_statistics"]["summarize"]["scored_candidate_count"] == 6
    assert report["scheduler"]["pending_queue"] == []


def test_global_score_budget_retains_pending_queue(sample_dir: Path, tmp_path: Path) -> None:
    def mutate(config: dict) -> None:
        config["global_score_budget"] = 5

    inputs = load_inputs(_prepare(sample_dir, mutate))
    report, code = run_full(inputs=inputs, output_dir=tmp_path / "out", reset=True)
    assert code == 0
    assert len(report["candidates"]) == 1
    score_failures = [item for item in report["failures"] if item["code"] == E_SCORE_BUDGET_EXHAUSTED]
    assert len(score_failures) == 15
    assert len(report["scheduler"]["pending_queue"]) == 15
    assert report["scheduler"]["pending_queue"][0]["candidate_id"] == "cand-classify-000002"
    ledger = report["scheduler"]["ledger"]
    assert ledger[0]["global_score"]["cost"] == 3
    assert ledger[1]["status"] == "generated_score_pending"
    assert ledger[1]["batch_generation_points"]["cost"] == 1
    assert ledger[1]["global_score"]["cost"] == 0


def test_generation_point_budget_stops_before_sampling(sample_dir: Path, tmp_path: Path) -> None:
    def mutate(config: dict) -> None:
        config["per_batch_generation_points"] = 2

    inputs = load_inputs(_prepare(sample_dir, mutate))
    report, code = run_full(inputs=inputs, output_dir=tmp_path / "out", reset=True)
    assert code == 0
    assert len(report["candidates"]) == 2
    generation_failures = [item for item in report["failures"] if item["code"] == E_GENERATION_BUDGET_EXHAUSTED]
    assert len(generation_failures) == 14
    assert len(report["scheduler"]["pending_queue"]) == 14
    assert all(item["stage"] == "sampling" for item in report["scheduler"]["pending_queue"])
    assert report["scheduler"]["remaining_budget"]["per_batch_generation_points"] == 0

