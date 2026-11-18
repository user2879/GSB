from __future__ import annotations

import json
from pathlib import Path

from columnar_pipeline.ledger import Ledger
from columnar_pipeline.model import RunConfig
from columnar_pipeline.runner import run_pipeline


def test_skew_split_budget_and_ledger(
    tmp_path: Path,
    skew_data: Path,
    params: Path,
) -> None:
    config = RunConfig.from_json_dict(
        json.loads((params / "skew.json").read_text(encoding="utf-8"))
    )
    summary = run_pipeline(
        skew_data,
        config,
        out_dir=tmp_path / "skew",
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        log_path=tmp_path / "pipeline.jsonl",
    )
    trace = json.loads((tmp_path / "skew" / "trace.json").read_text(encoding="utf-8"))
    assert summary["split_count"] == 3
    assert summary["split_children_count"] == 12
    assert summary["task_status_counts"]["split_parent"] == 3
    split_decisions = [item for item in trace["skew_decisions"] if item["action"] == "split"]
    assert len(split_decisions) == 3
    for decision in split_decisions:
        children = decision["children"]
        assert len(children) == 4
        for left, right in zip(children, children[1:]):
            assert left["row_end"] == right["row_start"]
        parent = next(
            task for task in trace["tasks"] if task["task_id"] == decision["task_id"]
        )
        assert children[0]["row_start"] == parent["row_start"]
        assert children[-1]["row_end"] == parent["row_end"]
        assert parent["status"] == "split_parent"
        assert parent["included"] is False
    for event in trace["budget_events"]:
        assert event["used_bytes"] <= event["budget_bytes"]
    queued_tasks = [task for task in trace["tasks"] if task["queue_count"] > 0]
    assert queued_tasks
    assert all(task["queue_started_ms"] is not None for task in queued_tasks)
    assert all(task["queue_started_ms"] <= task["start_ms"] for task in queued_tasks)
    assert any(event["event"] == "queued" for event in trace["budget_events"])
    assert trace["peak_memory_bytes"] <= summary["effective_memory_budget_bytes"]
    assert all(task["start_ms"] <= task["end_ms"] for task in trace["tasks"])

    with Ledger(tmp_path / "ledger.sqlite") as ledger:
        tasks = ledger.get_tasks(summary["run_id"])
        skews = ledger.get_skew_decisions(summary["run_id"])
        budgets = ledger.get_budget_events(summary["run_id"])
        batches = ledger.get_batch_states(summary["run_id"])
    assert len(tasks) == summary["task_count"]
    assert len(skews) >= 3
    assert budgets
    assert batches
    assert all(task["aggregate_json"] for task in tasks)
    assert all(task["queue_count"] >= 0 for task in tasks)
    assert any(task["split_source"] for task in tasks)


def test_tight_budget_produces_identical_result_bytes(
    tmp_path: Path,
    skew_data: Path,
    params: Path,
) -> None:
    normal_config = RunConfig.from_json_dict(
        json.loads((params / "skew.json").read_text(encoding="utf-8"))
    )
    tight_config = RunConfig.from_json_dict(
        json.loads((params / "tight_budget.json").read_text(encoding="utf-8"))
    )
    normal = run_pipeline(
        skew_data,
        normal_config,
        out_dir=tmp_path / "normal",
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        log_path=tmp_path / "pipeline.jsonl",
    )
    tight = run_pipeline(
        skew_data,
        tight_config,
        out_dir=tmp_path / "tight",
        ledger_path=tmp_path / "ledger.sqlite",
        snapshot_dir=tmp_path / "snapshots",
        log_path=tmp_path / "pipeline.jsonl",
    )
    assert normal["result_digest"] == tight["result_digest"]
    assert (tmp_path / "normal" / "final_result.json").read_bytes() == (
        tmp_path / "tight" / "final_result.json"
    ).read_bytes()
    tight_trace = json.loads((tmp_path / "tight" / "trace.json").read_text(encoding="utf-8"))
    assert any(event["event"] == "start" for event in tight_trace["budget_events"])
    assert tight_trace["peak_memory_bytes"] <= tight_config.memory_budget_bytes
