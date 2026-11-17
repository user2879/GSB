from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from transfer_pipeline.canonical import canonical_json
from transfer_pipeline.data import read_structured_file
from transfer_pipeline.edits import apply_edit
from transfer_pipeline.pipeline import run_pipeline
from transfer_pipeline.verification import _outcomes

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture()
def documents():
    return {
        "tasks": read_structured_file(EXAMPLES / "tasks.json"),
        "features": read_structured_file(EXAMPLES / "features.json"),
        "encoder": read_structured_file(EXAMPLES / "encoder.json"),
        "config": read_structured_file(EXAMPLES / "adapter_config.json"),
        "injections": read_structured_file(EXAMPLES / "injections.json"),
    }


def run(documents, state_directory, **kwargs):
    return run_pipeline(
        documents["tasks"],
        documents["features"],
        documents["encoder"],
        documents["config"],
        state_directory=state_directory,
        **kwargs,
    )


def assert_same_outcomes(left, right):
    assert canonical_json(_outcomes(left.report)) == canonical_json(_outcomes(right.report))


def test_full_runs_are_bitwise_deterministic_across_workspaces(documents, tmp_path):
    first = run(documents, tmp_path / "state-a")
    second = run(documents, tmp_path / "state-b")
    assert first.exit_code == second.exit_code == 0
    assert_same_outcomes(first, second)
    snapshot_a = json.loads((tmp_path / "state-a" / "snapshot.json").read_text())
    snapshot_b = json.loads((tmp_path / "state-b" / "snapshot.json").read_text())
    assert snapshot_a["representation_cache"] == snapshot_b["representation_cache"]
    assert snapshot_a["adapter_cache"] == snapshot_b["adapter_cache"]
    assert snapshot_a["evaluation_cache"] == snapshot_b["evaluation_cache"]


def test_noop_rerun_reuses_every_adapter_and_metric(documents, tmp_path):
    first = run(documents, tmp_path / "state")
    assert first.report["cache"]["training_invocations"] == 4
    second = run(documents, tmp_path / "state")
    assert second.report["cache"]["training_invocations"] == 0
    assert second.report["cache"]["adapter_hits"] == 4
    assert second.report["cache"]["evaluation_hits"] == 4
    assert second.report["incremental"]["reused_tasks"] == [
        "intent",
        "mood",
        "sentiment",
        "topic",
    ]
    assert second.report["snapshot_version"] == first.report["snapshot_version"] + 1


def test_incremental_edit_sequence_matches_full_rebuild_at_every_step(documents, tmp_path):
    edits = read_structured_file(EXAMPLES / "edits.json")
    state = tmp_path / "incremental"
    current = run(documents, state)
    assert current.exit_code == 0
    for index, edit in enumerate(edits, start=1):
        expected_invalidated = set()
        if edit["op"] == "add_task":
            expected_invalidated = {edit["task"]["name"]}
        elif edit["op"] == "delete_task":
            expected_invalidated = set()
        elif edit["op"] == "set_classes":
            expected_invalidated = {edit["name"]}
        elif edit["op"] == "replace_sample":
            expected_invalidated = set(
                current.report["sample_references"].get(edit["sample_id"], [])
            )
        elif edit["op"] == "set_adapter_params":
            expected_invalidated = {edit["name"]}
        apply_edit(documents, edit)
        incremental = run(documents, state)
        full = run(documents, tmp_path / f"full-{index}")
        assert incremental.exit_code == full.exit_code == 0
        assert canonical_json(_outcomes(incremental.report)) == canonical_json(
            _outcomes(full.report)
        )
        assert set(incremental.report["incremental"]["invalidated_tasks"]) == (
            expected_invalidated
        )
        current_names = {task["name"] for task in incremental.report["tasks"]}
        assert set(incremental.report["incremental"]["reused_tasks"]) == (
            current_names - expected_invalidated
        )
        current = incremental


def test_validation_failures_are_sorted_and_do_not_block_valid_tasks(documents, tmp_path):
    valid = copy.deepcopy(documents["tasks"]["tasks"][0])
    duplicate = copy.deepcopy(valid)
    empty_classes = copy.deepcopy(documents["tasks"]["tasks"][1])
    empty_classes["name"] = "empty-classes"
    empty_classes["classes"] = []
    missing = copy.deepcopy(documents["tasks"]["tasks"][2])
    missing["name"] = "missing-sample"
    missing["train_ids"] = ["not-present"]
    wrong_dim = copy.deepcopy(documents["tasks"]["tasks"][3])
    wrong_dim["name"] = "wrong-dim"
    wrong_dim["shared_feature_dim"] = 7
    documents["tasks"] = {
        "shared_feature_dim": 8,
        "tasks": [
            valid,
            duplicate,
            empty_classes,
            missing,
            wrong_dim,
            copy.deepcopy(documents["tasks"]["tasks"][1]),
            copy.deepcopy(documents["tasks"]["tasks"][2]),
            copy.deepcopy(documents["tasks"]["tasks"][3]),
        ],
    }
    result = run(documents, tmp_path / "state")
    assert [task["name"] for task in result.report["tasks"]] == [
        "intent",
        "mood",
        "topic",
    ]
    assert len(result.report["failures"]) == 4
    assert all(
        failure["code"] == "E_TASK_DESC_INVALID"
        for failure in result.report["failures"]
    )
    keys = [
        (failure["task_name"], failure["stage"], failure["code"])
        for failure in result.report["failures"]
    ]
    assert keys == sorted(keys)


def test_injection_ids_prove_all_business_fault_hits(documents, tmp_path):
    result = run(
        documents,
        tmp_path / "state",
        injection_document=documents["injections"],
    )
    assert result.exit_code == 0
    assert result.report["tasks"] == []
    hits = {item["id"]: item for item in result.report["injection_hits"]}
    assert set(hits) == {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
    assert all(item["hit"] for item in hits.values())
    assert {
        failure["code"] for failure in result.report["failures"]
    } == {"E_TASK_DESC_INVALID", "E_ADAPTER_CONFIG_INVALID"}


def test_train_and_evaluation_exceptions_are_isolated(documents, tmp_path):
    injections = {
        "injections": [
            {"id": "INJ-T", "type": "train_error", "task": "sentiment"},
            {"id": "INJ-E", "type": "evaluate_error", "task": "topic"},
        ]
    }
    result = run(
        documents,
        tmp_path / "state",
        injection_document=injections,
    )
    assert result.exit_code == 0
    assert [task["name"] for task in result.report["tasks"]] == ["intent", "mood"]
    failures = {failure["task_name"]: failure for failure in result.report["failures"]}
    assert failures["sentiment"]["stage"] == "adapter_train"
    assert failures["sentiment"]["code"] == "E_TRAINING_FAILED"
    assert failures["sentiment"]["exception_type"] == "RuntimeError"
    assert failures["topic"]["stage"] == "evaluation"
    assert failures["topic"]["code"] == "E_EVALUATION_FAILED"
    assert {item["id"] for item in result.report["injection_hits"] if item["hit"]} == {
        "INJ-T",
        "INJ-E",
    }


@pytest.mark.parametrize("concurrency", [1, 2, 8])
def test_concurrency_does_not_change_serial_outputs(documents, tmp_path, concurrency):
    documents["config"]["concurrency"] = concurrency
    documents["config"]["snapshot_commit_interval"] = 1
    result = run(documents, tmp_path / f"state-{concurrency}")
    reference_documents = copy.deepcopy(documents)
    reference_documents["config"]["concurrency"] = 1
    reference = run(reference_documents, tmp_path / f"reference-{concurrency}")
    assert canonical_json(_outcomes(result.report)) == canonical_json(
        _outcomes(reference.report)
    )
    assert [task["name"] for task in result.report["tasks"]] == sorted(
        task["name"] for task in result.report["tasks"]
    )


def test_interrupted_snapshot_resumes_to_full_result(documents, tmp_path):
    documents["config"]["concurrency"] = 1
    documents["config"]["snapshot_commit_interval"] = 1
    state = tmp_path / "state"
    partial = run(documents, state, stop_after_completed=1)
    assert partial.exit_code == 0
    assert partial.report["interrupted"] is True
    assert partial.report["run_state"] == "interrupted"
    assert 0 < len(partial.report["tasks"]) < 4
    partial_version = partial.report["snapshot_version"]
    snapshot_before = json.loads((state / "snapshot.json").read_text())
    assert snapshot_before["global_version"] == partial_version
    assert snapshot_before["run_state"] == "interrupted"
    resumed = run(
        documents,
        state,
        stop_event=threading.Event(),
        expected_version=partial_version,
        require_resume=True,
    )
    assert resumed.exit_code == 0
    assert resumed.report["resumed"] is True
    assert resumed.report["run_state"] == "committed"
    assert resumed.report["snapshot_version"] == partial_version
    full = run(documents, tmp_path / "full")
    assert canonical_json(_outcomes(resumed.report)) == canonical_json(
        _outcomes(full.report)
    )


def test_snapshot_config_and_version_mismatch_use_stable_code(documents, tmp_path):
    state = tmp_path / "state"
    first = run(documents, state)
    assert first.exit_code == 0
    changed_encoder = copy.deepcopy(documents["encoder"])
    changed_encoder["bias"][0] += 0.25
    mismatch = run_pipeline(
        documents["tasks"],
        documents["features"],
        changed_encoder,
        documents["config"],
        state_directory=state,
    )
    assert mismatch.exit_code == 2
    assert mismatch.report["failures"][0]["code"] == "E_CONFIG_MISMATCH"
    version_mismatch = run(
        documents,
        state,
        expected_version=first.report["snapshot_version"] + 99,
    )
    assert version_mismatch.exit_code == 2
    assert version_mismatch.report["failures"][0]["code"] == "E_CONFIG_MISMATCH"
    snapshot_path = state / "snapshot.json"
    corrupted = json.loads(snapshot_path.read_text())
    corrupted["global_version"] = first.report["snapshot_version"] + 99
    snapshot_path.write_text(json.dumps(corrupted), encoding="utf-8")
    corrupt_version = run(documents, state)
    assert corrupt_version.exit_code == 2
    assert corrupt_version.report["failures"][0]["code"] == "E_CONFIG_MISMATCH"
    resume_mismatch = run(documents, state, require_resume=True)
    assert resume_mismatch.exit_code == 2
    assert resume_mismatch.report["failures"][0]["code"] == "E_CONFIG_MISMATCH"


def test_validation_only_sample_change_reuses_adapter_and_recomputes_metrics(
    documents, tmp_path
):
    state = tmp_path / "state"
    first = run(documents, state)
    assert first.exit_code == 0
    replacement = read_structured_file(EXAMPLES / "replace_sample.json")
    edit = {
        "op": "replace_sample",
        "sample_id": replacement["sample_id"],
        "features": replacement["features"],
        "label": replacement["label"],
    }
    apply_edit(documents, edit)
    incremental = run(documents, state)
    assert incremental.report["incremental"]["invalidated_tasks"] == ["sentiment"]
    assert "sentiment" in incremental.report["incremental"]["adapter_reused_tasks"]
    assert "sentiment" in incremental.report["incremental"][
        "evaluation_recomputed_tasks"
    ]
    assert "sentiment" not in incremental.report["incremental"]["trained_tasks"]
    sentiment = next(
        task for task in incremental.report["tasks"] if task["name"] == "sentiment"
    )
    assert sentiment["cache"] == {"adapter_hit": True, "evaluation_hit": False}
    full = run(documents, tmp_path / "full")
    assert canonical_json(_outcomes(incremental.report)) == canonical_json(
        _outcomes(full.report)
    )


def test_threshold_change_reuses_adapter_and_recomputes_only_evaluation(
    documents, tmp_path
):
    state = tmp_path / "state"
    run(documents, state)
    apply_edit(
        documents,
        {
            "op": "set_adapter_params",
            "name": "sentiment",
            "changes": {"validation_threshold": 0.95},
        },
    )
    incremental = run(documents, state)
    assert incremental.report["incremental"]["adapter_reused_tasks"] == [
        "intent",
        "mood",
        "sentiment",
        "topic",
    ]
    assert incremental.report["incremental"]["evaluation_recomputed_tasks"] == [
        "sentiment"
    ]
    assert incremental.report["incremental"]["trained_tasks"] == []
    sentiment = next(
        task for task in incremental.report["tasks"] if task["name"] == "sentiment"
    )
    assert sentiment["metrics"]["threshold"] == 0.95
    full = run(documents, tmp_path / "full")
    assert canonical_json(_outcomes(incremental.report)) == canonical_json(
        _outcomes(full.report)
    )


def test_snapshot_and_report_contain_required_diagnostics(documents, tmp_path):
    report_path = tmp_path / "report.json"
    result = run(documents, tmp_path / "state", report_path=report_path)
    report = json.loads(report_path.read_text())
    assert result.report == report
    assert report["snapshot_version"] == 1
    assert report["snapshot"]["completed_tasks"] == [
        "intent",
        "mood",
        "sentiment",
        "topic",
    ]
    assert report["snapshot"]["completed_stages"] == [
        "adapter",
        "evaluation",
        "shared_forward",
    ]
    assert report["sample_references"]["s001"] == ["sentiment"]
    for task in report["tasks"]:
        summary = task["adapter_parameter_summary"]
        assert summary["digest"]
        assert summary["parameter_count"] > 0
        assert "accuracy" in task["metrics"]
        assert "macro_f1" in task["metrics"]
    assert not list((tmp_path / "state").glob("*.tmp"))


def test_actual_sigterm_finishes_current_task_then_exits_zero(documents, tmp_path):
    base_task = documents["tasks"]["tasks"][0]
    tasks = []
    for index in range(100):
        task = copy.deepcopy(base_task)
        task["name"] = f"task-{index:03d}"
        tasks.append(task)
    documents["tasks"] = {"shared_feature_dim": 8, "tasks": tasks}
    documents["config"]["concurrency"] = 1
    documents["config"]["snapshot_commit_interval"] = 1
    documents["config"]["epochs"] = 40
    tasks_path = tmp_path / "tasks.json"
    config_path = tmp_path / "config.json"
    tasks_path.write_text(json.dumps(documents["tasks"]), encoding="utf-8")
    config_path.write_text(json.dumps(documents["config"]), encoding="utf-8")
    state = tmp_path / "state"
    report_path = tmp_path / "report.json"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "transfer_pipeline",
            "run",
            "--tasks",
            str(tasks_path),
            "--features",
            str(EXAMPLES / "features.json"),
            "--encoder",
            str(EXAMPLES / "encoder.json"),
            "--config",
            str(config_path),
            "--state-dir",
            str(state),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10.0
    observed_progress = False
    while time.monotonic() < deadline:
        snapshot_path = state / "snapshot.json"
        if snapshot_path.exists():
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                snapshot = {}
            if snapshot.get("completed_tasks"):
                observed_progress = True
                break
        if process.poll() is not None:
            break
        time.sleep(0.01)
    assert observed_progress
    process.terminate()
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, (stdout, stderr)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exit_code"] == 0
    assert report["run_state"] == "interrupted"
    assert report["interrupted"] is True
    assert 0 < len(report["tasks"]) < 100
    snapshot = json.loads((state / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["run_state"] == "interrupted"


def test_common_field_aliases_numeric_labels_and_optional_defaults(documents, tmp_path):
    base_task = documents["tasks"]["tasks"][0]
    class_map = {"negative": 0, "positive": 1}
    sample_ids = list(base_task["train_ids"]) + list(base_task["val_ids"])
    feature_document = {
        "shared_feature_dim": 8,
        "samples": [
            {
                "id": sample_id,
                "vector": documents["features"]["samples"][sample_id]["features"],
                "class": class_map[documents["features"]["samples"][sample_id]["label"]],
            }
            for sample_id in sample_ids
        ],
    }
    task_document = {
        "feature_dim": 8,
        "tasks": [
            {
                "task_name": "alias-task",
                "class_set": [0, 1],
                "training_sample_ids": base_task["train_ids"],
                "validation_sample_ids": base_task["val_ids"],
            }
        ],
    }
    encoder = documents["encoder"]
    encoder_document = {
        "input_dim": 8,
        "output_dim": 6,
        "W": [list(row) for row in zip(*encoder["weights"])],
        "bias": encoder["bias"],
        "mean": encoder["normalization"]["mean"],
        "std": encoder["normalization"]["std"],
    }
    config = {
        "type": "linear_probe",
        "adapter_rank": 2,
        "learning_rate": 0.4,
        "num_epochs": 80,
        "batch": 8,
        "validation_metric_threshold": 0.5,
        "max_concurrency": 2,
        "snapshot_interval": 1,
    }
    result = run_pipeline(
        task_document,
        feature_document,
        encoder_document,
        config,
        state_directory=tmp_path / "state",
    )
    assert result.exit_code == 0
    assert [task["name"] for task in result.report["tasks"]] == ["alias-task"]
    assert result.report["tasks"][0]["metrics"]["accuracy"] >= 0.5
