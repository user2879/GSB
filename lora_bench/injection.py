"""Fault injection scenarios, isolation and atomic diagnostic reporting."""

from __future__ import annotations

import os
import shutil

import numpy as np

from .errors import (
    E_BASE_FINITE_MISMATCH,
    E_RANK_INVALID,
    E_SAMPLE_INVALID,
    E_TARGET_LAYER_UNKNOWN,
    E_VALIDATION_EMPTY,
    WorkbenchError,
)
from .generator import build_default_dataset
from .loader import load_base_pack
from .pipeline import run_pipeline
from .utils import write_json_atomic

INJECTION_SPECS = {
    "INJ-01": {
        "description": "样本缺少必需字段",
        "expected_code": E_SAMPLE_INVALID,
    },
    "INJ-02": {
        "description": "基座权重出现非有限数",
        "expected_code": E_BASE_FINITE_MISMATCH,
    },
    "INJ-03": {
        "description": "目标层不在基座层范围内",
        "expected_code": E_TARGET_LAYER_UNKNOWN,
    },
    "INJ-04": {
        "description": "秩为零",
        "expected_code": E_RANK_INVALID,
    },
    "INJ-05": {
        "description": "验证样本列表为空",
        "expected_code": E_VALIDATION_EMPTY,
    },
    "INJ-06": {
        "description": "训练阶段任务级异常隔离",
        "expected_code": "E_TRAIN_TASK_FAULT",
    },
}


def _base_config() -> dict:
    return {
        "rank": 2,
        "alpha": 4.0,
        "target_layers": [0, 2],
        "learning_rate": 0.02,
        "batch_size": 4,
        "epochs": 3,
        "metric": "accuracy",
        "metric_threshold": 0.5,
        "concurrency": 2,
        "commit_interval": 2,
    }


def _prepare_case_dir(root: str, injection_id: str) -> tuple[str, str]:
    case_input = os.path.join(root, injection_id, "input")
    case_workspace = os.path.join(root, injection_id, "workspace")
    if os.path.exists(os.path.join(root, injection_id)):
        shutil.rmtree(os.path.join(root, injection_id))
    os.makedirs(case_input, exist_ok=True)
    build_default_dataset(case_input)
    return case_input, case_workspace


def _corrupt_sample_line(case_input: str) -> None:
    path = os.path.join(case_input, "samples.jsonl")
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    import json

    record = json.loads(lines[0])
    del record["input_tokens"]
    lines[0] = json.dumps(record, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)


def _corrupt_base_tensor(case_input: str) -> None:
    base_dir = os.path.join(case_input, "base_v1")
    tensor_path = os.path.join(base_dir, "layer1.weight.bin")
    raw = np.fromfile(tensor_path, dtype=np.float32).copy()
    raw[0] = np.inf
    raw.tofile(tensor_path)


def run_injection_suite(report_root: str) -> dict:
    """Execute every injection in an isolated directory and write one report."""
    if os.path.exists(report_root):
        shutil.rmtree(report_root)
    os.makedirs(report_root, exist_ok=True)
    report_entries = []
    for injection_id, spec in INJECTION_SPECS.items():
        case_input, case_workspace = _prepare_case_dir(
            report_root, injection_id
        )
        config = _base_config()
        entry = {
            "injection_id": injection_id,
            "description": spec["description"],
            "expected_code": spec["expected_code"],
            "hit": False,
            "observed_code": None,
            "stage": None,
            "detail": None,
        }
        try:
            if injection_id == "INJ-01":
                _corrupt_sample_line(case_input)
                outcome = run_pipeline(case_input, case_workspace, config)
                codes = [failure["code"] for failure in outcome["failures"]]
                entry["hit"] = E_SAMPLE_INVALID in codes
                entry["observed_code"] = E_SAMPLE_INVALID if entry["hit"] else (
                    codes[0] if codes else None
                )
                entry["stage"] = "load"
                entry["detail"] = outcome["failures"][0]["message"] if outcome["failures"] else None
            elif injection_id == "INJ-02":
                _corrupt_base_tensor(case_input)
                load_base_pack(os.path.join(case_input, "base_v1"))
            elif injection_id == "INJ-03":
                config["target_layers"] = [0, 99]
                run_pipeline(case_input, case_workspace, config)
            elif injection_id == "INJ-04":
                config["rank"] = 0
                run_pipeline(case_input, case_workspace, config)
            elif injection_id == "INJ-05":
                tasks_path = os.path.join(case_input, "tasks.json")
                from .utils import read_json

                manifest = read_json(tasks_path)
                manifest["tasks"][0]["val_sample_ids"] = []
                write_json_atomic(tasks_path, manifest)
                outcome = run_pipeline(case_input, case_workspace, config)
                target_failure = next(
                    (
                        failure
                        for failure in outcome["failures"]
                        if failure["code"] == E_VALIDATION_EMPTY
                    ),
                    None,
                )
                entry["hit"] = target_failure is not None
                entry["observed_code"] = (
                    target_failure["code"] if target_failure else None
                )
                entry["stage"] = "validate"
                entry["detail"] = (
                    target_failure["message"] if target_failure else None
                )
            elif injection_id == "INJ-06":
                from .utils import read_json

                manifest = read_json(os.path.join(case_input, "tasks.json"))
                victim = manifest["tasks"][0]["task"]
                hook = WorkbenchError(
                    "E_TRAIN_TASK_FAULT",
                    "injected training failure",
                    stage="train",
                )
                outcome = run_pipeline(
                    case_input,
                    case_workspace,
                    config,
                    fault_hooks={(victim, "train"): hook},
                )
                target_failure = next(
                    (
                        failure
                        for failure in outcome["failures"]
                        if failure["code"] == "E_TRAIN_TASK_FAULT"
                    ),
                    None,
                )
                entry["hit"] = target_failure is not None
                entry["observed_code"] = (
                    target_failure["code"] if target_failure else None
                )
                entry["stage"] = "train"
                entry["detail"] = (
                    target_failure["message"] if target_failure else None
                )
                entry["isolated"] = (
                    target_failure is not None
                    and len(outcome["task_results"])
                    == len(manifest["tasks"]) - 1
                )
        except WorkbenchError as exc:
            entry["hit"] = exc.code == spec["expected_code"]
            entry["observed_code"] = exc.code
            entry["stage"] = exc.stage
            entry["detail"] = exc.message
        report_entries.append(entry)

    report = {
        "report": "injection_diagnostics",
        "total": len(report_entries),
        "hit_count": sum(1 for entry in report_entries if entry["hit"]),
        "entries": sorted(report_entries, key=lambda item: item["injection_id"]),
    }
    # Atomic write: temp file in same directory then os.replace (rule 8).
    write_json_atomic(os.path.join(report_root, "injection_report.json"), report)
    return report
