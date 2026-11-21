"""End-to-end local verification demo and acceptance evidence generator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, read_json
from .compare import compare_reports
from .generator import (
    Fixture,
    generate_fixture,
    replace_sequence_values,
    rewrite_config,
)
from .pipeline import PipelineOptions, run_pipeline


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _copy_dataset(source: Path, destination: Path) -> None:
    _reset(destination)
    shutil.copytree(source, destination)


def _run(
    config_path: Path,
    state_dir: Path,
    report_path: Path,
    *,
    reset: bool = False,
    stop_after_stage: str | None = None,
) -> dict[str, Any]:
    if reset:
        _reset(state_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(
        PipelineOptions(
            config_path=config_path,
            state_dir=state_dir,
            report_path=report_path,
            stop_after_stage=stop_after_stage,
        )
    )
    if result.exit_code != 0:
        raise RuntimeError(
            f"pipeline returned {result.exit_code}; see {result.report_path}"
        )
    return result.report


def _code_set(report: dict[str, Any]) -> list[str]:
    return sorted({failure["code"] for failure in report.get("failures", [])})


def _incremental_step(
    name: str,
    report: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    audit = report.get("incremental_audit", {})
    invalidation = audit.get("invalidation", {})
    return {
        "name": name,
        "report": str(report_path),
        "snapshot_status": report.get("snapshot", {}).get("status_code"),
        "invalidated_task_count": invalidation.get("invalidated_count"),
        "unchanged_task_count": invalidation.get("unchanged_count"),
        "training_reused": len(audit.get("training_reused", [])),
        "training_computed": len(audit.get("training_computed", [])),
        "prediction_reused": len(audit.get("prediction_reused", [])),
        "prediction_computed": len(audit.get("prediction_computed", [])),
        "metric_reused": len(audit.get("metric_reused", [])),
        "metric_computed": len(audit.get("metric_computed", [])),
        "failure_codes": _code_set(report),
    }


def run_demo(root: Path, *, seed: int = 20261122, force: bool = False) -> dict[str, Any]:
    root = root.resolve()
    if force:
        _reset(root)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    fixture: Fixture = generate_fixture(data_dir, seed=seed, force=True)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    incremental_state = root / "state_incremental"

    reports: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []

    initial_path = reports_dir / "01_initial.json"
    reports["initial"] = _run(
        fixture.config_path,
        incremental_state,
        initial_path,
        reset=True,
    )
    steps.append(_incremental_step("initial_full", reports["initial"], initial_path))

    replace_sequence_values(fixture.samples_path, "s-beta", seed=seed)
    sequence_path = reports_dir / "02_incremental_sequence_replace.json"
    reports["sequence_replace"] = _run(
        fixture.config_path,
        incremental_state,
        sequence_path,
    )
    steps.append(
        _incremental_step("incremental_sequence_replace", reports["sequence_replace"], sequence_path)
    )

    rewrite_config(
        fixture.config_path,
        models={
            "moving_average_window": 4,
            "feature_lag_order": 4,
            "autoregressive_regularization": 1.0e-5,
        },
    )
    model_path = reports_dir / "03_incremental_model_params.json"
    reports["model_params"] = _run(
        fixture.config_path,
        incremental_state,
        model_path,
    )
    steps.append(
        _incremental_step("incremental_model_params", reports["model_params"], model_path)
    )

    rewrite_config(fixture.config_path, quantization={"quantization_level": 2})
    quantization_path = reports_dir / "04_incremental_quantization.json"
    reports["quantization"] = _run(
        fixture.config_path,
        incremental_state,
        quantization_path,
    )
    steps.append(
        _incremental_step("incremental_quantization", reports["quantization"], quantization_path)
    )

    rewrite_config(fixture.config_path, backtest={"window_length": 10, "step": 3})
    final_incremental_path = reports_dir / "05_incremental_window_params.json"
    reports["final_incremental"] = _run(
        fixture.config_path,
        incremental_state,
        final_incremental_path,
    )
    steps.append(
        _incremental_step(
            "incremental_window_params",
            reports["final_incremental"],
            final_incremental_path,
        )
    )

    final_dataset = root / "final_dataset"
    _copy_dataset(data_dir, final_dataset)
    full_state = root / "state_full"
    full_path = reports_dir / "06_full_rebuild.json"
    reports["full"] = _run(
        final_dataset / "config.yaml",
        full_state,
        full_path,
        reset=True,
    )
    full_parity = compare_reports(reports["final_incremental"], reports["full"])
    atomic_write_json(reports_dir / "07_parity_incremental_vs_full.json", full_parity)
    if not full_parity["equal"]:
        raise RuntimeError("incremental and full rebuild reports are not equal")

    repeat_state = root / "state_repeat"
    _reset(repeat_state)
    shutil.copytree(incremental_state, repeat_state)
    repeat_path = reports_dir / "08_repeat_incremental.json"
    reports["repeat"] = _run(
        fixture.config_path,
        repeat_state,
        repeat_path,
    )
    repeat_parity = compare_reports(reports["repeat"], reports["full"])
    atomic_write_json(reports_dir / "09_parity_repeat_vs_full.json", repeat_parity)
    if not repeat_parity["equal"]:
        raise RuntimeError("repeat incremental report and full rebuild are not equal")

    interrupted_state = root / "state_interrupted"
    interrupted_path = reports_dir / "10_interrupted_after_windows.json"
    reports["interrupted"] = _run(
        final_dataset / "config.yaml",
        interrupted_state,
        interrupted_path,
        reset=True,
        stop_after_stage="windows",
    )
    resume_path = reports_dir / "11_resumed_after_interrupt.json"
    reports["resumed"] = _run(
        final_dataset / "config.yaml",
        interrupted_state,
        resume_path,
    )
    resume_parity = compare_reports(reports["resumed"], reports["full"])
    atomic_write_json(reports_dir / "12_parity_resumed_vs_full.json", resume_parity)
    if not resume_parity["equal"]:
        raise RuntimeError("snapshot-resumed report and full rebuild are not equal")

    concurrency_dataset = root / "concurrency_dataset"
    _copy_dataset(final_dataset, concurrency_dataset)
    rewrite_config(concurrency_dataset / "config.yaml", scheduler={"concurrency": 1})
    concurrency_state = root / "state_concurrency1"
    concurrency_path = reports_dir / "13_concurrency_1.json"
    reports["concurrency1"] = _run(
        concurrency_dataset / "config.yaml",
        concurrency_state,
        concurrency_path,
        reset=True,
    )
    concurrency_parity = compare_reports(reports["concurrency1"], reports["full"])
    atomic_write_json(reports_dir / "14_parity_concurrency_1_vs_4.json", concurrency_parity)
    if not concurrency_parity["equal"]:
        raise RuntimeError("concurrency=1 report and concurrency=4 report are not equal")

    quota_dataset = root / "quota_dataset"
    _copy_dataset(final_dataset, quota_dataset)
    rewrite_config(
        quota_dataset / "config.yaml",
        scheduler={"per_series_point_quota": 16, "batch_point_budget": 500000},
    )
    quota_state = root / "state_quota"
    quota_path = reports_dir / "15_quota_exhausted.json"
    reports["quota"] = _run(
        quota_dataset / "config.yaml",
        quota_state,
        quota_path,
        reset=True,
    )

    budget_dataset = root / "budget_dataset"
    _copy_dataset(final_dataset, budget_dataset)
    rewrite_config(
        budget_dataset / "config.yaml",
        scheduler={"per_series_point_quota": 500000, "batch_point_budget": 20},
    )
    budget_state = root / "state_budget"
    budget_path = reports_dir / "16_budget_exhausted.json"
    reports["budget"] = _run(
        budget_dataset / "config.yaml",
        budget_state,
        budget_path,
        reset=True,
    )

    injection_hits = reports["full"]["injections"]["hits"]
    injection_mapping = {
        hit["injection_id"]: hit["business_code"] for hit in injection_hits
    }
    summary = {
        "status": "ok",
        "seed": seed,
        "reports_dir": str(reports_dir),
        "final_dataset": str(final_dataset),
        "incremental_steps": steps,
        "parity": {
            "incremental_vs_full": full_parity,
            "repeat_vs_full": repeat_parity,
            "resumed_vs_full": resume_parity,
            "concurrency_1_vs_4": concurrency_parity,
        },
        "injection_mapping": injection_mapping,
        "injection_unhit_ids": reports["full"]["injections"]["unhit_injection_ids"],
        "snapshot_interrupt": {
            "status": reports["interrupted"]["status"],
            "exit_code": reports["interrupted"]["exit_code"],
            "resume_parity": resume_parity["equal"],
        },
        "quota_failure_codes": _code_set(reports["quota"]),
        "budget_failure_codes": _code_set(reports["budget"]),
        "full_failure_codes": _code_set(reports["full"]),
        "full_summary": reports["full"]["summary"],
        "model_aggregates": {
            model_name: payload.get("aggregate")
            for model_name, payload in reports["full"]["models"].items()
        },
    }
    if "E_SERIES_QUOTA_EXHAUSTED" not in summary["quota_failure_codes"]:
        raise RuntimeError("quota verification did not emit E_SERIES_QUOTA_EXHAUSTED")
    if "E_POINT_BUDGET_EXHAUSTED" not in summary["budget_failure_codes"]:
        raise RuntimeError("budget verification did not emit E_POINT_BUDGET_EXHAUSTED")
    atomic_write_json(root / "demo_summary.json", summary)
    return summary
