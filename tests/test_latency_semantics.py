from __future__ import annotations

import time
from pathlib import Path

from quant_workbench.engine import WorkbenchEngine
from quant_workbench.loading import load_calibration, load_commands, load_config, load_weight_pack


def engine_with_latency(
    sample_paths: dict[str, Path], *, budget_ms: float, delay_ms: float, point_budget: int = 1_000_000
) -> WorkbenchEngine:
    config = load_config(sample_paths["config"])
    config.latency_budget_ms = budget_ms
    config.worker_delay_ms = delay_ms
    config.point_batch_budget = point_budget
    weights, w_issues = load_weight_pack(sample_paths["weights"])
    calibration, c_issues = load_calibration(sample_paths["calibration"])
    return WorkbenchEngine(
        config,
        weights,
        calibration,
        base_issues=w_issues + c_issues,
        command_base_dir=sample_paths["commands"].parent,
    )


def test_stale_summary_returns_within_budget_and_never_claims_latest(
    sample_paths: dict[str, Path],
) -> None:
    engine = engine_with_latency(sample_paths, budget_ms=12, delay_ms=40)
    try:
        commands, _ = load_commands(sample_paths["commands"])
        started = time.perf_counter()
        stale = engine.process_command(commands[0])
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert elapsed_ms <= 60  # local wall clock; cooperative cancellation leaves headroom
        assert stale["status"] == "stale"
        assert stale["is_latest"] is False
        assert stale["result_generation"] == 0
        assert stale["requested_generation"] == 1
        assert "E_RESULT_STALE" in stale["codes"]
        assert stale["within_budget"] is True
        assert stale["pending_layers"]
    finally:
        engine.close()


def test_latest_completed_generation_is_used_while_new_generation_runs(
    sample_paths: dict[str, Path],
) -> None:
    engine = engine_with_latency(sample_paths, budget_ms=1_000, delay_ms=0)
    commands, _ = load_commands(sample_paths["commands"])
    try:
        initial = engine.process_command(commands[0])
        assert initial["status"] == "complete"
        complete_generation = initial["result_generation"]
        prior_max = initial["max_abs_error"]
        prior_mse = initial["mse"]

        engine.config.latency_budget_ms = 10
        engine.config.worker_delay_ms = 40
        stale = engine.process_command(commands[4])  # set_temperature
        assert stale["status"] == "stale"
        assert stale["is_latest"] is False
        assert stale["result_generation"] == complete_generation
        assert stale["max_abs_error"] == prior_max
        assert stale["mse"] == prior_mse
        assert stale["progress"]["completed"] < stale["progress"]["total"]

        engine.config.latency_budget_ms = 1_000
        engine.config.worker_delay_ms = 0
        final = engine.drain_all()
        assert final["status"] == "complete"
        assert final["is_latest"] is True
        assert final["requested_generation"] == final["result_generation"]
    finally:
        engine.close()


def test_point_budget_pending_is_explicit(sample_paths: dict[str, Path]) -> None:
    engine = engine_with_latency(
        sample_paths, budget_ms=1_000, delay_ms=0, point_budget=64
    )
    try:
        engine.drain_all()
        commands, _ = load_commands(sample_paths["commands"])
        response = engine.process_command(commands[3])  # replace only layer.4 (12 points)
        assert response["status"] == "complete"
        response = engine.process_command(commands[0])
        assert "E_POINT_BUDGET_PENDING" in response["codes"]
        assert response["status"] == "stale"
        assert engine.diagnostics()["scheduling"]["batches"]
    finally:
        engine.close()
