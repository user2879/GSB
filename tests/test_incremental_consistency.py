from __future__ import annotations

import copy
from pathlib import Path

from quant_workbench.engine import WorkbenchEngine
from quant_workbench.loading import load_calibration, load_commands, load_config, load_weight_pack
from quant_workbench.util import canonical_json


def make_engine(sample_paths: dict[str, Path], *, snapshot_store=None) -> WorkbenchEngine:
    config = load_config(sample_paths["config"])
    config.latency_budget_ms = 10_000
    config.point_batch_budget = 1_000_000
    config.worker_delay_ms = 0.0
    weights, weight_issues = load_weight_pack(sample_paths["weights"])
    calibration, calibration_issues = load_calibration(sample_paths["calibration"])
    return WorkbenchEngine(
        config,
        weights,
        calibration,
        snapshot_store=snapshot_store,
        base_issues=weight_issues + calibration_issues,
        command_base_dir=sample_paths["commands"].parent,
    )


def test_incremental_matches_full_rebuild_for_all_six_command_types(
    sample_paths: dict[str, Path],
) -> None:
    engine = make_engine(sample_paths)
    try:
        commands, issues = load_commands(sample_paths["commands"])
        assert not issues
        for command in commands:
            response = engine.process_command(command)
            assert response["status"] == "complete"
            assert response["is_latest"] is True
        engine.drain_all()
        incremental = engine.result_payload()
        assert incremental["summary"]["completed_layers"] == 5
        assert incremental["failures"] == []
        for layer in incremental["layers"]:
            assert layer["evaluation"]["correction_accumulation_order"] == [
                sample["sample_id"] for sample in layer["evaluation"]["samples"]
            ]
            assert layer["evaluation"]["samples"][0]["accumulation_order"] == list(
                range(len(layer["evaluation"]["samples"][0]["original_output"]))
            )

        rebuilt = engine.full_rebuild()
        assert canonical_json(incremental) == canonical_json(rebuilt)
        assert [row["layer"] for row in rebuilt["layers"]] == sorted(
            row["layer"] for row in rebuilt["layers"]
        )
    finally:
        engine.close()


def test_only_affected_layer_and_downstream_are_recomputed(
    sample_paths: dict[str, Path],
) -> None:
    engine = make_engine(sample_paths)
    try:
        engine.drain_all()
        layer0_quant = engine._cache["quantization"]["layer.0"]
        layer4_quant = engine._cache["quantization"]["layer.4"]
        before = engine.diagnostics()["cache_stats"]["misses"]

        command = load_commands(sample_paths["commands"])[0][0]  # set_bits layer.1
        response = engine.process_command(command)
        assert response["status"] == "complete"
        after = engine.diagnostics()["cache_stats"]["misses"]
        new_misses = after[len(before) :]
        missed_quant_layers = {
            entry["layer"] for entry in new_misses if entry["stage"] == "quantization"
        }
        assert missed_quant_layers == {"layer.1", "layer.2", "layer.3", "layer.4"}
        assert engine._cache["quantization"]["layer.0"] is layer0_quant
        assert engine._cache["quantization"]["layer.4"] is not layer4_quant

        last_command_entries = [
            entry
            for entry in engine.diagnostics()["invalidation_reuse_ledger"]
            if entry["command_index"] == command.command_index
        ]
        assert [entry["layer"] for entry in last_command_entries] == sorted(
            entry["layer"] for entry in last_command_entries
        )
        actions = {entry["layer"]: entry["action"] for entry in last_command_entries}
        assert actions == {
            "layer.0": "reuse",
            "layer.1": "recompute",
            "layer.2": "recompute",
            "layer.3": "recompute",
            "layer.4": "recompute",
        }
    finally:
        engine.close()


def test_temperature_reuses_quantization_but_invalidates_evaluation(
    sample_paths: dict[str, Path],
) -> None:
    engine = make_engine(sample_paths)
    try:
        engine.drain_all()
        quant_before = {layer: payload for layer, payload in engine._cache["quantization"].items()}
        before = engine.diagnostics()["cache_stats"]["misses"]
        command = load_commands(sample_paths["commands"])[0][4]  # set_temperature
        response = engine.process_command(command)
        assert response["status"] == "complete"
        after = engine.diagnostics()["cache_stats"]["misses"]
        new_misses = after[len(before) :]
        assert all(entry["stage"] != "quantization" for entry in new_misses)
        assert {entry["layer"] for entry in new_misses if entry["stage"] == "evaluation"} == {
            f"layer.{index}" for index in range(5)
        }
        for layer, payload in quant_before.items():
            assert engine._cache["quantization"][layer] is payload
    finally:
        engine.close()


def test_thread_count_does_not_change_stable_computational_output(
    sample_paths: dict[str, Path],
) -> None:
    engine = make_engine(sample_paths)
    try:
        commands, _ = load_commands(sample_paths["commands"])
        for command in commands:
            engine.process_command(command)
        engine.drain_all()
        final_config = copy.deepcopy(engine.config)
        final_weights = {name: weight.copy() for name, weight in engine.state.weights.items()}
        final_calibration = engine.state.calibration.copy()

        results = []
        for thread_count in (1, 3):
            config = copy.deepcopy(final_config)
            config.max_threads = thread_count
            config.thread_limit = max(8, thread_count)
            config.latency_budget_ms = 10_000
            config.point_batch_budget = 1_000_000
            candidate = WorkbenchEngine.from_state(
                config, final_weights, final_calibration
            )
            try:
                candidate.drain_all()
                payload = candidate.result_payload()
            finally:
                candidate.close()
            results.append(
                {
                    "layers": payload["layers"],
                    "deployment": payload["deployment"],
                    "scheduling": payload["scheduling"],
                    "failures": payload["failures"],
                }
            )
        assert canonical_json(results[0]) == canonical_json(results[1])
    finally:
        engine.close()


def test_fractional_width_change_reuses_calibration_stage_only(
    sample_paths: dict[str, Path],
) -> None:
    from quant_workbench.models import Command

    engine = make_engine(sample_paths)
    try:
        engine.drain_all()
        calibration_before = engine._cache["calibration"]["layer.0"]
        quant_before = engine._cache["quantization"]["layer.0"]
        before_misses = engine.diagnostics()["cache_stats"]["misses"]
        command = Command(
            99,
            99,
            "set_fractional_bits",
            "layer.0",
            {"fractional_bits": 2},
            {"cmd": "set_fractional_bits"},
        )
        response = engine.process_command(command)
        assert response["status"] == "complete"
        new_misses = engine.diagnostics()["cache_stats"]["misses"][len(before_misses) :]
        assert all(entry["stage"] != "calibration" for entry in new_misses)
        assert engine._cache["calibration"]["layer.0"] is calibration_before
        assert engine._cache["quantization"]["layer.0"] is not quant_before
        ledger = [
            entry
            for entry in engine.diagnostics()["invalidation_reuse_ledger"]
            if entry["command_index"] == 99 and entry["layer"] == "layer.0"
        ][0]
        assert ledger["stages"]["calibration"] == "reuse"
        assert ledger["stages"]["quantization"] == "recompute"
    finally:
        engine.close()
