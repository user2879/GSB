from __future__ import annotations

import json
from pathlib import Path

import yaml

from .helpers import (
    SAMPLE_CONFIG,
    SAMPLE_SEQUENCES,
    assert_canonical_equal,
    event_actions,
    replace_sequence,
    run_pipeline,
    write_config,
)


def _damping_config(path: Path) -> Path:
    return write_config(path, lambda raw: raw["motion_model"].__setitem__("damping", 0.7))


def _quantization_config(path: Path, base: Path) -> Path:
    source = yaml.safe_load(base.read_text(encoding="utf-8"))
    source["quantization_levels"] = [2, 4, 8]
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def _horizon_config(path: Path, base: Path) -> Path:
    source = yaml.safe_load(base.read_text(encoding="utf-8"))
    source["prediction_steps"] = 3
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def test_exact_stage_invalidation_and_reuse(tmp_path: Path) -> None:
    state = tmp_path / "state"
    config = write_config(tmp_path / "config.yaml")
    first = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-first",
        fresh=True,
    )
    assert first.diagnostic_report["stage_cache_totals"] == {"hit": 0, "recompute": 8}

    rerun = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-rerun",
    )
    assert rerun.diagnostic_report["stage_cache_totals"] == {"hit": 8, "recompute": 0}

    replaced_sequences = replace_sequence(
        SAMPLE_SEQUENCES,
        tmp_path / "sequences-replaced.jsonl",
        sequence_index=1,
        mutate=lambda sequence: sequence["frames"][0]["trajectories"][0].__setitem__(
            "color", [1, 2, 3]
        ),
    )
    replaced = run_pipeline(
        config_path=config,
        sequence_path=replaced_sequences,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-replaced",
    )
    actions = event_actions(replaced)
    for stage in ("encode", "motion", "predict", "decode"):
        assert actions[("seq-0001", stage)] == "HIT"
        assert actions[("seq-0002", stage)] == "RECOMPUTE"
    assert replaced.diagnostic_report["stage_cache_totals"] == {"hit": 4, "recompute": 4}

    damping_config = _damping_config(tmp_path / "config-damping.yaml")
    damping = run_pipeline(
        config_path=damping_config,
        sequence_path=replaced_sequences,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-damping",
    )
    assert damping.diagnostic_report["stage_cache"] == {
        "encode": {"hit": 2, "recompute": 0},
        "motion": {"hit": 2, "recompute": 0},
        "predict": {"hit": 0, "recompute": 2},
        "decode": {"hit": 0, "recompute": 2},
    }
    assert any(
        event["business_code"] == "E_SNAPSHOT_STALE"
        and event["reason"] == "CONFIG_HASH_MISMATCH"
        for event in damping.diagnostic_report["snapshot"]["stale_events"]
    )

    quant_config = _quantization_config(tmp_path / "config-quant.yaml", damping_config)
    quant = run_pipeline(
        config_path=quant_config,
        sequence_path=replaced_sequences,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-quant",
    )
    assert quant.diagnostic_report["stage_cache"]["encode"]["hit"] == 2
    assert quant.diagnostic_report["stage_cache"]["motion"]["hit"] == 2
    assert quant.diagnostic_report["stage_cache"]["predict"]["recompute"] == 2
    assert quant.diagnostic_report["stage_cache"]["decode"]["recompute"] == 2

    horizon_config = _horizon_config(tmp_path / "config-horizon.yaml", quant_config)
    horizon = run_pipeline(
        config_path=horizon_config,
        sequence_path=replaced_sequences,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-horizon",
    )
    assert horizon.decision_record["metrics"]["prediction_steps"] == 3
    assert horizon.diagnostic_report["stage_cache"]["encode"]["hit"] == 2
    assert horizon.diagnostic_report["stage_cache"]["motion"]["hit"] == 2
    assert horizon.diagnostic_report["stage_cache"]["predict"]["recompute"] == 2
    assert horizon.diagnostic_report["stage_cache"]["decode"]["recompute"] == 2


def test_incremental_edit_sequence_equals_blank_full_rebuild(tmp_path: Path) -> None:
    state = tmp_path / "incremental-state"
    config = write_config(tmp_path / "config.yaml")
    run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "initial",
        fresh=True,
    )

    replaced_sequences = replace_sequence(
        SAMPLE_SEQUENCES,
        tmp_path / "edited.jsonl",
        sequence_index=1,
        mutate=lambda sequence: sequence["frames"][2]["trajectories"][0].__setitem__(
            "x", sequence["frames"][2]["trajectories"][0]["x"] + 2.75
        ),
    )
    run_pipeline(
        config_path=config,
        sequence_path=replaced_sequences,
        tier=2,
        state_dir=state,
        output_dir=tmp_path / "edit-tier-2",
    )

    edited_config = _damping_config(tmp_path / "config-damping.yaml")
    source = yaml.safe_load(edited_config.read_text(encoding="utf-8"))
    source["prediction_steps"] = 3
    edited_config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    run_pipeline(
        config_path=edited_config,
        sequence_path=replaced_sequences,
        tier=4,
        state_dir=state,
        output_dir=tmp_path / "edit-tier-4",
    )

    final_config = _quantization_config(tmp_path / "config-final.yaml", edited_config)
    incremental = run_pipeline(
        config_path=final_config,
        sequence_path=replaced_sequences,
        tier=2,
        state_dir=state,
        output_dir=tmp_path / "incremental-final",
    )
    full = run_pipeline(
        config_path=final_config,
        sequence_path=replaced_sequences,
        tier=2,
        state_dir=tmp_path / "blank-state",
        output_dir=tmp_path / "full-final",
        fresh=True,
    )

    assert_canonical_equal(incremental, full)
    assert incremental.diagnostic_report["stage_cache_totals"]["hit"] > 0
    assert full.diagnostic_report["stage_cache_totals"]["hit"] == 0
    assert incremental.predictions == full.predictions
    assert incremental.cache_ledger == full.cache_ledger


def test_thread_pool_parallel_and_serial_results_match(tmp_path: Path) -> None:
    serial_config = write_config(
        tmp_path / "config-serial.yaml",
        lambda raw: raw.__setitem__("thread_pool_workers", 1),
    )
    parallel_config = write_config(
        tmp_path / "config-parallel.yaml",
        lambda raw: raw.__setitem__("thread_pool_workers", 4),
    )
    serial = run_pipeline(
        config_path=serial_config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=2,
        state_dir=tmp_path / "state-serial",
        output_dir=tmp_path / "out-serial",
        fresh=True,
    )
    parallel = run_pipeline(
        config_path=parallel_config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=2,
        state_dir=tmp_path / "state-parallel",
        output_dir=tmp_path / "out-parallel",
        fresh=True,
    )

    assert serial.predictions == parallel.predictions
    assert serial.decision_record == parallel.decision_record
    assert serial.cache_ledger == parallel.cache_ledger
    assert serial.failures == parallel.failures
    assert serial.diagnostic_report["thread_pool"]["observed_max_concurrency"] <= 1
    assert parallel.diagnostic_report["thread_pool"]["observed_max_concurrency"] <= 4
    assert parallel.diagnostic_report["thread_pool"]["observed_max_concurrency"] >= 2

