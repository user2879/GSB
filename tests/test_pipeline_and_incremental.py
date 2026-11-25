from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from conftest import EXAMPLES, cluster_projection, write_variant
from ins_diag.cli import canonical_analysis
from ins_diag.pipeline import run_pipeline


def test_default_pipeline_outputs_window_views_and_boundaries(tmp_path: Path) -> None:
    report = run_pipeline(EXAMPLES / "config.yaml", state_dir=tmp_path / "state")
    assert report["status"] == "ok"
    assert report["data"]["window_count"] == 43
    assert len(report["windows"]) == 43
    assert report["clusters"]["converged"] is True
    assert len(report["clusters"]["centroids_units"]) == 4
    assert report["snapshot"]["commit"]["committed"] is True
    assert report["boundary_windows"] == [
        "imu-alpha:gy:000096",
        "imu-alpha:gy:000128",
    ]
    for window in report["windows"]:
        assert window["cluster_id"] in {0, 1, 2, 3}
        assert window["density"] is not None
        assert "score_fixed_units" in window["density"]
        assert "boundary" in window["density"]
    for window in report["windows"]:
        if window["density"]["boundary"]:
            difference = abs(
                window["density"]["relative_permille"]
                - window["density"]["threshold_permille"]
            )
            assert difference <= window["density"]["quantization_half_step_permille"] + 1e-12
            assert window["density"]["reference"]["fractional_bits"] == (
                report["clusters"]["fixed_point"]["fractional_bits"] + 1
            )


def test_missing_samples_are_filled_from_window_seen_mean(tmp_path: Path) -> None:
    report = run_pipeline(EXAMPLES / "config.yaml", state_dir=tmp_path / "state")
    filled = [
        window
        for window in report["windows"]
        if window["fill_source"]["rule"] == "window_seen_mean"
    ]
    assert filled
    assert any(window["fill_source"]["missing_count"] > 0 for window in filled)
    for window in filled:
        assert window["fill_source"]["seen_count"] + window["fill_source"]["missing_count"] == 64
        assert all(
            0 <= index < 64 for index in window["fill_source"]["missing_indices"]
        )


def test_standardization_changes_when_participating_window_set_changes(
    tmp_path: Path,
) -> None:
    first = run_pipeline(
        EXAMPLES / "config.yaml",
        state_dir=tmp_path / "first",
        edits_limit=0,
    )
    replaced = run_pipeline(
        EXAMPLES / "config.yaml",
        state_dir=tmp_path / "replaced",
        edits_limit=1,
    )
    assert first["data"]["window_set_fingerprint"] != replaced["data"]["window_set_fingerprint"]
    assert first["standardization"]["means"] != replaced["standardization"]["means"]
    first_vectors = [window["standardized_vector"] for window in first["windows"]]
    replaced_vectors = [window["standardized_vector"] for window in replaced["windows"]]
    assert first_vectors != replaced_vectors


def test_quantization_and_fractional_bits_change_assignments_and_sse(
    tmp_path: Path, base_edits: dict
) -> None:
    config = {
        "schema_version": 1,
        "seed": 310,
        "window": {"length": 64, "shift": 32},
        "features": {"bands_hz": [[0.0, 8.0], [8.0, 24.0], [24.0, 64.0]]},
        "standardization": {"dispersion": "population_std"},
        "clustering": {
            "families": 3,
            "family_limit": 6,
            "max_iterations": 20,
            "fixed_point": {"fractional_bits": 10, "quantization_units": 2},
        },
        "kde": {"bandwidth": "0.20", "threshold_permille": 478},
        "runtime": {"max_workers": 4, "state_dir": "state"},
    }
    edits = copy.deepcopy(base_edits)

    def execute(name: str, bits: int, units: int) -> dict:
        variant = copy.deepcopy(edits)
        for command in variant["commands"]:
            if command["op"] == "set_fractional_bits":
                command["value"] = bits
            if command["op"] == "set_quantization_units":
                command["value"] = units
        config_path = write_variant(tmp_path / name, config, variant)
        return run_pipeline(config_path, state_dir=tmp_path / name / "state")

    base = execute("base", 10, 2)
    coarse = execute("coarse", 10, 4)
    wider = execute("wider", 12, 2)
    base_assignments = tuple(
        row["cluster_id"] for row in base["clusters"]["assignments"]
    )
    assert tuple(row["cluster_id"] for row in coarse["clusters"]["assignments"]) != base_assignments
    assert coarse["clusters"]["sse_units"] != base["clusters"]["sse_units"]
    assert wider["clusters"]["sse_units"] != base["clusters"]["sse_units"]
    assert wider["clusters"]["fixed_point"]["fractional_bits"] == 12
    assert coarse["clusters"]["fixed_point"]["quantization_units"] == 4


def test_kde_bandwidth_changes_scores(tmp_path: Path) -> None:
    common = {
        "schema_version": 1,
        "seed": 310,
        "window": {"length": 64, "shift": 32},
        "features": {"bands_hz": [[0.0, 8.0], [8.0, 24.0], [24.0, 64.0]]},
        "standardization": {"dispersion": "population_std"},
        "clustering": {
            "families": 3,
            "family_limit": 6,
            "max_iterations": 20,
            "fixed_point": {"fractional_bits": 10, "quantization_units": 2},
        },
        "kde": {"bandwidth": "0.15", "threshold_permille": 478},
        "runtime": {"max_workers": 3, "state_dir": "state"},
    }
    first_path = write_variant(tmp_path / "bw15", common, bandwidth="0.15")
    second_path = write_variant(tmp_path / "bw30", common, bandwidth="0.30")
    first = run_pipeline(first_path, state_dir=tmp_path / "bw15" / "state")
    second = run_pipeline(second_path, state_dir=tmp_path / "bw30" / "state")
    first_scores = [row["score_fixed_units"] for row in first["density"]["window_scores"]]
    second_scores = [row["score_fixed_units"] for row in second["density"]["window_scores"]]
    assert first_scores != second_scores
    assert first["density"]["bandwidth"] != second["density"]["bandwidth"]


def test_incremental_sequence_matches_clean_full_rebuild(tmp_path: Path) -> None:
    incremental_state = tmp_path / "incremental"
    config_path = EXAMPLES / "config.yaml"
    from ins_diag.config import load_config

    command_count = len(load_config(config_path).edits)
    for limit in range(command_count + 1):
        report = run_pipeline(config_path, state_dir=incremental_state, edits_limit=limit)
        assert report["status"] in {"ok", "partial"}
    incremental = run_pipeline(
        config_path, state_dir=incremental_state, edits_limit=command_count
    )
    full = run_pipeline(
        config_path, state_dir=tmp_path / "full", edits_limit=command_count
    )
    assert canonical_analysis(incremental) == canonical_analysis(full)
    assert incremental["cache"]["recomputed_count"] < full["cache"]["recomputed_count"]
    assert incremental["cache"]["reused_count"] == 87


def test_concurrency_degree_does_not_change_analysis(tmp_path: Path) -> None:
    config = {
        "schema_version": 1,
        "seed": 310,
        "waveform_path": str((EXAMPLES / "waveform.yaml").resolve()),
        "edits_path": str((EXAMPLES / "edits.yaml").resolve()),
        "lease_declaration_path": str((EXAMPLES / "lease.yaml").resolve()),
        "window": {"length": 64, "shift": 32},
        "features": {"bands_hz": [[0.0, 8.0], [8.0, 24.0], [24.0, 64.0]]},
        "standardization": {"dispersion": "population_std"},
        "clustering": {
            "families": 3,
            "family_limit": 6,
            "max_iterations": 25,
            "fixed_point": {"fractional_bits": 12, "quantization_units": 4},
        },
        "kde": {"bandwidth": "0.20", "threshold_permille": 478},
        "runtime": {"max_workers": 1, "state_dir": "state"},
    }
    serial_path = write_variant(tmp_path / "serial", config, max_workers=1)
    parallel_path = write_variant(tmp_path / "parallel", config, max_workers=4)
    serial = run_pipeline(serial_path, state_dir=tmp_path / "serial" / "state")
    parallel = run_pipeline(parallel_path, state_dir=tmp_path / "parallel" / "state")
    assert cluster_projection(serial) == cluster_projection(parallel)
    assert serial["workers"]["max_workers"] == 1
    assert parallel["workers"]["max_workers"] == 4


def test_repeated_run_reuses_raw_features_statistics_and_observations(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    first = run_pipeline(EXAMPLES / "config.yaml", state_dir=state)
    second = run_pipeline(EXAMPLES / "config.yaml", state_dir=state)
    assert first["cache"]["reused_count"] == 0
    assert second["cache"]["reused_count"] == 87
    assert second["cache"]["recomputed_count"] == 3
    assert canonical_analysis(first) == canonical_analysis(
        run_pipeline(EXAMPLES / "config.yaml", state_dir=tmp_path / "clean")
    )


def test_single_window_replacement_reuses_unaffected_raw_features(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    run_pipeline(EXAMPLES / "config.yaml", state_dir=state, edits_limit=0)
    replaced = run_pipeline(EXAMPLES / "config.yaml", state_dir=state, edits_limit=1)
    raw_events = [
        event
        for event in replaced["cache"]["events"]
        if event["kind"] == "raw_window_feature"
    ]
    raw_reused = [event for event in raw_events if event["outcome"] == "reused"]
    raw_recomputed = [event for event in raw_events if event["outcome"] == "recomputed"]
    assert len(raw_reused) == 39
    assert len(raw_recomputed) == 3
    stats_events = [
        event
        for event in replaced["cache"]["events"]
        if event["kind"] == "standardization_stats"
    ]
    assert stats_events[0]["outcome"] == "recomputed"
    observation_events = [
        event
        for event in replaced["cache"]["events"]
        if event["kind"] == "standardized_observation"
    ]
    assert len(observation_events) == 42
    assert all(event["outcome"] == "recomputed" for event in observation_events)
