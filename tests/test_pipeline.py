from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import quant_pipeline.pipeline as pipeline_module
from quant_pipeline.errors import (
    E_CALIBRATION_INVALID,
    E_CONFIG_INVALID,
    E_LAYER_QUANTIZATION,
    E_POINT_BUDGET_PENDING,
    E_RUN_INTERRUPTED,
    E_SNAPSHOT_STALE,
    E_WEIGHT_PACK_INVALID,
    PipelineError,
)
from quant_pipeline.pipeline import QuantizationPipeline, run_pipeline


def test_full_pipeline_and_incremental_reuse(sample_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = run_pipeline(sample_dir, output)
    assert first["status"] == "completed"
    assert first["failures"] == []
    assert len(first["layers"]) == 8
    assert all(layer["status"] == "ok" for layer in first["layers"])
    for layer in first["layers"]:
        assert layer["calibration"]["groups"]
        assert layer["quantization"]["fixed_ints"]
        assert layer["quantization"]["dequantized"]
        assert layer["error"]["statistics"]["element_count"] > 0
        assert layer["distillation"]["summary"]["sample_count"] == 4
        assert layer["cache_keys"].keys() >= {
            "calibration",
            "quantization",
            "error_analysis",
            "distillation",
        }
    schedule_keys = [task["sort_key"] for task in first["schedule"]["tasks"]]
    assert schedule_keys == sorted(schedule_keys)
    assert first["schedule"]["max_workers"] == 4

    second = run_pipeline(sample_dir, output)
    assert second["result_fingerprint"] == first["result_fingerprint"]
    assert second["cache_ledger"]["recomputed"] == []
    assert second["cache_ledger"]["reused"] == sorted(layer["name"] for layer in first["layers"])
    assert second["snapshot"]["previous_status"] == "completed"


def test_fresh_build_matches_incremental(sample_dir: Path, tmp_path: Path) -> None:
    incremental_dir = tmp_path / "incremental"
    fresh_dir = tmp_path / "fresh"
    first = run_pipeline(sample_dir, incremental_dir)
    second = run_pipeline(sample_dir, incremental_dir)
    fresh = run_pipeline(sample_dir, fresh_dir)
    assert first["result_fingerprint"] == second["result_fingerprint"] == fresh["result_fingerprint"]
    assert fresh["cache_ledger"]["recomputed"] == sorted(layer["name"] for layer in fresh["layers"])
    assert second["cache_ledger"]["recomputed"] == []


def test_single_layer_edit_invalidates_layer_and_downstream_distillation(
    sample_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "incremental"
    before = run_pipeline(sample_dir, output)
    weights_path = sample_dir / "weights.jsonl"
    records = [json.loads(line) for line in weights_path.read_text(encoding="utf-8").splitlines()]
    target_index = next(i for i, record in enumerate(records) if record["layer"] == "block0.mlp.down")
    records[target_index]["weights"][2] += 0.031
    records[target_index]["version"] += 1
    weights_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )
    after = run_pipeline(sample_dir, output)
    stages = after["cache_ledger"]["stages"]
    assert "block0.attn.k" in stages["calibration"]["reused"]
    assert "block0.attn.q" in stages["quantization"]["reused"]
    assert "block0.mlp.down" in stages["calibration"]["recomputed"]
    assert "block0.mlp.down" in stages["quantization"]["recomputed"]
    assert "block1.attn.k" in stages["quantization"]["reused"]
    assert "block1.attn.k" in stages["distillation"]["recomputed"]
    assert "block0.attn.k" in stages["distillation"]["reused"]
    assert after["result_fingerprint"] != before["result_fingerprint"]
    fresh = run_pipeline(sample_dir, tmp_path / "fresh_after_edit")
    assert fresh["result_fingerprint"] == after["result_fingerprint"]


def test_layer_calibration_edit_invalidates_only_referenced_layer_chain(
    sample_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "calibration_edit"
    run_pipeline(sample_dir, output)
    calibration_path = sample_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["samples"][0]["inputs"]["block0.mlp.down"][1] += 0.05
    calibration_path.write_text(json.dumps(calibration, ensure_ascii=False), encoding="utf-8")
    after = run_pipeline(sample_dir, output)
    stages = after["cache_ledger"]["stages"]
    assert "block0.attn.k" in stages["calibration"]["reused"]
    assert "block0.mlp.down" in stages["calibration"]["recomputed"]
    assert "block0.attn.q" in stages["quantization"]["reused"]
    fresh = run_pipeline(sample_dir, tmp_path / "calibration_fresh")
    assert fresh["result_fingerprint"] == after["result_fingerprint"]


def test_profile_and_fractional_bits_switch_real_values(sample_dir: Path, tmp_path: Path) -> None:
    q8 = run_pipeline(sample_dir, tmp_path / "q8", profile="q8", fractional_bits=8)
    q6 = run_pipeline(sample_dir, tmp_path / "q6", profile="q6", fractional_bits=6)
    q6_fine = run_pipeline(sample_dir, tmp_path / "q6f", profile="q6", fractional_bits=10)
    q8_layer = next(layer for layer in q8["layers"] if layer["name"] == "block0.attn.k")
    q6_layer = next(layer for layer in q6["layers"] if layer["name"] == "block0.attn.k")
    q6f_layer = next(layer for layer in q6_fine["layers"] if layer["name"] == "block0.attn.k")
    assert q8_layer["quantization"]["target_bits"] == 8
    assert q6_layer["quantization"]["target_bits"] == 6
    assert q8_layer["quantization"]["fixed_ints"] != q6_layer["quantization"]["fixed_ints"]
    assert q8_layer["quantization"]["dequantized"] != q6_layer["quantization"]["dequantized"]
    assert q6_layer["quantization"]["fixed_ints"] != q6f_layer["quantization"]["fixed_ints"]
    assert q6_layer["quantization"]["dequantized"] != q6f_layer["quantization"]["dequantized"]


def test_point_budget_pending_then_resume_matches_fresh(sample_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "resume"
    partial = run_pipeline(sample_dir, output, point_budget=50, max_workers=2)
    assert partial["status"] == "partial"
    assert partial["pending"]
    assert all(item["code"] == E_POINT_BUDGET_PENDING for item in partial["pending"])
    assert partial["snapshot"]["status"] == "interrupted"
    resumed = run_pipeline(sample_dir, output, point_budget=1000, max_workers=4)
    assert resumed["status"] == "completed"
    assert resumed["pending"] == []
    assert resumed["snapshot"]["status"] == "completed"
    assert resumed["snapshot"]["recovery"] == "resumed"
    fresh = run_pipeline(sample_dir, tmp_path / "fresh_budget", point_budget=1000)
    assert resumed["result_fingerprint"] == fresh["result_fingerprint"]
    assert resumed["cache_ledger"]["reused"]


def test_graceful_stop_writes_interrupted_manifest(sample_dir: Path, tmp_path: Path) -> None:
    stop_event = threading.Event()
    stop_event.set()
    report = run_pipeline(sample_dir, tmp_path / "stopped", stop_event=stop_event)
    assert report["status"] == "partial"
    assert report["pending"]
    assert all(item["code"] == E_RUN_INTERRUPTED for item in report["pending"])
    manifest = json.loads((tmp_path / "stopped" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"


def test_stale_unfinished_snapshot_forces_full_rebuild(sample_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "stale"
    partial = run_pipeline(sample_dir, output, point_budget=50)
    assert partial["snapshot"]["status"] == "interrupted"
    assert partial["snapshot"]["recovery"] == "new"
    config_path = sample_dir / "config.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["distillation"]["temperature"] = 1.75
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    rebuilt = run_pipeline(sample_dir, output, point_budget=1000)
    assert rebuilt["snapshot"]["status"] == "completed"
    assert rebuilt["snapshot"]["recovery"] == "full"
    assert any(event["business_code"] == E_SNAPSHOT_STALE for event in rebuilt["events"])
    assert rebuilt["status"] == "completed"


def test_concurrency_does_not_change_result_fingerprint(sample_dir: Path, tmp_path: Path) -> None:
    one = run_pipeline(sample_dir, tmp_path / "one", max_workers=1)
    four = run_pipeline(sample_dir, tmp_path / "four", max_workers=4)
    assert one["result_fingerprint"] == four["result_fingerprint"]


def test_thread_pool_never_exceeds_configured_workers(
    sample_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    original = pipeline_module.calibrate_layer
    lock = threading.Lock()
    active = 0
    maximum = 0

    def wrapped(layer, config):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.01)
            return original(layer, config)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(pipeline_module, "calibrate_layer", wrapped)
    report = run_pipeline(sample_dir, tmp_path / "workers", max_workers=3)
    assert report["status"] == "completed"
    assert 1 <= maximum <= 3


def test_cached_layers_do_not_call_calculation_functions(
    sample_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "cached"
    first = run_pipeline(sample_dir, output)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cached layer was recomputed")

    monkeypatch.setattr(pipeline_module, "calibrate_layer", forbidden)
    monkeypatch.setattr(pipeline_module, "quantize_layer", forbidden)
    monkeypatch.setattr(pipeline_module, "analyze_errors", forbidden)
    monkeypatch.setattr(pipeline_module, "distill_layer", forbidden)
    second = run_pipeline(sample_dir, output)
    assert second["status"] == "completed"
    assert second["result_fingerprint"] == first["result_fingerprint"]
    assert second["cache_ledger"]["recomputed"] == []


def test_layer_stage_failure_is_isolated_and_excluded_from_error_stats(
    sample_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    original = pipeline_module.quantize_layer

    def flaky(layer, calibration, config):
        if layer.name == "block0.attn.q":
            raise PipelineError(
                E_LAYER_QUANTIZATION,
                "injected quantization failure",
                stage="quantization",
                layer=layer.name,
            )
        return original(layer, calibration, config)

    monkeypatch.setattr(pipeline_module, "quantize_layer", flaky)
    report = run_pipeline(sample_dir, tmp_path / "isolated", max_workers=2)
    assert report["status"] == "partial"
    failed = next(layer for layer in report["layers"] if layer["name"] == "block0.attn.q")
    assert failed["status"] == "failed"
    assert "error" not in failed
    assert any(
        item["layer"] == "block0.attn.q" and item["business_code"] == E_LAYER_QUANTIZATION
        for item in report["failures"]
    )
    assert sum(layer["status"] == "ok" for layer in report["layers"]) == 7


def test_all_injections_are_isolated_and_reported(sample_dir: Path, tmp_path: Path) -> None:
    report = run_pipeline(
        sample_dir,
        tmp_path / "injections",
        injections_path=sample_dir / "injections_all.yaml",
    )
    assert report["status"] in {"partial", "failed"}
    assert len(report["injections"]) == 6
    assert all(item["hit"] for item in report["injections"])
    codes = {item["injection_id"]: item["business_code"] for item in report["injections"]}
    assert codes == {
        "INJ-001": E_WEIGHT_PACK_INVALID,
        "INJ-002": E_WEIGHT_PACK_INVALID,
        "INJ-003": E_CONFIG_INVALID,
        "INJ-004": E_CALIBRATION_INVALID,
        "INJ-005": E_CONFIG_INVALID,
        "INJ-006": E_WEIGHT_PACK_INVALID,
    }
    failed_layers = {
        failure["layer"] for failure in report["failures"] if failure.get("layer") is not None
    }
    assert len(failed_layers) >= 4
    assert any(layer["status"] == "quantized_only" for layer in report["layers"])
