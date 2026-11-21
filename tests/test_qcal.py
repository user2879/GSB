from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from qcal.acceptance import run_acceptance
from qcal.calibration import (
    V1_BITS_WIDTH,
    V1_LINE_WIDTH,
    V1_SCALE_WIDTH,
    V1_ZERO_WIDTH,
    downgrade_v2_to_v1,
    parse_calibration_file,
    reverse_downgrade,
    upgrade_v1_to_v2,
)
from qcal.codes import (
    E_CALIB_PARSE,
    E_SIDECAR_MISSING,
    E_WEIGHT_NONFINITE,
)
from qcal.distillation import distill_layer
from qcal.edits import replacement_weights
from qcal.engine import ForcedInterrupt, QcalEngine, RunPaths, prepare_inputs, snapshot_summary
from qcal.inputs import operation_apply
from qcal.models import GroupQuantParam, LayerQuantResult, WeightPack
from qcal.quantile import calibrate_layer, sorted_quantile
from qcal.quantize import quantize_layer
from qcal.io_utils import array_fingerprint, combined_fingerprint


SAMPLES = ROOT / "samples"
WEIGHT_LAYERS = {"layer.00", "layer.01", "layer.02", "layer.03"}


def _pack(weights: np.ndarray, layer: str = "layer.test") -> WeightPack:
    return WeightPack(
        layer=layer,
        shape=tuple(weights.shape),
        weights=np.asarray(weights, dtype=np.float64),
        content_version="test-1",
        content_fingerprint=combined_fingerprint(array_fingerprint(weights), "test-1"),
    )


def _base_config() -> dict:
    return {
        "exchange_version": "v1",
        "calibration": {
            "quantile_lower": 0.02,
            "quantile_upper": 0.98,
            "interpolation": "linear",
            "tie_break": "lower_index",
            "scheme": "asymmetric",
            "per_channel": False,
            "group_size": 0,
        },
        "fixed_point": {"fraction_bits": 10, "rounding_mode": "nearest_even", "saturation": "saturate"},
        "distillation": {"temperature": 1.5, "soft_label_weight": 0.35},
        "scheduler": {"workers": 4, "quant_points_budget": 0},
        "layers": {},
        "full_precision_layers": [],
    }


def test_v1_fixed_width_and_v2_named_fields_parse() -> None:
    text = (SAMPLES / "calibration_v1.txt").read_text(encoding="utf-8")
    data_line = [line for line in text.splitlines() if line.startswith("layer.00")][0]
    assert len(data_line) == V1_LINE_WIDTH
    assert int(data_line[-(V1_BITS_WIDTH + V1_ZERO_WIDTH + V1_SCALE_WIDTH) : -(V1_BITS_WIDTH + V1_ZERO_WIDTH)].strip()) == 8
    assert int(data_line[-V1_BITS_WIDTH:].strip()) == 8

    parsed_v1 = parse_calibration_file(SAMPLES / "calibration_v1.txt", "v1", WEIGHT_LAYERS)
    parsed_v2 = parse_calibration_file(SAMPLES / "calibration_v2.json", "v2", WEIGHT_LAYERS)
    assert parsed_v1.records["layer.00"].scale == pytest.approx(0.008)
    assert parsed_v1.records["layer.02"].bit_width == 4
    assert parsed_v2.records["layer.01"].per_channel is True
    assert parsed_v2.records["layer.01"].rounding_mode == "nearest_away"
    assert parsed_v2.records["layer.03"].precision_label == "int8-outlier"
    assert not parsed_v1.failures and not parsed_v2.failures


def test_calibration_validation_injection_has_business_code() -> None:
    injections = [{"id": "T-ZERO", "type": "zero_scale", "layer": "layer.02"}]
    parsed = parse_calibration_file(
        SAMPLES / "calibration_v1.txt", "v1", WEIGHT_LAYERS, injections
    )
    failures = [failure for failure in parsed.failures if failure.layer == "layer.02"]
    assert failures and all(failure.code == E_CALIB_PARSE for failure in failures)
    assert failures[0].injection_id == "T-ZERO"
    assert "layer.02" not in parsed.records


def test_v2_declared_exchange_version_is_validated(tmp_path: Path) -> None:
    payload = json.loads((SAMPLES / "calibration_v2.json").read_text(encoding="utf-8"))
    payload["exchange_version"] = "v1"
    path = tmp_path / "bad_v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = parse_calibration_file(path, "v2", WEIGHT_LAYERS)
    assert any("exchange_version" in failure.message for failure in parsed.failures)
    assert not parsed.records


def test_downgrade_sidecar_round_trip_and_missing_error(tmp_path: Path) -> None:
    prepared = prepare_inputs(RunPaths.discover(SAMPLES))
    upgraded = upgrade_v1_to_v2(prepared.parsed_v1.records)
    assert upgraded["layer.00"].scale == 0.008
    assert upgraded["layer.00"].precision_label == "v1-thousandth-exact"
    downgraded, sidecar, thousandths = downgrade_v2_to_v1(prepared.parsed_v2.records)
    assert thousandths["layer.00"] == 8
    assert sidecar["layers"]["layer.01"]["per_channel"] is True
    assert sidecar["layers"]["layer.01"]["rounding_mode"] == "nearest_away"
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    restored, failures, status = reverse_downgrade(downgraded, sidecar_path)
    assert status == "restored" and not failures
    for layer, record in prepared.parsed_v2.records.items():
        assert restored[layer].to_dict() == record.to_dict()

    missing, missing_failures, missing_status = reverse_downgrade(
        downgraded, tmp_path / "absent.json"
    )
    assert not missing
    assert missing_status == "sidecar_missing"
    assert {failure.code for failure in missing_failures} == {E_SIDECAR_MISSING}


def test_quantile_rules_and_repeatability() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 4.0])
    assert sorted_quantile(values, 0.5, "linear", "lower_index") == 2.0
    assert sorted_quantile(np.asarray([1.0, 3.0]), 0.5, "linear", "lower_index") == 2.0
    kernel = np.asarray([[0.1, -0.2, 0.3, -0.4], [0.25, -0.75, 0.0, 0.5]])
    pack = _pack(kernel)
    layer_cfg = {"bit_width": 4, "group_size": 4, "per_channel": False, "scheme": "asymmetric"}
    calibration_cfg = _base_config()["calibration"]
    first = calibrate_layer(pack, layer_cfg, calibration_cfg, "v1").to_dict()
    second = calibrate_layer(pack, layer_cfg, calibration_cfg, "v1").to_dict()
    assert first == second
    symmetric_cfg = dict(layer_cfg)
    symmetric_cfg["scheme"] = "symmetric"
    symmetric = calibrate_layer(pack, symmetric_cfg, calibration_cfg, "v1")
    assert all(group.zero_point == 0 for group in symmetric.groups)


def test_fixed_point_fraction_and_bit_width_change_integers_and_errors() -> None:
    rng = np.random.default_rng(7)
    pack = _pack(rng.normal(0.0, 0.4, size=(4, 8)))
    calibration_cfg = _base_config()["calibration"]
    layer_cfg = {"bit_width": 8, "group_size": 0, "per_channel": False, "scheme": "asymmetric"}
    quant = calibrate_layer(pack, layer_cfg, calibration_cfg, "v1")
    coarse = quantize_layer(pack, quant, {"fraction_bits": 4, "rounding_mode": "nearest_even", "saturation": "saturate"}, full_precision=False)
    fine = quantize_layer(pack, quant, {"fraction_bits": 10, "rounding_mode": "nearest_even", "saturation": "saturate"}, full_precision=False)
    assert coarse.integers != fine.integers
    assert coarse.mse != fine.mse

    narrow_cfg = dict(layer_cfg)
    narrow_cfg["bit_width"] = 4
    narrow = quantize_layer(pack, calibrate_layer(pack, narrow_cfg, calibration_cfg, "v1"), {"fraction_bits": 10, "rounding_mode": "nearest_even", "saturation": "saturate"}, full_precision=False)
    assert narrow.mse != fine.mse
    assert narrow.bit_width == 4 and fine.bit_width == 8
    assert len(coarse.clip_directions) == pack.numel
    assert coarse.metadata["error_metadata"]["rounding_mode"] == "nearest_even"


def test_rounding_mode_affects_tied_integers() -> None:
    weights = np.asarray([[0.25, 0.75]])
    pack = _pack(weights)
    group = GroupQuantParam(
        index=0,
        start=0,
        end=2,
        q_low=-1.0,
        q_high=1.0,
        observed_min=0.25,
        observed_max=0.75,
        raw_scale=0.5,
        scale=0.5,
        zero_point=0,
        qmin=-8,
        qmax=7,
        scheme="asymmetric",
        interpolation="linear",
        tie_break="lower_index",
        fixed_fraction_bits=0,
    )
    quant = LayerQuantResult(
        layer="layer.test",
        shape=[1, 2],
        bit_width=4,
        qmin=-8,
        qmax=7,
        group_size=2,
        per_channel=False,
        scheme="asymmetric",
        interpolation="linear",
        tie_break="lower_index",
        quantile_lower=0.0,
        quantile_upper=1.0,
        fixed_fraction_bits=0,
        groups=[group],
        content_fingerprint=pack.content_fingerprint,
        param_fingerprint="test",
        exchange_version="v1",
    )
    even = quantize_layer(pack, quant, {"fraction_bits": 10, "rounding_mode": "nearest_even", "saturation": "saturate"}, full_precision=False)
    away = quantize_layer(pack, quant, {"fraction_bits": 10, "rounding_mode": "nearest_away", "saturation": "saturate"}, full_precision=False)
    assert even.integers != away.integers


def test_distillation_order_and_error_reduction() -> None:
    prepared = prepare_inputs(RunPaths.discover(SAMPLES))
    pack = prepared.weights["layer.01"]
    layer_cfg = prepared.config["layers"]["layer.01"]
    quant = calibrate_layer(pack, layer_cfg, prepared.config["calibration"], "v1")
    quantized = quantize_layer(pack, quant, prepared.config["fixed_point"], full_precision=False)
    records = distill_layer(pack, quantized, list(reversed(prepared.samples)), prepared.config["distillation"])
    assert [record["sample_id"] for record in records] == sorted(record["sample_id"] for record in records)
    assert all(record["after_max_abs_deviation"] <= record["before_max_abs_deviation"] + 1e-15 for record in records)
    assert any(abs(value) > 0 for record in records for value in record["correction"])


def test_engine_incremental_scope_and_digest_equality(tmp_path: Path) -> None:
    paths = RunPaths.discover(SAMPLES)
    base = prepare_inputs(paths)
    state = tmp_path / "state"
    baseline = QcalEngine(
        paths,
        output_dir=tmp_path / "out0",
        state_dir=state,
        prepared=base,
        fresh=True,
    ).run()
    assert baseline["failure_count"] == 0
    assert baseline["distillation_record_count"] == 24

    config = json.loads(json.dumps(base.config))
    config["layers"]["layer.02"]["bit_width"] = 8
    edited = prepare_inputs(paths, config_override=config, weights_override=base.weights)
    incremental = QcalEngine(
        paths,
        output_dir=tmp_path / "out1",
        state_dir=state,
        prepared=edited,
    ).run()
    rebuild = QcalEngine(
        paths,
        output_dir=tmp_path / "out2",
        state_dir=tmp_path / "state2",
        prepared=edited,
        fresh=True,
    ).run()
    assert incremental["result_digest"] == rebuild["result_digest"]
    recomputed = incremental["reuse"]["recomputed"]
    assert recomputed["parse_validation"] == []
    assert recomputed["conversion"] == []
    assert recomputed["quantile_calibration"] == ["layer.02"]
    assert recomputed["fixed_point_rescale"] == ["layer.02"]
    assert recomputed["distillation_correction"] == ["layer.02"]
    ledger = incremental["cache_ledger"]["distillation_correction"]["layer.02"]
    assert ledger["calibration_samples"] == [f"sample-{index:02d}" for index in range(6)]

    config2 = json.loads(json.dumps(config))
    config2["distillation"]["temperature"] = 2.0
    warmed = prepare_inputs(paths, config_override=config2, weights_override=edited.weights)
    distill_only = QcalEngine(
        paths,
        output_dir=tmp_path / "out3",
        state_dir=state,
        prepared=warmed,
    ).run()
    assert distill_only["reuse"]["recomputed"]["distillation_correction"] == [
        "layer.00",
        "layer.01",
        "layer.02",
        "layer.03",
    ]
    assert distill_only["reuse"]["recomputed"]["fixed_point_rescale"] == []


def test_weight_edit_invalidates_only_that_layer(tmp_path: Path) -> None:
    paths = RunPaths.discover(SAMPLES)
    base = prepare_inputs(paths)
    state = tmp_path / "state"
    QcalEngine(paths, output_dir=tmp_path / "o0", state_dir=state, prepared=base, fresh=True).run()
    operation = {
        "id": "replace",
        "replace_weight": {"layer": "layer.01", "seed": 99, "scale": 0.3, "content_version": "r2"},
    }
    weights, config, _ = operation_apply(base.weights, base.config, operation, replacement_weights)
    edited = prepare_inputs(paths, config_override=config, weights_override=weights)
    report = QcalEngine(paths, output_dir=tmp_path / "o1", state_dir=state, prepared=edited).run()
    recomputed = report["reuse"]["recomputed"]
    for stage in recomputed:
        assert recomputed[stage] == ["layer.01"], stage


def test_budget_pending_and_thread_order_equivalence(tmp_path: Path) -> None:
    paths = RunPaths.discover(SAMPLES)
    budget_report = QcalEngine(
        paths,
        output_dir=tmp_path / "budget",
        state_dir=tmp_path / "state_budget",
        fresh=True,
        budget_override=96,
        workers_override=3,
    ).run()
    assert budget_report["pending"] == [
        {"layer": "layer.02", "code": "E_QUANT_BUDGET_PENDING"},
        {"layer": "layer.03", "code": "E_QUANT_BUDGET_PENDING"},
    ]
    assert sorted(budget_report["quantization_errors"]) == ["layer.00", "layer.01"]
    single = QcalEngine(
        paths,
        output_dir=tmp_path / "single",
        state_dir=tmp_path / "state_single",
        fresh=True,
        workers_override=1,
    ).run()
    parallel = QcalEngine(
        paths,
        output_dir=tmp_path / "parallel",
        state_dir=tmp_path / "state_parallel",
        fresh=True,
        workers_override=8,
    ).run()
    assert single["result_digest"] == parallel["result_digest"]
    assert list(single["quantization_errors"]) == list(parallel["quantization_errors"])


def test_forced_interrupt_snapshot_recovery(tmp_path: Path) -> None:
    paths = RunPaths.discover(SAMPLES)
    full = QcalEngine(
        paths,
        output_dir=tmp_path / "full",
        state_dir=tmp_path / "state_full",
        fresh=True,
    ).run()
    state = tmp_path / "state_interrupt"
    with pytest.raises(ForcedInterrupt) as caught:
        QcalEngine(
            paths,
            output_dir=tmp_path / "interrupt",
            state_dir=state,
            fresh=True,
            force_interrupt_after_stage="quantile_calibration",
        ).run()
    assert caught.value.exit_code == 75
    status = snapshot_summary(state)
    assert status["recovered_stages"] == {
        "parse_validation": True,
        "quantile_calibration": True,
        "fixed_point_rescale": False,
        "distillation_correction": False,
    }
    resumed = QcalEngine(
        paths,
        output_dir=tmp_path / "resumed",
        state_dir=state,
    ).run()
    assert resumed["result_digest"] == full["result_digest"]
    assert resumed["reuse"]["recomputed"]["parse_validation"] == []
    assert resumed["reuse"]["recomputed"]["quantile_calibration"] == []


def test_injections_are_isolated_and_mapped(tmp_path: Path) -> None:
    paths = RunPaths.discover(SAMPLES)
    report = QcalEngine(
        paths,
        output_dir=tmp_path / "inject",
        state_dir=tmp_path / "state_inject",
        fresh=True,
        injection_path=SAMPLES / "injections.yaml",
        budget_override=32,
    ).run()
    by_id = {failure.get("injection_id"): failure["code"] for failure in report["failures"]}
    assert by_id["INJ-001"] == E_WEIGHT_NONFINITE
    assert by_id["INJ-002"] == E_CALIB_PARSE
    assert by_id["INJ-003"] == E_CALIB_PARSE
    assert by_id["INJ-004"] == E_SIDECAR_MISSING
    assert list(report["quantization_errors"]) == ["layer.00"]


def test_sample_generator_is_byte_deterministic(tmp_path: Path) -> None:
    from generate_samples import generate

    left = tmp_path / "left"
    right = tmp_path / "right"
    generate(left, 20261121)
    generate(right, 20261121)
    for path in sorted(left.iterdir()):
        if path.is_file():
            assert path.read_bytes() == (right / path.name).read_bytes(), path.name


def test_full_acceptance_suite(tmp_path: Path) -> None:
    report = run_acceptance(SAMPLES, tmp_path / "artifacts")
    assert all(report["verification"].values()), report["verification"]
    assert report["edit_sequence"]["all_match"]
