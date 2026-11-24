from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quant_pipeline.config import E_CONFIG_INVALID, parse_config
from quant_pipeline.errors import E_CALIBRATION_INVALID, E_WEIGHT_PACK_INVALID
from quant_pipeline.inputs import (
    InjectionEngine,
    apply_injections,
    build_layer_calibration_views,
    load_injections,
    load_weight_records,
    normalize_calibration,
    validate_weight_pack,
)
from quant_pipeline.samples import generate_samples
from quant_pipeline.yaml_lite import loads_yaml


def test_sample_generation_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_samples(first, tmp_path / "c1")
    generate_samples(second, tmp_path / "c2")
    for name in ("weights.jsonl", "calibration.json", "config.yaml", "injections.yaml", "injections_all.yaml"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_valid_config_and_weight_pack(sample_dir: Path) -> None:
    raw = json.loads((sample_dir / "config.yaml").read_text(encoding="utf-8"))
    config, issues = parse_config(raw)
    assert issues == []
    records = load_weight_records(sample_dir / "weights.jsonl")
    assert len(records) == 8
    injection = InjectionEngine()
    layers, failures = validate_weight_pack(records, config, injection)
    assert failures == []
    assert sorted(layers) == sorted(config.layer_overrides)
    first = layers["block0.attn.k"]
    assert first.shape == (6, 8)
    assert first.numel == 48
    assert first.version == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["quantization"].__setitem__("scheme", "mystery"),
        lambda raw: raw["quantization"].__setitem__("rounding", "mystery"),
        lambda raw: raw["quantization"].__setitem__("saturation", "mystery"),
        lambda raw: raw["quantization"].__setitem__("percentile_low", 1.1),
        lambda raw: raw["quantization"]["profiles"][1].__setitem__("bits", 6),
        lambda raw: raw["distillation"].__setitem__("temperature", 0),
        lambda raw: raw["distillation"].__setitem__("soft_label_weight", 1.5),
        lambda raw: raw["runtime"].__setitem__("max_workers", 0),
        lambda raw: raw["layers"]["block0.attn.k"].__setitem__("target_bits", 99),
    ],
)
def test_invalid_configs_are_reported(sample_dir: Path, mutation) -> None:
    raw = json.loads((sample_dir / "config.yaml").read_text(encoding="utf-8"))
    mutation(raw)
    _config, issues = parse_config(raw)
    assert issues
    assert all(issue.code == E_CONFIG_INVALID for issue in issues)


def test_weight_pack_validation_failures(sample_dir: Path) -> None:
    raw_config = json.loads((sample_dir / "config.yaml").read_text(encoding="utf-8"))
    config, issues = parse_config(raw_config)
    assert not issues
    records = load_weight_records(sample_dir / "weights.jsonl")

    duplicate = json.loads((sample_dir / "weights.jsonl").read_text(encoding="utf-8").splitlines()[0])
    records.append(type(records[0])(line_no=99, data=duplicate))
    layers, failures = validate_weight_pack(records, config, InjectionEngine())
    assert "block0.attn.k" in layers
    assert any(item.code == E_WEIGHT_PACK_INVALID and "duplicate" in item.message for item in failures)

    records = load_weight_records(sample_dir / "weights.jsonl")
    records[0].data["version"] = 99
    records[1].data["version"] = 3
    _layers, failures = validate_weight_pack(records, config, InjectionEngine())
    assert any("lower than previous version" in item.message for item in failures)

    records = load_weight_records(sample_dir / "weights.jsonl")
    records[0].data["shape"] = [0, 8]
    _layers, failures = validate_weight_pack(records, config, InjectionEngine())
    assert any("positive integer" in item.message for item in failures)

    records = load_weight_records(sample_dir / "weights.jsonl")
    records[0].data["weights"][0] = float("nan")
    _layers, failures = validate_weight_pack(records, config, InjectionEngine())
    assert any("non-finite" in item.message for item in failures)


def test_calibration_normalization_and_dimension_contract(sample_dir: Path) -> None:
    raw = json.loads((sample_dir / "calibration.json").read_text(encoding="utf-8"))
    calibration = normalize_calibration(raw)
    assert len(calibration.samples) == 4
    inputs, references = calibration.for_layer("block0.attn.k")
    assert len(inputs) == len(references) == 4
    assert len(inputs[0]) == 8
    assert len(references[0]) == 6

    raw["samples"][0]["inputs"]["block0.attn.k"].append(1.0)
    mismatched = normalize_calibration(raw)
    raw_config = json.loads((sample_dir / "config.yaml").read_text(encoding="utf-8"))
    config, issues = parse_config(raw_config)
    assert not issues
    records = load_weight_records(sample_dir / "weights.jsonl")
    layers, failures = validate_weight_pack(records, config, InjectionEngine())
    assert not failures
    views, failures = build_layer_calibration_views(layers, mismatched, InjectionEngine())
    assert "block0.attn.k" not in views
    assert any(item.code == E_CALIBRATION_INVALID for item in failures)


def test_injection_config_maps_all_six_types(sample_dir: Path) -> None:
    items, failures = load_injections(sample_dir / "injections_all.yaml")
    assert failures == []
    assert len(items) == 6
    assert {item.expected_code for item in items} == {
        E_WEIGHT_PACK_INVALID,
        E_CONFIG_INVALID,
        E_CALIBRATION_INVALID,
    }


def test_yaml_fallback_accepts_block_configuration() -> None:
    text = """
quantization:
  active_profile: q8
  profiles:
    - name: q6
      bits: 6
      fractional_bits: 6
    - name: q8
      bits: 8
      fractional_bits: 8
  scheme: symmetric
  allowed_bit_widths: [2, 4, 6, 8]
distillation:
  temperature: 1.5
  soft_label_weight: 0.35
runtime:
  max_workers: 4
tolerances:
  absolute: 0.01
  relative: 0.1
layers:
  layer.a:
    group_size: 4
"""
    parsed = loads_yaml(text)
    config, issues = parse_config(parsed)
    assert issues == []
    assert config.active_profile == "q8"
    assert config.profiles["q6"].bits == 6
    assert config.layer_overrides["layer.a"]["group_size"] == 4


def test_layer_list_configuration_survives_empty_injection_pass() -> None:
    raw = {
        "quantization": {
            "active_profile": "q8",
            "profiles": {"q8": {"bits": 8, "fractional_bits": 8}},
            "scheme": "symmetric",
            "rounding": "nearest_even",
            "saturation": "saturate",
        },
        "distillation": {"temperature": 1.0, "soft_label_weight": 0.2},
        "runtime": {"max_workers": 2, "point_budget": 100, "snapshot_interval": 1},
        "tolerances": {"absolute": 0.1, "relative": 0.1},
        "layers": [{"name": "layer.a", "group_size": 4}],
    }
    engine = apply_injections([], [], raw, {})
    assert engine.failures == []
    config, issues = parse_config(raw)
    assert issues == []
    assert config.layer_overrides["layer.a"]["group_size"] == 4


def test_mapping_profiles_are_sorted_by_bits() -> None:
    raw = {
        "quantization": {
            "active_profile": "q8",
            "profiles": {
                "q8": {"bits": 8, "fractional_bits": 8},
                "q4": {"bits": 4, "fractional_bits": 4},
            },
        }
    }
    config, issues = parse_config(raw)
    assert issues == []
    assert config.profile_order == ("q4", "q8")
