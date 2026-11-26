from __future__ import annotations

import copy

import pytest

from vibstream.blocks import parse_block, parse_block_stream
from vibstream.config import ConfigError, parse_config
from vibstream.engine import MonitorEngine
from vibstream.errors import E_CONFIG_INVALID


def _base_config():
    return {
        "schema_version": 1,
        "channels": [
            {"id": "ch01", "sample_rate_hz": 2000},
            {"id": "ch02", "sample_rate_hz": 2000},
        ],
        "window": {"length": 96, "step": 32, "retention_depth": 8},
        "bands": [
            {"name": "low", "low_hz": 0, "high_hz": 80},
            {"name": "mid", "low_hz": 80, "high_hz": 160},
        ],
        "clustering": {"count": 3, "min_members": 2, "merge_rule": "nearest_l1"},
        "fixed_point": {"fraction_bits": 16, "quantization_levels": 256},
        "scheduler": {"batch_point_budget": 500, "cluster_point_quota": 300},
        "segment": {"min_windows": 2},
    }


def _base_block():
    return {
        "block_id": "b1",
        "arrival_seq": 1,
        "acquisition_ms": 0,
        "signals": {
            "ch01": [0.1, 0.2, 0.3, 0.4],
            "ch02": [0.2, 0.3, 0.4, 0.5],
        },
        "missing_segments": [],
        "resampling_markers": [],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["channels"][1].update({"id": "ch01"}),
        lambda data: data["channels"][0].update({"sample_rate_hz": 0}),
        lambda data: data["window"].update({"length": 0}),
        lambda data: data["window"].update({"step": 100}),
        lambda data: data["clustering"].update({"count": 0}),
        lambda data: data["fixed_point"].update({"fraction_bits": 14}),
        lambda data: data["fixed_point"].update({"quantization_levels": 100}),
    ],
)
def test_invalid_config_is_rejected(mutate):
    data = _base_config()
    mutate(data)
    with pytest.raises(ConfigError) as caught:
        parse_config(data)
    assert caught.value.code == E_CONFIG_INVALID


def test_invalid_block_inputs_use_e_config_invalid():
    config = parse_config(_base_config())
    cases = []
    unknown = _base_block()
    unknown["signals"]["ghost"] = [0.0, 0.0, 0.0, 0.0]
    cases.append(unknown)
    negative_time = _base_block()
    negative_time["acquisition_ms"] = -1
    cases.append(negative_time)
    missing_outside = _base_block()
    missing_outside["missing_segments"] = [{"channel": "ch01", "start": 2, "end": 8}]
    cases.append(missing_outside)
    resampling_outside = _base_block()
    resampling_outside["resampling_markers"] = [
        {"channel": "ch02", "start": 3, "end": 9, "source_rate_hz": 2000, "target_rate_hz": 1500}
    ]
    cases.append(resampling_outside)
    for block in cases:
        with pytest.raises(Exception) as caught:
            parse_block(block, config)
        assert getattr(caught.value, "code", None) == E_CONFIG_INVALID


def test_block_stream_requires_strictly_increasing_arrival():
    config = parse_config(_base_config())
    first = _base_block()
    second = copy.deepcopy(first)
    second["block_id"] = "b2"
    second["arrival_seq"] = 1
    with pytest.raises(Exception) as caught:
        parse_block_stream({"blocks": [first, second]}, config)
    assert caught.value.code == E_CONFIG_INVALID


def test_engine_rejects_bad_block_without_mutating_state(sample_config, sample_blocks):
    engine = MonitorEngine(sample_config, sample_blocks)
    bad = copy.deepcopy(sample_blocks[0].to_dict())
    bad["block_id"] = "bad-unknown"
    bad["arrival_seq"] = 1000
    bad["signals"]["ghost"] = [0.0] * len(bad["signals"]["ch01"])
    report = engine.apply_command({"type": "append", "blocks": [bad]})
    assert report["status"] == "REJECTED"
    assert report["failures"][0]["code"] == E_CONFIG_INVALID
    assert "bad-unknown" not in engine.blocks


def test_engine_rejects_invalid_window_command_without_mutating_config(sample_config, sample_blocks):
    engine = MonitorEngine(sample_config, sample_blocks)
    old_hash = engine.config.config_hash
    report = engine.apply_command(
        {"type": "set_window", "window": {"length": 20, "step": 40, "retention_depth": 4}}
    )
    assert report["status"] == "REJECTED"
    assert report["failures"][0]["code"] == E_CONFIG_INVALID
    assert engine.config.config_hash == old_hash
