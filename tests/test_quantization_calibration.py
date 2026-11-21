from __future__ import annotations

import copy

import numpy as np

from layerpack.calibration import evaluate_layer
from layerpack.config import EffectiveLayerSettings
from layerpack.loader import load_inputs
from layerpack.quantize import quantize_layer


def _settings(bits=4, scheme="symmetric", group_size=4, per_channel=True):
    return EffectiveLayerSettings(
        bits=bits,
        scheme=scheme,
        group_size=group_size,
        group_axis=-1,
        per_channel=per_channel,
        scale_significant_digits=7,
    )


def test_quantization_is_elementwise_deterministic(sample_paths):
    state, config = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    layer = state.layers["encoder.attn.v"]
    first = quantize_layer(layer, _settings())
    second = quantize_layer(layer, _settings())
    assert np.array_equal(first.dequantized, second.dequantized)
    assert np.array_equal(first.codes, second.codes)
    assert first.parameters == second.parameters
    assert first.distortion == second.distortion
    assert first.parameters["degenerated_group_count"] >= 1
    assert set(first.parameters["scheme_candidates"]) == {
        "symmetric",
        "asymmetric",
    }
    assert any(
        source["source"] == "whole_layer"
        for source in first.parameters["degeneration_sources"]
    )


def test_symmetric_and_asymmetric_produce_finite_results(sample_paths):
    state, _ = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    layer = state.layers["head"]
    symmetric = quantize_layer(layer, _settings(bits=8, scheme="symmetric"))
    asymmetric = quantize_layer(layer, _settings(bits=8, scheme="asymmetric"))
    assert np.all(np.isfinite(symmetric.dequantized))
    assert np.all(np.isfinite(asymmetric.dequantized))
    assert symmetric.distortion["mse"] >= 0.0
    assert asymmetric.distortion["mse"] >= 0.0
    assert symmetric.quantized_sha256 != asymmetric.quantized_sha256


def test_per_channel_toggle_changes_dequantized_weights(sample_paths):
    state, _ = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    layer = state.layers["embed"]
    channel = quantize_layer(layer, _settings(per_channel=True))
    group_only = quantize_layer(layer, _settings(per_channel=False))
    assert not np.array_equal(channel.dequantized, group_only.dequantized)


def test_calibration_changes_with_bit_width_and_reduces_error(sample_paths):
    state, config = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    layer = state.layers["head"]
    results = {}
    for bits in (2, 4, 8):
        quant = quantize_layer(layer, _settings(bits=bits, group_size=8))
        result = evaluate_layer(
            layer=layer,
            quant=quant,
            samples=state.calibration.samples,
            config=config,
        )
        results[bits] = result
        assert result.summary["after"]["mse"] <= result.summary["before"]["mse"]
    assert results[2].summary["before"]["mse"] > results[4].summary["before"]["mse"]
    assert results[4].summary["before"]["mse"] > results[8].summary["before"]["mse"]


def test_zero_soft_label_weight_is_marked_not_skipped(sample_paths):
    state, config = load_inputs(
        sample_paths["weights"], sample_paths["calibration"], sample_paths["config"]
    )
    config = copy.deepcopy(config)
    config.soft_label_weight = 0.0
    layer = state.layers["head"]
    quant = quantize_layer(layer, _settings())
    result = evaluate_layer(
        layer=layer,
        quant=quant,
        samples=state.calibration.samples,
        config=config,
    )
    assert result.summary["soft_label_disabled"] is True
    assert len(result.samples) == len(state.calibration.samples)
    assert all(sample["correction_disabled"] for sample in result.samples)
    assert all(sample["disabled_reason"] == "soft_label_weight_zero" for sample in result.samples)
