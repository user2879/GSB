from __future__ import annotations

import numpy as np
import pytest

from quant_pipeline.calibration import build_groups, calibrate_layer, quantile_sorted
from quant_pipeline.config import EffectiveLayerConfig
from quant_pipeline.errors import E_SATURATION_EXCEEDED, PipelineError
from quant_pipeline.inputs import WeightLayer
from quant_pipeline.quantization import quantize_layer
from quant_pipeline.util import sha256_json


def make_layer(values, shape=None, group_size=4, version=1, name="unit.layer") -> WeightLayer:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    shape = tuple(shape or (array.size,))
    return WeightLayer(
        name=name,
        shape=shape,
        weights=array,
        version=version,
        marker_group_size=group_size,
        line_no=1,
        content_fingerprint=sha256_json({"name": name, "values": array.tolist(), "version": version}),
    )


def make_config(**updates) -> EffectiveLayerConfig:
    base = dict(
        layer="unit.layer",
        profile="q8",
        target_bits=8,
        fractional_bits=8,
        group_size=4,
        min_group_size=2,
        scheme="symmetric",
        saturation="saturate",
        rounding="nearest_even",
        percentile_low=0.05,
        percentile_high=0.95,
        interpolation="linear",
        tie_break="lower_index",
    )
    base.update(updates)
    return EffectiveLayerConfig(**base)


def test_quantile_linear_rule_and_ties() -> None:
    values = np.asarray([4.0, 1.0, 1.0, 8.0, 2.0])
    assert quantile_sorted(values, 0.0) == 1.0
    assert quantile_sorted(values, 1.0) == 8.0
    assert quantile_sorted(values, 0.5) == 2.0
    assert quantile_sorted(values, 0.25) == 1.0
    assert quantile_sorted(values, 0.3) == pytest.approx(1.2)


def test_group_merge_records_basis() -> None:
    groups = build_groups("layer", 10, 2, 3)
    assert groups[0]["group_id"] == "layer#g0000"
    assert groups[0]["element_start"] == 0
    assert groups[0]["element_end"] == 10
    assert groups[0]["merged"] is True
    assert "merged into previous group" in groups[0]["merge_basis"]


def test_symmetric_and_asymmetric_calibration_are_deterministic() -> None:
    values = np.asarray([-3.0, -1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 9.0])
    layer = make_layer(values)
    symmetric = make_config(scheme="symmetric", group_size=4, min_group_size=2)
    asymmetric = make_config(scheme="asymmetric", group_size=4, min_group_size=2)
    first = calibrate_layer(layer, symmetric)
    second = calibrate_layer(layer, symmetric)
    assert first.data == second.data
    assert len(first.data["groups"]) == 2
    assert np.all(first.zero_points == 0)
    assert all(group["scale"] > 0 for group in first.data["groups"])
    asym = calibrate_layer(layer, asymmetric)
    assert asym.data != first.data
    assert any(group["zero_point"] != 0 for group in asym.data["groups"])
    assert asym.data["percentile_metadata"]["interpolation"] == "linear"
    assert asym.data["percentile_metadata"]["tie_break"] == "lower_index"


def test_bit_width_and_fractional_width_change_integers_and_dequantization() -> None:
    values = np.sin(np.arange(32, dtype=np.float64) * 1.13) * 2.0 + 0.03
    layer = make_layer(values, shape=(8, 4), group_size=4)
    base = make_config(target_bits=8, fractional_bits=4, group_size=4)
    cal_base = calibrate_layer(layer, base)
    q_base = quantize_layer(layer, cal_base, base)

    wide = make_config(target_bits=10, fractional_bits=4, group_size=4)
    q_wide = quantize_layer(layer, calibrate_layer(layer, wide), wide)
    assert q_base.fixed_ints.tolist() != q_wide.fixed_ints.tolist()
    assert q_base.dequantized.tolist() != q_wide.dequantized.tolist()

    finer = make_config(target_bits=8, fractional_bits=8, group_size=4)
    q_finer = quantize_layer(layer, calibrate_layer(layer, finer), finer)
    assert q_base.fixed_ints.tolist() != q_finer.fixed_ints.tolist()
    assert q_base.dequantized.tolist() != q_finer.dequantized.tolist()
    assert q_base.data["fixed_multiplier"] == 16
    assert q_finer.data["fixed_multiplier"] == 256


def test_saturation_records_are_index_sorted_and_switch_takes_effect() -> None:
    values = np.asarray([0.0, 0.1, -0.2, 0.0, 8.0, -9.0, 0.05, 0.02])
    layer = make_layer(values, shape=(2, 4), group_size=4)
    config = make_config(
        percentile_low=0.4,
        percentile_high=0.6,
        group_size=4,
        min_group_size=2,
    )
    calibration = calibrate_layer(layer, config)
    result = quantize_layer(layer, calibration, config)
    assert result.data["clipped_count"] > 0
    indices = [item["element_index"] for item in result.data["clipped_elements"]]
    assert indices == sorted(indices)
    assert all(item["direction"] in {"negative", "positive"} for item in result.data["clipped_elements"])
    assert result.data["fixed_ints"] != values.tolist()

    error_config = make_config(
        saturation="error",
        percentile_low=0.4,
        percentile_high=0.6,
        group_size=4,
        min_group_size=2,
    )
    with pytest.raises(PipelineError) as exc:
        quantize_layer(layer, calibration, error_config)
    assert exc.value.code == E_SATURATION_EXCEEDED
