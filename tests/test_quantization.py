from __future__ import annotations

import numpy as np
import pytest

from quant_workbench.models import LayerConfig, WeightLayer
from quant_workbench.quantization import quantize_layer


def make_layer(values: list[float], shape: tuple[int, ...] | None = None) -> WeightLayer:
    shape = shape or (1, len(values))
    return WeightLayer(
        layer="layer.x",
        version=1,
        shape=shape,
        weights=np.asarray(values, dtype=np.float64).reshape(shape),
        order=0,
    )


def test_groups_scales_zeros_and_saturation_are_recorded() -> None:
    layer = make_layer([-3.0, 0.1, 1.0, 3.4])
    result = quantize_layer(
        layer,
        LayerConfig(bits=4, group_size=4, fractional_bits=3, scheme="asymmetric"),
        significant_digits=1,
    )
    assert len(result["groups"]) == 1
    for group in result["groups"]:
        assert group["symmetric_scale"] > 0
        assert group["asymmetric_scale"] > 0
        assert 0 <= group["asymmetric_zero_point"] <= 15
    assert result["zero_points"] != [0]
    assert result["saturation_count"] > 0
    saturated = [row for row in result["truncations"] if row["direction"] != "none"]
    assert saturated
    assert all(row["direction"] in {"low", "high"} for row in saturated)
    assert all(isinstance(row["preclip_int"], int) for row in saturated)


def test_bit_and_fractional_width_changes_are_real() -> None:
    layer = make_layer([-1.7, -0.4, 0.2, 0.8, 1.3, 2.1, -2.0, 0.0])
    low_bits = quantize_layer(
        layer,
        LayerConfig(bits=4, group_size=4, fractional_bits=6, scheme="symmetric"),
        significant_digits=6,
    )
    high_bits = quantize_layer(
        layer,
        LayerConfig(bits=8, group_size=4, fractional_bits=6, scheme="symmetric"),
        significant_digits=6,
    )
    assert low_bits["integers"] != high_bits["integers"]
    assert low_bits["dequantized"] != high_bits["dequantized"]
    assert low_bits["cache_key"] != high_bits["cache_key"]

    coarse_fraction = quantize_layer(
        layer,
        LayerConfig(bits=8, group_size=4, fractional_bits=2, scheme="symmetric"),
        significant_digits=6,
    )
    assert high_bits["integers"] == coarse_fraction["integers"]
    assert high_bits["fixed_integers"] != coarse_fraction["fixed_integers"]
    assert high_bits["dequantized"] != coarse_fraction["dequantized"]
    assert high_bits["calibration_cache_key"] == coarse_fraction["calibration_cache_key"]
    assert high_bits["cache_key"] != coarse_fraction["cache_key"]


def test_symmetric_zero_points_are_zero_and_scales_trimmed() -> None:
    layer = make_layer([-2.34567, 0.1, 1.23456, 2.0])
    result = quantize_layer(
        layer,
        LayerConfig(bits=8, group_size=4, fractional_bits=4, scheme="symmetric"),
        significant_digits=4,
    )
    assert result["zero_points"] == [0]
    for value in result["symmetric_scales"]:
        assert float(f"{value:.4g}") == value
    assert result["truncations"][0]["direction"] in {"none", "low", "high"}
