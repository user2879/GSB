from __future__ import annotations

import numpy as np

from quant_pipeline.calibration import CalibrationResult
from quant_pipeline.config import DistillationConfig, EffectiveLayerConfig, ToleranceConfig
from quant_pipeline.distillation import distill_layer
from quant_pipeline.error_analysis import analyze_errors
from quant_pipeline.inputs import LayerCalibrationView, WeightLayer
from quant_pipeline.quantization import QuantizationResult
from quant_pipeline.util import sha256_json


def make_effective(**updates) -> EffectiveLayerConfig:
    body = dict(
        layer="unit.layer",
        profile="q8",
        target_bits=8,
        fractional_bits=8,
        group_size=2,
        min_group_size=1,
        scheme="symmetric",
        saturation="saturate",
        rounding="nearest_even",
        percentile_low=0.0,
        percentile_high=1.0,
        interpolation="linear",
        tie_break="lower_index",
    )
    body.update(updates)
    return EffectiveLayerConfig(**body)


def make_calibration(groups, scales, zeros) -> CalibrationResult:
    return CalibrationResult(
        data={
            "shape": [len(scales)],
            "groups": groups,
            "scales": scales,
            "zero_points": zeros,
        },
        scales=np.asarray(scales, dtype=np.float64),
        zero_points=np.asarray(zeros, dtype=np.int64),
        group_index=np.zeros(len(scales), dtype=np.int64),
    )


def test_zero_truth_uses_absolute_branch_and_tie_priority() -> None:
    layer = WeightLayer(
        name="unit.layer",
        shape=(2,),
        weights=np.asarray([0.0, 1.0]),
        version=1,
        marker_group_size=2,
        line_no=1,
        content_fingerprint=sha256_json({"x": [0.0, 1.0]}),
    )
    groups = [
        {
            "group_id": "unit.layer#g0000",
            "element_start": 0,
            "element_end": 2,
            "element_count": 2,
            "percentile_low_value": 0.0,
            "percentile_high_value": 1.0,
        }
    ]
    calibration = make_calibration(groups, [1.0, 1.0], [0, 0])
    quantization = QuantizationResult(
        data={},
        fixed_ints=np.asarray([0, 1]),
        dequantized=np.asarray([0.1, 1.0]),
        base_dequantized=np.asarray([0.05, 1.0]),
        unclipped_dequantized=np.asarray([0.1, 1.0]),
        clipped_mask=np.asarray([False, False]),
    )
    report = analyze_errors(
        layer,
        quantization,
        calibration,
        make_effective(),
        ToleranceConfig(absolute=0.01, relative=0.5),
    )
    assert report["statistics"]["zero_truth_absolute_branch_count"] == 1
    assert report["relative_errors"][0] is None
    assert report["relative_branches"][0] == "absolute_for_zero_truth"
    first = report["over_tolerance_elements"][0]
    assert first["element_index"] == 0
    assert first["contributions"]["scaling_quantization"] == 0.05
    assert first["contributions"]["dequantization_rounding"] == 0.05
    assert first["primary_source"] == "scaling_quantization"


def test_distillation_reduces_mean_bias_with_soft_labels() -> None:
    weights = np.asarray(
        [
            0.4,
            -0.3,
            0.7,
            0.1,
            0.8,
            -0.6,
        ],
        dtype=np.float64,
    )
    layer = WeightLayer(
        name="unit.layer",
        shape=(2, 3),
        weights=weights,
        version=1,
        marker_group_size=3,
        line_no=1,
        content_fingerprint=sha256_json({"weights": weights.tolist()}),
    )
    inputs = (
        np.asarray([1.0, -0.5, 0.25]),
        np.asarray([0.2, 0.4, -1.0]),
        np.asarray([-0.7, 0.3, 0.9]),
    )
    references = tuple(layer.apply(vector) for vector in inputs)
    view = LayerCalibrationView(
        layer=layer.name,
        sample_ids=("s0", "s1", "s2"),
        inputs=inputs,
        references=references,
    )
    quantized_weights = weights + np.asarray([0.08, -0.05, 0.02, -0.04, 0.03, 0.06])
    result = distill_layer(
        layer,
        quantized_weights,
        view,
        DistillationConfig(temperature=1.2, soft_label_weight=0.8),
    )
    assert result["summary"]["sample_count"] == 3
    assert result["summary"]["mean_absolute_bias_after"] <= result["summary"][
        "mean_absolute_bias_before"
    ]
    assert all(sample["correction"] for sample in result["samples"])

