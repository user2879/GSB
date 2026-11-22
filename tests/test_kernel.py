from __future__ import annotations

import numpy as np

from qcolumn_diag.calibration import derive_quant_params, quantile_stage_artifact
from qcolumn_diag.kernel import analyze_column_width
from qcolumn_diag.models import GroupSpec, TensorConfig
from qcolumn_diag.reference import verify_active_width


def make_tensor(**overrides: object) -> TensorConfig:
    base = dict(
        tensor_id="t",
        shape=(7, 1),
        column_axis=1,
        quantile_lower_permille=200,
        quantile_upper_permille=800,
        scheme="asymmetric",
        bit_width_candidates=(4, 8),
        active_bit_width=8,
        fractional_bits=4,
        saturation_strategy="saturate",
        zero_scale_policy="clamp_to_min_scale",
        tolerance=0.01,
        group_over_tolerance_limit=0.5,
        groups=(GroupSpec("all", (0,)),),
    )
    base.update(overrides)
    return TensorConfig(**base)


def analyze(values: np.ndarray, tensor: TensorConfig, width: int):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    missing = np.isnan(values)
    calibration, failure = quantile_stage_artifact(
        values,
        missing,
        tensor,
        interpolation="linear",
        tie_break="stable_ascending_first_occurrence",
    )
    assert failure is None
    params = derive_quant_params(
        lower_quantile=calibration["lower_quantile"],
        upper_quantile=calibration["upper_quantile"],
        tensor=tensor,
        bit_width=width,
    )
    refs = [
        {"row": row, "index": [row, 0], "injection_ids": [], "missing": bool(missing[row])}
        for row in range(values.size)
    ]
    analysis, failures = analyze_column_width(
        values=values,
        missing_mask=missing,
        tensor=tensor,
        calibration=calibration,
        params=params,
        sample_refs=refs,
    )
    return values, missing, calibration, params, analysis, failures


def test_scale_is_quantized_to_fractional_grid_and_width_changes_codes() -> None:
    values = np.asarray([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    tensor = make_tensor()
    values, missing, calibration, params4, analysis4, failures4 = analyze(values, tensor, 4)
    _, _, _, params8, analysis8, failures8 = analyze(values, tensor, 8)
    assert not failures4 and not failures8
    for params in (params4, params8):
        unit = params["scale_unit"]
        units = params["quantized_scale"] / unit
        assert abs(units - round(units)) < 1e-12
        assert params["scale_quantization_delta"] >= 0.0
        assert params["zero_scale"] is False
    codes4 = [item["quantized_code"] for item in analysis4["elements"]]
    codes8 = [item["quantized_code"] for item in analysis8["elements"]]
    assert codes4 != codes8
    verification = verify_active_width(
        values=values,
        missing_mask=missing,
        tensor=tensor,
        calibration=calibration,
        params=params8,
        analysis=analysis8,
    )
    assert verification["passed"]


def test_fractional_bits_really_change_intermediate_codes() -> None:
    values = np.asarray([-3.25, -1.5, 0.0, 1.5, 3.25])
    tensor_low = make_tensor(fractional_bits=2)
    tensor_high = make_tensor(fractional_bits=10)
    _, _, _, params_low, analysis_low, _ = analyze(values, tensor_low, 8)
    _, _, _, params_high, analysis_high, _ = analyze(values, tensor_high, 8)
    assert params_low["quantized_scale"] != params_high["quantized_scale"]
    assert [e["quantized_code"] for e in analysis_low["elements"]] != [
        e["quantized_code"] for e in analysis_high["elements"]
    ]
    assert [e["dequantized"] for e in analysis_low["elements"]] != [
        e["dequantized"] for e in analysis_high["elements"]
    ]


def test_saturation_direction_preclip_and_zero_branch() -> None:
    values = np.asarray([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    tensor = make_tensor()
    _, _, _, params, analysis, failures = analyze(values, tensor, 8)
    assert not failures
    low = analysis["elements"][0]
    high = analysis["elements"][-1]
    assert low["clipped"] and low["clip_direction"] == "low"
    assert high["clipped"] and high["clip_direction"] == "high"
    assert low["preclip_code"] < params["code_min"]
    assert high["preclip_code"] > params["code_max"]
    zero = analysis["elements"][3]
    assert zero["value"] == 0.0
    assert zero["relative_error"] is None
    assert zero["error_branch"] == "absolute_at_zero"


def test_symmetric_scheme_has_integer_zero_point_and_zero_scale_degrades() -> None:
    tensor = make_tensor(scheme="symmetric")
    _, _, _, params, _, _ = analyze(
        np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]), tensor, 8
    )
    assert params["zero_point"] == 0
    zero_tensor = make_tensor()
    _, _, _, zero_params, zero_analysis, failures = analyze(
        np.asarray([0.0, 0.0, 0.0]), zero_tensor, 8
    )
    assert zero_params["zero_scale"] is True
    assert zero_params["degraded_scale"] is True
    assert zero_params["quantized_scale"] == zero_params["scale_unit"]
    assert any(item["error_code"] == "E_SCALE_ZERO" for item in failures)
    assert zero_analysis["summary"]["analyzed_count"] == 3


def test_all_attribution_shares_sum_to_element_error() -> None:
    values = np.asarray([-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
    tensor = make_tensor()
    _, _, _, _, analysis, _ = analyze(values, tensor, 8)
    assert any(item["clipped"] for item in analysis["elements"])
    for element in analysis["elements"]:
        attribution = element["attribution"]
        assert attribution is not None
        total = sum(attribution["shares"].values())
        assert abs(total - element["abs_error"]) <= 1e-12
        assert attribution["share_sum_error"] <= tensor.tolerance
    clipped = [item for item in analysis["elements"] if item["clipped"]]
    assert any(item["dominant_source"] == "saturation_clipping" for item in clipped)
