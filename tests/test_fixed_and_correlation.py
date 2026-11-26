from __future__ import annotations

from decimal import Decimal

from fixed_lab.dataset import Dataset, SampleRecord
from fixed_lab.features import correlation_analysis, quantize_dataset
from fixed_lab.fixedpoint import FixedParams
from fixed_lab.pathing import classify_zero


def test_quantization_records_grid_step_and_residual() -> None:
    params = FixedParams(decimal_places=4, quant_level=1)
    dataset = Dataset(columns=["a"])
    dataset.rows["S1"] = SampleRecord("S1", 1, [Decimal("0.123456")], Decimal("1"), Decimal("1"))
    qd = quantize_dataset(dataset, params)
    record = qd.feature_records[0][0]
    assert record["integer"] % params.grid == 0
    assert record["step"] == "0.001"
    assert abs(Decimal(record["residual"])) <= Decimal("0.0005")


def test_exact_threshold_correlation_is_marked_boundary() -> None:
    params = FixedParams(decimal_places=6, quant_level=0)
    dataset = Dataset(columns=["x", "y"])
    for index in range(6):
        value = Decimal(index + 1)
        dataset.rows[f"S{index}"] = SampleRecord(f"S{index}", index, [value, value], value, Decimal("1"))
    qd = quantize_dataset(dataset, params)
    result = correlation_analysis(qd, Decimal("1"), Decimal("1"))
    pair = result["pairs"][0]
    assert pair["strong_match"] is True
    assert pair["boundary_exact_threshold"] is True
    assert pair["boundary"] is True


def test_half_step_zero_tie_has_fixed_decision() -> None:
    params = FixedParams(decimal_places=4, quant_level=1)  # grid=10, half=5
    zero = classify_zero(0, params, "zero")
    tie = classify_zero(5, params, "zero")
    nonzero = classify_zero(6, params, "zero")
    assert zero["decision"] == "zero_exact"
    assert tie["is_zero"] is True and tie["boundary"] is True
    assert nonzero["is_zero"] is False
