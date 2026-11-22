import numpy as np
import pytest

from edge_lease import fixedpoint as fp
from edge_lease.errors import GridError


def test_scale_must_be_power_of_ten():
    assert fp.validate_scale(1) == 1
    assert fp.validate_scale(1000) == 1000
    for bad in (0, -1, 3, 105, 1.0, True, "100"):
        with pytest.raises(GridError) as exc:
            fp.validate_scale(bad)
        assert exc.value.code == "E_GRID"


def test_quantize_truncates_toward_zero_and_reports_single_error():
    assert fp.quantize_scalar("0.125", 1000) == (125, 0)
    assert fp.quantize_scalar(0.1255, 1000) == (125, 1)
    assert fp.quantize_scalar(-0.1255, 1000) == (-125, 1)


def test_quantize_array_matches_scalar_semantics():
    values = ["0.125", "0.375", "0.5"]
    q, flags = fp.quantize_array(values, 1000)
    assert q.tolist() == [125, 375, 500]
    assert flags.tolist() == [0, 0, 0]
    q2, flags2 = fp.quantize_array(np.asarray([0.1, 0.2]), 1000)
    assert q2.dtype == np.int64 and flags2.dtype == np.uint8


def test_saturation_clips_to_int32():
    assert fp.saturate(fp.INT32_MAX + 5) == (fp.INT32_MAX, True)
    assert fp.saturate(fp.INT32_MIN - 5) == (fp.INT32_MIN, True)
    assert fp.saturate(123) == (123, False)


def test_trunc_division_is_toward_zero():
    assert fp.trunc_div(7, 3) == 2
    assert fp.trunc_div(-7, 3) == -2
    assert fp.trunc_div(7, -3) == -2


def test_derived_digest_and_error_split():
    scale = 1000
    c1, e1 = fp.quantize_scalar(0.6, scale)
    c2, e2 = fp.quantize_scalar(0.4, scale)
    digest, saturated, raw = fp.derived_digest(
        [(c1, 1000), (c2, 2000)], scale
    )
    assert digest == 1400 and saturated is False
    assert raw["raw_before_saturate"] == 1400
    # 0.3331 truncates at 1000 grid -> one quantum of single-value error
    c3, e3 = fp.quantize_scalar("0.3331", scale)
    assert e3 == 1 and c3 == 333
    bounds = fp.derived_error_bound(
        [(c3, e3, 1000, 2), (c2, e2, 2000, 1)], scale
    )
    assert bounds["single"] >= 1
    assert bounds["propagated"] >= 1
    assert bounds["bound"] >= bounds["single"] + bounds["propagated"]


def test_leaf_error_bound_distinguishes_error_kinds():
    exact = fp.leaf_error_bound(0)
    truncated = fp.leaf_error_bound(1)
    assert exact == {"single": 0, "propagated": 0, "bound": 0}
    assert truncated["single"] == 1 and truncated["propagated"] == 0


def test_contract_descriptor_is_stable():
    contract = fp.describe_contract(1000)
    assert contract["quant_version"] == "Q32-S1000"
    assert contract["truncation"] == "toward_zero"
    assert contract["saturation"] == "signed_int32"
