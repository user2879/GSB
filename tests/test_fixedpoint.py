from __future__ import annotations

from reco_shard_diag.fixedpoint import aggregate_groups, evaluate_watermark, quantize_signed


def test_aggregate_groups_uses_deterministic_order_and_saturation():
    samples = [
        {"group": "g1", "values": [900, -900]},
        {"group": "g0", "values": [500, 500]},
        {"group": "g1", "values": [800, 100]},
    ]
    summary = aggregate_groups(samples, 2, 10)
    assert summary["group_order"] == ["g0", "g1"]
    assert summary["groups"][0]["raw_sum"] == [500, 500]
    assert summary["groups"][1]["raw_sum"] == [1700, -800]
    assert summary["groups"][1]["quantized_sum"] == [511, -512]
    assert summary["overflow_count"] == 2


def test_high_width_reference_changes_quantized_result():
    samples = [{"group": "g0", "values": [1200, 1200]}]
    low = aggregate_groups(samples, 2, 10)
    high = aggregate_groups(samples, 2, 12)
    assert low["groups"][0]["quantized_sum"] == [511, 511]
    assert high["groups"][0]["quantized_sum"] == [1200, 1200]


def test_watermark_floor_and_boundaries():
    watermark = evaluate_watermark(4, 3, 7000, 10000, 16)
    assert watermark["soft_limit"] == 2
    assert watermark["hard_limit"] == 4
    assert watermark["state"] == "SOFT_WARN"
    assert watermark["code"] == "E_QUOTA_SOFT_LIMIT"
    at_hard = evaluate_watermark(4, 4, 7000, 10000, 16)
    assert at_hard["state"] == "SOFT_WARN"
    above_hard = evaluate_watermark(4, 5, 7000, 10000, 16)
    assert above_hard["state"] == "HARD_REJECT"
    assert above_hard["code"] == "E_QUOTA_HARD_LIMIT"


def test_quantize_signed_bounds():
    assert quantize_signed(999, 8) == 127
    assert quantize_signed(-999, 8) == -128
