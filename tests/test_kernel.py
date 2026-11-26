import math

import pytest

from decchain.config import DecodeConfig
from decchain.kernel import run_kernel


TOKENS = [100, 101, 102, 103, 104]
LOGPROBS = [0.0, -0.4, -1.1, -2.0, -3.1]


def test_quantized_distribution_sums_to_scale(config):
    result = run_kernel(TOKENS, LOGPROBS, 1.0, config)
    assert result.total_int == config.scale
    assert sum(result.quant.values()) == config.scale
    assert all(v >= 0 for v in result.quant.values())


def test_temperature_changes_integers(config):
    cold = run_kernel(TOKENS, LOGPROBS, 0.5, config).quant
    hot = run_kernel(TOKENS, LOGPROBS, 2.0, config).quant
    assert cold != hot
    # Higher temperature flattens the distribution: top token loses mass.
    assert cold[100] > hot[100]


def test_width_and_tier_change_integers():
    base = DecodeConfig(width=4, tier="ROUND_HALF_UP")
    wider = DecodeConfig(width=5, tier="ROUND_HALF_UP")
    floor = DecodeConfig(width=4, tier="FLOOR")
    a = run_kernel(TOKENS, LOGPROBS, 1.0, base)
    b = run_kernel(TOKENS, LOGPROBS, 1.0, wider)
    c = run_kernel(TOKENS, LOGPROBS, 1.0, floor)
    assert a.quant != b.quant
    raw_round = {r.token: r.raw_int for r in a.records}
    raw_floor = {r.token: r.raw_int for r in c.records}
    assert raw_round != raw_floor
    assert b.total_int == wider.scale
    assert c.total_int == base.scale


def test_reference_is_one_digit_wider(config):
    result = run_kernel(TOKENS, LOGPROBS, 1.0, config)
    assert result.ref_width == config.width + 1
    assert sum(result.ref_quant.values()) == result.ref_scale


def test_records_carry_residuals(config):
    result = run_kernel(TOKENS, LOGPROBS, 1.0, config)
    for record in result.records:
        assert record.final_int == record.raw_int + record.residual
        assert math.isfinite(record.scaled_logit)


def test_rejects_non_finite_and_bad_temperature(config):
    with pytest.raises(ValueError):
        run_kernel(TOKENS, [0.0, float("nan"), -1.0, -2.0, -3.0], 1.0, config)
    with pytest.raises(ValueError):
        run_kernel(TOKENS, LOGPROBS, 0.0, config)
