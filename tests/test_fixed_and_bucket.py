from __future__ import annotations

import pytest

from qosfdiag.fixed import FixedPointSpec, ceil_div
from qosfdiag.token_bucket import TokenBucket


def test_scale_contract() -> None:
    assert FixedPointSpec(0).tick_ns == 1_000_000_000
    assert FixedPointSpec(3).tick_ns == 1_000_000
    assert FixedPointSpec(6).tick_ns == 1_000
    assert FixedPointSpec(3).token_units_per_byte == 1000
    with pytest.raises(ValueError):
        FixedPointSpec(-1)
    with pytest.raises(ValueError):
        FixedPointSpec(7)


def test_rate_is_exact_and_truncation_rules_are_declared() -> None:
    spec = FixedPointSpec(3)
    analysis = spec.rate_quantization_analysis(1234)
    assert analysis["represented_units_per_tick"] == 1234 * 125
    assert analysis["represented_bytes_per_s"] == 1234 * 125
    assert analysis["rate_quantization_error_bytes_per_s"] == 0
    assert analysis["refill_error_units_per_boundary"] == 1234 * 125
    assert spec.quantize_time_ns(1_999_999) == 1
    assert spec.as_dict()["delay_rounding"].startswith("ceiling")


def test_queue_delay_uses_ceiling_integer_ticks() -> None:
    spec = FixedPointSpec(3)
    assert spec.queue_delay_ticks(1, 1) == 1
    assert spec.queue_delay_ticks(1001, 1000) == 2
    assert ceil_div(10, 3) == 4
    delay = spec.delay_quantization_analysis()
    assert delay["queue_delay_error_bound_ns"] == 2 * spec.tick_ns


def test_token_bucket_saturates_and_consumes_integers() -> None:
    bucket = TokenBucket.create(
        scale=3,
        committed_rate_kbps=8,
        burst_bytes=1000,
    )
    # 8 kbps = 1000 bytes/s, represented as 1000 token units/tick at scale 3.
    assert bucket.rate_units_per_tick == 1000
    assert bucket.tokens == 1_000_000
    assert bucket.refill(10_000) == 1_000_000
    ok, before, after = bucket.consume(200_000, 10_000)
    assert ok and before == 1_000_000 and after == 800_000
    assert bucket.release_delay_ticks(300_000) == 0
    assert bucket.release_delay_ticks(900_000) == 100
    ok, _, after = bucket.consume(800_001, 10_000)
    assert not ok and after == 800_000
    ok, _, after = bucket.consume(800_001, 20_000)
    assert ok and after == 199_999


def test_bucket_rejects_time_reversal() -> None:
    bucket = TokenBucket.create(scale=3, committed_rate_kbps=1, burst_bytes=10)
    bucket.refill(5)
    with pytest.raises(ValueError):
        bucket.refill(4)
