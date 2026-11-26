import pytest

from dispatch_tokens.bucket import TokenBucket
from dispatch_tokens.fixedpoint import (quantize, rescale, saturate,
                                        trunc_div, validate_scale)


def test_refill_uses_integer_math_and_remainder():
    bucket = TokenBucket("n", rate=3, burst=10, scale=100, bucket_key=1,
                         start_ms=0, param_version=1)
    first = bucket.refill(100)
    # 3 * 100 * 100 / 1000 = 30；初始余额已到上限 1000，触发饱和夹取。
    assert first.added == 30
    assert first.before_clamp == 1030
    assert first.after_clamp == 1000
    assert first.clamp_applied is True
    bucket.state.balance = 0
    bucket.state.remainder = 0
    bucket.state.last_ms = 0
    record = bucket.refill(50)
    # 50*3*100/1000 = 15 令牌，余 0。
    assert record.added == 15
    assert record.after_clamp == 15


def test_refill_split_batches_are_identical():
    bucket_one = TokenBucket("n", rate=7, burst=100, scale=10, bucket_key=1,
                             start_ms=0, param_version=1)
    bucket_two = TokenBucket("n", rate=7, burst=100, scale=10, bucket_key=1,
                             start_ms=0, param_version=1)
    bucket_one.state.balance = 0
    bucket_two.state.balance = 0
    one = bucket_one.refill(900)
    split_a = bucket_two.refill(300)
    split_b = bucket_two.refill(900)
    assert one.after_clamp == split_b.after_clamp
    assert bucket_one.state.balance == bucket_two.state.balance
    assert bucket_one.state.remainder == bucket_two.state.remainder
    assert split_a.added + split_b.added == one.added


def test_revision_carryover_does_not_recompute_history():
    bucket = TokenBucket("n", rate=10, burst=5, scale=100, bucket_key=1,
                         start_ms=0, param_version=1)
    bucket.refill(250)
    bucket.try_consume(3)
    # 提交时刻先补充并饱和到旧上限 500（=5 逻辑令牌），再换因子。
    carry = bucket.apply_revision(new_rate=20, new_burst=8, new_scale=1000,
                                  committed_ms=750, new_version=2)
    assert carry["old_scale"] == 100
    assert carry["new_scale"] == 1000
    assert carry["carried_balance"] == 5000
    assert bucket.state.rate == 20
    assert bucket.state.burst == 8
    assert bucket.state.param_version == 2
    after = bucket.refill(1000)
    assert after.rate == 20
    assert after.scale == 1000


def test_rescale_truncation_recorded():
    assert rescale(1234, 1000, 100) == 100
    assert rescale(500, 100, 1000) == 5000
    with pytest.raises(ValueError):
        validate_scale(50)


def test_saturate_and_trunc():
    assert saturate(99, upper=10) == 10
    assert saturate(-3, lower=0) == 0
    assert trunc_div(7, 2) == 3
    assert quantize(4, 100) == 400


def test_refill_record_contains_required_audit_fields():
    bucket = TokenBucket("n", rate=5, burst=8, scale=100, bucket_key=1,
                         start_ms=0, param_version=1)
    record = bucket.refill(400)
    for field_name in ("elapsed_ms", "added", "before_clamp",
                       "after_clamp"):
        assert hasattr(record, field_name)
    assert record.truncation == "TRUNC_TOWARD_ZERO"
    assert record.saturation == "SATURATE"
