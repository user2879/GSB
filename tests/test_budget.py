"""要求2/3：双路径判定、路径选择规则与偏差界。"""

import pytest

from slo_burn.budget import (
    choose_path,
    deviation_bounds,
    run_exact,
    run_incremental,
)
from slo_burn.divergence import verify_paths
from slo_burn.errors import DivergenceError
from slo_burn.models import MinutePoint, PATH_EXACT, PATH_INCREMENTAL, ServiceSpec


def _points(minutes, successes=100, failures=5):
    return [
        MinutePoint(
            minute=minute,
            successes=successes,
            failures=failures,
            source_line=minute + 1,
            source_batch_seq=minute + 1,
            source_stream="regular",
        )
        for minute in range(minutes)
    ]


def _spec(window_minutes=30, target=9990, burn=20000):
    return ServiceSpec(
        service_id="svc",
        target_permyriad=target,
        window_minutes=window_minutes,
        burn_threshold=burn,
        budget_floor=0,
    )


def test_exact_window_sliding_budget_identity():
    spec = _spec(window_minutes=10)
    conclusions = run_exact(spec, _points(25))
    for item in conclusions:
        expected_remaining = item.budget_total_scaled - item.failures * 10000
        assert item.remaining_budget_scaled == expected_remaining
    # 窗口填满后总量稳定
    assert conclusions[-1].total == 1050
    assert conclusions[-1].failures == 50


def test_choose_path_explicit_rules():
    short = choose_path(window_minutes=30, batch_count=100)
    assert short.path == PATH_EXACT
    long_enough = choose_path(window_minutes=120, batch_count=150)
    assert long_enough.path == PATH_INCREMENTAL
    assert long_enough.incremental_state_ratio == 8 / 120
    few_batches = choose_path(window_minutes=120, batch_count=3)
    assert few_batches.path == PATH_EXACT


def test_incremental_state_is_bounded_and_reuses_chunks():
    # 增量路径以块级环形缓冲复用状态：状态条目数远小于逐分钟精确路径，
    # 且与精确路径的偏差被尾部边界块上界严格约束。
    spec = _spec(window_minutes=32)
    points = _points(90)
    exact = run_exact(spec, points)
    incremental, states = run_incremental(spec, points, chunk_capacity=8)
    by_minute = {point.minute: point for point in points}
    for left, right in zip(exact, incremental):
        bounds = deviation_bounds(left, 8, by_minute)
        assert abs(right.total - left.total) <= bounds["bound_total_requests"]
        assert abs(right.failures - left.failures) <= bounds["bound_failures"]
    # 长窗口下环形块状态规模显著小于逐分钟缓冲
    assert states[-1].ring_chunks <= 6
    assert states[-1].entries < spec.window_minutes


def test_incremental_error_is_confined_to_boundary_chunk():
    spec = _spec(window_minutes=30)
    points = _points(90)
    exact = run_exact(spec, points)
    incremental, _ = run_incremental(spec, points, chunk_capacity=8)
    by_minute = {point.minute: point for point in points}
    for left, right in zip(exact, incremental):
        bounds = deviation_bounds(left, 8, by_minute)
        delta = abs(right.remaining_budget_scaled - left.remaining_budget_scaled)
        assert delta <= bounds["bound_remaining_scaled"]


def test_verify_paths_within_bound_on_misaligned_window():
    spec = _spec(window_minutes=30)
    points = _points(90)
    exact = run_exact(spec, points)
    incremental, _ = run_incremental(spec, points, chunk_capacity=8)
    records = verify_paths("svc", exact, incremental, 8, points)
    # 所有偏差必须可解释且不越界（越界会抛 E_DIVERGE）
    assert isinstance(records, list)


def test_deviation_bounds_are_quantified():
    spec = _spec(window_minutes=30)
    points = _points(90)
    exact = run_exact(spec, points)
    by_minute = {point.minute: point for point in points}
    bounds = deviation_bounds(exact[40], 8, by_minute)
    assert bounds["bound_remaining_scaled"] == 10000 * bounds["bound_total_requests"]
    assert bounds["bound_total_requests"] >= 0
    assert "basis" in bounds


def test_divergence_is_raised_on_corrupted_incremental():
    spec = _spec(window_minutes=30)
    points = _points(90)
    exact = run_exact(spec, points)
    incremental, _ = run_incremental(spec, points, chunk_capacity=8)
    by_minute = {point.minute: point for point in points}
    bounds = deviation_bounds(exact[45], 8, by_minute)
    # 破坏幅度必须超过偏差上界，才能证明 E_DIVERGE 不是误报
    corruption = bounds["bound_remaining_scaled"] + 10_000_000
    broken = list(incremental)
    broken[45] = broken[45].__class__(
        **{
            **broken[45].__dict__,
            "remaining_budget_scaled": broken[45].remaining_budget_scaled - corruption,
        }
    )
    with pytest.raises(DivergenceError) as exc:
        verify_paths("svc", exact, broken, 8, points)
    assert exc.value.code == "E_DIVERGE"
    assert exc.value.context["service_id"] == "svc"
    assert exc.value.context["minute"] == 45
    assert exc.value.context["field"] == "remaining_budget_scaled"


def test_zero_traffic_has_defined_burn_rate():
    points = [
        MinutePoint(0, 0, 0, 1, 1, "regular"),
        MinutePoint(1, 0, 0, 2, 2, "regular"),
    ]
    conclusions = run_exact(_spec(), points)
    assert all(item.burn_rate_num == 0 for item in conclusions)
    assert all(item.alert is False for item in conclusions)
