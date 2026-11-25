"""要求4：迟到批窗口归属、预算重算与稳定去重。"""

from slo_burn.late import recompute_with_late, windows_owning_minute
from slo_burn.models import MinuteBatch, ServiceSpec
from slo_burn.stream import merge_streams


def _spec():
    return ServiceSpec(
        service_id="svc",
        target_permyriad=9990,
        window_minutes=5,
        burn_threshold=20000,
        budget_floor=0,
    )


def _regular():
    return [
        MinuteBatch("svc", line=i + 1, batch_seq=i + 1, minute=i,
                    successes=100, failures=1, stream="regular")
        for i in range(10)
    ]


def test_windows_owning_minute_reference_chain():
    assert windows_owning_minute(3, window_minutes=5, horizon=8) == [3, 4, 5, 6, 7]
    assert windows_owning_minute(7, window_minutes=5, horizon=8) == [7, 8]


def test_late_batch_recomputes_owned_windows():
    late = [
        MinuteBatch("svc", line=101, batch_seq=1, minute=2,
                    successes=100, failures=40, stream="late")
    ]
    conclusions, points, coverage, impacts = recompute_with_late(
        spec=_spec(), regular=_regular(), late=late, path="exact"
    )
    impacted_minutes = [item.minute for item in impacts]
    # 分钟 2 归属窗口终点 2..6
    assert impacted_minutes == [2, 3, 4, 5, 6]
    assert all("remaining_budget_scaled" in item.changed_fields for item in impacts)
    assert len(points) == 10


def test_duplicate_minute_uses_stable_rule_and_records_coverage():
    late = [
        MinuteBatch("svc", line=101, batch_seq=1, minute=2,
                    successes=100, failures=40, stream="late"),
        MinuteBatch("svc", line=102, batch_seq=2, minute=2,
                    successes=100, failures=5, stream="late"),
    ]
    _, points, coverage, _ = recompute_with_late(
        spec=_spec(), regular=_regular(), late=late, path="exact"
    )
    point = next(item for item in points if item.minute == 2)
    # 同流内行号小者优先
    assert point.failures == 40
    assert point.source_line == 101
    assert any(record.minute == 2 and record.covered_line == 102 for record in coverage)


def test_merge_is_deterministic_across_input_order():
    regular = _regular()
    late_a = [
        MinuteBatch("svc", 101, 1, 2, 100, 40, "late"),
        MinuteBatch("svc", 102, 2, 4, 100, 30, "late"),
    ]
    merged_a, coverage_a = merge_streams(regular, late_a)
    merged_b, coverage_b = merge_streams(list(reversed(regular)), list(reversed(late_a)))
    assert merged_a == merged_b
    assert coverage_a == coverage_b


def test_late_impact_sorted_by_service_and_minute():
    late = [MinuteBatch("svc", 101, 1, 0, 100, 50, "late")]
    _, _, _, impacts = recompute_with_late(
        spec=_spec(), regular=_regular(), late=late, path="exact"
    )
    minutes = [item.minute for item in impacts]
    assert minutes == sorted(minutes)
