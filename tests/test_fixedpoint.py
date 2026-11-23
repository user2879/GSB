from __future__ import annotations

from ossdiag.fixedpoint import FixedPoint, ObjectNumericStats, compute_object_stats, rank_objects_by_completion


def test_fixed_point_floor_and_saturation() -> None:
    fixed = FixedPoint(3)
    assert fixed.scale == 1000
    assert fixed.ratio(1, 3) == 333
    assert fixed.ratio(0, 3) == 0
    assert fixed.saturate(-10) == 0
    assert fixed.saturate(10**30) == (1 << 63) - 1
    assert fixed.policy()["rounding"].startswith("floor")


def test_object_stats_and_error_bounds() -> None:
    fixed = FixedPoint(6)
    stats = compute_object_stats(
        object_key="a",
        total_bytes=300,
        shard_rows=[
            {"byte_start": 0, "byte_end": 100, "progress_cursor": 100, "status": "complete", "digest_duration_ns": 1_000_000},
            {"byte_start": 100, "byte_end": 300, "progress_cursor": 200, "status": "uploading", "digest_duration_ns": 3_000_000},
        ],
        fixed=fixed,
        lease_grants=4,
        renew_count=1,
    )
    assert stats.completed_bytes == 200
    assert stats.completion_fixed == 666666
    assert stats.average_digest_ms_fixed == 2_000_000
    assert stats.renewal_rate_fixed == 250000
    assert stats.comparison_error_bound == 2


def test_error_bound_ties_use_lexicographic_object_key() -> None:
    fixed = FixedPoint(6)
    stats = [
        ObjectNumericStats("zeta", 100, 100, 1_000_000, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2),
        ObjectNumericStats("alpha", 100, 100, 999_999, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2),
        ObjectNumericStats("beta", 100, 50, 500_000, 1, 2, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2),
    ]
    ranked, pairs = rank_objects_by_completion(stats, fixed)
    assert [item["object_key"] for item in ranked] == ["alpha", "zeta", "beta"]
    assert ranked[0]["ordering_affected_by_error_bound"] is True
    assert pairs and pairs[0]["bound_fixed"] == 2
