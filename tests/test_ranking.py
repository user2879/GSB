from __future__ import annotations

from slowprof import aggregate_samples, build_ranking
from slowprof.models import ProfilePolicy, Sample


def _sample(request_id: str, frames, duration: int, weight: int = 1) -> Sample:
    return Sample(
        request_id=request_id,
        timestamp_ms=len(request_id),
        duration_ms=duration,
        service_id="checkout",
        frames=tuple(frames),
        weight=weight,
    )


def test_ranking_order_and_stable_ties():
    policy = ProfilePolicy(top_k=10)
    samples = [
        _sample("1", ["root.api", "a.method"], 100),
        _sample("2", ["root.api", "a.method", "child.work"], 50),
        _sample("3", ["root.api", "b.method"], 100),
        _sample("4", ["root.api", "y.method"], 50),
        _sample("5", ["root.api", "x.method"], 50),
    ]
    result = aggregate_samples(samples, policy, path="full")
    ranking = build_ranking(result, policy)["ranking"]
    frames = [item["frame"] for item in ranking]
    assert frames[:3] == ["a.method", "b.method", "child.work"]
    assert frames.index("x.method") < frames.index("y.method")


def test_fixed_point_error_bounds_and_error_bound_dominated_tie():
    policy = ProfilePolicy(scale=3, top_k=2)
    samples = [
        _sample("a", ["root.api", "a.hit"], 3),
        _sample("b", ["root.api", "b.hit"], 4),
    ]
    result = aggregate_samples(samples, policy, path="full")
    report = build_ranking(result, policy)
    entries = report["ranking"]
    assert report["reported_k"] == 2
    assert report["error_annotation"]["top_k_contains_true_top_k"] is True
    assert entries[0]["frame"] == "b.hit"
    assert entries[0]["self_ratio_fixed"] == 2
    assert entries[1]["frame"] == "a.hit"
    assert entries[1]["self_ratio_fixed"] == 1
    assert entries[0]["quantization_order_error_bound_dominated"] is True
    assert entries[1]["quantization_order_error_bound_dominated"] is True
    assert report["error_annotation"]["error_bound_dominated_pair_count"] == 2
    assert report["error_annotation"]["combined_pair_error_bound"]["fixed_point_units"] == 1
    assert report["error_annotation"]["top_k_total_ratio_error_upper_bound"] == {
        "numerator": 1,
        "denominator": 3,
        "source_numerator": 2,
        "source_denominator": 6,
        "explanation": "sum of k half-unit per-frame bounds",
    }


def test_quantized_boundary_tie_is_annotated():
    policy = ProfilePolicy(scale=3, top_k=1)
    samples = [
        _sample("a", ["root.api", "a.hit"], 3),
        _sample("b", ["root.api", "b.hit"], 4),
    ]
    report = build_ranking(aggregate_samples(samples, policy, path="full"), policy)
    boundary = report["error_annotation"]["top_k_boundary_check"]
    assert report["ranking"][0]["frame"] == "b.hit"
    assert report["ranking"][0]["self_ratio_fixed"] == 2
    assert boundary["first_excluded_self_ratio_fixed"] == 1
    assert boundary["quantization_order_error_bound_dominated_with_first_excluded"] is True
    assert report["ranking"][0]["quantization_order_error_bound_dominated"] is True
    assert report["error_annotation"]["error_bound_dominated_pair_count"] == 1


def test_k_shortage_and_context_stability(samples_dir, policy, small_interval):
    from slowprof import profile_file

    big_k = ProfilePolicy(top_k=100)
    session = profile_file(samples_dir / "small.ndjson", big_k, small_interval)
    report = session.ranking_report
    assert report["requested_k"] == 100
    assert report["reported_k"] == len(session.aggregation.frame_stats)
    assert report["error_annotation"]["top_k_boundary_check"]["first_excluded_frame"] is None
    first = report["ranking"][0]
    assert first["callers"] == sorted(
        first["callers"], key=lambda item: (-item["inclusive_time_ms"], item["frame"])
    )


def test_top_k_boundary_evidence(samples_dir, policy, small_interval):
    from slowprof import profile_file

    report = profile_file(samples_dir / "small.ndjson", policy, small_interval).ranking_report
    boundary = report["error_annotation"]["top_k_boundary_check"]
    assert boundary["last_selected_frame"] == "billing.charge"
    assert boundary["first_excluded_frame"] == "redis.get"
    assert boundary["excluded_is_not_better"] is True
