from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from slowprof import (
    aggregate_file,
    aggregate_samples,
    build_ranking,
    canonical_core_bytes,
    profile_file,
)
from slowprof.ndjson import load_samples


def _core(samples, policy, path):
    result = aggregate_samples(samples, policy, path=path)
    ranking = build_ranking(result, policy)
    return canonical_core_bytes(result, ranking)


def test_automatic_path_selection(samples_dir, policy, small_interval, large_interval):
    small = aggregate_file(samples_dir / "small.ndjson", policy, small_interval)
    large = aggregate_file(samples_dir / "large.ndjson", policy, large_interval)
    assert small.selection.path == "full"
    assert small.selection.forced is False
    assert small.folded_stacks is not None
    assert large.selection.path == "streaming"
    assert large.selection.forced is False
    assert large.folded_stacks is None
    assert large.selection.unique_stack_count == 125
    assert large.selection.total_weight == 64760
    assert large.memory_estimate["streaming_no_path_strings"] is True
    assert large.memory_estimate["fixed_depth_working_buffer_bytes"] == 64 * 8


def test_full_and_streaming_are_byte_identical_small(samples_dir, policy, small_interval):
    full = aggregate_file(samples_dir / "small.ndjson", policy, small_interval, force_path="full")
    streaming = aggregate_file(
        samples_dir / "small.ndjson", policy, small_interval, force_path="streaming"
    )
    assert canonical_core_bytes(full, build_ranking(full, policy)) == canonical_core_bytes(
        streaming, build_ranking(streaming, policy)
    )


def test_full_and_streaming_are_byte_identical_large(samples_dir, policy, large_interval):
    full = aggregate_file(samples_dir / "large.ndjson", policy, large_interval, force_path="full")
    streaming = aggregate_file(
        samples_dir / "large.ndjson", policy, large_interval, force_path="streaming"
    )
    assert len(full.folded_stacks or ()) == 125
    assert canonical_core_bytes(full, build_ranking(full, policy)) == canonical_core_bytes(
        streaming, build_ranking(streaming, policy)
    )


def test_fold_merges_path_weights(samples_dir, policy, small_interval):
    result = aggregate_file(samples_dir / "small.ndjson", policy, small_interval)
    folds = dict(result.folded_stacks or ())
    assert folds[("api.handle", "db.query", "sql.execute")] == 6
    assert folds[("api.handle", "billing.charge", "http.call")] == 3


def test_self_and_cumulative_metric_contract(samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    frames = {stat.frame: stat for stat in session.aggregation.frame_stats}
    assert frames["api.handle"].self_time_ms == 0
    assert frames["api.handle"].cumulative_time_ms == 2600
    assert frames["api.handle"].participant_weight == 19
    assert frames["sql.execute"].self_time_ms == 600
    assert frames["sql.execute"].cumulative_time_ms == 600
    assert frames["billing.charge"].self_time_ms == 90
    assert frames["billing.charge"].cumulative_time_ms == 1570
    assert sum(stat.self_time_ms for stat in frames.values()) == 2725
    assert session.aggregation.total_time_ms == 2725


def test_context_direct_callers_and_callees(samples_dir, policy, small_interval):
    session = profile_file(samples_dir / "small.ndjson", policy, small_interval)
    top = session.ranking_report["ranking"][0]
    assert top["frame"] == "http.call"
    assert top["callers"] == [
        {"frame": "billing.charge", "inclusive_time_ms": 1480, "participant_weight": 3}
    ]
    assert top["callees"] == []


def test_duplicate_frame_names_are_deduped_per_stack(policy):
    from slowprof.models import Sample

    sample = Sample(
        request_id="r",
        timestamp_ms=1,
        duration_ms=10,
        service_id="checkout",
        frames=("a.top", "b.mid", "a.top"),
        weight=3,
    )
    full = aggregate_samples([sample], policy, path="full")
    streaming = aggregate_samples([sample], policy, path="streaming")
    assert canonical_core_bytes(full, build_ranking(full, policy)) == canonical_core_bytes(
        streaming, build_ranking(streaming, policy)
    )
    stats = {stat.frame: stat for stat in full.frame_stats}
    assert stats["a.top"].self_time_ms == 30
    assert stats["a.top"].cumulative_time_ms == 30
    assert stats["a.top"].participant_weight == 3
    assert stats["a.top"].occurrence_count == 1
    assert stats["b.mid"].self_time_ms == 0
    assert stats["b.mid"].cumulative_time_ms == 30


def test_input_order_independence(samples_dir, policy, small_interval):
    samples, _ = load_samples(samples_dir / "small.ndjson", policy, small_interval)
    reordered = tuple(reversed(samples))
    assert _core(samples, policy, "full") == _core(reordered, policy, "full")
    assert _core(samples, policy, "streaming") == _core(reordered, policy, "streaming")


def test_concurrent_profiles_are_stable(samples_dir, policy, large_interval):
    expected = profile_file(
        samples_dir / "large.ndjson", policy, large_interval
    ).conclusion_sha256

    def run_one(_):
        return profile_file(samples_dir / "large.ndjson", policy, large_interval).conclusion_sha256

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_one, range(8)))
    assert results == [expected] * len(results)
