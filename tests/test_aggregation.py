"""Fixed-point aggregation, error bounds, saturation and determinism."""
from __future__ import annotations

from dataclasses import asdict

from obsflow.aggregation import (
    aggregate_all,
    aggregate_window,
    group_by_metric,
    ordered_records,
    quantize_value,
    window_slices,
)
from obsflow.canonical import canonical_bytes
from obsflow.config import load_engine_config, load_source_set
from obsflow.codec import load_stream
from obsflow.models import INT64_MAX, NormalizedRecord, WindowParams


def _records(sample_dir):
    config = load_source_set(sample_dir["declaration_path"])
    records = []
    for decl in config.sources:
        records.extend(load_stream(
            f"{sample_dir['output_dir']}/{decl.file_path}", decl))
    return ordered_records(records)


def test_quantize_truncation_and_bounds():
    # raw 1 at scale 10000, session scale 100 -> trunc(1*100/10000)=0
    q, sat = quantize_value(1, 10_000, 100)
    assert q == 0 and sat is False
    q, _ = quantize_value(-1, 10_000, 100)  # toward zero, not floor
    assert q == 0
    q, _ = quantize_value(250, 100, 100)
    assert q == 250
    huge, saturated = quantize_value(10**30, 1, 1)
    assert saturated is True and huge == INT64_MAX


def test_window_error_bounds_are_strict(sample_dir):
    params = load_engine_config(sample_dir["window_path"]).windows
    windows = aggregate_all(group_by_metric(_records(sample_dir)), params)
    assert windows
    for window in windows:
        assert window.value_error_bound_q == 1
        assert window.sum_error_bound_q == window.count
        assert window.truncation == "toward_zero"
        assert window.saturation in {"none", "value_or_sum_clamped",
                                     "mean_clamped"}


def test_batching_and_advance_rhythm_byte_identical(sample_dir):
    params = load_engine_config(sample_dir["window_path"]).windows
    records = _records(sample_dir)

    full = aggregate_all(group_by_metric(records), params)
    # Simulate arbitrary batch splits then a final merge.
    chunks = [records[:7], records[7:20], records[20:47], records[47:]]
    merged: list[NormalizedRecord] = []
    partial_digests = []
    for chunk in chunks:
        merged = ordered_records(merged + chunk)
        partial = aggregate_all(group_by_metric(merged), params)
        partial_digests.append(
            [canonical_bytes(asdict(w)) for w in partial])
    batched = aggregate_all(group_by_metric(merged), params)
    assert [canonical_bytes(asdict(w)) for w in batched] == \
           [canonical_bytes(asdict(w)) for w in full]
    # Every already-final window stays stable across later batches. Windows
    # are keyed by (metric, start_seq) because window indices shift as the
    # per-metric record count grows across batches.
    final_by_key = {
        (w.metric_key, w.start_seq): canonical_bytes(asdict(w))
        for w in batched
    }
    for prior_windows in partial_digests:
        prior = aggregate_all  # noqa: keep reference readable
        partial = aggregate_all(group_by_metric(merged), params)
    # Recompute each partial snapshot explicitly and compare by stable key.
    merged_check: list[NormalizedRecord] = []
    for chunk in chunks:
        merged_check = ordered_records(merged_check + chunk)
        for window in aggregate_all(group_by_metric(merged_check), params):
            key = (window.metric_key, window.start_seq)
            assert canonical_bytes(asdict(window)) == final_by_key[key]


def test_window_slices_only_full_windows():
    assert window_slices(4, 5, 1) == []
    assert window_slices(5, 5, 1) == [(0, 5)]
    assert window_slices(7, 5, 2) == [(0, 5), (2, 7)]


def test_mean_truncates_toward_zero(sample_dir):
    params = WindowParams(window_size=3, slide_step=1, scale=1)
    records = [
        NormalizedRecord("s", 1, 1, "m", -1, 1, 1, (), {}, 2),
        NormalizedRecord("s", 2, 2, "m", -1, 1, 1, (), {}, 3),
        NormalizedRecord("s", 3, 3, "m", -2, 1, 1, (), {}, 4),
    ]
    window = aggregate_window(records, "m", 0, params, 1)
    assert window.sum_q == -4
    assert window.mean_q == -1  # trunc(-4/3) toward zero
