from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from waterlease.hashing import canonical_bytes, sha256_hex
from waterlease.sampling import (
    aggregate_samples,
    build_summary_core,
    combine_aggregates,
    generate_samples,
    split_batches,
)
from waterlease.water import FixedPointMeter, summarize_attribution


def test_samples_are_deterministic_for_seed_and_shard_id() -> None:
    first = generate_samples(1701, "s1", 128)
    second = generate_samples(1701, "s1", 128)
    third = generate_samples(1701, "s2", 128)
    assert canonical_bytes(first) == canonical_bytes(second)
    assert canonical_bytes(first) != canonical_bytes(third)


def test_order_batch_split_and_concurrency_do_not_change_digest() -> None:
    rows = generate_samples(1701, "shard-x", 144)
    full = aggregate_samples(rows)

    reordered = [rows[(index * 37) % len(rows)] for index in range(len(rows))]
    reordered_full = aggregate_samples(reordered)
    assert sha256_hex(full) == sha256_hex(reordered_full)

    for batch_size in (1, 7, 32, 144):
        batches = split_batches(rows, batch_size)
        combined = combine_aggregates(aggregate_samples(batch) for batch in batches)
        assert sha256_hex(combined) == sha256_hex(full)

    def one_run(_: int) -> str:
        local_rows = generate_samples(1701, "shard-x", 144)
        aggregates = [aggregate_samples(batch) for batch in split_batches(local_rows, 13)]
        summary = build_summary_core(
            node_id="node-a",
            shard_id="shard-x",
            capacity_version="cap-v1",
            aggregate=combine_aggregates(aggregates),
        )
        return summary["aggregate_digest"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(one_run, range(32)))
    assert len(set(digests)) == 1


def test_summary_binds_required_identity_and_counts() -> None:
    aggregate = aggregate_samples(generate_samples(7, "s", 40))
    summary = build_summary_core(
        node_id="node-7",
        shard_id="shard-7",
        capacity_version="cap-9",
        aggregate=aggregate,
        upstream_digests={"u": "abc"},
    )
    assert summary["node_id"] == "node-7"
    assert summary["shard_id"] == "shard-7"
    assert summary["capacity_version"] == "cap-9"
    assert summary["sample_rows"] == 40
    assert 0 < summary["participated_rows"] <= 40
    assert summary["aggregate_digest"]


def test_fixed_point_ratio_truncates_toward_floor_and_reports_error() -> None:
    meter = FixedPointMeter(2)
    ratio = meter.quantize_ratio(1, 3, name="test")
    assert ratio["quantized_q"] == 33
    assert ratio["remainder"] == 1
    assert ratio["truncation_error_num"] == 1
    assert ratio["truncation_error_den"] == 300


def test_global_error_summary_reports_dominance_threshold() -> None:
    meter = FixedPointMeter(3)
    records = []
    for index in range(3):
        records.append(
            {
                "scale_exp": 3,
                "segments": {
                    "capacity_version_chain": {"attribution_units_1e9": 10},
                    "lease_generation": {"attribution_units_1e9": 5},
                    "schedule_queue": {"attribution_units_1e9": 85},
                },
            }
        )
    summary = summarize_attribution(records)
    assert summary["dominant_segment"] == "schedule_queue"
    assert summary["dominance_threshold_bp"] == 5000
    assert summary["global_budget_aggregation_error_upper_bound"]["fixed_q_units_bound"] == 3
