from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from workbench.samples import (
    generate_samples,
    make_batch_summary,
    merge_batch_summaries,
    sample_digest,
)


def test_samples_are_deterministic_by_seed_and_shard():
    first = generate_samples(123, "shard-a", 50)
    second = generate_samples(123, "shard-a", 50)
    other = generate_samples(123, "shard-b", 50)
    assert first == second
    assert first != other
    assert sample_digest(123, "shard-a", 50) == sample_digest(123, "shard-a", 50)


def _summary_for_split(rows, split, order=None):
    samples = generate_samples(77, "s9", rows)
    batches = []
    for index, start in enumerate(range(0, rows, split)):
        batches.append(
            make_batch_summary(
                node_id="n1",
                shard_id="s9",
                capacity_version=2,
                samples=samples[start : start + split],
                batch_index=index,
            )
        )
    if order is not None:
        batches = [batches[i] for i in order]
    return merge_batch_summaries(
        batches,
        node_id="n1",
        shard_id="s9",
        sample_rows=rows,
        capacity_version=2,
        fixed_point_exponent=0,
        full_sample_digest=sample_digest(77, "s9", rows),
    )


def test_batch_splits_and_orders_give_same_digest():
    baseline = _summary_for_split(65, 65)
    splits = [1, 3, 7, 13]
    for split in splits:
        merged = _summary_for_split(65, split)
        assert merged["summary_digest"] == baseline["summary_digest"]
    shuffled = _summary_for_split(65, 7, order=[4, 0, 8, 2, 6, 1, 9, 3, 7, 5])
    assert shuffled["summary_digest"] == baseline["summary_digest"]
    assert baseline["participating_rows"] == 65


def test_concurrent_generation_is_stable():
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: sample_digest(9, "parallel", 100), range(32)))
    assert len(set(results)) == 1
