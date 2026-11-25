"""Requirement 3: seed-deterministic generation, null bitmap, load-order and
batch-split independence."""

import random

from obs_pipeline import gen
from obs_pipeline.records import decode_stream, encode_line


def test_same_seed_bit_identical(sample_spec):
    for source in sample_spec.sources:
        first = gen.generate_source_records(source)
        second = gen.generate_source_records(source)
        assert gen.records_digest(first) == gen.records_digest(second)
        nulls_a = [i for i, r in enumerate(first) if r.value_fixed is None]
        nulls_b = [i for i, r in enumerate(source.null_bitmap) if r == 1]
        assert nulls_a == nulls_b


def test_load_order_independence(sample_spec):
    for source in sample_spec.sources:
        canonical = gen.render_stream_lines(source)
        shuffled = gen.render_stream_lines(source, arrival_order=True)
        assert shuffled != canonical or source.record_count <= 1
        # Loader restores canonical order by sorting on obs_seq.
        restored = sorted(shuffled, key=lambda ln: int(ln.split("|")[1]))
        assert gen.stream_digest(restored) == gen.stream_digest(canonical)
        records = decode_stream(restored, source)
        reference = decode_stream(canonical, source)
        assert gen.records_digest(records) == gen.records_digest(reference)
        if source.record_format == 3:
            # v3 is lossless: disk round-trip matches in-memory generation.
            assert gen.records_digest(records) == gen.records_digest(
                gen.generate_source_records(source)
            )


def test_batch_split_independence(sample_spec):
    for source in sample_spec.sources:
        lines = gen.render_stream_lines(source)
        full = decode_stream(lines, source)
        cuts = (0, 1, 7, source.record_count)
        chunks = []
        for start, end in zip(cuts, cuts[1:]):
            chunk_source = _slice_source(source, start, end)
            chunks.append(decode_stream(lines[start:end], chunk_source))
        reassembled = [r for chunk in chunks for r in chunk]
        assert gen.records_digest(reassembled) == gen.records_digest(full)


def _slice_source(source, start, end):
    from obs_pipeline.spec import Source

    return Source(
        id=source.id, seed=source.seed, record_count=end - start,
        record_format=source.record_format, metrics=source.metrics,
        default_scale=source.default_scale, null_bitmap=source.null_bitmap[start:end],
    )


def test_arrival_order_deterministic(sample_spec):
    for source in sample_spec.sources:
        assert gen.generate_arrival_order(source) == gen.generate_arrival_order(source)
        assert sorted(gen.generate_arrival_order(source)) == list(range(source.record_count))
