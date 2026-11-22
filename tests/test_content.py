from edge_lease.content import (
    canonical_json,
    content_hash,
    fnv1a_32,
    generate_bytes,
    shard_hash,
    shard_hit_digest,
    shard_intervals,
)


def test_generated_bytes_are_deterministic():
    first = generate_bytes(42, "media/a.bin", 64)
    second = generate_bytes(42, "media/a.bin", 64)
    other_seed = generate_bytes(43, "media/a.bin", 64)
    other_path = generate_bytes(42, "media/b.bin", 64)
    assert first == second
    assert first != other_seed and first != other_path
    assert len(first) == 64


def test_hashes_are_stable_32bit():
    data = generate_bytes(7, "x", 128)
    assert fnv1a_32(data) == content_hash(7, "x", 128)
    assert 0 <= fnv1a_32(data) <= 0xFFFFFFFF
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_shard_intervals_cover_full_content():
    intervals = shard_intervals(
        [{"start": 0, "end": 4}, {"start": 4, "end": 10}], 10
    )
    assert intervals == [(0, 4), (4, 10)]
    full = generate_bytes(9, "p", 10)
    assert shard_hash(9, "p", 0, 4, 10) == fnv1a_32(full[0:4])


def test_invalid_shard_intervals_raise():
    import pytest

    with pytest.raises(ValueError):
        shard_intervals([{"start": 0, "end": 4}, {"start": 5, "end": 10}], 10)
    with pytest.raises(ValueError):
        shard_intervals([{"start": 0, "end": 11}], 10)


def test_shard_hit_digest_is_order_sensitive():
    a = shard_hit_digest([1, 2, 3])
    b = shard_hit_digest([3, 2, 1])
    assert a != b and a == shard_hit_digest([1, 2, 3])
