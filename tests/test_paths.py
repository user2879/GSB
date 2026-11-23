from __future__ import annotations

from ossdiag.content import LEAF_SIZE, generate_segment
from ossdiag.digest import digest_chunk, make_aggregator, object_digest
from ossdiag.models import load_spec
from ossdiag.planner import choose_path

from conftest import MiB, object_doc, spec_doc


def _digest_object_with_path(spec, path: str, chunk_size: int) -> str:
    descriptors = []
    for shard_index in range(spec.shard_count):
        start, end = spec.shard_bounds(shard_index)
        shard_len = end - start
        aggregator = make_aggregator(path)
        relative = 0
        while relative < shard_len:
            length = min(chunk_size, shard_len - relative)
            data = generate_segment(spec.object_key, shard_index, spec.seed_for(shard_index), relative, length)
            digests = digest_chunk(spec.object_key, shard_index, relative, shard_len, data)
            for _, leaf_hex in digests.leaves:
                aggregator.add_leaf(bytes.fromhex(leaf_hex))
            relative += length
        root = aggregator.finalize()
        descriptors.append((shard_index, start, end, root))
    return object_digest(spec.object_key, spec.total_bytes, descriptors).hex()


def test_all_paths_and_batch_boundaries_produce_identical_object_digest() -> None:
    item = load_spec_data_item = load_spec_data_from(
        spec_doc(objects=[object_doc(total_bytes=5 * LEAF_SIZE + 123, shard_size=2 * LEAF_SIZE, batch_size=LEAF_SIZE)])
    )
    values = set()
    for path in ("direct", "sliding", "hierarchical"):
        for chunk_size in (LEAF_SIZE, 2 * LEAF_SIZE):
            values.add(_digest_object_with_path(item, path, chunk_size))
    assert len(values) == 1


def load_spec_data_from(doc):
    from ossdiag.models import load_spec_data

    return load_spec_data(doc).objects[0]


def test_deterministic_content_is_batch_independent() -> None:
    full = generate_segment("key", 3, 77, 0, 5 * LEAF_SIZE + 19)
    pieces = []
    offset = 0
    for step in (1, 7, 12345, LEAF_SIZE, 999):
        if offset >= len(full):
            break
        length = min(step, len(full) - offset)
        pieces.append(generate_segment("key", 3, 77, offset, length))
        offset += length
    if offset < len(full):
        pieces.append(generate_segment("key", 3, 77, offset, len(full) - offset))
    assert b"".join(pieces) == full


def test_path_selection_is_deterministic_and_marks_boundary() -> None:
    session = load_spec("samples/demo_spec.json")
    choices = {item.object_key: choose_path(item, session.tie_threshold_work) for item in session.objects}
    assert choices["scale/tiny-1m.bin"].path == "direct"
    assert choices["scale/medium-32m.bin"].path == "sliding"
    assert choices["scale/medium-32m.bin"].boundary_case is True
    assert choices["scale/large-256m.bin"].path == "hierarchical"
    for selection in choices.values():
        assert selection.total_bytes > 0
        assert selection.shard_count > 0
        assert selection.batch_count > 0
        assert len(selection.estimates) == 3
        assert all(estimate.score >= 0 for estimate in selection.estimates)


def test_tiny_object_selects_direct_path() -> None:
    item = load_spec_data_from(spec_doc(objects=[object_doc(total_bytes=64 * 1024)]))
    choice = choose_path(item, 32)
    assert choice.path == "direct"
    assert choice.boundary_case is False
