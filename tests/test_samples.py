from __future__ import annotations

from gsb.samples import (
    aggregate_groups,
    canonical_json_bytes,
    final_aggregate,
    generate_samples,
    sample_digest,
)
from gsb.spec import parse_spec
from helpers import base_raw_spec


def test_samples_and_digest_are_independent_of_batch_slicing() -> None:
    spec = parse_spec(base_raw_spec())
    node = spec.node_map()["node-c"]
    shard = spec.shard_map()["join-ab"]
    rows = generate_samples(node, shard)
    assert canonical_json_bytes(rows) == canonical_json_bytes(generate_samples(node, shard))
    assert sample_digest(rows) == sample_digest(generate_samples(node, shard))

    def final_for(batch_size: int) -> dict:
        stages = [
            aggregate_groups(rows[offset : offset + batch_size], shard.group_count, node.fixed_scale)
            for offset in range(0, len(rows), batch_size)
        ]
        return final_aggregate(node, shard, "cap-test", rows, stages)

    one = final_for(1)
    seven = final_for(7)
    all_rows = final_for(len(rows))
    assert one["aggregate_digest"] == seven["aggregate_digest"] == all_rows["aggregate_digest"]
    assert one["sample_digest"] == all_rows["sample_digest"]
    assert one["participating_rows"] == len(rows)

