from pathlib import Path

from dispatch_tokens.sample import aggregate_batch, generate_batch
from dispatch_tokens.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[1] / "samples" / "demo_spec.json"


def test_batch_is_deterministic_across_generations():
    first = generate_batch(7, "sh-0", 200, 1)
    second = generate_batch(7, "sh-0", 200, 1)
    assert first == second


def test_aggregate_is_integer_and_stable():
    batch = generate_batch(11, "sh-0", 200, 1)
    summary_a = aggregate_batch(batch, "node-a", 100)
    summary_b = aggregate_batch(batch, "node-a", 100)
    assert summary_a.checksum == summary_b.checksum
    assert summary_a.row_count == 200
    assert all(isinstance(v, int) for v in summary_a.group_sums.values())
    assert all(isinstance(v, int) for v in summary_a.group_means.values())


def test_demo_spec_all_shards_have_integer_weights():
    spec = load_spec(SPEC_PATH)
    for shard in spec.shards:
        assert isinstance(shard.weight, int)
        assert shard.weight >= 0
