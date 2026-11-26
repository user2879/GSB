from edge_cache.arbitration import arbitrate, merge_conclusions
from edge_cache.config import load_config_dir
from edge_cache.receipts import Receipt, integrity_label_for, precision_label_for


def _receipt(shard_id, index, node_id, quota_version, quota_code, clock_ms,
             duplicate_ref=None):
    receipt = Receipt(
        shard_id=shard_id,
        index=index,
        node_id=node_id,
        check_rate_permille=900,
        residual=1,
        check_rate_float=0.900001,
        quota_version=quota_version,
        quota_code=quota_code,
        digest=f"digest-{node_id}-{shard_id}",
        duplicate_ref=duplicate_ref,
        fixed_point=4,
        precision_label=precision_label_for(4, 1),
        clock_ms=clock_ms,
    )
    receipt.integrity_label = integrity_label_for(receipt)
    return receipt


def test_arbitration_priority_order():
    quota_seq = {"q1": 100, "q2": 200, "q3": 300}
    candidates = [
        _receipt("s1", 1, "node-a", "q1", 100, 500),
        _receipt("s1", 1, "node-b", "q3", 300, 100),
        _receipt("s1", 1, "node-c", "q3", 300, 400),
    ]
    result = arbitrate("s1", candidates, quota_seq)
    assert result.authority.node_id == "node-c"
    assert not result.tied
    assert [candidate.node_id for candidate in result.candidates] == [
        "node-c", "node-b", "node-a"
    ]


def test_override_reference_wins_even_on_older_quota():
    quota_seq = {"q1": 100, "q2": 200}
    candidates = [
        _receipt("s1", 1, "node-a", "q2", 200, 500),
        _receipt("s1", 1, "node-b", "q1", 100, 100, duplicate_ref="node-a::s1::dup"),
    ]
    result = arbitrate("s1", candidates, quota_seq)
    assert result.authority.node_id == "node-b"


def test_full_tie_is_flagged_and_all_candidates_kept():
    quota_seq = {"q1": 100}
    candidates = [
        _receipt("s1", 1, "node-a", "q1", 100, 100),
        _receipt("s1", 1, "node-b", "q1", 100, 100),
    ]
    # Distinct node ids break the tie lexicographically; force a true tie by
    # reusing identical node identity values.
    candidates[1].node_id = "node-a"
    candidates[1].integrity_label = integrity_label_for(candidates[1])
    result = arbitrate("s1", candidates, quota_seq)
    assert result.tied
    assert len(result.candidates) == 2


def test_conclusion_propagates_upstream_failure(sample_config_dir):
    config = load_config_dir(sample_config_dir)
    quota_seq = {quota.version: quota.seq for quota in config.quotas}
    arbitration = {}
    for shard in config.shards:
        candidates = []
        if shard.shard_id != "s1":
            candidates = [
                _receipt(shard.shard_id, shard.index, "node-a", "q1", 100, 10)
            ]
        arbitration[shard.shard_id] = arbitrate(shard.shard_id, candidates, quota_seq)
    conclusions = merge_conclusions(config, arbitration)
    assert not conclusions["s1"].available
    assert not conclusions["s2"].available
    assert conclusions["s2"].upstream_unavailable == ("s1",)
    assert not conclusions["s4"].available
    assert set(conclusions["s4"].upstream_unavailable) == {"s2", "s3"}
