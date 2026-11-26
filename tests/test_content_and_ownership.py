from cacheflow.content import deterministic_shard_bytes, shard_digest
from cacheflow.ownership import arbitrate_ownership
from cacheflow.spec import ArtifactSpec, NodeSpec, ShardSpec


def test_deterministic_bytes_and_digest_bind_all_fields():
    first = deterministic_shard_bytes("seed", "node-a", "art", 0, 16)
    second = deterministic_shard_bytes("seed", "node-a", "art", 0, 16)
    assert first == second and len(first) == 16

    digest = shard_digest("node-a", "art", 0, first)
    assert digest == shard_digest("node-a", "art", 0, first)
    assert digest != shard_digest("node-b", "art", 0, first)
    assert digest != shard_digest("node-a", "other", 0, first)
    assert digest != shard_digest("node-a", "art", 1, first)
    assert digest != shard_digest("node-a", "art", 0, first + b"x")


def node(node_id, free_capacity, max_leases=3):
    return NodeSpec(node_id, "127.0.0.1", 40000 + hash(node_id) % 1000, free_capacity, max_leases, "seed-" + node_id)


def test_arbitration_uses_free_slots_then_node_id():
    artifact = ArtifactSpec("art", 4, (ShardSpec(0, 0, 4),))
    nodes = [node("node-a", 2), node("node-b", 2), node("node-c", 4)]
    result = arbitrate_ownership(
        artifact=artifact,
        nodes=nodes,
        free_slots={"node-a": 2, "node-b": 2, "node-c": 4},
        active_lease_counts={"node-a": 0, "node-b": 0, "node-c": 0},
    )
    assert result.owner_node_id == "node-c"

    tied = arbitrate_ownership(
        artifact=artifact,
        nodes=[node("node-b", 2), node("node-a", 2)],
        free_slots={"node-a": 2, "node-b": 2},
        active_lease_counts={"node-a": 0, "node-b": 0},
    )
    assert tied.owner_node_id == "node-a"
    losers = [c for c in tied.candidates if not c.selected]
    assert losers[0].veto_reasons[-1] == "node_id_order_lost"


def test_digest_match_has_priority():
    artifact = ArtifactSpec("art", 4, (ShardSpec(0, 0, 4),))
    nodes = [node("node-a", 9), node("node-b", 1)]
    derived_b = None
    result = arbitrate_ownership(
        artifact=artifact,
        nodes=nodes,
        free_slots={"node-a": 9, "node-b": 1},
        active_lease_counts={"node-a": 0, "node-b": 0},
        observed_digests={"node-a": {0: "bad-digest"}},
    )
    assert result.owner_node_id == "node-b"
    candidate_a = next(c for c in result.candidates if c.node_id == "node-a")
    assert any(reason.startswith("digest_mismatch") for reason in candidate_a.veto_reasons)


def test_capacity_and_lease_limits_veto_candidates():
    artifact = ArtifactSpec("art", 8, (ShardSpec(0, 0, 4), ShardSpec(1, 4, 8)))
    nodes = [node("node-a", 1, max_leases=1), node("node-b", 2, max_leases=1)]
    result = arbitrate_ownership(
        artifact=artifact,
        nodes=nodes,
        free_slots={"node-a": 1, "node-b": 2},
        active_lease_counts={"node-a": 1, "node-b": 0},
    )
    assert result.owner_node_id == "node-b"
    candidate_a = next(c for c in result.candidates if c.node_id == "node-a")
    assert "insufficient_capacity" in candidate_a.veto_reasons
    assert "lease_limit_reached" in candidate_a.veto_reasons
