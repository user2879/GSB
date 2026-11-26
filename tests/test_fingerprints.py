from dataclasses import replace

from edge_cache.config import load_config_dir
from edge_cache.fingerprints import (
    authority_key,
    conclusion_key,
    node_summary_key,
    serialization_key,
    upstream_closure,
)
from edge_cache.invalidation import (
    apply_plan,
    empty_plan,
    invalidate_format,
    invalidate_node_quota,
    invalidate_shard,
)
from edge_cache.ledger import Ledger
from edge_cache.receipts import C1, C2, C3


def test_upstream_closure_includes_transitive_dependencies(sample_config_dir):
    config = load_config_dir(sample_config_dir)
    assert upstream_closure(config, "s4") == ("s1", "s2", "s3", "s4")


def test_shard_replacement_only_invalidates_downstream_keys(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    ledger = Ledger(tmp_workdir)
    # Pretend every artifact exists.
    for shard in config.shards:
        for node in config.nodes:
            ledger.store_artifact(node_summary_key(config, node, shard.shard_id), {"v": 1})
        ledger.store_artifact(authority_key(config, shard.shard_id), {"v": 1})
        ledger.store_artifact(conclusion_key(config, shard.shard_id), {"v": 1})
        for fmt in (C1, C2, C3):
            ledger.store_artifact(serialization_key(config, shard.shard_id, fmt), {"v": 1})

    plan = invalidate_shard(config, "s3", empty_plan())
    removed = apply_plan(ledger, plan)
    # s3 and its downstream s4/s5 are affected; s1 and s2 artifacts survive.
    surviving_shards = {
        name for name in ledger.all_artifacts()
    }
    assert not any(key in surviving_shards for key in plan.artifact_keys)
    s2_key = conclusion_key(config, "s2")
    assert ledger.has_artifact(s2_key)
    assert {"s3", "s4", "s5"}.issubset(plan.affected_shards)
    assert removed


def test_quota_change_invalidates_only_changed_node_summaries(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    ledger = Ledger(tmp_workdir)
    node_b = replace(config.node("node-b"), quota_version="q3")
    changed = replace(config, nodes=tuple(
        node_b if node.node_id == "node-b" else node for node in config.nodes
    ))
    for shard in changed.shards:
        for node in changed.nodes:
            ledger.store_artifact(node_summary_key(changed, node, shard.shard_id), {"v": 1})
    plan = invalidate_node_quota(changed, "node-b", empty_plan())
    apply_plan(ledger, plan)
    assert plan.affected_nodes == {"node-b"}
    assert not ledger.has_artifact(node_summary_key(changed, node_b, "s1"))
    assert ledger.has_artifact(node_summary_key(changed, changed.node("node-a"), "s1"))


def test_format_switch_only_touches_serialization(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    ledger = Ledger(tmp_workdir)
    for shard in config.shards:
        ledger.store_artifact(conclusion_key(config, shard.shard_id), {"v": 1})
        for fmt in (C1, C2, C3):
            ledger.store_artifact(serialization_key(config, shard.shard_id, fmt), {"v": 1})
    plan = invalidate_format(config, empty_plan())
    apply_plan(ledger, plan)
    assert all(
        ledger.has_artifact(conclusion_key(config, shard.shard_id))
        for shard in config.shards
    )
    assert not any(
        ledger.has_artifact(serialization_key(config, shard.shard_id, C3))
        for shard in config.shards
    )
