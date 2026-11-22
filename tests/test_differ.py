"""Requirement 2: diff measurement and derivation impact chains."""

from config_regression.differ import (
    ADDED,
    COEFFICIENT_CHANGED,
    REMOVED,
    TARGETS_CHANGED,
    VALUE_CHANGED,
    diff_manifests,
)
from config_regression.model import load_manifest, load_registry


def _registry():
    return load_registry({"nodes": [
        {"node_id": n, "port": 0} for n in ("node-a", "node-b", "node-c")]})


def _load(data):
    return load_manifest(data, _registry())


BASE = {"version": 1, "items": [
    {"key": "root", "value": 1.0, "targets": ["node-a"], "derived_from": []},
    {"key": "mid", "value": 2.0, "targets": ["node-a", "node-b"],
     "derived_from": [{"key": "root", "coefficient": 2.0}]},
    {"key": "leaf", "value": 3.0, "targets": ["node-b"],
     "derived_from": [{"key": "mid", "coefficient": 1.0}]},
    {"key": "steady", "value": 9.0, "targets": ["node-c"],
     "derived_from": []},
    {"key": "doomed", "value": 4.0, "targets": ["node-b"],
     "derived_from": []},
]}

CANDIDATE = {"version": 2, "items": [
    {"key": "root", "value": 1.5, "targets": ["node-a"], "derived_from": []},
    {"key": "mid", "value": 2.0, "targets": ["node-a"],
     "derived_from": [{"key": "root", "coefficient": 2.5}]},
    {"key": "leaf", "value": 3.0, "targets": ["node-b"],
     "derived_from": [{"key": "mid", "coefficient": 1.0}]},
    {"key": "steady", "value": 9.0, "targets": ["node-c"],
     "derived_from": []},
    {"key": "newborn", "value": 5.0, "targets": ["node-c"],
     "derived_from": [{"key": "root", "coefficient": 1.0}]},
]}


def _diff():
    return diff_manifests(_load(BASE), _load(CANDIDATE))


def test_change_type_classification():
    result = _diff()
    assert VALUE_CHANGED in result.item_diffs["root"].change_types
    assert TARGETS_CHANGED in result.item_diffs["mid"].change_types
    assert COEFFICIENT_CHANGED in result.item_diffs["mid"].change_types
    assert ADDED in result.item_diffs["newborn"].change_types
    assert REMOVED in result.item_diffs["doomed"].change_types
    assert "steady" in result.unchanged_keys
    assert "leaf" not in result.changed_keys  # value/edges identical


def test_impact_chain_follows_full_downstream_in_topo_order():
    result = _diff()
    root_chain = next(c for c in result.impact_chains if c.root_key == "root")
    # root -> mid -> leaf and root -> newborn; candidate topo order preserved
    assert root_chain.downstream_order == ["mid", "leaf", "newborn"]
    assert set(root_chain.affected_nodes) == {"node-a", "node-b", "node-c"}


def test_unchanged_item_never_appears_as_impact_root():
    result = _diff()
    roots = {chain.root_key for chain in result.impact_chains}
    assert "steady" not in roots
    assert "leaf" not in roots


def test_removed_item_keeps_baseline_nodes_and_sorted_output():
    result = _diff()
    removed = next(c for c in result.impact_chains if c.root_key == "doomed")
    assert removed.affected_nodes == ["node-b"]
    assert [c.root_key for c in result.impact_chains] == sorted(
        c.root_key for c in result.impact_chains)


def test_result_independent_of_dict_iteration_order():
    import random
    from config_regression.canonical import canonical_bytes
    first = canonical_bytes(_diff().as_dict())
    rng = random.Random(0)
    for _ in range(5):
        shuffled = dict(CANDIDATE)
        items = list(CANDIDATE["items"])
        rng.shuffle(items)
        shuffled["items"] = items
        again = diff_manifests(_load(BASE), _load(shuffled))
        assert canonical_bytes(again.as_dict()) == first
