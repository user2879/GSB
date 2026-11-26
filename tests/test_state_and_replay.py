from pathlib import Path

import pytest

from cacheflow.canonical import canonical_bytes
from cacheflow.engine import Engine, replay_bundle
from cacheflow.errors import DivergenceError, GenerationError
from cacheflow.evolve import EvolutionChange, evolve_and_verify
from cacheflow.report import conclusion_from_engine
from cacheflow.spec import load_bundle
from cacheflow.statemachine import GenerationChain


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(ROOT / "examples/nodes.json", ROOT / "examples/artifacts.json", ROOT / "examples/events.json")


def test_generation_chain_rejects_jump_and_rollback_but_preserves_generation():
    chain = GenerationChain("lease-x", 1)
    valid = {
        "event_seq": 2,
        "ordinal": 0,
        "node_id": "node-a",
        "trigger": "renew",
        "reason": "ok",
        "state_before": "granted",
        "state_after": "renewed",
        "carried_label": 1,
        "quota_version": 1,
    }
    chain.append(**valid, generation_before=1, generation_after=2)
    assert chain.generation == 2

    with pytest.raises(GenerationError):
        chain.append(**{**valid, "event_seq": 3}, generation_before=3, generation_after=4)
    assert chain.generation == 2

    with pytest.raises(GenerationError):
        chain.append(**{**valid, "event_seq": 4}, generation_before=1, generation_after=2)
    assert chain.generation == 2


def test_full_replay_is_unique_and_byte_stable(bundle):
    engine = replay_bundle(bundle)
    conclusion = conclusion_from_engine(bundle, engine)
    assert conclusion["uniqueness"]["unique_holding_ok"] is True
    assert conclusion["capacity"]["node-c"]["used_slots"] == 3
    assert conclusion["capacity"]["node-a"]["used_slots"] == 0
    assert canonical_bytes(conclusion) == canonical_bytes(conclusion_from_engine(bundle, replay_bundle(bundle)))


def test_arbitrary_prefixes_and_chunked_replay_are_stable(bundle):
    seqs = [int(event["seq"]) for event in bundle.events]
    for cutoff in seqs:
        first = conclusion_from_engine(bundle, replay_bundle(bundle, through_seq=cutoff), cutoff_seq=cutoff)
        second = conclusion_from_engine(bundle, replay_bundle(bundle, through_seq=cutoff), cutoff_seq=cutoff)
        assert canonical_bytes(first) == canonical_bytes(second)

    chunked = Engine(bundle)
    for index, event in enumerate(bundle.events):
        chunked.apply_spec_event(event)
        if index % 5 == 4:
            _ = canonical_bytes(conclusion_from_engine(bundle, chunked))
    one_shot = replay_bundle(bundle)
    assert canonical_bytes(conclusion_from_engine(bundle, chunked)) == canonical_bytes(
        conclusion_from_engine(bundle, one_shot)
    )


def test_stale_event_does_not_advance_planned_generation(bundle):
    engine = Engine(bundle)
    for event in bundle.events:
        result = engine.apply_spec_event(event)
        if int(event["seq"]) == 22:
            assert result["expected_error"] == "E_STALE"
            assert result["current_generation"] == 2
    alpha = engine.leases["lease-alpha"]
    assert alpha["generation"] == 4
    assert alpha["state"] == "expired"


def test_evolution_subset_matches_full_rebuild(bundle):
    change = EvolutionChange(seq=40, removed_node_ids=("node-c",), reason="remove c")
    evolved_nodes = [node for node in bundle.nodes if node.node_id != "node-c"]
    result = evolve_and_verify(
        bundle,
        cutoff_seq=24,
        change=change,
        evolved_nodes=evolved_nodes,
    )
    impact = result["impact"]
    assert impact["affected_lease_ids"] == ["lease-alpha"]
    assert "lease-beta" in impact["unaffected_lease_ids"]
    assert "lease-delta" in impact["unaffected_lease_ids"]
    assert result["untouched_ok"] is True
    assert result["incremental_matches_full_rebuild"] is True


def test_divergence_locator():
    from cacheflow.canonical import assert_consistent

    with pytest.raises(DivergenceError) as exc:
        assert_consistent({"nodes": {"a": {"free": 1}}}, {"nodes": {"a": {"free": 2}}}, DivergenceError)
    assert "$.nodes.a.free" in str(exc.value)
