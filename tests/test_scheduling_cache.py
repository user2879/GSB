from __future__ import annotations

from ner_workbench.config import load_config
from ner_workbench.generator import generate_dataset
from ner_workbench.injector import InjectionSet, InjectionSpec
from ner_workbench.loader import load_corpus
from ner_workbench.scheduling import (
    CacheEvent,
    build_cache_events,
    plan_budget,
    simulate_lru_cache,
)


def test_budget_quota_injection_and_global_exhaustion(tmp_path):
    samples_dir = tmp_path / "samples"
    generate_dataset(samples_dir, seed=91, sample_count=6)
    cfg = load_config(samples_dir / "inference.yaml")
    samples = load_corpus(samples_dir / "corpus.jsonl").samples
    first = samples[0]
    injection = InjectionSpec("INJ-Q", first.sample_id, "schedule", "quota_lt_min_chars")
    injection_set = InjectionSet(
        by_sample={first.sample_id: (injection,)},
        all=(injection,),
        source_hash="test",
    )
    decisions = plan_budget(samples, cfg, injection_set, initial_remaining=10)
    assert decisions[0].sample_id == first.sample_id
    assert decisions[0].code == "E_SAMPLE_QUOTA_EXCEEDED"
    assert decisions[0].injection_id == "INJ-Q"
    assert decisions[0].allocated == 10
    assert decisions[-1].after <= 10
    assert all(decision.after <= decision.before for decision in decisions)


def test_lru_eviction_tie_and_recompute_are_deterministic():
    events = (
        CacheEvent("e1", 1, "S", 0, "emission", "S:block:000000"),
        CacheEvent("e2", 2, "S", 0, "emission", "S:block:000001"),
        CacheEvent("e3", 3, "S", 0, "emission", "S:block:000002"),
        CacheEvent("e4", 4, "S", 0, "decode", "S:block:000000"),
    )

    def signature(key: str) -> str:
        return "checksum:" + key

    first = simulate_lru_cache(events, 2, signature)
    second = simulate_lru_cache(events, 2, signature)
    assert first.as_dict() == second.as_dict()
    assert first.evictions[0]["evicted_block"] == "S:block:000000"
    assert first.recomputations == 1
    assert first.recompilation_log[0]["matches_first_compute"] is True
    assert first.checksum_mismatches == 0


def test_generated_cache_access_causes_evictions_and_recompute(tmp_path):
    samples_dir = tmp_path / "samples"
    generate_dataset(samples_dir, seed=92, sample_count=8)
    cfg = load_config(samples_dir / "inference.yaml")
    samples = load_corpus(samples_dir / "corpus.jsonl").samples
    events = build_cache_events(samples, cfg)
    simulation = simulate_lru_cache(events, cfg.cache_block_budget, lambda key: "sha:" + key)
    assert simulation.evictions
    assert simulation.recomputations > 0
    assert simulation.checksum_mismatches == 0
