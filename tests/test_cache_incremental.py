from __future__ import annotations

from dataclasses import replace

from candidate_diagnostics.engine import IncrementalEngine, compare_result_fields


def test_single_channel_weight_invalidates_only_referencing_requests(config, requests):
    engine = IncrementalEngine(config)
    engine.run(requests)
    engine.update_config({"channels": {"search": 9.99}})
    run = engine.run(requests)
    assert "req-feedonly" in run.ledger["hit_requests"]
    assert "req-001" in run.ledger["recomputed_requests"]
    assert "req-feedonly:v1" in run.ledger["hit_products"]
    assert "req-feedonly:v2" in run.ledger["hit_products"]
    assert all(item.startswith("req-feedonly:") is False for item in run.ledger["recomputed_products"])
    assert "req-001:search" in run.ledger["recomputed_segments"]
    assert "req-001:feed" in run.ledger["hit_segments"]


def test_single_request_candidate_replacement_invalidates_only_request(config, requests):
    engine = IncrementalEngine(config)
    engine.run(requests)
    changed = dict(requests)
    request = changed["req-005"]
    changed_candidates = list(request.candidates)
    changed_candidates[0] = replace(changed_candidates[0], raw_score=1.5)
    request.candidates = changed_candidates
    run = engine.run(changed)
    assert run.ledger["recomputed_requests"] == ["req-005"]
    assert set(run.ledger["hit_requests"]) == set(requests) - {"req-005"}
    assert run.ledger["recomputed_products"] == ["req-005:v2", "req-005:v1"]


def test_topk_change_recomputes_results_but_reuses_normalization_segments(config, requests):
    engine = IncrementalEngine(config)
    engine.run(requests)
    engine.update_config({"top_k": 3})
    run = engine.run(requests)
    assert "req-001" in run.ledger["recomputed_requests"]
    assert "req-001:search" in run.ledger["hit_segments"]
    assert run.results["req-001"].top_k_ids == [101, 102, 103]


def test_incremental_final_equals_blank_rebuild(config, requests):
    engine = IncrementalEngine(config)
    engine.run(requests)
    engine.update_config({"channels": {"search": 1.25}})
    changed_requests = dict(requests)
    req3 = changed_requests["req-003"]
    req3.candidates = [
        replace(candidate, raw_score=77.7 if candidate.candidate_id == 301 else candidate.raw_score)
        for candidate in req3.candidates
    ]
    engine.run(changed_requests)
    engine.update_config({"top_k": 4, "sampling": {"ratio_permille": 500, "ratio": 0.5}})
    incremental = engine.run(changed_requests)
    blank = IncrementalEngine(engine.config).run(changed_requests)
    comparison = compare_result_fields(incremental, blank)
    assert comparison["identical"] is True, comparison["mismatches"]
