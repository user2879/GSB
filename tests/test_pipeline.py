from __future__ import annotations

from candidate_diagnostics.attribution import make_reference_config
from candidate_diagnostics.config import Config, deep_merge
from candidate_diagnostics.models import Candidate
from candidate_diagnostics.pipeline import analyze_many, analyze_request, normalize_candidate


def test_topk_boundary_tie_uses_candidate_id_asc(config, requests):
    result = analyze_request(requests["req-001"], config)
    assert result.ranked_ids[:5] == [101, 102, 103, 104, 105]
    assert result.top_k_ids[-1] == 105
    assert result.boundary["boundary_selected_id"] == 105
    assert result.boundary["nearest_rejected_id"] == 106
    assert result.boundary["tie_at_cutoff"] is True
    assert set(result.boundary["boundary_selected_ids"]) == {103, 104, 105}
    assert result.boundary["boundary_rejected_ids"] == [106]


def test_sampling_quota_remainder_and_deterministic_shuffle(config, requests):
    first = analyze_request(requests["req-001"], config)
    second = analyze_request(requests["req-001"], config)
    assert first.to_dict() == second.to_dict()
    assert first.sampled_ids == [103, 105]
    assert first.unsampled_topk_ids == [101, 102, 104]
    assert first.shuffle_order == [105, 103]
    assert first.layers["feed"]["quota"] == 1
    assert first.layers["recall"]["top_up"] == 1
    assert first.layers["search"]["quota"] == 0
    assert "remainder_by_channel_name_asc" in first.layers["recall"]["basis"]


def test_many_is_stable_in_request_order(config, requests):
    results = analyze_many(requests, config)
    assert list(results) == sorted(requests)
    assert [rid for rid in results] == sorted(results)


def test_quantization_width_and_gear_change_values_for_fractional_score(config):
    candidate = Candidate(999, 80.00006, "feed", 1)
    baseline = normalize_candidate(candidate, config)
    coarse = Config(deep_merge(config.raw, {"quantization": {"fixed_decimals": 1, "gear_exponent": 1}}))
    coarsened = normalize_candidate(candidate, coarse)
    assert baseline["fixed_score"] != coarsened["fixed_score"]
    assert baseline["dequant_score"] != coarsened["dequant_score"]
    assert coarsened["fixed_score"] % 10 == 0


def test_higher_width_reference_changes_boundary_decision(config, requests):
    coarse = analyze_request(requests["req-bcmp"], config)
    reference_config = make_reference_config(config)
    reference = analyze_request(requests["req-bcmp"], reference_config)
    assert 705 in coarse.top_k_ids
    assert 706 not in coarse.top_k_ids
    assert 706 in reference.top_k_ids
    assert 705 not in reference.top_k_ids


def test_aux_feature_groups_are_recorded(config, requests):
    result = analyze_request(requests["req-001"], config)
    feed_groups = result.layers["feed"]["feature_groups"]
    assert "20" in feed_groups
    assert set(feed_groups["20"]) == {102, 105}
