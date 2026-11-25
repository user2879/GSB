"""Fixed-point arbitration for confusion-candidate template pairs."""

from __future__ import annotations

from typing import Any

from .fingerprint import sha256_json


def arbitrate_confusion_pairs(
    confusion_candidates: list[dict[str, Any]],
    representatives: dict[str, dict[str, Any]],
    cluster_assignments: dict[str, str],
    *,
    cluster_threshold_permille: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in sorted(confusion_candidates, key=lambda item: item["pair_key"]):
        template_a = candidate["template_a"]
        template_b = candidate["template_b"]
        word_a = template_a.split(":", 1)[0]
        word_b = template_b.split(":", 1)[0]
        score_a = representatives[word_a]["composite_score_permille"]
        score_b = representatives[word_b]["composite_score_permille"]
        score_gap = abs(score_a - score_b)
        threshold_delta = candidate["threshold_delta_permille"]
        divergence = (1000 - candidate["similarity_permille"]) * 2 + score_gap // 2 + max(0, threshold_delta) // 2
        cluster_a = cluster_assignments.get(word_a, word_a)
        cluster_b = cluster_assignments.get(word_b, word_b)
        tie_detected = score_gap == 0 or candidate["similarity_permille"] == candidate["candidate_threshold_permille"]
        if cluster_a == cluster_b:
            decision = "merge_to_consensus"
            consensus_word = min(cluster_a, cluster_b)
            decision_reason = "templates_already_share_cluster"
        elif score_gap <= 25 and candidate["similarity_permille"] >= cluster_threshold_permille:
            decision = "unresolvable"
            consensus_word = None
            decision_reason = "similarity_and_representative_confidence_are_tied"
        else:
            decision = "retain_both"
            consensus_word = None
            decision_reason = "distinct_clusters_with_resolvable_confidence_gap"
        result = {
            "pair_key": candidate["pair_key"],
            "template_a": template_a,
            "template_b": template_b,
            "command_word_a": word_a,
            "command_word_b": word_b,
            "similarity_permille": candidate["similarity_permille"],
            "candidate_threshold_permille": candidate["candidate_threshold_permille"],
            "cluster_threshold_permille": cluster_threshold_permille,
            "threshold_delta_permille": threshold_delta,
            "representative_score_a_permille": score_a,
            "representative_score_b_permille": score_b,
            "representative_score_gap_permille": score_gap,
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "divergence_score": divergence,
            "decision": decision,
            "decision_reason": decision_reason,
            "consensus_command_word": consensus_word,
            "tie_detected": tie_detected,
            "tie_break_basis": "template_pair_key_lexicographic" if tie_detected else None,
            "tie_reason": "equal_score_or_exact_candidate_threshold" if tie_detected else None,
            "references": [
                template_a,
                template_b,
                representatives[word_a]["template_key"],
                representatives[word_b]["template_key"],
            ],
        }
        result["cache_key"] = "arbitration:" + sha256_json(
            {
                "candidate_cache_key": candidate["candidate_cache_key"],
                "score_a": score_a,
                "score_b": score_b,
                "representative_a": representatives[word_a]["template_key"],
                "representative_b": representatives[word_b]["template_key"],
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "cluster_threshold_permille": cluster_threshold_permille,
            }
        )
        results.append(result)
    return results
