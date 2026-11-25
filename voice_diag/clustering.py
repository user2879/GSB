"""Deterministic representative selection and command-word cluster merges."""

from __future__ import annotations

from typing import Any

from .fingerprint import sha256_json


def pair_key(template_a: str, template_b: str) -> str:
    first, second = sorted((template_a, template_b))
    return f"{first}|{second}"


def _round_div(numerator: int, denominator: int) -> int:
    return (numerator * 2 + denominator) // (2 * denominator)


def select_representatives(
    quantized: dict[str, dict[str, Any]],
    config: Any,
    similarities: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], str]:
    words: dict[str, list[dict[str, Any]]] = {}
    for template in quantized.values():
        words.setdefault(template["command_word"], []).append(template)
    representatives: dict[str, dict[str, Any]] = {}
    for command_word in sorted(words):
        members = sorted(words[command_word], key=lambda item: item["template_key"])
        scores: list[dict[str, Any]] = []
        for candidate in members:
            total = 0
            compared = 0
            for other in members:
                if candidate["template_key"] == other["template_key"]:
                    total += 1000
                    compared += 1
                    continue
                lookup = (similarities or {}).get(pair_key(candidate["template_key"], other["template_key"]))
                if lookup is not None:
                    total += lookup["similarity_permille"]
                    compared += 1
            score = _round_div(total, compared) if compared else 0
            scores.append(
                {
                    "template_key": candidate["template_key"],
                    "composite_score_permille": score,
                    "effective_frames": int(candidate.get("effective_frames", {}).get("total", 0)),
                    "comparison_count": compared,
                    "comparison_complete": compared == len(members),
                    "integer_fingerprint": candidate["integer_fingerprint"],
                }
            )
        rule = config.representative_rule
        if rule == "lexicographic":
            ordered = sorted(scores, key=lambda item: item["template_key"])
            selected = ordered[0]
            tie_basis = "lexicographic_template_key"
        elif rule == "most_frames":
            ordered = sorted(
                scores,
                key=lambda item: (-item["effective_frames"], -item["composite_score_permille"], item["template_key"]),
            )
            selected = ordered[0]
            tied = [
                item
                for item in scores
                if item["effective_frames"] == selected["effective_frames"]
                and item["composite_score_permille"] == selected["composite_score_permille"]
            ]
            tie_basis = "effective_frames_then_composite_then_template_key" if len(tied) > 1 else "higher_effective_frames"
        else:
            ordered = sorted(scores, key=lambda item: (-item["composite_score_permille"], item["template_key"]))
            selected = ordered[0]
            tied = [item for item in scores if item["composite_score_permille"] == selected["composite_score_permille"]]
            tie_basis = "composite_score_then_template_key" if len(tied) > 1 else "higher_composite_score"
        if len(members) == 1:
            tie_basis = "sole_template"
        representatives[command_word] = {
            "command_word": command_word,
            "template_key": selected["template_key"],
            "composite_score_permille": selected["composite_score_permille"],
            "representative_rule": rule,
            "tie_break_basis": tie_basis,
            "degenerate": len(members) == 1,
            "skip_cross_word_merge": len(members) == 1,
            "candidate_scores": scores,
            "member_template_keys": [member["template_key"] for member in members],
            "references": [member["template_key"] for member in members],
            "cache_key": "representative:" + sha256_json(
                {
                    "command_word": command_word,
                    "rule": rule,
                    "members": [
                        {
                            "key": member["template_key"],
                            "integer_fingerprint": member["integer_fingerprint"],
                            "effective_frames": member.get("effective_frames", {}).get("total", 0),
                        }
                        for member in members
                    ],
                }
            ),
        }
    stage_key = "representatives:" + sha256_json(
        {word: value["cache_key"] for word, value in sorted(representatives.items())}
    )
    return representatives, stage_key


def build_clusters(
    quantized: dict[str, dict[str, Any]],
    representatives: dict[str, dict[str, Any]],
    similarities: dict[str, dict[str, Any]],
    config: Any,
) -> tuple[dict[str, Any], str]:
    words = sorted(representatives)
    active: dict[str, dict[str, Any]] = {}
    assignment: dict[str, dict[str, Any]] = {}
    traces: list[dict[str, Any]] = []
    threshold = config.cluster_threshold

    for command_word in words:
        representative = representatives[command_word]
        candidates = sorted(active)
        comparisons: list[dict[str, Any]] = []
        merged_into: str | None = None
        if not representative["degenerate"]:
            for target_word in candidates:
                target = active[target_word]
                if representatives[target_word]["degenerate"]:
                    comparisons.append(
                        {
                            "target_cluster": target_word,
                            "compared": False,
                            "reason": "target_degenerate_cluster_exempt",
                        }
                    )
                    continue
                key = pair_key(representative["template_key"], representatives[target_word]["template_key"])
                similarity = similarities.get(key)
                if similarity is None:
                    comparisons.append(
                        {
                            "target_cluster": target_word,
                            "comparison_pair": key,
                            "compared": False,
                            "reason": "comparison_unavailable",
                        }
                    )
                    continue
                accepted = similarity["similarity_permille"] >= threshold
                comparisons.append(
                    {
                        "target_cluster": target_word,
                        "comparison_pair": key,
                        "compared": True,
                        "similarity_permille": similarity["similarity_permille"],
                        "threshold_permille": threshold,
                        "accepted": accepted,
                    }
                )
                if accepted:
                    merged_into = target_word
                    break
        if merged_into is None:
            active[command_word] = {
                "cluster_id": command_word,
                "kept_command_word": command_word,
                "merged_command_words": [command_word],
                "member_template_keys": sorted(
                    key for key, value in quantized.items() if value["command_word"] == command_word
                ),
                "degenerate": representative["degenerate"],
                "merge_evidence": [],
            }
            assignment[command_word] = {
                "command_word": command_word,
                "cluster_id": command_word,
                "rewritten": False,
                "reason": "degenerate_single_template" if representative["degenerate"] else "no_qualifying_cluster",
            }
        else:
            target = active[merged_into]
            target["merged_command_words"].append(command_word)
            target["merged_command_words"] = sorted(set(target["merged_command_words"]))
            target["member_template_keys"] = sorted(
                set(
                    target["member_template_keys"]
                    + [key for key, value in quantized.items() if value["command_word"] == command_word]
                )
            )
            target["merge_evidence"].append(
                {
                    "source_command_word": command_word,
                    "target_cluster": merged_into,
                    "comparison_pair": pair_key(
                        representative["template_key"], representatives[merged_into]["template_key"]
                    ),
                    "similarity_permille": comparisons[-1]["similarity_permille"],
                    "threshold_permille": threshold,
                }
            )
            target["merge_evidence"].sort(key=lambda item: item["source_command_word"])
            assignment[command_word] = {
                "command_word": command_word,
                "cluster_id": merged_into,
                "rewritten": True,
                "reason": "first_lexicographic_qualifying_cluster",
            }
        trace = {
                "command_word": command_word,
                "representative_template_key": representative["template_key"],
                "degenerate": representative["degenerate"],
                "comparisons": comparisons,
                "decision": "merged" if merged_into else ("exempt_degenerate" if representative["degenerate"] else "standalone"),
                "merged_into": merged_into,
            }
        trace["cache_key"] = "cluster-trace:" + sha256_json(trace)
        traces.append(trace)

    clusters = []
    for cluster_id in sorted(active):
        cluster = active[cluster_id]
        representative = representatives[cluster["kept_command_word"]]
        cluster["representative_template_key"] = representative["template_key"]
        cluster["representative_composite_score_permille"] = representative["composite_score_permille"]
        cluster["member_template_keys"] = sorted(cluster["member_template_keys"])
        cluster["references"] = cluster["member_template_keys"]
        cluster["cache_key"] = "cluster:" + sha256_json(
            {
                "cluster_id": cluster["cluster_id"],
                "kept_command_word": cluster["kept_command_word"],
                "merged_command_words": cluster["merged_command_words"],
                "member_template_keys": cluster["member_template_keys"],
                "merge_evidence": cluster["merge_evidence"],
                "representative_template_key": cluster["representative_template_key"],
                "representative_composite_score_permille": cluster[
                    "representative_composite_score_permille"
                ],
                "degenerate": cluster["degenerate"],
            }
        )
        clusters.append(cluster)
    for assignment_record in assignment.values():
        assignment_record["cache_key"] = "cluster-assignment:" + sha256_json(assignment_record)

    result = {
        "cluster_threshold_permille": threshold,
        "representative_rule": config.representative_rule,
        "clusters": clusters,
        "assignments": [assignment[word] for word in sorted(assignment)],
        "comparison_trace": traces,
        "merge_count": sum(len(cluster["merged_command_words"]) - 1 for cluster in clusters),
    }
    result["cache_key"] = "clusters:" + sha256_json(
        {
            "threshold": threshold,
            "representatives": {word: value["cache_key"] for word, value in sorted(representatives.items())},
            "merge_trace": [
                {
                    "command_word": item["command_word"],
                    "comparisons": item["comparisons"],
                    "merged_into": item["merged_into"],
                }
                for item in traces
            ],
        }
    )
    return result, result["cache_key"]
