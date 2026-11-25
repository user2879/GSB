"""Deterministic forward maximum matching and rule-based disambiguation."""

from __future__ import annotations

from typing import Any

from .hashing import fingerprint


def build_index(lexicon: dict[str, Any]) -> dict[str, Any]:
    entries = lexicon["entries"]
    by_id = {entry["id"]: entry for entry in entries}
    enabled = [entry for entry in entries if entry.get("enabled")]
    by_term = {entry["term"]: entry for entry in enabled}
    pos_tags = lexicon.get("pos_tags", [])
    code_to_pos = {tag["code"]: tag["name"] for tag in pos_tags}
    pos_to_code = {tag["name"]: tag["code"] for tag in pos_tags}
    return {
        "entries": entries,
        "by_id": by_id,
        "enabled": enabled,
        "by_term": by_term,
        "code_to_pos": code_to_pos,
        "pos_to_code": pos_to_code,
    }


def rule_package_fingerprint(rules_document: dict[str, Any]) -> str:
    return fingerprint({"rules": rules_document.get("rules", [])}, 20)


def estimate_match_points(doc: dict[str, Any], index: dict[str, Any]) -> int:
    """Estimate one batch's matching points.

    The declared document term count is multiplied by every enabled candidate
    term length.  The formula is deliberately integer-only and stable.
    """

    term_count = int(doc.get("term_count") or max(1, len(doc["text"])))
    candidate_length_sum = sum(len(entry["term"]) for entry in index["enabled"])
    return term_count * candidate_length_sum


def _all_candidates(text: str, index: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for start in range(len(text) + 1):
        for entry_index, entry in enumerate(index["enabled"]):
            term = entry["term"]
            end = start + len(term)
            if end <= len(text) and text.startswith(term, start):
                candidates.append(
                    {
                        "candidate_id": f"{entry['id']}@{start}:{end}",
                        "entry_id": entry["id"],
                        "entry_index": entry_index,
                        "term": term,
                        "start": start,
                        "end": end,
                        "length": len(term),
                        "frequency": entry["frequency"],
                    }
                )
    return candidates


def _local_fmm_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda c: (c["start"], -c["length"], -c["frequency"], c["term"], c["entry_id"]))


def _overlap_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda c: (-c["length"], c["start"], c["term"], c["entry_id"]))


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _select_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = _overlap_order(candidates)
    winners: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    winner_by_id: dict[str, dict[str, Any]] = {}

    for candidate in ordered:
        overlapping_winners = [winner for winner in winners if _overlaps(candidate, winner)]
        if not overlapping_winners:
            winners.append(candidate)
            winner_by_id[candidate["candidate_id"]] = candidate
            continue
        winner = min(overlapping_winners, key=lambda item: (-item["length"], item["start"], item["term"], item["entry_id"]))
        suppressed.append(
            {
                "candidate_id": candidate["candidate_id"],
                "entry_id": candidate["entry_id"],
                "term": candidate["term"],
                "start": candidate["start"],
                "end": candidate["end"],
                "length": candidate["length"],
                "superseded_by": winner["candidate_id"],
                "superseding_entry_id": winner["entry_id"],
                "superseding_term": winner["term"],
                "superseding_start": winner["start"],
                "superseding_end": winner["end"],
            }
        )

    suppressed.sort(key=lambda item: (item["start"], item["end"], item["term"], item["entry_id"]))
    winners.sort(key=lambda item: (item["start"], item["end"], item["entry_id"]))
    return winners, suppressed


def _base_fragments(text: str, winners: list[dict[str, Any]], suppressed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    suppressions_by_winner: dict[str, list[dict[str, Any]]] = {}
    for item in suppressed:
        suppressions_by_winner.setdefault(item["superseded_by"], []).append(item)

    fragments: list[dict[str, Any]] = []
    covered = [False] * len(text)
    for winner in winners:
        for pos in range(winner["start"], winner["end"]):
            covered[pos] = True
        winner_suppressed = suppressions_by_winner.get(winner["candidate_id"], [])
        refs = [winner["entry_id"]] + [item["entry_id"] for item in winner_suppressed]
        fragments.append(
            {
                "start": winner["start"],
                "end": winner["end"],
                "entry_id": winner["entry_id"],
                "candidate_entry_ids": [winner["entry_id"]],
                "entry_refs": _dedupe(refs),
                "rule_ids": [],
                "origin": "fmm",
            }
        )

    pos = 0
    while pos < len(text):
        if covered[pos]:
            pos += 1
            continue
        end = pos + 1
        fragments.append(
            {
                "start": pos,
                "end": end,
                "entry_id": None,
                "candidate_entry_ids": [],
                "entry_refs": [],
                "rule_ids": [],
                "origin": "unknown-character",
            }
        )
        pos = end

    fragments.sort(key=lambda item: (item["start"], item["end"]))
    return fragments


def _rules_for_doc(rules_document: dict[str, Any], doc: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = doc["doc_id"]
    return [
        rule
        for rule in rules_document.get("rules", [])
        if rule.get("enabled", True) and rule.get("doc_id", "*") in ("*", doc_id)
    ]


def _resolve_rule_conflicts(rules: list[dict[str, Any]], text_length: int) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    issues: list[dict[str, Any]] = []
    for rule in rules:
        position = rule["position"]
        if position < 0 or position > text_length:
            issues.append(
                {
                    "code": "E_RULE_INVALID",
                    "rule_id": rule["rule_id"],
                    "position": position,
                    "message": f"position {position} is outside text length {text_length}",
                }
            )
            continue
        groups.setdefault((rule.get("doc_id", "*"), position), []).append(rule)

    decisions: list[dict[str, Any]] = []
    active: dict[int, dict[str, Any]] = {}
    for (_, position), group in sorted(groups.items(), key=lambda item: item[0][1]):
        distinct_actions = {rule["action"] for rule in group}
        winner = min(group, key=lambda rule: (-int(rule["version"]), int(rule["order"]), rule["rule_id"]))
        decision = {
            "position": position,
            "winning_rule_id": winner["rule_id"],
            "action": winner["action"],
            "version": winner["version"],
            "order": winner["order"],
            "candidate_rule_ids": [rule["rule_id"] for rule in sorted(group, key=lambda r: (-r["version"], r["order"], r["rule_id"]))],
            "conflict": len(distinct_actions) > 1,
        }
        decisions.append(decision)
        active[position] = winner
    return active, decisions, issues


def _split_fragment(fragments: list[dict[str, Any]], position: int, rule_id: str) -> list[dict[str, Any]]:
    for index, fragment in enumerate(fragments):
        if fragment["start"] < position < fragment["end"]:
            left = dict(fragment)
            right = dict(fragment)
            left["end"] = position
            right["start"] = position
            left["candidate_entry_ids"] = list(fragment["candidate_entry_ids"])
            right["candidate_entry_ids"] = list(fragment["candidate_entry_ids"])
            left["entry_refs"] = list(fragment["entry_refs"])
            right["entry_refs"] = list(fragment["entry_refs"])
            left["rule_ids"] = _dedupe(fragment["rule_ids"] + [rule_id])
            right["rule_ids"] = _dedupe(fragment["rule_ids"] + [rule_id])
            left["origin"] = "rule-split"
            right["origin"] = "rule-split"
            return fragments[:index] + [left, right] + fragments[index + 1 :]
    for fragment in fragments:
        if fragment["start"] <= position <= fragment["end"]:
            fragment["rule_ids"] = _dedupe(fragment["rule_ids"] + [rule_id])
    return fragments


def _merge_at_position(fragments: list[dict[str, Any]], position: int, rule_id: str) -> tuple[list[dict[str, Any]], bool]:
    for index in range(len(fragments) - 1):
        left = fragments[index]
        right = fragments[index + 1]
        if left["end"] == position and right["start"] == position:
            merged = {
                "start": left["start"],
                "end": right["end"],
                "entry_id": None,
                "candidate_entry_ids": _dedupe(list(left["candidate_entry_ids"]) + list(right["candidate_entry_ids"])),
                "entry_refs": _dedupe(list(left["entry_refs"]) + list(right["entry_refs"])),
                "rule_ids": _dedupe(list(left["rule_ids"]) + list(right["rule_ids"]) + [rule_id]),
                "origin": "rule-merge",
            }
            return fragments[:index] + [merged] + fragments[index + 2 :], True
    return fragments, False


def segment_document(
    doc: dict[str, Any],
    index: dict[str, Any],
    rules_document: dict[str, Any],
    lexicon_revision: int | str,
    package_fingerprint: str | None = None,
) -> dict[str, Any]:
    text = doc["text"]
    all_candidates = _all_candidates(text, index)
    fmm_order = _local_fmm_order(all_candidates)
    winners, suppressed = _select_candidates(all_candidates)
    fragments = _base_fragments(text, winners, suppressed)

    doc_rules = _rules_for_doc(rules_document, doc)
    active_rules, rule_decisions, rule_issues = _resolve_rule_conflicts(doc_rules, len(text))
    effective_rule_fp = fingerprint(doc_rules, 20)

    for position, rule in sorted(active_rules.items()):
        if rule["action"] == "split":
            fragments = _split_fragment(fragments, position, rule["rule_id"])

    merge_results: list[dict[str, Any]] = []
    for position, rule in sorted(active_rules.items()):
        if rule["action"] == "merge":
            fragments, merged = _merge_at_position(fragments, position, rule["rule_id"])
            merge_results.append({"position": position, "rule_id": rule["rule_id"], "merged": merged})

    enriched: list[dict[str, Any]] = []
    for seq, fragment in enumerate(sorted(fragments, key=lambda item: (item["start"], item["end"])), start=1):
        surface = text[fragment["start"] : fragment["end"]]
        exact_entry = index["by_term"].get(surface)
        entry_refs = list(fragment.get("entry_refs", []))
        candidate_entry_ids = list(fragment.get("candidate_entry_ids", []))
        exact_entry_id = None
        if exact_entry is not None:
            exact_entry_id = exact_entry["id"]
            candidate_entry_ids = [exact_entry_id]
            entry_refs.append(exact_entry_id)
        entry_refs = _dedupe(entry_refs)
        candidate_entry_ids = _dedupe(candidate_entry_ids)
        enriched.append(
            {
                "fragment_id": f"{doc['doc_id']}:F{seq:04d}",
                "fragment_seq": seq,
                "start": fragment["start"],
                "end": fragment["end"],
                "surface": surface,
                "entry_id": exact_entry_id or fragment.get("entry_id"),
                "exact_entry_id": exact_entry_id,
                "candidate_entry_ids": candidate_entry_ids,
                "entry_refs": sorted(entry_refs),
                "rule_ids": sorted(fragment.get("rule_ids", [])),
                "origin": fragment.get("origin", "fmm"),
            }
        )

    package_fp = package_fingerprint or rule_package_fingerprint(rules_document)
    segmentation_fp = fingerprint(
        {
            "algorithm": "deterministic-fmm-rule-v1",
            "lexicon_revision": lexicon_revision,
            "rule_package_fingerprint": package_fp,
            "effective_rule_fingerprint": effective_rule_fp,
            "fragments": [(f["start"], f["end"], f["surface"], f["entry_refs"]) for f in enriched],
        },
        20,
    )
    return {
        "doc_id": doc["doc_id"],
        "doc_seq": doc["doc_seq"],
        "text": text,
        "match_points": estimate_match_points(doc, index),
        "fmm_candidate_order": [c["candidate_id"] for c in fmm_order],
        "suppressed": suppressed,
        "rule_decisions": rule_decisions,
        "rule_issues": rule_issues,
        "merge_results": merge_results,
        "effective_rule_fingerprint": effective_rule_fp,
        "rule_package_fingerprint": package_fp,
        "segmentation_fingerprint": segmentation_fp,
        "fragments": enriched,
    }
