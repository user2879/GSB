from __future__ import annotations

from typing import Any

from .canonical import fingerprint


def phrase_char_length(tokens: list[str], start: int, end: int) -> int:
    selected = tokens[start:end]
    return sum(len(token) for token in selected) + max(0, len(selected) - 1)


def find_term_occurrences(tokens: list[str], term: str) -> list[tuple[int, int]]:
    phrase = term.split()
    if not phrase:
        return []
    occurrences: list[tuple[int, int]] = []
    width = len(phrase)
    for start in range(0, len(tokens) - width + 1):
        if tokens[start : start + width] == phrase:
            occurrences.append((start, start + width))
    return occurrences


def trigger_rule_fingerprint(schema_event_type: dict[str, Any]) -> str:
    return fingerprint(
        {"trigger_terms": sorted(schema_event_type.get("trigger_terms", []))}, 16
    )


def trigger_key_parts(
    event: dict[str, Any],
    schema_event_type: dict[str, Any],
    revision: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, str]:
    content_fp = fingerprint(
        {"event_id": event["event_id"], "tokens": event["tokens"]}, 16
    )
    rule_fp = trigger_rule_fingerprint(schema_event_type)
    cache_key = fingerprint(
        {
            "artifact": "trigger_match",
            "event_content_fingerprint": content_fp,
            "trigger_rule_revision_id": (
                scope["trigger_rule_revision_id"] if scope else revision["revision_id"]
            ),
            "trigger_rule_revision_version": (
                scope["trigger_rule_revision_version"] if scope else revision["version"]
            ),
            "trigger_rule_fingerprint": rule_fp,
            "algorithm": "char_span_desc/term_asc/start_asc/greedy/v1",
        }
    )
    return {
        "cache_key": cache_key,
        "content_fingerprint": content_fp,
        "rule_pack_fingerprint": rule_fp,
        "revision_fingerprint": (
            scope["schema_slice_fingerprint"] if scope else fingerprint(revision, 16)
        ),
    }


def match_event_triggers(
    event: dict[str, Any],
    schema_event_type: dict[str, Any],
    revision: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = event["event_id"]
    tokens = event["tokens"]
    terms = sorted(set(schema_event_type.get("trigger_terms", [])))
    candidates: list[dict[str, Any]] = []

    for term in terms:
        for start, end in find_term_occurrences(tokens, term):
            char_length = phrase_char_length(tokens, start, end)
            candidate_key = fingerprint(
                {"event_id": event_id, "term": term, "start": start, "end": end}, 10
            )
            candidates.append(
                {
                    "candidate_id": f"TRGC-{event_id}-{start:02d}-{end:02d}-{candidate_key}",
                    "match_id": f"TRG-{event_id}-{start:02d}-{end:02d}-{candidate_key}",
                    "event_id": event_id,
                    "term": term,
                    "token_start": start,
                    "token_end": end,
                    "char_span_length": char_length,
                    "priority_key": [-char_length, term, start, end],
                }
            )

    candidates.sort(key=lambda item: tuple(item["priority_key"]))
    accepted: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []

    for rank, candidate in enumerate(candidates):
        candidate["greedy_rank"] = rank
        overlapper = next(
            (
                accepted_item
                for accepted_item in accepted
                if candidate["token_start"] < accepted_item["token_end"]
                and candidate["token_end"] > accepted_item["token_start"]
            ),
            None,
        )
        if overlapper is None:
            candidate["selection"] = "accepted"
            candidate["selection_basis"] = (
                "longest_char_span:first_non_overlapping_in_priority_order"
            )
            accepted.append(candidate)
        else:
            candidate["selection"] = "suppressed"
            candidate["suppressed_by"] = overlapper["match_id"]
            candidate["suppressed_by_term"] = overlapper["term"]
            candidate["suppression_reason"] = "overlapping_longer_or_prior_trigger"
            suppressed.append(candidate)

    accepted.sort(key=lambda item: (item["token_start"], item["token_end"], item["term"]))
    suppressed.sort(
        key=lambda item: (item["token_start"], item["token_end"], item["term"])
    )
    for index, match in enumerate(accepted):
        match["output_order"] = index

    key_parts = trigger_key_parts(event, schema_event_type, revision, scope)
    content_fp = key_parts["content_fingerprint"]
    rule_fp = key_parts["rule_pack_fingerprint"]
    cache_key = key_parts["cache_key"]
    return {
        "artifact_id": f"trigger:{event_id}",
        "cache_key": cache_key,
        "event_id": event_id,
        "event_type": event["event_type"],
        "active": bool(accepted),
        "content_fingerprint": content_fp,
        "schema_revision_id": revision["revision_id"],
        "schema_version": revision["version"],
        "rule_pack_fingerprint": rule_fp,
        "matching_rule": "char_span_length_desc,term_lexicographic_asc,start_asc",
        "matches": accepted,
        "suppressed": suppressed,
        "candidate_count": len(candidates),
        "match_count": len(accepted),
        "basis": [
            "controlled_token_sequence_exact_phrase",
            "character_span_length_used_for_primary_order",
            "overlap_candidate_recorded_with_suppressed_by",
            f"schema_revision={revision['revision_id']}@v{revision['version']}",
        ],
    }


def empty_trigger_result(event_id: str, reason: str) -> dict[str, Any]:
    return {
        "artifact_id": f"trigger:{event_id}",
        "event_id": event_id,
        "active": False,
        "matches": [],
        "suppressed": [],
        "match_count": 0,
        "reason": reason,
    }
