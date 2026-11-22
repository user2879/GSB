"""Longest-match entity slot extraction with deterministic overlap resolution."""

from __future__ import annotations

from typing import Any

from .canonical import digest
from .normalization import normalize_match_text, normalize_match_tokens


SLOT_ALGORITHM_VERSION = "slots-v1"


def build_entity_patterns(
    config: dict[str, Any], rules: dict[str, Any]
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(config.get("entities", [])):
        normalized_term = normalize_match_text(entry["term"], rules, config)
        pattern_tokens = normalize_match_tokens(entry["term"], rules, config)
        if not pattern_tokens:
            continue
        patterns.append(
            {
                "entry_index": entry_index,
                "entry_id": entry["entry_id"],
                "dictionary_term": entry["term"],
                "normalized_term": normalized_term,
                "pattern_tokens": pattern_tokens,
                "entity_type": entry["entity_type"],
                "priority": float(entry["priority"]),
            }
        )
    return patterns


def find_candidates(
    tokens: list[str], patterns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pattern in patterns:
        pattern_tokens = pattern["pattern_tokens"]
        width = len(pattern_tokens)
        if width > len(tokens):
            continue
        for start in range(0, len(tokens) - width + 1):
            if tokens[start : start + width] != pattern_tokens:
                continue
            candidates.append(
                {
                    "entry_id": pattern["entry_id"],
                    "entry_index": pattern["entry_index"],
                    "dictionary_term": pattern["dictionary_term"],
                    "normalized_term": pattern["normalized_term"],
                    "entity_type": pattern["entity_type"],
                    "priority": pattern["priority"],
                    "token_start": start,
                    "token_end": start + width,
                    "token_length": width,
                    "matched_text": " ".join(tokens[start : start + width]),
                }
            )
    candidates.sort(
        key=lambda item: (
            -item["token_length"],
            -item["priority"],
            item["token_start"],
            item["entry_index"],
            item["entry_id"],
        )
    )
    for rank, candidate in enumerate(candidates):
        candidate["rank"] = rank
    return candidates


def _overlaps(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return (
        first["token_start"] < second["token_end"]
        and first["token_end"] > second["token_start"]
    )


def resolve_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedily accept best candidates; retain every swallowed overlap."""

    accepted: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in candidates:
        overlaps = [slot for slot in accepted if _overlaps(candidate, slot)]
        if overlaps:
            record = dict(candidate)
            record["suppression_reason"] = "overlap"
            record["overlaps_slot_entry_ids"] = [slot["entry_id"] for slot in overlaps]
            suppressed.append(record)
        else:
            accepted.append(dict(candidate))
    accepted.sort(
        key=lambda item: (
            item["token_start"],
            item["token_end"],
            -item["priority"],
            item["entry_id"],
        )
    )
    suppressed.sort(key=lambda item: item["rank"])
    return accepted, suppressed


def slot_cache_key(
    content_fingerprint: str,
    normalization_key: str,
    candidates: list[dict[str, Any]],
    rules: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    references = sorted(
        [
            {
                "entry_id": candidate["entry_id"],
                "entry_index": candidate["entry_index"],
                "normalized_term": candidate["normalized_term"],
                "entity_type": candidate["entity_type"],
                "priority": candidate["priority"],
            }
            for candidate in candidates
        ],
        key=lambda item: (item["entry_index"], item["entry_id"]),
    )
    key_material = {
        "algorithm": SLOT_ALGORITHM_VERSION,
        "content_fingerprint": content_fingerprint,
        "normalization_key": normalization_key,
        "rule_table_version": rules.get("version"),
        "dictionary_references": references,
        "entity_type_vocabulary": sorted(config.get("entity_types", [])),
    }
    return digest(key_material), references


def slot_output_fingerprint(
    slots: list[dict[str, Any]], suppressed: list[dict[str, Any]]
) -> str:
    return digest(
        {
            "algorithm": SLOT_ALGORITHM_VERSION,
            "slots": slots,
            "suppressed": suppressed,
        }
    )
