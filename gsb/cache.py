from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import cache_key, fingerprint


class ContentCache:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self.store.get(key)

    def put(self, key: str, value: Any) -> None:
        self.store[key] = value

    def invalidate(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    def keys(self) -> list[str]:
        return sorted(self.store)


@dataclass
class ReuseLedger:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        entity_type: str,
        entity_id: str,
        status: str,
        reason: str,
        cache_key: str | None = None,
        old_cache_key: str | None = None,
        new_cache_key: str | None = None,
        basis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {
            "seq": len(self.entries) + 1,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "status": status,
            "reason": reason,
            "cache_key": cache_key or new_cache_key,
            "old_cache_key": old_cache_key,
            "new_cache_key": new_cache_key,
            "basis": basis or {},
        }
        self.entries.append(entry)
        return entry

    def sorted_entries(self) -> list[dict[str, Any]]:
        ordered = sorted(
            self.entries,
            key=lambda entry: (
                str(entry["basis"].get("ticket_id", "")),
                entry["entity_type"],
                str(entry["entity_id"]),
                entry["status"],
                entry["seq"],
            ),
        )
        for index, entry in enumerate(ordered, 1):
            entry["canonical_order"] = index
        return ordered

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for entry in self.entries:
            result[entry["status"]] = result.get(entry["status"], 0) + 1
        result["total"] = len(self.entries)
        return result


def ticket_content_fingerprint(ticket: dict[str, Any]) -> str:
    return fingerprint({"ticket_id": ticket["ticket_id"], "tokens": ticket["tokens"]})


def ticket_full_fingerprint(ticket: dict[str, Any]) -> str:
    return fingerprint(ticket)


def _rows_for_terms(rows: list[dict[str, Any]], terms: set[str]) -> list[dict[str, Any]]:
    return sorted(
        ({key: row[key] for key in sorted(row)} for row in rows if row.get("term") in terms),
        key=lambda row: (int(row.get("order", 0)), str(row.get("term", ""))),
    )


def ticket_term_set(ticket: dict[str, Any]) -> set[str]:
    return set(ticket["tokens"])


def negation_dependency_fingerprint(bundle: dict[str, Any], ticket: dict[str, Any]) -> str:
    terms = ticket_term_set(ticket)
    return fingerprint(_rows_for_terms(bundle["triggers_doc"]["triggers"], terms))


def conflict_dependency_fingerprint(bundle: dict[str, Any], terms: set[str]) -> str:
    return fingerprint(_rows_for_terms(bundle["conflicts_doc"]["conflicts"], terms))


def vocab_dependency_fingerprint(bundle: dict[str, Any], terms: set[str] | None = None) -> str:
    rows = bundle["vocab_list"]
    if terms is not None:
        rows = [row for row in rows if row["term"] in terms]
    return fingerprint(rows)


def hit_cache_key(bundle: dict[str, Any], ticket: dict[str, Any], gear: str) -> str:
    terms = ticket_term_set(ticket)
    return cache_key(
        "hit",
        ticket_content_fingerprint(ticket),
        vocab_dependency_fingerprint(bundle, terms),
        fingerprint({"gear": gear}),
    )


def scope_cache_key(
    bundle: dict[str, Any], ticket: dict[str, Any], hit_key: str, gear: str
) -> str:
    scan_params = fingerprint({
        "scope_max_tokens": bundle["config"]["scope_max_tokens"],
        "punctuation_terms": sorted(
            row["term"] for row in bundle["vocab_list"] if row.get("kind") == "punctuation"
        ),
    })
    return cache_key(
        "scope",
        hit_key,
        negation_dependency_fingerprint(bundle, ticket),
        scan_params,
        gear,
    )


def settlement_cache_key(
    bundle: dict[str, Any],
    ticket: dict[str, Any],
    scope: dict[str, Any],
    scan: dict[str, Any],
    gear: str,
) -> str:
    token_set = set(scope["token_positions"] and [ticket["tokens"][pos] for pos in scope["token_positions"]])
    conflict_fp = conflict_dependency_fingerprint(bundle, token_set)
    config = bundle["config"]
    has_conflict = any(ticket["tokens"][pos] in bundle["conflicts"] for pos in scope["token_positions"])
    cancellation_fp = fingerprint({
        "cancellation_strength_points": config["cancellation_strength_points"],
        "score_lower_bound_points": config["score_lower_bound_points"],
        "rounding_mode": config["fixed_point"]["rounding_mode"],
    }) if has_conflict else "no-conflict"
    penalty_fp = fingerprint({
        "length_penalty_bp_per_token": config["length_penalty_bp_per_token"],
        "length_min_retention_bp": config["length_min_retention_bp"],
        "length": scope["length"],
        "rounding_mode": config["fixed_point"]["rounding_mode"],
    }) if scope["length"] > 1 else "length-one-no-penalty"
    return cache_key(
        "settlement",
        scope["scope_id"],
        scope_cache_key(bundle, ticket, hit_cache_key(bundle, ticket, gear), gear),
        conflict_fp,
        cancellation_fp,
        penalty_fp,
        gear,
    )


def conclusion_cache_key(
    bundle: dict[str, Any],
    ticket: dict[str, Any],
    settlement_keys: list[str],
    gear: str,
) -> str:
    config = bundle["config"]
    composition_fp = fingerprint({
        "positive_threshold_points": config["positive_threshold_points"],
        "negative_threshold_points": config["negative_threshold_points"],
        "risk_thresholds_points": config["risk_thresholds_points"],
        "boundary_half_step_decimal_width": config.get("boundary_half_step_decimal_width", 2),
        "rounding_mode": config["fixed_point"]["rounding_mode"],
    })
    channel_fp = fingerprint({
        "channel_id": ticket["channel_id"],
        "weight_permille": bundle["channels"][ticket["channel_id"]],
    })
    metadata_fp = fingerprint({
        "annotator_version": ticket["annotator_version"],
        "completion_status": ticket["completion_status"],
    })
    return cache_key(
        "conclusion",
        ticket_full_fingerprint(ticket),
        fingerprint(settlement_keys),
        composition_fp,
        channel_fp,
        metadata_fp,
        gear,
    )
