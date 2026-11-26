from __future__ import annotations

from typing import Any

from .canonical import fingerprint


def make_entry(
    *,
    stage: str,
    subject_id: str,
    config: dict[str, Any],
    cache_key: str,
    upstream: dict[str, Any],
    artifact_fingerprint: str,
    disposition: str,
    invalidation_reason: str,
    scope_fingerprint: str | None = None,
    reused_from_version: int | None = None,
) -> dict[str, Any]:
    return {
        "entry_id": f"{stage}:{subject_id}",
        "stage": stage,
        "subject_id": subject_id,
        "config_version": config["config_version"],
        "config_hash": config["config_hash"],
        "scope_fingerprint": scope_fingerprint,
        "cache_key": cache_key,
        "upstream_fingerprints": upstream,
        "artifact_fingerprint": artifact_fingerprint,
        "disposition": disposition,
        "reuse_hit": disposition == "reused",
        "invalidation_reason": invalidation_reason,
        "reused_from_global_version": reused_from_version,
    }


def assign_ordinals(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=lambda entry: entry["entry_id"])
    for ordinal, entry in enumerate(ordered):
        entry["ordinal"] = ordinal
    return ordered


def baseline_entry_map(baseline: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not baseline:
        return {}
    return {
        (entry["stage"], entry["subject_id"]): entry
        for entry in baseline.get("ledger", {}).get("entries", [])
    }


def explain_miss(
    *,
    stage: str,
    subject_id: str,
    cache_key: str,
    scope_fingerprint: str,
    upstream: dict[str, Any],
    baseline_entries: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, list[str]]:
    old = baseline_entries.get((stage, subject_id))
    if old is None:
        return "new_subject", ["subject_absent_from_baseline"]
    if old.get("cache_key") == cache_key:
        return "persisted_partial_result", ["cache_key_match_in_committed_partial_snapshot"]
    changed: list[str] = []
    if old.get("scope_fingerprint") != scope_fingerprint:
        changed.append("relevant_config_fingerprint")
    old_upstream = old.get("upstream_fingerprints", {})
    for key, value in upstream.items():
        if old_upstream.get(key) != value:
            changed.append(f"upstream:{key}")
    if not changed:
        changed.append("cache_key_algorithm_or_quantization")
    return "cache_miss", changed


def ledger_fingerprint(entries: list[dict[str, Any]]) -> str:
    return fingerprint(
        [
            {
                "entry_id": entry["entry_id"],
                "cache_key": entry["cache_key"],
                "artifact_fingerprint": entry["artifact_fingerprint"],
                "upstream_fingerprints": entry["upstream_fingerprints"],
            }
            for entry in sorted(entries, key=lambda item: item["ordinal"])
        ]
    )
