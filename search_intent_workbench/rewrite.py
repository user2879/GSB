"""Deterministic rewrite suggestion generation."""

from __future__ import annotations

import re
from typing import Any

from .canonical import digest


REWRITE_ALGORITHM_VERSION = "rewrite-v1"
ENTITY_PLACEHOLDER = re.compile(r"\{entity:([a-z_]+)\}")


def select_rewrite_rule(
    primary_intent: str | None, rules: dict[str, Any]
) -> dict[str, Any] | None:
    candidates = [
        rule
        for rule in rules.get("rewrite_rules", [])
        if isinstance(rule, dict) and rule.get("intent") == primary_intent
    ]
    candidates.sort(key=lambda rule: (rule.get("priority", 0), rule.get("id", "")))
    return candidates[0] if candidates else None


def rewrite_cache_key(
    content_fingerprint: str,
    normalization_key: str,
    intent_key: str,
    slot_key: str,
    rewrite_rule: dict[str, Any] | None,
    rules: dict[str, Any],
) -> str:
    return digest(
        {
            "algorithm": REWRITE_ALGORITHM_VERSION,
            "content_fingerprint": content_fingerprint,
            "normalization_key": normalization_key,
            "intent_key": intent_key,
            "slot_key": slot_key,
            "rule_table_version": rules.get("version"),
            "rewrite_rule": rewrite_rule,
        }
    )


def build_rewrite(
    normalized_text: str,
    primary_intent: str | None,
    slots: list[dict[str, Any]],
    rules: dict[str, Any],
) -> dict[str, Any]:
    rule = select_rewrite_rule(primary_intent, rules)
    if rule is None:
        return {
            "rule_id": None,
            "rule_revision": None,
            "template": None,
            "text": normalized_text,
            "referenced_entity_ids": [],
            "placeholders": [],
        }

    by_type: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        by_type.setdefault(slot["entity_type"], []).append(slot)
    placeholder_records: list[dict[str, Any]] = []

    def replace_entity(match: re.Match[str]) -> str:
        entity_type = match.group(1)
        matches = by_type.get(entity_type, [])
        if not matches:
            placeholder_records.append(
                {
                    "placeholder": match.group(0),
                    "entity_type": entity_type,
                    "resolved": False,
                    "entity_ids": [],
                    "value": "相关商品",
                }
            )
            return "相关商品"
        seen: set[tuple[str, str]] = set()
        values: list[str] = []
        for slot in sorted(matches, key=lambda item: item["token_start"]):
            key = (slot["entry_id"], slot["matched_text"])
            if key in seen:
                continue
            seen.add(key)
            values.append(slot["matched_text"])
        value = "、".join(values)
        placeholder_records.append(
            {
                "placeholder": match.group(0),
                "entity_type": entity_type,
                "resolved": True,
                "entity_ids": [slot["entry_id"] for slot in matches],
                "value": value,
            }
        )
        return value

    text = ENTITY_PLACEHOLDER.sub(replace_entity, rule["template"])
    text = text.replace("{normalized_query}", normalized_text or "空查询")
    text = text.replace("{query}", normalized_text or "空查询")
    text = text.replace("{intent}", primary_intent or "unknown")
    referenced_entity_ids = sorted(
        {
            slot["entry_id"]
            for placeholder in placeholder_records
            for slot_id in placeholder["entity_ids"]
            for slot in slots
            if slot["entry_id"] == slot_id
        }
    )
    return {
        "rule_id": rule["id"],
        "rule_revision": rule["revision"],
        "rule_priority": rule["priority"],
        "template": rule["template"],
        "text": text,
        "referenced_entity_ids": referenced_entity_ids,
        "placeholders": placeholder_records,
    }


def rewrite_output_fingerprint(result: dict[str, Any]) -> str:
    return digest({"algorithm": REWRITE_ALGORITHM_VERSION, "result": result})
