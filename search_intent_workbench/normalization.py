"""Deterministic normalization and tokenization."""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

from .canonical import canonical_json, digest


NORMALIZATION_ALGORITHM_VERSION = "norm-v1"


def _ordered_normalization_rules(
    rules: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {
        rule["id"]: rule
        for rule in rules.get("normalization_rules", [])
        if isinstance(rule, dict) and isinstance(rule.get("id"), str)
    }
    return [
        by_id[rule_id]
        for rule_id in config.get("normalization", {}).get("rule_ids", [])
        if rule_id in by_id
    ]


def _rule_is_applicable(rule: dict[str, Any], text: str) -> bool:
    kind = rule.get("kind")
    if kind == "case_fold":
        return text.casefold() != text
    if kind == "width_fold":
        return unicodedata.normalize("NFKC", text) != text
    if kind == "whitespace_collapse":
        return text != " ".join(text.split())
    if kind == "token_split":
        return any(character.isspace() for character in text)
    return True


def relevant_rule_snapshots(
    source_terms: list[str], rules: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return only rule settings that can affect this query's normalization."""

    snapshots: list[dict[str, Any]] = []
    for rule in _ordered_normalization_rules(rules, config):
        if any(_rule_is_applicable(rule, term) for term in source_terms):
            snapshots.append(
                {
                    "id": rule["id"],
                    "kind": rule["kind"],
                    "enabled": rule["enabled"],
                    "revision": rule["revision"],
                    "position": config["normalization"]["rule_ids"].index(rule["id"]),
                }
            )
    return snapshots


def normalization_cache_key(
    content_fingerprint: str,
    source_terms: list[str],
    rules: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    references = relevant_rule_snapshots(source_terms, rules, config)
    key_material = {
        "algorithm": NORMALIZATION_ALGORITHM_VERSION,
        "content_fingerprint": content_fingerprint,
        "rule_table_version": rules.get("version"),
        "rule_references": references,
    }
    return digest(key_material), references


def _apply_text_rules(text: str, rules: list[dict[str, Any]]) -> str:
    value = text
    for rule in rules:
        if not rule.get("enabled"):
            continue
        kind = rule.get("kind")
        if kind == "case_fold":
            value = value.casefold()
        elif kind == "width_fold":
            value = unicodedata.normalize("NFKC", value)
        elif kind == "whitespace_collapse":
            value = " ".join(value.split())
        elif kind == "token_split":
            continue
    return value


def normalize_tokens(
    source_terms: list[str], rules: dict[str, Any], config: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize and tokenize terms, retaining source-term traceability."""

    ordered_rules = _ordered_normalization_rules(rules, config)
    split_enabled = any(
        rule.get("kind") == "token_split" and rule.get("enabled")
        for rule in ordered_rules
    )
    tokens: list[str] = []
    token_records: list[dict[str, Any]] = []
    normalized_terms: list[dict[str, Any]] = []
    for source_index, term in enumerate(source_terms):
        normalized = _apply_text_rules(term, ordered_rules)
        normalized_terms.append(
            {
                "source_index": source_index,
                "source_term": term,
                "normalized_term": normalized,
            }
        )
        pieces = normalized.split() if split_enabled else ([normalized] if normalized else [])
        for piece in pieces:
            if not piece:
                continue
            token_records.append(
                {
                    "token_index": len(tokens),
                    "text": piece,
                    "source_term_index": source_index,
                }
            )
            tokens.append(piece)
    return tokens, token_records, normalized_terms


def normalize_match_text(
    text: str, rules: dict[str, Any], config: dict[str, Any]
) -> str:
    """Normalize a feature or dictionary phrase for matching."""

    tokens, _, _ = normalize_tokens([text], rules, config)
    return " ".join(tokens)


def normalize_match_tokens(
    text: str, rules: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    tokens, _, _ = normalize_tokens([text], rules, config)
    return tokens


def output_fingerprint(
    tokens: list[str],
    normalized_terms: list[dict[str, Any]],
    rule_references: list[dict[str, Any]],
) -> str:
    material = {
        "algorithm": NORMALIZATION_ALGORITHM_VERSION,
        "tokens": tokens,
        "normalized_terms": normalized_terms,
        "rules": rule_references,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()
