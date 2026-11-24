"""Stable partition and window identities, independent of dict iteration order."""

from __future__ import annotations

from typing import Mapping
from urllib.parse import quote

from .canonical import canonical_json
from .spec import RuleSpec


def partition_dict(
    key_values: Mapping[str, object], rule: RuleSpec
) -> dict[str, object]:
    return {key: key_values[key] for key in rule.partition_keys}


def partition_label(partition: Mapping[str, object]) -> str:
    return quote(canonical_json(dict(partition)), safe="")


def make_window_id(rule_id: str, partition: Mapping[str, object], start_ms: int) -> str:
    return f"{rule_id}|{partition_label(partition)}|{int(start_ms)}"
