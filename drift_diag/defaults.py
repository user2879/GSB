"""Explicit, versioned default rules used when older declarations are normalized."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .errors import format_error


@dataclass(frozen=True)
class DefaultRule:
    field: str
    value: Any
    rule_id: str

    def materialize(self) -> Any:
        return deepcopy(self.value)


@dataclass(frozen=True)
class DefaultRuleSet:
    version: str
    rules: dict[str, DefaultRule]


def _rule_set(version: str, image: str, resource_limit: int, probe_ms: int, labels: dict[str, str]) -> DefaultRuleSet:
    return DefaultRuleSet(
        version=version,
        rules={
            "image": DefaultRule("image", image, f"{version}/image/registry-baseline"),
            "resource_limit": DefaultRule(
                "resource_limit", resource_limit, f"{version}/resource_limit/baseline"
            ),
            "probe_ms": DefaultRule("probe_ms", probe_ms, f"{version}/probe_ms/baseline"),
            "labels": DefaultRule("labels", labels, f"{version}/labels/baseline"),
            "parent_ref": DefaultRule("parent_ref", None, f"{version}/parent_ref/no-parent"),
        },
    )


DEFAULT_RULE_SETS: dict[str, DefaultRuleSet] = {
    "defaults-v1": _rule_set(
        "defaults-v1",
        image="registry.local/default:1.0",
        resource_limit=1024,
        probe_ms=500,
        labels={"defaulted": "v1"},
    ),
    "defaults-v2": _rule_set(
        "defaults-v2",
        image="registry.local/default:2.0",
        resource_limit=2048,
        probe_ms=750,
        labels={"defaulted": "v2"},
    ),
}


def get_rule_set(version: str) -> DefaultRuleSet:
    try:
        return DEFAULT_RULE_SETS[version]
    except KeyError as exc:
        raise format_error(
            f"unknown defaults rule version: {version}",
            item=version,
            field="defaults_version",
            details={"known_versions": sorted(DEFAULT_RULE_SETS)},
        ) from exc


def get_default_rule(version: str, field: str) -> DefaultRule:
    rule_set = get_rule_set(version)
    try:
        return rule_set.rules[field]
    except KeyError as exc:
        raise format_error(
            f"no default rule for field {field} in {version}",
            item=version,
            field=field,
        ) from exc
