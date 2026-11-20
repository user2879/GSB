"""Immutable domain model for the scheduling workbench."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PLUGIN_NAMES: tuple[str, ...] = (
    "node_affinity",
    "anti_affinity",
    "resource_balance",
    "image_locality",
)
TAINT_EFFECTS = frozenset({"NoSchedule", "PreferNoSchedule", "NoExecute"})
SELECTOR_OPERATORS = frozenset({"In", "NotIn", "Exists", "DoesNotExist"})

_LABEL_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[-A-Za-z0-9_.]*[A-Za-z0-9])?$")
_LABEL_PREFIX_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*$")
_NODE_NAME_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9.]*[a-z0-9])?$")
_DNS_SUBDOMAIN_MAX = 253
_LABEL_NAME_MAX = 63
_LABEL_VALUE_RE = _LABEL_NAME_RE


def valid_label_key(key: str) -> bool:
    if not isinstance(key, str) or not key or len(key) > 317:
        return False
    if "/" in key:
        prefix, name = key.split("/", 1)
        return (
            bool(prefix)
            and len(prefix) <= _DNS_SUBDOMAIN_MAX
            and bool(_LABEL_PREFIX_RE.fullmatch(prefix))
            and bool(name)
            and len(name) <= _LABEL_NAME_MAX
            and bool(_LABEL_NAME_RE.fullmatch(name))
        )
    return len(key) <= _LABEL_NAME_MAX and bool(_LABEL_NAME_RE.fullmatch(key))


def valid_label_value(value: str) -> bool:
    if not isinstance(value, str) or len(value) > _LABEL_NAME_MAX:
        return False
    return value == "" or bool(_LABEL_VALUE_RE.fullmatch(value))


def valid_node_name(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _DNS_SUBDOMAIN_MAX
        and value == value.lower()
        and bool(_NODE_NAME_RE.fullmatch(value))
    )


def valid_pod_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _DNS_SUBDOMAIN_MAX
        and value == value.lower()
        and bool(_NODE_NAME_RE.fullmatch(value))
    )


@dataclass(frozen=True)
class Expression:
    key: str
    operator: str
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class Selector:
    match_labels: dict[str, str] = field(default_factory=dict)
    match_expressions: tuple[Expression, ...] = ()

    def referenced_keys(self) -> set[str]:
        return set(self.match_labels) | {expression.key for expression in self.match_expressions}

    def matches(self, labels: dict[str, str]) -> bool:
        for key, value in self.match_labels.items():
            if labels.get(key) != value:
                return False
        for expression in self.match_expressions:
            present = expression.key in labels
            actual = labels.get(expression.key)
            if expression.operator == "In" and (not present or actual not in expression.values):
                return False
            if expression.operator == "NotIn" and present and actual in expression.values:
                return False
            if expression.operator == "Exists" and not present:
                return False
            if expression.operator == "DoesNotExist" and present:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.match_labels:
            result["matchLabels"] = dict(sorted(self.match_labels.items()))
        if self.match_expressions:
            result["matchExpressions"] = [
                {
                    "key": expression.key,
                    "operator": expression.operator,
                    **({"values": list(expression.values)} if expression.operator in {"In", "NotIn"} else {}),
                }
                for expression in self.match_expressions
            ]
        return result


@dataclass(frozen=True)
class SoftRule:
    selector: Selector
    weight: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"selector": self.selector.to_dict(), "weight": self.weight}


@dataclass(frozen=True)
class AntiRule:
    selector: Selector
    topology_key: str
    weight: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "selector": self.selector.to_dict(),
            "topologyKey": self.topology_key,
        }
        if self.weight is not None:
            result["weight"] = self.weight
        return result


@dataclass(frozen=True)
class Toleration:
    key: str = ""
    operator: str = "Equal"
    value: str = ""
    effect: str | None = None

    def tolerates(self, taint: "Taint") -> bool:
        if self.effect is not None and self.effect != taint.effect:
            return False
        if self.operator == "Exists":
            return self.key == "" or self.key == taint.key
        return self.key == taint.key and self.value == taint.value

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"key": self.key, "operator": self.operator}
        if self.value:
            result["value"] = self.value
        if self.effect is not None:
            result["effect"] = self.effect
        return result


@dataclass(frozen=True)
class Taint:
    key: str
    value: str = ""
    effect: str = "NoSchedule"

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "effect": self.effect}


@dataclass(frozen=True)
class Node:
    name: str
    labels: dict[str, str]
    cpu: int
    memory: int
    taints: tuple[Taint, ...] = ()
    images: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "labels": dict(sorted(self.labels.items())),
            "allocatable": {"cpu": self.cpu, "memory": self.memory},
            "taints": [taint.to_dict() for taint in self.taints],
            "images": list(self.images),
        }


@dataclass(frozen=True)
class Pod:
    pod_id: str
    labels: dict[str, str]
    cpu: int
    memory: int
    image: str | None = None
    node_affinity_hard: Selector | None = None
    node_affinity_soft: tuple[SoftRule, ...] = ()
    anti_affinity_hard: tuple[AntiRule, ...] = ()
    anti_affinity_soft: tuple[AntiRule, ...] = ()
    tolerations: tuple[Toleration, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.pod_id,
            "labels": dict(sorted(self.labels.items())),
            "resources": {"cpu": self.cpu, "memory": self.memory},
        }
        if self.image is not None:
            result["image"] = self.image
        if self.node_affinity_hard is not None or self.node_affinity_soft:
            affinity: dict[str, Any] = {}
            if self.node_affinity_hard is not None:
                affinity["hard"] = self.node_affinity_hard.to_dict()
            if self.node_affinity_soft:
                affinity["soft"] = [rule.to_dict() for rule in self.node_affinity_soft]
            result["nodeAffinity"] = affinity
        if self.anti_affinity_hard or self.anti_affinity_soft:
            anti: dict[str, Any] = {}
            if self.anti_affinity_hard:
                anti["hard"] = [rule.to_dict() for rule in self.anti_affinity_hard]
            if self.anti_affinity_soft:
                anti["soft"] = [rule.to_dict() for rule in self.anti_affinity_soft]
            result["antiAffinity"] = anti
        if self.tolerations:
            result["tolerations"] = [toleration.to_dict() for toleration in self.tolerations]
        return result

    def node_selector_keys(self) -> set[str]:
        """Label keys that must exist in the node label universe."""
        keys: set[str] = set()
        if self.node_affinity_hard is not None:
            keys |= self.node_affinity_hard.referenced_keys()
        for rule in self.node_affinity_soft:
            keys |= rule.selector.referenced_keys()
        for rule in self.anti_affinity_hard + self.anti_affinity_soft:
            keys.add(rule.topology_key)
        return keys

    def anti_selector_keys(self) -> set[str]:
        """Label keys that anti-affinity selectors read from other pods."""
        keys: set[str] = set()
        for rule in self.anti_affinity_hard + self.anti_affinity_soft:
            keys |= rule.selector.referenced_keys()
        return keys

    def selector_keys(self) -> set[str]:
        return self.node_selector_keys() | self.anti_selector_keys()


@dataclass(frozen=True)
class PluginConfig:
    weights: dict[str, int]
    version: int = 1

    def normalized(self) -> "PluginConfig":
        return PluginConfig(weights={name: int(self.weights[name]) for name in PLUGIN_NAMES}, version=self.version)

    def total_weight(self) -> int:
        return sum(self.weights.values())

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "weights": dict(sorted(self.weights.items()))}


DEFAULT_PLUGINS = PluginConfig(
    weights={
        "node_affinity": 3,
        "anti_affinity": 2,
        "resource_balance": 3,
        "image_locality": 2,
    },
    version=1,
)
