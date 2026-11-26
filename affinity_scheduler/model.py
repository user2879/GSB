from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


PLUGIN_NAMES: tuple[str, ...] = (
    "taint_toleration",
    "node_affinity",
    "pod_anti_affinity",
    "resource_fit",
)

TAINT_EFFECTS: dict[int, str] = {
    0: "NoSchedule",
    1: "PreferNoSchedule",
    2: "NoExecute",
}

TAINT_FILTER_EFFECTS = frozenset({0, 2})

VETO_CODES: dict[int, str] = {
    0: "NONE",
    10: "TAINT_TOLERATION",
    20: "NODE_AFFINITY",
    30: "POD_ANTI_AFFINITY",
    40: "RESOURCE_FIT",
    50: "NO_FEASIBLE_NODE",
}

FILTER_ORDER: tuple[str, ...] = (
    "taint_toleration",
    "node_affinity",
    "pod_anti_affinity",
    "resource_fit",
)

VETO_BY_PLUGIN: dict[str, int] = {
    "taint_toleration": 10,
    "node_affinity": 20,
    "pod_anti_affinity": 30,
    "resource_fit": 40,
}

DEFAULT_TIE_BREAK: tuple[str, ...] = (*PLUGIN_NAMES, "node_name")


@dataclass(frozen=True)
class Taint:
    key: str
    value: str
    effect: int

    @property
    def effect_name(self) -> str:
        return TAINT_EFFECTS[self.effect]


@dataclass(frozen=True)
class Toleration:
    key: str
    operator: str
    value: str
    effect: int


@dataclass(frozen=True)
class LabelRequirement:
    key: str
    operator: str
    values: tuple[str, ...] = ()
    weight: int = 1


@dataclass(frozen=True)
class PodSelectorRequirement:
    key: str
    operator: str
    values: tuple[str, ...]
    topology_key: str
    weight: int = 1


@dataclass(frozen=True)
class Node:
    name: str
    labels: Mapping[str, str]
    cpu: int
    memory: int
    taints: tuple[Taint, ...] = ()


@dataclass(frozen=True)
class Pod:
    pod_id: str
    labels: Mapping[str, str]
    request_cpu: int
    request_memory: int
    node_hard: tuple[LabelRequirement, ...] = ()
    node_soft: tuple[LabelRequirement, ...] = ()
    anti_hard: tuple[PodSelectorRequirement, ...] = ()
    anti_soft: tuple[PodSelectorRequirement, ...] = ()
    tolerations: tuple[Toleration, ...] = ()

    @property
    def plugin_dependencies(self) -> frozenset[str]:
        dependencies: set[str] = set()
        if self.node_hard or self.node_soft:
            dependencies.add("node_affinity")
        if self.anti_hard or self.anti_soft:
            dependencies.add("pod_anti_affinity")
        if self.tolerations:
            dependencies.add("taint_toleration")
        dependencies.add("resource_fit")
        return frozenset(dependencies)


@dataclass(frozen=True)
class ServiceGroup:
    name: str
    node_names: tuple[str, ...]
    port: int


@dataclass(frozen=True)
class ServiceConfig:
    host: str
    port_base: int
    timeout_ms: int
    groups: tuple[ServiceGroup, ...]


@dataclass(frozen=True)
class Spec:
    source_path: str
    parameter_version: int
    scale_exponent: int
    weights_myriad: Mapping[str, int]
    nodes: tuple[Node, ...]
    pods: tuple[Pod, ...]
    service: ServiceConfig

    @property
    def scale_factor(self) -> int:
        return 10**self.scale_exponent

    @property
    def node_by_name(self) -> Mapping[str, Node]:
        return {node.name: node for node in self.nodes}

    @property
    def pod_by_id(self) -> Mapping[str, Pod]:
        return {pod.pod_id: pod for pod in self.pods}
