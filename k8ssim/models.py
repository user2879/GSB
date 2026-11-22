"""Typed domain models for nodes, pods and scheduling conclusions.

The models are plain immutable dataclasses with explicit ``to_dict`` /
``from_dict`` conversions; nothing here performs I/O or validation policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

from .constants import PLUGIN_ORDER

# --------------------------------------------------------------------------- #
# Selectors and affinity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabelRequirement:
    key: str
    operator: str           # In | NotIn | Exists | DoesNotExist
    values: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"key": self.key, "operator": self.operator, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: dict) -> "LabelRequirement":
        return cls(
            key=data["key"],
            operator=data["operator"],
            values=tuple(data.get("values", ())),
        )


@dataclass(frozen=True)
class PreferredTerm:
    weight: int
    requirement: LabelRequirement

    def to_dict(self) -> dict:
        return {"weight": self.weight, "requirement": self.requirement.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "PreferredTerm":
        return cls(
            weight=int(data["weight"]),
            requirement=LabelRequirement.from_dict(data["requirement"]),
        )


@dataclass(frozen=True)
class NodeAffinity:
    required: Tuple[LabelRequirement, ...] = ()
    preferred: Tuple[PreferredTerm, ...] = ()

    def to_dict(self) -> dict:
        return {
            "required": [r.to_dict() for r in self.required],
            "preferred": [t.to_dict() for t in self.preferred],
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NodeAffinity":
        if not data:
            return cls()
        return cls(
            required=tuple(LabelRequirement.from_dict(r) for r in data.get("required", [])),
            preferred=tuple(PreferredTerm.from_dict(t) for t in data.get("preferred", [])),
        )


@dataclass(frozen=True)
class PodAntiAffinity:
    # Required hard terms: do not co-locate with a matching pod on the topology key.
    required: Tuple[LabelRequirement, ...] = ()
    required_topology_key: Optional[str] = None
    # Preferred soft terms with their positive weights.
    preferred: Tuple[PreferredTerm, ...] = ()
    preferred_topology_key: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "required": [r.to_dict() for r in self.required],
            "required_topology_key": self.required_topology_key,
            "preferred": [t.to_dict() for t in self.preferred],
            "preferred_topology_key": self.preferred_topology_key,
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PodAntiAffinity":
        if not data:
            return cls()
        return cls(
            required=tuple(LabelRequirement.from_dict(r) for r in data.get("required", [])),
            required_topology_key=data.get("required_topology_key"),
            preferred=tuple(PreferredTerm.from_dict(t) for t in data.get("preferred", [])),
            preferred_topology_key=data.get("preferred_topology_key"),
        )


# --------------------------------------------------------------------------- #
# Taints and tolerations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Taint:
    key: str
    value: str
    effect: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Taint":
        return cls(key=data["key"], value=data.get("value", ""), effect=data["effect"])


@dataclass(frozen=True)
class Toleration:
    key: str
    operator: str           # Exists | Equal
    value: str = ""
    effect: str = ""        # empty means every effect is tolerated

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Toleration":
        return cls(
            key=data.get("key", ""),
            operator=data.get("operator", "Equal"),
            value=data.get("value", ""),
            effect=data.get("effect", ""),
        )


# --------------------------------------------------------------------------- #
# Nodes and pods
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Node:
    name: str
    labels: Dict[str, str]
    cpu: int
    memory: int
    taints: Tuple[Taint, ...]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "labels": dict(self.labels),
            "cpu": self.cpu,
            "memory": self.memory,
            "taints": [t.to_dict() for t in self.taints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        return cls(
            name=data["name"],
            labels=dict(data.get("labels", {})),
            cpu=int(data["cpu"]),
            memory=int(data["memory"]),
            taints=tuple(Taint.from_dict(t) for t in data.get("taints", [])),
        )


@dataclass(frozen=True)
class Pod:
    pod_id: str
    cpu_request: int
    memory_request: int
    labels: Dict[str, str] = field(default_factory=dict)
    node_affinity: NodeAffinity = field(default_factory=NodeAffinity)
    pod_anti_affinity: PodAntiAffinity = field(default_factory=PodAntiAffinity)
    tolerations: Tuple[Toleration, ...] = ()

    def to_dict(self) -> dict:
        return {
            "pod_id": self.pod_id,
            "cpu_request": self.cpu_request,
            "memory_request": self.memory_request,
            "labels": dict(self.labels),
            "node_affinity": self.node_affinity.to_dict(),
            "pod_anti_affinity": self.pod_anti_affinity.to_dict(),
            "tolerations": [t.to_dict() for t in self.tolerations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pod":
        return cls(
            pod_id=data["pod_id"],
            cpu_request=int(data["cpu_request"]),
            memory_request=int(data["memory_request"]),
            labels=dict(data.get("labels", {})),
            node_affinity=NodeAffinity.from_dict(data.get("node_affinity")),
            pod_anti_affinity=PodAntiAffinity.from_dict(data.get("pod_anti_affinity")),
            tolerations=tuple(Toleration.from_dict(t) for t in data.get("tolerations", [])),
        )


# --------------------------------------------------------------------------- #
# Scheduling conclusions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rejection:
    node: str
    gate: str               # taint | node_affinity_hard | pod_anti_affinity_hard | resource
    reason: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PluginScore:
    plugin: str
    raw_score: int                  # 0..100 integer
    weight: int                     # basis points
    contribution_mp: int            # micro-points (score * weight)
    truncation: str
    quant_error_mp: int             # worst-case quantization error in micro-points

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NodeScore:
    node: str
    plugins: Tuple[PluginScore, ...]
    total_mp: int                   # sum of contributions in micro-points
    total_points: int               # floor(total_mp / 1_000_000)
    comp_trunc_error_mp: int        # single final floor division error
    error_bound_mp: int             # propagated total error bound
    error_bound_points: int         # ceil(error_bound_mp / 1_000_000)

    def plugin_map(self) -> Dict[str, PluginScore]:
        return {p.plugin: p for p in self.plugins}

    def vector(self) -> Tuple[int, ...]:
        mapping = self.plugin_map()
        return tuple(mapping[name].raw_score for name in PLUGIN_ORDER)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "plugins": [p.to_dict() for p in self.plugins],
            "total_mp": self.total_mp,
            "total_points": self.total_points,
            "comp_trunc_error_mp": self.comp_trunc_error_mp,
            "error_bound_mp": self.error_bound_mp,
            "error_bound_points": self.error_bound_points,
        }


@dataclass(frozen=True)
class Decision:
    pod_id: str
    feasible: bool
    candidates: Tuple[str, ...]
    rejections: Tuple[Rejection, ...]
    node_scores: Tuple[NodeScore, ...]
    bound_node: Optional[str]
    tie_reason: str                 # unique | error_bound_tie | exact_tie | unschedulable
    error_influenced: bool
    runner_up: Optional[str]
    score_gap_mp: Optional[int]
    combined_error_mp: Optional[int]
    parameter_version: int

    def to_dict(self) -> dict:
        return {
            "pod_id": self.pod_id,
            "feasible": self.feasible,
            "candidates": list(self.candidates),
            "rejections": [r.to_dict() for r in self.rejections],
            "node_scores": [s.to_dict() for s in self.node_scores],
            "bound_node": self.bound_node,
            "tie_reason": self.tie_reason,
            "error_influenced": self.error_influenced,
            "runner_up": self.runner_up,
            "score_gap_mp": self.score_gap_mp,
            "combined_error_mp": self.combined_error_mp,
            "parameter_version": self.parameter_version,
        }
