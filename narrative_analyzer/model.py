"""Immutable narrative specification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DIRECTIONS = (">", ">=", "<", "<=", "==", "!=")


@dataclass(frozen=True)
class Expr:
    kind: str
    value: Optional[float] = None
    name: Optional[str] = None
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None
    coefficient_key: Optional[str] = None


@dataclass(frozen=True)
class Coefficient:
    key: str
    edge_id: str
    effect_index: int
    value: float


@dataclass(frozen=True)
class Effect:
    target: str
    mode: str
    expression: Expr
    source: str
    effect_index: int


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    choice: str
    when: tuple[str, ...]
    effects: tuple[Effect, ...]
    declaration_index: int


@dataclass(frozen=True)
class Condition:
    id: str
    component: str
    threshold: float
    direction: str


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    outcome: Optional[str]
    declaration_index: int


@dataclass(frozen=True)
class NumericSpec:
    coefficient_relative_error: float
    initial_relative_error: float
    unit_roundoff: float
    max_abs_error: float


@dataclass(frozen=True)
class StateSpec:
    components: tuple[str, ...]
    initial: tuple[float, ...]


@dataclass(frozen=True)
class Spec:
    spec_version: int
    seed: int
    start: str
    comparison: str
    boundary_epsilon: float
    path_tolerance: float
    numeric: NumericSpec
    state: StateSpec
    conditions: tuple[Condition, ...]
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    coefficients: tuple[Coefficient, ...]
    source_sha256: str

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def condition_map(self) -> dict[str, Condition]:
        return {condition.id: condition for condition in self.conditions}

    def edge_map(self) -> dict[str, Edge]:
        return {edge.id: edge for edge in self.edges}

    def outgoing(self) -> dict[str, tuple[Edge, ...]]:
        result: dict[str, list[Edge]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            result[edge.source].append(edge)
        return {node.id: tuple(result[node.id]) for node in self.nodes}

    def coefficient_map(self) -> dict[str, Coefficient]:
        return {coefficient.key: coefficient for coefficient in self.coefficients}

