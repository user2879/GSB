"""Immutable narrative specification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, double, sha256_json


@dataclass(frozen=True)
class Condition:
    id: str
    component: str
    op: str
    threshold: float
    threshold_key: str
    declaration_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "component": self.component,
            "op": self.op,
            "threshold": double(self.threshold),
            "threshold_key": self.threshold_key,
            "declaration_index": self.declaration_index,
        }


@dataclass(frozen=True)
class Effect:
    id: str
    target: str
    expression: str
    ast: tuple[Any, ...]
    coefficient_keys: tuple[str, ...]
    declaration_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "expression": self.expression,
            "coefficient_keys": list(self.coefficient_keys),
            "declaration_index": self.declaration_index,
        }


@dataclass(frozen=True)
class Edge:
    index: int
    id: str
    source: str
    target: str
    label: str
    effects: tuple[Effect, ...]
    conditions: tuple[Condition, ...]

    @property
    def coefficient_keys(self) -> tuple[str, ...]:
        return tuple(key for effect in self.effects for key in effect.coefficient_keys)

    @property
    def threshold_keys(self) -> tuple[str, ...]:
        return tuple(condition.threshold_key for condition in self.conditions)

    @property
    def trigger_keys(self) -> tuple[str, ...]:
        return (f"effect_set:{self.id}", *self.coefficient_keys, *self.threshold_keys)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "from": self.source,
            "to": self.target,
            "label": self.label,
            "effects": [effect.as_dict() for effect in self.effects],
            "conditions": [condition.as_dict() for condition in self.conditions],
            "trigger_keys": list(self.trigger_keys),
        }


@dataclass(frozen=True)
class Node:
    index: int
    id: str
    kind: str
    title: str
    ending: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "ending": self.ending,
        }


@dataclass(frozen=True)
class SequenceSample:
    id: str
    description: str
    edges: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description, "edges": list(self.edges)}


@dataclass(frozen=True)
class NarrativeSpec:
    schema: str
    id: str
    title: str
    seed: int
    components: tuple[str, ...]
    initial_state: tuple[float, ...]
    boundary_tolerance: float
    merge_tolerance: float
    start: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    samples: tuple[SequenceSample, ...]
    source_name: str = ""

    def node_by_id(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def edge_by_id(self, edge_id: str) -> Edge:
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        raise KeyError(edge_id)

    def outgoing(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if edge.source == node_id)

    def initial_map(self) -> dict[str, float]:
        return {component: self.initial_state[index] for index, component in enumerate(self.components)}

    def sample_by_id(self, sample_id: str) -> SequenceSample:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        raise KeyError(sample_id)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "id": self.id,
            "title": self.title,
            "seed": self.seed,
            "components": list(self.components),
            "initial_state": {
                component: double(self.initial_state[index])
                for index, component in enumerate(self.components)
            },
            "boundary_tolerance": double(self.boundary_tolerance),
            "merge_tolerance": double(self.merge_tolerance),
            "start": self.start,
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "samples": [sample.as_dict() for sample in self.samples],
        }

    @property
    def spec_hash(self) -> str:
        return sha256_json(self.canonical_dict())

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_dict())

    def with_initial_state(self, values: Sequence[float]) -> "NarrativeSpec":
        return NarrativeSpec(
            schema=self.schema,
            id=self.id,
            title=self.title,
            seed=self.seed,
            components=self.components,
            initial_state=tuple(double(value) for value in values),
            boundary_tolerance=self.boundary_tolerance,
            merge_tolerance=self.merge_tolerance,
            start=self.start,
            nodes=self.nodes,
            edges=self.edges,
            samples=self.samples,
            source_name=self.source_name,
        )

    def with_edges(self, edges: Sequence[Edge]) -> "NarrativeSpec":
        return NarrativeSpec(
            schema=self.schema,
            id=self.id,
            title=self.title,
            seed=self.seed,
            components=self.components,
            initial_state=self.initial_state,
            boundary_tolerance=self.boundary_tolerance,
            merge_tolerance=self.merge_tolerance,
            start=self.start,
            nodes=self.nodes,
            edges=tuple(edges),
            samples=self.samples,
            source_name=self.source_name,
        )

    def component_index(self, component: str) -> int:
        for index, candidate in enumerate(self.components):
            if candidate == component:
                return index
        raise KeyError(component)

    def initial_payload(self) -> dict[str, float]:
        return {
            component: double(self.initial_state[index])
            for index, component in enumerate(self.components)
        }

    def declared_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for component in self.components:
            keys.append(f"initial:{component}:value")
        for edge in self.edges:
            keys.extend(edge.trigger_keys)
        return tuple(keys)


def effects_from_ids(spec: NarrativeSpec, edge_ids: Iterable[str]) -> tuple[tuple[Edge, Effect], ...]:
    pairs: list[tuple[Edge, Effect]] = []
    for edge_id in edge_ids:
        edge = spec.edge_by_id(edge_id)
        for effect in edge.effects:
            pairs.append((edge, effect))
    return tuple(pairs)


def state_mapping(components: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {component: double(values[index]) for index, component in enumerate(components)}


def state_tuple(components: Sequence[str], mapping: Mapping[str, float]) -> tuple[float, ...]:
    return tuple(double(mapping[component]) for component in components)
