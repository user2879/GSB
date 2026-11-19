"""Dependency graph for primitives, slices, frames, and digest nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import stable_hash
from .constants import CHANNEL_INDEX
from .frames import BatchDigest, FrameRecord
from .model import PARAMETER_KEYS, EffectGraph

NODE_PRIMITIVE = "primitive"
NODE_SLICE = "slice"
NODE_FRAME = "frame"
NODE_DIGEST = "digest"
NODE_PROTOCOL = "protocol"

PROTOCOL_TRIGGER = "protocol.version"
CONTENT_HASH_TRIGGER = "content_hash"


@dataclass(frozen=True)
class DependencyNode:
    key: str
    kind: str
    version_stamp: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "version_stamp": self.version_stamp,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyNode":
        return cls(
            key=str(value["key"]),
            kind=str(value["kind"]),
            version_stamp=str(value["version_stamp"]),
            content_hash=str(value["content_hash"]),
        )


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    trigger_keys: tuple[str, ...]
    channels: tuple[str, ...] = ()
    protocol_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "trigger_keys": list(self.trigger_keys),
            "channels": list(self.channels),
            "protocol_key": self.protocol_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyEdge":
        return cls(
            source=str(value["source"]),
            target=str(value["target"]),
            trigger_keys=tuple(value["trigger_keys"]),
            channels=tuple(value.get("channels", [])),
            protocol_key=(
                None if value.get("protocol_key") is None else str(value["protocol_key"])
            ),
        )


@dataclass(frozen=True)
class DependencyGraph:
    nodes: tuple[DependencyNode, ...]
    edges: tuple[DependencyEdge, ...]
    protocol_version: int

    @property
    def node_by_key(self) -> dict[str, DependencyNode]:
        return {node.key: node for node in self.nodes}

    @property
    def outgoing(self) -> dict[str, list[DependencyEdge]]:
        result: dict[str, list[DependencyEdge]] = {}
        for edge in self.edges:
            result.setdefault(edge.source, []).append(edge)
        return result

    def downstream_closure(
        self,
        seed_keys: Iterable[str],
        changed_triggers: Iterable[str],
        include_protocol_switch: bool = False,
    ) -> tuple[str, ...]:
        active_triggers = set(changed_triggers)
        if include_protocol_switch:
            active_triggers.add(PROTOCOL_TRIGGER)
        visited: set[str] = set()
        queue = list(dict.fromkeys(seed_keys))
        outgoing = self.outgoing
        while queue:
            source = queue.pop(0)
            if source in visited:
                continue
            visited.add(source)
            source_node = self.node_by_key.get(source)
            if source_node is not None and source_node.kind == NODE_FRAME:
                active_triggers.add(CONTENT_HASH_TRIGGER)
            for edge in outgoing.get(source, []):
                if edge.protocol_key == PROTOCOL_TRIGGER and PROTOCOL_TRIGGER in active_triggers:
                    matched = True
                elif PROTOCOL_TRIGGER in edge.trigger_keys and PROTOCOL_TRIGGER in active_triggers:
                    matched = True
                else:
                    matched = bool(set(edge.trigger_keys) & active_triggers)
                if matched and edge.target not in visited:
                    queue.append(edge.target)
        return tuple(sorted(visited))

    def closure_excluding_roots(
        self,
        seed_keys: Iterable[str],
        changed_triggers: Iterable[str],
        include_protocol_switch: bool = False,
    ) -> tuple[str, ...]:
        seeds = set(seed_keys)
        return tuple(
            key
            for key in self.downstream_closure(
                seed_keys,
                changed_triggers,
                include_protocol_switch=include_protocol_switch,
            )
            if key not in seeds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencyGraph":
        return cls(
            protocol_version=int(value["protocol_version"]),
            nodes=tuple(DependencyNode.from_dict(item) for item in value["nodes"]),
            edges=tuple(DependencyEdge.from_dict(item) for item in value["edges"]),
        )


def _node_for_primitive(primitive: Any) -> DependencyNode:
    content_hash = stable_hash(primitive.to_dict())
    return DependencyNode(
        key=f"primitive:{primitive.id}",
        kind=NODE_PRIMITIVE,
        version_stamp=f"src:{content_hash[:16]}",
        content_hash=content_hash,
    )


def _node_for_slice(time_slice: Any) -> DependencyNode:
    content_hash = stable_hash(time_slice.to_dict())
    return DependencyNode(
        key=f"slice:{time_slice.id}",
        kind=NODE_SLICE,
        version_stamp=f"src:{content_hash[:16]}",
        content_hash=content_hash,
    )


def build_dependency_graph(
    graph: EffectGraph,
    frames: Iterable[FrameRecord],
    digests: Iterable[BatchDigest],
    protocol_version: int,
) -> DependencyGraph:
    ordered_frames = sorted(frames, key=lambda frame: frame.seq)
    ordered_digests = sorted(digests, key=lambda digest: digest.first_seq)
    primitive_by_id = graph.primitive_by_id
    channel_members = {
        time_slice.id: tuple(
            sorted(
                {channel for primitive_id in time_slice.primitive_ids for channel in primitive_by_id[primitive_id].channels},
                key=CHANNEL_INDEX.__getitem__,
            )
        )
        for time_slice in graph.slices
    }
    nodes: list[DependencyNode] = [
        DependencyNode(
            key="protocol:current",
            kind=NODE_PROTOCOL,
            version_stamp=f"protocol:v{protocol_version}",
            content_hash=stable_hash({"protocol_version": protocol_version}),
        )
    ]
    nodes.extend(_node_for_primitive(primitive) for primitive in graph.primitives)
    nodes.extend(_node_for_slice(time_slice) for time_slice in graph.slices)
    nodes.extend(
        DependencyNode(
            key=frame.frame_key,
            kind=NODE_FRAME,
            version_stamp=frame.version_stamp,
            content_hash=frame.content_hash,
        )
        for frame in ordered_frames
    )
    nodes.extend(
        DependencyNode(
            key=digest.batch_key,
            kind=NODE_DIGEST,
            version_stamp=f"v{digest.protocol_version}:{digest.digest[:16]}",
            content_hash=digest.digest,
        )
        for digest in ordered_digests
    )

    edges: list[DependencyEdge] = []
    for time_slice in graph.slices:
        for primitive_id in time_slice.primitive_ids:
            primitive = primitive_by_id[primitive_id]
            edges.append(
                DependencyEdge(
                    source=f"primitive:{primitive_id}",
                    target=f"slice:{time_slice.id}",
                    trigger_keys=PARAMETER_KEYS,
                    channels=primitive.channels,
                )
            )
        for frame in ordered_frames:
            if time_slice.overlaps_frame(frame.timestamp_us, graph.frame_period_us):
                edges.append(
                    DependencyEdge(
                        source=f"slice:{time_slice.id}",
                        target=frame.frame_key,
                        trigger_keys=PARAMETER_KEYS,
                        channels=channel_members[time_slice.id],
                    )
                )

    for primitive in graph.primitives:
        primitive_frame_start = max(0, primitive.start_us // graph.frame_period_us)
        primitive_frame_end = min(
            graph.frame_count,
            (primitive.end_us + graph.frame_period_us - 1) // graph.frame_period_us,
        )
        for frame in ordered_frames[primitive_frame_start:primitive_frame_end]:
            edges.append(
                DependencyEdge(
                    source=f"primitive:{primitive.id}",
                    target=frame.frame_key,
                    trigger_keys=PARAMETER_KEYS,
                    channels=primitive.channels,
                )
            )

    for frame in ordered_frames:
        edges.append(
            DependencyEdge(
                source="protocol:current",
                target=frame.frame_key,
                trigger_keys=(PROTOCOL_TRIGGER,),
                protocol_key=PROTOCOL_TRIGGER,
            )
        )

    seq_to_batch: dict[int, str] = {}
    for digest in ordered_digests:
        for seq in range(digest.first_seq, digest.last_seq + 1):
            seq_to_batch[seq] = digest.batch_key
        edges.append(
            DependencyEdge(
                source="protocol:current",
                target=digest.batch_key,
                trigger_keys=(PROTOCOL_TRIGGER,),
                protocol_key=PROTOCOL_TRIGGER,
            )
        )
    for frame in ordered_frames:
        batch_key = seq_to_batch.get(frame.seq)
        if batch_key is None:
            raise ValueError(f"frame {frame.seq} has no batch digest")
        edges.append(
            DependencyEdge(
                source=frame.frame_key,
                target=batch_key,
                trigger_keys=(CONTENT_HASH_TRIGGER,),
            )
        )
    return DependencyGraph(
        nodes=tuple(sorted(nodes, key=lambda node: node.key)),
        edges=tuple(
            sorted(
                edges,
                key=lambda edge: (edge.source, edge.target, edge.trigger_keys, edge.channels),
            )
        ),
        protocol_version=protocol_version,
    )

