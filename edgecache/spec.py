"""Validation and typed access for deterministic edge-cache specifications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import content_digest
from .errors import SpecError


RATE_SCALE = 1_000_000


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _need_int(value: object, field: str, minimum: int | None = None) -> int:
    if not _is_int(value):
        raise SpecError(field, "must be an integer", value=repr(value))
    number = int(value)
    if minimum is not None and number < minimum:
        raise SpecError(field, f"must be >= {minimum}", value=number)
    return number


def _need_nonempty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(field, "must be a non-empty string", value=repr(value))
    return value


def _need_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(field, "must be a JSON object")
    return value


def _need_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(field, "must be a JSON array")
    return value


@dataclass(frozen=True)
class GlobalSpec:
    concurrency_limit: int
    lease_limit_per_node: int
    lease_ttl_ms: int
    socket_timeout_ms: int
    available_quota: int
    block_weight: int
    max_unknown_batch_evictions: int
    block_weight_quantum_ppm: int
    dominant_threshold_ppm: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "concurrency_limit": self.concurrency_limit,
            "lease_limit_per_node": self.lease_limit_per_node,
            "lease_ttl_ms": self.lease_ttl_ms,
            "socket_timeout_ms": self.socket_timeout_ms,
            "available_quota": self.available_quota,
            "block_weight": self.block_weight,
            "max_unknown_batch_evictions": self.max_unknown_batch_evictions,
            "block_weight_quantum_ppm": self.block_weight_quantum_ppm,
            "dominant_threshold_ppm": self.dominant_threshold_ppm,
        }


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    seed: int
    fixed_scale: int
    block_count: int
    lease_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "seed": self.seed,
            "fixed_scale": self.fixed_scale,
            "block_count": self.block_count,
            "lease_limit": self.lease_limit,
        }


@dataclass(frozen=True)
class RevisionSpec:
    revision_id: int
    node_id: str
    parent_revision: int | None
    quota: int
    fragment_rate_ppm: int
    block_rate_ppm: int

    def sampling_fields(self) -> tuple[int, int, int]:
        return self.quota, self.fragment_rate_ppm, self.block_rate_ppm

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "node_id": self.node_id,
            "parent_revision": self.parent_revision,
            "quota": self.quota,
            "fragment_rate_ppm": self.fragment_rate_ppm,
            "block_rate_ppm": self.block_rate_ppm,
        }


@dataclass(frozen=True)
class FragmentEvent:
    fragment_id: str
    node_id: str
    batch_id: str
    sequence: int
    block_count: int
    evict_blocks: int
    quota_revision: int
    corrupt_blocks: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "node_id": self.node_id,
            "batch_id": self.batch_id,
            "sequence": self.sequence,
            "block_count": self.block_count,
            "evict_blocks": self.evict_blocks,
            "quota_revision": self.quota_revision,
            "corrupt_blocks": {str(key): value for key, value in sorted(self.corrupt_blocks.items())},
        }


@dataclass(frozen=True)
class EdgeSpec:
    path: Path
    format_version: int
    global_spec: GlobalSpec
    nodes: tuple[NodeSpec, ...]
    revisions: tuple[RevisionSpec, ...]
    fragments: tuple[FragmentEvent, ...]
    raw: dict[str, Any]
    digest: str

    def node_by_id(self) -> dict[str, NodeSpec]:
        return {node.node_id: node for node in self.nodes}

    def revision_by_id(self) -> dict[int, RevisionSpec]:
        return {revision.revision_id: revision for revision in self.revisions}

    def fragment_by_id(self) -> dict[str, FragmentEvent]:
        return {fragment.fragment_id: fragment for fragment in self.fragments}

    def fragment_order_key(self, fragment: FragmentEvent) -> tuple[str, int, str]:
        return fragment.node_id, fragment.sequence, fragment.fragment_id

    def ordered_fragments(self) -> tuple[FragmentEvent, ...]:
        return tuple(sorted(self.fragments, key=self.fragment_order_key))

    def effective_revision(self, fragment: FragmentEvent) -> RevisionSpec:
        return self.revision_by_id()[fragment.quota_revision]

    def latest_revision_for_node(self) -> dict[str, RevisionSpec]:
        result: dict[str, RevisionSpec] = {}
        for revision in self.revisions:
            previous = result.get(revision.node_id)
            if previous is None or revision.revision_id > previous.revision_id:
                result[revision.node_id] = revision
        return result


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SpecError("$", "spec file does not exist", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "$",
            "spec is not valid JSON",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    except OSError as exc:
        raise SpecError("$", "spec file cannot be read", path=str(path), error=str(exc)) from exc


def load_spec(path: str | Path) -> EdgeSpec:
    source = Path(path)
    raw = _load_json(source)
    root = _need_object(raw, "$")
    format_version = _need_int(root.get("format_version"), "$.format_version", 1)
    if format_version != 1:
        raise SpecError("$.format_version", "only integer format version 1 is supported", value=format_version)

    nodes = _parse_nodes(root.get("nodes"))
    revisions = _parse_revisions(root.get("quota_revisions"), nodes)
    fragments = _parse_fragments(root.get("fragments"), nodes, revisions)
    global_spec = _parse_global(root.get("global"), nodes, fragments)
    digest = content_digest(root)
    return EdgeSpec(
        path=source,
        format_version=format_version,
        global_spec=global_spec,
        nodes=tuple(nodes),
        revisions=tuple(revisions),
        fragments=tuple(fragments),
        raw=root,
        digest=digest,
    )


def _parse_global(value: object, nodes: Iterable[NodeSpec], fragments: Iterable[FragmentEvent]) -> GlobalSpec:
    obj = _need_object(value, "$.global")
    concurrency_limit = _need_int(obj.get("concurrency_limit"), "$.global.concurrency_limit", 1)
    lease_limit_per_node = _need_int(obj.get("lease_limit_per_node"), "$.global.lease_limit_per_node", 1)
    lease_ttl_ms = _need_int(obj.get("lease_ttl_ms"), "$.global.lease_ttl_ms", 1)
    socket_timeout_ms = _need_int(obj.get("socket_timeout_ms"), "$.global.socket_timeout_ms", 1)
    available_quota = _need_int(obj.get("available_quota"), "$.global.available_quota", 0)
    block_weight = _need_int(obj.get("block_weight"), "$.global.block_weight", 1)
    max_unknown_batch_evictions = _need_int(
        obj.get("max_unknown_batch_evictions"),
        "$.global.max_unknown_batch_evictions",
        0,
    )
    block_weight_quantum_ppm = _need_int(
        obj.get("block_weight_quantum_ppm", 0),
        "$.global.block_weight_quantum_ppm",
        0,
    )
    dominant_threshold_ppm = _need_int(
        obj.get("dominant_threshold_ppm", 500_000),
        "$.global.dominant_threshold_ppm",
        0,
    )
    if dominant_threshold_ppm > RATE_SCALE:
        raise SpecError(
            "$.global.dominant_threshold_ppm",
            f"must be <= {RATE_SCALE}",
            value=dominant_threshold_ppm,
        )
    if not nodes:
        raise SpecError("$.nodes", "at least one node is required")
    if not fragments:
        raise SpecError("$.fragments", "at least one fragment event is required")
    return GlobalSpec(
        concurrency_limit=concurrency_limit,
        lease_limit_per_node=lease_limit_per_node,
        lease_ttl_ms=lease_ttl_ms,
        socket_timeout_ms=socket_timeout_ms,
        available_quota=available_quota,
        block_weight=block_weight,
        max_unknown_batch_evictions=max_unknown_batch_evictions,
        block_weight_quantum_ppm=block_weight_quantum_ppm,
        dominant_threshold_ppm=dominant_threshold_ppm,
    )


def _parse_nodes(value: object) -> list[NodeSpec]:
    entries = _need_list(value, "$.nodes")
    if not entries:
        raise SpecError("$.nodes", "must contain at least one node")
    nodes: list[NodeSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, entry in enumerate(entries):
        base = f"$.nodes[{index}]"
        obj = _need_object(entry, base)
        node_id = _need_nonempty_str(obj.get("node_id"), f"{base}.node_id")
        if node_id in seen_ids:
            raise SpecError(f"{base}.node_id", "node identifier must be unique", duplicate=node_id)
        seen_ids.add(node_id)
        host = obj.get("host")
        if host != "127.0.0.1":
            raise SpecError(f"{base}.host", "must be the literal 127.0.0.1", value=repr(host))
        port = _need_int(obj.get("port"), f"{base}.port", 1)
        if port > 65535:
            raise SpecError(f"{base}.port", "must be <= 65535", value=port)
        if port in seen_ports:
            raise SpecError(f"{base}.port", "loopback port must be globally unique", duplicate=port)
        seen_ports.add(port)
        seed = _need_int(obj.get("seed"), f"{base}.seed", 0)
        fixed_scale = _need_int(obj.get("fixed_scale"), f"{base}.fixed_scale", 1)
        block_count = _need_int(obj.get("block_count"), f"{base}.block_count", 1)
        lease_limit = _need_int(obj.get("lease_limit"), f"{base}.lease_limit", 1)
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=port,
                seed=seed,
                fixed_scale=fixed_scale,
                block_count=block_count,
                lease_limit=lease_limit,
            )
        )
    return nodes


def _parse_revisions(value: object, nodes: Iterable[NodeSpec]) -> list[RevisionSpec]:
    entries = _need_list(value, "$.quota_revisions")
    if not entries:
        raise SpecError("$.quota_revisions", "must contain at least one revision")
    node_ids = {node.node_id for node in nodes}
    revisions: list[RevisionSpec] = []
    known: dict[int, RevisionSpec] = {}
    previous_version: int | None = None
    for index, entry in enumerate(entries):
        base = f"$.quota_revisions[{index}]"
        obj = _need_object(entry, base)
        revision_id = _need_int(obj.get("revision_id"), f"{base}.revision_id", 0)
        if revision_id in known:
            raise SpecError(f"{base}.revision_id", "revision version must be unique", duplicate=revision_id)
        if previous_version is not None and revision_id <= previous_version:
            raise SpecError(
                f"{base}.revision_id",
                "revision version must be strictly monotonic in chain order",
                previous=previous_version,
                value=revision_id,
            )
        previous_version = revision_id
        node_id = _need_nonempty_str(obj.get("node_id"), f"{base}.node_id")
        if node_id not in node_ids:
            raise SpecError(f"{base}.node_id", "revision references an unknown node", value=node_id)
        parent_value = obj.get("parent_revision")
        if parent_value is None:
            parent_revision = None
        else:
            parent_revision = _need_int(parent_value, f"{base}.parent_revision", 0)
            if parent_revision not in known:
                raise SpecError(
                    f"{base}.parent_revision",
                    "parent revision must already exist",
                    value=parent_revision,
                )
            if known[parent_revision].node_id != node_id:
                raise SpecError(
                    f"{base}.parent_revision",
                    "parent revision belongs to a different node",
                    parent_node=known[parent_revision].node_id,
                )
        quota = _need_int(obj.get("quota"), f"{base}.quota", 0)
        fragment_rate_ppm = _need_int(obj.get("fragment_rate_ppm"), f"{base}.fragment_rate_ppm", 0)
        block_rate_ppm = _need_int(obj.get("block_rate_ppm"), f"{base}.block_rate_ppm", 0)
        revision = RevisionSpec(
            revision_id=revision_id,
            node_id=node_id,
            parent_revision=parent_revision,
            quota=quota,
            fragment_rate_ppm=fragment_rate_ppm,
            block_rate_ppm=block_rate_ppm,
        )
        revisions.append(revision)
        known[revision_id] = revision
    return revisions


def _parse_fragments(
    value: object,
    nodes: Iterable[NodeSpec],
    revisions: Iterable[RevisionSpec],
) -> list[FragmentEvent]:
    entries = _need_list(value, "$.fragments")
    if not entries:
        raise SpecError("$.fragments", "must contain at least one fragment event")
    node_ids = {node.node_id for node in nodes}
    revision_by_id = {revision.revision_id: revision for revision in revisions}
    fragments: list[FragmentEvent] = []
    seen_fragment_ids: set[str] = set()
    seen_batch_ids: set[str] = set()
    seen_sequences: dict[str, set[int]] = {}
    for index, entry in enumerate(entries):
        base = f"$.fragments[{index}]"
        obj = _need_object(entry, base)
        fragment_id = _need_nonempty_str(obj.get("fragment_id"), f"{base}.fragment_id")
        if fragment_id in seen_fragment_ids:
            raise SpecError(
                f"{base}.fragment_id",
                "fragment identifier must be unique",
                duplicate=fragment_id,
            )
        seen_fragment_ids.add(fragment_id)
        node_id = _need_nonempty_str(obj.get("node_id"), f"{base}.node_id")
        if node_id not in node_ids:
            raise SpecError(f"{base}.node_id", "fragment references an unknown node", value=node_id)
        batch_id = _need_nonempty_str(obj.get("batch_id"), f"{base}.batch_id")
        if batch_id in seen_batch_ids:
            raise SpecError(f"{base}.batch_id", "batch identifier must be unique", duplicate=batch_id)
        seen_batch_ids.add(batch_id)
        sequence = _need_int(obj.get("sequence"), f"{base}.sequence", 0)
        node_sequences = seen_sequences.setdefault(node_id, set())
        if sequence in node_sequences:
            raise SpecError(
                f"{base}.sequence",
                "fragment sequence must be unique within a node",
                duplicate=sequence,
            )
        node_sequences.add(sequence)
        block_count = _need_int(obj.get("block_count"), f"{base}.block_count", 1)
        evict_blocks = _need_int(obj.get("evict_blocks"), f"{base}.evict_blocks", 0)
        if evict_blocks > block_count:
            raise SpecError(
                f"{base}.evict_blocks",
                "cannot exceed block_count",
                value=evict_blocks,
                block_count=block_count,
            )
        quota_revision = _need_int(obj.get("quota_revision"), f"{base}.quota_revision", 0)
        if quota_revision not in revision_by_id:
            raise SpecError(
                f"{base}.quota_revision",
                "fragment references an unknown quota revision",
                value=quota_revision,
            )
        if revision_by_id[quota_revision].node_id != node_id:
            raise SpecError(
                f"{base}.quota_revision",
                "quota revision belongs to a different node",
                revision_node=revision_by_id[quota_revision].node_id,
            )
        corrupt_value = obj.get("corrupt_blocks", {})
        if not isinstance(corrupt_value, dict):
            raise SpecError(f"{base}.corrupt_blocks", "must be a JSON object")
        corrupt_blocks: dict[int, int] = {}
        for raw_key, raw_delta in corrupt_value.items():
            try:
                block_index = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise SpecError(f"{base}.corrupt_blocks", "block key must be an integer") from exc
            if block_index < 0 or block_index >= block_count:
                raise SpecError(
                    f"{base}.corrupt_blocks",
                    "block key is outside the fragment",
                    key=raw_key,
                    block_count=block_count,
                )
            corrupt_blocks[block_index] = _need_int(
                raw_delta,
                f"{base}.corrupt_blocks[{raw_key}]",
            )
        fragments.append(
            FragmentEvent(
                fragment_id=fragment_id,
                node_id=node_id,
                batch_id=batch_id,
                sequence=sequence,
                block_count=block_count,
                evict_blocks=evict_blocks,
                quota_revision=quota_revision,
                corrupt_blocks=corrupt_blocks,
            )
        )
    return fragments

