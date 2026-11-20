"""Canonical data model, digests, cache keys and dependency traversal."""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .errors import EdgeError


EVICTION_RULES = frozenset({"stable_fifo", "smallest_key", "nearest_expiry"})
KEY_FIELDS = frozenset({"node_id", "path", "etag"})
DERIVATION_RULES = frozenset({"concat_v1"})


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


@dataclass(frozen=True, order=True)
class NodePolicy:
    node_id: str
    capacity: int
    eviction_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "capacity": self.capacity,
            "eviction_rule": self.eviction_rule,
        }


@dataclass(frozen=True)
class Policy:
    version: str
    key_fields: tuple[str, ...]
    nodes: tuple[NodePolicy, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "key_fields": list(self.key_fields),
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def node(self, node_id: str) -> NodePolicy:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True)
class Derivation:
    rule: str
    separator: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "separator": self.separator}


@dataclass(frozen=True)
class ContentRecord:
    path: str
    etag: str
    content: bytes
    ttl_ms: int
    derived_from: tuple[str, ...] = ()
    derivation: Derivation | None = None

    @property
    def content_sha256(self) -> str:
        return sha256_hex(self.content)

    def to_declared_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "etag": self.etag,
            "ttl_ms": self.ttl_ms,
            "content_b64": b64encode(self.content),
            "content_sha256": self.content_sha256,
            "derived_from": list(self.derived_from),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    def version_fingerprint(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "etag": self.etag,
            "content_sha256": self.content_sha256,
            "ttl_ms": self.ttl_ms,
            "derived_from": list(self.derived_from),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }


def derive_bytes(rule: str, separator: str, dependencies: Sequence[ContentRecord]) -> bytes:
    if rule == "concat_v1":
        return separator.encode("utf-8").join(dep.content for dep in dependencies)
    raise EdgeError(
        code="E_SPEC",
        message=f"unsupported derivation rule: {rule}",
        field_path="derivation.rule",
        details={"rule": rule},
    )


@dataclass(frozen=True)
class ContentSnapshot:
    records: tuple[ContentRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda item: item.path)))

    @property
    def by_path(self) -> dict[str, ContentRecord]:
        return {record.path: record for record in self.records}

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        return {record.path: record.derived_from for record in self.records}

    @property
    def dependents(self) -> dict[str, tuple[str, ...]]:
        values: dict[str, list[str]] = {record.path: [] for record in self.records}
        for record in self.records:
            for dependency in record.derived_from:
                values[dependency].append(record.path)
        return {path: tuple(sorted(items)) for path, items in sorted(values.items())}

    @property
    def etag_set(self) -> dict[str, str]:
        return {record.path: record.etag for record in self.records}

    @property
    def digest(self) -> str:
        payload = [record.version_fingerprint() for record in self.records]
        return sha256_hex(canonical_json_bytes(payload))

    @property
    def version(self) -> str:
        return "cv-" + self.digest[:16]

    def to_declared_list(self) -> list[dict[str, Any]]:
        return [record.to_declared_dict() for record in self.records]

    def graph_dict(self) -> dict[str, list[str]]:
        return {record.path: list(record.derived_from) for record in self.records}

    def topological_order(self) -> list[str]:
        indegree = {path: len(deps) for path, deps in self.dependencies.items()}
        ready = [path for path, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        dependents = self.dependents
        while ready:
            path = heapq.heappop(ready)
            ordered.append(path)
            for child in dependents.get(path, ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(ordered) != len(self.records):
            raise EdgeError(
                code="E_SPEC",
                message="derivation graph contains a cycle",
                field_path="derived_from",
                details={"cycle_members": sorted(set(indegree) - set(ordered))},
            )
        return ordered

    def descendant_order(self, trigger_path: str) -> list[str]:
        """Return every transitive dependent in stable topological order.

        The trigger itself is deliberately not included.  A heap, rather than
        an adjacency iteration order, supplies the deterministic tie-break.
        """

        if trigger_path not in self.by_path:
            raise EdgeError(
                code="E_STATE",
                message="trigger path is not present in the content snapshot",
                path=trigger_path,
                field_path="trigger_path",
            )
        dependents = self.dependents
        seen: set[str] = set()
        stack = [trigger_path]
        while stack:
            current = stack.pop()
            for child in dependents.get(current, ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        if not seen:
            return []
        indegree: dict[str, int] = {path: 0 for path in seen}
        for path in seen:
            for dependency in self.dependencies[path]:
                if dependency in seen:
                    indegree[path] += 1
        ready = [path for path, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            path = heapq.heappop(ready)
            ordered.append(path)
            for child in dependents.get(path, ()):
                if child not in seen:
                    continue
                indegree[child] -= 1
                if indegree[child] == 0:
                    heapq.heappush(ready, child)
        if len(ordered) != len(seen):
            raise EdgeError(
                code="E_SPEC",
                message="cycle detected while ordering propagation",
                field_path="derived_from",
            )
        return ordered

    def propagation_plan(self, triggers: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sequence = 0
        for trigger in sorted(set(triggers)):
            for depth_path in self.descendant_order(trigger):
                depth = self.depth_from(trigger, depth_path)
                sequence += 1
                rows.append(
                    {
                        "seq": sequence,
                        "trigger_path": trigger,
                        "affected_path": depth_path,
                        "depth": depth,
                    }
                )
        return rows

    def depth_from(self, trigger: str, target: str) -> int:
        dependents = self.dependents
        queue: list[tuple[int, str]] = [(0, trigger)]
        visited = {trigger}
        while queue:
            depth, current = heapq.heappop(queue)
            if current == target:
                return depth
            for child in dependents.get(current, ()):
                if child not in visited:
                    visited.add(child)
                    heapq.heappush(queue, (depth + 1, child))
        return -1


@dataclass(frozen=True)
class LoadedSpec:
    content: ContentSnapshot
    policy: Policy
    source_path: str | None = None

    @property
    def content_digest(self) -> str:
        return self.content.digest

    @property
    def content_version(self) -> str:
        return self.content.version

    @property
    def policy_digest(self) -> str:
        return self.policy.digest

    @property
    def policy_version(self) -> str:
        return self.policy.version

    @property
    def digest(self) -> str:
        payload = {
            "content_digest": self.content_digest,
            "policy_digest": self.policy_digest,
        }
        return sha256_hex(canonical_json_bytes(payload))

    def to_raw(self) -> dict[str, Any]:
        return {
            "schema": "edgecache.spec/v1",
            "content": self.content.to_declared_list(),
            "policy": self.policy.to_dict(),
        }


def cache_key(
    policy: Policy,
    *,
    node_id: str,
    path: str,
    etag: str,
) -> str:
    values = {"node_id": node_id, "path": path, "etag": etag}
    payload = {"fields": list(policy.key_fields), "values": {key: values[key] for key in policy.key_fields}}
    return canonical_json_bytes(payload).decode("utf-8")


def canonical_trace_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: int(row["seq"]))
    normalized = [
        {
            "seq": int(row["seq"]),
            "trigger_path": str(row["trigger_path"]),
            "affected_path": str(row["affected_path"]),
            "depth": int(row["depth"]),
        }
        for row in ordered
    ]
    return canonical_json_bytes(normalized)
