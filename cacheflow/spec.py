from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes, sha256_hex
from .errors import SpecError


EVENT_TYPES = {
    "grant",
    "write",
    "read",
    "renew",
    "release",
    "expire",
    "quota_bump",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read spec: {exc}", entry=str(path), field="$") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
            entry=str(path),
            field=f"$.line-{exc.lineno}",
        ) from exc
    if not isinstance(value, dict):
        raise SpecError("top-level JSON value must be an object", entry=str(path), field="$")
    value["__path__"] = str(path)
    return value


def _require_object(value: Any, entry: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError("value must be an object", entry=entry, field=field)
    return value


def _require_str(obj: dict[str, Any], key: str, entry: Any, *, nonempty: bool = True) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or (nonempty and not value):
        raise SpecError(f"{key} must be a non-empty string", entry=entry, field=key)
    return value


def _require_positive_int(obj: dict[str, Any], key: str, entry: Any) -> int:
    value = obj.get(key)
    if not _is_int(value) or value <= 0:
        raise SpecError(f"{key} must be a positive integer", entry=entry, field=key)
    return value


def _optional_positive_int(obj: dict[str, Any], key: str, entry: Any, default: int) -> int:
    if key not in obj:
        return default
    return _require_positive_int(obj, key, entry)


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    host: str
    port: int
    capacity_slots: int
    max_leases: int
    seed: str


@dataclass(frozen=True)
class ShardSpec:
    index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_key: str
    total_bytes: int
    shards: tuple[ShardSpec, ...]

    def shard(self, index: int) -> ShardSpec:
        for shard in self.shards:
            if shard.index == index:
                return shard
        raise KeyError(index)


@dataclass(frozen=True)
class Bundle:
    nodes_doc: dict[str, Any]
    artifacts_doc: dict[str, Any]
    events_doc: dict[str, Any] | None
    nodes: tuple[NodeSpec, ...]
    artifacts: tuple[ArtifactSpec, ...]
    events: tuple[dict[str, Any], ...]
    nodes_path: str | None = None
    artifacts_path: str | None = None
    events_path: str | None = None

    @property
    def quota_version(self) -> int:
        return int(self.nodes_doc["quota_version"])

    @property
    def global_concurrency(self) -> int:
        return int(self.nodes_doc["global_concurrency"])

    def node(self, node_id: str) -> NodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def artifact(self, artifact_key: str) -> ArtifactSpec:
        for artifact in self.artifacts:
            if artifact.artifact_key == artifact_key:
                return artifact
        raise KeyError(artifact_key)

    def spec_digest(self) -> str:
        public = {
            "nodes": {k: v for k, v in self.nodes_doc.items() if not k.startswith("__")},
            "artifacts": {k: v for k, v in self.artifacts_doc.items() if not k.startswith("__")},
            "events": None
            if self.events_doc is None
            else {k: v for k, v in self.events_doc.items() if not k.startswith("__")},
        }
        return sha256_hex(canonical_bytes(public))


def load_nodes_spec(path: str | Path) -> tuple[dict[str, Any], tuple[NodeSpec, ...]]:
    doc = _load_json(path)
    entry = str(path)
    if "quota_version" in doc:
        _require_positive_int(doc, "quota_version", entry)
    else:
        raise SpecError("quota_version is required", entry=entry, field="quota_version")
    if "global_concurrency" in doc:
        _require_positive_int(doc, "global_concurrency", entry)
    else:
        raise SpecError("global_concurrency is required", entry=entry, field="global_concurrency")

    raw_nodes = doc.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SpecError("nodes must be a non-empty array", entry=entry, field="nodes")

    nodes: list[NodeSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[tuple[str, int]] = set()
    for position, raw_node in enumerate(raw_nodes):
        node_entry = f"nodes[{position}]"
        node = _require_object(raw_node, node_entry, "nodes")
        node_id = _require_str(node, "node_id", node_entry)
        if node_id in seen_ids:
            raise SpecError("duplicate node_id", entry=node_entry, field="node_id")
        seen_ids.add(node_id)

        host = _require_str(node, "host", node_entry)
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", entry=node_entry, field="host")

        port = node.get("port")
        if not _is_int(port) or not (1 <= int(port) <= 65535):
            raise SpecError("port must be an integer in 1..65535", entry=node_entry, field="port")
        port_key = (host, int(port))
        if port_key in seen_ports:
            raise SpecError("duplicate loopback port", entry=node_entry, field="port")
        seen_ports.add(port_key)

        capacity_slots = _require_positive_int(node, "capacity_slots", node_entry)
        max_leases = _require_positive_int(node, "max_leases", node_entry)
        seed = _require_str(node, "seed", node_entry)
        nodes.append(
            NodeSpec(
                node_id=node_id,
                host=host,
                port=int(port),
                capacity_slots=capacity_slots,
                max_leases=max_leases,
                seed=seed,
            )
        )
    return doc, tuple(sorted(nodes, key=lambda item: item.node_id))


def load_artifacts_spec(path: str | Path) -> tuple[dict[str, Any], tuple[ArtifactSpec, ...]]:
    doc = _load_json(path)
    entry = str(path)
    raw_artifacts = doc.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise SpecError("artifacts must be a non-empty array", entry=entry, field="artifacts")

    artifacts: list[ArtifactSpec] = []
    seen_keys: set[str] = set()
    for position, raw_artifact in enumerate(raw_artifacts):
        artifact_entry = f"artifacts[{position}]"
        artifact = _require_object(raw_artifact, artifact_entry, "artifacts")
        key = _require_str(artifact, "artifact_key", artifact_entry)
        if key in seen_keys:
            raise SpecError("duplicate artifact_key", entry=artifact_entry, field="artifact_key")
        seen_keys.add(key)

        total_bytes = _require_positive_int(artifact, "total_bytes", artifact_entry)
        raw_shards = artifact.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise SpecError("shards must be a non-empty array", entry=artifact_entry, field="shards")

        shards: list[ShardSpec] = []
        seen_indexes: set[int] = set()
        for shard_position, raw_shard in enumerate(raw_shards):
            shard_entry = f"artifacts[{position}].shards[{shard_position}]"
            shard = _require_object(raw_shard, shard_entry, "shards")
            raw_index = shard.get("index")
            if not _is_int(raw_index) or int(raw_index) < 0:
                raise SpecError("shard index must be non-negative", entry=shard_entry, field="index")
            index = int(raw_index)
            if index in seen_indexes:
                raise SpecError("duplicate shard index", entry=shard_entry, field="index")
            seen_indexes.add(index)
            start = shard.get("start")
            end = shard.get("end")
            if not _is_int(start) or int(start) < 0:
                raise SpecError("start must be a non-negative integer", entry=shard_entry, field="start")
            if not _is_int(end) or int(end) <= int(start):
                raise SpecError("end must be an integer greater than start", entry=shard_entry, field="end")
            if int(end) > total_bytes:
                raise SpecError("shard range exceeds total_bytes", entry=shard_entry, field="end")
            shards.append(ShardSpec(index=index, start=int(start), end=int(end)))

        ordered = sorted(shards, key=lambda item: item.index)
        if [shard.index for shard in ordered] != list(range(len(ordered))):
            raise SpecError("shard indexes must start at 0 and be consecutive", entry=artifact_entry, field="shards")
        cursor = 0
        for shard in ordered:
            if shard.start != cursor:
                field = "start" if shard.start != cursor else "end"
                raise SpecError(
                    "shard ranges must be head-to-tail contiguous from byte 0",
                    entry=f"{artifact_entry}.shards[{shard.index}]",
                    field=field,
                )
            cursor = shard.end
        if cursor != total_bytes:
            raise SpecError("final shard end must equal total_bytes", entry=artifact_entry, field="total_bytes")
        artifacts.append(ArtifactSpec(artifact_key=key, total_bytes=total_bytes, shards=tuple(ordered)))
    return doc, tuple(sorted(artifacts, key=lambda item: item.artifact_key))


def load_events_spec(
    path: str | Path | None,
    *,
    nodes: Iterable[NodeSpec] | None = None,
    artifacts: Iterable[ArtifactSpec] | None = None,
    quota_version: int | None = None,
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    if path is None:
        return None, ()
    doc = _load_json(path)
    raw_events = doc.get("events")
    entry = str(path)
    if not isinstance(raw_events, list):
        raise SpecError("events must be an array", entry=entry, field="events")

    node_ids = {node.node_id for node in (nodes or [])}
    artifact_keys = {artifact.artifact_key for artifact in (artifacts or [])}
    events: list[dict[str, Any]] = []
    last_seq = 0
    grant_leases: dict[str, int] = {}
    known_leases: set[str] = set()

    for position, raw_event in enumerate(raw_events):
        event_entry = f"events[{position}]"
        event = _require_object(raw_event, event_entry, "events")
        seq = event.get("seq")
        if not _is_int(seq) or int(seq) <= 0:
            raise SpecError("seq must be a positive integer", entry=event_entry, field="seq")
        if int(seq) <= last_seq:
            raise SpecError("seq must be strictly increasing and unique", entry=event_entry, field="seq")
        last_seq = int(seq)

        event_type = _require_str(event, "type", event_entry)
        if event_type not in EVENT_TYPES:
            raise SpecError(f"unknown event type: {event_type}", entry=event_entry, field="type")

        lease_id: str | None = None
        if event_type != "quota_bump":
            lease_id = _require_str(event, "lease_id", event_entry)

        if event_type == "grant":
            artifact_key = _require_str(event, "artifact_key", event_entry)
            if artifact_keys and artifact_key not in artifact_keys:
                raise SpecError("artifact_key is not declared", entry=event_entry, field="artifact_key")
            if lease_id in grant_leases:
                raise SpecError("lease_id has more than one grant", entry=event_entry, field="lease_id")
            grant_leases[lease_id] = int(seq)
            known_leases.add(lease_id)
            if "node_id" in event and event["node_id"] is not None:
                node_id = _require_str(event, "node_id", event_entry)
                if node_ids and node_id not in node_ids:
                    raise SpecError("node_id is not declared", entry=event_entry, field="node_id")
            _optional_positive_int(event, "ttl_ms", event_entry, 600_000)
            qv = _optional_positive_int(event, "quota_version", event_entry, int(quota_version or 1))
            event["quota_version"] = qv
        elif event_type in {"write", "read"}:
            if lease_id not in known_leases:
                raise SpecError("event references lease before its grant", entry=event_entry, field="lease_id")
            if "shard_index" in event and (not _is_int(event["shard_index"]) or int(event["shard_index"]) < 0):
                raise SpecError("shard_index must be a non-negative integer", entry=event_entry, field="shard_index")
            if "generation" in event:
                _require_positive_int(event, "generation", event_entry)
        elif event_type in {"renew", "release", "expire"}:
            if lease_id not in known_leases:
                raise SpecError("event references lease before its grant", entry=event_entry, field="lease_id")
            _require_positive_int(event, "generation", event_entry)
            if event_type == "renew":
                _optional_positive_int(event, "ttl_ms", event_entry, 600_000)
        elif event_type == "quota_bump":
            new_version = _require_positive_int(event, "quota_version", event_entry)
            if quota_version is not None and new_version <= quota_version and position == 0:
                raise SpecError("first quota_bump must exceed configured quota_version", entry=event_entry, field="quota_version")

        if "node_id" in event and event["node_id"] is not None and event_type != "grant":
            node_id = _require_str(event, "node_id", event_entry)
            if node_ids and node_id not in node_ids:
                raise SpecError("node_id is not declared", entry=event_entry, field="node_id")

        events.append(dict(event))
    return doc, tuple(events)


def load_bundle(
    nodes_path: str | Path,
    artifacts_path: str | Path,
    events_path: str | Path | None = None,
) -> Bundle:
    nodes_doc, nodes = load_nodes_spec(nodes_path)
    artifacts_doc, artifacts = load_artifacts_spec(artifacts_path)
    events_doc, events = load_events_spec(
        events_path,
        nodes=nodes,
        artifacts=artifacts,
        quota_version=int(nodes_doc["quota_version"]),
    )
    return Bundle(
        nodes_doc=nodes_doc,
        artifacts_doc=artifacts_doc,
        events_doc=events_doc,
        nodes=nodes,
        artifacts=artifacts,
        events=events,
        nodes_path=str(nodes_path),
        artifacts_path=str(artifacts_path),
        events_path=None if events_path is None else str(events_path),
    )


def iter_json_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.json") if path.is_file())
