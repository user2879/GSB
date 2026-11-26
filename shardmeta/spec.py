"""Loading and strict validation of local JSON declarations (requirement 1).

Failures always raise :class:`SpecError` (E_SPEC) naming the offending node id
or object key and the specific field. No network or account access happens.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .errors import SpecError
from .models import NodeSpec, ObjectManifest, Request, ShardPlacement

VALID_OPS = frozenset({"grant", "upload", "renew", "abort", "read"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise SpecError(f"file not found: {path}", field="path")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc.msg}", field="json", line=exc.lineno) from exc


def load_node_specs(path: str) -> dict[str, NodeSpec]:
    data = load_json(path)
    if not isinstance(data, dict) or "nodes" not in data:
        raise SpecError("node declaration must be an object with a 'nodes' array", field="nodes")
    raw_nodes = data["nodes"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise SpecError("'nodes' must be a non-empty array", field="nodes")

    specs: dict[str, NodeSpec] = {}
    seen_ports: dict[int, str] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise SpecError("each node must be an object", field=f"nodes[{index}]")
        node_id = raw.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise SpecError("node_id missing or not a non-empty string", node=str(node_id),
                            field="node_id")
        if node_id in specs:
            raise SpecError("duplicate node_id", node=node_id, field="node_id")

        host = raw.get("host")
        if host != "127.0.0.1":
            raise SpecError("host must be the literal 127.0.0.1", node=node_id, field="host")
        port = raw.get("port")
        if not _is_positive_int(port) or not (1024 <= port <= 65535):
            raise SpecError("port must be a positive integer in [1024,65535]",
                            node=node_id, field="port")
        if port in seen_ports:
            raise SpecError(
                f"port {port} already used by node {seen_ports[port]}",
                node=node_id, field="port",
            )

        for field_name in ("capacity_slots", "lease_limit"):
            if not _is_positive_int(raw.get(field_name)):
                raise SpecError(f"{field_name} must be a positive integer",
                                node=node_id, field=field_name)
        for field_name in ("seed", "meta_version"):
            if not _is_nonneg_int(raw.get(field_name)):
                raise SpecError(f"{field_name} must be a non-negative integer",
                                node=node_id, field=field_name)

        spec = NodeSpec(
            node_id=node_id,
            host=host,
            port=port,
            capacity_slots=int(raw["capacity_slots"]),
            lease_limit=int(raw["lease_limit"]),
            seed=int(raw["seed"]),
            meta_version=int(raw["meta_version"]),
        )
        specs[node_id] = spec
        seen_ports[port] = node_id
    return specs


def load_object_manifests(path: str, nodes: dict[str, NodeSpec]) -> dict[str, ObjectManifest]:
    data = load_json(path)
    if not isinstance(data, dict) or "objects" not in data:
        raise SpecError("manifest must be an object with an 'objects' array", field="objects")
    raw_objects = data["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise SpecError("'objects' must be a non-empty array", field="objects")

    manifests: dict[str, ObjectManifest] = {}
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict):
            raise SpecError("each object must be an object", field=f"objects[{index}]")
        key = raw.get("object_key")
        if not isinstance(key, str) or not key:
            raise SpecError("object_key missing or not a non-empty string",
                            key=str(key), field="object_key")
        if key in manifests:
            raise SpecError("duplicate object_key", key=key, field="object_key")

        total_bytes = raw.get("total_bytes")
        shard_size = raw.get("shard_size")
        if not _is_positive_int(total_bytes):
            raise SpecError("total_bytes must be a positive integer", key=key,
                            field="total_bytes")
        if not _is_positive_int(shard_size):
            raise SpecError("shard_size must be a positive integer", key=key,
                            field="shard_size")
        if total_bytes % shard_size != 0:
            raise SpecError(
                f"shard_size {shard_size} does not divide total_bytes {total_bytes}",
                key=key, field="shard_size",
            )
        if not _is_nonneg_int(raw.get("content_seed")):
            raise SpecError("content_seed must be a non-negative integer",
                            key=key, field="content_seed")

        placements_raw = raw.get("placements")
        if not isinstance(placements_raw, list) or not placements_raw:
            raise SpecError("placements must be a non-empty array", key=key,
                            field="placements")
        expected_count = total_bytes // shard_size
        if len(placements_raw) != expected_count:
            raise SpecError(
                f"expected {expected_count} placements for {expected_count} shards, "
                f"got {len(placements_raw)}",
                key=key, field="placements",
            )

        placements: list[ShardPlacement] = []
        seen_indexes: set[int] = set()
        for pos, item in enumerate(placements_raw):
            if not isinstance(item, dict):
                raise SpecError("each placement must be an object", key=key,
                                field=f"placements[{pos}]")
            shard_index = item.get("shard_index")
            node_id = item.get("node_id")
            if not _is_nonneg_int(shard_index):
                raise SpecError("shard_index must be a non-negative integer",
                                key=key, field=f"placements[{pos}].shard_index")
            if shard_index in seen_indexes:
                raise SpecError(f"duplicate shard_index {shard_index}", key=key,
                                field="placements.shard_index")
            if node_id not in nodes:
                raise SpecError(f"placement references unknown node {node_id}",
                                key=key, field="placements.node_id")
            size = item.get("size", shard_size)
            if not _is_positive_int(size):
                raise SpecError("placement size must be a positive integer",
                                key=key, field="placements.size")
            seen_indexes.add(shard_index)
            placements.append(ShardPlacement(shard_index=int(shard_index),
                                              node_id=node_id, size=int(size)))
        if seen_indexes != set(range(expected_count)):
            raise SpecError("placement shard_indexes must form 0..N-1",
                            key=key, field="placements.shard_index")

        manifests[key] = ObjectManifest(
            object_key=key,
            total_bytes=int(total_bytes),
            shard_size=int(shard_size),
            content_seed=int(raw["content_seed"]),
            placements=tuple(placements),
        )
    return manifests


def load_request_stream(path: str, manifests: dict[str, ObjectManifest]) -> list[Request]:
    if not os.path.exists(path):
        raise SpecError(f"request stream not found: {path}", field="path")
    requests: list[Request] = []
    previous_seq = None
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpecError(f"invalid JSONL: {exc.msg}", field="request",
                                line=line_number) from exc
            if not isinstance(raw, dict):
                raise SpecError("each request must be a JSON object",
                                field="request", line=line_number)
            seq = raw.get("seq")
            if not _is_nonneg_int(seq):
                raise SpecError("seq must be a non-negative integer",
                                field="seq", line=line_number)
            if previous_seq is not None and seq <= previous_seq:
                raise SpecError(
                    f"seq must be strictly increasing (previous {previous_seq}, got {seq})",
                    field="seq", line=line_number,
                )
            op = raw.get("op")
            if op not in VALID_OPS:
                raise SpecError(f"op must be one of {sorted(VALID_OPS)}",
                                field="op", line=line_number)
            key = raw.get("object_key")
            if key not in manifests:
                raise SpecError(f"request references unknown object_key {key}",
                                key=str(key), field="object_key", line=line_number)

            shard_index = raw.get("shard_index")
            if shard_index is not None:
                if not _is_nonneg_int(shard_index) or shard_index >= manifests[key].shard_count:
                    raise SpecError("shard_index out of manifest range",
                                    key=key, field="shard_index", line=line_number)
            expected_version = raw.get("expected_lease_version")
            if expected_version is not None and not _is_nonneg_int(expected_version):
                raise SpecError("expected_lease_version must be a non-negative integer",
                                key=key, field="expected_lease_version", line=line_number)
            ttl_ms = raw.get("ttl_ms")
            if ttl_ms is not None and not _is_positive_int(ttl_ms):
                raise SpecError("ttl_ms must be a positive integer",
                                key=key, field="ttl_ms", line=line_number)
            requests.append(
                Request(
                    seq=int(seq),
                    op=op,
                    object_key=key,
                    shard_index=None if shard_index is None else int(shard_index),
                    lease_id=raw.get("lease_id"),
                    expected_lease_version=None
                    if expected_version is None else int(expected_version),
                    ttl_ms=None if ttl_ms is None else int(ttl_ms),
                    source_file=raw.get("source_file"),
                )
            )
            previous_seq = seq
    if not requests:
        raise SpecError("request stream is empty", field="request")
    return requests


def load_bundle(nodes_path: str, manifest_path: str, request_path: str):
    nodes = load_node_specs(nodes_path)
    manifests = load_object_manifests(manifest_path, nodes)
    requests = load_request_stream(request_path, manifests)
    return nodes, manifests, requests


__all__ = [
    "load_json",
    "load_node_specs",
    "load_object_manifests",
    "load_request_stream",
    "load_bundle",
    "VALID_OPS",
]
