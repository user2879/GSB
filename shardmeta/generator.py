"""Deterministic local sample generation (fixed integer seeds, no network).

Produces node declaration, object manifest, c1/c2/c3 metadata exchange files
and a JSONL request stream. Ports are probed on 127.0.0.1 and, when busy, the
generator walks to the next free high port before regenerating everything.
"""
from __future__ import annotations

import json
import os
import socket
from typing import Any

from .content import compute_checksum, generate_shard_bytes
from .metadata import encode_c1_line, encode_json_line
from .models import FORMAT_C1, FORMAT_C2, FORMAT_C3, NormalizedRecord

DEFAULT_PORTS = (41101, 41102, 41103)


def port_free(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def choose_ports(count: int, start: int = 41101) -> list[int]:
    ports: list[int] = []
    candidate = start
    while len(ports) < count:
        if port_free("127.0.0.1", candidate):
            ports.append(candidate)
        candidate += 1
        if candidate > 65000:
            raise RuntimeError("no free high ports available")
    return ports


def build_node_specs(ports: list[int]) -> dict[str, Any]:
    nodes = []
    for index, port in enumerate(ports):
        nodes.append({
            "node_id": f"node-{index + 1}",
            "host": "127.0.0.1",
            "port": port,
            "capacity_slots": 4,
            "lease_limit": 3,
            "seed": 1000 + index * 100,
            "meta_version": 1 + index,
        })
    return {"schema": "shardmeta/node-declaration/v1", "nodes": nodes}


def build_manifest(node_ids: list[str]) -> dict[str, Any]:
    objects = [
        {
            "object_key": "obj-alpha",
            "total_bytes": 24,
            "shard_size": 8,
            "content_seed": 7001,
            "placements": [
                {"shard_index": 0, "node_id": node_ids[0], "size": 8},
                {"shard_index": 1, "node_id": node_ids[1], "size": 8},
                {"shard_index": 2, "node_id": node_ids[2], "size": 8},
            ],
        },
        {
            "object_key": "obj-beta",
            "total_bytes": 16,
            "shard_size": 8,
            "content_seed": 7002,
            "placements": [
                {"shard_index": 0, "node_id": node_ids[1], "size": 8},
                {"shard_index": 1, "node_id": node_ids[0], "size": 8},
            ],
        },
        {
            "object_key": "obj-gamma",
            "total_bytes": 8,
            "shard_size": 8,
            "content_seed": 7003,
            "placements": [
                {"shard_index": 0, "node_id": node_ids[2], "size": 8},
            ],
        },
    ]
    return {"schema": "shardmeta/object-manifest/v1", "objects": objects}


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _record(node_seed: int, object_seed: int, node_id: str, key: str,
            shard_index: int, size: int, merge_id: str, meta_version: int,
            parent: str | None, cursor: int, fmt: int, line: int) -> NormalizedRecord:
    data = generate_shard_bytes(node_seed, object_seed, key, shard_index, size)
    checksum = compute_checksum(node_id, key, shard_index, data)
    return NormalizedRecord(
        object_key=key, shard_index=shard_index, shard_digest=checksum.shard_digest,
        merge_id=merge_id, shard_bytes=size, meta_version=meta_version,
        content_check=checksum.content_check, parent_merge_id=parent,
        commit_cursor=cursor, source_format=fmt, source_line=line,
    )


def build_metadata_files(out_dir: str, node_specs: list[dict],
                         manifest: dict) -> dict[str, str]:
    seeds = {node["node_id"]: node["seed"] for node in node_specs}
    versions = {node["node_id"]: node["meta_version"] for node in node_specs}
    obj_index = {obj["object_key"]: obj for obj in manifest["objects"]}

    def placement(key: str, shard_index: int):
        return next(item for item in obj_index[key]["placements"]
                    if item["shard_index"] == shard_index)

    # c1: fixed width, alpha shards 0 and 1 (lowest format, minimal fields).
    c1_records = []
    line = 1
    for key, shard_index in (("obj-alpha", 0), ("obj-alpha", 1)):
        place = placement(key, shard_index)
        node_id = place["node_id"]
        obj = obj_index[key]
        record = _record(seeds[node_id], obj["content_seed"], node_id, key,
                         shard_index, place["size"], f"C1-{key}-{shard_index}",
                         versions[node_id], None, 0, FORMAT_C1, line)
        c1_records.append(record)
        line += 1
    c1_path = os.path.join(out_dir, "metadata.c1.txt")
    with open(c1_path, "w", encoding="utf-8") as handle:
        for record in c1_records:
            handle.write(encode_c1_line(record) + "\n")

    # c2: JSON lines, alpha shard 2 and beta shard 0 (adds merge/bytes/version).
    c2_specs = [
        ("obj-alpha", 2, "C2-alpha-2", None),
        ("obj-beta", 0, "C2-beta-0", None),
    ]
    c2_path = os.path.join(out_dir, "metadata.c2.jsonl")
    with open(c2_path, "w", encoding="utf-8") as handle:
        for line, (key, shard_index, merge_id, parent) in enumerate(c2_specs, start=1):
            place = placement(key, shard_index)
            node_id = place["node_id"]
            obj = obj_index[key]
            record = _record(seeds[node_id], obj["content_seed"], node_id, key,
                             shard_index, place["size"], merge_id,
                             versions[node_id], parent, line, FORMAT_C2, line)
            handle.write(encode_json_line(record, FORMAT_C2) + "\n")

    # c3: JSON lines with content check, parent chain and commit cursor.
    c3_specs = [
        ("obj-beta", 1, "C3-beta-1", None, 1),
        ("obj-gamma", 0, "C3-gamma-0", None, 2),
        # A newer re-upload of alpha#0 (higher meta version) -> adoption winner.
        ("obj-alpha", 0, "C3-alpha-0-reup", "C3-beta-1", 3),
    ]
    c3_path = os.path.join(out_dir, "metadata.c3.jsonl")
    with open(c3_path, "w", encoding="utf-8") as handle:
        for line, (key, shard_index, merge_id, parent, cursor) in enumerate(c3_specs, start=1):
            place = placement(key, shard_index)
            node_id = place["node_id"]
            obj = obj_index[key]
            record = _record(seeds[node_id], obj["content_seed"], node_id, key,
                             shard_index, place["size"], merge_id,
                             versions[node_id] + 1, parent, cursor, FORMAT_C3, line)
            handle.write(encode_json_line(record, FORMAT_C3) + "\n")

    return {"c1": c1_path, "c2": c2_path, "c3": c3_path}


def build_request_stream() -> list[dict]:
    return [
        {"seq": 1, "op": "grant", "object_key": "obj-alpha", "shard_index": 0,
         "ttl_ms": 300},
        {"seq": 2, "op": "upload", "object_key": "obj-alpha", "shard_index": 0},
        {"seq": 3, "op": "grant", "object_key": "obj-alpha", "shard_index": 1,
         "ttl_ms": 300},
        {"seq": 4, "op": "renew", "object_key": "obj-alpha", "shard_index": 1,
         "lease_id": "resolved-at-runtime", "expected_lease_version": 1,
         "ttl_ms": 300, "runtime_lease": True},
        {"seq": 5, "op": "upload", "object_key": "obj-alpha", "shard_index": 1},
        {"seq": 6, "op": "grant", "object_key": "obj-beta", "shard_index": 0,
         "ttl_ms": 120},
        {"seq": 7, "op": "upload", "object_key": "obj-beta", "shard_index": 0},
        {"seq": 8, "op": "read", "object_key": "obj-alpha"},
        {"seq": 9, "op": "grant", "object_key": "obj-gamma", "shard_index": 0,
         "ttl_ms": 300},
        {"seq": 10, "op": "abort", "object_key": "obj-gamma", "shard_index": 0,
         "lease_id": "resolved-at-runtime", "runtime_lease": True},
        {"seq": 11, "op": "read", "object_key": "obj-beta"},
        {"seq": 12, "op": "read", "object_key": "obj-gamma"},
    ]


def write_request_stream(path: str, requests: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, sort_keys=True) + "\n")


def generate_sample_bundle(out_dir: str, start_port: int = 41101) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ports = choose_ports(3, start_port)
    nodes_doc = build_node_specs(ports)
    node_ids = [node["node_id"] for node in nodes_doc["nodes"]]
    manifest_doc = build_manifest(node_ids)

    nodes_path = os.path.join(out_dir, "nodes.json")
    manifest_path = os.path.join(out_dir, "objects.json")
    request_path = os.path.join(out_dir, "requests.jsonl")
    _write_json(nodes_path, nodes_doc)
    _write_json(manifest_path, manifest_doc)
    metadata_paths = build_metadata_files(out_dir, nodes_doc["nodes"], manifest_doc)
    write_request_stream(request_path, build_request_stream())

    return {
        "nodes": nodes_path,
        "objects": manifest_path,
        "requests": request_path,
        **metadata_paths,
    }


__all__ = [
    "generate_sample_bundle",
    "choose_ports",
    "port_free",
    "build_node_specs",
    "build_manifest",
    "build_request_stream",
]
