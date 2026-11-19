"""Strict local JSON specification loading and validation."""

from __future__ import annotations

import heapq
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .canonical import ensure_relative_path, sha256_object
from .errors import CycleError, SpecValidationError
from .models import BlockDecl, LogicalShard, NodeDecl, RelationDecl, ShardDecl, Spec

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecValidationError("must be a JSON object", field_path=field_path)
    return value


def _require_list(value: Any, field_path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecValidationError("must be a JSON array", field_path=field_path)
    return value


def _require_string(value: Any, field_path: str, *, shard_id: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError(
            "must be a non-empty string", field_path=field_path, shard_id=shard_id
        )
    return value


def validate_digest(value: Any, field_path: str, *, shard_id: str | None = None) -> str:
    text = _require_string(value, field_path, shard_id=shard_id)
    if not _DIGEST_RE.match(text):
        raise SpecValidationError(
            "must be a lowercase sha256 digest (sha256:<64 hex>)",
            field_path=field_path,
            shard_id=shard_id,
        )
    return text


def compute_block_checksum(
    block_id: str,
    seq_start: int,
    seq_end: int,
    logical_meta: dict[str, tuple[int, int, str]],
    seq_to_logical: dict[int, str],
) -> str:
    shards = []
    for seq in range(seq_start, seq_end + 1):
        logical_id = seq_to_logical[seq]
        logical_seq, byte_count, digest = logical_meta[logical_id]
        shards.append(
            {
                "seq": logical_seq,
                "logical_id": logical_id,
                "bytes": byte_count,
                "digest": digest,
            }
        )
    return sha256_object(
        {
            "block_id": block_id,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "shards": shards,
        }
    )


def stable_topological_order(
    nodes: Iterable[str], relations: Iterable[RelationDecl]
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    node_set = set(nodes)
    adjacency_sets: dict[str, set[str]] = {node: set() for node in node_set}
    predecessor_sets: dict[str, set[str]] = {node: set() for node in node_set}
    indegree: dict[str, int] = {node: 0 for node in node_set}
    for relation in relations:
        for source in relation.sources:
            if relation.target not in adjacency_sets[source]:
                adjacency_sets[source].add(relation.target)
                predecessor_sets[relation.target].add(source)
                indegree[relation.target] += 1

    ready = [node for node, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for target in sorted(adjacency_sets[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(order) != len(node_set):
        remaining = sorted(node_set - set(order))
        # Deterministically expose one concrete cycle by walking the residual graph.
        seen: set[str] = set()
        current = remaining[0]
        while current not in seen:
            seen.add(current)
            successors = [target for target in sorted(adjacency_sets[current]) if target in remaining]
            if not successors:
                break
            current = successors[0]
        raise CycleError(
            f"mirror relation graph contains a cycle through {current}",
            shard_id=current,
        )
    adjacency = {node: tuple(sorted(adjacency_sets[node])) for node in node_set}
    predecessors = {node: tuple(sorted(predecessor_sets[node])) for node in node_set}
    return tuple(order), adjacency, predecessors


def load_spec(path: str | Path, *, raw_override: dict[str, Any] | None = None) -> Spec:
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve()
    spec_dir = source_path.parent
    if raw_override is None:
        try:
            text = source_path.read_text(encoding="utf-8-sig")
            raw = json.loads(text)
        except FileNotFoundError as exc:
            raise SpecValidationError("specification file does not exist", field_path="$.path") from exc
        except json.JSONDecodeError as exc:
            raise SpecValidationError(
                f"invalid JSON: {exc.msg}",
                field_path=f"$ line {exc.lineno} column {exc.colno}",
            ) from exc
        except OSError as exc:
            raise SpecValidationError("cannot read specification file", field_path="$.path") from exc
    else:
        raw = raw_override
    root = _require_object(raw, "$")

    if root.get("format", "shardrep-spec") != "shardrep-spec":
        raise SpecValidationError("format must be 'shardrep-spec'", field_path="format")
    format_version = root.get("format_version", 1)
    if not _is_int(format_version) or format_version != 1:
        raise SpecValidationError("only integer format_version 1 is supported", field_path="format_version")
    group_id = _require_string(root.get("group_id"), "group_id")
    group_version = root.get("group_version")
    if not _is_int(group_version) or group_version < 1:
        raise SpecValidationError("must be a positive integer", field_path="group_version")
    graph_version = root.get("mirror_graph_version")
    if not _is_int(graph_version) or graph_version < 1:
        raise SpecValidationError("must be a positive integer", field_path="mirror_graph_version")

    content = _require_object(root.get("content"), "content")
    seed = content.get("seed")
    if not _is_int(seed):
        raise SpecValidationError("must be an integer", field_path="content.seed")
    engine = content.get("engine", "shake256-v1")
    _require_string(engine, "content.engine")

    nodes_raw = _require_list(root.get("nodes"), "nodes")
    if not nodes_raw:
        raise SpecValidationError("at least one node is required", field_path="nodes")

    nodes: list[NodeDecl] = []
    node_ids: set[str] = set()
    shard_ids: set[str] = set()
    logical_meta: dict[str, tuple[int, int, str]] = {}
    all_shards: list[ShardDecl] = []
    for node_index, node_value in enumerate(nodes_raw):
        node_path = f"nodes[{node_index}]"
        node_obj = _require_object(node_value, node_path)
        node_id = _require_string(node_obj.get("node_id"), f"{node_path}.node_id")
        if node_id in node_ids:
            raise SpecValidationError("duplicate node_id", field_path=f"{node_path}.node_id", shard_id=None)
        node_ids.add(node_id)
        directory = _require_string(node_obj.get("directory"), f"{node_path}.directory")
        node_dir = ensure_relative_path(directory, f"{node_path}.directory", base=spec_dir)
        if not node_dir.exists() or not node_dir.is_dir():
            raise SpecValidationError("node directory does not exist", field_path=f"{node_path}.directory")
        pack_quota = node_obj.get("pack_quota", 1)
        if not _is_int(pack_quota) or pack_quota < 0:
            raise SpecValidationError("must be a non-negative integer", field_path=f"{node_path}.pack_quota")
        repair_enabled = node_obj.get("repair_enabled", True)
        if not isinstance(repair_enabled, bool):
            raise SpecValidationError("must be a boolean", field_path=f"{node_path}.repair_enabled")
        shards_raw = _require_list(node_obj.get("shards"), f"{node_path}.shards")
        node_shards: list[ShardDecl] = []
        seen_logical: set[str] = set()
        seen_files: set[str] = set()
        for shard_index, shard_value in enumerate(shards_raw):
            shard_path = f"{node_path}.shards[{shard_index}]"
            shard_obj = _require_object(shard_value, shard_path)
            shard_id = _require_string(shard_obj.get("shard_id"), f"{shard_path}.shard_id")
            if shard_id in shard_ids:
                raise SpecValidationError(
                    "duplicate shard_id across the specification",
                    field_path=f"{shard_path}.shard_id",
                    shard_id=shard_id,
                )
            shard_ids.add(shard_id)
            logical_id = _require_string(
                shard_obj.get("logical_id"), f"{shard_path}.logical_id", shard_id=shard_id
            )
            if logical_id in seen_logical:
                raise SpecValidationError(
                    "a node may hold only one replica of a logical shard",
                    field_path=f"{shard_path}.logical_id",
                    shard_id=shard_id,
                )
            seen_logical.add(logical_id)
            seq = shard_obj.get("seq")
            if not _is_int(seq) or seq < 0:
                raise SpecValidationError(
                    "must be a non-negative integer", field_path=f"{shard_path}.seq", shard_id=shard_id
                )
            byte_count = shard_obj.get("bytes")
            if not _is_int(byte_count) or byte_count <= 0:
                raise SpecValidationError(
                    "must be a positive integer", field_path=f"{shard_path}.bytes", shard_id=shard_id
                )
            digest = validate_digest(shard_obj.get("digest"), f"{shard_path}.digest", shard_id=shard_id)
            file_name = _require_string(shard_obj.get("file"), f"{shard_path}.file", shard_id=shard_id)
            shard_file = ensure_relative_path(file_name, f"{shard_path}.file", base=node_dir)
            if file_name in seen_files:
                raise SpecValidationError(
                    "duplicate shard file within a node",
                    field_path=f"{shard_path}.file",
                    shard_id=shard_id,
                )
            seen_files.add(file_name)
            meta = (seq, byte_count, digest)
            previous = logical_meta.get(logical_id)
            if previous is not None and previous != meta:
                raise SpecValidationError(
                    "all replicas of a logical shard must declare the same seq, bytes and digest",
                    field_path=f"{shard_path}.digest",
                    shard_id=shard_id,
                )
            logical_meta[logical_id] = meta
            shard_decl = ShardDecl(
                node_id=node_id,
                shard_id=shard_id,
                logical_id=logical_id,
                seq=seq,
                byte_count=byte_count,
                digest=digest,
                file_name=file_name,
                path=shard_file,
            )
            node_shards.append(shard_decl)
            all_shards.append(shard_decl)
        nodes.append(
            NodeDecl(
                node_id=node_id,
                directory=directory,
                path=node_dir,
                pack_quota=pack_quota,
                repair_enabled=repair_enabled,
                shards=tuple(sorted(node_shards, key=lambda item: (item.seq, item.logical_id, item.shard_id))),
            )
        )

    # A logical sequence number must identify exactly one logical shard.
    seq_to_logical: dict[int, str] = {}
    for logical_id, (seq, _byte_count, _digest) in logical_meta.items():
        previous = seq_to_logical.get(seq)
        if previous is not None and previous != logical_id:
            raise SpecValidationError(
                f"logical sequence {seq} is assigned to both {previous} and {logical_id}",
                field_path="nodes",
                shard_id=logical_id,
            )
        seq_to_logical[seq] = logical_id
    if not seq_to_logical:
        raise SpecValidationError("at least one physical shard is required", field_path="nodes")
    expected_seqs = set(range(max(seq_to_logical) + 1))
    if set(seq_to_logical) != expected_seqs:
        missing = sorted(expected_seqs - set(seq_to_logical))
        raise SpecValidationError(
            f"logical sequence range is not contiguous; missing {missing}",
            field_path="nodes",
        )

    blocks_raw = _require_list(root.get("blocks"), "blocks")
    if not blocks_raw:
        raise SpecValidationError("at least one block is required", field_path="blocks")
    blocks: list[BlockDecl] = []
    block_ids: set[str] = set()
    for block_index, block_value in enumerate(blocks_raw):
        block_path = f"blocks[{block_index}]"
        block_obj = _require_object(block_value, block_path)
        block_id = _require_string(block_obj.get("block_id"), f"{block_path}.block_id")
        if block_id in block_ids:
            raise SpecValidationError("duplicate block_id", field_path=f"{block_path}.block_id")
        block_ids.add(block_id)
        start = block_obj.get("seq_start")
        end = block_obj.get("seq_end")
        if not _is_int(start) or start < 0:
            raise SpecValidationError("must be a non-negative integer", field_path=f"{block_path}.seq_start")
        if not _is_int(end) or end < start:
            raise SpecValidationError("must be an integer >= seq_start", field_path=f"{block_path}.seq_end")
        if start > max(seq_to_logical) or end > max(seq_to_logical):
            raise SpecValidationError("block range exceeds declared shard sequences", field_path=f"{block_path}.seq_end")
        for seq in range(start, end + 1):
            if seq not in seq_to_logical:
                raise SpecValidationError(
                    f"block references undeclared shard sequence {seq}",
                    field_path=f"{block_path}.seq_start",
                )
        checksum = validate_digest(block_obj.get("checksum"), f"{block_path}.checksum")
        blocks.append(BlockDecl(block_id=block_id, seq_start=start, seq_end=end, checksum=checksum))

    ordered_blocks = sorted(blocks, key=lambda item: (item.seq_start, item.seq_end, item.block_id))
    if ordered_blocks[0].seq_start != 0:
        raise SpecValidationError("first block must start at sequence 0", field_path="blocks[0].seq_start")
    for previous, current in zip(ordered_blocks, ordered_blocks[1:]):
        if current.seq_start != previous.seq_end + 1:
            raise SpecValidationError(
                "block sequence ranges must meet end-to-start without gaps or overlap",
                field_path=f"blocks[{blocks.index(current)}].seq_start",
            )
    if ordered_blocks[-1].seq_end != max(seq_to_logical):
        raise SpecValidationError("block sequence ranges do not cover all declared shards", field_path="blocks")

    block_for_seq: dict[int, str] = {}
    for block in ordered_blocks:
        for seq in block.seqs:
            block_for_seq[seq] = block.block_id

    for block in ordered_blocks:
        actual = compute_block_checksum(
            block.block_id, block.seq_start, block.seq_end, logical_meta, seq_to_logical
        )
        if actual != block.checksum:
            raise SpecValidationError(
                f"block checksum mismatch; expected {actual}",
                field_path=f"blocks[{blocks.index(block)}].checksum",
            )

    relations_raw = _require_list(root.get("relations", []), "relations")
    relations: list[RelationDecl] = []
    for relation_index, relation_value in enumerate(relations_raw):
        relation_path = f"relations[{relation_index}]"
        relation_obj = _require_object(relation_value, relation_path)
        kind = relation_obj.get("kind")
        if kind not in {"mirror", "join"}:
            raise SpecValidationError("must be 'mirror' or 'join'", field_path=f"{relation_path}.kind")
        target = _require_string(relation_obj.get("target"), f"{relation_path}.target")
        if target not in logical_meta:
            raise SpecValidationError(
                "relation target does not identify a declared logical shard",
                field_path=f"{relation_path}.target",
                shard_id=target,
            )
        if kind == "mirror":
            source = _require_string(relation_obj.get("source"), f"{relation_path}.source")
            sources = (source,)
        else:
            source_values = _require_list(relation_obj.get("sources"), f"{relation_path}.sources")
            if not source_values:
                raise SpecValidationError("join requires at least one source", field_path=f"{relation_path}.sources")
            sources = tuple(
                _require_string(value, f"{relation_path}.sources[{index}]")
                for index, value in enumerate(source_values)
            )
            if len(set(sources)) != len(sources):
                raise SpecValidationError("join sources must be unique", field_path=f"{relation_path}.sources")
        for source in sources:
            if source not in logical_meta:
                raise SpecValidationError(
                    "relation source does not identify a declared logical shard",
                    field_path=f"{relation_path}.source",
                    shard_id=source,
                )
        if target in sources:
            raise CycleError("relation cannot reference its own target", shard_id=target)
        relations.append(RelationDecl(kind=kind, sources=tuple(sorted(sources)), target=target))

    topo_order, adjacency, predecessors = stable_topological_order(logical_meta, relations)

    replicas_by_logical: dict[str, list[ShardDecl]] = {logical_id: [] for logical_id in logical_meta}
    for shard in all_shards:
        replicas_by_logical[shard.logical_id].append(shard)
    logical_shards: dict[str, LogicalShard] = {}
    for logical_id, (seq, byte_count, digest) in logical_meta.items():
        replicas = tuple(
            sorted(
                replicas_by_logical[logical_id],
                key=lambda item: (item.node_id, item.shard_id),
            )
        )
        if not replicas:
            raise SpecValidationError(
                "logical shard has no replicas", field_path="nodes", shard_id=logical_id
            )
        logical_shards[logical_id] = LogicalShard(
            logical_id=logical_id,
            seq=seq,
            byte_count=byte_count,
            digest=digest,
            block_id=block_for_seq[seq],
            replicas=replicas,
        )

    block_by_id = {block.block_id: block for block in ordered_blocks}
    node_by_id = {node.node_id: node for node in sorted(nodes, key=lambda item: item.node_id)}
    shard_by_id = {shard.shard_id: shard for shard in sorted(all_shards, key=lambda item: item.shard_id)}
    return Spec(
        raw=root,
        source_path=source_path,
        spec_dir=spec_dir,
        format_version=format_version,
        group_id=group_id,
        group_version=group_version,
        mirror_graph_version=graph_version,
        seed=seed,
        content_engine=engine,
        nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
        blocks=tuple(ordered_blocks),
        relations=tuple(sorted(relations, key=lambda item: (item.target, item.sources, item.kind))),
        logical_shards=logical_shards,
        seq_to_logical=seq_to_logical,
        block_by_id=block_by_id,
        node_by_id=node_by_id,
        shard_by_id=shard_by_id,
        topo_order=topo_order,
        adjacency=adjacency,
        predecessors=predecessors,
    )


def spec_manifest_digest(spec: Spec) -> str:
    return sha256_object(
        {
            "group_id": spec.group_id,
            "group_version": spec.group_version,
            "mirror_graph_version": spec.mirror_graph_version,
            "nodes": [node.as_dict() for node in spec.nodes],
            "blocks": [block.as_dict() for block in spec.blocks],
            "relations": [relation.as_dict() for relation in spec.relations],
        }
    )
