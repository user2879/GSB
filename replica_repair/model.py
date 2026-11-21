"""Strict v1/v2/v3 replica-manifest model and field-path validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .errors import FormatError
from .util import canonical_json_dumps, ensure_int, sha256_text, validate_simple_id


CURRENT_MANIFEST_VERSION = 3
SUPPORTED_MANIFEST_VERSIONS = (1, 2, 3)
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RELATION_ALGORITHM = "sha256-concat-v1"


def validate_digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise FormatError("digest must be sha256:<64 lowercase hex>", path=path)
    return value


def validate_positive_int(value: Any, path: str, *, maximum: int | None = None) -> int:
    if not ensure_int(value) or value <= 0:
        raise FormatError("value must be a positive integer", path=path)
    if maximum is not None and value > maximum:
        raise FormatError(f"value must be <= {maximum}", path=path)
    return value


@dataclass(frozen=True)
class ChecksumDerivation:
    algorithm: str
    source_block_ids: tuple[str, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "source_block_ids": list(self.source_block_ids),
            "digest": self.digest,
        }


@dataclass(frozen=True)
class BlockRecord:
    block_id: str
    size_bytes: int
    digest: str | None = None
    repair_weight: int = 1
    checksum_derivation: ChecksumDerivation | None = None
    repair_source: str = "origin"

    def to_dict(self, version: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "block_id": self.block_id,
            "size_bytes": self.size_bytes,
        }
        if version == 1:
            return payload
        if self.digest is None:
            raise FormatError(
                "v2/v3 block requires a digest",
                path=f"$.nodes[].blocks[{self.block_id}].digest",
            )
        payload["digest"] = self.digest
        payload["repair_weight"] = self.repair_weight
        if version == 2:
            return payload
        if version != 3:
            raise FormatError("unsupported manifest version", path="$.format_version")
        payload["checksum_derivation"] = (
            self.checksum_derivation.to_dict() if self.checksum_derivation is not None else None
        )
        payload["repair_source"] = self.repair_source
        return payload


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    priority: int | None
    blocks: tuple[BlockRecord, ...]

    def block_map(self) -> dict[str, BlockRecord]:
        return {block.block_id: block for block in self.blocks}

    def to_dict(self, version: int) -> dict[str, Any]:
        payload: dict[str, Any] = {"node_id": self.node_id}
        if version >= 2:
            if self.priority is None:
                raise FormatError(
                    "v2/v3 node requires a priority",
                    path=f"$.nodes[{self.node_id}].priority",
                )
            payload["priority"] = self.priority
        payload["blocks"] = [block.to_dict(version) for block in self.blocks]
        return payload


@dataclass(frozen=True)
class Manifest:
    format_version: int
    group_id: str
    manifest_version: int
    nodes: tuple[NodeRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        return self.format_version == CURRENT_MANIFEST_VERSION

    def node_map(self) -> dict[str, NodeRecord]:
        return {node.node_id: node for node in self.nodes}

    def block_map(self) -> dict[str, BlockRecord]:
        result: dict[str, BlockRecord] = {}
        for node in self.nodes:
            for block in node.blocks:
                result[block.block_id] = block
        return result

    def canonical_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def fingerprint(self, *, include_manifest_version: bool = True) -> str:
        payload = self.to_dict()
        if not include_manifest_version:
            payload.pop("manifest_version", None)
        return sha256_text(canonical_json_dumps(payload))

    def to_dict(self, version: int | None = None) -> dict[str, Any]:
        target_version = self.format_version if version is None else version
        if target_version not in SUPPORTED_MANIFEST_VERSIONS:
            raise FormatError("unsupported manifest version", path="$.format_version")
        payload: dict[str, Any] = {
            "format_version": target_version,
            "group_id": self.group_id,
            "manifest_version": self.manifest_version,
            "nodes": [node.to_dict(target_version) for node in sorted(self.nodes, key=lambda n: n.node_id)],
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def to_json(self, *, version: int | None = None) -> str:
        return canonical_json_dumps(self.to_dict(version), pretty=True, newline=True)

    @classmethod
    def from_file(cls, path: Path | str) -> "Manifest":
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FormatError(
                f"cannot read manifest: {exc}",
                path="$",
                details={"file": str(path)},
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FormatError(
                "manifest is not valid JSON",
                path="$",
                details={"line": exc.lineno, "column": exc.colno},
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> "Manifest":
        if not isinstance(payload, dict):
            raise FormatError("manifest root must be an object", path="$")
        allowed = {"format_version", "group_id", "manifest_version", "nodes", "metadata"}
        _reject_unknown(payload, allowed, "$")

        if "format_version" not in payload:
            raise FormatError("required field is missing", path="$.format_version")
        version = payload["format_version"]
        if not ensure_int(version):
            raise FormatError("format_version must be an integer", path="$.format_version")
        if version not in SUPPORTED_MANIFEST_VERSIONS:
            raise FormatError(
                "unknown manifest version",
                path="$.format_version",
                details={"supported": list(SUPPORTED_MANIFEST_VERSIONS)},
            )

        if "group_id" not in payload:
            raise FormatError("required field is missing", path="$.group_id")
        group_id = validate_simple_id(payload["group_id"], "$.group_id")

        if "manifest_version" not in payload:
            raise FormatError("required field is missing", path="$.manifest_version")
        manifest_version = validate_positive_int(payload["manifest_version"], "$.manifest_version")

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise FormatError("metadata must be an object", path="$.metadata")

        if "nodes" not in payload:
            raise FormatError("required field is missing", path="$.nodes")
        raw_nodes = payload["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise FormatError("nodes must be a non-empty array", path="$.nodes")

        nodes: list[NodeRecord] = []
        seen_nodes: set[str] = set()
        for node_index, raw_node in enumerate(raw_nodes):
            node = _decode_node(raw_node, node_index, version)
            if node.node_id in seen_nodes:
                raise FormatError("duplicate node_id", path=f"$.nodes[{node_index}].node_id")
            seen_nodes.add(node.node_id)
            nodes.append(node)
        nodes.sort(key=lambda node: node.node_id)

        manifest = cls(
            format_version=version,
            group_id=group_id,
            manifest_version=manifest_version,
            nodes=tuple(nodes),
            metadata=metadata,
        )
        if version == 3:
            _validate_relations(manifest)
        return manifest


def _decode_node(raw_node: Any, node_index: int, version: int) -> NodeRecord:
    base_path = f"$.nodes[{node_index}]"
    if not isinstance(raw_node, dict):
        raise FormatError("node must be an object", path=base_path)
    if version == 1:
        allowed = {"node_id", "blocks"}
    else:
        allowed = {"node_id", "priority", "blocks"}
    _reject_unknown(raw_node, allowed, base_path)

    if "node_id" not in raw_node:
        raise FormatError("required field is missing", path=f"{base_path}.node_id")
    node_id = validate_simple_id(raw_node["node_id"], f"{base_path}.node_id")

    priority: int | None = None
    if version >= 2:
        if "priority" not in raw_node:
            raise FormatError("required field is missing", path=f"{base_path}.priority")
        priority = validate_positive_int(raw_node["priority"], f"{base_path}.priority", maximum=1_000_000)

    if "blocks" not in raw_node:
        raise FormatError("required field is missing", path=f"{base_path}.blocks")
    raw_blocks = raw_node["blocks"]
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise FormatError("blocks must be a non-empty array", path=f"{base_path}.blocks")

    blocks: list[BlockRecord] = []
    seen_blocks: set[str] = set()
    for block_index, raw_block in enumerate(raw_blocks):
        block = _decode_block(raw_block, base_path, block_index, version)
        if block.block_id in seen_blocks:
            raise FormatError("duplicate block_id", path=f"{base_path}.blocks[{block_index}].block_id")
        seen_blocks.add(block.block_id)
        blocks.append(block)
    blocks.sort(key=lambda block: block.block_id)
    return NodeRecord(node_id=node_id, priority=priority, blocks=tuple(blocks))


def _decode_block(
    raw_block: Any,
    node_path: str,
    block_index: int,
    version: int,
) -> BlockRecord:
    base_path = f"{node_path}.blocks[{block_index}]"
    if not isinstance(raw_block, dict):
        raise FormatError("block must be an object", path=base_path)
    if version == 1:
        allowed = {"block_id", "size_bytes"}
    elif version == 2:
        allowed = {"block_id", "size_bytes", "digest", "repair_weight"}
    else:
        allowed = {
            "block_id",
            "size_bytes",
            "digest",
            "repair_weight",
            "checksum_derivation",
            "repair_source",
        }
    _reject_unknown(raw_block, allowed, base_path)

    if "block_id" not in raw_block:
        raise FormatError("required field is missing", path=f"{base_path}.block_id")
    block_id = validate_simple_id(raw_block["block_id"], f"{base_path}.block_id")

    if "size_bytes" not in raw_block:
        raise FormatError("required field is missing", path=f"{base_path}.size_bytes")
    size_bytes = validate_positive_int(raw_block["size_bytes"], f"{base_path}.size_bytes")

    digest: str | None = None
    repair_weight = 1
    derivation: ChecksumDerivation | None = None
    repair_source = "origin"

    if version >= 2:
        if "digest" not in raw_block:
            raise FormatError("required field is missing", path=f"{base_path}.digest")
        digest = validate_digest(raw_block["digest"], f"{base_path}.digest")

        if "repair_weight" not in raw_block:
            raise FormatError("required field is missing", path=f"{base_path}.repair_weight")
        repair_weight = validate_positive_int(
            raw_block["repair_weight"],
            f"{base_path}.repair_weight",
            maximum=1_000_000,
        )

    if version == 3:
        if "checksum_derivation" not in raw_block:
            raise FormatError("required field is missing", path=f"{base_path}.checksum_derivation")
        derivation_value = raw_block["checksum_derivation"]
        if derivation_value is not None:
            derivation = _decode_derivation(derivation_value, f"{base_path}.checksum_derivation")

        if "repair_source" not in raw_block:
            raise FormatError("required field is missing", path=f"{base_path}.repair_source")
        repair_source = raw_block["repair_source"]
        if not isinstance(repair_source, str) or not repair_source:
            raise FormatError("repair_source must be a non-empty string", path=f"{base_path}.repair_source")
        if repair_source != "origin":
            validate_simple_id(repair_source, f"{base_path}.repair_source")

    return BlockRecord(
        block_id=block_id,
        size_bytes=size_bytes,
        digest=digest,
        repair_weight=repair_weight,
        checksum_derivation=derivation,
        repair_source=repair_source,
    )


def _decode_derivation(value: Any, path: str) -> ChecksumDerivation:
    if not isinstance(value, dict):
        raise FormatError("checksum_derivation must be an object or null", path=path)
    _reject_unknown(value, {"algorithm", "source_block_ids", "digest"}, path)
    for required in ("algorithm", "source_block_ids", "digest"):
        if required not in value:
            raise FormatError("required field is missing", path=f"{path}.{required}")
    if value["algorithm"] != RELATION_ALGORITHM:
        raise FormatError(
            f"algorithm must be {RELATION_ALGORITHM}",
            path=f"{path}.algorithm",
        )
    raw_sources = value["source_block_ids"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise FormatError("source_block_ids must be a non-empty array", path=f"{path}.source_block_ids")
    sources: list[str] = []
    for index, source in enumerate(raw_sources):
        sources.append(validate_simple_id(source, f"{path}.source_block_ids[{index}]"))
    if len(set(sources)) != len(sources):
        raise FormatError("source_block_ids must be unique", path=f"{path}.source_block_ids")
    if sources != sorted(sources):
        raise FormatError("source_block_ids must be lexicographically sorted", path=f"{path}.source_block_ids")
    digest = validate_digest(value["digest"], f"{path}.digest")
    return ChecksumDerivation(
        algorithm=RELATION_ALGORITHM,
        source_block_ids=tuple(sources),
        digest=digest,
    )


def _validate_relations(manifest: Manifest) -> None:
    for node_index, node in enumerate(manifest.nodes):
        blocks = node.block_map()
        for block_index, block in enumerate(node.blocks):
            relation = block.checksum_derivation
            if relation is None:
                continue
            path = (
                f"$.nodes[{node_index}].blocks[{block_index}].checksum_derivation"
            )
            for source_id in relation.source_block_ids:
                if source_id not in blocks:
                    raise FormatError(
                        "derivation source block is not present in the same node",
                        path=f"{path}.source_block_ids",
                        details={"missing_block_id": source_id, "node_id": node.node_id},
                    )
                if source_id == block.block_id:
                    raise FormatError(
                        "derivation cannot reference itself",
                        path=f"{path}.source_block_ids",
                    )
            expected = derive_relation_digest(
                relation.algorithm,
                relation.source_block_ids,
                {source_id: blocks[source_id].digest for source_id in relation.source_block_ids},
            )
            if relation.digest != expected:
                raise FormatError(
                    "derivation digest does not match source block digests",
                    path=f"{path}.digest",
                    details={"expected": expected, "actual": relation.digest},
                )
        _validate_relation_cycles(node)


def _validate_relation_cycles(node: NodeRecord) -> None:
    blocks = node.block_map()
    state: dict[str, int] = {}

    def visit(block_id: str) -> None:
        marker = state.get(block_id, 0)
        if marker == 1:
            raise FormatError(
                "cyclic checksum derivation",
                path=f"$.nodes[{node.node_id}].blocks[{block_id}].checksum_derivation",
            )
        if marker == 2:
            return
        state[block_id] = 1
        relation = blocks[block_id].checksum_derivation
        if relation is not None:
            for source_id in relation.source_block_ids:
                visit(source_id)
        state[block_id] = 2

    for block_id in sorted(blocks):
        visit(block_id)


def derive_relation_digest(
    algorithm: str,
    source_block_ids: Iterable[str],
    source_digests: dict[str, str | None],
) -> str:
    if algorithm != RELATION_ALGORITHM:
        raise FormatError("unsupported relation algorithm", path="$.checksum_derivation.algorithm")
    sources = []
    for block_id in source_block_ids:
        digest = source_digests.get(block_id)
        if digest is None:
            raise FormatError(
                "derivation source has no digest",
                path="$.checksum_derivation.source_block_ids",
                details={"block_id": block_id},
            )
        sources.append({"block_id": block_id, "digest": digest})
    return sha256_text(
        canonical_json_dumps(
            {
                "algorithm": algorithm,
                "sources": sources,
            }
        )
    )


def recompute_derivations(nodes: Iterable[NodeRecord]) -> tuple[NodeRecord, ...]:
    """Recompute all v3 relation digests after content-level manifest edits."""

    result: list[NodeRecord] = []
    for node in nodes:
        updated: dict[str, BlockRecord] = {block.block_id: block for block in node.blocks}
        visiting: set[str] = set()
        done: set[str] = set()

        def resolve(block_id: str) -> BlockRecord:
            block = updated[block_id]
            relation = block.checksum_derivation
            if relation is None:
                done.add(block_id)
                return block
            if block_id in visiting:
                raise FormatError(
                    "cyclic checksum derivation",
                    path=f"$.nodes[{node.node_id}].blocks[{block_id}].checksum_derivation",
                )
            if block_id in done:
                return block
            visiting.add(block_id)
            resolved_sources: dict[str, str | None] = {}
            for source_id in relation.source_block_ids:
                resolved_sources[source_id] = resolve(source_id).digest
            visiting.remove(block_id)
            new_relation = replace(
                relation,
                digest=derive_relation_digest(
                    relation.algorithm,
                    relation.source_block_ids,
                    resolved_sources,
                ),
            )
            updated[block_id] = replace(block, checksum_derivation=new_relation)
            done.add(block_id)
            return updated[block_id]

        for block_id in sorted(updated):
            resolve(block_id)
        result.append(replace(node, blocks=tuple(updated[block_id] for block_id in sorted(updated))))
    return tuple(result)


def with_manifest_version(manifest: Manifest, version: int) -> Manifest:
    return replace(manifest, manifest_version=version)


def _reject_unknown(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FormatError(
            "unknown field",
            path=f"{path}.{unknown[0]}",
            details={"unknown_fields": unknown},
        )
