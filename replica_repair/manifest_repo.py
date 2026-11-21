"""Current-version manifest writes, monotonic evolution, and disk state digests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import FormatError, StateError
from .model import (
    CURRENT_MANIFEST_VERSION,
    Manifest,
    NodeRecord,
    recompute_derivations,
)
from .util import atomic_write_bytes, canonical_json_dumps, sha256_bytes, sha256_file


@dataclass(frozen=True)
class ManifestChange:
    affected_blocks: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    details: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_blocks": list(self.affected_blocks),
            "affected_nodes": list(self.affected_nodes),
            "details": list(self.details),
        }


@dataclass(frozen=True)
class SaveResult:
    manifest: Manifest
    previous_manifest_version: int | None
    version_incremented: bool
    changes: ManifestChange

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest.manifest_version,
            "previous_manifest_version": self.previous_manifest_version,
            "version_incremented": self.version_incremented,
            "changes": self.changes.to_dict(),
        }


class ManifestRepository:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def save_current(
        self,
        manifest: Manifest,
        *,
        recompute_relations: bool = False,
        force_manifest_version: int | None = None,
    ) -> SaveResult:
        previous: Manifest | None = None
        if self.path.exists():
            previous = Manifest.from_file(self.path)
            if previous.format_version != CURRENT_MANIFEST_VERSION:
                raise StateError(
                    "cannot evolve a legacy manifest before upgrading it",
                    path="$.format_version",
                    details={"found": previous.format_version},
                )
        if manifest.format_version != CURRENT_MANIFEST_VERSION:
            raise FormatError(
                "only the current manifest version may be written",
                path="$.format_version",
                details={"required": CURRENT_MANIFEST_VERSION, "found": manifest.format_version},
            )

        normalized = manifest
        if recompute_relations:
            normalized = replace(normalized, nodes=recompute_derivations(normalized.nodes))
        normalized = Manifest.from_dict(normalized.to_dict())

        target_version = normalized.manifest_version
        if force_manifest_version is not None:
            if not isinstance(force_manifest_version, int) or isinstance(force_manifest_version, bool):
                raise FormatError("manifest version must be an integer", path="$.manifest_version")
            target_version = force_manifest_version
        elif previous is not None:
            target_version = max(target_version, previous.manifest_version)

        version_incremented = False
        if previous is not None:
            semantic_changed = normalized.fingerprint(include_manifest_version=False) != previous.fingerprint(
                include_manifest_version=False
            )
            if target_version <= previous.manifest_version and (
                semantic_changed or target_version < previous.manifest_version
            ):
                target_version = previous.manifest_version + 1
                version_incremented = True

        normalized = replace(normalized, manifest_version=target_version)
        normalized = Manifest.from_dict(normalized.to_dict())
        atomic_write_bytes(self.path, normalized.to_json().encode("utf-8"))

        if previous is None:
            changes = compute_manifest_diff(None, normalized)
        else:
            changes = compute_manifest_diff(previous, normalized)
        return SaveResult(
            manifest=normalized,
            previous_manifest_version=previous.manifest_version if previous else None,
            version_incremented=version_incremented,
            changes=changes,
        )


def compute_manifest_diff(old: Manifest | None, new: Manifest) -> ManifestChange:
    details: set[tuple[str, str, str]] = set()

    def add(block_id: str, node_id: str, reason: str) -> None:
        details.add((block_id, node_id, reason))

    def all_for(node: NodeRecord, reason: str) -> None:
        for block in node.blocks:
            add(block.block_id, node.node_id, reason)

    if old is None:
        for node in new.nodes:
            all_for(node, "initial")
    elif old.group_id != new.group_id:
        for node in old.nodes:
            all_for(node, "group_replaced")
        for node in new.nodes:
            all_for(node, "group_replaced")
    else:
        old_nodes = old.node_map()
        new_nodes = new.node_map()
        for node_id in sorted(set(old_nodes) | set(new_nodes)):
            old_node = old_nodes.get(node_id)
            new_node = new_nodes.get(node_id)
            if old_node is None and new_node is not None:
                all_for(new_node, "node_added")
                continue
            if new_node is None and old_node is not None:
                all_for(old_node, "node_removed")
                continue
            assert old_node is not None and new_node is not None
            if old_node.priority != new_node.priority:
                all_for(new_node, "node_priority")
            old_blocks = old_node.block_map()
            new_blocks = new_node.block_map()
            for block_id in sorted(set(old_blocks) | set(new_blocks)):
                old_block = old_blocks.get(block_id)
                new_block = new_blocks.get(block_id)
                if old_block is None and new_block is not None:
                    add(block_id, node_id, "block_added")
                    continue
                if new_block is None and old_block is not None:
                    add(block_id, node_id, "block_removed")
                    continue
                assert old_block is not None and new_block is not None
                for field in (
                    "size_bytes",
                    "digest",
                    "repair_weight",
                    "repair_source",
                    "checksum_derivation",
                ):
                    if getattr(old_block, field) != getattr(new_block, field):
                        add(block_id, node_id, f"block_{field}")
                if old_block.checksum_derivation != new_block.checksum_derivation:
                    sources = set()
                    if old_block.checksum_derivation is not None:
                        sources.update(old_block.checksum_derivation.source_block_ids)
                    if new_block.checksum_derivation is not None:
                        sources.update(new_block.checksum_derivation.source_block_ids)
                    for source_id in sorted(sources):
                        add(source_id, node_id, "checksum_derivation_source")

        if old.manifest_version != new.manifest_version and not details:
            for node in new.nodes:
                all_for(node, "manifest_version")

    ordered = tuple(
        {"block_id": block_id, "node_id": node_id, "reason": reason}
        for block_id, node_id, reason in sorted(details)
    )
    affected_blocks = tuple(sorted({item[0] for item in details}))
    affected_nodes = tuple(sorted({item[1] for item in details}))
    return ManifestChange(
        affected_blocks=affected_blocks,
        affected_nodes=affected_nodes,
        details=ordered,
    )


def node_blocks_dir(root: Path | str, node_id: str) -> Path:
    return Path(root) / "nodes" / node_id / "blocks"


def block_file(root: Path | str, node_id: str, block_id: str) -> Path:
    return node_blocks_dir(root, node_id) / f"{block_id}.bin"


def compute_state_digest(root: Path | str, manifest: Manifest) -> str:
    """Hash manifest semantics plus persisted block bytes, never mtimes."""

    disk_nodes: list[dict[str, Any]] = []
    for node in sorted(manifest.nodes, key=lambda item: item.node_id):
        blocks: list[dict[str, Any]] = []
        for block in sorted(node.blocks, key=lambda item: item.block_id):
            path = block_file(root, node.node_id, block.block_id)
            if path.is_file():
                blocks.append(
                    {
                        "block_id": block.block_id,
                        "present": True,
                        "size_bytes": path.stat().st_size,
                        "digest": sha256_file(path),
                    }
                )
            else:
                blocks.append(
                    {
                        "block_id": block.block_id,
                        "present": False,
                        "size_bytes": None,
                        "digest": None,
                    }
                )
        disk_nodes.append({"node_id": node.node_id, "blocks": blocks})
    payload = {
        "manifest": manifest.to_dict(),
        "disk": {"nodes": disk_nodes},
    }
    return sha256_bytes(canonical_json_dumps(payload).encode("utf-8"))
