"""Manifest loading, v1/v2 -> v3 upgrades, and resumable checkpoints."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .errors import BlockError, FormatError, StateError
from .model import (
    CURRENT_MANIFEST_VERSION,
    BlockRecord,
    Manifest,
    NodeRecord,
    validate_digest,
)
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    is_readonly,
    make_readonly,
    sha256_bytes,
    sha256_file,
)


CHECKPOINT_VERSION = 1
DigestResolver = Callable[[str, str, int], str]
AfterRecordHook = Callable[[int], None]


class FileDigestResolver:
    """Resolve a v1 block digest from its persisted node block file."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def __call__(self, node_id: str, block_id: str, size_bytes: int) -> str:
        path = self.root / "nodes" / node_id / "blocks" / f"{block_id}.bin"
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise BlockError(
                "cannot resolve v1 digest because block data is unavailable",
                path=f"$.nodes[{node_id}].blocks[{block_id}]",
                details={"node_id": node_id, "block_id": block_id, "file": str(path)},
            ) from exc
        if len(data) != size_bytes:
            raise BlockError(
                "block data size does not match manifest size_bytes",
                path=f"$.nodes[{node_id}].blocks[{block_id}].size_bytes",
                details={
                    "node_id": node_id,
                    "block_id": block_id,
                    "expected": size_bytes,
                    "actual": len(data),
                },
            )
        return sha256_bytes(data)


@dataclass(frozen=True)
class UpgradeResult:
    manifest: Manifest
    source_version: int
    target_version: int
    upgraded: bool
    resumed: bool
    processed_now: int
    skipped_committed: int
    backup_path: str | None
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "upgraded": self.upgraded,
            "resumed": self.resumed,
            "processed_now": self.processed_now,
            "skipped_committed": self.skipped_committed,
            "backup_path": self.backup_path,
            "checkpoint_path": self.checkpoint_path,
            "manifest_version": self.manifest.manifest_version,
        }


class ManifestStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def default_resolver(self) -> DigestResolver:
        return FileDigestResolver(self.root)

    def load_current(
        self,
        path: Path | str,
        *,
        upgrade: bool = True,
        resolver: DigestResolver | None = None,
        after_record: AfterRecordHook | None = None,
    ) -> UpgradeResult:
        path = Path(path)
        source = Manifest.from_file(path)
        checkpoint_path = self.checkpoint_path(path)
        if source.format_version == CURRENT_MANIFEST_VERSION:
            if checkpoint_path.exists():
                checkpoint = self._read_checkpoint(checkpoint_path)
                if checkpoint.get("phase") != "complete":
                    return self.upgrade_in_place(
                        path,
                        resolver=resolver,
                        after_record=after_record,
                    )
            return UpgradeResult(
                manifest=source,
                source_version=source.format_version,
                target_version=CURRENT_MANIFEST_VERSION,
                upgraded=False,
                resumed=False,
                processed_now=0,
                skipped_committed=0,
                backup_path=None,
                checkpoint_path=None,
            )
        if not upgrade:
            raise FormatError(
                "manifest is not at the current format version",
                path="$.format_version",
                details={
                    "found": source.format_version,
                    "required": CURRENT_MANIFEST_VERSION,
                },
            )
        return self.upgrade_in_place(path, resolver=resolver, after_record=after_record)

    def upgrade_in_place(
        self,
        path: Path | str,
        *,
        resolver: DigestResolver | None = None,
        after_record: AfterRecordHook | None = None,
    ) -> UpgradeResult:
        path = Path(path)
        if not path.exists():
            raise FormatError("manifest file does not exist", path="$", details={"file": str(path)})
        target = Manifest.from_file(path)
        checkpoint_path = self.checkpoint_path(path)

        if checkpoint_path.exists():
            checkpoint = self._read_checkpoint(checkpoint_path)
            backup_path = Path(checkpoint["backup_path"])
            if target.format_version == CURRENT_MANIFEST_VERSION and checkpoint.get("phase") == "complete":
                # The checkpoint is an audit record. Later repairs/evolutions may
                # legitimately advance the current manifest beyond this hash.
                return UpgradeResult(
                    manifest=target,
                    source_version=int(checkpoint["source_format_version"]),
                    target_version=CURRENT_MANIFEST_VERSION,
                    upgraded=True,
                    resumed=True,
                    processed_now=0,
                    skipped_committed=len(checkpoint["processed_node_ids"]),
                    backup_path=str(backup_path) if backup_path.exists() else None,
                    checkpoint_path=str(checkpoint_path),
                )
            return self._resume_upgrade(path, checkpoint_path, backup_path, resolver, after_record)

        if target.format_version == CURRENT_MANIFEST_VERSION:
            return UpgradeResult(
                manifest=target,
                source_version=target.format_version,
                target_version=CURRENT_MANIFEST_VERSION,
                upgraded=False,
                resumed=False,
                processed_now=0,
                skipped_committed=0,
                backup_path=None,
                checkpoint_path=None,
            )

        backup_path = self.backup_path(path, target.format_version)
        self._create_readonly_backup(path, backup_path)
        checkpoint = self._new_checkpoint(path, backup_path, target)
        atomic_write_json(checkpoint_path, checkpoint)
        return self._complete_upgrade(
            path=path,
            backup_path=backup_path,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            source=target,
            resolver=resolver or self.default_resolver(),
            after_record=after_record,
            resumed=False,
        )

    def _resume_upgrade(
        self,
        path: Path,
        checkpoint_path: Path,
        backup_path: Path,
        resolver: DigestResolver | None,
        after_record: AfterRecordHook | None,
    ) -> UpgradeResult:
        checkpoint = self._read_checkpoint(checkpoint_path)
        if checkpoint.get("target_format_version") != CURRENT_MANIFEST_VERSION:
            raise FormatError(
                "checkpoint target version is unsupported",
                path="$.target_format_version",
                details={"checkpoint": str(checkpoint_path)},
            )
        if not backup_path.exists():
            raise StateError(
                "upgrade checkpoint exists but its read-only source backup is missing",
                path="$",
                details={"backup_path": str(backup_path), "checkpoint_path": str(checkpoint_path)},
            )
        self._verify_backup(backup_path, checkpoint)
        source = Manifest.from_file(backup_path)
        if source.group_id != checkpoint["group_id"]:
            raise StateError("checkpoint group does not match source backup", path="$.group_id")
        if checkpoint["source_format_version"] != source.format_version:
            raise StateError("checkpoint source version does not match source backup", path="$.format_version")

        current_target = Manifest.from_file(path)
        if current_target.format_version == CURRENT_MANIFEST_VERSION:
            if checkpoint.get("phase") == "complete":
                return UpgradeResult(
                    manifest=current_target,
                    source_version=source.format_version,
                    target_version=CURRENT_MANIFEST_VERSION,
                    upgraded=True,
                    resumed=True,
                    processed_now=0,
                    skipped_committed=len(checkpoint["processed_node_ids"]),
                    backup_path=str(backup_path),
                    checkpoint_path=str(checkpoint_path),
                )
            if len(checkpoint["processed_node_ids"]) != len(checkpoint["ordered_node_ids"]):
                raise StateError(
                    "target was already rewritten but checkpoint is incomplete; refusing ambiguous recovery",
                    path="$",
                    details={"checkpoint_path": str(checkpoint_path)},
                )
        elif sha256_file(path) != checkpoint["source_sha256"]:
            raise StateError(
                "target changed after upgrade checkpoint was created",
                path="$",
                details={"checkpoint_path": str(checkpoint_path)},
            )
        if current_target.format_version == CURRENT_MANIFEST_VERSION:
            if len(checkpoint["processed_node_ids"]) != len(checkpoint["ordered_node_ids"]):
                raise StateError(
                    "target is current but checkpoint has uncommitted records",
                    path="$",
                    details={"checkpoint_path": str(checkpoint_path)},
                )

        return self._complete_upgrade(
            path=path,
            backup_path=backup_path,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            source=source,
            resolver=resolver or self.default_resolver(),
            after_record=after_record,
            resumed=True,
        )

    def _complete_upgrade(
        self,
        *,
        path: Path,
        backup_path: Path,
        checkpoint_path: Path,
        checkpoint: dict[str, Any],
        source: Manifest,
        resolver: DigestResolver,
        after_record: AfterRecordHook | None,
        resumed: bool,
    ) -> UpgradeResult:
        ordered_ids = list(checkpoint["ordered_node_ids"])
        processed_ids = list(checkpoint["processed_node_ids"])
        records = {item["node_id"]: copy.deepcopy(item["record"]) for item in checkpoint["records"]}

        if ordered_ids != sorted(node.node_id for node in source.nodes):
            raise StateError("checkpoint record order does not match source manifest", path="$.nodes")
        if processed_ids != ordered_ids[: len(processed_ids)]:
            raise StateError("checkpoint processed records are not a committed prefix", path="$.nodes")
        if set(records) != set(processed_ids):
            raise StateError("checkpoint records do not match processed node ids", path="$.nodes")
        self._validate_partial_records(source, checkpoint, records)

        source_nodes = source.node_map()
        processed_now = 0
        for node_id in ordered_ids[len(processed_ids) :]:
            node = source_nodes[node_id]
            upgraded_node = self._upgrade_node(node, source.format_version, resolver)
            records[node_id] = upgraded_node.to_dict(CURRENT_MANIFEST_VERSION)
            processed_ids.append(node_id)
            checkpoint["records"] = [
                {"node_id": record_id, "record": records[record_id]}
                for record_id in processed_ids
            ]
            checkpoint["processed_node_ids"] = list(processed_ids)
            checkpoint["phase"] = "upgrading"
            atomic_write_json(checkpoint_path, checkpoint)
            processed_now += 1
            if after_record is not None:
                after_record(len(processed_ids))

        all_nodes: list[NodeRecord] = []
        for node_id in ordered_ids:
            record = records[node_id]
            node_manifest = Manifest.from_dict(
                {
                    "format_version": 3,
                    "group_id": source.group_id,
                    "manifest_version": checkpoint["new_manifest_version"],
                    "metadata": source.metadata,
                    "nodes": [record],
                }
            )
            all_nodes.append(node_manifest.nodes[0])

        upgraded_manifest = Manifest(
            format_version=CURRENT_MANIFEST_VERSION,
            group_id=source.group_id,
            manifest_version=int(checkpoint["new_manifest_version"]),
            nodes=tuple(sorted(all_nodes, key=lambda item: item.node_id)),
            metadata=copy.deepcopy(source.metadata),
        )
        # Enforce full-document validation before replacing the legacy file.
        upgraded_manifest = Manifest.from_dict(upgraded_manifest.to_dict())
        target_bytes = upgraded_manifest.to_json().encode("utf-8")
        atomic_write_bytes(path, target_bytes)

        checkpoint["phase"] = "complete"
        checkpoint["target_sha256"] = sha256_bytes(target_bytes)
        checkpoint["completed_manifest_version"] = upgraded_manifest.manifest_version
        atomic_write_json(checkpoint_path, checkpoint)

        return UpgradeResult(
            manifest=upgraded_manifest,
            source_version=source.format_version,
            target_version=CURRENT_MANIFEST_VERSION,
            upgraded=True,
            resumed=resumed,
            processed_now=processed_now,
            skipped_committed=len(processed_ids) - processed_now,
            backup_path=str(backup_path),
            checkpoint_path=str(checkpoint_path),
        )

    def _upgrade_node(self, node: NodeRecord, source_version: int, resolver: DigestResolver) -> NodeRecord:
        if node.priority is None:
            priority = 1
        else:
            priority = node.priority
        upgraded_blocks: list[BlockRecord] = []
        for block in node.blocks:
            if block.digest is None:
                digest = resolver(node.node_id, block.block_id, block.size_bytes)
                validate_digest(
                    digest,
                    f"$.nodes[{node.node_id}].blocks[{block.block_id}].digest",
                )
            else:
                digest = block.digest
            upgraded_blocks.append(
                replace(
                    block,
                    digest=digest,
                    repair_weight=block.repair_weight if source_version >= 2 else 1,
                    checksum_derivation=None if source_version < 3 else block.checksum_derivation,
                    repair_source="origin" if source_version < 3 else block.repair_source,
                )
            )
        return NodeRecord(
            node_id=node.node_id,
            priority=priority,
            blocks=tuple(sorted(upgraded_blocks, key=lambda item: item.block_id)),
        )

    def _new_checkpoint(self, path: Path, backup_path: Path, source: Manifest) -> dict[str, Any]:
        ordered_ids = sorted(node.node_id for node in source.nodes)
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "group_id": source.group_id,
            "source_format_version": source.format_version,
            "target_format_version": CURRENT_MANIFEST_VERSION,
            "source_sha256": sha256_file(path),
            "source_manifest_version": source.manifest_version,
            "new_manifest_version": source.manifest_version + 1,
            "backup_path": str(backup_path),
            "target_path": str(path),
            "ordered_node_ids": ordered_ids,
            "processed_node_ids": [],
            "records": [],
            "phase": "upgrading",
            "target_sha256": None,
        }

    def _validate_partial_records(
        self,
        source: Manifest,
        checkpoint: dict[str, Any],
        records: dict[str, dict[str, Any]],
    ) -> None:
        if not records:
            return
        ordered_ids = checkpoint["ordered_node_ids"]
        selected = [records[node_id] for node_id in ordered_ids if node_id in records]
        Manifest.from_dict(
            {
                "format_version": 3,
                "group_id": source.group_id,
                "manifest_version": int(checkpoint["new_manifest_version"]),
                "nodes": selected,
            }
        )

    def _create_readonly_backup(self, path: Path, backup_path: Path) -> None:
        if backup_path.exists():
            self._verify_backup(backup_path, {"source_sha256": sha256_file(path)})
            if not is_readonly(backup_path):
                make_readonly(backup_path)
            return
        data = path.read_bytes()
        atomic_write_bytes(backup_path, data)
        make_readonly(backup_path)

    def _verify_backup(self, backup_path: Path, checkpoint: dict[str, Any]) -> None:
        actual = sha256_file(backup_path)
        if actual != checkpoint["source_sha256"]:
            raise StateError(
                "read-only source backup content changed",
                path="$",
                details={
                    "backup_path": str(backup_path),
                    "expected": checkpoint["source_sha256"],
                    "actual": actual,
                },
            )
        if not is_readonly(backup_path):
            raise StateError(
                "source backup must remain read-only",
                path="$",
                details={"backup_path": str(backup_path)},
            )

    def _verify_completed_target(self, path: Path, checkpoint: dict[str, Any]) -> None:
        expected = checkpoint.get("target_sha256")
        actual = sha256_file(path)
        if expected is None or actual != expected:
            raise StateError(
                "completed upgrade target no longer matches checkpoint",
                path="$",
                details={"expected": expected, "actual": actual},
            )

    @staticmethod
    def backup_path(path: Path, source_version: int) -> Path:
        return path.with_name(f"{path.name}.v{source_version}.readonly.bak")

    @staticmethod
    def checkpoint_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.upgrade.checkpoint.json")

    @staticmethod
    def _read_checkpoint(path: Path) -> dict[str, Any]:
        import json

        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StateError(
                "cannot read upgrade checkpoint",
                path="$",
                details={"checkpoint_path": str(path)},
            ) from exc
        if not isinstance(value, dict):
            raise StateError("upgrade checkpoint must be an object", path="$")
        if value.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise StateError(
                "unsupported upgrade checkpoint version",
                path="$.checkpoint_version",
                details={"checkpoint_path": str(path)},
            )
        required = {
            "group_id",
            "source_format_version",
            "target_format_version",
            "source_sha256",
            "new_manifest_version",
            "ordered_node_ids",
            "processed_node_ids",
            "records",
            "phase",
        }
        missing = sorted(required - set(value))
        if missing:
            raise StateError(
                "upgrade checkpoint is missing required fields",
                path=f"$.{missing[0]}",
                details={"checkpoint_path": str(path)},
            )
        return value


def crash_process_after(committed_records: int, crash_after: int) -> None:
    """Used by the CLI's deterministic forced-termination demo hook."""

    if crash_after > 0 and committed_records >= crash_after:
        os._exit(86)
