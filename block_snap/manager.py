"""Snapshot manager: CoW snapshots, manifest digest and validation entry point.

This module owns the domain rules; :mod:`block_snap.rollback` owns the three
rollback execution paths and crash recovery.  All metadata changes happen in
explicit SQLite transactions and all failures flow through :class:`DomainError`.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Optional

from .bitmap import Bitmap
from .errors import DomainError, StructuredLogger, E_CHAIN, E_INPUT, E_INTEGRITY, E_SHAPE
from .storage import BlockStore, MetadataStore
from .volume import Volume, block_digest

MANIFEST_FORMAT_VERSION = 2
DEFAULT_BITMAP_THRESHOLD = 0.25
DEFAULT_CHAIN_SPAN = 2


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def manifest_digest(rows: list[Any], refs_by_snapshot: dict[str, dict[int, str]]) -> str:
    """Re-derive the manifest content digest from entries in stable order."""
    hasher = hashlib.blake2b(digest_size=32, person=b"manifest")
    for row in sorted(rows, key=lambda r: int(r["version_no"])):
        hasher.update(str(row["snapshot_id"]).encode())
        hasher.update(b"|")
        hasher.update(str(row["name"]).encode())
        hasher.update(b"|")
        hasher.update(str(row["version_no"]).encode())
        hasher.update(b"|")
        hasher.update(str(row["parent_id"] or "").encode())
        hasher.update(b"|")
        hasher.update(bytes(row["bitmap"]))
        hasher.update(b"|refs|")
        refs = refs_by_snapshot.get(row["snapshot_id"], {})
        for block_no in sorted(refs):
            hasher.update(f"{block_no}:{refs[block_no]};".encode())
        hasher.update(b"#")
    return hasher.hexdigest()


class SnapshotManager:
    """High-level package API bound to one workspace directory."""

    def __init__(self, workspace: str, log_path: Optional[str] = None) -> None:
        self.workspace = os.path.abspath(workspace)
        self.volumes_dir = os.path.join(self.workspace, "volumes")
        self.blocks_dir = os.path.join(self.workspace, "blocks")
        self.tmp_dir = os.path.join(self.workspace, "tmp")
        self.backup_dir = os.path.join(self.workspace, "backups")
        self.meta_dir = os.path.join(self.workspace, "meta")
        for directory in (
            self.volumes_dir,
            self.blocks_dir,
            self.tmp_dir,
            self.backup_dir,
            self.meta_dir,
        ):
            os.makedirs(directory, exist_ok=True)
        self.store = MetadataStore(os.path.join(self.meta_dir, "block_snap.sqlite"))
        self.blocks = BlockStore(self.blocks_dir)
        self.logger = StructuredLogger(
            log_path or os.path.join(self.meta_dir, "events.jsonl")
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "SnapshotManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- volume lifecycle ----------------------------------------------
    def load_volume(self, name: str, block_size: int, block_count: int, seed: str = "vol") -> Volume:
        """Register (or re-validate) a volume and verify its image shape."""
        path = os.path.join(self.volumes_dir, f"{name}.img")
        volume = Volume(path, block_size, block_count, seed)
        volume.validate_shape()
        self.store.upsert_volume(name, path, block_size, block_count, seed, now_iso())
        self.store.commit()
        self.logger.log("INFO", "volume_loaded", manifest_version=self._manifest_version(name),
                        **{"block_count": block_count, "block_size": block_size})
        return volume

    def create_volume(self, name: str, block_size: int, block_count: int, seed: str = "vol") -> Volume:
        """Create a deterministic baseline image and register the volume."""
        path = os.path.join(self.volumes_dir, f"{name}.img")
        volume = Volume(path, block_size, block_count, seed)
        volume.create_initialized()
        self.store.upsert_volume(name, path, block_size, block_count, seed, now_iso())
        if self.store.get_manifest(name) is None:
            self.store.put_manifest(name, MANIFEST_FORMAT_VERSION, manifest_digest([], {}), now_iso())
        self.store.replace_current_state(name, volume.block_digests())
        self.store.commit()
        self.logger.log("INFO", "volume_created", **{"block_count": block_count, "block_size": block_size})
        return volume

    def _require_volume_row(self, name: str) -> Any:
        row = self.store.get_volume(name)
        if row is None:
            raise DomainError(E_INPUT, f"unknown volume {name!r}", field="volume_name")
        return row

    def open_volume(self, name: str) -> Volume:
        row = self._require_volume_row(name)
        volume = Volume(row["path"], int(row["block_size"]), int(row["block_count"]), row["seed"])
        volume.validate_shape()
        return volume

    def _manifest_version(self, name: str) -> Optional[int]:
        manifest = self.store.get_manifest(name)
        return int(manifest["format_version"]) if manifest else None

    # -- snapshot creation ---------------------------------------------
    def create_snapshot(self, volume_name: str, snapshot_name: str) -> dict[str, Any]:
        row = self._require_volume_row(volume_name)
        block_count = int(row["block_count"])
        block_size = int(row["block_size"])
        volume = self.open_volume(volume_name)
        if self.store.snapshot_exists(volume_name, snapshot_name):
            raise DomainError(
                E_INPUT,
                f"snapshot name {snapshot_name!r} already exists",
                field="snapshot_name",
                manifest_version=self._manifest_version(volume_name),
            )
        live_digests = volume.block_digests()
        previous = self.store.current_map(volume_name)
        if previous:
            changed = sorted(
                block_no
                for block_no, digest in live_digests.items()
                if previous.get(block_no) != digest
            )
        else:
            # First snapshot establishes the full initial block state while
            # keeping an empty "changed since snapshot" bitmap.
            changed = []
        bitmap = Bitmap.from_blocks(block_count, changed)
        if len(bitmap.to_bytes()) != (block_count + 7) // 8:
            raise DomainError(E_INPUT, "bitmap length incompatible with block count", field="bitmap")

        snapshots = self.store.list_snapshots(volume_name)
        version_no = self.store.max_version(volume_name) + 1
        parent_id = snapshots[-1]["snapshot_id"] if snapshots else None
        snapshot_id = f"s{version_no:06d}-{uuid.uuid4().hex[:8]}"

        stored_blocks = changed if snapshots else list(range(block_count))
        for block_no in stored_blocks:
            block_hash = self.blocks.put(volume.read_block(block_no))
            self.store.insert_ref(volume_name, snapshot_id, block_no, block_hash)
        self.store.insert_snapshot(
            volume_name, snapshot_id, snapshot_name, version_no, parent_id,
            bitmap.to_bytes(), now_iso(),
        )
        self.store.patch_current_state(volume_name, live_digests)
        self._refresh_manifest(volume_name)
        self.store.commit()
        self.logger.log(
            "INFO", "snapshot_created", snapshot_id=snapshot_id,
            manifest_version=self._manifest_version(volume_name),
            **{"snapshot_name": snapshot_name, "version_no": version_no,
               "changed_blocks": len(changed), "bitmap_bytes": len(bitmap.to_bytes())},
        )
        return {
            "snapshot_id": snapshot_id,
            "name": snapshot_name,
            "version_no": version_no,
            "parent_id": parent_id,
            "changed_blocks": len(changed),
            "bitmap": bitmap.to_bytes().hex(),
        }

    def _refresh_manifest(self, volume_name: str) -> None:
        rows = self.store.list_snapshots(volume_name)
        refs = {row["snapshot_id"]: self.store.ref_map(volume_name, row["snapshot_id"]) for row in rows}
        digest = manifest_digest(rows, refs)
        manifest = self.store.get_manifest(volume_name)
        version = int(manifest["format_version"]) if manifest else MANIFEST_FORMAT_VERSION
        self.store.put_manifest(volume_name, version, digest, now_iso())

    def list_snapshots(self, volume_name: str) -> list[dict[str, Any]]:
        self._require_volume_row(volume_name)
        result = []
        for row in self.store.list_snapshots(volume_name):
            bitmap = Bitmap(int(self._require_volume_row(volume_name)["block_count"]), row["bitmap"])
            refs = self.store.ref_map(volume_name, row["snapshot_id"])
            result.append({
                "snapshot_id": row["snapshot_id"],
                "name": row["name"],
                "version_no": int(row["version_no"]),
                "parent_id": row["parent_id"],
                "changed_blocks": bitmap.count(),
                "bitmap_ratio": round(bitmap.ratio(), 4),
                "ref_count": len(refs),
                "created_at": row["created_at"],
            })
        return result

    # -- chain reconstruction and validation ---------------------------
    def validate_chain(self, volume_name: str) -> None:
        rows = self.store.list_snapshots(volume_name)
        ids = {row["snapshot_id"] for row in rows}
        for row in rows:
            parent = row["parent_id"]
            if parent is not None and parent not in ids:
                err = DomainError(
                    E_CHAIN,
                    f"snapshot {row['snapshot_id']} references missing parent {parent}",
                    snapshot_id=row["snapshot_id"],
                    manifest_version=self._manifest_version(volume_name),
                    extra={"missing_parent": parent},
                )
                self.logger.log_error(err)
                raise err

    def reconstruct_state(self, volume_name: str, snapshot_id: str) -> dict[int, bytes]:
        """Replay the chain from the base snapshot to build full block state."""
        self.validate_chain(volume_name)
        rows = self.store.list_snapshots(volume_name)
        target_version: Optional[int] = None
        for row in rows:
            if row["snapshot_id"] == snapshot_id:
                target_version = int(row["version_no"])
        if target_version is None:
            raise DomainError(
                E_CHAIN, f"unknown snapshot {snapshot_id}", snapshot_id=snapshot_id,
                manifest_version=self._manifest_version(volume_name),
            )
        state: dict[int, bytes] = {}
        for row in rows[:target_version]:
            refs = self.store.ref_map(volume_name, row["snapshot_id"])
            for block_no, block_hash in refs.items():
                state[block_no] = self.blocks.get(block_hash)
        return state

    def verify_block_integrity(self, volume_name: str, snapshot_id: str) -> None:
        """Compare every referenced block object digest with its registered digest."""
        refs = self.store.ref_map(volume_name, snapshot_id)
        for block_no in sorted(refs):
            block_hash = refs[block_no]
            if not self.blocks.exists(block_hash):
                err = DomainError(
                    E_CHAIN,
                    f"block object missing for snapshot {snapshot_id}",
                    snapshot_id=snapshot_id, block_no=block_no,
                    manifest_version=self._manifest_version(volume_name),
                )
                self.logger.log_error(err)
                raise err
            actual = block_digest(self.blocks.get(block_hash))
            if actual != block_hash:
                err = DomainError(
                    E_INTEGRITY,
                    f"block {block_no} digest mismatch",
                    snapshot_id=snapshot_id, block_no=block_no,
                    manifest_version=self._manifest_version(volume_name),
                    extra={"expected": block_hash, "actual": actual},
                )
                self.logger.log_error(err)
                raise err

    def verify_volume_shape(self, volume_name: str) -> Volume:
        try:
            return self.open_volume(volume_name)
        except DomainError as err:
            err.manifest_version = self._manifest_version(volume_name)
            self.logger.log_error(err)
            raise

    def manifest_view(self, volume_name: str) -> dict[str, Any]:
        self._require_volume_row(volume_name)
        manifest = self.store.get_manifest(volume_name)
        rows = self.store.list_snapshots(volume_name)
        refs = {row["snapshot_id"]: self.store.ref_map(volume_name, row["snapshot_id"]) for row in rows}
        recomputed = manifest_digest(rows, refs)
        return {
            "format_version": int(manifest["format_version"]) if manifest else None,
            "stored_digest": manifest["content_digest"] if manifest else None,
            "recomputed_digest": recomputed,
            "digest_match": bool(manifest and manifest["content_digest"] == recomputed),
            "entry_count": len(rows),
        }

    def query_logs(self, **filters: Any) -> list[dict[str, Any]]:
        return self.logger.query(**filters)
