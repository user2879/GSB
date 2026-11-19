"""Single facade exposing the full package API used by CLI and Tkinter UI."""
from __future__ import annotations

import os
from typing import Any, Optional

from .manager import SnapshotManager
from .manifest import ManifestService
from .rollback import RollbackEngine
from .volume import Volume


class BlockSnapService:
    """Convenience facade; thin orchestration over the domain objects."""

    def __init__(self, workspace: str, log_path: Optional[str] = None) -> None:
        self.workspace = os.path.abspath(workspace)
        self.manager = SnapshotManager(workspace, log_path=log_path)
        self.rollback_engine = RollbackEngine(self.manager)
        self.manifests = ManifestService(self.manager)

    def close(self) -> None:
        self.manager.close()

    def __enter__(self) -> "BlockSnapService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # volume -----------------------------------------------------------
    def create_volume(self, name: str, block_size: int, block_count: int, seed: str = "vol") -> Volume:
        return self.manager.create_volume(name, block_size, block_count, seed)

    def load_volume(self, name: str, block_size: int, block_count: int, seed: str = "vol") -> Volume:
        return self.manager.load_volume(name, block_size, block_count, seed)

    def open_volume(self, name: str) -> Volume:
        return self.manager.open_volume(name)

    def verify_volume_shape(self, name: str) -> Volume:
        return self.manager.verify_volume_shape(name)

    def inject_changes(self, name: str, change_spec: str) -> list[int]:
        volume = self.manager.open_volume(name)
        changed = volume.inject_changes(change_spec)
        return changed

    # snapshots --------------------------------------------------------
    def create_snapshot(self, volume_name: str, snapshot_name: str) -> dict[str, Any]:
        return self.manager.create_snapshot(volume_name, snapshot_name)

    def list_snapshots(self, volume_name: str) -> list[dict[str, Any]]:
        return self.manager.list_snapshots(volume_name)

    def manifest_view(self, volume_name: str) -> dict[str, Any]:
        return self.manager.manifest_view(volume_name)

    def replace_manifest(self, volume_name: str, format_version: int, entries: list[dict[str, Any]],
                         claimed_digest: Optional[str] = None) -> dict[str, Any]:
        return self.manifests.replace_manifest(
            volume_name, format_version, entries, claimed_digest=claimed_digest
        )

    # rollback ---------------------------------------------------------
    def plan_rollback(self, volume_name: str, snapshot_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.rollback_engine.plan_path(volume_name, snapshot_id, **kwargs)

    def rollback(self, volume_name: str, snapshot_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.rollback_engine.rollback(volume_name, snapshot_id, **kwargs)

    def cross_check_paths(self, volume_name: str, snapshot_id: str) -> dict[str, Any]:
        return self.rollback_engine.cross_check_paths(volume_name, snapshot_id)

    def recover(self, volume_name: str, job_id: Optional[str] = None) -> dict[str, Any]:
        return self.rollback_engine.recover(volume_name, job_id)

    def completion_token(self, volume_name: str, snapshot_id: str) -> dict[str, Any]:
        return self.rollback_engine.completion_token(volume_name, snapshot_id)

    # audit & logs -----------------------------------------------------
    def audit(self, volume_name: str) -> dict[str, Any]:
        return self.manifests.audit(volume_name)

    def audit_verdict(self, volume_name: str) -> Optional[dict[str, Any]]:
        return self.manifests.audit_verdict(volume_name)

    def query_logs(self, **filters: Any) -> list[dict[str, Any]]:
        return self.manager.query_logs(**filters)
