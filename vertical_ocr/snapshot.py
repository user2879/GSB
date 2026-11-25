from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .ledger import atomic_write_json


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "snapshots"
        self.versions = self.root / "versions"
        self.current_pointer = self.root / "current.json"
        self.versions.mkdir(parents=True, exist_ok=True)

    def current(self) -> dict[str, Any] | None:
        if not self.current_pointer.exists():
            return None
        pointer = json.loads(self.current_pointer.read_text(encoding="utf-8"))
        manifest_path = Path(pointer["manifest_path"])
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def is_fresh(self, global_version: Any, config_hash: str) -> tuple[bool, str, dict[str, Any] | None]:
        current = self.current()
        if current is None:
            return False, "no_committed_snapshot", None
        if str(current.get("global_version")) != str(global_version):
            return False, "global_version_mismatch", current
        if current.get("config_hash") != config_hash:
            return False, "config_hash_mismatch", current
        return True, "fresh", current

    def commit(
        self,
        *,
        global_version: Any,
        config_hash: str,
        template_version: str,
        keypoint_fingerprint: str,
        completed_images: list[str],
        consumed_budget: int,
        pending: list[dict[str, Any]],
        stage_products: dict[str, Any],
        kill_before_switch: bool = False,
    ) -> dict[str, Any]:
        snapshot_id = f"snap-v{global_version}-{config_hash[:10]}-{os.getpid()}-{time.monotonic_ns()}"
        tmp = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(self.root)))
        try:
            for stage_name, product in stage_products.items():
                atomic_write_json(tmp / f"{stage_name}.json", product)
            manifest = {
                "snapshot_id": snapshot_id,
                "schema": "vertical-ocr-snapshot/v1",
                "global_version": str(global_version),
                "config_hash": config_hash,
                "template_version": template_version,
                "keypoint_fingerprint": keypoint_fingerprint,
                "completed_images": completed_images,
                "consumed_budget": consumed_budget,
                "pending_count": len(pending),
                "pending": pending,
                "stages": sorted(stage_products.keys()),
            }
            atomic_write_json(tmp / "manifest.json", manifest)
            if kill_before_switch:
                os.kill(os.getpid(), 9)
            final_dir = self.versions / snapshot_id
            os.replace(tmp, final_dir)
            atomic_write_json(self.current_pointer, {
                "snapshot_id": snapshot_id,
                "manifest_path": str(Path("versions") / snapshot_id / "manifest.json"),
            })
            return {**manifest, "manifest_path": str(final_dir / "manifest.json")}
        finally:
            if tmp.exists():
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    def recovery_summary(self) -> dict[str, Any]:
        current = self.current()
        if current is None:
            return {"recoverable": False, "scope": "no committed snapshot; no completed stage can be promised"}
        return {
            "recoverable": True,
            "snapshot_id": current["snapshot_id"],
            "global_version": current["global_version"],
            "config_hash": current["config_hash"],
            "completed_images": current.get("completed_images", []),
            "consumed_budget": current.get("consumed_budget", 0),
            "pending": current.get("pending", []),
            "scope": "only atomically committed stage products are recoverable; a force-killed run has no exit report",
        }
