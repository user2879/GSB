from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .hashing import canonical_json


def read_current_manifest(root: str | Path) -> dict[str, Any] | None:
    current = Path(root) / "snapshots" / "current" / "manifest.json"
    if not current.exists():
        return None
    try:
        return json.loads(current.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_atomic_snapshot(
    root: str | Path,
    result: dict[str, Any],
    global_version: int,
    config_hash: str,
) -> Path:
    """Write all stage artifacts to a temp directory and atomically swap symlink."""
    snapshot_root = Path(root) / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-v{global_version:06d}-", dir=snapshot_root))

    stages = {
        "windows.json": result.get("windows", []),
        "components.json": result.get("components", []),
        "keypoints.json": result.get("keypoints", []),
        "reading.json": result.get("reading", []),
        "conflicts.json": result.get("conflicts", []),
        "queue_budget.json": {
            "pending_queue": result.get("pending_queue", []),
            "budget": result.get("budget", {}),
        },
        "failures.json": result.get("failures", []),
        "result.json": result,
    }
    for filename, payload in stages.items():
        (staging / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    manifest = {
        "global_version": global_version,
        "config_hash": config_hash,
        "previous_global_version": result.get("previous_global_version"),
        "version_changed": result.get("global_version") != result.get("previous_global_version"),
        "files": sorted(stages),
        "canonical_result_sha256": __import__("vertical_workbench.hashing", fromlist=["sha256_text"]).sha256_text(
            canonical_json(result)
        ),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    final_name = f"release-v{global_version:06d}-{config_hash[:12]}-{uuid.uuid4().hex[:8]}"
    final_path = snapshot_root / final_name
    os.replace(staging, final_path)

    link_path = snapshot_root / "current"
    tmp_link = snapshot_root / f".current-tmp-{global_version}-{os.getpid()}"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    os.symlink(final_path.name, tmp_link)
    os.replace(tmp_link, link_path)
    return link_path
