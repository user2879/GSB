"""Atomic staged snapshots and restart/resume decisions."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_INVALID, diagnostic
from .fixed import digest


SNAPSHOT_SCHEMA = 1
STAGE_FILES = {
    "features": "features.json",
    "match_pairs": "match_pairs.json",
    "points": "points.json",
    "depth_maps": "depth_maps.json",
    "voxels": "voxels.json",
    "ledger": "ledger.json",
    "pending": "pending.json",
    "budget": "budget.json",
    "strategy": "strategy.json",
    "adjacency": "adjacency.json",
}


def _snapshot_root(project_dir: str | Path) -> Path:
    return Path(project_dir) / ".mgirecon" / "current"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_versions(result: dict[str, Any]) -> dict[str, str]:
    return {
        "load": digest(result.get("validation", {})),
        "features": digest(result.get("features", {})),
        "matching": digest(result.get("match_pairs", [])),
        "triangulation": digest(result.get("points", [])),
        "depth": digest(result.get("depth_maps", {})),
        "voxel": digest({"voxels": result.get("voxels", []), "adjacency": result.get("adjacency", [])}),
        "ledger": digest(result.get("ledger", {})),
    }


def load_snapshot(project_dir: str | Path, config_hash: str | None = None) -> dict[str, Any]:
    current = _snapshot_root(project_dir)
    manifest_path = current / "manifest.json"
    if not manifest_path.exists():
        return {
            "state": {},
            "manifest": {},
            "resume_mode": "fresh",
            "errors": [],
        }
    try:
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported snapshot schema")
        state: dict[str, Any] = {"manifest": manifest}
        for key, filename in STAGE_FILES.items():
            path = current / filename
            if path.exists():
                state[key] = _read_json(path)
        if config_hash is None:
            resume_mode = "resume"
        elif manifest.get("config_hash") == config_hash:
            resume_mode = "resume"
        else:
            resume_mode = "incremental"
        return {"state": state, "manifest": manifest, "resume_mode": resume_mode, "errors": []}
    except Exception as exc:
        return {
            "state": {},
            "manifest": {},
            "resume_mode": "fresh",
            "errors": [
                diagnostic(
                    E_SNAPSHOT_INVALID,
                    f"snapshot cannot be resumed: {exc}",
                    stage="snapshot",
                    detail={"path": str(current)},
                )
            ],
        }


def write_snapshot(
    project_dir: str | Path,
    result: dict[str, Any],
    *,
    previous_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_dir).resolve()
    state_root = root / ".mgirecon"
    state_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="snapshot-tmp-", dir=state_root))
    previous_version = int((previous_manifest or {}).get("global_version", 0))
    global_version = previous_version + 1
    completed_views = sorted(result.get("features", {}))
    completed_voxels = sorted(voxel["path"] for voxel in result.get("voxels", []))
    stage_versions = _stage_versions(result)
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA,
        "global_version": global_version,
        "config_hash": result.get("config_hash"),
        "camera_hash": result.get("camera_hash"),
        "completed_views": completed_views,
        "completed_voxel_paths": completed_voxels,
        "budget_consumed": result.get("budget", {}).get("global_consumed", 0),
        "budget": result.get("budget", {}),
        "stage_versions": stage_versions,
        "pending_count": len(result.get("pending", [])),
        "error_codes": sorted({error["code"] for error in result.get("errors", [])}),
        "status": result.get("status"),
        "resume_rule": "schema+config_hash; changed config enters incremental path",
    }
    try:
        _write_json(temporary / "manifest.json", manifest)
        for key, filename in STAGE_FILES.items():
            _write_json(temporary / filename, result.get(key, {} if key in {"features", "depth_maps"} else []))
        current = _snapshot_root(root)
        backup = state_root / "previous"
        if backup.exists():
            shutil.rmtree(backup)
        if current.exists():
            os.replace(current, backup)
        os.replace(temporary, current)
        if backup.exists():
            shutil.rmtree(backup)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
