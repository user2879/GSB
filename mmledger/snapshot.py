"""Immutable staged snapshots with an atomically replaced current pointer."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .jsonutil import atomic_write_json, pretty_dumps


POINTER_NAME = "CURRENT.json"
STAGES = ("parse", "ownership", "resources", "ledger")


@dataclass(frozen=True)
class CurrentSnapshot:
    global_version: int
    generation_name: str
    generation_path: Path
    manifest: Mapping[str, Any]

    def stage_path(self, stage: str) -> Path:
        return self.generation_path / f"{stage}.json"

    def load_stage(self, stage: str) -> Any:
        path = self.stage_path(stage)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def cache(self) -> Mapping[str, Any]:
        path = self.generation_path / "cache.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.generations = self.root / "generations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations.mkdir(parents=True, exist_ok=True)

    @property
    def pointer_path(self) -> Path:
        return self.root / POINTER_NAME

    def current(self) -> CurrentSnapshot | None:
        if not self.pointer_path.exists():
            return None
        try:
            pointer = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        generation_name = pointer.get("generation")
        if not isinstance(generation_name, str):
            return None
        generation_path = self.generations / generation_name
        manifest_path = generation_path / "manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        version = manifest.get("global_version")
        if not isinstance(version, int) or isinstance(version, bool):
            return None
        return CurrentSnapshot(version, generation_name, generation_path, manifest)

    def next_version(self) -> int:
        current = self.current()
        versions = []
        for path in self.generations.iterdir():
            match = re.fullmatch(r"g(\d{8})", path.name)
            if match and path.is_dir():
                versions.append(int(match.group(1)))
        highest = max(versions, default=0)
        if current is not None:
            highest = max(highest, current.global_version)
        return highest + 1

    def staging(self) -> Path:
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=str(self.root)))

    def write_stage(self, staging: Path, stage: str, payload: Any) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown snapshot stage {stage}")
        stage_dir = staging / f".{stage}.tmp"
        stage_dir.mkdir(parents=True, exist_ok=False)
        stage_file = stage_dir / "payload.json"
        stage_file.write_text(pretty_dumps(payload), encoding="utf-8")
        final_dir = staging / stage
        os.replace(stage_dir, final_dir)
        return final_dir / "payload.json"

    def write_support_file(self, staging: Path, name: str, payload: Any) -> Path:
        if "/" in name or "\\" in name or not name:
            raise ValueError(f"invalid support file {name}")
        temp = staging / f".{name}.tmp"
        temp.write_text(pretty_dumps(payload), encoding="utf-8")
        final = staging / name
        os.replace(temp, final)
        return final

    def commit(self, staging: Path, *, global_version: int, manifest: Mapping[str, Any]) -> CurrentSnapshot:
        generation_name = f"g{global_version:08d}"
        generation_path = self.generations / generation_name
        if generation_path.exists():
            raise FileExistsError(f"snapshot generation already exists: {generation_path}")
        manifest_payload = dict(manifest)
        manifest_payload["global_version"] = global_version
        manifest_payload["generation"] = generation_name
        manifest_payload["completed_stages"] = list(STAGES)
        self.write_support_file(staging, "manifest.json", manifest_payload)
        os.replace(staging, generation_path)
        pointer = {"generation": generation_name, "global_version": global_version}
        atomic_write_json(self.pointer_path, pointer)
        return CurrentSnapshot(global_version, generation_name, generation_path, manifest_payload)

    def discard(self, staging: Path) -> None:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def restore_decision(
    current: CurrentSnapshot | None,
    *,
    config_hash: str,
    receipt_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    if current is None:
        return {
            "decision": "cold_full_rebuild",
            "reuse_snapshot": False,
            "reason": "no_current_snapshot",
        }
    manifest = current.manifest
    reasons = []
    if manifest.get("config_hash") != config_hash:
        reasons.append("config_hash_changed")
    if dict(manifest.get("receipt_fingerprints", {})) != dict(sorted(receipt_fingerprints.items())):
        reasons.append("receipt_fingerprints_changed")
    if manifest.get("pending_batches"):
        reasons.append("pending_batches_exist")
    if not reasons:
        return {
            "decision": "identical_snapshot_reuse",
            "reuse_snapshot": True,
            "reason": "config_hash_and_receipt_fingerprints_match",
            "global_version": current.global_version,
        }
    return {
        "decision": "incremental_dependency_rebuild",
        "reuse_snapshot": False,
        "reason": ",".join(reasons),
        "global_version": current.global_version,
    }
