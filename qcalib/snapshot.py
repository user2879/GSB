"""Stage-scoped atomic snapshots with whole-directory replacement and recovery."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .hashing import stable_json

STAGES = ("parse_validate", "quantile_calibration", "fixed_point_requantization",
          "distillation_correction", "ledger_aggregation")


class ForcedAbort(RuntimeError):
    """Raised by a test fault point to simulate a hard process termination."""


class SnapshotStore:
    def __init__(self, root: str | Path,
                 fault_point: str | None = None) -> None:
        self.root = Path(root)
        self.tmp_root = self.root / "tmp"
        self.stage_root = self.root / "stages"
        self.manifest_path = self.root / "manifest.json"
        self.fault_point = fault_point
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"stages": {}, "committed": []}

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        tmp_manifest = self.root / "manifest.json.tmp"
        tmp_manifest.write_text(stable_json(manifest) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, self.manifest_path)

    def maybe_abort(self, fault_point: str) -> None:
        if self.fault_point == fault_point:
            raise ForcedAbort(f"fault point triggered: {fault_point}")

    def commit_stage(self, stage: str, payload: dict[str, Any],
                     config_hash: str,
                     content_versions: dict[str, int]) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage: {stage}")
        self.maybe_abort(f"before_commit:{stage}")
        tmp_dir = self.tmp_root / stage
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        document = {
            "stage": stage,
            "config_hash": config_hash,
            "content_versions": content_versions,
            "payload": payload,
        }
        (tmp_dir / "data.json").write_text(stable_json(document) + "\n",
                                           encoding="utf-8")
        final_dir = self.stage_root / stage
        # Whole-directory atomic replacement on the same filesystem.
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)

        manifest = self.load_manifest()
        manifest["stages"][stage] = {
            "config_hash": config_hash,
            "content_versions": content_versions,
        }
        if stage not in manifest["committed"]:
            manifest["committed"].append(stage)
        manifest["committed"] = [s for s in STAGES if s in manifest["committed"]]
        self.save_manifest(manifest)
        self.maybe_abort(f"after_commit:{stage}")
        return manifest

    def load_stage(self, stage: str) -> dict[str, Any] | None:
        path = self.stage_root / stage / "data.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def recoverable_stages(self, config_hash: str,
                           content_versions: dict[str, int]) -> list[str]:
        manifest = self.load_manifest()
        recoverable: list[str] = []
        for stage in STAGES:
            meta = manifest.get("stages", {}).get(stage)
            data = self.load_stage(stage)
            if meta is None or data is None:
                continue
            if meta.get("config_hash") != config_hash:
                continue
            if meta.get("content_versions") != content_versions:
                continue
            recoverable.append(stage)
        return recoverable

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)
