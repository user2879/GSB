"""Staged, content-addressed snapshot store with whole-store replacement."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .config import canonical_json
from .errors import E_SNAPSHOT_MISMATCH, FailureLog

MANIFEST_NAME = "manifest.json"
GLOBAL_VERSION = 3


class SnapshotStore:
    """Stage products in ``tmp/`` then atomically publish them per stage.

    Artifacts are keyed by their content hash, so unchanged products are never
    rewritten.  A stage commit replaces the whole stage directory rather than
    touching individual files, and the manifest commit is the atomic marker.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tmp = self.root / "tmp"
        self.manifest_path = self.root / MANIFEST_NAME

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def stage_path(self, stage: str, name: str) -> Path:
        path = self.tmp / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def commit_stage(self, stage: str) -> None:
        staged = self.tmp / stage
        target = self.root / stage
        if not staged.exists():
            return
        live = self.root / (stage + ".new")
        if live.exists():
            shutil.rmtree(live)
        shutil.copytree(staged, live)
        if target.exists():
            shutil.rmtree(target)
        live.replace(target)
        shutil.rmtree(staged)

    def product_path(self, stage: str, key: str, suffix: str = ".json") -> Path:
        return self.root / stage / f"{key}{suffix}"

    def has_product(self, stage: str, key: str, suffix: str = ".json") -> bool:
        return self.product_path(stage, key, suffix).exists()

    def read_json(self, stage: str, key: str) -> dict:
        return json.loads(self.product_path(stage, key).read_text("utf-8"))

    def write_staged_json(self, stage: str, key: str, payload: dict) -> Path:
        path = self.stage_path(stage, f"{key}.json")
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return path

    def write_staged_bytes(self, stage: str, key: str, data: bytes) -> Path:
        path = self.stage_path(stage, f"{key}.pcm")
        path.write_bytes(data)
        return path

    def publish_manifest(self, manifest: dict) -> None:
        target = dict(manifest)
        target["global_version"] = GLOBAL_VERSION
        path = self.stage_path("manifest", MANIFEST_NAME)
        path.write_text(canonical_json(target) + "\n", encoding="utf-8")
        if self.manifest_path.exists():
            self.manifest_path.unlink()
        path.replace(self.manifest_path)
        shutil.rmtree(self.tmp / "manifest", ignore_errors=True)

    def read_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text("utf-8"))

    def verify_resume(self, config_fp: str, exchange_fp: str,
                      audio_fp: str, failures: FailureLog) -> dict | None:
        """Return the manifest when resumable, otherwise trigger a rebuild.

        Only the global schema version, config and source audio are hard
        resume gates.  An exchange-content change is an expected edit: it is
        recorded in the manifest but resolved through per-segment cache keys,
        so untouched segments keep their cached products.
        """
        manifest = self.read_manifest()
        if manifest is None:
            return None
        mismatches = []
        if manifest.get("global_version") != GLOBAL_VERSION:
            mismatches.append("global_version")
        if manifest.get("config_hash") != config_fp:
            mismatches.append("config_hash")
        if manifest.get("audio_fingerprint") != audio_fp:
            mismatches.append("audio_fingerprint")
        if mismatches:
            failures.add(E_SNAPSHOT_MISMATCH, "load",
                         field=",".join(mismatches),
                         message="snapshot mismatch; full rebuild required")
            self.reset()
            return None
        return manifest
