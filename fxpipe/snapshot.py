"""Staged snapshots with temp-dir build + atomic whole-directory swap.

Only fully committed snapshots are recoverable. A run that is killed while a
stage is being written leaves an un-referenced temp directory; restart ignores
it and resumes from the last committed manifest.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fingerprint import config_hash, fingerprint
from .ledger import Ledger

MANIFEST_NAME = "manifest.json"
CURRENT_LINK = "current"
STAGE_FILES = ("stage_embed.json", "stage_pool.json", "stage_attn.json",
               "stage_norm.json", "stage_final.json")


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    exchange_version: str
    completed_ops: list[str]
    stage_versions: dict[str, int]
    consumed_points: int
    quota_spent: dict[str, int]
    pending: list[dict]
    stage_points: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp_root = self.root / "tmp"
        self.tmp_root.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    def current_dir(self) -> Path | None:
        link = self.root / CURRENT_LINK
        if link.is_symlink():
            target = os.readlink(link)
            target_path = Path(target)
            return target_path if target_path.is_absolute() else (self.root / target_path)
        if link.exists():
            return link
        return None

    def load_manifest(self) -> Manifest | None:
        current = self.current_dir()
        if current is None:
            return None
        path = Path(current) / MANIFEST_NAME
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(**data)

    def should_resume(self, config_dict: dict, exchange_version: str) -> Manifest | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        if manifest.config_hash != config_hash(config_dict):
            return None
        if manifest.exchange_version != exchange_version:
            return None
        return manifest

    # ------------------------------------------------------------------ #
    def commit(
        self,
        manifest: Manifest,
        ledger: Ledger,
        stage_payloads: dict[str, object],
    ) -> Path:
        version_dir = self.root / f"v{manifest.global_version}"
        staging = self.tmp_root / f"v{manifest.global_version}.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)

        ledger.save(staging / "ledger.json")
        for stage, payload in stage_payloads.items():
            (staging / f"stage_{stage}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        # Atomic whole-directory replacement on the local filesystem.
        if version_dir.exists():
            shutil.rmtree(version_dir)
        os.replace(staging, version_dir)

        link = self.root / CURRENT_LINK
        tmp_link = self.root / "current.new"
        if tmp_link.is_symlink() or tmp_link.exists():
            tmp_link.unlink()
        os.symlink(version_dir.name, tmp_link)
        os.replace(tmp_link, link)
        return version_dir

    def load_ledger(self) -> Ledger:
        current = self.current_dir()
        if current is None:
            return Ledger()
        return Ledger.load(Path(current) / "ledger.json")

    def prune_old_versions(self, keep: int = 3) -> None:
        versions = sorted(
            (path for path in self.root.glob("v*") if path.is_dir()),
            key=lambda path: int(path.name[1:]),
        )
        for stale in versions[:-keep]:
            shutil.rmtree(stale, ignore_errors=True)

    @staticmethod
    def new_manifest(config_dict: dict, exchange_version: str, global_version: int) -> Manifest:
        return Manifest(
            global_version=global_version,
            config_hash=config_hash(config_dict),
            exchange_version=exchange_version,
            completed_ops=[],
            stage_versions={},
            consumed_points=0,
            quota_spent={},
            pending=[],
            stage_points={},
        )

    def max_disk_version(self) -> int:
        versions = [int(path.name[1:]) for path in self.root.glob("v*")
                    if path.is_dir() and path.name[1:].isdigit()]
        return max(versions, default=0)

    def next_version(self, previous: Manifest | None) -> int:
        # Never reuse an existing on-disk version directory, even when the
        # in-memory manifest is absent after a config-hash rebuild.
        floor = self.max_disk_version() + 1
        if previous is None:
            return floor
        return max(floor, previous.global_version + 1)
