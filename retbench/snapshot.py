"""阶段化快照：临时目录写入 + 整体替换 + 清单版本判定。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from .errors import E_SNAPSHOT_STALE
from .ledger import STAGES

MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotManifest:
    global_version: int
    config_hash: str
    command_cursor: int
    stage_versions: dict[str, int]
    stage_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "command_cursor": self.command_cursor,
            "stage_versions": dict(self.stage_versions),
            "stage_hashes": dict(self.stage_hashes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotManifest":
        return cls(
            global_version=int(data["global_version"]),
            config_hash=str(data["config_hash"]),
            command_cursor=int(data["command_cursor"]),
            stage_versions={k: int(v) for k, v in data["stage_versions"].items()},
            stage_hashes={k: str(v) for k, v in data["stage_hashes"].items()},
        )


class SnapshotStore:
    def __init__(self, root: str, temp_dir: str, config_hash: str):
        self.root = root
        self.temp_dir = temp_dir
        self.config_hash = config_hash
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)

    def load_manifest(self) -> SnapshotManifest | None:
        if not os.path.isfile(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return SnapshotManifest.from_dict(json.load(handle))

    def resume_manifest(self) -> tuple[SnapshotManifest | None, str | None]:
        """启动判定：配置哈希一致才允许续跑，否则返回 E_SNAPSHOT_STALE。"""

        manifest = self.load_manifest()
        if manifest is None:
            return None, None
        if manifest.config_hash != self.config_hash:
            return None, E_SNAPSHOT_STALE
        return manifest, None

    def commit_stage(self, stage: str, payload: Any, current: SnapshotManifest,
                     changed: bool) -> tuple[SnapshotManifest, str]:
        """先写临时目录再 os.replace 整体替换；版本变化才替换，不逐文件覆盖。"""

        stage_dir = os.path.join(self.root, stage)
        stage_tmp = tempfile.mkdtemp(prefix=f"{stage}-", dir=self.temp_dir)
        file_path = os.path.join(stage_tmp, f"{stage}.json")
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(body)
        payload_hash = digest_text(body)

        new_manifest = SnapshotManifest(
            global_version=current.global_version + (1 if changed else 0),
            config_hash=self.config_hash,
            command_cursor=current.command_cursor,
            stage_versions=dict(current.stage_versions),
            stage_hashes=dict(current.stage_hashes),
        )
        if changed or current.stage_hashes.get(stage) != payload_hash:
            new_manifest.stage_versions[stage] = current.stage_versions.get(stage, 0) + 1
            os.makedirs(stage_dir, exist_ok=True)
            os.replace(file_path, os.path.join(stage_dir, f"{stage}.json"))
            new_manifest.stage_hashes[stage] = payload_hash
        shutil.rmtree(stage_tmp, ignore_errors=True)
        self._write_manifest_atomic(new_manifest)
        return new_manifest, payload_hash

    def update_cursor(self, manifest: SnapshotManifest, cursor: int) -> SnapshotManifest:
        updated = SnapshotManifest(
            global_version=manifest.global_version,
            config_hash=manifest.config_hash,
            command_cursor=cursor,
            stage_versions=dict(manifest.stage_versions),
            stage_hashes=dict(manifest.stage_hashes),
        )
        self._write_manifest_atomic(updated)
        return updated

    def _write_manifest_atomic(self, manifest: SnapshotManifest) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="manifest-", dir=self.temp_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, ensure_ascii=False,
                      sort_keys=True, indent=2)
        os.replace(tmp_path, self.manifest_path)

    def reset(self) -> None:
        """配置漂移导致 E_SNAPSHOT_STALE 后转入全量：清空旧产物。"""

        for stage in STAGES:
            shutil.rmtree(os.path.join(self.root, stage), ignore_errors=True)
        if os.path.isfile(self.manifest_path):
            os.remove(self.manifest_path)


from .fingerprint import digest_text


def initial_manifest(config_hash: str) -> SnapshotManifest:
    return SnapshotManifest(
        global_version=0,
        config_hash=config_hash,
        command_cursor=0,
        stage_versions={stage: 0 for stage in STAGES},
        stage_hashes={stage: "" for stage in STAGES},
    )
