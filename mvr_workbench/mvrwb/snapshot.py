"""阶段化快照：临时目录写入 + 整体替换提交 + 清单版本判定与恢复。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .errors import E_SNAPSHOT_STALE, StageError
from .util import commit_directory, content_fingerprint, read_json, write_json_atomic

SNAPSHOT_VERSION = 3
STAGE_NAMES = ("feature", "matching", "depth", "cloud")


@dataclass
class SnapshotHandle:
    root: str
    tmp_root: str

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    def stage_dir(self, stage: str) -> str:
        return os.path.join(self.root, stage)

    def stage_tmp(self, stage: str) -> str:
        return os.path.join(self.tmp_root, f"{stage}.tmp")

    def artifact_path(self, stage: str, artifact_id: str) -> str:
        return os.path.join(self.stage_dir(stage), f"{artifact_id}.json")

    def tmp_artifact_path(self, stage: str, artifact_id: str) -> str:
        return os.path.join(self.stage_tmp(stage), f"{artifact_id}.json")


class SnapshotStore:
    def __init__(self, root: str) -> None:
        self.handle = SnapshotHandle(root=root, tmp_root=os.path.join(root, "tmp"))
        os.makedirs(root, exist_ok=True)
        os.makedirs(self.handle.tmp_root, exist_ok=True)

    # -- 清单 ---------------------------------------------------------------

    def exists(self) -> bool:
        return os.path.exists(self.handle.manifest_path)

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        payload = dict(manifest)
        payload["manifest_fingerprint"] = content_fingerprint(
            {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
        )
        write_json_atomic(self.handle.manifest_path, payload)

    def read_manifest(self) -> dict[str, Any]:
        return read_json(self.handle.manifest_path)

    def check_resumable(self, config_fingerprint: str, camera_fingerprint: str) -> dict[str, Any]:
        """启动时以快照版本号判定续跑资格。

        配置哈希仅作为清单信息记录：阈值 / 体素等配置变更通过各阶段的
        参数指纹走精确失效复用，而不是整体判陈旧；只有快照 schema 版本
        不被当前代码支持时才输出 E_SNAPSHOT_STALE。相机集哈希不一致
        （视图增删 / 相机更换）同样由缓存键精确失效处理。
        """
        if not self.exists():
            return {"resumable": False, "reason": "no-snapshot", "recovery_view_id": None}
        manifest = self.read_manifest()
        if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
            stale = StageError(
                E_SNAPSHOT_STALE,
                f"快照版本 {manifest.get('snapshot_version')} 与当前版本 {SNAPSHOT_VERSION} 不一致",
                stage="snapshot",
            )
            stale.recovery_view_id = manifest.get("recovery_view_id")
            raise stale
        return {
            "resumable": True,
            "reason": "resume",
            "recovery_view_id": manifest.get("recovery_view_id"),
            "manifest": manifest,
            "config_changed": manifest.get("config_fingerprint") != config_fingerprint,
            "cameras_changed": manifest.get("camera_bundle_fingerprint") != camera_fingerprint,
        }

    # -- 阶段产物提交 -------------------------------------------------------

    def stage_artifact(self, artifact: dict[str, Any]) -> None:
        stage = artifact["stage"]
        path = self.handle.tmp_artifact_path(stage, artifact["artifact_id"])
        write_json_atomic(path, artifact)

    def commit_stage(self, stage: str) -> None:
        tmp_dir = self.handle.stage_tmp(stage)
        os.makedirs(tmp_dir, exist_ok=True)
        target_dir = self.handle.stage_dir(stage)
        commit_directory(tmp_dir, target_dir)
        os.makedirs(tmp_dir, exist_ok=True)

    def load_stage(self, stage: str) -> list[dict[str, Any]]:
        directory = self.handle.stage_dir(stage)
        if not os.path.isdir(directory):
            return []
        artifacts = []
        for name in sorted(os.listdir(directory)):
            if name.endswith(".json"):
                artifacts.append(read_json(os.path.join(directory, name)))
        return artifacts

    def load_artifact_map(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for stage in STAGE_NAMES:
            for artifact in self.load_stage(stage):
                result[artifact["artifact_id"]] = artifact
        return result
