"""阶段化快照整体替换、恢复判定（要求 8）。

提交采用"版本目录 + current 符号链接"原子切换：
- 各阶段产物先写入 snapshot_tmp/<version>/ 下的阶段临时目录；
- 全部写齐后整体替换为 snapshot_commit/versions/v<N>/，再以
  os.replace 原子切换 current 链接；绝不逐文件覆盖已提交产物；
- 启动比对全局版本号与配置哈希，不符则输出 E_SNAPSHOT_STALE 并隔离丢弃。
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List

from .errors import E_SNAPSHOT_STALE, Failure
from .serialization import content_hash, read_json, write_json

MANIFEST_NAME = "manifest.json"
CURRENT_LINK = "current"

STAGE_LEDGER = "ledger"
STAGE_FLIPS = "flips"
STAGE_BOUNDARY = "boundary"
STAGE_REPORT = "report"


@dataclass
class SnapshotPaths:
    tmp_root: str
    commit_root: str

    @property
    def versions_dir(self) -> str:
        return os.path.join(self.commit_root, "versions")

    @property
    def current_path(self) -> str:
        return os.path.join(self.commit_root, CURRENT_LINK)

    @property
    def stale_dir(self) -> str:
        return os.path.join(self.commit_root, "stale")


class SnapshotStore:
    def __init__(self, tmp_root: str, commit_root: str):
        self.paths = SnapshotPaths(tmp_root, commit_root)
        os.makedirs(self.paths.tmp_root, exist_ok=True)
        os.makedirs(self.paths.versions_dir, exist_ok=True)
        os.makedirs(self.paths.stale_dir, exist_ok=True)

    # ---------- 恢复判定 ----------
    def load_manifest(self) -> Dict[str, Any] | None:
        cur = self.paths.current_path
        if not os.path.islink(cur) and not os.path.exists(cur):
            return None
        manifest_path = os.path.join(cur, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            return None
        return read_json(manifest_path)

    def validate(self, expected_config_hash: str
                 ) -> tuple[Dict[str, Any] | None, List[Failure]]:
        manifest = self.load_manifest()
        if manifest is None:
            return None, []
        if manifest.get("config_hash") != expected_config_hash:
            failure = Failure(
                E_SNAPSHOT_STALE, self.paths.current_path,
                "配置哈希与快照不一致，丢弃旧产物并全量重建",
                {"manifest_config_hash": manifest.get("config_hash"),
                 "expected_config_hash": expected_config_hash})
            self._quarantine(manifest)
            return None, [failure]
        if not isinstance(manifest.get("global_version"), int):
            failure = Failure(
                E_SNAPSHOT_STALE, self.paths.current_path,
                "全局版本号缺失或非法，丢弃旧产物")
            self._quarantine(manifest)
            return None, [failure]
        return manifest, []

    def _quarantine(self, manifest: Dict[str, Any]) -> None:
        target = os.readlink(self.paths.current_path) \
            if os.path.islink(self.paths.current_path) else None
        if target:
            version_dir = os.path.join(self.paths.commit_root, target)
            if os.path.isdir(version_dir):
                dest = os.path.join(self.paths.stale_dir,
                                    os.path.basename(version_dir))
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.move(version_dir, dest)
            os.unlink(self.paths.current_path)

    # ---------- 临时写入与整体提交 ----------
    def new_version_dir(self, version: int) -> str:
        version_dir = os.path.join(self.paths.tmp_root, f"v{version}")
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        os.makedirs(version_dir, exist_ok=True)
        return version_dir

    def write_stage(self, version_dir: str, stage: str,
                    payload: Any) -> str:
        stage_dir = os.path.join(version_dir, stage)
        os.makedirs(stage_dir, exist_ok=True)
        path = os.path.join(stage_dir, "data.json")
        write_json(path, payload)
        return content_hash(payload)

    def commit(self, version_dir: str, manifest: Dict[str, Any]) -> None:
        write_json(os.path.join(version_dir, MANIFEST_NAME), manifest)
        version_name = os.path.basename(version_dir)
        final_dir = os.path.join(self.paths.versions_dir, version_name)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        # 临时目录整体替换进提交区（非逐文件覆盖）。
        shutil.move(version_dir, final_dir)

        new_link = self.paths.current_path + ".new"
        if os.path.lexists(new_link):
            os.unlink(new_link)
        os.symlink(os.path.join("versions", version_name), new_link)
        os.replace(new_link, self.paths.current_path)

    def current_manifest_safe(self) -> Dict[str, Any] | None:
        return self.load_manifest()
