"""阶段化快照：临时目录写入 + 整体替换提交 + 清单版本判定。"""
from __future__ import annotations

import os
import shutil

from .errors import E_SNAPSHOT_STALE, FailureList
from .utils import ensure_dir, fingerprint, read_json, write_json

MANIFEST_NAME = "snapshot_manifest.json"
SNAPSHOT_VERSION = 3
STAGE_ORDER = ("load", "features_match", "triangulate", "depth_voxel")


class SnapshotStore:
    def __init__(self, root: str):
        self.root = ensure_dir(root)
        self.tmp_root = ensure_dir(os.path.join(root, "tmp"))
        self.stage_root = ensure_dir(os.path.join(root, "stages"))
        self.manifest_path = os.path.join(root, MANIFEST_NAME)

    # ---- 清单 ----
    def load_manifest(self) -> dict | None:
        if os.path.exists(self.manifest_path):
            return read_json(self.manifest_path)
        return None

    def save_manifest(self, manifest: dict) -> None:
        write_json(self.manifest_path, manifest)

    # ---- 阶段产物：先写临时目录，再整体替换 ----
    def commit_stage(self, stage: str, payload: dict, stage_version: str) -> str:
        stage_tmp = os.path.join(self.tmp_root, stage)
        if os.path.exists(stage_tmp):
            shutil.rmtree(stage_tmp)
        ensure_dir(stage_tmp)
        write_json(os.path.join(stage_tmp, "artifact.json"), payload)
        target = os.path.join(self.stage_root, stage)
        # 识别版本变化而非逐文件覆盖：目标已存在且版本相同则跳过替换
        marker = os.path.join(target, "artifact.json")
        if os.path.exists(marker):
            existing = read_json(marker)
            if existing.get("__stage_version__") == stage_version:
                shutil.rmtree(stage_tmp)
                return "unchanged"
        if os.path.exists(target):
            shutil.rmtree(target)
        os.replace(stage_tmp, target)
        return "replaced"

    def load_stage(self, stage: str) -> dict | None:
        marker = os.path.join(self.stage_root, stage, "artifact.json")
        if os.path.exists(marker):
            return read_json(marker)
        return None

    # ---- 恢复判定 ----
    def check_resume(self, config_fp: str, camera_fp: str,
                     exchange_fp: str, failures: FailureList) -> tuple:
        """返回 (可续跑阶段起点, 清单或 None)；不一致记 E_SNAPSHOT_STALE。"""
        manifest = self.load_manifest()
        if manifest is None:
            return STAGE_ORDER[0], None
        mismatch = []
        if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
            mismatch.append("snapshot_version")
        if manifest.get("config_hash") != config_fp:
            mismatch.append("config_hash")
        if manifest.get("camera_hash") != camera_fp:
            mismatch.append("camera_hash")
        if manifest.get("exchange_fingerprint") != exchange_fp:
            mismatch.append("exchange_fingerprint")
        if mismatch:
            failures.fail(
                E_SNAPSHOT_STALE,
                "snapshot mismatch: " + ",".join(mismatch),
                stage="snapshot",
                sort_key=f"snapshot|0000|{E_SNAPSHOT_STALE}",
                context={"mismatch": mismatch},
            )
            return self._affected_start(mismatch), manifest
        return None, manifest  # None 表示全部阶段可复用

    @staticmethod
    def _affected_start(mismatch: list) -> str:
        if "camera_hash" in mismatch or "exchange_fingerprint" in mismatch:
            return STAGE_ORDER[0]
        if "config_hash" in mismatch:
            return STAGE_ORDER[1]
        return STAGE_ORDER[0]

    def build_manifest(self, config_fp: str, camera_fp: str, exchange_fp: str,
                       completed_views: list, stage_versions: dict) -> dict:
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "global_version": fingerprint([config_fp, camera_fp, exchange_fp,
                                           sorted(completed_views), stage_versions]),
            "config_hash": config_fp,
            "camera_hash": camera_fp,
            "exchange_fingerprint": exchange_fp,
            "completed_views": sorted(completed_views),
            "stage_versions": stage_versions,
        }

    def reset(self) -> None:
        for d in (self.tmp_root, self.stage_root):
            if os.path.exists(d):
                shutil.rmtree(d)
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
        ensure_dir(self.tmp_root)
        ensure_dir(self.stage_root)
