"""阶段化快照：临时目录写入 + 整体替换提交 + 启动恢复判定。"""

from __future__ import annotations

import json
import os
import shutil

from .errors import Code

GLOBAL_VERSION = 1
STAGES = ("load_validate", "block_score", "softmax_reduce", "output_summary")


class SnapshotManager:
    def __init__(self, root, config_hash: str, tensor_pack_fp: str,
                 quant_level: int):
        self.root = root
        self.config_hash = config_hash
        self.tensor_pack_fp = tensor_pack_fp
        self.quant_level = quant_level
        self.tmp_dir = os.path.join(root, "tmp")
        self.current_dir = os.path.join(root, "current")
        self.manifest_path = os.path.join(root, "manifest.json")
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def check_resume(self):
        """返回 (已完成批次游标, 问题列表)。版本或配置哈希不一致时输出
        E_SNAPSHOT_VERSION_MISMATCH 并从零续跑。"""
        manifest = self.load_manifest()
        if manifest is None:
            return 0, []
        issues = []
        if (manifest.get("global_version") != GLOBAL_VERSION
                or manifest.get("config_hash") != self.config_hash):
            issues.append({
                "code": Code.E_SNAPSHOT_VERSION_MISMATCH,
                "message": "快照版本号或配置哈希不一致，放弃续跑并重新初始化",
                "manifest_version": manifest.get("global_version"),
                "manifest_config_hash": manifest.get("config_hash"),
                "current_config_hash": self.config_hash,
            })
            return 0, issues
        return int(manifest.get("completed_batch_cursor", 0)), issues

    def commit(self, completed_batch_cursor: int, stage_artifacts: dict):
        """各阶段产物先写入临时目录，再整体替换提交，最后落清单。"""
        manifest = self.load_manifest() or {}
        artifact_versions = dict(manifest.get("artifact_versions", {}))
        for stage in STAGES:
            payload = stage_artifacts.get(stage)
            if payload is None:
                continue
            tmp_path = os.path.join(self.tmp_dir, f"{stage}.tmp")
            final_path = os.path.join(self.current_dir, f"{stage}.bin")
            with open(tmp_path, "wb") as fh:
                fh.write(payload)
            os.replace(tmp_path, final_path)  # 同分区整体替换
            artifact_versions[stage] = artifact_versions.get(stage, 0) + 1
        new_manifest = {
            "global_version": GLOBAL_VERSION,
            "config_hash": self.config_hash,
            "tensor_pack_fp": self.tensor_pack_fp,
            "quant_level": self.quant_level,
            "completed_batch_cursor": completed_batch_cursor,
            "artifact_versions": artifact_versions,
        }
        tmp_manifest = os.path.join(self.tmp_dir, "manifest.tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as fh:
            json.dump(new_manifest, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_manifest, self.manifest_path)
        return new_manifest

    def reset(self):
        shutil.rmtree(self.root, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.current_dir, exist_ok=True)
