"""阶段快照的持久化、整体替换与恢复判定。

状态目录布局::

    state/
      manifest.json            全局版本号/配置哈希/种子版本/已完成样本/已消耗预算
      samples/<sid>/...        活跃样本三件套（内容寻址保存）
      cache/feature/<sid>.json
      cache/signature/<sid>.json
      cache/bucket/<sid>.json
      cache/cluster/ledger.json
      pending.json             待办队列
      tmp/<version>/           阶段临时目录，提交时整体替换
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from .hashing import fingerprint


def write_json_atomic(path: str, obj: Any) -> None:
    """同目录临时文件 + os.replace 实现单文件原子写。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class SnapshotStore:
    """封装状态目录、临时阶段目录与整体替换提交。"""

    def __init__(self, state_dir: str):
        self.state_dir = os.path.abspath(state_dir)
        self.tmp_root = os.path.join(self.state_dir, "tmp")
        os.makedirs(os.path.join(self.state_dir, "cache", "feature"), exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "cache", "signature"), exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "cache", "bucket"), exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "cache", "cluster"), exist_ok=True)
        os.makedirs(os.path.join(self.state_dir, "samples"), exist_ok=True)

    # ---- 临时阶段目录与整体替换 --------------------------------------
    def stage_dir(self, version: int, stage: str) -> str:
        path = os.path.join(self.tmp_root, f"v{version}", stage)
        os.makedirs(path, exist_ok=True)
        return path

    def commit_stage(self, version: int, stage: str) -> str:
        """把阶段临时目录整体替换到 state/stages/<version>/<stage>。"""
        src = os.path.join(self.tmp_root, f"v{version}", stage)
        dst = os.path.join(self.state_dir, "stages", f"v{version}", stage)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        return dst

    def cleanup_tmp(self, version: int) -> None:
        path = os.path.join(self.tmp_root, f"v{version}")
        if os.path.exists(path):
            shutil.rmtree(path)

    # ---- manifest 恢复判定 -------------------------------------------
    def manifest_path(self) -> str:
        return os.path.join(self.state_dir, "manifest.json")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["manifest_fingerprint"] = fingerprint(
            {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
        )
        write_json_atomic(self.manifest_path(), manifest)

    def read_manifest(self) -> dict[str, Any] | None:
        manifest = read_json(self.manifest_path(), None)
        if manifest is None:
            return None
        stored = manifest.pop("manifest_fingerprint", None)
        if stored != fingerprint(manifest):
            raise ValueError("manifest 内容指纹不匹配，快照可能已损坏")
        return manifest

    def can_resume(self, config_hash: str) -> tuple[bool, str]:
        """启动时以全局版本号与配置哈希判定是否续跑。"""
        manifest = read_json(self.manifest_path(), None)
        if manifest is None:
            return False, "无历史快照，从空白启动"
        if manifest.get("config_hash") != config_hash:
            return False, "配置哈希变化，按新配置重建受影响产物"
        return True, f"命中版本号 {manifest.get('version')} 的快照，按清单续跑"
