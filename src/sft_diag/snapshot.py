"""阶段化快照：临时目录写入、单文件整体原子替换、清单与恢复判定。

磁盘布局：
  snapshot/
    manifest.json
    tmp.<pid>/                 临时目录（阶段写入期间存在）
    stages/
      sampling.json
      scoring.json
      dedup.json

每个阶段先写 tmp 目录下的新文件，再用 os.replace 原子替换正式文件，
最后原子替换 manifest。强制终止后只可能出现：
- manifest 已替换 -> 整个批次的阶段产物均已落盘（按阶段顺序提交）；
- manifest 未替换 -> 恢复到上一完整批次。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from .errors import E_CONFIG_MISMATCH, E_SNAPSHOT_CORRUPT, Failure
from .utils import canonical_json, fingerprint

STAGE_FILES = {
    "sampling": "sampling.json",
    "scoring": "scoring.json",
    "dedup": "dedup.json",
}
MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    global_version: int
    base_fingerprint: str
    combined_fingerprint: str
    config_fp: str
    params_fp: str
    completed_batch_no: int
    stage_versions: dict[str, int]
    stage_cache_keys: dict[str, str]
    snapshot_interval_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_version": self.global_version,
            "base_fingerprint": self.base_fingerprint,
            "combined_fingerprint": self.combined_fingerprint,
            "config_fp": self.config_fp,
            "params_fp": self.params_fp,
            "completed_batch_no": self.completed_batch_no,
            "stage_versions": dict(self.stage_versions),
            "stage_cache_keys": dict(self.stage_cache_keys),
            "snapshot_interval_batches": self.snapshot_interval_batches,
            "manifest_fp": fingerprint(
                {
                    "global_version": self.global_version,
                    "base_fingerprint": self.base_fingerprint,
                    "combined_fingerprint": self.combined_fingerprint,
                    "config_fp": self.config_fp,
                    "params_fp": self.params_fp,
                    "completed_batch_no": self.completed_batch_no,
                    "stage_versions": dict(self.stage_versions),
                    "stage_cache_keys": dict(self.stage_cache_keys),
                    "snapshot_interval_batches": self.snapshot_interval_batches,
                }
            ),
        }


class SnapshotStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.stage_dir = os.path.join(self.root, "stages")
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)
        os.makedirs(self.stage_dir, exist_ok=True)

    # ---------- 基础读写 ----------
    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        text = canonical_json(payload)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def _atomic_replace_json(self, relative_path: str, payload: Any) -> None:
        target = os.path.join(self.root, relative_path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="tmp.", dir=self.root)
        try:
            temp_file = os.path.join(temp_dir, os.path.basename(relative_path))
            self._write_json(temp_file, payload)
            os.replace(temp_file, target)
        finally:
            if os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def commit_stage(
        self,
        stage: str,
        payload: Any,
        cache_key: str,
        manifest: Manifest,
    ) -> Manifest:
        relative = os.path.join("stages", STAGE_FILES[stage])
        self._atomic_replace_json(relative, payload)
        manifest.stage_versions[stage] = manifest.global_version
        manifest.stage_cache_keys[stage] = cache_key
        self._atomic_replace_json(MANIFEST_NAME, manifest.to_dict())
        return manifest

    def read_stage(self, stage: str) -> Any | None:
        path = os.path.join(self.stage_dir, STAGE_FILES[stage])
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ---------- 清单与恢复 ----------
    def write_manifest(self, manifest: Manifest) -> None:
        self._atomic_replace_json(MANIFEST_NAME, manifest.to_dict())

    def read_manifest(self) -> Manifest | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        stored_fp = data.pop("manifest_fp", None)
        manifest = Manifest(
            global_version=int(data["global_version"]),
            base_fingerprint=data["base_fingerprint"],
            combined_fingerprint=data["combined_fingerprint"],
            config_fp=data["config_fp"],
            params_fp=data["params_fp"],
            completed_batch_no=int(data["completed_batch_no"]),
            stage_versions=dict(data["stage_versions"]),
            stage_cache_keys=dict(data["stage_cache_keys"]),
            snapshot_interval_batches=int(data["snapshot_interval_batches"]),
        )
        if stored_fp != manifest.to_dict()["manifest_fp"]:
            raise ValueError("manifest fingerprint mismatch")
        return manifest

    def resume_check(
        self,
        base_fingerprint: str,
        current_combined: str,
    ) -> tuple[Manifest | None, Failure | None]:
        """启动恢复判定。

        - 无清单：全新运行；
        - base_fingerprint 不一致：E_CONFIG_MISMATCH，拒绝续跑；
        - 清单完整且当前输入已包含已完成批次：返回清单续跑。
        """
        try:
            manifest = self.read_manifest()
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            return None, Failure(
                E_SNAPSHOT_CORRUPT,
                self.manifest_path,
                MANIFEST_NAME,
                f"unreadable manifest: {exc}",
                stage="recovery",
            )
        if manifest is None:
            return None, None
        if manifest.base_fingerprint != base_fingerprint:
            return None, Failure(
                E_CONFIG_MISMATCH,
                self.manifest_path,
                MANIFEST_NAME,
                "base input fingerprint differs from snapshot; refusing to resume",
                batch_no=manifest.completed_batch_no,
                stage="recovery",
            )
        if current_combined != manifest.combined_fingerprint:
            return manifest, None
        return manifest, None

    def cleanup_temp_dirs(self) -> list[str]:
        removed: list[str] = []
        for name in os.listdir(self.root):
            if name.startswith("tmp."):
                path = os.path.join(self.root, name)
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)
        return removed
