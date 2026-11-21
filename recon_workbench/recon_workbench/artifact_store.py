"""产物缓存键、阶段快照与原子整体替换。"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import BizCode, WorkbenchError
from .utils import (
    atomic_replace_dir,
    atomic_write_json,
    content_hash,
    read_json,
)

STAGE_FEATURE = "feature"
STAGE_MATCH = "match"
STAGE_DEPTH = "depth"
STAGE_CLOUD = "point_cloud"
STAGES = (STAGE_FEATURE, STAGE_MATCH, STAGE_DEPTH, STAGE_CLOUD)


@dataclass
class CacheDecision:
    reused: List[str] = field(default_factory=list)
    recomputed: List[str] = field(default_factory=list)

    def record(self, key: str, reused: bool) -> None:
        (self.reused if reused else self.recomputed).append(key)

    def to_dict(self) -> Dict[str, Any]:
        return {"reused": list(self.reused),
                "recomputed": list(self.recomputed)}


class ArtifactStore:
    """工作区：内容寻址缓存 + 阶段快照，全部 JSON 原子写。"""

    def __init__(self, work_dir: str) -> None:
        self.work_dir = os.path.abspath(work_dir)
        self.cache_dir = os.path.join(self.work_dir, "cache")
        self.snapshot_dir = os.path.join(self.work_dir, "snapshots")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    # ---- 缓存键（上游内容指纹 + 参数指纹逐层派生）----
    @staticmethod
    def feature_key(view_fingerprint: str, config_fingerprint: str,
                    precision: str) -> str:
        return "feat:" + content_hash(
            {"view": view_fingerprint, "cfg": config_fingerprint,
             "stage": STAGE_FEATURE, "precision": precision}
        )

    @staticmethod
    def match_key(key_a: str, key_b: str, config_fingerprint: str,
                  precision: str) -> str:
        return "match:" + content_hash(
            {"a": key_a, "b": key_b, "cfg": config_fingerprint,
             "stage": STAGE_MATCH, "precision": precision}
        )

    @staticmethod
    def depth_key(match_key: str, camera_fingerprint: str,
                  config_fingerprint: str, precision: str) -> str:
        return "depth:" + content_hash(
            {"match": match_key, "camera": camera_fingerprint,
             "cfg": config_fingerprint, "stage": STAGE_DEPTH,
             "precision": precision}
        )

    @staticmethod
    def cloud_key(depth_keys: List[str], config_fingerprint: str,
                  precision: str) -> str:
        return "cloud:" + content_hash(
            {"depths": depth_keys, "cfg": config_fingerprint,
             "stage": STAGE_CLOUD, "precision": precision}
        )

    def _cache_path(self, key: str) -> str:
        digest = key.split(":", 1)[1]
        return os.path.join(self.cache_dir, f"{digest}.json")

    def has(self, key: str) -> bool:
        return os.path.exists(self._cache_path(key))

    def load(self, key: str) -> Any:
        return read_json(self._cache_path(key))

    def save(self, key: str, payload: Any) -> None:
        atomic_write_json(self._cache_path(key), payload)

    # ---- 阶段快照（临时目录写入 + 整体替换）----
    def _stage_dir(self, version: int, stage: str) -> str:
        return os.path.join(
            self.snapshot_dir, f"v{version:04d}", stage
        )

    def commit_stage(self, version: int, stage: str,
                     files: Dict[str, Any], manifest: Dict[str, Any]) -> str:
        target = self._stage_dir(version, stage)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        tmp_root = os.path.join(
            self.snapshot_dir, f".staging-v{version:04d}-{stage}"
        )
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        os.makedirs(tmp_root, exist_ok=True)
        for name, payload in files.items():
            atomic_write_json(os.path.join(tmp_root, name), payload)
        atomic_write_json(os.path.join(tmp_root, "_manifest.json"), manifest)
        atomic_replace_dir(tmp_root, target)
        return target

    def load_stage(self, version: int, stage: str) -> Optional[Dict[str, Any]]:
        stage_dir = self._stage_dir(version, stage)
        if not os.path.isdir(stage_dir):
            return None
        out: Dict[str, Any] = {}
        for name in sorted(os.listdir(stage_dir)):
            if name == "_manifest.json":
                continue
            out[name] = read_json(os.path.join(stage_dir, name))
        return out

    def stage_manifest(self, version: int, stage: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self._stage_dir(version, stage), "_manifest.json")
        if not os.path.exists(path):
            return None
        return read_json(path)

    def latest_version(self) -> Optional[int]:
        if not os.path.isdir(self.snapshot_dir):
            return None
        versions = []
        for name in os.listdir(self.snapshot_dir):
            if name.startswith("v") and name[1:].isdigit():
                versions.append(int(name[1:]))
        return max(versions) if versions else None

    def versions(self) -> List[int]:
        if not os.path.isdir(self.snapshot_dir):
            return []
        return sorted(
            int(name[1:]) for name in os.listdir(self.snapshot_dir)
            if name.startswith("v") and name[1:].isdigit()
        )

    def discard_version(self, version: int) -> None:
        path = os.path.join(self.snapshot_dir, f"v{version:04d}")
        if os.path.isdir(path):
            shutil.rmtree(path)

    def verify_resume(self, version: int, config_hash: str,
                      input_hash: str) -> Dict[str, Any]:
        """启动续跑判定：版本号、配置哈希与输入指纹须全部一致。"""
        completed: Dict[str, Any] = {}
        for stage in STAGES:
            manifest = self.stage_manifest(version, stage)
            if manifest is None:
                continue
            if manifest.get("config_hash") != config_hash:
                raise WorkbenchError(
                    f"快照 {version} 配置哈希不一致",
                    stage=stage, code=BizCode.E_SNAPSHOT_MISMATCH,
                    detail={"snapshot": manifest.get("config_hash"),
                            "current": config_hash},
                )
            if manifest.get("input_hash") != input_hash:
                raise WorkbenchError(
                    f"快照 {version} 输入指纹不一致",
                    stage=stage, code=BizCode.E_SNAPSHOT_MISMATCH,
                    detail={"snapshot": manifest.get("input_hash"),
                            "current": input_hash},
                )
            completed[stage] = manifest
        return completed
