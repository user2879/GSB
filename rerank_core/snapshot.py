"""阶段化快照：临时目录写入、整体原子替换、版本识别与恢复判定。

阶段产物按缓存键内容寻址存储；每次提交先写入新的临时目录，再以
``os.replace`` 整体替换 manifest，替换前对比版本号识别变化，绝不逐文件覆盖。
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import RerankConfig, groups_fingerprint
from .errors import ConfigMismatchError

MANIFEST_NAME = "manifest.json"
STAGE_LOAD = "load"
STAGE_NORMALIZE = "normalize"
STAGE_FUSION = "fusion"
STAGE_RERANK = "rerank"
STAGES = (STAGE_LOAD, STAGE_NORMALIZE, STAGE_FUSION, STAGE_RERANK)


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.tmp_dir = self.root / "tmp"
        self.manifest_path = self.root / MANIFEST_NAME
        self.manifest: dict | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---- 恢复判定 ----
    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> dict:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        return self.manifest

    def verify_or_reject(
        self, config: RerankConfig, candidates_fingerprint: str
    ) -> dict:
        if not self.exists():
            raise FileNotFoundError(self.manifest_path)
        manifest = self.load_manifest()
        expected_config = config.fingerprint()
        expected_groups = groups_fingerprint(config)
        if manifest.get("config_hash") != expected_config:
            raise ConfigMismatchError(
                "快照配置哈希与当前配置不一致，拒绝续跑",
                details={
                    "snapshot_config_hash": manifest.get("config_hash"),
                    "current_config_hash": expected_config,
                },
            )
        if manifest.get("groups_hash") != expected_groups:
            raise ConfigMismatchError(
                "快照特征分组表哈希与当前分组表不一致，拒绝续跑",
                details={
                    "snapshot_groups_hash": manifest.get("groups_hash"),
                    "current_groups_hash": expected_groups,
                },
            )
        if manifest.get("candidates_hash") != candidates_fingerprint:
            raise ConfigMismatchError(
                "快照候选集内容指纹与当前候选集不一致，拒绝续跑",
                details={
                    "snapshot_candidates_hash": manifest.get("candidates_hash"),
                    "current_candidates_hash": candidates_fingerprint,
                },
            )
        return manifest

    # ---- 内容寻址产物 ----
    def _artifact_path(self, key: str) -> Path:
        return self.artifacts_dir / f"{key}.json"

    def has_artifact(self, key: str) -> bool:
        return self._artifact_path(key).exists()

    def read_artifact(self, key: str):
        with self._artifact_path(key).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def stage_artifact(self, key: str, payload) -> Path:
        """先把产物写到临时目录，提交时再整体并入。"""
        target_tmp = self.tmp_dir / f"{key}.json"
        with target_tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        return target_tmp

    def commit(
        self,
        *,
        global_version: int,
        config: RerankConfig,
        candidates_fingerprint: str,
        completed_cursor: list[str],
        stage_versions: dict[str, str],
        staged_files: list[tuple[Path, str]],
        stage_cache_keys: dict,
        extra: dict | None = None,
    ) -> dict:
        previous = self.manifest or (
            self.load_manifest() if self.exists() else None
        )
        previous_version = previous.get("global_version", 0) if previous else 0
        if global_version <= previous_version:
            raise RuntimeError(
                f"全局版本号必须单调递增: {global_version} <= {previous_version}"
            )

        # 临时产物原子并入内容寻址目录（仅新增本次变化的键）
        for tmp_path, key in staged_files:
            final_path = self._artifact_path(key)
            if not final_path.exists():
                os.replace(tmp_path, final_path)
            else:
                tmp_path.unlink(missing_ok=True)

        manifest = {
            "global_version": global_version,
            "config_hash": config.fingerprint(),
            "groups_hash": groups_fingerprint(config),
            "candidates_hash": candidates_fingerprint,
            "completed_request_cursor": list(completed_cursor),
            "stage_versions": dict(stage_versions),
            "stage_cache_keys": stage_cache_keys,
        }
        if extra:
            manifest.update(extra)

        tmp_manifest = self.tmp_dir / MANIFEST_NAME
        with tmp_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp_manifest, self.manifest_path)  # 整体替换，非逐文件覆盖
        self.manifest = manifest
        return manifest

    def reset(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = None
