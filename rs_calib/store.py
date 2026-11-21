"""状态存储：阶段化快照、临时目录写入 + 原子整体替换、恢复判定。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

from .errors import E_SNAPSHOT_STALE, StageError
from .fingerprint import calibration_table_fingerprint, regression_config_fingerprint

SNAPSHOT_VERSION = 3
MANIFEST_NAME = "manifest.json"


class StateStore:
    def __init__(self, snapshot_dir: str, config: dict, table: dict[str, dict]):
        self.snapshot_dir = snapshot_dir
        self.config = config
        self.table = table
        self.config_hash = regression_config_fingerprint(config)
        self.table_hash = calibration_table_fingerprint(table)
        # 产物：以缓存键为索引，保证内容寻址、可精确复用/失效
        self.calib: dict[str, dict] = {}
        self.models: dict[str, dict] = {}
        self.alignments: dict[str, dict] = {}
        self.failures: list[dict] = []
        self.ledger: list[dict] = []
        self.reuse_log: list[dict] = []
        self.completed_tiles: list[str] = []
        self.global_version = 0
        self.stale_detected = False

    # ---------- 恢复 ----------
    def try_resume(self) -> bool:
        manifest_path = os.path.join(self.snapshot_dir, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            return False
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if manifest.get("snapshot_version") != SNAPSHOT_VERSION:
            self.stale_detected = True
            raise StageError(E_SNAPSHOT_STALE,
                             f"快照版本 {manifest.get('snapshot_version')} != {SNAPSHOT_VERSION}，重建")
        if manifest.get("config_hash") != self.config_hash:
            self.stale_detected = True
            raise StageError(E_SNAPSHOT_STALE, "回归配置哈希与快照不一致，重建",
                             stage="snapshot")
        if manifest.get("table_hash") != self.table_hash:
            # 定标表变更属正常增量失效场景，由指纹失效处理，不判 stale
            pass
        for stage in ("calib", "models", "alignments", "failures", "ledger", "reuse_log"):
            path = os.path.join(self.snapshot_dir, f"{stage}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    setattr(self, stage, json.load(fh))
        self.global_version = manifest["global_version"]
        self.completed_tiles = list(manifest.get("completed_tiles", []))
        return True

    # ---------- 原子提交 ----------
    def commit(self, stage: str, completed_tiles: list[str]) -> str:
        self.global_version += 1
        parent = os.path.dirname(os.path.abspath(self.snapshot_dir))
        os.makedirs(parent, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix=".snap-tmp-", dir=parent)
        try:
            payload = {
                "calib": self.calib,
                "models": self.models,
                "alignments": self.alignments,
                "failures": self.failures,
                "ledger": self.ledger,
                "reuse_log": self.reuse_log,
            }
            for name, data in payload.items():
                with open(os.path.join(tmp_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=2)
            manifest = {
                "snapshot_version": SNAPSHOT_VERSION,
                "global_version": self.global_version,
                "stage": stage,
                "config_hash": self.config_hash,
                "table_hash": self.table_hash,
                "completed_tiles": sorted(completed_tiles),
                "artifact_versions": {
                    "calib": self._version(self.calib),
                    "models": self._version(self.models),
                    "alignments": self._version(self.alignments),
                },
            }
            with open(os.path.join(tmp_dir, MANIFEST_NAME), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, sort_keys=True, indent=2)
            # 整体替换：移除旧快照目录后原子换入新目录
            if os.path.exists(self.snapshot_dir):
                shutil.rmtree(self.snapshot_dir)
            os.replace(tmp_dir, self.snapshot_dir)
            self.completed_tiles = sorted(completed_tiles)
            return self.snapshot_dir
        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _version(mapping: dict) -> str:
        import hashlib
        digest = hashlib.sha256(
            json.dumps(mapping, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def reset_for_rebuild(self) -> None:
        if os.path.exists(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)
        self.calib.clear()
        self.models.clear()
        self.alignments.clear()
        self.failures.clear()
        self.ledger.clear()
        self.reuse_log.clear()
        self.completed_tiles.clear()
        self.global_version = 0
