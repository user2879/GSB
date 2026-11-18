"""阶段快照整体替换、清单版本管理与崩溃恢复。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ProofreadError, E_CONFIG_MISMATCH, E_STATE_CORRUPT

MANIFEST_NAME = "manifest.json"
STAGE_DIR = "stages"


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    rule_fingerprint: str
    scoring_fingerprint: str
    completed_cursor: List[str]
    stage_versions: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_version": self.global_version,
            "config_hash": self.config_hash,
            "rule_fingerprint": self.rule_fingerprint,
            "scoring_fingerprint": self.scoring_fingerprint,
            "completed_cursor": list(self.completed_cursor),
            "stage_versions": dict(self.stage_versions),
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Manifest":
        return cls(
            global_version=int(obj["global_version"]),
            config_hash=obj["config_hash"],
            rule_fingerprint=obj["rule_fingerprint"],
            scoring_fingerprint=obj["scoring_fingerprint"],
            completed_cursor=list(obj.get("completed_cursor", [])),
            stage_versions=dict(obj.get("stage_versions", {})),
        )


class SnapshotStore:
    """所有阶段产物先写临时目录，再 os.replace 整体替换提交。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.stage_root = self.root / STAGE_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME

    # ---- 清单 ----
    def manifest_exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> Manifest:
        if not self.manifest_exists():
            raise ProofreadError(E_STATE_CORRUPT, "快照清单不存在", source=str(self.root))
        try:
            return Manifest.from_dict(json.loads(self.manifest_path.read_text("utf-8")))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ProofreadError(E_STATE_CORRUPT, f"快照清单损坏: {exc}",
                                 source=str(self.root)) from exc

    def write_manifest(self, manifest: Manifest) -> None:
        self._atomic_write_json(self.manifest_path, manifest.to_dict())

    def resume_guard(self, config_hash: str) -> Optional[Manifest]:
        """启动续跑守卫：版本/配置哈希不一致直接拒绝。"""
        if not self.manifest_exists():
            return None
        manifest = self.load_manifest()
        if manifest.config_hash != config_hash:
            raise ProofreadError(
                E_CONFIG_MISMATCH,
                f"快照配置哈希 {manifest.config_hash} 与当前配置 {config_hash} 不一致，拒绝续跑",
                source=str(self.root))
        return manifest

    # ---- 阶段产物 ----
    def stage_path(self, stage: str, version: int) -> Path:
        return self.stage_root / f"{stage}.v{version}.json"

    def commit_stage(self, stage: str, version: int, payload: Any) -> Path:
        target = self.stage_path(stage, version)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{stage}.", dir=self.root))
        try:
            tmp_file = tmp_dir / "payload.json"
            tmp_file.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")),
                encoding="utf-8")
            os.replace(tmp_file, target)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return target

    def load_stage(self, stage: str, version: int) -> Any:
        path = self.stage_path(stage, version)
        if not path.exists():
            raise ProofreadError(E_STATE_CORRUPT,
                                 f"阶段 {stage} v{version} 快照缺失", source=str(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def recovery_point(self) -> Dict[str, Any]:
        """强制终止后仅依据完整替换的快照报告恢复点。"""
        if not self.manifest_exists():
            return {"recoverable": False, "reason": "无快照清单"}
        manifest = self.load_manifest()
        intact: Dict[str, int] = {}
        corrupt: List[str] = []
        for stage, version in manifest.stage_versions.items():
            path = self.stage_path(stage, version)
            if not path.exists():
                corrupt.append(stage)
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                corrupt.append(stage)
                continue
            intact[stage] = version
        return {
            "recoverable": len(corrupt) == 0,
            "global_version": manifest.global_version,
            "completed_cursor": list(manifest.completed_cursor),
            "intact_stages": intact,
            "corrupt_stages": corrupt,
            "stage_versions": dict(manifest.stage_versions),
        }

    def reset(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
