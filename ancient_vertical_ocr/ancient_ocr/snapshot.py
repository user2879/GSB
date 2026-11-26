"""阶段化快照：临时目录写入 + os.replace 整体提交，崩溃恢复只认已提交快照。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import E_RECOVERY_FALLBACK, E_SNAPSHOT_VERSION_MISMATCH, ValidationIssue
from .hashing import canonical_json, digest

SNAPSHOT_NAME = "current"
STAGE_ORDER = ("load", "coarse", "fine", "intra", "merge")


@dataclass
class RecoveryInfo:
    recoverable: bool
    stage: str | None
    global_version: int
    completed_units: list[str]
    issue: ValidationIssue | None


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = self.root / SNAPSHOT_NAME

    def commit(self, state: dict, global_version: int, config_hash: str) -> str:
        tmp = self.root / f"tmp-{os.getpid()}-{global_version}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        manifest = {
            "global_version": global_version,
            "config_hash": config_hash,
            "stage": state.get("stage"),
            "state_digest": digest(state),
        }
        (tmp / "state.json").write_text(
            canonical_json(state), encoding="utf-8")
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp / "COMMITTED").write_text(str(global_version), encoding="utf-8")
        backup = self.root / f"backup-{os.getpid()}-{global_version}"
        if self.current.exists():
            os.replace(self.current, backup)
        os.replace(tmp, self.current)
        if backup.exists():
            shutil.rmtree(backup)
        return str(self.current)

    def latest(self) -> dict | None:
        state_path = self.current / "state.json"
        marker = self.current / "COMMITTED"
        if not state_path.exists() or not marker.exists():
            return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads((self.current / "manifest.json").read_text(encoding="utf-8"))
        if digest(state) != manifest["state_digest"]:
            return None
        return {"state": state, "manifest": manifest}

    def recover(self, expected_config_hash: str) -> RecoveryInfo:
        loaded = self.latest()
        if loaded is None:
            return RecoveryInfo(False, None, 0, [], ValidationIssue(
                E_RECOVERY_FALLBACK, "<snapshot>", "root",
                "不存在完整已提交快照，只能从空白全量重建"))
        manifest = loaded["manifest"]
        if manifest["config_hash"] != expected_config_hash:
            return RecoveryInfo(
                False, None, manifest["global_version"], [],
                ValidationIssue(
                    E_SNAPSHOT_VERSION_MISMATCH, "<snapshot>", "manifest.config_hash",
                    "配置哈希与当前配置不一致，拒绝逐文件覆盖，回退全量重建",
                    {"snapshot_hash": manifest["config_hash"],
                     "expected_hash": expected_config_hash}))
        state = loaded["state"]
        completed = sorted({
            unit["unit_key"]
            for sample in state.get("samples", {}).values()
            for unit in sample.get("completed_units", [])
        })
        return RecoveryInfo(
            recoverable=True, stage=state.get("stage"),
            global_version=manifest["global_version"],
            completed_units=completed, issue=None)

    def load_state(self) -> dict | None:
        loaded = self.latest()
        return loaded["state"] if loaded else None
