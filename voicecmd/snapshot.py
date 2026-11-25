"""阶段化快照：临时目录写入 + 原子整体替换 + 版本清单恢复判定。"""

import json
import os
import shutil
from pathlib import Path

from .config import PipelineConfig

MANIFEST_NAME = "snapshot_manifest.json"
STATE_NAME = "state.json"
STAGE_VERSION_INIT = 0


def write_snapshot(snapshot_dir: str | Path, state: dict, cfg: PipelineConfig,
                   stage_versions: dict[str, int], global_version: int) -> None:
    base = Path(snapshot_dir)
    tmp_dir = base.parent / f".{base.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "global_version": global_version,
        "config_hash": cfg.config_fingerprint(),
        "vocab_fingerprint": cfg.vocab_fingerprint(),
        "stage_versions": stage_versions,
    }
    (tmp_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_dir / STATE_NAME).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    backup_dir = base.parent / f".{base.name}.old.{os.getpid()}"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if base.exists():
        os.replace(base, backup_dir)
    os.replace(tmp_dir, base)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def load_snapshot(snapshot_dir: str | Path, cfg: PipelineConfig):
    base = Path(snapshot_dir)
    manifest_path = base / MANIFEST_NAME
    state_path = base / STATE_NAME
    if not manifest_path.exists() or not state_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != cfg.config_fingerprint():
        return {"action": "rebuild", "reason": "config_hash_mismatch", "manifest": manifest}
    if manifest.get("vocab_fingerprint") != cfg.vocab_fingerprint():
        return {"action": "rebuild", "reason": "vocab_fingerprint_mismatch",
                "manifest": manifest}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {"action": "resume", "reason": "matched", "manifest": manifest, "state": state}
