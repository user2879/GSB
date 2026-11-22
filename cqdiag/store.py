"""阶段产物快照存储: 临时目录写入 + 整体替换提交 + 启动恢复 (业务要求 8)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from . import FORMAT_VERSION

STAGES = ("calib", "scale", "quant", "dequant", "stratify")


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class SnapshotStore:
    """目录布局:
    workdir/
      snapshots/<stage>/<tensor>__<col>.json   已提交阶段产物
      staging/<stage>/...                      提交前临时目录
      manifest.json                            清单
    """

    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self.snap_dir = self.workdir / "snapshots"
        self.staging_dir = self.workdir / "staging"
        self.manifest_path = self.workdir / "manifest.json"
        self.recovery_log: list[str] = []
        self.manifest = self._load_or_init_manifest()

    def _load_or_init_manifest(self) -> dict:
        # 启动恢复判定: 残留 staging 说明上次提交中断, 丢弃并按清单恢复
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
            self.recovery_log.append(
                "discarded incomplete staging dir from interrupted commit"
            )
        if not self.manifest_path.exists():
            return {
                "format_version": FORMAT_VERSION,
                "global_version": 0,
                "config_hash": None,
                "completed_columns": [],
                "artifacts": {s: {} for s in STAGES},
            }
        manifest = _read_json(self.manifest_path)
        if manifest.get("format_version") != FORMAT_VERSION:
            self.recovery_log.append(
                "format version changed, previous snapshot artifacts invalidated"
            )
            shutil.rmtree(self.snap_dir, ignore_errors=True)
            return {
                "format_version": FORMAT_VERSION,
                "global_version": 0,
                "config_hash": None,
                "completed_columns": [],
                "artifacts": {s: {} for s in STAGES},
            }
        # 校验清单引用的产物文件真实存在且可解析, 缺失/损坏则剔除
        for stage in STAGES:
            for col_key in list(manifest["artifacts"].get(stage, {})):
                path = self._artifact_path(stage, col_key)
                ok = path.exists()
                if ok:
                    try:
                        _read_json(path)
                    except (json.JSONDecodeError, OSError):
                        ok = False
                if not ok:
                    del manifest["artifacts"][stage][col_key]
                    self.recovery_log.append(
                        f"dropped corrupt/missing artifact {stage}/{col_key}"
                    )
        return manifest

    def _artifact_path(self, stage: str, col_key: str) -> Path:
        safe = col_key.replace(":", "__")
        return self.snap_dir / stage / f"{safe}.json"

    def get(self, stage: str, col_key: str) -> dict | None:
        entry = self.manifest["artifacts"].get(stage, {}).get(col_key)
        if entry is None:
            return None
        path = self._artifact_path(stage, col_key)
        if not path.exists():
            return None
        return _read_json(path)

    def get_key(self, stage: str, col_key: str) -> str | None:
        entry = self.manifest["artifacts"].get(stage, {}).get(col_key)
        return None if entry is None else entry["key"]

    def commit(self, new_artifacts: dict[str, dict[str, dict]], config_hash: str,
               completed_columns: list[str]) -> None:
        """先把产物写入临时目录, 再整体替换提交并更新清单."""
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        for stage, items in new_artifacts.items():
            for col_key, payload in items.items():
                safe = col_key.replace(":", "__")
                path = self.staging_dir / stage / f"{safe}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_json_atomic(path, payload)
        # 整体替换: staging -> snapshots
        for stage in STAGES:
            stage_dir = self.staging_dir / stage
            if not stage_dir.exists():
                continue
            target = self.snap_dir / stage
            target.mkdir(parents=True, exist_ok=True)
            for f in sorted(stage_dir.iterdir()):
                os.replace(f, target / f.name)
        shutil.rmtree(self.staging_dir, ignore_errors=True)
        # 更新清单
        self.manifest["global_version"] += 1
        self.manifest["config_hash"] = config_hash
        self.manifest["completed_columns"] = sorted(completed_columns)
        for stage, items in new_artifacts.items():
            for col_key, payload in items.items():
                self.manifest["artifacts"].setdefault(stage, {})[col_key] = {
                    "key": payload["cache_key"],
                    "version": self.manifest["global_version"],
                }
        _write_json_atomic(self.manifest_path, self.manifest)

    def detect_version_change(self, config_hash: str) -> str:
        stored = self.manifest.get("config_hash")
        if stored is None:
            return "fresh_workspace"
        if stored != config_hash:
            return "config_changed"
        return "unchanged"
