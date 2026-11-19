"""版本化快照：临时目录写入 + 原子整体替换 + 恢复一致性。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ESnapshotStale
from .fingerprints import content_hash, sha256_text
from .models import Corpus

MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def commit(
        self,
        corpus: Corpus,
        *,
        global_version: int,
        model_fp: str,
        mapping_fp: str,
        artifacts: dict[str, dict[str, str]],
        degradation: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """先写临时目录，再整体替换目标目录。"""
        parent = self.root.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=".snap-", dir=parent))
        try:
            artifacts_dir = tmp_dir / "artifacts"
            artifacts_dir.mkdir()
            artifact_index: dict[str, dict[str, str]] = {}
            for sid, per_version in artifacts.items():
                artifact_index[sid] = {}
                for version, content in per_version.items():
                    fp = sha256_text(content)
                    (artifacts_dir / f"{sid}.{version}").write_text(content, encoding="utf-8")
                    artifact_index[sid][version] = fp
            state_payload = {
                sid: sent.to_dict() for sid, sent in sorted(corpus.sentences.items())
            }
            (tmp_dir / "state.json").write_text(
                json.dumps(state_payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            (tmp_dir / "degradation.json").write_text(
                json.dumps(degradation, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            manifest = {
                "global_version": global_version,
                "model_fp": model_fp,
                "mapping_fp": mapping_fp,
                "state_fp": content_hash(state_payload),
                "degradation_fp": content_hash(degradation),
                "sentence_artifacts": artifact_index,
                "diagnostics": diagnostics or {},
            }
            (tmp_dir / MANIFEST_NAME).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            # 原子替换：rename 在同文件系统上为原子操作。
            backup = None
            if self.root.exists():
                backup = Path(tempfile.mkdtemp(prefix=".snap-old-", dir=parent))
                shutil.move(str(self.root), str(backup / "old"))
            os.rename(tmp_dir, self.root)
            if backup is not None:
                shutil.rmtree(backup, ignore_errors=True)
            return manifest
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def load_state(self) -> dict[str, Any]:
        return json.loads((self.root / "state.json").read_text(encoding="utf-8"))

    def verify_recovery(
        self,
        corpus: Corpus,
        *,
        expected_version: int,
        model_fp: str,
        mapping_fp: str,
    ) -> dict[str, Any]:
        """启动恢复判定：版本号与当前内容指纹不一致即 E_SNAPSHOT_STALE，保留旧快照。"""
        manifest = self.load_manifest()
        if manifest is None:
            raise ESnapshotStale("无可用快照清单，无法恢复")
        current_state = {
            sid: sent.to_dict() for sid, sent in sorted(corpus.sentences.items())
        }
        current_fp = content_hash(current_state)
        mismatches: list[str] = []
        if manifest.get("global_version") != expected_version:
            mismatches.append(
                f"global_version: 清单 {manifest.get('global_version')} != 期望 {expected_version}"
            )
        if manifest.get("state_fp") != current_fp:
            mismatches.append("state_fp: 清单与当前统一表示内容指纹不一致")
        if manifest.get("model_fp") != model_fp:
            mismatches.append("model_fp 不一致")
        if manifest.get("mapping_fp") != mapping_fp:
            mismatches.append("mapping_fp 不一致")
        if mismatches:
            # 不覆盖旧快照，直接拒绝。
            raise ESnapshotStale("快照过期: " + "; ".join(mismatches))
        return manifest
