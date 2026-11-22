"""阶段化快照：事务式版本目录 + current 符号链接原子切换。

一次快照事务 (begin -> write_stage* -> complete) 构建一个候选目录，
complete 时一次性切换 current 符号链接；任一阶段缺失则候选目录永不链接，
强制终止后只信任 current 指向的完整版本。
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

STAGES = ("load", "segment", "oov", "tag")
MANIFEST_NAME = "manifest.json"


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.tmp = self.root / "tmp"
        self.versions = self.root / "versions"
        self.current = self.root / "current"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.versions.mkdir(parents=True, exist_ok=True)
        self._candidate: Path | None = None
        self._written_stages: set[str] = set()

    def _current_real(self) -> Path | None:
        if not self.current.exists():
            return None
        return self.current.resolve()

    def load_manifest(self) -> dict | None:
        real = self._current_real()
        if real is None:
            return None
        path = real / MANIFEST_NAME
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def begin(self) -> None:
        self._candidate = Path(tempfile.mkdtemp(prefix="ver-", dir=self.tmp))
        self._written_stages = set()

    def write_stage(self, stage: str, version: int, payload: dict) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        if self._candidate is None:
            raise RuntimeError("snapshot transaction not begun")
        atomic_write_json(self._candidate / f"{stage}.json", {"version": version, "payload": payload})
        self._written_stages.add(stage)

    def complete(
        self,
        global_version: int,
        lexicon_hash: str,
        prob_hash: str,
        budget_consumed: bool,
        pending: list[str],
        stage_versions: dict[str, int],
    ) -> dict:
        if self._candidate is None or set(STAGES) - self._written_stages:
            raise RuntimeError("incomplete snapshot transaction")
        new_manifest = {
            "global_version": global_version,
            "lexicon_hash": lexicon_hash,
            "prob_table_hash": prob_hash,
            "budget_consumed": budget_consumed,
            "pending_sentences": list(pending),
            "stage_versions": dict(stage_versions),
            "stages": list(STAGES),
        }
        atomic_write_json(self._candidate / MANIFEST_NAME, new_manifest)
        final_dir = self.versions / f"v{global_version}"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(self._candidate, final_dir)
        self._candidate = None

        link_tmp = self.root / ".current.tmp"
        if link_tmp.is_symlink() or link_tmp.is_file():
            link_tmp.unlink()
        elif link_tmp.exists():
            shutil.rmtree(link_tmp)
        os.symlink(os.path.relpath(final_dir, self.root), link_tmp)
        if self.current.exists() and not self.current.is_symlink():
            shutil.rmtree(self.current)
        os.replace(link_tmp, self.current)
        return new_manifest

    def read_stage(self, stage: str) -> dict | None:
        real = self._current_real()
        if real is None:
            return None
        path = real / f"{stage}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def restore(self) -> dict | None:
        manifest = self.load_manifest()
        if manifest is None:
            return None
        real = self._current_real()
        for stage in manifest.get("stages", STAGES):
            if not (real / f"{stage}.json").exists():
                return None
        if not isinstance(manifest.get("stage_versions"), dict):
            return None
        return manifest

    def cleanup_tmp(self) -> None:
        for child in self.tmp.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
