"""依赖图、内容哈希与地图版本戳的持久化及崩溃恢复。

落盘布局（全部 UTF-8；数值经 :mod:`encoding` 以固定小端编码）::

    <workspace>/
      manifest.json        # 已提交的完整清单（原子替换）
      manifest.json.tmp    # 暂态文件（崩溃后按孤儿回收）
      exit-report.json     # 仅正常退出写入；强制终止不会产生
      orphan/              # 未提交缓存回收目录
      journal.log          # 提交事件追加日志

强制终止后重启只信任 ``manifest.json``：.tmp 与无清单引用的缓存文件
视为孤儿，移动到 ``orphan/``，绝不据此伪造退出报告。
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .encoding import content_hash, windows_path, write_text_file, read_text_file
from .errors import DependencyStaleError

MANIFEST_NAME = "manifest.json"
TMP_NAME = "manifest.json.tmp"
EXIT_REPORT_NAME = "exit-report.json"
ORPHAN_DIR = "orphan"
JOURNAL_NAME = "journal.log"


@dataclass
class RecoveryResult:
    recovered: bool
    manifest_path: str
    manifest_hash: str
    orphan_files: List[str]
    had_clean_exit_report: bool
    journal_tail: List[str]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "recovered": self.recovered,
            "manifest_path": self.manifest_path,
            "manifest_hash": self.manifest_hash,
            "orphan_files": self.orphan_files,
            "had_clean_exit_report": self.had_clean_exit_report,
            "journal_tail": self.journal_tail,
        }


class PersistenceStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.workspace / MANIFEST_NAME
        self.tmp_path = self.workspace / TMP_NAME
        self.exit_report_path = self.workspace / EXIT_REPORT_NAME
        self.orphan_dir = self.workspace / ORPHAN_DIR
        self.journal_path = self.workspace / JOURNAL_NAME

    # ---------- 原子提交 ----------
    def commit_manifest(self, payload: Dict[str, Any], map_version: int) -> str:
        canonical_digest = content_hash(payload)
        envelope = {
            "map_version": map_version,
            "manifest_hash": canonical_digest,
            "serialized_hash": content_hash(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            ),
            "payload": payload,
        }
        rendered = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2)
        write_text_file(self.tmp_path, rendered)
        os.replace(self.tmp_path, self.manifest_path)
        with self.journal_path.open("a", encoding="utf-8") as journal:
            journal.write(f"COMMIT {map_version} {canonical_digest}\n")
        return canonical_digest

    def write_exit_report(self, report: Dict[str, Any]) -> str:
        if not self.manifest_path.exists():
            raise DependencyStaleError(
                "无已提交清单，禁止写退出报告",
                manifest_path=windows_path(self.manifest_path),
            )
        rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        write_text_file(self.exit_report_path, rendered)
        return content_hash(report)

    def load_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            raise DependencyStaleError(
                "持久化清单不存在",
                manifest_path=windows_path(self.manifest_path),
            )
        envelope = json.loads(read_text_file(self.manifest_path))
        payload = envelope["payload"]
        serialized_actual = content_hash(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        )
        if serialized_actual != envelope["serialized_hash"]:
            raise DependencyStaleError(
                "清单内容哈希不匹配",
                stored=envelope["serialized_hash"],
                actual=serialized_actual,
            )
        return payload

    # ---------- 崩溃恢复 ----------
    def recover(self) -> RecoveryResult:
        orphans: List[str] = []
        if self.tmp_path.exists():
            orphans.extend(self._quarantine(self.tmp_path))

        tracked: set[str] = set()
        manifest_hash = ""
        recovered = False
        if self.manifest_path.exists():
            envelope = json.loads(read_text_file(self.manifest_path))
            payload = envelope["payload"]
            manifest_hash = envelope["manifest_hash"]
            serialized_actual = content_hash(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
            )
            if serialized_actual != envelope["serialized_hash"]:
                raise DependencyStaleError(
                    "恢复时清单哈希不匹配",
                    stored=envelope["serialized_hash"],
                    actual=serialized_actual,
                )
            tracked = _collect_tracked_files(payload)
            recovered = True

        for child in sorted(self.workspace.iterdir()):
            if child.name in {MANIFEST_NAME, JOURNAL_NAME, EXIT_REPORT_NAME, ORPHAN_DIR}:
                continue
            if child.name in tracked or child.name == TMP_NAME:
                continue
            if child.is_file():
                orphans.extend(self._quarantine(child))

        clean_exit = self.exit_report_path.exists()
        journal_tail = self._read_journal_tail()
        return RecoveryResult(
            recovered=recovered,
            manifest_path=windows_path(self.manifest_path),
            manifest_hash=manifest_hash,
            orphan_files=[windows_path(path) for path in orphans],
            had_clean_exit_report=clean_exit,
            journal_tail=journal_tail,
        )

    def remove_exit_report(self) -> None:
        if self.exit_report_path.exists():
            self.exit_report_path.unlink()

    def _quarantine(self, path: Path) -> List[str]:
        self.orphan_dir.mkdir(parents=True, exist_ok=True)
        target = self.orphan_dir / path.name
        if target.exists():
            target = self.orphan_dir / f"{path.stem}.{os.getpid()}{path.suffix}"
        shutil.move(str(path), str(target))
        return [str(target)]

    def _read_journal_tail(self, limit: int = 20) -> List[str]:
        if not self.journal_path.exists():
            return []
        lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        return lines[-limit:]

    def write_loose_cache(self, name: str, payload: Dict[str, Any]) -> str:
        """模拟“未提交缓存”：独立文件，未进入清单。"""
        path = self.workspace / name
        write_text_file(path, json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return windows_path(path)


def _collect_tracked_files(payload: Dict[str, Any]) -> set[str]:
    tracked: set[str] = set()
    for key, value in payload.items():
        if isinstance(value, str) and value.endswith(".json"):
            tracked.add(Path(value.replace("\\", os.sep)).name)
    return tracked
