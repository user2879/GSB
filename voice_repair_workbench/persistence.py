"""工程与缓存的原子持久化、版本识别、受控退出与强制终止恢复。

提交语义：
  * 所有写入先落 .tmp 再 os.replace 原子替换，写入未完成 = 未提交；
  * snapshot.json 记录最近一次已提交稳定状态（工程指纹、素材哈希、
    各阶段键/输出哈希、引擎版本戳）；
  * 正常退出额外写 CLOSED.flag 与 exit_report.json；
  * 强制终止后重启只认 snapshot.json，.tmp 与无主缓存作为孤儿回收，
    且绝不伪造退出报告。
"""
from __future__ import annotations

import json
import os
import time

from . import config
from .cache import StageCache
from .errors import E_SNAPSHOT_STALE, RepairError
from .hashing import sha256_json


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json_if_exists(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Workspace:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.cache_dir = os.path.join(self.root, config.CACHE_DIR)
        self.cache = StageCache(self.cache_dir)
        self.project_path = os.path.join(self.root, config.PROJECT_FILE)
        self.snapshot_path = os.path.join(self.root, config.SNAPSHOT_FILE)
        self.closed_flag = os.path.join(self.root, config.CLOSED_MARKER)
        self.exit_report_path = os.path.join(self.root, config.EXIT_REPORT_FILE)
        self.orphan_log = os.path.join(self.root, config.ORPHAN_LOG_FILE)
        os.makedirs(self.root, exist_ok=True)

    # ---------- 工程 ----------
    def save_project(self, project_v2: dict) -> None:
        atomic_write_json(self.project_path, project_v2)

    def load_project_raw(self) -> dict:
        data = read_json_if_exists(self.project_path)
        if data is None:
            raise RepairError(
                E_SNAPSHOT_STALE,
                f"project file missing in workspace: {self.project_path}",
                field=config.PROJECT_FILE,
            )
        return data

    # ---------- 快照 ----------
    def commit_snapshot(self, snapshot: dict) -> None:
        payload = dict(snapshot)
        payload["committed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        payload["snapshot_hash"] = sha256_json(
            {k: v for k, v in payload.items() if k not in ("committed_at", "snapshot_hash")}
        )
        atomic_write_json(self.snapshot_path, payload)

    def load_snapshot(self) -> dict | None:
        return read_json_if_exists(self.snapshot_path)

    def verify_snapshot_freshness(
        self, snapshot: dict, content_hash: str, project_fingerprint: str
    ) -> None:
        """目录整体替换/参数变化后，用版本戳+哈希判定新旧，拒绝盲目复用。"""
        problems = []
        if snapshot.get("engine_version") != config.ENGINE_VERSION:
            problems.append("engine_version")
        if snapshot.get("content_hash") != content_hash:
            problems.append("content_hash")
        if snapshot.get("project_fingerprint") != project_fingerprint:
            problems.append("project_fingerprint")
        if problems:
            raise RepairError(
                E_SNAPSHOT_STALE,
                "snapshot does not match current project/material; cache reuse forbidden",
                field=",".join(problems),
                context={
                    "stale_fields": problems,
                    "snapshot_engine": snapshot.get("engine_version"),
                    "current_engine": config.ENGINE_VERSION,
                },
            )

    def verify_stage_hashes(self, snapshot: dict) -> None:
        """重启后逐阶段核对输出哈希，任何不符抛 E_CACHE_HASH（由 cache.get 抛）。"""
        for entry in snapshot.get("stages", []):
            self.cache.get(entry["key"])

    # ---------- 崩溃恢复 ----------
    def was_cleanly_closed(self) -> bool:
        return os.path.exists(self.closed_flag)

    def recover(self, live_keys: set[str]) -> dict:
        """重启恢复：回收 .tmp 与孤儿缓存，返回恢复审计信息。"""
        uncommitted = self.cache.wipe_uncommitted()
        orphans = self.cache.delete_orphans(live_keys)
        audit = {
            "uncommitted_tmp_reclaimed": uncommitted,
            "orphan_cache_reclaimed": orphans,
            "clean_close": self.was_cleanly_closed(),
            "exit_report_present": os.path.exists(self.exit_report_path),
        }
        with open(self.orphan_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **audit}, sort_keys=True) + "\n")
        # 恢复场景下旧关闭标记与退出报告不再代表本次运行状态。
        if os.path.exists(self.closed_flag):
            os.remove(self.closed_flag)
        return audit

    # ---------- 受控退出 ----------
    def write_close_marker(self, manifest: dict) -> None:
        atomic_write_json(self.closed_flag, {"closed": True, **manifest})

    def write_exit_report(self, report: dict) -> None:
        atomic_write_json(self.exit_report_path, report)

    def output_path(self, name: str) -> str:
        return os.path.join(self.root, name)
