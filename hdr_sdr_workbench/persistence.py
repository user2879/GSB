"""崩溃安全的工作区持久化。

目录布局::

    <workspace>/
      frames/                 合成 half 源帧（frame_XXXX.half.z）
      outputs/                8 位 SDR 输出（out_XXXX_<版本戳>.u8）
      metrics/                指标报告（metric_XXXX_<版本戳>.json）
      tmp/                    原子写临时区（崩溃后整体视为孤儿）
      manifest.json           最近一次已提交稳定快照（原子替换）
      operations.jsonl        已提交操作序列（每行一条，原子追加）
      shutdown_report.json    仅受控退出时存在
      recovery.log            每次恢复的孤儿回收/哈希校验记录

提交协议（保证“写入未完成即未提交”）：
  1. 所有新产物先写到 tmp/；
  2. fsync 后 rename 到正式目录；
  3. 重写 operations.jsonl 与 manifest.json 到 tmp 再原子 replace；
  4. 只有 manifest.json 替换成功，该版本戳才算已提交。

恢复时只信任 manifest.json 中的版本戳与内容哈希；tmp/ 内任何文件、
outputs/metrics 中未被 manifest 引用的文件一律作为孤儿回收。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from .errors import (E_HASH_MISMATCH, E_RECOVER_ORPHAN, E_SNAPSHOT_STALE,
                     E_STATE_VERSION, WorkbenchError)
from .state import STATE_SCHEMA_VERSION, StateTable, canonical_json

MANIFEST_NAME = "manifest.json"
OPERATIONS_NAME = "operations.jsonl"
SHUTDOWN_NAME = "shutdown_report.json"
RECOVERY_LOG_NAME = "recovery.log"
DIR_FRAMES = "frames"
DIR_OUTPUTS = "outputs"
DIR_METRICS = "metrics"
DIR_TMP = "tmp"


def ensure_dirs(workspace: str) -> dict:
    paths = {name: os.path.join(workspace, name) for name in
             (DIR_FRAMES, DIR_OUTPUTS, DIR_METRICS, DIR_TMP)}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: str, data: bytes) -> None:
    """tmp 目录落盘 + fsync + 原子 rename。"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(directory), DIR_TMP) \
        if os.path.basename(directory) != DIR_TMP else directory
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def atomic_write_json(path: str, obj) -> None:
    atomic_write(path, canonical_json(obj) + b"\n")


def output_name(frame: int, version: int) -> str:
    return f"out_{frame:04d}_v{version:06d}.u8"


def metric_name(frame: int, version: int) -> str:
    return f"metric_{frame:04d}_v{version:06d}.json"


class WorkspaceStore:
    """工作区读写、提交与恢复。"""

    def __init__(self, workspace: str):
        self.workspace = os.path.abspath(workspace)
        self.paths = ensure_dirs(self.workspace)
        self.manifest_path = os.path.join(self.workspace, MANIFEST_NAME)
        self.operations_path = os.path.join(self.workspace, OPERATIONS_NAME)
        self.recovery_log_path = os.path.join(self.workspace, RECOVERY_LOG_NAME)

    # -- 快照读写 ---------------------------------------------------------
    def write_manifest(self, manifest: dict) -> None:
        if int(manifest.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise WorkbenchError(E_STATE_VERSION,
                                 f"manifest 版本不受支持: {manifest.get('schema_version')}")
        atomic_write_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))

    def write_operations(self, operations: list[dict]) -> None:
        lines = [canonical_json(op).decode("utf-8") for op in operations]
        atomic_write(self.operations_path, ("\n".join(lines) + "\n").encode("utf-8"))

    def read_operations(self) -> list[dict]:
        if not os.path.exists(self.operations_path):
            return []
        ops = []
        with open(self.operations_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ops.append(json.loads(line))
        return ops

    def load_state_table(self) -> StateTable | None:
        manifest = self.read_manifest()
        if manifest is None:
            return None
        if int(manifest.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise WorkbenchError(E_STATE_VERSION,
                                 f"manifest 版本冲突: {manifest.get('schema_version')}")
        return StateTable.from_dict(manifest["state_table"])

    def commit(self, state_table: StateTable, frame_entries: dict,
               source_manifest: dict) -> dict:
        """提交一个稳定快照。frame_entries: {frame: {version, output_hash,...}}。"""
        manifest = {
            "schema_version": STATE_SCHEMA_VERSION,
            "committed_at_seq": len(state_table.operations) - 1,
            "state_table": state_table.to_dict(),
            "frames": {str(k): v for k, v in sorted(frame_entries.items())},
            "source_config": source_manifest.get("config", {}),
        }
        # 先保证操作序列落盘，再原子替换 manifest
        self.write_operations(state_table.operations)
        self.write_manifest(manifest)
        return manifest

    # -- 快照新旧判定 -----------------------------------------------------
    def reject_if_stale(self, incoming: dict) -> None:
        """整体导入外部快照时以版本戳判定新旧，拒绝盲目录入旧缓存。"""
        current = self.read_manifest()
        if current is None:
            return
        in_seq = int(incoming.get("committed_at_seq", -1))
        cur_seq = int(current.get("committed_at_seq", -1))
        if in_seq < cur_seq:
            raise WorkbenchError(
                E_SNAPSHOT_STALE,
                f"拒绝旧快照: 传入版本戳 {in_seq} < 当前版本戳 {cur_seq}",
                details={"incoming": in_seq, "current": cur_seq})
        if int(incoming.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise WorkbenchError(E_STATE_VERSION,
                                 f"快照版本不受支持: {incoming.get('schema_version')}")

    # -- 恢复 -------------------------------------------------------------
    def recover(self) -> dict:
        """按最近一次已提交稳定状态恢复，回收孤儿产物并校验哈希。"""
        report = {"event": "recover", "orphans": [], "hash_failures": [],
                  "committed_at_seq": None, "recovered_frames": 0}
        manifest = self.read_manifest()
        if manifest is None:
            # 没有任何已提交状态：清理全部临时/产物
            report["orphans"] = self._collect_orphans(set())
            self._append_recovery_log(report)
            return report

        report["committed_at_seq"] = int(manifest["committed_at_seq"])
        referenced: set[str] = set()
        for frame, entry in manifest["frames"].items():
            referenced.add(entry["output_file"])
            referenced.add(entry["metric_file"])
            out_path = os.path.join(self.paths[DIR_OUTPUTS], entry["output_file"])
            if not os.path.exists(out_path) or sha256_file(out_path) != entry["output_hash"]:
                report["hash_failures"].append(
                    {"frame": int(frame), "file": entry["output_file"]})

        report["orphans"] = self._collect_orphans(referenced)
        if report["orphans"]:
            raise WorkbenchError(
                E_RECOVER_ORPHAN,
                f"恢复时回收 {len(report['orphans'])} 个未提交孤儿产物",
                details=report)
        if report["hash_failures"]:
            raise WorkbenchError(
                E_HASH_MISMATCH,
                f"{len(report['hash_failures'])} 个已提交产物哈希校验失败",
                details=report)
        report["recovered_frames"] = len(manifest["frames"])
        self._append_recovery_log(report)
        return report

    def _collect_orphans(self, referenced: set[str]) -> list[str]:
        import re
        orphans = []
        # tmp/ 下全部是未提交临时文件
        for name in sorted(os.listdir(self.paths[DIR_TMP])):
            orphans.append(self._recover_one(os.path.join(self.paths[DIR_TMP], name)))
        # outputs/metrics 中：保留当前 manifest 引用项与版本戳命名合法的历史
        # 产物（历史状态重建需要），只回收命名不合法/未纳入任何提交的孤儿。
        patterns = (re.compile(r"out_\d{4}_v\d{6}\.u8$"),
                    re.compile(r"metric_\d{4}_v\d{6}\.json$"))
        for directory in (self.paths[DIR_OUTPUTS], self.paths[DIR_METRICS]):
            for name in sorted(os.listdir(directory)):
                if name in referenced:
                    continue
                if any(pat.match(name) for pat in patterns):
                    continue  # 合法历史版本产物，保留以供历史状态重建
                orphans.append(self._recover_one(os.path.join(directory, name)))
        return [o for o in orphans if o]

    def _recover_one(self, path: str) -> str | None:
        try:
            os.remove(path)
            return os.path.relpath(path, self.workspace)
        except FileNotFoundError:
            return None

    def _append_recovery_log(self, report: dict) -> None:
        line = canonical_json({"ts_monotonic": time.monotonic_ns(), **report}).decode("utf-8")
        with open(self.recovery_log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def write_shutdown_report(self, manifest: dict, reason: str = "controlled") -> dict:
        report = {
            "event": "shutdown",
            "reason": reason,
            "committed_at_seq": int(manifest["committed_at_seq"]),
            "frame_count": len(manifest["frames"]),
            "output_hashes": {f: e["output_hash"] for f, e in sorted(manifest["frames"].items())},
        }
        atomic_write_json(os.path.join(self.workspace, SHUTDOWN_NAME), report)
        return report

    def shutdown_report_exists(self) -> bool:
        return os.path.exists(os.path.join(self.workspace, SHUTDOWN_NAME))

    def clear_shutdown_report(self) -> None:
        path = os.path.join(self.workspace, SHUTDOWN_NAME)
        if os.path.exists(path):
            os.remove(path)
