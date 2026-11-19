"""依赖图、对齐产物与快照的原子持久化及强制终止恢复。

工作区布局：
  store/
    blobs/<sha256>          内容寻址的不可变二进制/JSON 块
    nodes/state.json        最近一次已提交的节点状态（原子替换）
    snapshots/<v>.json      已提交不可变快照（原子替换）
    journal.jsonl           追加式事件日志（原子 append）
    shutdown.marker         受控退出完整清单+关闭标记
    orphans.log             启动时回收的未提交孤儿记录

提交协议：先写 .tmp 临时文件，fsync 后 os.replace 原子改名；
没有完成 replace 的文件一律视为未提交，重启作为孤儿回收。
恢复只信任最近一次已提交稳定状态，绝不伪造退出报告。
"""
from __future__ import annotations

import json
import os
from typing import Any

from .hashing import canonical_bytes, sha256_hex

STORE_DIR = "store"
BLOB_DIR = "blobs"
NODE_STATE = "nodes" + os.sep + "state.json"
SNAP_DIR = "snapshots"
JOURNAL = "journal.jsonl"
SHUTDOWN = "shutdown.marker"
ORPHANS = "orphans.log"


class Store:
    def __init__(self, ws_dir: str) -> None:
        self.ws_dir = ws_dir
        self.root = os.path.join(ws_dir, STORE_DIR)
        self.blobs = os.path.join(self.root, BLOB_DIR)
        self.snaps = os.path.join(self.root, SNAP_DIR)
        self.nodes_dir = os.path.join(self.root, "nodes")
        os.makedirs(self.blobs, exist_ok=True)
        os.makedirs(self.snaps, exist_ok=True)
        os.makedirs(self.nodes_dir, exist_ok=True)

    # -- 原子写 ---------------------------------------------------------------
    def _atomic_write(self, path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def write_blob(self, payload: Any) -> tuple[str, str]:
        data = canonical_bytes(payload)
        digest = sha256_hex(data)
        path = os.path.join(self.blobs, digest)
        if not os.path.exists(path):
            self._atomic_write(path, data)
        return digest, os.path.relpath(path, self.root).replace(os.sep, "/")

    def write_raw_blob(self, data: bytes) -> str:
        digest = sha256_hex(data)
        path = os.path.join(self.blobs, digest)
        if not os.path.exists(path):
            self._atomic_write(path, data)
        return digest

    def commit_node_state(self, graph_manifest: dict[str, Any],
                          param_version: str, seq: int,
                          config_snapshot: dict[str, Any]) -> str:
        body = {
            "schema": "syncbench-state/1",
            "seq": seq,
            "param_version": param_version,
            "config_snapshot": config_snapshot,
            "graph": graph_manifest,
        }
        path = os.path.join(self.root, NODE_STATE)
        self._atomic_write(path, canonical_bytes(body))
        return sha256_hex(canonical_bytes(body))

    def commit_snapshot(self, snapshot_dict: dict[str, Any]) -> str:
        version = snapshot_dict["version"]
        path = os.path.join(self.snaps, f"snapshot_{version:04d}.json")
        self._atomic_write(path, canonical_bytes(snapshot_dict))
        return path

    def append_journal(self, event: dict[str, Any]) -> None:
        path = os.path.join(self.root, JOURNAL)
        # 追加式：读旧内容+新事件后原子替换，保证行级完整
        existing = b""
        if os.path.exists(path):
            with open(path, "rb") as fh:
                existing = fh.read()
        line = canonical_bytes(event) + b"\n"
        self._atomic_write(path, existing + line)

    def write_shutdown(self, report: dict[str, Any]) -> None:
        report = dict(report)
        report["kind"] = "controlled_shutdown"
        self._atomic_write(os.path.join(self.root, SHUTDOWN),
                           canonical_bytes(report))

    # -- 读取 -----------------------------------------------------------------
    def load_node_state(self) -> dict[str, Any] | None:
        path = os.path.join(self.root, NODE_STATE)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return json.loads(fh.read())

    def latest_snapshot(self) -> dict[str, Any] | None:
        if not os.path.isdir(self.snaps):
            return None
        names = sorted(n for n in os.listdir(self.snaps)
                       if n.startswith("snapshot_") and n.endswith(".json"))
        if not names:
            return None
        with open(os.path.join(self.snaps, names[-1]), "rb") as fh:
            return json.loads(fh.read())

    def journal_events(self) -> list[dict[str, Any]]:
        path = os.path.join(self.root, JOURNAL)
        if not os.path.exists(path):
            return []
        out = []
        with open(path, "rb") as fh:
            for line in fh.read().splitlines():
                if line.strip():
                    out.append(json.loads(line))
        return out

    def has_clean_shutdown(self) -> bool:
        return os.path.exists(os.path.join(self.root, SHUTDOWN))

    # -- 恢复/孤儿回收 ---------------------------------------------------------
    def recover(self) -> dict[str, Any]:
        """启动恢复：回收 .tmp 孤儿，返回恢复摘要（不伪造退出状态）。"""
        orphans: list[str] = []
        for base, _dirs, files in os.walk(self.root):
            for name in files:
                if name.endswith(".tmp"):
                    full = os.path.join(base, name)
                    orphans.append(os.path.relpath(full, self.root)
                                   .replace(os.sep, "/"))
                    os.remove(full)
        clean = self.has_clean_shutdown()
        state = self.load_node_state()
        snap = self.latest_snapshot()
        summary = {
            "kind": "recovery",
            "clean_shutdown": clean,
            "orphans_reclaimed": sorted(orphans),
            "restored_seq": state["seq"] if state else 0,
            "restored_param_version": state["param_version"] if state else None,
            "restored_snapshot_version": snap["version"] if snap else None,
            "forced_termination_detected": (not clean) and state is not None,
        }
        with open(os.path.join(self.root, ORPHANS), "ab") as fh:
            fh.write(canonical_bytes(summary) + b"\n")
        self.append_journal(summary)
        return summary
