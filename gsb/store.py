"""原子持久化与恢复。

目录布局：
  snapshots/v{n}.json   快照（内容、版本戳、父版本戳、状态哈希）
  logs/v{n}.jsonl       达到该版本所应用的操作日志
  manifest.json         链清单（提交点，最后原子写入）
  chain_ledger.jsonl    版本链账本（压缩记录）
  adjudication.jsonl    裁决账本
  online_hashes.json    在线演进逐版本哈希（供独立验证器比对）
  SHUTDOWN_OK           正常退出标记
"""

from __future__ import annotations

import json
import os

from .model import Snapshot, canonical_json, content_hash

SHUTDOWN_MARKER = "SHUTDOWN_OK"


class Store:
    def __init__(self, root: str):
        self.root = root
        self.snapshots_dir = os.path.join(root, "snapshots")
        self.logs_dir = os.path.join(root, "logs")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    # ---- 原子写原语：临时文件 + fsync + os.replace + 目录 fsync ----
    def _atomic_write(self, path: str, data: bytes) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def write_json(self, rel: str, obj) -> None:
        self._atomic_write(
            os.path.join(self.root, rel),
            (canonical_json(obj) + "\n").encode("utf-8"),
        )

    def append_ledger(self, name: str, record: dict) -> None:
        path = os.path.join(self.root, name)
        line = (canonical_json(record) + "\n").encode("utf-8")
        with open(path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    # ---- 提交 ----
    def commit_version(self, snapshot: Snapshot, ops: list[dict], manifest: dict) -> None:
        """先写日志与快照，最后更新清单；清单是提交点。"""
        v = snapshot.version
        self.write_json(f"logs/v{v}.jsonl", ops)
        self.write_json(f"snapshots/v{v}.json", snapshot.to_dict())
        self.write_json("manifest.json", manifest)

    def mark_clean_shutdown(self) -> None:
        self._atomic_write(os.path.join(self.root, SHUTDOWN_MARKER), b"clean\n")

    # ---- 读取 ----
    def read_json(self, rel: str):
        with open(os.path.join(self.root, rel), encoding="utf-8") as fh:
            return json.load(fh)

    def has(self, rel: str) -> bool:
        return os.path.exists(os.path.join(self.root, rel))

    def load_manifest(self) -> dict:
        return self.read_json("manifest.json")

    def load_snapshot(self, version: int) -> Snapshot:
        return Snapshot.from_dict(self.read_json(f"snapshots/v{version}.json"))

    def load_ops(self, version: int) -> list[dict]:
        return self.read_json(f"logs/v{version}.jsonl")

    # ---- 恢复 ----
    def recover(self) -> dict:
        """强制终止后的恢复：只依据已提交清单，回收未提交尾部，不伪造退出报告。"""
        report = {"previous_clean_shutdown": self.has(SHUTDOWN_MARKER),
                  "reclaimed": [], "head": None}
        if self.has(SHUTDOWN_MARKER):
            os.remove(os.path.join(self.root, SHUTDOWN_MARKER))
        # 清理未完成的临时文件
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                if name.endswith(".tmp"):
                    os.remove(os.path.join(dirpath, name))
                    report["reclaimed"].append(os.path.relpath(os.path.join(dirpath, name), self.root))
        if not self.has("manifest.json"):
            return report
        manifest = self.load_manifest()
        committed = {entry["version"] for entry in manifest["versions"]}
        report["head"] = manifest["head"]
        # 回收未提交尾部：清单之外的快照与日志一律作废
        for sub, prefix in ((self.snapshots_dir, "snapshots"), (self.logs_dir, "logs")):
            for name in sorted(os.listdir(sub)):
                if not name.startswith("v"):
                    continue
                version = int(name.split(".")[0][1:])
                if version not in committed:
                    os.remove(os.path.join(sub, name))
                    report["reclaimed"].append(f"{prefix}/{name}")
        report["reclaimed"].sort()
        return report

    # ---- 快照目录整体替换：以版本戳与内容哈希判定新旧，禁止盲目录入 ----
    def adopt_snapshot_dir(self, incoming_root: str) -> dict:
        incoming = Store(incoming_root)
        if not incoming.has("manifest.json"):
            return {"adopted": False, "reason": "incoming 缺少清单"}
        in_manifest = incoming.load_manifest()
        for entry in in_manifest["versions"]:
            snap = incoming.load_snapshot(entry["version"])
            if snap.compute_hash() != entry["hash"]:
                return {"adopted": False,
                        "reason": f"incoming 版本 {entry['version']} 内容哈希校验失败"}
        if not self.has("manifest.json"):
            decision, reason = True, "本地为空，接受经校验的 incoming"
        else:
            cur_manifest = self.load_manifest()
            if in_manifest["head"] > cur_manifest["head"]:
                decision, reason = True, (
                    f"incoming 版本戳 {in_manifest['head']} 新于本地 {cur_manifest['head']}")
            else:
                decision, reason = False, (
                    f"incoming 版本戳 {in_manifest['head']} 不新于本地 "
                    f"{cur_manifest['head']}，拒绝盲目录入")
        if not decision:
            return {"adopted": False, "reason": reason}
        for entry in in_manifest["versions"]:
            v = entry["version"]
            self.write_json(f"snapshots/v{v}.json", incoming.read_json(f"snapshots/v{v}.json"))
            if incoming.has(f"logs/v{v}.jsonl"):
                self.write_json(f"logs/v{v}.jsonl", incoming.read_json(f"logs/v{v}.jsonl"))
        self.write_json("manifest.json", in_manifest)
        return {"adopted": True, "reason": reason, "head": in_manifest["head"]}
