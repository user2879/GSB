"""内容寻址存储、复用账本与阶段快照。

目录布局（全部位于工作区子目录，不污染源样本目录）::

    <root>/
      blobs/<key>.json            # 不可变内容寻址产物（分块/特征束/嵌入/索引条目）
      state/ledger.json           # 当前账本（原子写）
      snapshots/v<N>/manifest.json + 状态副本
      tmp/                        # 提交前临时目录

快照采用“先写临时目录 -> 整体替换 -> 指针更新”的方式；
强制终止后只能恢复到最后一次完整替换的快照。
"""

from __future__ import annotations

import os
import shutil

from .jsonio import read_json, write_json
from .errors import Failure

SNAPSHOT_PREFIX = "snap-v"
LEDGER_NAME = "ledger.json"
POINTER_NAME = "latest"


class Store:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.blob_dir = os.path.join(self.root, "blobs")
        self.state_dir = os.path.join(self.root, "state")
        self.snapshot_dir = os.path.join(self.root, "snapshots")
        self.tmp_dir = os.path.join(self.root, "tmp")
        for directory in (
            self.root,
            self.blob_dir,
            self.state_dir,
            self.snapshot_dir,
            self.tmp_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    # ---- 不可变内容寻址产物 ----
    def blob_path(self, key: str) -> str:
        return os.path.join(self.blob_dir, f"{key}.json")

    def has_blob(self, key: str) -> bool:
        return os.path.exists(self.blob_path(key))

    def get_blob(self, key: str):
        return read_json(self.blob_path(key))

    def put_blob(self, key: str, payload) -> None:
        path = self.blob_path(key)
        if os.path.exists(path):
            # 内容寻址：同键即同内容，禁止覆盖写以暴露键冲突缺陷。
            existing = read_json(path)
            if existing != payload:
                raise RuntimeError(f"内容寻址键冲突: {key}")
            return
        write_json(path, payload)

    # ---- 账本 ----
    @property
    def ledger_path(self) -> str:
        return os.path.join(self.state_dir, LEDGER_NAME)

    def load_ledger(self):
        if os.path.exists(self.ledger_path):
            return read_json(self.ledger_path)
        return new_ledger()

    def save_ledger(self, ledger: dict) -> None:
        write_json(self.ledger_path, ledger)

    # ---- 快照 ----
    def _snapshot_path(self, version: int) -> str:
        return os.path.join(self.snapshot_dir, f"{SNAPSHOT_PREFIX}{version}")

    def commit_snapshot(self, ledger: dict, stage_versions: dict) -> dict:
        """整体写入新版本快照并原子更新 latest 指针。返回快照清单。"""
        version = int(ledger["global_version"])
        target = self._snapshot_path(version)
        if os.path.exists(target):
            raise RuntimeError(f"快照版本已存在: v{version}")
        staging = os.path.join(
            self.tmp_dir, f"{SNAPSHOT_PREFIX}{version}-{os.getpid()}"
        )
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)

        ledger_copy = dict(ledger)
        ledger_copy["snapshot_version"] = version
        write_json(os.path.join(staging, LEDGER_NAME), ledger_copy)
        manifest = {
            "global_version": version,
            "config_hash": ledger["config_hash"],
            "quant_level": ledger["quant_level"],
            "completed_images": sorted(ledger["images"].keys()),
            "stage_versions": dict(sorted(stage_versions.items())),
            "ledger_file": LEDGER_NAME,
        }
        write_json(os.path.join(staging, "manifest.json"), manifest)

        # 整体替换：rename 在同文件系统上为原子操作。
        os.rename(staging, target)
        pointer = os.path.join(self.snapshot_dir, POINTER_NAME)
        write_json(pointer, {"latest_version": version})
        return manifest

    def latest_snapshot_version(self) -> int | None:
        pointer = os.path.join(self.snapshot_dir, POINTER_NAME)
        if not os.path.exists(pointer):
            return None
        return int(read_json(pointer)["latest_version"])

    def recover(self) -> tuple[dict | None, dict | None]:
        """仅依据完整替换的快照恢复；返回 (账本, 快照清单)。"""
        version = self.latest_snapshot_version()
        if version is None:
            return None, None
        snapshot = self._snapshot_path(version)
        manifest_path = os.path.join(snapshot, "manifest.json")
        ledger_snapshot = os.path.join(snapshot, LEDGER_NAME)
        if not (os.path.exists(manifest_path) and os.path.exists(ledger_snapshot)):
            return None, None
        manifest = read_json(manifest_path)
        ledger = read_json(ledger_snapshot)
        # 恢复点覆盖当前工作账本（未完成提交的状态一律丢弃）。
        write_json(self.ledger_path, ledger)
        return ledger, manifest

    def cleanup_tmp(self) -> None:
        for name in os.listdir(self.tmp_dir):
            path = os.path.join(self.tmp_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


def new_ledger() -> dict:
    return {
        "schema": "vrep-ledger-v1",
        "global_version": 0,
        "config_hash": None,
        "quant_level": None,
        "images": {},
        "index": None,
        "failures": [],
        "runtime": {
            "reuse": [],
            "recompute": [],
            "budget_audit": [],
            "pending": [],
        },
        "snapshots": [],
        "last_recovery": None,
    }


def sort_failures(failures: list[dict]) -> list[dict]:
    return sorted(
        failures,
        key=lambda item: Failure.sort_key(
            Failure(
                image=item["image"],
                stage=item["stage"],
                code=item["code"],
                detail=item["detail"],
                injection_id=item.get("injection_id"),
            )
        ),
    )
