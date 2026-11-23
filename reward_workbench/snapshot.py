"""阶段化快照：临时目录写入 + 整体替换提交 + 启动陈旧判定。

阶段划分：load / feature / score / boundary / summary。
每阶段产物先写到 runtime/tmp/<staging>，再用 os.replace 原子整体提交到
runtime/snapshots/<version>/，绝不逐文件覆盖在线快照。
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Any

from .errors import E_SNAPSHOT_STALE, FailureEntry, FailureLog
from .hashing import sha256_hex

STAGES = ("load", "feature", "score", "boundary", "summary")
MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    global_version: int
    config_hash: str
    data_fingerprint: str
    processed_cursor: int
    consumed_budget: int
    stage_versions: dict[str, int]
    parallel: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SnapshotStore:
    def __init__(self, root: str, tmp_root: str) -> None:
        self.root = root
        self.tmp_root = tmp_root
        os.makedirs(root, exist_ok=True)
        os.makedirs(tmp_root, exist_ok=True)

    # ---- 写：先临时目录后整体替换 ----
    def commit_stage(self, version: int, stage: str, payload: dict,
                     manifest: Manifest) -> str:
        staging = os.path.join(self.tmp_root, f"v{version}-{stage}-{os.getpid()}")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(staging, exist_ok=True)
        with open(os.path.join(staging, f"{stage}.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
        with open(os.path.join(staging, MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump(manifest.to_dict(), fh, ensure_ascii=False, sort_keys=True, indent=2)

        target_dir = os.path.join(self.root, f"v{version}")
        os.makedirs(target_dir, exist_ok=True)
        # 逐文件 os.replace 是同卷原子操作；目标版本目录在提交窗口内自洽
        for name in (f"{stage}.json", MANIFEST_NAME):
            os.replace(
                os.path.join(staging, name),
                os.path.join(target_dir, name),
            )
        shutil.rmtree(staging, ignore_errors=True)
        return target_dir

    def latest_version(self) -> int | None:
        versions = []
        for name in os.listdir(self.root):
            if name.startswith("v") and name[1:].isdigit():
                versions.append(int(name[1:]))
        return max(versions) if versions else None

    def load_manifest(self, version: int) -> Manifest | None:
        path = os.path.join(self.root, f"v{version}", MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return Manifest(**data)

    def load_stage(self, version: int, stage: str) -> dict | None:
        path = os.path.join(self.root, f"v{version}", f"{stage}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ---- 启动判定：版本号 + 配置哈希 ----
    def resume_or_detect_stale(self, expected_config_hash: str,
                               expected_data_fp: str,
                               failures: FailureLog) -> tuple[Manifest | None, bool]:
        """返回 (manifest, is_stale)。

        - 无快照：全新启动。
        - 配置哈希/数据指纹不匹配或版本落后：E_SNAPSHOT_STALE，不逐文件覆盖，
          由上层从失败清单/空白续跑重建。
        """
        version = self.latest_version()
        if version is None:
            return None, False
        manifest = self.load_manifest(version)
        if manifest is None:
            failures.add(FailureEntry(
                sort_key=f"v{version}", code=E_SNAPSHOT_STALE,
                stage="snapshot", message="manifest unreadable",
            ))
            return None, True
        if manifest.config_hash != expected_config_hash or \
                manifest.data_fingerprint != expected_data_fp:
            failures.add(FailureEntry(
                sort_key=f"v{version}", code=E_SNAPSHOT_STALE, stage="snapshot",
                message="snapshot config/data mismatch; rebuild from failure list",
                context=f"version={version}",
            ))
            return manifest, True
        return manifest, False


def fingerprint_config(cfg_dict: dict[str, Any]) -> str:
    canonical = json.dumps(cfg_dict, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return sha256_hex(["config", canonical])


def fingerprint_pairs(pairs_dict: dict[str, Any]) -> str:
    canonical = json.dumps(pairs_dict, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return sha256_hex(["pairs", canonical])
