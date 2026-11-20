"""批次快照整体替换、恢复判定与续跑。

每次提交先写入独立临时目录（不可变版本目录），再把 manifest.json
以 os.replace 原子换入快照根目录，保证“整体替换”语义：任一时刻
manifest 要么指向旧版本、要么指向新版本，不会出现半写状态。
恢复时以版本号、配置哈希与张量包指纹共同判定，不一致输出
E_SNAPSHOT_STALE 并从头重算。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass
from pathlib import Path

from .errors import E_SNAPSHOT_STALE, FailureLedger

MANIFEST_NAME = "manifest.json"


@dataclass
class SnapshotState:
    version: int
    config_hash: str
    package_fingerprint: str
    completed_batches: int
    consumed_budget: int
    payload: dict
    resumed: bool = False


def _manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_NAME


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                              default=_json_default, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"不可 JSON 序列化的对象: {type(obj)!r}")


def commit_snapshot(
    root: str | Path,
    *,
    version: int,
    config_hash: str,
    package_fingerprint: str,
    completed_batches: int,
    consumed_budget: int,
    payload: dict,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    version_dir = root / f"v{version:06d}"
    staging = root / f".staging-v{version:06d}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _write_json(staging / "state.json", payload)
    manifest = {
        "version": version,
        "config_hash": config_hash,
        "package_fingerprint": package_fingerprint,
        "completed_batches": completed_batches,
        "consumed_budget": consumed_budget,
        "state_path": f"{version_dir.name}/state.json",
    }
    # 临时目录整体落盘后再发布版本目录；陈旧版本目录先回收再原子改名。
    if version_dir.exists():
        shutil.rmtree(version_dir)
    os.replace(staging, version_dir)
    _write_json(_manifest_path(root), manifest)
    return version_dir


def load_snapshot(
    root: str | Path,
    *,
    config_hash: str,
    package_fingerprint: str,
    ledger: FailureLedger,
) -> SnapshotState | None:
    manifest_path = _manifest_path(root)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ledger.add(
            E_SNAPSHOT_STALE,
            "快照清单无法解析，按全新运行重算",
            "snapshot",
            (0, "manifest"),
        )
        return None

    mismatches = []
    if manifest.get("config_hash") != config_hash:
        mismatches.append("config_hash")
    if manifest.get("package_fingerprint") != package_fingerprint:
        mismatches.append("package_fingerprint")
    if mismatches:
        ledger.add(
            E_SNAPSHOT_STALE,
            f"快照与当前输入不一致（{', '.join(mismatches)}），版本 "
            f"{manifest.get('version')} 作废并重新计算",
            "snapshot",
            (0, "stale", *mismatches),
        )
        return None

    state_path = Path(root) / manifest["state_path"]
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        ledger.add(
            E_SNAPSHOT_STALE,
            f"快照状态文件缺失或损坏: {state_path}",
            "snapshot",
            (0, "state"),
        )
        return None
    return SnapshotState(
        version=int(manifest["version"]),
        config_hash=manifest["config_hash"],
        package_fingerprint=manifest["package_fingerprint"],
        completed_batches=int(manifest["completed_batches"]),
        consumed_budget=int(manifest["consumed_budget"]),
        payload=payload,
        resumed=True,
    )
